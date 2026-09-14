from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vlm_benchmark.reviews import export_reviews
from vlm_benchmark.dataset import load_datasets
from vlm_benchmark.validation import decision_matches, strict_json_loads, validate_decision


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/model-comparison-20260913"
REPORT_DIR = ROOT / "reports"
MODELS = (
    ("qwen", "qwen3.5-4b-q4km"),
    ("smolvlm2", "smolvlm2-2.2b-q4km"),
    ("internvl", "internvl3.5-4b-q4km"),
    ("gemma3", "gemma3-4b-it-q4km"),
)
DATASET_CASES = {case["case_id"]: case for case in load_datasets([ROOT / "datasets/expanded/cases.jsonl"])}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(value: int, denominator: int) -> float | None:
    return round(value / denominator, 6) if denominator else None


def outer_fence_diagnostic(cases: list[dict[str, Any]]) -> dict[str, int]:
    fence = re.compile(r"\A\s*```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*\Z", re.IGNORECASE | re.DOTALL)
    raw_valid = 0
    tool_match = 0
    args_match_non_scene = 0
    non_scene_denominator = 0
    recoverable_fence_only = 0
    for result in cases:
        case = DATASET_CASES[result["case_id"]]
        is_scene = case["oracle"].get("scene_rubric") is not None
        if not is_scene:
            non_scene_denominator += 1
        raw = result.get("raw_response")
        if not isinstance(raw, str):
            continue
        match = fence.fullmatch(raw)
        candidate = match.group(1) if match else raw
        try:
            parsed = strict_json_loads(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
        if validate_decision(parsed):
            continue
        raw_valid += 1
        acceptable = case["oracle"]["acceptable_decisions"]
        tool_match += int(any(parsed["tool"] == item["tool"] for item in acceptable))
        if not is_scene:
            args_match_non_scene += int(decision_matches(parsed, acceptable))
        if match and not result["scores"]["raw_valid"]:
            recoverable_fence_only += 1
    return {
        "raw_valid": raw_valid,
        "tool_match": tool_match,
        "tool_args_match_non_scene": args_match_non_scene,
        "tool_args_match_non_scene_denominator": non_scene_denominator,
        "recoverable_fence_only": recoverable_fence_only,
    }


def environment() -> dict[str, Any]:
    gpu: dict[str, str] | None = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        name, memory_mib, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        gpu = {"name": name, "memory_total_mib": memory_mib, "driver": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
        "runtime": {
            "name": "llama.cpp b10941 Vulkan",
            "commit": "4a8993735",
            "context_size": 8192,
            "parallel": 1,
            "gpu_layers": "all",
            "host": "127.0.0.1",
        },
    }


def collect_model(model_key: str, run_name: str, catalog: dict[str, Any]) -> dict[str, Any]:
    run = RUN_ROOT / run_name
    plan = read_json(run / "execution_plan.json")
    if plan.get("status") != "complete" or plan.get("completed_cases") != 106:
        raise ValueError(f"incomplete run: {run}")
    cases = read_jsonl(run / "cases.jsonl")
    analysis = read_json(run / "analysis.json")
    summary = read_json(run / "summary.json")
    if len(cases) != 106 or analysis["denominator"] != 106 or summary["n_called"] != 106:
        raise ValueError(f"denominator mismatch: {run}")
    raw_valid = sum(item["scores"]["raw_valid"] for item in cases)
    tool_match = sum(item["scores"]["tool_correct"] for item in cases)
    abstain = sum(bool(item.get("parsed") and item["parsed"].get("abstain")) for item in cases)
    backend_errors = sum(item["error_kind"] == "backend_error" for item in cases)
    scene_pending = sum(bool(item.get("scene_review")) for item in cases)
    automatic = analysis["automatic"]
    fence_diagnostic = outer_fence_diagnostic(cases)
    row = {
        "model_key": model_key,
        "model": run_name,
        "family": catalog["models"][model_key]["family"],
        "precision": catalog["models"][model_key]["precision"],
        "n": 106,
        "raw_valid": raw_valid,
        "raw_valid_rate": rate(raw_valid, 106),
        "tool_match": tool_match,
        "tool_match_rate": rate(tool_match, 106),
        "tool_args_match_non_scene": automatic["tool_args_match_non_scene"],
        "tool_args_match_non_scene_denominator": automatic["tool_args_match_non_scene_denominator"],
        "tool_args_match_non_scene_rate": rate(
            automatic["tool_args_match_non_scene"], automatic["tool_args_match_non_scene_denominator"]
        ),
        "raw_abstain": abstain,
        "raw_abstain_rate": rate(abstain, 106),
        "backend_errors": backend_errors,
        "latency_p50_ms": analysis["latency_ms"]["p50"],
        "latency_p95_ms": analysis["latency_ms"]["p95"],
        "scene_outputs_pending_review": scene_pending,
        "scene_content_score": None,
        "outer_fence_diagnostic": fence_diagnostic,
        "by_primary_acceptable_tool": analysis["by_primary_oracle_tool"],
        "run_paths": {
            "cases": str((run / "cases.jsonl").relative_to(ROOT)).replace("\\", "/"),
            "summary": str((run / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            "analysis": str((run / "analysis.json").relative_to(ROOT)).replace("\\", "/"),
            "manifest": str((run / "manifest.json").relative_to(ROOT)).replace("\\", "/"),
            "manual_review": str((run / "reviews.todo.jsonl").relative_to(ROOT)).replace("\\", "/"),
        },
    }
    manifest = {
        **plan,
        "result_files": {
            "cases.jsonl": sha256(run / "cases.jsonl"),
            "summary.json": sha256(run / "summary.json"),
            "analysis.json": sha256(run / "analysis.json"),
        },
    }
    migration_path = run / "scene-metadata-migration.json"
    if migration_path.exists():
        manifest["posthoc_metadata_migrations"] = [read_json(migration_path)]
    (run / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    export_reviews(run / "cases.jsonl", run / "reviews.todo.jsonl")
    return row


def link(path: str, label: str) -> str:
    return f"[{label}](../{path})"


def main() -> int:
    catalog = read_json(ROOT / "runtime/comparison/models.json")
    rows = [collect_model(key, name, catalog) for key, name in MODELS]
    env = environment()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    machine = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic dev A-proxy; not a holdout, real-camera, ROS, navigation, or safety result",
        "dataset": {
            "path": "datasets/expanded/cases.jsonl",
            "sha256": "e9772611575f4565041367af1e384e7f50caff238a1f09c832af6b4f8153b266",
            "cases": 106,
            "groups": 20,
            "images": 20,
            "split": "dev",
        },
        "protocol": {
            "prompt_sha256": "4192a7dd5d52ab71e29510bdb45cf7bdd9195b72b05babc127a5da1482fd6bcd",
            "temperature": 0,
            "max_tokens": 256,
            "timeout_seconds": 180,
            "case_order": "dataset order",
            "differences": "Each GGUF uses its embedded native chat template and llama.cpp model-specific vision preprocessing; prompt text and input image bytes are unchanged.",
        },
        "environment": env,
        "models": rows,
        "molmo2_feasibility": {
            "status": "not_tested_inference_unsupported_by_frozen_runtime",
            "runtime_commit": "4a8993735",
            "source_tree_recursive_search_truncated": False,
            "molmo_named_paths_in_runtime_source": 0,
            "available_third_party_gguf": "reubk/Molmo2-4B-GGUF@330afa8d9c9206086ab1a0f93b297393e9dd8e2f",
            "reason": "The exact llama.cpp runtime source tree contains no Molmo/Molmo2 implementation path; adding a different runtime or Transformers stack would violate the bounded single-runtime comparison.",
        },
    }
    json_path = REPORT_DIR / "model-comparison-summary.json"
    json_path.write_text(json.dumps(machine, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in {"by_primary_acceptable_tool", "run_paths", "outer_fence_diagnostic"}}
        flat.update({f"outer_fence_{key}": value for key, value in row["outer_fence_diagnostic"].items()})
        csv_rows.append(flat)
    csv_fields = list(csv_rows[0])
    with (REPORT_DIR / "model-comparison-summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    table = []
    for row in rows:
        table.append(
            f"| {row['model']} | {row['raw_valid']}/106 | {row['tool_match']}/106 | "
            f"{row['tool_args_match_non_scene']}/{row['tool_args_match_non_scene_denominator']} | "
            f"{row['outer_fence_diagnostic']['tool_match']}/106 | "
            f"{row['outer_fence_diagnostic']['tool_args_match_non_scene']}/{row['outer_fence_diagnostic']['tool_args_match_non_scene_denominator']} | "
            f"{row['outer_fence_diagnostic']['recoverable_fence_only']} | "
            f"{row['raw_abstain']}/106 | {row['backend_errors']} | {row['latency_p50_ms']:.1f} | "
            f"{row['latency_p95_ms']:.1f} | N/A ({row['scene_outputs_pending_review']} pending) |"
        )
    artifacts = []
    for row in rows:
        paths = row["run_paths"]
        artifacts.append(
            f"- **{row['model']}** — {link(paths['cases'], 'raw cases')}, {link(paths['manifest'], 'manifest')}, "
            f"{link(paths['analysis'], 'analysis')}, {link(paths['manual_review'], 'manual review export')}"
        )
    report = f"""# Сравнение локальных VLM для decision benchmark

Дата: 13.09.2026. Это один фиксированный прогон четырёх quantized VLM на синтетическом dev-наборе:
106 cases, но только 20 связанных групп/изображений. Эти 106 строк нельзя трактовать как 106
независимых наблюдений; результат не доказывает перенос на реальные камеры, ROS, навигацию или
физическую безопасность. Статистическая значимость не заявляется и абсолютный «победитель» не
назначается.

## Общий протокол

Dataset SHA-256: `{machine['dataset']['sha256']}`. Prompt SHA-256:
`{machine['protocol']['prompt_sha256']}`. Для всех моделей: temperature 0, max output 256 tokens,
timeout 180 s, один порядок cases, те же prompt и байты изображений. Различались только встроенный
native chat template GGUF и обязательная model-specific vision preprocessing в llama.cpp. Qwen3.5
запущен с `--reasoning off`; другим моделям отдельные смысловые prompts не давались. Все ошибки и
abstain остаются в знаменателе 106. `Raw abstain` означает, что parsed JSON содержит
`abstain:true`, даже если весь объект затем не прошёл schema validation. Model-quality retry,
output selection, LLM repair и prompt tuning отсутствуют. Один упавший до записи InternVL request
был операционно повторён после исправления evaluator; это явно записано в code lineage.

## Автоматические результаты

| Модель | Raw valid | Tool match | Tool+args, non-scene | Fence diag tool | Fence diag tool+args | Fence-only recovered | Raw abstain | Backend errors | p50 ms | p95 ms | Scene content |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(table)}

`Tool+args` не включает cases со свободным `describe_scene`: для них exact match/keywords не
используются. Scene content остаётся N/A до независимой ручной проверки по сохранённым rubrics.
`Fence diag` — только post-hoc диагностика: вокруг **всего** ответа удаляется ровно одна внешняя
пара `````json ... ````` или ````` ... `````, затем применяется тот же strict JSON/domain validator.
Внутренний JSON не извлекается, потерянные поля не дополняются, основной strict score не меняется.
Группировка `by_primary_acceptable_tool` в machine-readable JSON означает **первый допустимый tool
oracle**, а не единственную внутреннюю природу case; некоторые cases допускают несколько решений.

InternVL3.5 и Qwen3.5 — разные VLM/vision stacks, но InternVL3.5 использует Qwen3 language
backbone, поэтому они не являются полностью независимыми семействами. Все значения относятся к
Q4_K_M language weights с F16 projector, а не к full-precision checkpoints.

## Molmo2 feasibility

Molmo2-4B не запускался. Для exact runtime commit `4a8993735` recursive GitHub source tree был
получен полностью (`truncated=false`) и не содержит ни одного Molmo/Molmo2 path. Доступен только
third-party GGUF `reubk/Molmo2-4B-GGUF@330afa8d...`, однако projector implementation в frozen
runtime не подтверждён. Проверка потребовала бы другого llama.cpp или Transformers stack, поэтому
она осознанно оставлена вне bounded comparison.

## Артефакты

{chr(10).join(artifacts)}

- {link('reports/model-comparison-summary.json', 'machine-readable JSON')}
- {link('reports/model-comparison-summary.csv', 'machine-readable CSV')}
- {link('runs/model-comparison-20260913/remaining-queue-v2.log', 'queue log with one recorded evaluator failure')}
- {link('runs/model-comparison-20260913/remaining-queue.log', 'preserved failed queue attempt')}

## Повторение

Exact model revisions/hashes находятся в `runtime/comparison/models.json`. Полное повторение нужно
писать в новый каталог, чтобы не перезаписывать историю:

```powershell
$env:PYTHONPATH=(Resolve-Path .\\src).Path
$repro = "runs\\model-comparison-repro"
powershell -ExecutionPolicy Bypass -File runtime\\stop-local-vlm.ps1
# Для каждого key/config/name: qwen, smolvlm2, internvl, gemma3
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\start-model.ps1 -Model qwen -LogDirectory "$repro\\server-logs\\qwen"
python runtime\\comparison\\run_checkpoint.py datasets\\expanded\\cases.jsonl --config configs\\comparison-qwen3.5-4b-q4km.json --model-key qwen --output "$repro\\qwen3.5-4b-q4km"
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\stop-model.ps1
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\start-model.ps1 -Model smolvlm2 -LogDirectory "$repro\\server-logs\\smolvlm2"
python runtime\\comparison\\run_checkpoint.py datasets\\expanded\\cases.jsonl --config configs\\comparison-smolvlm2-2.2b-q4km.json --model-key smolvlm2 --output "$repro\\smolvlm2-2.2b-q4km"
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\stop-model.ps1
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\start-model.ps1 -Model internvl -LogDirectory "$repro\\server-logs\\internvl"
python runtime\\comparison\\run_checkpoint.py datasets\\expanded\\cases.jsonl --config configs\\comparison-internvl3.5-4b-q4km.json --model-key internvl --output "$repro\\internvl3.5-4b-q4km"
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\stop-model.ps1
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\start-model.ps1 -Model gemma3 -LogDirectory "$repro\\server-logs\\gemma3"
python runtime\\comparison\\run_checkpoint.py datasets\\expanded\\cases.jsonl --config configs\\comparison-gemma3-4b-it-q4km.json --model-key gemma3 --output "$repro\\gemma3-4b-it-q4km"
powershell -ExecutionPolicy Bypass -File runtime\\comparison\\stop-model.ps1
powershell -ExecutionPolicy Bypass -File runtime\\start-local-vlm.ps1
```

`run_checkpoint.py` допускает только resume точного dataset-order prefix и проверяет frozen
dataset/config/prompt/code hashes. Исторический InternVL run содержит явный `code_lineage`: после
74 cases исправлен evaluator crash на ошибочном `describe_scene` для non-scene oracle; первые 74
cases не запускались повторно. Упавший до записи 75-й request был операционно выполнен снова после
restart; это не выбор лучшего output и не model-quality retry. Post-hoc
`scene-review-null-rubric-v1` меняет только ошибочно прикреплённый `scene_review` metadata; raw,
parsed, scores и latency защищены одинаковым before/after digest в каждом manifest.
"""
    (REPORT_DIR / "model-comparison.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "models": len(rows), "report": str(REPORT_DIR / 'model-comparison.md')}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
