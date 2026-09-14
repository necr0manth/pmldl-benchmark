from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vlm_benchmark.core import configure
from vlm_benchmark.dataset import fixture_image, load_datasets
from vlm_benchmark.evaluation import evaluate_case, summarize
from vlm_benchmark.runner import sha256, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append-only, resumable one-pass VLM benchmark runner")
    parser.add_argument("dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", required=True, choices=("qwen", "smolvlm2", "internvl", "gemma3"))
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--resume-after-evaluation-bugfix", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt checkpoint line {number}: {exc}") from exc
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def analysis(results: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {case["case_id"]: case for case in cases}
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        case = indexed[result["case_id"]]
        primary_tool = case["oracle"]["acceptable_decisions"][0]["tool"]
        groups.setdefault(primary_tool, []).append(result)

    def group_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        scene = all(indexed[item["case_id"]]["oracle"].get("scene_rubric") is not None for item in items)
        return {
            "n": count,
            "raw_valid": sum(item["scores"]["raw_valid"] for item in items),
            "tool_match": sum(item["scores"]["tool_correct"] for item in items),
            "tool_args_match": None if scene else sum(item["scores"]["raw_semantic_correct"] for item in items),
            "valid_non_abstain": sum(item["scores"]["valid_non_abstain"] for item in items),
            "errors": sum(item["error_kind"] is not None for item in items),
        }

    latencies = [float(item["latency_ms"]) for item in results]
    non_scene = [item for item in results if indexed[item["case_id"]]["oracle"].get("scene_rubric") is None]
    return {
        "denominator": len(results),
        "automatic": {
            "raw_valid": sum(item["scores"]["raw_valid"] for item in results),
            "tool_match": sum(item["scores"]["tool_correct"] for item in results),
            "tool_args_match_non_scene": sum(item["scores"]["raw_semantic_correct"] for item in non_scene),
            "tool_args_match_non_scene_denominator": len(non_scene),
            "valid_non_abstain": sum(item["scores"]["valid_non_abstain"] for item in results),
            "errors": sum(item["error_kind"] is not None for item in results),
        },
        "by_primary_oracle_tool": {key: group_metrics(groups[key]) for key in sorted(groups)},
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "first_case_cold_context": latencies[0] if latencies else None,
            "note": "Same fixed case order; later requests may benefit from llama.cpp prompt/image caching.",
        },
        "scene_content": {
            "automatic_content_score": None,
            "status": "pending_human_review",
            "note": "Only tool/shape are automatic; free description content is not exact-match graded.",
        },
    }


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_datasets([dataset])
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"unknown --case-id values: {sorted(missing)}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    catalog_path = Path(__file__).with_name("models.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = catalog["models"][args.model_key]
    if config["backend"]["model"] != entry["alias"]:
        raise ValueError("config alias does not match model catalog")

    prompt_path = (config_path.parent / config["prompt_path"]).resolve()
    plan_path = output / "execution_plan.json"
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_key": args.model_key,
        "model": entry,
        "runtime": catalog["runtime"],
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "config": {"path": str(config_path), "sha256": sha256(config_path), "value": config},
        "prompt": {"path": str(prompt_path), "sha256": sha256(prompt_path)},
        "code": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("models.json"),
                Path(__file__).resolve().with_name("start-model.ps1"),
                Path(__file__).resolve().with_name("stop-model.ps1"),
                Path(__file__).resolve().parents[2] / "src/vlm_benchmark/core.py",
                Path(__file__).resolve().parents[2] / "src/vlm_benchmark/backend.py",
                Path(__file__).resolve().parents[2] / "src/vlm_benchmark/dataset.py",
                Path(__file__).resolve().parents[2] / "src/vlm_benchmark/evaluation.py",
                Path(__file__).resolve().parents[2] / "src/vlm_benchmark/runner.py",
                Path(__file__).resolve().parents[2] / "src/vlm_benchmark/validation.py",
            )
        },
        "case_ids": [case["case_id"] for case in cases],
        "case_order": "dataset order",
        "native_model_behavior": "GGUF-embedded native chat template and llama.cpp model-specific vision preprocessing; semantic prompt and image bytes are identical.",
    }
    if plan_path.exists():
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in ("model_key", "model", "runtime", "dataset", "config", "prompt", "case_ids", "case_order"):
            if frozen[key] != plan[key]:
                raise ValueError(f"frozen execution plan mismatch: {key}")
        if frozen["code"] != plan["code"]:
            old_hashes = {
                "run_checkpoint.py": "819ac0ca915e8e18dfad19eef0d0e44345b32d3d4eac283aad752893fa1264a4",
                "evaluation.py": "b6526540b7bff66827404650c52786ced11795969b2b27ef2183c1b34d62542e",
            }
            changed = {
                Path(path).name
                for path in set(frozen["code"]) | set(plan["code"])
                if frozen["code"].get(path) != plan["code"].get(path)
            }
            old_matches = all(
                any(Path(path).name == name and value == expected for path, value in frozen["code"].items())
                for name, expected in old_hashes.items()
            )
            if not args.resume_after_evaluation_bugfix or changed != set(old_hashes) or not old_matches:
                raise ValueError("frozen execution plan mismatch: code")
            completed_before_fix = len(load_rows(output / "cases.jsonl"))
            frozen.setdefault("code_lineage", []).append(
                {
                    "reason": "evaluation bugfix: unexpected describe_scene on a non-scene oracle lacked a rubric and crashed before recording the case",
                    "old_code": frozen["code"],
                    "new_code": plan["code"],
                    "completed_cases_before_fix": completed_before_fix,
                }
            )
            frozen["code"] = plan["code"]
        plan = frozen
        plan["status"] = "running"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with urllib.request.urlopen(config["backend"]["base_url"].rstrip("/") + "/models", timeout=10) as response:
        endpoint_models = json.loads(response.read().decode("utf-8"))
    served_ids = [item.get("id") for item in endpoint_models.get("data", [])]
    if entry["alias"] not in served_ids:
        raise RuntimeError(f"endpoint does not serve expected alias {entry['alias']!r}: {served_ids}")
    (output / "endpoint_models.json").write_text(
        json.dumps(endpoint_models, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    results_path = output / "cases.jsonl"
    results = load_rows(results_path)
    existing_ids = [item["case_id"] for item in results]
    expected_prefix = [case["case_id"] for case in cases[: len(results)]]
    if existing_ids != expected_prefix:
        raise ValueError("checkpoint case IDs are not the expected dataset-order prefix")
    engine = configure(config_path)
    for case in cases[len(results) :]:
        detail = engine.decide_detailed(
            case["transcript"], fixture_image(case), case["mission_state"], case["extra_context"]
        )
        result = evaluate_case(case, detail)
        append_row(results_path, result)
        results.append(result)
        plan["completed_cases"] = len(results)
        plan["last_case_id"] = case["case_id"]
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write_jsonl(results_path, results)
    (output / "summary.json").write_text(
        json.dumps(summarize(results), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "analysis.json").write_text(
        json.dumps(analysis(results, cases), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plan["status"] = "complete"
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan["completed_cases"] = len(results)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "cases": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
