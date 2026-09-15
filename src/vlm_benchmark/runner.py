from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import configure
from .dataset import fixture_image, load_datasets
from .evaluation import evaluate_case, summarize


def run_benchmark(
    dataset_paths: list[str | Path],
    config_path: str | Path,
    output_dir: str | Path,
    case_ids: list[str] | None = None,
) -> Path:
    datasets = [Path(path).resolve() for path in dataset_paths]
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_datasets(datasets)
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"unknown --case-id values: {sorted(missing)}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    engine = configure(config_file)
    results: list[dict[str, Any]] = []
    for case in cases:
        image = fixture_image(case)
        if case.get("method") == "check_people":
            detail = engine.check_people_detailed(image)
        elif case.get("method") == "describe_image":
            detail = engine.describe_image_detailed(image)
        else:
            detail = engine.decide_detailed(case["transcript"], image, case["mission_state"], case["extra_context"])
        results.append(evaluate_case(case, detail))
    write_jsonl(output / "cases.jsonl", results)
    (output / "summary.json").write_text(
        json.dumps(summarize(results), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": "0.1.0",
        "contract_version": "implementation-v1",
        "config": _safe_config(config),
        "config_sha256": sha256(config_file),
        "dataset_files": [{"path": str(path), "sha256": sha256(path)} for path in datasets],
        "image_assets": _image_assets(cases),
        "splits": {split: sum(case["split"] == split for case in cases) for split in sorted({case["split"] for case in cases})},
        "group_count": len({case["group_id"] for case in cases}),
        "prompt_sha256": hashlib.sha256(engine.prompt_template.encode("utf-8")).hexdigest(),
        "people_prompt_sha256": hashlib.sha256(engine.people_prompt_template.encode("utf-8")).hexdigest(),
        "description_prompt_sha256": hashlib.sha256(engine.description_prompt_template.encode("utf-8")).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "limitations": [
            "offline A-proxy; not a Robo-guide/ROS integration benchmark",
            "mock backend validates harness behavior only and does not measure VLM quality"
            if engine.backend.backend_type == "mock"
            else "external endpoint latency includes network transport",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in items), encoding="utf-8"
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    safe = json.loads(json.dumps(config))
    if "backend" in safe:
        safe["backend"].pop("api_key", None)
    return safe


def _image_assets(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    paths = {
        Path(case["_dataset_path"]).parent / case["image"]["path"]
        for case in cases
        if case["image"]["status"] == "available"
    }
    return [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sorted(paths)]
