from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vlm_benchmark.dataset import load_datasets
from vlm_benchmark.evaluation import summarize
from vlm_benchmark.runner import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/model-comparison-20260913"
RUNS = (
    "qwen3.5-4b-q4km",
    "smolvlm2-2.2b-q4km",
    "internvl3.5-4b-q4km",
    "gemma3-4b-it-q4km",
)
PRESERVED_FIELDS = (
    "case_id", "group_id", "split", "raw_response", "parsed", "parse_succeeded",
    "validation_errors", "error_kind", "effective", "latency_ms", "image_included", "scores",
)


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    dataset_path = ROOT / "datasets/expanded/cases.jsonl"
    dataset = {case["case_id"]: case for case in load_datasets([dataset_path])}
    for run_name in RUNS:
        run = RUN_ROOT / run_name
        cases_path = run / "cases.jsonl"
        migration_path = run / "scene-metadata-migration.json"
        if migration_path.exists():
            migration = json.loads(migration_path.read_text(encoding="utf-8"))
            if file_sha256(cases_path) != migration["after_cases_sha256"]:
                raise ValueError(f"post-migration cases hash mismatch: {run_name}")
            print(json.dumps({"run": run_name, "status": "already_migrated"}))
            continue
        rows = read_jsonl(cases_path)
        before_hash = file_sha256(cases_path)
        preserved_before = digest([{key: row[key] for key in PRESERVED_FIELDS} for row in rows])
        changed = []
        for row in rows:
            rubric = dataset[row["case_id"]]["oracle"].get("scene_rubric")
            if row.get("scene_review") is not None and rubric is None:
                row["scene_review"] = None
                changed.append(row["case_id"])
        preserved_after = digest([{key: row[key] for key in PRESERVED_FIELDS} for row in rows])
        if preserved_before != preserved_after:
            raise AssertionError(f"protected result fields changed: {run_name}")
        write_jsonl(cases_path, rows)
        (run / "summary.json").write_text(
            json.dumps(summarize(rows), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        migration = {
            "migration_id": "scene-review-null-rubric-v1",
            "rule": "set scene_review to null iff the dataset oracle scene_rubric is null",
            "reason": "older evaluator attached review metadata to an unexpected describe_scene prediction on a non-scene case",
            "dataset_sha256": file_sha256(dataset_path),
            "before_cases_sha256": before_hash,
            "after_cases_sha256": file_sha256(cases_path),
            "changed_case_ids": changed,
            "preserved_fields": list(PRESERVED_FIELDS),
            "preserved_fields_sha256_before": preserved_before,
            "preserved_fields_sha256_after": preserved_after,
        }
        migration_path.write_text(
            json.dumps(migration, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"run": run_name, "status": "migrated", "changed": len(changed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
