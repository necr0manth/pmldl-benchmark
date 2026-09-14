from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import summarize
from .runner import read_jsonl, write_jsonl


def export_reviews(cases_path: str | Path, output_path: str | Path) -> int:
    rows = []
    for item in read_jsonl(cases_path):
        scene = item.get("scene_review")
        if scene and scene.get("status") == "pending":
            rows.append(
                {
                    "case_id": item["case_id"],
                    "text": scene["text"],
                    "scene_rubric": scene["scene_rubric"],
                    "reviewer": "",
                    "verdict": "",
                    "observation_coverage": None,
                    "unsupported_claims": None,
                    "notes": "",
                }
            )
    write_jsonl(Path(output_path), rows)
    return len(rows)


def import_reviews(run_dir: str | Path, review_path: str | Path) -> int:
    run = Path(run_dir)
    results = read_jsonl(run / "cases.jsonl")
    reviews = read_jsonl(review_path)
    indexed = {item["case_id"]: item for item in results}
    seen: set[str] = set()
    for review in reviews:
        errors = validate_review(review)
        if errors:
            raise ValueError(f"review {review.get('case_id')!r}: " + "; ".join(errors))
        case_id = review["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate review for {case_id}")
        seen.add(case_id)
        if case_id not in indexed or indexed[case_id].get("scene_review") is None:
            raise ValueError(f"unknown/non-scene case_id: {case_id}")
        current = indexed[case_id]["scene_review"]
        for immutable in ("text", "scene_rubric"):
            if immutable in review and review[immutable] != current[immutable]:
                raise ValueError(f"review {case_id!r} changes immutable {immutable}")
        review_fields = {"reviewer", "verdict", "observation_coverage", "unsupported_claims", "notes"}
        indexed[case_id]["scene_review"] = {
            **current,
            "status": "reviewed",
            **{key: review[key] for key in review_fields},
        }
    write_jsonl(run / "cases.jsonl", results)
    (run / "summary.json").write_text(
        json.dumps(summarize(results), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return len(reviews)


def validate_review(review: Any) -> list[str]:
    if not isinstance(review, dict):
        return ["must be an object"]
    errors = []
    if not isinstance(review.get("case_id"), str) or not review["case_id"]:
        errors.append("case_id must be non-empty")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"]:
        errors.append("reviewer must be non-empty")
    if review.get("verdict") not in {"pass", "fail"}:
        errors.append("verdict must be pass or fail")
    coverage = review.get("observation_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
        errors.append("observation_coverage must be in [0,1]")
    unsupported = review.get("unsupported_claims")
    if isinstance(unsupported, bool) or not isinstance(unsupported, int) or unsupported < 0:
        errors.append("unsupported_claims must be integer >=0")
    if not isinstance(review.get("notes"), str):
        errors.append("notes must be a string")
    return errors
