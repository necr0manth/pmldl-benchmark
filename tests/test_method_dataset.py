import json
from pathlib import Path

from vlm_benchmark.evaluation import evaluate_case


def test_people_labels_use_facing_annotation_not_legacy_recommendation():
    path = Path(__file__).parents[1] / "datasets" / "methods" / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_origin = {row["metadata"]["origin_case_ids"][0]: row for row in rows if row.get("method") == "check_people"}
    for case_id in ("v2-xa01-a01", "v2-xa03-a01", "v2-xa04-a01", "v2-xa06-a01", "v2-xa07-a01", "v2-xa08-a01"):
        assert by_origin[case_id]["expected"]["tool"] == "idle"
    for case_id in ("v2-xa02-a01", "v2-xa05-a01"):
        assert by_origin[case_id]["expected"]["tool"] == "interrupt"
    assert by_origin["v2-vb04-audience"]["expected"]["tool"] == "idle"


def test_unified_decide_cases_use_v2_scored_fields():
    path = Path(__file__).parents[1] / "datasets" / "methods" / "cases.jsonl"
    case = next(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if '"method": "decide"' in line)
    detail = {"raw_response": '{"tool":"start_tour","args":{"tour_id":"tour_demo"},"confidence":1,"abstain":false}', "parsed": {"tool": "start_tour", "args": {"tour_id": "tour_demo"}, "confidence": 1, "abstain": False}, "error_kind": None, "normalization": "fence-only", "normalized_response": None, "validation_errors": [], "effective": None}
    result = evaluate_case(case, detail)
    assert set(result["scores"]["fields"]) == set(case["scored_fields"])
