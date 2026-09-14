import json

from vlm_benchmark.v2 import parse_v2, score_v2, summarize_v2, validate_v2_case


def case(expected=None, fields=None):
    return {
        "case_id": "x", "group_id": "g", "image_group": "img", "category": "basic",
        "expected": expected or {"tool": "goto_exhibit", "args": {"location_id": "hall"}, "abstain": False},
        "scored_fields": fields or ["tool", "abstain", "location_id"],
    }


def test_v2_strict_and_outer_fence_only():
    raw = '```json\n{"tool":"idle","args":{},"confidence":0,"abstain":true}\n```'
    assert parse_v2(raw, "strict")[0] is None
    assert parse_v2(raw, "fence-only")[0]["tool"] == "idle"
    assert parse_v2('{"x":{"tool":"idle"}}', "fence-only")[0] is None


def test_v2_scored_mask_does_not_hide_wrong_tool_or_missing_field():
    detail = {"raw_response": json.dumps({"tool": "start_tour", "args": {"tour_id": "main"}, "confidence": 1, "abstain": False})}
    result = score_v2(case(), detail)
    assert result["scores"]["fields"]["tool"] is False
    assert result["scores"]["composite"] is False


def test_scene_text_is_not_compared():
    expected = {"tool": "describe_scene", "args": {"text": "reference"}, "abstain": False}
    detail = {"raw_response": '{"tool":"describe_scene","args":{"text":"anything else"},"confidence":1,"abstain":false}'}
    result = score_v2(case(expected, ["tool", "abstain"]), detail)
    assert result["scores"]["composite"] is True
    assert result["scene_diagnostic"] == {"text_nonempty": True}


def test_category_summary_has_explicit_denominators():
    item = score_v2(case(), {"raw_response": '{"tool":"goto_exhibit","args":{"location_id":"hall"},"confidence":1,"abstain":false}'})
    summary = summarize_v2([item])
    assert summary["overall"]["fields"]["location_id"]["denominator"] == 1


def test_v2_rejects_legacy_ask_and_wrong_field_mask():
    bad = case({"tool": "ask_visitor", "args": {"question_id": "clarify"}, "abstain": False}, ["tool", "abstain"])
    bad.update({"split": "dev", "transcript": "x", "image": {"status": "available", "path": "missing"}, "mission_state": {}, "extra_context": "", "metadata": {"source": "synthetic", "benchmark_version": "v2"}})
    errors = validate_v2_case(bad, ".")
    assert any("not allowed in v2" in error for error in errors)
    assert parse_v2('{"tool":"ask_visitor","args":{"question_id":"x"},"confidence":1,"abstain":false}')[0] is None


def test_v2_does_not_score_schema_error_as_valid_parsed_dict():
    detail = {"raw_response": '{"tool":"idle","args":{},"confidence":0,"abstain":true}', "normalized_response": '{"tool":"idle"}', "normalization": "strict", "error_kind": "schema_error", "parsed": {"tool": "idle"}}
    result = score_v2(case({"tool": "idle", "args": {}, "abstain": True}, ["tool", "abstain"]), detail)
    assert result["scores"]["raw_valid"] is False
