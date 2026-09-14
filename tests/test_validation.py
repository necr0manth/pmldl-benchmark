from vlm_benchmark.validation import strict_json_loads, validate_decision


def test_strict_json_rejects_duplicate_and_nonfinite():
    for raw in ('{"tool":"idle","tool":"idle"}', '{"confidence": NaN}'):
        try:
            strict_json_loads(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JSON extension was accepted")


def test_typed_tool_args_and_abstention_are_strict():
    valid = {
        "tool": "report_audience",
        "args": {"visible_count": 2, "facing_robot_count": 1, "recommendation": "continue"},
        "confidence": 0.7,
        "abstain": False,
    }
    assert validate_decision(valid) == []
    invalid = {**valid, "args": {**valid["args"], "visible_count": True}}
    assert "invalid args for report_audience" in validate_decision(invalid)
    bad_abstain = {"tool": "idle", "args": {}, "confidence": 0.1, "abstain": False, "extra": 1}
    assert validate_decision(bad_abstain)
