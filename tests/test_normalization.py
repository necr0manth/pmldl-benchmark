from vlm_benchmark.backend import MockBackend
from vlm_benchmark.core import DecisionEngine
from vlm_benchmark.validation import normalize_outer_fence


def test_outer_fence_normalizer_is_exact_and_strict_default():
    raw = '```json\n{"tool":"idle","args":{},"confidence":0,"abstain":true}\n```'
    normalized, applied = normalize_outer_fence(raw, "fence-only")
    assert applied and normalized.startswith("{") and normalized.endswith("}")
    assert normalize_outer_fence(raw, "strict") == (raw, False)
    assert normalize_outer_fence('{"x":"```"}', "fence-only")[1] is False


def test_core_effective_uses_configured_fence_normalization():
    raw = '```json\n{"tool":"idle","args":{},"confidence":0,"abstain":true}\n```'
    engine = DecisionEngine(MockBackend({}, raw), "{{TRANSCRIPT_JSON}}", "fence-only")
    detail = engine.decide_detailed("x", None, {}, "")
    assert detail["error_kind"] is None
    assert detail["normalization_applied"] is True
    assert detail["effective"]["reason"] == "model_abstained"
