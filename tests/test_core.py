import json

from vlm_benchmark.backend import MockBackend
from vlm_benchmark.core import DecisionEngine


class CaptureBackend:
    backend_type = "capture"

    def __init__(self, response):
        self.response = response
        self.prompt = None

    def generate(self, prompt, image, mime_type):
        self.prompt = prompt
        return self.response


def test_prompt_receives_only_public_inputs_not_oracle():
    backend = CaptureBackend('{"tool":"idle","args":{},"confidence":0,"abstain":true}')
    engine = DecisionEngine(backend, "T={{TRANSCRIPT_JSON}} S={{MISSION_STATE_JSON}} X={{EXTRA_CONTEXT}} I={{IMAGE_STATUS_JSON}}")
    engine.decide_detailed("hello", None, {"state": "IDLE", "oracle": "SECRET_GOLD"}, "public")
    assert "hello" in backend.prompt and "public" in backend.prompt
    assert "SECRET_GOLD" not in backend.prompt and "oracle" not in backend.prompt


def test_prompt_substitution_does_not_reprocess_markers_from_user_data():
    backend = CaptureBackend('{"tool":"idle","args":{},"confidence":0,"abstain":true}')
    template = "T={{TRANSCRIPT_JSON}} X={{EXTRA_CONTEXT}} S={{MISSION_STATE_JSON}} I={{IMAGE_STATUS_JSON}}"
    engine = DecisionEngine(backend, template)
    prompt = engine.build_prompt("literal {{EXTRA_CONTEXT}}", "missing", {"state": "{{EXTRA_CONTEXT}}"}, "PUBLIC")
    assert prompt.count("PUBLIC") == 1
    assert prompt.count("{{EXTRA_CONTEXT}}") == 2


def test_prompt_declares_visitor_request_priority():
    prompt = DecisionEngine(MockBackend({}, "" )).prompt_template
    assert "explicit visitor command or request takes priority" in prompt


def test_invalid_and_backend_error_have_fallback_but_are_raw_failures():
    invalid = DecisionEngine(MockBackend({}, "not json"), "{{TRANSCRIPT_JSON}}")
    detail = invalid.decide_detailed("x", None, {}, "")
    assert detail["error_kind"] == "invalid_json"
    assert detail["parsed"] is None
    assert detail["effective"]["status"] == "fallback"

    class Broken:
        backend_type = "broken"

        def generate(self, prompt, image, mime_type):
            raise TimeoutError("deadline")

    error = DecisionEngine(Broken(), "{{TRANSCRIPT_JSON}}").decide_detailed("x", None, {}, "")
    assert error["error_kind"] == "backend_error"
    assert error["effective"]["reason"] == "backend_error"


def test_public_decide_returns_safe_canonical_shape_on_failure():
    engine = DecisionEngine(MockBackend({}, "bad"), "{{TRANSCRIPT_JSON}}")
    assert engine.decide("x", None, {}, "") == {
        "tool": "idle",
        "args": {},
        "confidence": 0.0,
        "abstain": True,
    }
