from vlm_benchmark.core import DecisionEngine


class MessageCapture:
    backend_type = "capture"

    def __init__(self, response):
        self.response = response
        self.messages = None

    def generate(self, prompt, image, mime_type):
        return self.response

    def generate_messages(self, messages):
        self.messages = messages
        return self.response


def test_image_methods_send_pinned_system_and_image_only_user_message():
    backend = MessageCapture('{"tool":"interrupt","args":{},"confidence":0.8,"abstain":false}')
    engine = DecisionEngine(backend, output_normalization="strict")
    result = engine.check_people(b"image")
    assert result["tool"] == "interrupt"
    assert backend.messages[0]["role"] == "system"
    assert "посетитель" in backend.messages[0]["content"]
    assert backend.messages[1]["role"] == "user"
    assert len(backend.messages[1]["content"]) == 1
    assert backend.messages[1]["content"][0]["type"] == "image_url"


def test_describe_image_is_text_smoke_without_keyword_router():
    backend = MessageCapture("В кадре виден зал.")
    engine = DecisionEngine(backend)
    assert engine.describe_image(b"image") == "В кадре виден зал."
    assert backend.messages[1]["content"]


def test_people_invalid_response_stays_raw_failure_and_public_abstains():
    engine = DecisionEngine(MessageCapture("not json"))
    detail = engine.check_people_detailed(b"image")
    assert detail["error_kind"] == "invalid_json"
    assert detail["parsed"] is None
    assert engine.check_people(b"image")["abstain"] is True
