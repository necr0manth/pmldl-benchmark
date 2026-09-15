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


def test_configurable_people_and_description_prompts():
    backend = MessageCapture('{"tool":"idle","args":{},"confidence":0.9,"abstain":false}')
    engine = DecisionEngine(
        backend,
        people_prompt_template="CUSTOM_PEOPLE_PROMPT",
        description_prompt_template="CUSTOM_DESCRIPTION_PROMPT",
    )
    engine.check_people(b"fake_image")
    assert backend.messages[0]["role"] == "system"
    assert backend.messages[0]["content"] == "CUSTOM_PEOPLE_PROMPT"

    engine.describe_image(b"fake_image")
    assert backend.messages[0]["role"] == "system"
    assert backend.messages[0]["content"] == "CUSTOM_DESCRIPTION_PROMPT"


def test_configure_loads_custom_prompt_paths(tmp_path):
    p_people = tmp_path / "custom_people.txt"
    p_people.write_text("CUSTOM_PEOPLE_FROM_FILE", encoding="utf-8")
    p_desc = tmp_path / "custom_desc.txt"
    p_desc.write_text("CUSTOM_DESC_FROM_FILE", encoding="utf-8")

    from vlm_benchmark.core import configure
    engine = configure({
        "backend": {"type": "mock", "default_response": '{"tool":"idle","args":{},"confidence":0,"abstain":true}'},
        "people_prompt_path": str(p_people),
        "description_prompt_path": str(p_desc),
    })
    assert engine.people_prompt_template == "CUSTOM_PEOPLE_FROM_FILE"
    assert engine.description_prompt_template == "CUSTOM_DESC_FROM_FILE"


def test_train_dataset_validity_and_isolation():
    from pathlib import Path
    from vlm_benchmark.dataset import load_datasets

    root = Path(__file__).resolve().parents[1]
    train_cases = root / "datasets" / "train" / "cases.jsonl"
    methods_cases = root / "datasets" / "methods" / "cases.jsonl"

    if train_cases.is_file():
        # Validate train alone
        cases = load_datasets([train_cases])
        assert len(cases) == 330
        assert all(c["split"] == "train" for c in cases)

        # Validate no split/group leakage with dev benchmark
        combined = load_datasets([methods_cases, train_cases])
        assert len(combined) == 125 + 330

