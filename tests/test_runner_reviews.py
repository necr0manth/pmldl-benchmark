import json
from pathlib import Path

import pytest

from conftest import make_case, write_cases
from vlm_benchmark.reviews import export_reviews, import_reviews
from vlm_benchmark.runner import run_benchmark


def write_config(path: Path, response: str):
    path.write_text(
        json.dumps({"name": "test", "backend": {"type": "mock", "default_response": response}}),
        encoding="utf-8",
    )


def test_errors_remain_in_denominators_and_effective_is_separate(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    write_cases(dataset, [make_case("a"), make_case("b", "g2")])
    config = tmp_path / "config.json"
    write_config(config, "invalid")
    output = run_benchmark([dataset], config, tmp_path / "run")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_cases"] == summary["n_called"] == 2
    assert summary["n_raw_valid"] == 0
    assert summary["n_effective_success"] == 2
    assert summary["rates"]["raw_semantic_accuracy"] == 0


def test_scene_text_is_pending_until_manual_review(tmp_path):
    case = make_case("scene")
    case["oracle"] = {
        "acceptable_decisions": [{"tool": "describe_scene", "args": {"text": "not compared"}, "abstain": False}],
        "acceptable_effective_actions": [[{"tool": "describe_scene", "args": {"text": "not compared"}}]],
        "scene_rubric": {
            "required_observations": ["one person"],
            "allowed_context_facts": [],
            "disallowed_claims": ["invented identity"],
        },
    }
    dataset = tmp_path / "cases.jsonl"
    write_cases(dataset, [case])
    config = tmp_path / "config.json"
    write_config(
        config,
        '{"tool":"describe_scene","args":{"text":"A person is visible."},"confidence":0.7,"abstain":false}',
    )
    output = run_benchmark([dataset], config, tmp_path / "run")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_raw_semantic_correct"] == 1
    assert summary["scene_content"]["pending"] == 1
    assert summary["scene_content"]["pass_rate"] is None
    todo = tmp_path / "todo.jsonl"
    assert export_reviews(output / "cases.jsonl", todo) == 1
    tampered = json.loads(todo.read_text(encoding="utf-8"))
    tampered.update(
        {
            "reviewer": "human",
            "verdict": "pass",
            "observation_coverage": 1.0,
            "unsupported_claims": 0,
            "notes": "tampered",
            "text": "replacement model output",
        }
    )
    tampered_path = tmp_path / "tampered.jsonl"
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable text"):
        import_reviews(output, tampered_path)
    review = tmp_path / "done.jsonl"
    review.write_text(
        json.dumps(
            {
                "case_id": "scene",
                "reviewer": "human",
                "verdict": "pass",
                "observation_coverage": 1.0,
                "unsupported_claims": 0,
                "notes": "checked against image",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert import_reviews(output, review) == 1
    reviewed = json.loads((output / "summary.json").read_text())["scene_content"]
    assert reviewed["review_coverage"] == 1 and reviewed["pass_rate"] == 1


def test_idle_non_abstain_does_not_inflate_subject_action_coverage(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    case = make_case("idle")
    case["oracle"]["acceptable_decisions"] = [{"tool": "idle", "args": {}, "abstain": False}]
    case["oracle"]["acceptable_effective_actions"] = [[{"tool": "idle", "args": {}}]]
    write_cases(dataset, [case])
    config = tmp_path / "config.json"
    write_config(config, '{"tool":"idle","args":{},"confidence":0.8,"abstain":false}')
    output = run_benchmark([dataset], config, tmp_path / "run")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_raw_semantic_correct"] == 1
    assert summary["n_valid_non_abstain"] == 0
    assert summary["rates"]["non_abstain_accuracy"] is None


def test_unexpected_scene_tool_on_non_scene_case_does_not_crash(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    case = make_case("unexpected-scene")
    case["oracle"].pop("scene_rubric", None)
    write_cases(dataset, [case])
    config = tmp_path / "config.json"
    write_config(
        config,
        '{"tool":"describe_scene","args":{"text":"Visible objects."},"confidence":0.7,"abstain":false}',
    )
    output = run_benchmark([dataset], config, tmp_path / "run")
    result = json.loads((output / "cases.jsonl").read_text())
    assert result["scores"]["raw_valid"] is True
    assert result["scores"]["tool_correct"] is False
    assert result["scene_review"] is None
