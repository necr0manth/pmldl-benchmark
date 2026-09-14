import json
from pathlib import Path


def make_case(case_id="case", group_id="group", split="dev", transcript="test"):
    return {
        "case_id": case_id,
        "group_id": group_id,
        "split": split,
        "transcript": transcript,
        "image": {"status": "missing", "path": None},
        "mission_state": {"state": "IDLE"},
        "extra_context": "",
        "oracle": {
            "acceptable_decisions": [{"tool": "idle", "args": {}, "abstain": True}],
            "acceptable_effective_actions": [
                [
                    {"tool": "idle", "args": {}},
                    {"tool": "ask_visitor", "args": {"question_id": "clarify"}},
                ]
            ],
            "scene_rubric": None,
        },
        "metadata": {"source": "other", "label_policy_version": "v1"},
    }


def write_cases(path: Path, cases):
    path.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
