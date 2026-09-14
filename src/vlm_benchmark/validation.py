from __future__ import annotations

import json
import math
from typing import Any


TOOLS = {"start_tour", "goto_exhibit", "report_audience", "describe_scene", "idle", "ask_visitor"}


class DuplicateKeyError(ValueError):
    pass


def normalize_outer_fence(raw: str, mode: str = "strict") -> tuple[str, bool]:
    """Remove exactly one outer markdown JSON fence; never extracts inner JSON."""
    if mode != "fence-only":
        return raw, False
    stripped = raw.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return raw, False
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"}:
        return "\n".join(lines[1:-1]).strip(), True
    return raw, False


def strict_json_loads(raw: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise DuplicateKeyError(f"duplicate key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"non-finite number: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_decision(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["decision must be an object"]
    errors: list[str] = []
    expected = {"tool", "args", "confidence", "abstain"}
    if set(value) != expected:
        errors.append(f"top-level fields must be exactly {sorted(expected)}")
    tool, args = value.get("tool"), value.get("args")
    confidence, abstain = value.get("confidence"), value.get("abstain")
    if tool not in TOOLS:
        errors.append("unknown tool")
    if not isinstance(args, dict):
        errors.append("args must be an object")
        args = {}
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence must be a number, not bool")
    elif not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append("confidence must be finite and in [0,1]")
    if not isinstance(abstain, bool):
        errors.append("abstain must be boolean")
    schemas = {
        "start_tour": ({"tour_id"}, lambda a: isinstance(a.get("tour_id"), str) and bool(a["tour_id"].strip())),
        "goto_exhibit": (
            {"location_id"},
            lambda a: isinstance(a.get("location_id"), str) and bool(a["location_id"].strip()),
        ),
        "describe_scene": ({"text"}, lambda a: isinstance(a.get("text"), str) and bool(a["text"].strip())),
        "ask_visitor": (
            {"question_id"},
            lambda a: isinstance(a.get("question_id"), str) and bool(a["question_id"].strip()),
        ),
        "idle": (set(), lambda a: True),
        "report_audience": (
            {"visible_count", "facing_robot_count", "recommendation"},
            lambda a: _is_int(a.get("visible_count"))
            and a["visible_count"] >= 0
            and _is_int(a.get("facing_robot_count"))
            and 0 <= a["facing_robot_count"] <= a["visible_count"]
            and a.get("recommendation") in {"continue", "wait"},
        ),
    }
    if tool in schemas:
        fields, predicate = schemas[tool]
        if set(args) != fields or not predicate(args):
            errors.append(f"invalid args for {tool}")
    if abstain is True and not (tool == "idle" and args == {}):
        errors.append("abstain must use canonical idle with empty args")
    return errors


def decision_matches(value: dict[str, Any], acceptable: list[dict[str, Any]]) -> bool:
    for candidate in acceptable:
        if value.get("tool") != candidate.get("tool") or value.get("abstain") != candidate.get("abstain"):
            continue
        if value.get("tool") == "describe_scene" or value.get("args") == candidate.get("args"):
            return True
    return False


def actions_match(actions: list[dict[str, Any]], acceptable: list[list[dict[str, Any]]]) -> bool:
    for candidate_actions in acceptable:
        if len(actions) != len(candidate_actions):
            continue
        matched = True
        for action, candidate in zip(actions, candidate_actions):
            if action.get("tool") != candidate.get("tool"):
                matched = False
                break
            if action.get("tool") != "describe_scene" and action.get("args") != candidate.get("args"):
                matched = False
                break
        if matched:
            return True
    return False
