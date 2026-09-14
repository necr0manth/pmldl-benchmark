from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .validation import strict_json_loads, validate_decision

SPLITS = {"train", "dev", "holdout"}
IMAGE_STATUSES = {"available", "missing", "stale", "unknown"}


def load_datasets(paths: list[str | Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for supplied in paths:
        path = Path(supplied).resolve()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = strict_json_loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            case["_dataset_path"] = str(path)
            case["_line_number"] = line_number
            errors = validate_case(case, path)
            if errors:
                raise ValueError(f"{path}:{line_number}: " + "; ".join(errors))
            case_id, group_id, split = case["case_id"], case["group_id"], case["split"]
            if case_id in seen_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen_ids.add(case_id)
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                raise ValueError(f"group_id {group_id!r} leaks across splits: {previous}, {split}")
            cases.append(case)
    if not cases:
        raise ValueError("dataset contains no cases")
    return cases


def validate_case(case: Any, dataset_path: Path) -> list[str]:
    if not isinstance(case, dict):
        return ["case must be an object"]
    errors: list[str] = []
    if case.get("method") == "decide":
        required = {"case_id", "group_id", "split", "transcript", "image", "mission_state", "extra_context", "expected", "metadata", "method"}
        missing = required - set(case)
        if missing:
            return [f"missing fields: {sorted(missing)}"]
        errors.extend(_validate_image(case["image"], dataset_path))
        if case["split"] not in SPLITS:
            errors.append("split must be train, dev, or holdout")
        if not isinstance(case["metadata"], dict) or case["metadata"].get("source") not in {"synthetic", "real", "other"}:
            errors.append("metadata.source must be synthetic, real, or other")
        if not isinstance(case["mission_state"], dict) or not _finite_tree(case["mission_state"]):
            errors.append("mission_state must be a finite object")
        expected = case["expected"]
        if not isinstance(expected, dict) or not isinstance(expected.get("tool"), str) or not isinstance(expected.get("args"), dict) or not isinstance(expected.get("abstain"), bool):
            errors.append("expected must contain tool,args,abstain")
        return errors
    if case.get("method") in {"check_people", "describe_image"}:
        required = {"case_id", "group_id", "split", "method", "image", "expected", "metadata"}
        missing = required - set(case)
        if missing:
            return [f"missing fields: {sorted(missing)}"]
        if case["method"] == "check_people":
            expected = case["expected"]
            if not isinstance(expected, dict) or expected.get("tool") not in {"idle", "interrupt"} or expected.get("abstain") is not False:
                errors.append("check_people expected must be non-abstaining idle or interrupt")
        elif not isinstance(case["expected"], dict) or case["expected"].get("kind") != "nonempty_text":
            errors.append("describe_image expected must have kind nonempty_text")
        errors.extend(_validate_image(case["image"], dataset_path))
        if case["split"] not in SPLITS:
            errors.append("split must be train, dev, or holdout")
        if not isinstance(case["metadata"], dict) or case["metadata"].get("source") not in {"synthetic", "real", "other"}:
            errors.append("metadata.source must be synthetic, real, or other")
        return errors
    required = {
        "case_id",
        "group_id",
        "split",
        "transcript",
        "image",
        "mission_state",
        "extra_context",
        "oracle",
        "metadata",
    }
    missing = required - set(case)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
        return errors
    for field in ("case_id", "group_id", "transcript", "extra_context"):
        if not isinstance(case[field], str) or (field in {"case_id", "group_id"} and not case[field]):
            errors.append(f"{field} must be a string" + ("" if field in {"transcript", "extra_context"} else " and non-empty"))
    if case["split"] not in SPLITS:
        errors.append("split must be train, dev, or holdout")
    errors.extend(_validate_image(case["image"], dataset_path))
    if not isinstance(case["mission_state"], dict):
        errors.append("mission_state must be an object")
    elif not _finite_tree(case["mission_state"]):
        errors.append("mission_state contains non-finite number")
    else:
        mission_errors = _validate_mission_state(case["mission_state"])
        errors.extend(mission_errors)
    oracle = case["oracle"]
    if not isinstance(oracle, dict):
        errors.append("oracle must be an object")
    else:
        acceptable = oracle.get("acceptable_decisions")
        if not isinstance(acceptable, list) or not acceptable:
            errors.append("oracle.acceptable_decisions must be a non-empty list")
        else:
            for index, decision in enumerate(acceptable):
                if not isinstance(decision, dict):
                    errors.append(f"acceptable_decisions[{index}] must be an object")
                    continue
                candidate = {**decision, "confidence": 0.5}
                decision_errors = validate_decision(candidate)
                if decision_errors:
                    errors.append(f"acceptable_decisions[{index}]: {', '.join(decision_errors)}")
            if any(item.get("tool") == "describe_scene" for item in acceptable if isinstance(item, dict)):
                rubric = oracle.get("scene_rubric")
                rubric_keys = {"required_observations", "allowed_context_facts", "disallowed_claims"}
                if not isinstance(rubric, dict) or set(rubric) != rubric_keys:
                    errors.append("describe_scene requires exact scene_rubric fields")
                elif any(not isinstance(rubric[key], list) or not all(isinstance(x, str) for x in rubric[key]) for key in rubric_keys):
                    errors.append("scene_rubric values must be string lists")
        actions = oracle.get("acceptable_effective_actions")
        if not isinstance(actions, list) or not actions or not all(isinstance(seq, list) for seq in actions):
            errors.append("oracle.acceptable_effective_actions must be a non-empty list of lists")
        else:
            for sequence_index, sequence in enumerate(actions):
                if not sequence:
                    errors.append(f"acceptable_effective_actions[{sequence_index}] must not be empty")
                for action_index, action in enumerate(sequence):
                    if not isinstance(action, dict) or set(action) != {"tool", "args"}:
                        errors.append(
                            f"acceptable_effective_actions[{sequence_index}][{action_index}] must contain tool,args"
                        )
                        continue
                    action_errors = validate_decision(
                        {**action, "confidence": 0.5, "abstain": False}
                    )
                    if action_errors:
                        errors.append(
                            f"acceptable_effective_actions[{sequence_index}][{action_index}]: "
                            + ", ".join(action_errors)
                        )
    metadata = case["metadata"]
    if not isinstance(metadata, dict) or metadata.get("source") not in {"synthetic", "real", "other"}:
        errors.append("metadata.source must be synthetic, real, or other")
    elif not isinstance(metadata.get("label_policy_version"), str):
        errors.append("metadata.label_policy_version must be a string")
    return errors


def _validate_image(image: Any, dataset_path: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(image, dict) or set(image) != {"status", "path"}:
        return ["image must contain exactly status and path"]
    if image["status"] not in IMAGE_STATUSES:
        errors.append("invalid image status")
    elif image["status"] == "available":
        if not isinstance(image["path"], str):
            errors.append("available image requires path")
        elif not (dataset_path.parent / image["path"]).is_file():
            errors.append(f"image does not exist: {image['path']}")
    elif image["path"] is not None:
        errors.append("non-available image path must be null")
    return errors


def fixture_image(case: dict[str, Any]) -> dict[str, Any]:
    image = case["image"]
    if image["status"] != "available":
        return {"status": image["status"], "path": None}
    base = Path(case["_dataset_path"]).parent
    return {"status": "available", "path": str(base / image["path"])}


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def _validate_mission_state(state: dict[str, Any]) -> list[str]:
    allowed = {
        "state",
        "interrupt",
        "stop_index",
        "stop_total",
        "stop_id",
        "exhibit_id",
        "resume_available",
        "resume_token",
        "stamp",
    }
    errors = []
    unknown = set(state) - allowed
    if unknown:
        errors.append(f"unknown mission_state fields: {sorted(unknown)}")
    if "state" in state and state["state"] is not None:
        value = state["state"]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            errors.append("mission_state.state must be string, integer, or null")
    for key in ("interrupt", "resume_available"):
        if key in state and state[key] is not None and not isinstance(state[key], bool):
            errors.append(f"mission_state.{key} must be boolean or null")
    for key in ("stop_index", "stop_total"):
        if key in state and state[key] is not None:
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"mission_state.{key} must be integer >=0 or null")
    for key in ("stop_id", "exhibit_id", "resume_token"):
        if key in state and state[key] is not None and not isinstance(state[key], str):
            errors.append(f"mission_state.{key} must be string or null")
    return errors
