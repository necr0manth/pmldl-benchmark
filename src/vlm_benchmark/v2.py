from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .validation import normalize_outer_fence, strict_json_loads, validate_decision
from .dataset import _finite_tree, _validate_mission_state

V2_TOOLS = {"start_tour", "goto_exhibit", "report_audience", "describe_scene", "idle"}
V2_CATEGORIES = {"audience", "basic", "scene_routing", "unambiguous_input", "policy_edge"}
V2_FIELDS = {
    "start_tour": {"tool", "abstain", "tour_id"},
    "goto_exhibit": {"tool", "abstain", "location_id"},
    "report_audience": {"tool", "abstain", "visible_count", "facing_robot_count", "recommendation"},
    "describe_scene": {"tool", "abstain"}, "idle": {"tool", "abstain"},
}


def load_v2_cases(path: str) -> list[dict[str, Any]]:
    from pathlib import Path
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = strict_json_loads(line)
        errors = validate_v2_case(case, Path(path).parent)
        if errors:
            raise ValueError(f"{path}:{line_no}: " + "; ".join(errors))
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
        cases.append(case)
    if not cases:
        raise ValueError("v2 dataset contains no cases")
    return cases


def validate_v2_case(case: Any, base_dir: Any) -> list[str]:
    from pathlib import Path
    if not isinstance(case, dict):
        return ["case must be an object"]
    required = {"case_id", "group_id", "image_group", "category", "split", "transcript", "image", "mission_state", "extra_context", "expected", "scored_fields", "metadata"}
    errors = [f"missing fields: {sorted(required - set(case))}"] if required - set(case) else []
    if errors:
        return errors
    if not all(isinstance(case[key], str) and case[key] for key in ("case_id", "group_id", "image_group", "category")):
        errors.append("IDs and category must be non-empty strings")
    if case["category"] not in V2_CATEGORIES:
        errors.append("unknown v2 category")
    if case["split"] != "dev":
        errors.append("v2 currently permits only dev split")
    image = case["image"]
    if not isinstance(image, dict) or set(image) != {"status", "path"} or image["status"] != "available" or not isinstance(image["path"], str):
        errors.append("image must be an available {status,path} object")
    elif not (Path(base_dir) / image["path"]).is_file():
        errors.append(f"image does not exist: {image['path']}")
    if not isinstance(case["transcript"], str) or not isinstance(case["mission_state"], dict) or not _finite_tree(case["mission_state"]) or not isinstance(case["extra_context"], str):
        errors.append("mission_state must be object and extra_context string")
    elif _validate_mission_state(case["mission_state"]):
        errors.extend(_validate_mission_state(case["mission_state"]))
    expected = case["expected"]
    if not isinstance(expected, dict) or set(expected) != {"tool", "args", "abstain"}:
        errors.append("expected must contain exactly tool,args,abstain")
    else:
        candidate = {**expected, "confidence": 0.5}
        errors.extend("expected: " + error for error in validate_decision(candidate))
        if expected.get("tool") not in V2_TOOLS:
            errors.append("expected tool is not allowed in v2")
        if expected.get("abstain") and expected.get("tool") != "idle":
            errors.append("abstain must use canonical idle")
    if not isinstance(case["scored_fields"], list) or not case["scored_fields"] or not all(isinstance(x, str) for x in case["scored_fields"]):
        errors.append("scored_fields must be a non-empty string list")
    elif "tool" not in case["scored_fields"]:
        errors.append("scored_fields must include tool")
    elif len(case["scored_fields"]) != len(set(case["scored_fields"])):
        errors.append("scored_fields must not contain duplicates")
    elif isinstance(expected, dict) and expected.get("tool") in V2_FIELDS and set(case["scored_fields"]) != V2_FIELDS[expected["tool"]]:
        errors.append("scored_fields must exactly match the expected tool mask")
    metadata = case["metadata"]
    if not isinstance(metadata, dict) or metadata.get("source") != "synthetic" or metadata.get("benchmark_version") != "v2":
        errors.append("metadata must identify synthetic benchmark v2")
    return errors


def parse_v2(raw: str | None, normalization: str = "strict") -> tuple[dict[str, Any] | None, str | None]:
    if raw is None:
        return None, "missing_response"
    candidate = raw
    if normalization == "fence-only":
        candidate, _ = normalize_outer_fence(raw, normalization)
    try:
        value = strict_json_loads(candidate)
    except (TypeError, ValueError) as exc:
        return None, f"invalid_json: {exc}"
    errors = validate_decision(value)
    if isinstance(value, dict) and value.get("tool") not in V2_TOOLS:
        errors.append("tool is not allowed in v2")
    if errors:
        return None, "invalid_decision: " + "; ".join(errors)
    return value, None


def score_v2(case: dict[str, Any], detail: dict[str, Any], normalization: str = "strict") -> dict[str, Any]:
    if detail.get("error_kind") is not None:
        parsed, parse_error = None, detail.get("error_kind")
    elif detail.get("normalization") == normalization and detail.get("normalized_response") is not None:
        parsed, parse_error = detail.get("parsed"), None if detail.get("error_kind") is None else detail.get("error_kind")
    else:
        parsed, parse_error = parse_v2(detail.get("raw_response"), normalization)
    expected = case["expected"]
    fields = case["scored_fields"]
    valid = parsed is not None
    field_scores: dict[str, bool | None] = {}
    for field in fields:
        if not valid:
            field_scores[field] = False
        elif field == "tool":
            field_scores[field] = parsed["tool"] == expected["tool"]
        elif field == "abstain":
            field_scores[field] = parsed["abstain"] == expected["abstain"]
        else:
            field_scores[field] = parsed["tool"] == expected["tool"] and parsed["args"].get(field) == expected["args"].get(field)
    composite = bool(valid and fields and all(field_scores.values()))
    diagnostic = None
    if valid and expected["tool"] == "describe_scene":
        diagnostic = {"text_nonempty": isinstance(parsed["args"].get("text"), str) and bool(parsed["args"]["text"].strip())}
    return {
        "case_id": case["case_id"], "group_id": case["group_id"], "image_group": case["image_group"],
        "category": case["category"], "expected_tool": expected["tool"], "pair_id": case.get("pair_id"),
        "raw_response": detail.get("raw_response"), "parsed": parsed, "parse_error": parse_error,
        "backend_error": detail.get("error_kind") == "backend_error", "error_kind": detail.get("error_kind"),
        "validation_errors": detail.get("validation_errors", []), "effective": detail.get("effective"),
        "normalized_response": detail.get("normalized_response"), "normalization_applied": detail.get("normalization_applied", False),
        "normalization": normalization, "scores": {"raw_valid": valid, "fields": field_scores, "composite": composite},
        "scene_diagnostic": diagnostic,
    }


def summarize_v2(results: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        fields = sorted({field for item in items for field in item["scores"]["fields"]})
        return {"n": len(items), "raw_valid": sum(item["scores"]["raw_valid"] for item in items),
                "composite": sum(item["scores"]["composite"] for item in items),
                "fields": {field: {"correct": sum(item["scores"]["fields"].get(field) is True for item in items),
                                   "denominator": sum(field in item["scores"]["fields"] for item in items)} for field in fields}}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        by_category[item["category"]].append(item)
        by_image[item["image_group"]].append(item)
        by_tool[item["expected_tool"]].append(item)
    return {"n_cases": len(results), "overall": summary(results), "categories": {key: summary(value) for key, value in sorted(by_category.items())},
            "image_groups": {key: summary(value) for key, value in sorted(by_image.items())},
            "expected_tool_groups": {key: summary(value) for key, value in sorted(by_tool.items())},
            "scene_text_content": "not_scored", "free_text_judgment": "not_used"}
