from __future__ import annotations

import statistics
from typing import Any

from .validation import actions_match, decision_matches


def evaluate_case(case: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    if case.get("method") == "decide":
        from .v2 import score_v2
        result = score_v2(case, detail, detail.get("normalization", "strict"))
        result["method"] = "decide"
        return result
    if case.get("method") == "describe_image":
        text = detail.get("raw_response") or ""
        return {"case_id": case["case_id"], "group_id": case["group_id"], "split": case["split"], "method": case["method"], "raw_response": detail.get("raw_response"), "error_kind": detail.get("error_kind"), "scores": {"completion_nonempty": bool(text.strip()), "backend_ok": detail.get("error_kind") != "backend_error"}}
    if case.get("method") == "check_people":
        parsed = detail["parsed"]
        valid = parsed is not None and detail["error_kind"] is None
        expected = case["expected"]
        return {"case_id": case["case_id"], "group_id": case["group_id"], "split": case["split"], "method": case["method"], "raw_response": detail["raw_response"], "parsed": parsed, "error_kind": detail["error_kind"], "scores": {"raw_valid": valid, "action_correct": bool(valid and parsed["tool"] == expected["tool"] and parsed["abstain"] == expected["abstain"]), "effective_success": False}}
    parsed = detail["parsed"]
    valid = parsed is not None and detail["error_kind"] is None
    acceptable = case["oracle"]["acceptable_decisions"]
    tool_correct = bool(valid and any(parsed["tool"] == item["tool"] for item in acceptable))
    semantic_correct = bool(valid and decision_matches(parsed, acceptable))
    # A meaningful idle may be non-abstaining, but it is not subject-action coverage.
    non_abstain = bool(valid and not parsed["abstain"] and parsed["tool"] != "idle")
    result = {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "split": case["split"],
        "raw_response": detail["raw_response"],
        "parsed": parsed,
        "parse_succeeded": detail["parse_succeeded"],
        "validation_errors": detail["validation_errors"],
        "error_kind": detail["error_kind"],
        "effective": detail["effective"],
        "latency_ms": detail["latency_ms"],
        "image_included": detail["image_included"],
        "scores": {
            "raw_valid": valid,
            "tool_correct": tool_correct,
            "raw_semantic_correct": semantic_correct,
            "valid_non_abstain": non_abstain,
            "correct_non_abstain": bool(non_abstain and semantic_correct),
            "effective_success": actions_match(
                detail["effective"]["actions"], case["oracle"]["acceptable_effective_actions"]
            ),
        },
        "scene_review": None,
    }
    scene_rubric = case["oracle"].get("scene_rubric")
    if valid and parsed["tool"] == "describe_scene" and scene_rubric is not None:
        result["scene_review"] = {
            "status": "pending",
            "text": parsed["args"]["text"],
            "scene_rubric": scene_rubric,
        }
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {item.get("method") for item in results}
    if len(methods) > 1:
        return {"n_cases": len(results), "methods": {str(method): summarize([item for item in results if item.get("method") == method]) for method in sorted(methods, key=lambda value: str(value))}}
    if results and results[0].get("method"):
        method = results[0]["method"]
        if method == "describe_image":
            return {"method": method, "n_cases": len(results), "n_backend_errors": sum(x["error_kind"] == "backend_error" for x in results), "n_nonempty": sum(x["scores"]["completion_nonempty"] for x in results), "n_backend_ok": sum(x["scores"]["backend_ok"] for x in results)}
        if method == "check_people":
            return {"method": method, "n_cases": len(results), "n_raw_valid": sum(x["scores"]["raw_valid"] for x in results), "n_action_correct": sum(x["scores"]["action_correct"] for x in results), "n_backend_errors": sum(x["error_kind"] == "backend_error" for x in results)}
        if method == "decide":
            from .v2 import summarize_v2
            return {"method": method, **summarize_v2(results)}
    n = len(results)
    counts = {
        "n_cases": n,
        "n_called": n,
        "n_backend_errors": sum(item["error_kind"] == "backend_error" for item in results),
        "n_raw_parsed": sum(item["parse_succeeded"] for item in results),
        "n_raw_valid": sum(item["scores"]["raw_valid"] for item in results),
        "n_tool_correct": sum(item["scores"]["tool_correct"] for item in results),
        "n_raw_semantic_correct": sum(item["scores"]["raw_semantic_correct"] for item in results),
        "n_valid_non_abstain": sum(item["scores"]["valid_non_abstain"] for item in results),
        "n_correct_non_abstain": sum(item["scores"]["correct_non_abstain"] for item in results),
        "n_effective_success": sum(item["scores"]["effective_success"] for item in results),
    }
    non_abstain_denominator = counts["n_valid_non_abstain"]
    scene_items = [item for item in results if item.get("scene_review") is not None]
    reviewed = [item for item in scene_items if item["scene_review"].get("status") == "reviewed"]
    coverages = [item["scene_review"]["observation_coverage"] for item in reviewed]
    latencies = [item["latency_ms"] for item in results]
    return {
        **counts,
        "rates": {
            "raw_valid": ratio(counts["n_raw_valid"], n),
            "raw_parse": ratio(counts["n_raw_parsed"], n),
            "tool_accuracy": ratio(counts["n_tool_correct"], n),
            "raw_semantic_accuracy": ratio(counts["n_raw_semantic_correct"], n),
            "non_abstain_coverage": ratio(non_abstain_denominator, n),
            "non_abstain_accuracy": ratio(counts["n_correct_non_abstain"], non_abstain_denominator),
            "effective_success": ratio(counts["n_effective_success"], n),
        },
        "scene_content": {
            "eligible": len(scene_items),
            "reviewed": len(reviewed),
            "pending": len(scene_items) - len(reviewed),
            "review_coverage": ratio(len(reviewed), len(scene_items)),
            "pass_rate": ratio(sum(item["scene_review"].get("verdict") == "pass" for item in reviewed), len(reviewed)),
            "mean_observation_coverage": round(statistics.fmean(coverages), 6) if coverages else None,
            "unsupported_claims": sum(item["scene_review"].get("unsupported_claims", 0) for item in reviewed),
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
