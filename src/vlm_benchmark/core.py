from __future__ import annotations

import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

from .backend import Backend, backend_from_config
from .validation import normalize_outer_fence, strict_json_loads, validate_decision

DEFAULT_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "decision-v1.txt"
MISSION_FIELDS = {
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


class DecisionEngine:
    def __init__(self, backend: Backend, prompt_template: str | None = None, output_normalization: str = "strict"):
        self.backend = backend
        self.prompt_template = prompt_template or DEFAULT_PROMPT.read_text(encoding="utf-8")
        if output_normalization not in {"strict", "fence-only"}:
            raise ValueError(f"unsupported output normalization: {output_normalization}")
        self.output_normalization = output_normalization

    def build_prompt(
        self, transcript: str, image_status: str, mission_state: dict[str, Any], extra_context: str
    ) -> str:
        clean_state = {key: value for key, value in mission_state.items() if key in MISSION_FIELDS}
        substitutions = {
            "{{TRANSCRIPT_JSON}}": json.dumps(transcript, ensure_ascii=False, allow_nan=False),
            "{{IMAGE_STATUS_JSON}}": json.dumps(image_status, ensure_ascii=False),
            "{{MISSION_STATE_JSON}}": json.dumps(clean_state, ensure_ascii=False, allow_nan=False, sort_keys=True),
            "{{EXTRA_CONTEXT}}": extra_context,
        }
        pattern = re.compile("|".join(re.escape(marker) for marker in substitutions))
        return pattern.sub(lambda match: substitutions[match.group(0)], self.prompt_template)

    def decide_detailed(
        self, transcript: str, image: Any, mission_state: dict[str, Any], extra_context: str
    ) -> dict[str, Any]:
        image_status, image_bytes, mime_type = normalize_image(image)
        prompt = self.build_prompt(transcript, image_status, mission_state, extra_context)
        started = time.perf_counter()
        raw_response: str | None = None
        normalized_response: str | None = None
        normalization_applied = False
        parsed: dict[str, Any] | None = None
        parse_succeeded = False
        errors: list[str] = []
        error_kind: str | None = None
        try:
            raw_response = self.backend.generate(prompt, image_bytes, mime_type)
        except Exception as exc:  # backend failures are benchmark observations
            error_kind = "backend_error"
            errors.append(f"{type(exc).__name__}: {exc}")
        if raw_response is not None:
            try:
                normalized_response, normalization_applied = normalize_outer_fence(raw_response, self.output_normalization)
                loaded = strict_json_loads(normalized_response)
                parse_succeeded = True
                if isinstance(loaded, dict):
                    parsed = loaded
                validation_errors = validate_decision(loaded)
                errors.extend(validation_errors)
                if validation_errors:
                    error_kind = "schema_error"
            except (ValueError, json.JSONDecodeError) as exc:
                error_kind = "invalid_json"
                errors.append(f"{type(exc).__name__}: {exc}")
        if error_kind is not None or parsed is None:
            effective = fallback(error_kind or "invalid_response")
        elif parsed["abstain"]:
            effective = fallback("model_abstained")
        else:
            effective = {
                "status": "accepted",
                "reason": "valid_model_decision",
                "actions": [{"tool": parsed["tool"], "args": parsed["args"]}],
            }
        return {
            "raw_response": raw_response,
            "normalized_response": normalized_response,
            "normalization": self.output_normalization,
            "normalization_applied": normalization_applied,
            "parsed": parsed,
            "parse_succeeded": parse_succeeded,
            "validation_errors": errors,
            "error_kind": error_kind,
            "effective": effective,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "image_included": image_bytes is not None,
        }

    def decide(self, transcript: str, image: Any, mission_state: dict[str, Any], extra_context: str) -> dict:
        detail = self.decide_detailed(transcript, image, mission_state, extra_context)
        if detail["error_kind"] is None and detail["parsed"] is not None:
            return detail["parsed"]
        return {"tool": "idle", "args": {}, "confidence": 0.0, "abstain": True}


def fallback(reason: str) -> dict[str, Any]:
    return {
        "status": "fallback",
        "reason": reason,
        "actions": [
            {"tool": "idle", "args": {}},
            {"tool": "ask_visitor", "args": {"question_id": "clarify"}},
        ],
    }


def normalize_image(image: Any) -> tuple[str, bytes | None, str | None]:
    if image is None:
        return "missing", None, None
    status = "available"
    mime_type: str | None = None
    if isinstance(image, dict):
        status = image.get("status", "unknown")
        if status != "available":
            return status, None, None
        mime_type = image.get("mime_type")
        image = image.get("data", image.get("path"))
    if isinstance(image, bytes):
        return status, image, mime_type or "application/octet-stream"
    path = Path(image)
    return status, path.read_bytes(), mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


_default_engine: DecisionEngine | None = None


def configure(config: str | Path | dict[str, Any]) -> DecisionEngine:
    global _default_engine
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else None
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path else dict(config)
    prompt_path = data.get("prompt_path")
    if prompt_path:
        path = Path(prompt_path)
        if config_path and not path.is_absolute():
            path = config_path.parent / path
        template = path.read_text(encoding="utf-8")
    else:
        template = None
    _default_engine = DecisionEngine(backend_from_config(data["backend"]), template, data.get("output_normalization", "strict"))
    return _default_engine


def decide(transcript: str, image: Any, mission_state: dict[str, Any], extra_context: str) -> dict:
    if _default_engine is None:
        raise RuntimeError("call vlm_benchmark.configure(...) before decide(...)")
    return _default_engine.decide(transcript, image, mission_state, extra_context)
