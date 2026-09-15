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
PEOPLE_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "people-system.txt"
DESCRIPTION_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "description-system.txt"
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
    def __init__(
        self,
        backend: Backend,
        prompt_template: str | None = None,
        output_normalization: str = "strict",
        people_prompt_template: str | None = None,
        description_prompt_template: str | None = None,
    ):
        self.backend = backend
        self.prompt_template = prompt_template or DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.people_prompt_template = people_prompt_template or PEOPLE_PROMPT.read_text(encoding="utf-8")
        self.description_prompt_template = description_prompt_template or DESCRIPTION_PROMPT.read_text(encoding="utf-8")
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

    def check_people_detailed(self, image: Any) -> dict[str, Any]:
        image_status, image_bytes, mime_type = normalize_image(image)
        return self._image_json_call(self.people_prompt_template, image_status, image_bytes, mime_type, validate_people)

    def check_people(self, image: Any) -> dict:
        detail = self.check_people_detailed(image)
        if detail["error_kind"] is None and detail["parsed"] is not None:
            return detail["parsed"]
        return {"tool": "idle", "args": {}, "confidence": 0.0, "abstain": True}

    def describe_image_detailed(self, image: Any) -> dict[str, Any]:
        image_status, image_bytes, mime_type = normalize_image(image)
        return self._image_text_call(self.description_prompt_template, image_status, image_bytes, mime_type)

    def describe_image(self, image: Any) -> str:
        detail = self.describe_image_detailed(image)
        return detail["raw_response"] if detail["error_kind"] is None and detail["raw_response"].strip() else ""

    def _image_json_call(self, system_prompt: str, image_status: str, image_bytes: bytes | None, mime_type: str | None, validator: Any) -> dict[str, Any]:
        started = time.perf_counter()
        raw_response = None
        parsed = None
        errors: list[str] = []
        error_kind = None
        try:
            raw_response = self.backend.generate_messages(_image_messages(system_prompt, image_bytes, mime_type))
            normalized, normalization_applied = normalize_outer_fence(raw_response, self.output_normalization)
            loaded = strict_json_loads(normalized)
            if isinstance(loaded, dict):
                parsed = loaded
            errors.extend(validator(loaded))
            if errors:
                error_kind = "schema_error"
        except (ValueError, json.JSONDecodeError) as exc:
            error_kind = "invalid_json"
            errors.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            error_kind = "backend_error"
            errors.append(f"{type(exc).__name__}: {exc}")
        return {"raw_response": raw_response, "parsed": parsed, "parse_succeeded": parsed is not None, "validation_errors": errors, "error_kind": error_kind, "normalization_applied": locals().get("normalization_applied", False), "effective": fallback(error_kind or "valid"), "latency_ms": round((time.perf_counter() - started) * 1000, 3), "image_included": image_bytes is not None}

    def _image_text_call(self, system_prompt: str, image_status: str, image_bytes: bytes | None, mime_type: str | None) -> dict[str, Any]:
        started = time.perf_counter()
        raw_response = None
        error_kind = None
        errors: list[str] = []
        try:
            raw_response = self.backend.generate_messages(_image_messages(system_prompt, image_bytes, mime_type))
        except Exception as exc:
            error_kind = "backend_error"
            errors.append(f"{type(exc).__name__}: {exc}")
        return {"raw_response": raw_response, "parsed": None, "parse_succeeded": bool(raw_response and raw_response.strip()), "validation_errors": errors, "error_kind": error_kind, "effective": None, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "image_included": image_bytes is not None}


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


def _image_messages(system_prompt: str, image_bytes: bytes | None, mime_type: str | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image_bytes is not None:
        import base64
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type or 'application/octet-stream'};base64,{base64.b64encode(image_bytes).decode('ascii')}"}})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def validate_people(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["people decision must be an object"]
    errors: list[str] = []
    if set(value) != {"tool", "args", "confidence", "abstain"}:
        errors.append("top-level fields must be exactly tool,args,confidence,abstain")
    if value.get("tool") not in {"idle", "interrupt"}:
        errors.append("tool must be idle or interrupt")
    if not isinstance(value.get("args"), dict):
        errors.append("args must be an object")
    elif value.get("tool") == "idle" and value["args"] != {}:
        errors.append("idle args must be empty")
    elif value.get("tool") == "interrupt" and not set(value["args"]).issubset({"reason", "people_count", "looking_at_robot"}):
        errors.append("interrupt args contain unsupported fields")
    if isinstance(value.get("confidence"), bool) or not isinstance(value.get("confidence"), (int, float)) or not 0 <= value.get("confidence", -1) <= 1:
        errors.append("confidence must be finite and in [0,1]")
    if not isinstance(value.get("abstain"), bool):
        errors.append("abstain must be boolean")
    if value.get("abstain") is True and not (value.get("tool") == "idle" and value.get("args") == {}):
        errors.append("abstain must use canonical idle with empty args")
    return errors


_default_engine: DecisionEngine | None = None


def configure(config: str | Path | dict[str, Any]) -> DecisionEngine:
    global _default_engine
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else None
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path else dict(config)

    def _read_prompt(key: str) -> str | None:
        path_str = data.get(key)
        if not path_str:
            return None
        path = Path(path_str)
        if config_path and not path.is_absolute():
            path = config_path.parent / path
        return path.read_text(encoding="utf-8")

    template = _read_prompt("prompt_path")
    people_template = _read_prompt("people_prompt_path")
    description_template = _read_prompt("description_prompt_path")

    _default_engine = DecisionEngine(
        backend=backend_from_config(data["backend"]),
        prompt_template=template,
        output_normalization=data.get("output_normalization", "strict"),
        people_prompt_template=people_template,
        description_prompt_template=description_template,
    )
    return _default_engine


def decide(transcript: str, image: Any, mission_state: dict[str, Any], extra_context: str) -> dict:
    if _default_engine is None:
        raise RuntimeError("call vlm_benchmark.configure(...) before decide(...)")
    return _default_engine.decide(transcript, image, mission_state, extra_context)


def check_people(image: Any) -> dict:
    if _default_engine is None:
        raise RuntimeError("call vlm_benchmark.configure(...) before check_people(...)")
    return _default_engine.check_people(image)


def describe_image(image: Any) -> str:
    if _default_engine is None:
        raise RuntimeError("call vlm_benchmark.configure(...) before describe_image(...)")
    return _default_engine.describe_image(image)
