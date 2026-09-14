from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class Backend(Protocol):
    backend_type: str

    def generate(self, prompt: str, image: bytes | None, mime_type: str | None) -> str: ...


@dataclass
class MockBackend:
    """Deterministic transport-free stub. It is not a VLM quality check."""

    responses: dict[str, str]
    default_response: str
    backend_type: str = "mock"

    def generate(self, prompt: str, image: bytes | None, mime_type: str | None) -> str:
        for marker, response in self.responses.items():
            if marker in prompt:
                return response
        return self.default_response


@dataclass
class OpenAICompatibleBackend:
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 30.0
    max_tokens: int = 256
    temperature: float = 0.0
    backend_type: str = "openai-compatible"

    def generate(self, prompt: str, image: bytes | None, mime_type: str | None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            encoded = base64.b64encode(image).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"},
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        message = body["choices"][0]["message"]["content"]
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if isinstance(part, dict))
        raise RuntimeError("response message content is not text")


def backend_from_config(config: dict[str, Any]) -> Backend:
    kind = config.get("type")
    if kind == "mock":
        return MockBackend(
            responses=dict(config.get("responses", {})),
            default_response=config.get(
                "default_response", '{"tool":"idle","args":{},"confidence":0,"abstain":true}'
            ),
        )
    if kind == "openai-compatible":
        return OpenAICompatibleBackend(
            base_url=config["base_url"],
            model=config["model"],
            api_key_env=config.get("api_key_env"),
            timeout_s=float(config.get("timeout_s", 30)),
            max_tokens=int(config.get("max_tokens", 256)),
            temperature=float(config.get("temperature", 0)),
        )
    raise ValueError(f"unsupported backend type: {kind!r}")
