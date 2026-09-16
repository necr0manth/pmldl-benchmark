from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class Backend(Protocol):
    backend_type: str

    def generate(self, prompt: str, image: bytes | None, mime_type: str | None) -> str: ...

    def generate_messages(self, messages: list[dict[str, Any]]) -> str: ...


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

    def generate_messages(self, messages: list[dict[str, Any]]) -> str:
        text = "\n".join(
            part.get("text", "")
            for message in messages
            for part in (message.get("content", []) if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        return self.generate(text, None, None)


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
        return self._complete({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })

    def generate_messages(self, messages: list[dict[str, Any]]) -> str:
        return self._complete({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })

    def _complete(self, payload: dict[str, Any]) -> str:
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


@dataclass
class TransformersLocalBackend:
    model_id: str = "Qwen/Qwen3.5-4B"
    adapter_path: str | None = None
    quantization: str = "4bit"
    max_tokens: int = 256
    temperature: float = 0.0
    backend_type: str = "transformers-local"

    def __post_init__(self) -> None:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import PeftModel

        self._torch = torch
        self._Image = Image

        if self.adapter_path:
            p = Path(self.adapter_path)
            if not p.is_absolute():
                root = Path(__file__).resolve().parents[2]
                if (root / p).exists():
                    p = root / p
                elif (Path.cwd() / p).exists():
                    p = Path.cwd() / p
            load_path = str(p)
            self.adapter_path = str(p)
        else:
            load_path = self.model_id

        self.processor = AutoProcessor.from_pretrained(load_path)

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant_config = None
        if self.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        base_model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            torch_dtype=compute_dtype if self.quantization == "none" else None,
            device_map="auto",
        )

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self.model = base_model

        self.model.eval()

    def generate(self, prompt: str, image: bytes | None, mime_type: str | None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            import io
            pil_img = self._Image.open(io.BytesIO(image)).convert("RGB")
            content.append({"type": "image", "image": pil_img})
        return self.generate_messages([{"role": "user", "content": content}])

    def generate_messages(self, messages: list[dict[str, Any]]) -> str:
        import io
        formatted_messages = []
        images = []
        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            if isinstance(raw_content, str):
                formatted_messages.append({"role": role, "content": raw_content})
            elif isinstance(raw_content, list):
                parts = []
                for part in raw_content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") in ("image", "image_url"):
                        if "image" in part and hasattr(part["image"], "convert"):
                            pil_img = part["image"]
                        elif "image_url" in part:
                            url = part["image_url"].get("url", "")
                            if url.startswith("data:") and ";base64," in url:
                                b64 = url.split(";base64,")[1]
                                raw_bytes = base64.b64decode(b64)
                                pil_img = self._Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                            else:
                                continue
                        else:
                            continue
                        images.append(pil_img)
                        parts.append({"type": "image", "image": pil_img})
                formatted_messages.append({"role": role, "content": parts})

        text = self.processor.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        kwargs: dict[str, Any] = {"text": [text]}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs, return_tensors="pt").to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


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
    if kind == "transformers-local":
        return TransformersLocalBackend(
            model_id=config.get("model_id", "Qwen/Qwen3.5-4B"),
            adapter_path=config.get("adapter_path"),
            quantization=config.get("quantization", "4bit"),
            max_tokens=int(config.get("max_tokens", 256)),
            temperature=float(config.get("temperature", 0)),
        )
    raise ValueError(f"unsupported backend type: {kind!r}")
