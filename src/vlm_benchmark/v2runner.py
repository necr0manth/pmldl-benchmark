from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import configure
from .v2 import load_v2_cases, score_v2, summarize_v2


def run_v2(dataset: str | Path, config: str | Path, output: str | Path, normalization: str | None = None) -> Path:
    dataset_path, config_path, output_path = Path(dataset).resolve(), Path(config).resolve(), Path(output).resolve()
    if output_path.exists() and any((output_path / name).exists() for name in ("cases.jsonl", "manifest.json", "summary.json")):
        raise ValueError(f"refusing to overwrite existing v2 run directory: {output_path}")
    cases = load_v2_cases(str(dataset_path))
    output_path.mkdir(parents=True, exist_ok=True)
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    normalization = normalization or config_data.get("output_normalization", "strict")
    engine = configure(config_path)
    results: list[dict[str, Any]] = []
    for case in cases:
        image = {"status": "available", "path": str(dataset_path.parent / case["image"]["path"])}
        detail = engine.decide_detailed(case["transcript"], image, case["mission_state"], case["extra_context"])
        results.append(score_v2(case, detail, normalization))
    (output_path / "cases.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in results), encoding="utf-8")
    (output_path / "summary.json").write_text(json.dumps(summarize_v2(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt_path = (config_path.parent / config_data["prompt_path"]).resolve()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "benchmark_version": "2.0.0", "normalization": normalization,
                "dataset": str(dataset_path), "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "prompt": str(prompt_path), "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                "case_count": len(cases), "limitations": ["synthetic dev only", "scene free text not scored"]}
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
