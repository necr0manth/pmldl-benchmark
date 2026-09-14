# VLM decision benchmark

Standalone benchmark for a robot-guide decision layer. It evaluates a VLM on a camera frame, visitor transcript, mission state, and optional confirmed context without ROS or direct robot control.

The Python API exposes the project contract:

```python
from vlm_benchmark import configure, decide

configure("configs/openai-compatible.local.json")
result: dict = decide(
    transcript="Что ты видишь?",
    image="frame.png",
    mission_state={"state": "NARRATING"},
    extra_context="",
)
```

`decide(transcript, image, mission_state, extra_context) -> dict` returns `tool`, `args`, `confidence`, and `abstain`. Invalid or failed backend output becomes a canonical abstention; benchmark artifacts retain raw, parsed, and effective/fallback results separately.

## Dataset

The v2 synthetic development set contains 144 cases over 40 images in `datasets/v2`. It covers `start_tour`, `goto_exhibit`, audience assessment, scene-description routing, and policy/input edge cases. The earlier pilot and expanded datasets remain because v2 references their images and assembly data.

## Install and verify

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"

vlm-benchmark v2-validate .\datasets\v2\cases.jsonl
python -m pytest
vlm-benchmark v2-run .\datasets\v2\cases.jsonl --config .\configs\v2-mock.json --output .\runs\v2-mock
```

The mock backend checks plumbing only; it does not measure VLM perception.

## Run a local VLM

Configs use an OpenAI-compatible multimodal endpoint. Copy `configs/openai-compatible.example.json` to an ignored `configs/*.local.json`, then set `base_url`, `model`, timeout, decoding limits, and (if needed) `api_key_env`. Do not put credentials in config files.

For the bundled local layout, downloaded GGUF weights belong under ignored `models/`, while the downloaded llama.cpp runtime belongs under ignored `runtime/llama.cpp-*`. Model hashes and expected paths are recorded in `runtime/comparison/models.json`. On Windows, the included scripts can start and stop the local endpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\runtime\start-local-vlm.ps1
vlm-benchmark v2-run .\datasets\v2\cases.jsonl --config .\configs\v2-qwen3.5-4b-q4km.json --output .\runs\v2-qwen
powershell -ExecutionPolicy Bypass -File .\runtime\stop-local-vlm.ps1
```

`v2-run` defaults to the config's normalization policy. `strict` accepts only a JSON object; `fence-only` additionally removes one outer Markdown code fence and performs no semantic repair. Override explicitly with `--normalization strict` or `--normalization fence-only` when comparing both policies.

Scene-description cases score only the selected tool and abstention behavior. Free scene prose is deliberately not scored automatically.

## Limits

The images are synthetic development data, not a holdout or evidence of real-camera performance. The benchmark does not validate navigation, physical safety, ROS integration, or factual exhibit knowledge. Model weights, downloaded runtimes, logs, and run outputs are local artifacts and are not versioned.
