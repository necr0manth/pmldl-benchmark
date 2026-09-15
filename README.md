# VLM decision benchmark

Standalone benchmark for a robot-guide decision layer. It sends a visitor transcript, one camera image, mission state, and optional confirmed context to a multimodal model, then evaluates the returned tool decision. It has no ROS executor and does not control robot motion.

The repository includes the complete synthetic v2 development dataset: 144 cases and 40 images. Model weights, downloaded inference runtimes, and run outputs are intentionally not stored in Git.

## Install

Use Python 3.10 or newer. After cloning, run all commands below in the repository root with Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Validate and run the mock

```powershell
vlm-benchmark v2-validate .\datasets\v2\cases.jsonl
python -m pytest
vlm-benchmark v2-run .\datasets\v2\cases.jsonl `
  --config .\configs\v2-mock.json `
  --output .\runs\v2-mock-01
```

The bundled mock always returns a canonical abstention. It verifies dataset loading, prompt assembly, validation, scoring, and output writing; its score is not a model-quality or perception result.

## Three-method API and image-only cases

The public API has three methods: the backward-compatible `decide(transcript, image, mission_state, extra_context) -> dict`, `check_people(image) -> dict`, and `describe_image(image) -> str`. Method cases set one `method` field; image-only cases contain only their image and method-specific expected output.

Validate and run the unified 125-case method set without a VLM (57 `decide`, 29 `check_people`, 39 `describe_image`):

```powershell
$env:PYTHONPATH = "src"
python -m vlm_benchmark validate .\datasets\methods\cases.jsonl
python -m vlm_benchmark run .\datasets\methods\cases.jsonl `
  --config .\configs\v2-mock.json `
  --output C:\Projects\pmldl\runs\methods-mock
```

`check_people` scores raw JSON validity and idle/interrupt action accuracy. `describe_image` records only non-empty completion and backend success; it does not judge prose content or turn a selected method into semantic correctness. The frozen v2 dataset and its runner remain available unchanged.

For the four sequential unified model runs, use `configs/methods-qwen3.5-4b-q4km.json`, `configs/methods-gemma3-4b-it-q4km.json`, `configs/methods-internvl3.5-4b-q4km.json`, and `configs/methods-smolvlm2-2.2b-q4km.json`. They retain the v2 decision prompt, `fence-only` normalization, and existing backend model aliases. The 57 `decide` rows use v2 field masks/scoring; the two image-only methods use their pinned prompts and method-specific scores.

### Method-specific prompt configuration

By default, the benchmark engine uses:
- `prompts/decision-v2.txt` for `decide`
- `prompts/people-system.txt` for `check_people`
- `prompts/description-system.txt` for `describe_image`

You can customize the prompt for each method independently in the run configuration JSON:
- `prompt_path` (or `decision_prompt_path`): Path to the `decide` prompt template.
- `people_prompt_path`: Path to the `check_people` prompt template.
- `description_prompt_path`: Path to the `describe_image` prompt template.

All configured prompt files are validated, resolved relative to the config file location, and recorded with their individual SHA-256 hashes in `manifest.json` for full reproducibility.

## Run a real VLM

The backend must accept multimodal OpenAI-compatible requests at `<base_url>/chat/completions`, normally `/v1/chat/completions`. Images are sent as base64 data URLs in message content.

### Option A: use an existing endpoint

Start from the v2 config so the v2 prompt and normalization policy are preserved:

```powershell
Copy-Item .\configs\v2-qwen3.5-4b-q4km.json .\configs\my-model.local.json
```

Edit the ignored `configs/my-model.local.json`. A complete example is:

```json
{
  "name": "my-v2-model",
  "prompt_path": "../prompts/decision-v2.txt",
  "output_normalization": "fence-only",
  "backend": {
    "type": "openai-compatible",
    "base_url": "http://127.0.0.1:8080/v1",
    "model": "exact-served-model-id",
    "api_key_env": "MY_VLM_API_KEY",
    "timeout_s": 180,
    "max_tokens": 256,
    "temperature": 0
  }
}
```

Set `name` to a descriptive run name, `base_url` to the endpoint prefix, and `model` to the exact model ID served by that endpoint. If authentication is unnecessary, remove `api_key_env`. Otherwise put only the environment-variable name in JSON and set the secret outside the repository:

```powershell
$env:MY_VLM_API_KEY = "your-secret"
```

Do not copy `configs/openai-compatible.example.json` for v2: it has no `prompt_path`, so the engine would select the v1 prompt and the v2 run manifest step would fail.

Run each attempt into a new output directory:

```powershell
vlm-benchmark v2-run .\datasets\v2\cases.jsonl `
  --config .\configs\my-model.local.json `
  --output .\runs\my-model-01
```

The runner writes `cases.jsonl`, `summary.json`, and `manifest.json` only after all 144 calls finish. It does not provide partial checkpoints, and it refuses to overwrite a directory containing an earlier completed run.

### Option B: reproduce the local Qwen setup on Windows Vulkan

Prerequisites are 64-bit Windows, a Vulkan-capable GPU with a working Vulkan driver, Python as above, and sufficient RAM, VRAM, and free disk space. The two model files alone use about 3.4 GB; actual speed and GPU offload depend on the hardware and driver.

The pinned artifacts are:

| Artifact | Pinned source | Expected SHA-256 |
| --- | --- | --- |
| llama.cpp b10941 Windows x64 Vulkan | [GitHub release asset](https://github.com/ggml-org/llama.cpp/releases/download/b10941/llama-b10941-bin-win-vulkan-x64.zip) | `daa2475068ceeef57aa0e5c70dde4739991ab1d0cc4aa601f9bd905ad96be79a` |
| Qwen3.5-4B Q4_K_M | [Hugging Face revision e87f176](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf?download=true) | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Qwen3.5-4B multimodal projector F16 | [Hugging Face revision e87f176](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/mmproj-F16.gguf?download=true) | `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864` |

Download and lay them out exactly as expected by `runtime/start-local-vlm.ps1`:

```powershell
$qwenRevision = "e87f176479d0855a907a41277aca2f8ee7a09523"
$llamaArchive = ".\runtime\downloads\llama-b10941-bin-win-vulkan-x64.zip"

New-Item -ItemType Directory -Force .\runtime\downloads | Out-Null
New-Item -ItemType Directory -Force .\models\Qwen3.5-4B-GGUF | Out-Null

Invoke-WebRequest `
  -Uri "https://github.com/ggml-org/llama.cpp/releases/download/b10941/llama-b10941-bin-win-vulkan-x64.zip" `
  -OutFile $llamaArchive
Expand-Archive -LiteralPath $llamaArchive `
  -DestinationPath .\runtime\llama.cpp-b10941-vulkan -Force

Invoke-WebRequest `
  -Uri "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/$qwenRevision/Qwen3.5-4B-Q4_K_M.gguf?download=true" `
  -OutFile .\models\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf
Invoke-WebRequest `
  -Uri "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/$qwenRevision/mmproj-F16.gguf?download=true" `
  -OutFile .\models\Qwen3.5-4B-GGUF\mmproj-F16.gguf

Get-FileHash -Algorithm SHA256 $llamaArchive
Get-FileHash -Algorithm SHA256 .\models\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf
Get-FileHash -Algorithm SHA256 .\models\Qwen3.5-4B-GGUF\mmproj-F16.gguf
```

Extract the whole archive, not only `llama-server.exe`: all DLLs must remain beside the executable. Before starting, create the log directory because the current start script does not create it:

```powershell
New-Item -ItemType Directory -Force .\runtime\logs | Out-Null
powershell -ExecutionPolicy Bypass -File .\runtime\start-local-vlm.ps1

vlm-benchmark v2-run .\datasets\v2\cases.jsonl `
  --config .\configs\v2-qwen3.5-4b-q4km.json `
  --output .\runs\qwen-v2-01

powershell -ExecutionPolicy Bypass -File .\runtime\stop-local-vlm.ps1
```

The bundled start/stop scripts are specifically wired to the Qwen paths and alias above, and safely handle path names containing spaces (e.g. OneDrive or desktop directories). For a different VLM server, start it with its own tooling and use an Option A config.

## Prompt ablation study and run comparison

The repository includes pre-built configs and prompt templates to evaluate the impact of in-context exemplars on small VLMs (e.g. Qwen3.5-4B):

- **Zero-Shot (`0shot`)**: `configs/methods-qwen3.5-4b-0shot.json` (uses standard instructions without exemplars).
- **One-Shot (`1shot`)**: `configs/methods-qwen3.5-4b-1shot.json` (uses `prompts/decision-v2-oneshot.txt` and `prompts/people-oneshot.txt`).
- **Few-Shot (`fewshot`)**: `configs/methods-qwen3.5-4b-fewshot.json` (uses `prompts/decision-v2-fewshot.txt` and `prompts/people-fewshot.txt` with balanced exemplars for attentive visitor, empty scene, turned away, and ambiguous).

### Running the ablation suite

With the local VLM server running:

```powershell
python -m vlm_benchmark run .\datasets\methods\cases.jsonl `
  --config .\configs\methods-qwen3.5-4b-0shot.json `
  --output .\runs\qwen-prompt-study\methods-qwen3_5-4b-0shot

python -m vlm_benchmark run .\datasets\methods\cases.jsonl `
  --config .\configs\methods-qwen3.5-4b-1shot.json `
  --output .\runs\qwen-prompt-study\methods-qwen3_5-4b-1shot

python -m vlm_benchmark run .\datasets\methods\cases.jsonl `
  --config .\configs\methods-qwen3.5-4b-fewshot.json `
  --output .\runs\qwen-prompt-study\methods-qwen3_5-4b-fewshot
```

### Side-by-side run comparison tool

Use `scripts/compare_runs.py` to compare any set of run directories or an entire study folder:

```powershell
python scripts/compare_runs.py .\runs\qwen-prompt-study
```


## Fine-tuning Qwen3.5-4B with Few-Shot Prompts

The repository includes a complete LoRA / QLoRA fine-tuning pipeline in [`scripts/train_lora_qwen.py`](scripts/train_lora_qwen.py) and a PowerShell launcher in [`scripts/run_finetune.ps1`](scripts/run_finetune.ps1). It fine-tunes `Qwen/Qwen3.5-4B` using the empirical findings from the prompt study:

- **Optimal Few-Shot Prompt Conditioning**: Uses [`prompts/people-fewshot.txt`](prompts/people-fewshot.txt) containing the 4 balanced exemplars (`idle` attentive, `interrupt` empty, `interrupt` turned away, `abstain` ambiguous) that achieved 100% JSON valid schema and 58.6% action accuracy.
- **Selective Loss Masking**: Masks all system prompt, user query, and image tokens to `-100`, computing backpropagation loss exclusively on the assistant's JSON tool decision.
- **8 GB VRAM Optimization**: Uses 4-bit NF4 QLoRA quantization via `bitsandbytes`, gradient checkpointing, and paged AdamW optimizer to run comfortably on consumer laptop GPUs (e.g. RTX 4060 8GB).
- **Multi-task Preservation**: Optionally trains across all three methods (`check_people`, `decide`, `describe_image`) to avoid catastrophic forgetting of robot navigation and tool schemas.

### Quick start

```powershell
# 1. Verify dataset, chat templating, and tokenization without training:
powershell -ExecutionPolicy Bypass -File .\scripts\run_finetune.ps1 -DryRun

# 2. Fine-tune on audience engagement (check_people):
powershell -ExecutionPolicy Bypass -File .\scripts\run_finetune.ps1 -MethodFilter "check_people" -Epochs 3

# 3. Full multi-task training with automated weight merging:
powershell -ExecutionPolicy Bypass -File .\scripts\run_finetune.ps1 -MethodFilter "all" -Epochs 3 -MergeAndSave
```

After training, the LoRA adapter is saved to `outputs/qwen3.5-4b-fewshot-lora`. When `-MergeAndSave` is passed, the script merges the adapter weights with the base model into `outputs/qwen3.5-4b-fewshot-lora-merged` for direct GGUF quantization and serving with `llama-server`.

## Normalization and results

`strict` accepts only a JSON object. `fence-only` may remove one outer Markdown code fence but performs no semantic repair. For a shared comparison, keep `output_normalization: "fence-only"` in the config; `v2-run --normalization strict` can then rescore each new run under the stricter rule. The CLI flag controls v2 scoring, while the public `decide(...)` API uses the config's `output_normalization` policy.

Every output directory contains:

- `summary.json`: `n_cases`; `overall` counts for `n`, `raw_valid`, `composite`, and per-field `correct`/`denominator`; the same breakdown under `categories`, `image_groups`, and `expected_tool_groups`. Audience fields such as `visible_count`, `facing_robot_count`, and `recommendation` appear in the applicable field summaries.
- `cases.jsonl`: per-case `raw_response`, `parsed`, `parse_error`, `effective`, and `scores.raw_valid`, `scores.fields`, and `scores.composite` for diagnosis.
- `manifest.json`: dataset, config, prompt hashes, selected normalization, case count, and stated limitations.

An operational fallback (`idle` followed by a clarification request) is recorded under `effective`; it is not semantic success by the model and does not turn invalid JSON into a correct raw decision.

Scene-description cases score only the selected tool and abstention behavior. Free scene prose is marked `not_scored` and is not judged automatically.

## Python API

After creating `configs/my-model.local.json` and starting its endpoint:

```python
from vlm_benchmark import configure, decide

configure("configs/my-model.local.json")
decision: dict = decide(
    transcript="Что ты видишь?",
    image="frame.png",
    mission_state={"state": "NARRATING"},
    extra_context="",
)
```

Image-only calls are available after the same `configure(...)`: `check_people("frame.png")` and `describe_image("frame.png")`. The real robot chat branch accepts a visitor utterance; `describe_image` intentionally fixes the benchmark contract to the pinned system prompt plus the picture, with no changing utterance/history or keyword router.

`decide(transcript, image, mission_state, extra_context) -> dict` returns `tool`, `args`, `confidence`, and `abstain`. Invalid or failed backend output returns canonical abstention.

## Limits

This is synthetic development data, not a holdout or evidence of real-camera performance. The benchmark does not validate navigation, physical safety, ROS integration, or factual exhibit knowledge.
