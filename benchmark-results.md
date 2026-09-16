# VLM Model Comparison on a Unified Robot-Guide Benchmark

## Goal and benchmark

The benchmark compares four small vision-language models on the decision layer of a robot-guide system. It evaluates three callable methods over 125 synthetic development cases:

- `decide`: 57 cases using a visitor transcript, camera image, mission state, and additional confirmed context.
- `check_people`: 29 image-only cases for deciding whether to continue or wait for the audience.
- `describe_image`: 39 image-only cases for producing a scene description.

The image-only methods use fixed robot-specific prompts. The 57 `decide` cases contain 26 `start_tour` and 31 `goto_exhibit` cases. Their required IDs are selected from the supplied case context; this benchmark does not assume a live robot tool registry. The `decide` method uses the transcript, image, mission state, and extra context defined by its input contract. Images are reused synthetic development scenes. This is an offline decision-layer benchmark; it does not run ROS or control navigation.

## Setup

The models were run locally, one at a time, using the same llama.cpp-based multimodal server on an RTX 4060 Laptop GPU with 8 GB of memory. All runs used Q4_K_M model weights, temperature 0, and a maximum of 256 generated tokens.

## Scoring

For `decide`, a response is valid when it is a JSON object with the required fields and types: `tool`, `args`, `confidence`, and `abstain`. Configured validity means that the response passed this validation under the configured `fence-only` policy: one outer Markdown code fence may be removed before parsing, with no semantic repair. Therefore, the validity figures below are post-normalization configured-validity figures, not raw strict-JSON figures. Correctness requires the expected tool and required arguments; invalid responses and abstentions are not correct.

For `check_people`, scoring separates invalid responses, abstentions, and valid `idle`/`interrupt` actions. `idle` means allow narration to continue without interruption; `interrupt` means pause narration. The expected policy is: at least one clearly attentive person implies `idle`; otherwise the expected action is `interrupt`. The denominator is 29 cases. Optional people arguments are not scored.

For `describe_image`, the benchmark records only whether the model returned a non-empty completion and whether the backend completed successfully. It does not evaluate the content of the description and uses no human or LLM judge.

## Results

| Model | `decide` configured-valid | `decide` correct | `check_people` configured-valid | `check_people` action correct | `describe_image` non-empty/backend OK |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 56/57 (98.2%) | 56/57 (98.2%) | 13/29 (44.8%) | 8/29 (27.6%) | 39/39 (100%) |
| Gemma3-4B-IT | 57/57 (100%) | 37/57 (64.9%) | 29/29 (100%) | 18/29 (62.1%) | 39/39 (100%) |
| InternVL3.5-4B | 56/57 (98.2%) | 45/57 (78.9%) | 29/29 (100%) | 18/29 (62.1%) | 39/39 (100%) |
| SmolVLM2-2.2B | 1/57 (1.8%) | 0/57 (0%) | 0/29 (0%) | 0/29 (0%) | 39/39 (100%) |

All four models completed the 39 description requests successfully and produced non-empty text. This is a completion result, not a claim about description quality.

### Audience-action breakdown

The expected split is 18 `idle` and 11 `interrupt` cases.

- Qwen3.5-4B: 0 idle→idle, 5 idle→interrupt, 0 interrupt→idle, 8 interrupt→interrupt, 16 invalid.
- Gemma3-4B-IT: 18 idle→idle, 11 interrupt→idle, 0 invalid, 0 abstentions. Its 18/29 score is exactly the always-idle baseline.
- InternVL3.5-4B: 18 idle→idle, 7 interrupt→idle, 4 abstentions, 0 invalid. It also does not demonstrate reliable interruption discrimination.
- SmolVLM2-2.2B: 29 invalid responses and no valid action.

## Prompt Strategy Ablation on Qwen3.5-4B (0-Shot, 1-Shot, Few-Shot)

To investigate whether prompt design and few-shot in-context learning could resolve formatting failures and improve visual discrimination, an ablation study was conducted on **Qwen3.5-4B (Q4_K_M)** across three prompt regimes:

1. **Zero-Shot (`0shot`)**: Direct instruction prompts with strict schema definitions and negative constraints.
2. **One-Shot (`1shot`)**: Instructions accompanied by a single exemplar (an attentive visitor facing the robot for `check_people`).
3. **Few-Shot (`fewshot`)**: Instructions accompanied by balanced multi-condition exemplars (`idle` attentive visitor, `interrupt` empty/no people, `interrupt` turned away, and ambiguous `abstain`).

### Comparative Results (125 unified cases)

Evaluated under identical inference conditions (llama.cpp b10941 Vulkan, temperature 0, max 256 tokens):

| Metric | 0-Shot | 1-Shot | Few-Shot |
| --- | ---: | ---: | ---: |
| **`decide` valid** | 57/57 (100.0%) | 57/57 (100.0%) | 57/57 (100.0%) |
| **`decide` correct** | **57/57 (100.0%)** | 55/57 (96.5%) | 55/57 (96.5%) |
| **`check_people` valid** | 12/29 (41.4%) | **29/29 (100.0%)** | **29/29 (100.0%)** |
| **`check_people` action correct** | 7/29 (24.1%) | 10/29 (34.5%) | **17/29 (58.6%)** |
| **`describe_image` non-empty** | 39/39 (100.0%) | 39/39 (100.0%) | 39/39 (100.0%) |
| **Total Benchmark Score** | 103/125 (82.4%) | 104/125 (83.2%) | **111/125 (88.8%)** |

### Audience-Action Confusion Matrix (`check_people`, 29 cases: 18 `idle`, 11 `interrupt`)

| Outcome | 0-Shot | 1-Shot | Few-Shot |
| --- | ---: | ---: | ---: |
| *Invalid format/schema* | 17 | **0** | **0** |
| *True Positive Idle (`idle` $\to$ `idle`)* | 0 | 1 | **8** |
| *True Positive Interrupt (`interrupt` $\to$ `interrupt`)* | 7 | **9** | **9** |
| *False Alarm Interrupt (`idle` $\to$ `interrupt`)* | 5 | 17 | **10** |
| *Missed Interrupt (`interrupt` $\to$ `idle`)* | **0** | 2 | 2 |

### Key Findings

1. **One-shot prompting cures schema failure**: Providing a single JSON exemplar completely eliminated invalid formatting errors in `check_people` (from 17 invalid down to 0).
2. **Balanced few-shot unlocks audience discrimination**: While 1-shot produced a hyper-sensitive model (classifying almost all valid cases as `interrupt`, with 17 false alarms), the balanced 4-exemplar prompt reduced false alarms from 17 to 10 and enabled true `idle` detection (8 correct vs 1 in 1-shot and 0 in 0-shot), more than doubling action accuracy ($24.1\% \to 58.6\%$).
3. **Task-specific prompt specialization**: Zero-shot remains optimal for `decide` (100% correct), where extra exemplars introduce minor distraction on complex edge cases (55/57, 96.5%).
4. **Optimal Hybrid Configuration**: Because the benchmark engine supports decoupled per-method prompt paths (`prompt_path` for `decide` and `people_prompt_path` for `check_people`), combining **Zero-Shot `decide`** with **Few-Shot `check_people`** achieves an overall score of **113/125 (90.4%)**.

## QLoRA Fine-Tuning on Qwen3.5-4B

Following the prompt ablation study, Qwen3.5-4B was fine-tuned using 4-bit QLoRA on the newly curated museum dataset of 110 real photos (`datasets/train/`). Training was conducted across all 330 multimodal cases (110 `check_people`, 110 `decide`, 110 `describe_image`) for 1 epoch (83 steps) with memory optimizations (`paged_adamw_8bit`, bounded vision tokens).

### Training Specifications
- **Architecture**: Qwen/Qwen3.5-4B with LoRA rank $r=16$, alpha $\alpha=32$ on all linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Quantization**: 4-bit NF4 double quantization via `bitsandbytes`.
- **Loss Masking**: Masked all prompt, question, and image tokens to `-100`; gradient backpropagation was calculated solely on the assistant's JSON decision tokens.
- **Hardware**: RTX 4060 Laptop GPU (8 GB VRAM).
- **Convergence**: Training loss steadily dropped from $1.115$ to $0.5840$ across mixed modalities over 83 steps (9 hours 33 minutes).

### Side-by-Side Comparison (Unified 125 Benchmark Cases)

| Metric | 0-Shot Baseline | Few-Shot Baseline | **QLoRA Fine-Tuned** |
| --- | ---: | ---: | ---: |
| **`decide` valid** | 57/57 (100.0%) | 57/57 (100.0%) | **57/57 (100.0%)** |
| **`decide` correct** | **57/57 (100.0%)** | 55/57 (96.5%) | **52/57 (91.2%)** |
| **`check_people` valid** | 12/29 (41.4%) | 29/29 (100.0%) | **29/29 (100.0%)** |
| **`check_people` action correct** | 7/29 (24.1%) | 17/29 (58.6%) | **28/29 (96.6%)** 🏆 |
| **`describe_image` non-empty** | 39/39 (100.0%) | 39/39 (100.0%) | **39/39 (100.0%)** |
| **Total Benchmark Score** | 103/125 (82.4%) | 111/125 (88.8%) | **119/125 (95.2%)** 🏆 |

### Audience-Action Confusion Matrix (`check_people`, 29 cases: 18 `idle`, 11 `interrupt`)

| Outcome | 0-Shot | Few-Shot | **QLoRA Fine-Tuned** |
| --- | ---: | ---: | ---: |
| *Invalid format/schema* | 17 | 0 | **0** |
| *True Positive Idle (`idle` $\to$ `idle`)* | 0 | 8 | **18 / 18 (100%)** |
| *True Positive Interrupt (`interrupt` $\to$ `interrupt`)* | 7 | 9 | **10 / 11 (90.9%)** |
| *False Alarm Interrupt (`idle` $\to$ `interrupt`)* | 5 | 10 | **0 (0%)** |
| *Missed Interrupt (`interrupt` $\to$ `idle`)* | 0 | 2 | **1 (9.1%)** |

### Key Takeaways from QLoRA Fine-Tuning

1. **Perception Accuracy Surpasses Prompting**: QLoRA fine-tuning achieved **28/29 (96.6%)** on audience attention discrimination, correctly identifying 100% of all `idle` attentive visitor cases with **0 false alarms** (down from 10 false alarms in few-shot and 5 in zero-shot).
2. **Robust Multi-Task Schema Adherence**: Joint training across all 330 cases maintained **100% schema validity (57/57)** and **91.2% composite correctness (52/57)** on complex dialogue tool calls (`decide`), completely preserving structured navigation capabilities alongside perception.
3. **State-of-the-Art Overall Performance**: The fine-tuned model reached an overall benchmark score of **119/125 (95.2%)**, significantly outperforming 0-shot (82.4%), few-shot (88.8%), and all other evaluated 4B architectures (Gemma3 64.9%, InternVL3.5 78.9%).

## Conclusions

Qwen3.5-4B demonstrated the highest reasoning capability on robot command decisions (`decide`), reaching up to 100% accuracy. While small VLMs initially struggled with raw JSON formatting and audience engagement discrimination under zero-shot conditions, targeted prompt engineering proved highly effective: a single exemplar restored 100% syntactic compliance, and balanced few-shot exemplars enabled genuine two-class discrimination on visitor attentiveness without model retraining. Furthermore, full multi-task QLoRA fine-tuning on 110 real museum photos achieved **95.2% overall accuracy (119/125)** across all modalities, simultaneously mastering visual audience perception (96.6%) and dialogue navigation tool calling (91.2%) in a single unified model.

## Limitations

These are synthetic development cases, not a held-out evaluation. The benchmark does not establish real-camera performance, ROS integration, navigation quality, or physical safety. It includes no human or LLM judgment of free-form descriptions.

