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

## Conclusions

Qwen3.5-4B performed best on the basic `decide` command cases in this benchmark. Gemma3-4B-IT and InternVL3.5-4B achieved the highest audience-action score, but that score is largely explained by the majority/always-idle baseline of 18/29 (62.1%). None of the models demonstrated robust discrimination between attentive and non-attentive people here.

The description method cannot be ranked for semantic quality because only completion was measured. Invalid responses combine formatting and decision-contract failures, so low audience-action results cannot be attributed to visual perception alone.

## Limitations

These are synthetic development cases, not a held-out evaluation. The benchmark does not establish real-camera performance, ROS integration, navigation quality, or physical safety. It includes no human or LLM judgment of free-form descriptions.
