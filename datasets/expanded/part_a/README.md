# Expanded benchmark shard: part A

Synthetic development-only expansion: 8 independent raster scenes and 40 cases, five cases per scene. All cases use `decide(transcript, image, mission_state, extra_context)` and frozen interface v1. Labels record only independently observed people, count, and visible head/body orientation; raised hands, sitting, conversation, height, and distance are distractors/context, not new skills.

The images were generated one per scene with built-in imagegen and each final result was inspected with `view_image`. No oracle labels or review rubrics are inserted into model-visible inputs. Scene descriptions use reviewer-only rubrics and prohibit identity, attention, intent, and unobserved exhibit claims. All records are `dev`; this shard is not a frozen holdout or a quality score.

Lineage, prompts, and SHA-256 hashes are in `manifest.json`. Images are under `assets/`; paths in `cases.jsonl` are relative to this directory.
