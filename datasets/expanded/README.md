# Expanded synthetic development set

This merged dataset contains 106 cases in 20 scene groups backed by 20 distinct synthetic images.
It is an exact merge of the original 4-image/26-case pilot, part A (8/40), and part B (8/40).
Original case IDs, group IDs, and `dev` splits are preserved. Image paths are rebased to the source
assets; images are not copied or duplicated.

## Run

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python .\datasets\expanded\generate_merge.py
python -m vlm_benchmark validate .\datasets\expanded\cases.jsonl
python -m vlm_benchmark run .\datasets\expanded\cases.jsonl --config .\configs\mock-smoke.json --output .\runs\expanded-mock
python -m vlm_benchmark review-export .\runs\expanded-mock\mock-smoke\cases.jsonl --output .\runs\expanded-mock\reviews.todo.jsonl
```

The deterministic mock verifies plumbing only. It does not inspect the images, and its score is not
a VLM-quality result. Scene-description content remains pending until a human review is imported.

## Coverage

| Shard | Images / cases | Visible real people | Distance and framing | Lighting | Main distractors | Tools represented |
| --- | ---: | --- | --- | --- | --- | --- |
| Pilot | 4 / 26 | 0–3, frontal/mixed/back | mostly near/medium | controlled gallery | sculptures and display objects | start, goto, audience, scene, ask/idle controls |
| Part A | 8 / 40 | 1–5, mixed orientation/posture | close, grouped, varied depth | bright, warm, dark | statues, seated visitor, raised hand, machinery | start, goto, audience, scene, clarify alternative |
| Part B | 8 / 40 | 0–3 | overlap, edge crop, long distance | even, backlit, dim/noisy | people poster, statue, phone, screens | start, goto, audience, scene, ask/idle under uncertainty |
| Combined | 20 / 106 | 0–5 | close through distant | broad synthetic variation | human-like and behavioral distractors | all six closed schemas appear through decisions/controls |

The 16 new images were independently inspected against their final labels. Printed people and
human-shaped statues are not visitors. `xb05` exposes a visible count but deliberately does not
force an orientation label at long distance. Phone use, gestures, posture, and gaze direction never
establish attention, readiness, identity, or mental state.

## Provenance and limits

- All records are synthetic `dev` data; this is not a retrospectively frozen holdout.
- `manifest.json` records source JSONL hashes, the merged JSONL hash, and hashes for all 20 referenced
  images. Per-shard manifests retain generation prompts and visual-review notes.
- Recommendation policy is case-visible context, not a claimed robot-wide policy. The pilot includes
  calibration variants with different explicit policies; part A/B clear audience cases use
  `continue` when at least one person visibly faces the robot and `wait` otherwise.
- Golden decisions and reviewer rubrics stay in `oracle` and are not inserted into the model prompt.
- Real robot-camera data and manual scene-description reviews are still required before transfer or
  model-quality claims.

