# Author manifest: `new_scenes_b`

Synthetic-dev observation shard for audience counting, observable facing direction, and scene routing. It contains ten independent RGB museum-camera frames (`vb01`–`vb10`) and no benchmark cases or model outputs.

## Coverage

- Empty gallery and framed-photo distractor (`vb01`).
- Live visitor versus clearly faceless mannequins (`vb02`).
- Counter occlusion, edge crop, hats, backlight, motion blur, distance, camera tilt, and mixed orientations (`vb03`–`vb09`).
- Phone-down, camera-facing, and exhibit-facing visitors with poster/sculpture context (`vb10`).

## Observed vs intended

Labels in `manifest.json` are based on visual inspection of the saved raster files, not on generation intent. `vb06`–`vb08` retain unknown facing where the frame does not support a reliable robot-facing claim. `vb10` has an ambiguous upper glass reflection; it is recorded as an uncertainty and is not counted as a live visitor. No gestures, emotions, demographics, attention, readiness, hidden people, or exhibit facts are labeled.

## Split and use

This shard is synthetic `dev` material only. It must not be treated as real-camera validation or a physical-safety result. Keep it grouped with related synthetic data when creating splits; do not use it as holdout evidence.

## Lineage

Each image was generated independently with one built-in imagegen call, copied into `assets/`, and inspected with `view_image` before labeling. The manifest stores relative paths, SHA-256 hashes, source=`synthetic`, and the generation prompt used for that image. No ROS package, model runtime, or source repository was modified.
