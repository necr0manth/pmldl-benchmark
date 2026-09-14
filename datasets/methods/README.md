# Image-only method set

This set contains one dispatchable set of 125 cases: the frozen 57 basic `decide` cases plus deduplicated cases for `check_people(image)` and `describe_image(image)`.
Images are reused from the frozen v2 set through relative paths; no image copies are made.
The 48 legacy audience/policy rows are re-labeled once under the fixed people policy and
deduplicated by `(method, image)`, preserving source IDs in `metadata.origin_case_ids`.

The real chat branch accepts a visitor utterance. `describe_image` deliberately fixes the
image-only contract: it sends only the pinned system prompt and the picture, without a
changing utterance/history and without keyword routing.
