# Synthetic pilot dataset v1

This is a compact development/calibration pilot for the VLM decision benchmark. It is synthetic-only and is not a final holdout, real-robot validation, or statistically independent test: all cases sharing an image stay in the same group and split.

- 26 cases, 4 images, 4 image groups, one `dev` split.
- Inputs follow `decide(transcript, image, mission_state, extra_context: str)`.
- Outputs follow `.ai_docs/implementation-interface-v1.md`; `oracle.acceptable_decisions` contains semantic alternatives, not a string target.
- Image paths are relative to this directory.
- `extra_context` contains synthetic registry/facts only. It does not prove that an object is present in the image.
- Scene descriptions are manually rubric-scored for observations, unsupported claims, and omissions; there is no single exact gold sentence.

Use `manifest.json` for hashes and generation provenance, `rubrics.md` for scene review, and `metadata-view.json` for an independent-checker summary.
