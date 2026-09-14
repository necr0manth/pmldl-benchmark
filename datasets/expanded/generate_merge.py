from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SOURCES = (
    ROOT / "datasets" / "pilot" / "cases.jsonl",
    OUTPUT_DIR / "part_a" / "cases.jsonl",
    OUTPUT_DIR / "part_b" / "cases.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative_posix(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def main() -> None:
    cases: list[dict] = []
    source_entries: list[dict] = []
    seen_case_ids: set[str] = set()
    group_splits: dict[str, str] = {}

    for source_path in SOURCES:
        if not source_path.is_file():
            raise FileNotFoundError(f"source shard is missing: {source_path}")
        source_cases = []
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id {case_id!r} in {source_path}:{line_number}")
            seen_case_ids.add(case_id)
            group_id, split = case["group_id"], case["split"]
            previous_split = group_splits.setdefault(group_id, split)
            if previous_split != split:
                raise ValueError(f"group_id {group_id!r} leaks across splits")
            if case["image"]["status"] == "available":
                absolute_image = (source_path.parent / case["image"]["path"]).resolve()
                if not absolute_image.is_file():
                    raise FileNotFoundError(f"missing image for {case_id}: {absolute_image}")
                case["image"]["path"] = relative_posix(absolute_image, OUTPUT_DIR)
            source_cases.append(case)
            cases.append(case)
        source_entries.append(
            {
                "path": relative_posix(source_path, OUTPUT_DIR),
                "case_count": len(source_cases),
                "sha256": sha256(source_path),
            }
        )

    output_path = OUTPUT_DIR / "cases.jsonl"
    output_text = "".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases
    )
    output_path.write_text(output_text, encoding="utf-8")

    referenced_images: dict[str, Path] = {}
    for case in cases:
        image = case["image"]
        if image["status"] == "available":
            referenced_images[image["path"]] = (OUTPUT_DIR / image["path"]).resolve()
    assets = [
        {"path": path, "sha256": sha256(absolute_path)}
        for path, absolute_path in sorted(referenced_images.items())
    ]
    manifest = {
        "dataset_version": "expanded-dev-v1",
        "created_at": "2026-09-13",
        "status": "synthetic_development_only_not_frozen_holdout",
        "split_policy": "all source-image families retain their original group_id and dev split",
        "lineage": "Exact merge of pilot-v1, expanded-part-a, and expanded-part-b; image paths are rebased and assets are not copied.",
        "case_count": len(cases),
        "group_count": len(group_splits),
        "image_count": len(assets),
        "sources": source_entries,
        "assets": assets,
        "cases_sha256": sha256(output_path),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Wrote {len(cases)} cases, {len(group_splits)} groups, "
        f"and {len(assets)} referenced images to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
