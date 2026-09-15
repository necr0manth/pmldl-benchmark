"""Build a complete training dataset from collected photos for VLM fine-tuning."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_COLLECTION = ROOT / "finetune-people-collected"
TRAIN_DIR = ROOT / "datasets" / "train"
ASSETS_DIR = TRAIN_DIR / "assets"

# Verified annotations for unreviewed images
MANUAL_ANNOTATIONS = {
    "ft_f_009.png": {
        "category": "inattentive",
        "visible_people_count": 2,
        "clearly_facing_count": 0,
        "proposed_tool": "interrupt",
        "review_status": "approved",
        "visual_review_notes": "Two visitors viewed from behind examining exhibition display; both clearly turned away from the camera.",
    },
    "ft_f_015.png": {
        "category": "attentive",
        "visible_people_count": 2,
        "clearly_facing_count": 2,
        "proposed_tool": "idle",
        "review_status": "approved",
        "visual_review_notes": "Two adult visitors standing side by side facing camera smiling; clearly attentive.",
    },
    "ft_f_020.png": {
        "category": "attentive",
        "visible_people_count": None,
        "clearly_facing_count": 1,
        "proposed_tool": "idle",
        "review_status": "approved",
        "visual_review_notes": "One foreground visitor looking directly back at the camera; background visitors examining gallery displays.",
    },
}

EXHIBIT_ONTOLOGY = {
    "exhibit_vessel": ["синяя ваза", "ваза", "древний сосуд", "сосуд", "керамика"],
    "exhibit_mechanism": ["механизм", "бронзовый механизм", "шестерни", "роботизированная рука", "аппарат"],
    "exhibit_sculpture": ["красная скульптура", "скульптура", "абстрактная фигура", "статуя"],
    "exhibit_painting": ["картина", "полотно", "галерея плакатов", "пейзаж"],
    "exhibit_dinosaur": ["окаменелость", "скелет", "минералы", "ископаемые"],
}
TOUR_LIST = ["tour_demo", "tour_robotics", "tour_art", "tour_quick"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw_manifest(src_manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for line in src_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            fn = row.get("relativeimagepath")
            if fn in MANUAL_ANNOTATIONS:
                row.update(MANUAL_ANNOTATIONS[fn])
            rows.append(row)
    return rows


def build_dataset() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    raw_manifest_path = SRC_COLLECTION / "manifest.jsonl"
    if not raw_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found at {raw_manifest_path}")

    rows = load_raw_manifest(raw_manifest_path)
    print(f"Loaded {len(rows)} raw image metadata rows.")

    # 1. Copy images into assets/
    copied_assets = []
    for row in rows:
        img_fn = row["relativeimagepath"]
        src_img = SRC_COLLECTION / img_fn
        dst_img = ASSETS_DIR / img_fn
        if not src_img.is_file():
            raise FileNotFoundError(f"Source image missing: {src_img}")
        shutil.copy2(src_img, dst_img)
        file_hash = sha256_file(dst_img)
        copied_assets.append({
            "filename": img_fn,
            "path": f"assets/{img_fn}",
            "sha256": file_hash,
            "size_bytes": dst_img.stat().st_size,
        })
    print(f"Copied {len(copied_assets)} image assets to {ASSETS_DIR}")

    # Load prompts for SFT conversion
    people_prompt = (ROOT / "prompts" / "people-system.txt").read_text(encoding="utf-8").strip()
    desc_prompt = (ROOT / "prompts" / "description-system.txt").read_text(encoding="utf-8").strip()
    decision_prompt_tmpl = (ROOT / "prompts" / "decision-v2.txt").read_text(encoding="utf-8").strip()

    cases: list[dict[str, Any]] = []
    sft_records: list[dict[str, Any]] = []
    sharegpt_records: list[dict[str, Any]] = []

    # 2. Build check_people cases (110 cases)
    for idx, row in enumerate(rows, 1):
        img_fn = row["relativeimagepath"]
        stem = Path(img_fn).stem
        group_id = f"train-{stem}"
        tool = row.get("proposed_tool")
        if tool not in ("idle", "interrupt"):
            raise ValueError(f"Invalid tool for {img_fn}: {tool}")

        case_id = f"train-people-{stem}"
        expected = {"tool": tool, "args": {}, "abstain": False}
        case = {
            "case_id": case_id,
            "group_id": group_id,
            "image_group": stem,
            "category": row.get("category") or "general",
            "split": "train",
            "method": "check_people",
            "image": {"status": "available", "path": f"assets/{img_fn}"},
            "expected": expected,
            "metadata": {
                "source": "synthetic",
                "category": row.get("category"),
                "visible_people_count": row.get("visible_people_count"),
                "clearly_facing_count": row.get("clearly_facing_count"),
                "notes": row.get("visual_review_notes"),
                "source_batch": row.get("source_batch"),
            },
        }
        cases.append(case)

        # SFT entry for check_people
        assistant_json = json.dumps(expected, ensure_ascii=False)
        sft_records.append({
            "id": case_id,
            "task": "check_people",
            "images": [f"assets/{img_fn}"],
            "messages": [
                {"role": "system", "content": people_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"assets/{img_fn}"},
                        {"type": "text", "text": "Определи статус внимания посетителей перед роботом."},
                    ],
                },
                {"role": "assistant", "content": assistant_json},
            ],
        })
        sharegpt_records.append({
            "id": case_id,
            "image": f"assets/{img_fn}",
            "conversations": [
                {"from": "system", "value": people_prompt},
                {"from": "user", "value": "<image>\nОпредели статус внимания посетителей перед роботом."},
                {"from": "assistant", "value": assistant_json},
            ],
        })

    # 3. Build describe_image cases (110 cases)
    for idx, row in enumerate(rows, 1):
        img_fn = row["relativeimagepath"]
        stem = Path(img_fn).stem
        group_id = f"train-{stem}"
        case_id = f"train-describe-{stem}"
        case = {
            "case_id": case_id,
            "group_id": group_id,
            "image_group": stem,
            "category": row.get("category") or "general",
            "split": "train",
            "method": "describe_image",
            "image": {"status": "available", "path": f"assets/{img_fn}"},
            "expected": {"kind": "nonempty_text"},
            "metadata": {
                "source": "synthetic",
                "category": row.get("category"),
                "notes": row.get("visual_review_notes"),
            },
        }
        cases.append(case)

        notes = row.get("visual_review_notes") or ""
        desc_text = f"В музейном зале: {notes}" if notes else "Музейная экспозиция с экспонатами и галерейным пространством."
        sft_records.append({
            "id": case_id,
            "task": "describe_image",
            "images": [f"assets/{img_fn}"],
            "messages": [
                {"role": "system", "content": desc_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"assets/{img_fn}"},
                        {"type": "text", "text": "Опиши сцену перед роботом-экскурсоводом."},
                    ],
                },
                {"role": "assistant", "content": desc_text},
            ],
        })
        sharegpt_records.append({
            "id": case_id,
            "image": f"assets/{img_fn}",
            "conversations": [
                {"from": "system", "value": desc_prompt},
                {"from": "user", "value": "<image>\nОпиши сцену перед роботом-экскурсоводом."},
                {"from": "assistant", "value": desc_text},
            ],
        })

    # 4. Build multi-task decide cases (110 cases)
    exhibit_keys = list(EXHIBIT_ONTOLOGY.keys())
    for idx, row in enumerate(rows, 1):
        img_fn = row["relativeimagepath"]
        stem = Path(img_fn).stem
        group_id = f"train-{stem}"
        case_id = f"train-decide-{stem}"

        modulo = idx % 4
        if modulo == 0:
            tour_id = TOUR_LIST[idx % len(TOUR_LIST)]
            transcript = f"Начни {tour_id}." if idx % 2 == 0 else f"Запусти экскурсию {tour_id}."
            extra_ctx = (
                f"TASK POLICY / CONFIRMED FACTS:\n"
                f"Allowed tour IDs: {', '.join(TOUR_LIST)}.\n"
                f"END TASK POLICY"
            )
            expected = {"tool": "start_tour", "args": {"tour_id": tour_id}, "abstain": False}
            scored_fields = ["tool", "abstain", "tour_id"]
            task = "start_tour"
        elif modulo in (1, 2):
            ex_id = exhibit_keys[idx % len(exhibit_keys)]
            aliases = EXHIBIT_ONTOLOGY[ex_id]
            chosen_alias = aliases[idx % len(aliases)]
            transcript = f"Отведи к {chosen_alias}." if idx % 2 == 0 else f"Покажи {chosen_alias}."
            extra_ctx = (
                f"TASK POLICY / CONFIRMED FACTS:\n"
                f"Allowed location IDs: {', '.join(exhibit_keys)}.\n"
                f"Alias {chosen_alias} maps to {ex_id}.\n"
                f"END TASK POLICY"
            )
            expected = {"tool": "goto_exhibit", "args": {"location_id": ex_id}, "abstain": False}
            scored_fields = ["tool", "abstain", "location_id"]
            task = "goto_exhibit"
        else:
            transcript = "Принеси мне чашку кофе из буфета."
            extra_ctx = (
                f"TASK POLICY / CONFIRMED FACTS:\n"
                f"Allowed tour IDs: {', '.join(TOUR_LIST)}.\n"
                f"Allowed location IDs: {', '.join(exhibit_keys)}.\n"
                f"END TASK POLICY"
            )
            expected = {"tool": "idle", "args": {}, "abstain": True}
            scored_fields = ["tool", "abstain"]
            task = "abstain"

        case = {
            "case_id": case_id,
            "group_id": group_id,
            "image_group": stem,
            "category": "basic",
            "split": "train",
            "method": "decide",
            "transcript": transcript,
            "image": {"status": "available", "path": f"assets/{img_fn}"},
            "mission_state": {"state": "NARRATING" if modulo != 0 else "IDLE"},
            "extra_context": extra_ctx,
            "expected": expected,
            "scored_fields": scored_fields,
            "metadata": {"source": "synthetic", "task": task},
        }
        cases.append(case)

        prompt_content = (
            decision_prompt_tmpl
            .replace("{transcript}", transcript)
            .replace("{image_status}", "available")
            .replace("{mission_state}", json.dumps(case["mission_state"], ensure_ascii=False))
            .replace("{extra_context}", extra_ctx)
        )
        assistant_json = json.dumps(expected, ensure_ascii=False)
        sft_records.append({
            "id": case_id,
            "task": "decide",
            "images": [f"assets/{img_fn}"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"assets/{img_fn}"},
                        {"type": "text", "text": prompt_content},
                    ],
                },
                {"role": "assistant", "content": assistant_json},
            ],
        })
        sharegpt_records.append({
            "id": case_id,
            "image": f"assets/{img_fn}",
            "conversations": [
                {"from": "user", "value": f"<image>\n{prompt_content}"},
                {"from": "assistant", "value": assistant_json},
            ],
        })

    # 5. Write cases.jsonl
    cases_file = TRAIN_DIR / "cases.jsonl"
    with open(cases_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} benchmark cases to {cases_file}")

    # 6. Write SFT files
    sft_file = TRAIN_DIR / "sft_qwen.jsonl"
    with open(sft_file, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sft_records)} SFT records to {sft_file}")

    sharegpt_file = TRAIN_DIR / "sft_sharegpt.json"
    sharegpt_file.write_text(json.dumps(sharegpt_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(sharegpt_records)} ShareGPT records to {sharegpt_file}")

    # 7. Compute statistics and write manifest.json
    from collections import Counter
    tool_counts = Counter(c["expected"]["tool"] for c in cases if c["method"] == "check_people")
    cat_counts = Counter(r.get("category") for r in rows)
    method_counts = Counter(c["method"] for c in cases)

    manifest = {
        "dataset_name": "robot-guide-train-v1",
        "dataset_version": "1.0.0",
        "split": "train",
        "case_count": len(cases),
        "image_count": len(copied_assets),
        "cases_sha256": sha256_file(cases_file),
        "sft_qwen_sha256": sha256_file(sft_file),
        "sft_sharegpt_sha256": sha256_file(sharegpt_file),
        "breakdown": {
            "methods": dict(method_counts),
            "check_people_tools": dict(tool_counts),
            "image_categories": dict(cat_counts),
        },
        "assets": copied_assets,
    }
    manifest_file = TRAIN_DIR / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_file}")


if __name__ == "__main__":
    build_dataset()
