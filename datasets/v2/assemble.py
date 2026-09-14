from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets" / "expanded" / "cases.jsonl"
OUT = Path(__file__).resolve().parent


def main() -> None:
    selected = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        choices = case["oracle"]["acceptable_decisions"]
        if len(choices) != 1 or choices[0]["tool"] == "ask_visitor":
            continue
        expected = {key: choices[0][key] for key in ("tool", "args", "abstain")}
        if expected["tool"] == "report_audience" and "Audience policy:" not in case["extra_context"]:
            continue
        task = case["metadata"].get("task", "")
        category = "audience" if expected["tool"] == "report_audience" else "scene_routing" if expected["tool"] == "describe_scene" else "basic"
        fields = ["tool", "abstain"]
        if expected["tool"] == "report_audience":
            fields += ["visible_count", "facing_robot_count", "recommendation"]
        elif expected["tool"] == "start_tour":
            fields += ["tour_id"]
        elif expected["tool"] == "goto_exhibit":
            fields += ["location_id"]
        elif expected["tool"] == "describe_scene":
            fields = ["tool", "abstain"]
        image_path = case["image"]["path"]
        source_path = (SOURCE.parent / image_path).resolve()
        selected.append({"case_id": "v2-" + case["case_id"], "group_id": "v2-" + case["group_id"],
            "image_group": source_path.stem, "category": category, "split": "dev", "transcript": case["transcript"],
            "image": {"status": "available", "path": os.path.relpath(source_path, OUT).replace("\\", "/")},
            "mission_state": case["mission_state"],
            "extra_context": "TASK POLICY / CONFIRMED FACTS:\n" + case["extra_context"].replace("Allowed exhibit IDs", "Allowed location IDs") + "\nEND TASK POLICY",
            "expected": expected, "scored_fields": fields,
            "metadata": {"source": "synthetic", "benchmark_version": "v2", "source_case_id": case["case_id"], "task": task}})
    author_scenes = []
    for author_dir in (OUT / "new_scenes_a", OUT / "new_scenes_b"):
        manifest_path = author_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for scene in manifest.get("scenes", []):
            image = (author_dir / scene["image_path"]).resolve()
            if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest().lower() != scene["sha256"].lower():
                raise ValueError(f"author scene asset/hash mismatch: {scene.get('scene_id')}")
            if not scene.get("count_known"):
                continue
            author_scenes.append((scene, image))
    pairs = {"va01": "pair-a", "va03": "pair-a", "va02": "pair-b", "va04": "pair-b", "va05": "pair-c", "va06": "pair-c",
             "va07": "pair-d", "va08": "pair-d", "va09": "pair-e", "va10": "pair-e", "vb01": "pair-f", "vb02": "pair-f",
             "vb03": "pair-g", "vb04": "pair-g"}
    for scene, image in author_scenes:
        sid = scene["scene_id"]
        common = {"split": "dev", "transcript": "Сколько людей перед роботом и сколько смотрит на робота?",
            "image": {"status": "available", "path": os.path.relpath(image, OUT).replace("\\", "/")},
            "mission_state": {"state": "NARRATING"},
            "extra_context": "TASK POLICY: wait when facing_robot_count > 0; continue when it equals 0.\nCONFIRMED FACTS: Allowed tour IDs: tour_demo. Allowed location IDs: main_hall (aliases: central hall, main gallery).\nEND TASK POLICY",
            "metadata": {"source": "synthetic", "benchmark_version": "v2", "author_scene_id": sid}}
        if scene.get("facing_known"):
            selected.append({**common, "case_id": f"v2-{sid}-audience", "group_id": f"v2-{sid}", "image_group": sid, "category": "audience",
                "expected": {"tool": "report_audience", "args": {"visible_count": scene["visible_count"], "facing_robot_count": scene["facing_robot_count"], "recommendation": "wait" if scene["facing_robot_count"] > 0 else "continue"}, "abstain": False},
                "scored_fields": ["tool", "abstain", "visible_count", "facing_robot_count", "recommendation"], "pair_id": pairs.get(sid)})
        for suffix, transcript, tool, args, category, fields in (("scene", "Опиши видимую сцену без выдуманных фактов.", "describe_scene", {"text": "nonempty visual description"}, "scene_routing", ["tool", "abstain"]),
            ("basic", "Начни экскурсию." if int(scene["visible_count"]) % 2 == 0 else "Отправь меня в центральный зал.", "start_tour" if int(scene["visible_count"]) % 2 == 0 else "goto_exhibit", {"tour_id": "tour_demo"} if int(scene["visible_count"]) % 2 == 0 else {"location_id": "main_hall"}, "basic", ["tool", "abstain", "tour_id"] if int(scene["visible_count"]) % 2 == 0 else ["tool", "abstain", "location_id"])):
            selected.append({**common, "case_id": f"v2-{sid}-{suffix}", "group_id": f"v2-{sid}", "image_group": sid, "category": category,
                "transcript": transcript, "expected": {"tool": tool, "args": args, "abstain": False}, "scored_fields": fields, "pair_id": None})
        if scene.get("facing_known"):
            edge_context = "TASK POLICY: wait when visible_count >= 3; continue when visible_count < 3. CONFIRMED FACTS: counts are evaluated from the image only. END TASK POLICY"
            selected.append({**common, "case_id": f"v2-{sid}-policy-edge", "group_id": f"v2-{sid}", "image_group": sid,
                "category": "policy_edge", "transcript": "Оцени аудиторию по заданной фоновой политике.", "extra_context": edge_context,
                "expected": {"tool": "report_audience", "args": {"visible_count": scene["visible_count"], "facing_robot_count": scene["facing_robot_count"], "recommendation": "wait" if scene["visible_count"] >= 3 else "continue"}, "abstain": False},
                "scored_fields": ["tool", "abstain", "visible_count", "facing_robot_count", "recommendation"], "pair_id": None})
    (OUT / "cases.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in selected), encoding="utf-8")
    assets = []
    for case in selected:
        path = (OUT / case["image"]["path"]).resolve()
        relative_path = os.path.relpath(path, OUT).replace("\\", "/")
        if not any(x["path"] == relative_path for x in assets):
            assets.append({"path": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"dataset_version": "v2.0.0-dev", "status": "synthetic_development_only", "split": "dev", "case_count": len(selected),
                "image_count": len(assets), "case_sha256": hashlib.sha256((OUT / "cases.jsonl").read_bytes()).hexdigest(), "assets": assets,
                "excluded": {"multi_valid": True, "ask_visitor_without_robot_catalog": True, "scene_free_text": "not_scored"},
                "pending_author_images": 20 - len(author_scenes)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(selected), "images": len(assets), "pending_author_images": 20 - len(author_scenes)}))


if __name__ == "__main__":
    main()
