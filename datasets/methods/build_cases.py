"""Build the image-only method set from the frozen v2 cases without copying images."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets" / "v2" / "cases.jsonl"
OUT = Path(__file__).resolve().parent / "cases.jsonl"

rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
people = {}
for row in rows:
    if row["category"] in {"audience", "policy_edge"}:
        key = row["image"]["path"]
        # Prefer the ordinary audience annotation deterministically. Policy-edge
        # rows intentionally vary the old background policy and are not labels
        # for the new fixed image-only contract.
        previous = people.get(key)
        if previous is None or (row["category"], row["case_id"]) < (previous["category"], previous["case_id"]):
            people[key] = row
scene = {row["image"]["path"]: row for row in rows if row["category"] == "scene_routing"}

out = []
for row in rows:
    if row["category"] == "basic":
        legacy = dict(row)
        legacy["method"] = "decide"
        legacy["image"] = {**row["image"], "path": str(Path("../v2") / row["image"]["path"]).replace("\\", "/")}
        out.append(legacy)
for method, selected in (("check_people", people), ("describe_image", scene)):
    for source_id, row in sorted(((row["case_id"], row) for row in selected.values())):
        actual = (ROOT / "datasets" / "v2" / row["image"]["path"]).resolve()
        image_path = Path(__file__).resolve().parent / Path(Path(row["image"]["path"]).name)
        relative = Path(__file__).resolve().parent / Path(Path(row["image"]["path"]).name)
        # Keep the original directory layout in the relative reference.
        rel = Path("../v2") / Path(row["image"]["path"])
        expected = (
            {"tool": "idle" if row["expected"]["args"].get("facing_robot_count", 0) > 0 else "interrupt", "args": {}, "abstain": False}
            if method == "check_people" else {"kind": "nonempty_text"}
        )
        out.append({"case_id": f"method-{method}-{row['case_id']}", "group_id": f"method-{method}-{row['image_group']}", "split": row["split"], "method": method, "image": {"status": "available", "path": str(rel).replace("\\", "/")}, "expected": expected, "metadata": {"source": row["metadata"].get("source", "synthetic"), "label_policy_version": "image-only-v1", "origin_case_ids": [row["case_id"]], "origin_image": row["image"]["path"], "provenance": "datasets/v2/cases.jsonl; fixed image-only method contract"}})
OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in out) + "\n", encoding="utf-8")
print(f"wrote {len(out)} cases: people={len(people)}, scene={len(scene)}")
