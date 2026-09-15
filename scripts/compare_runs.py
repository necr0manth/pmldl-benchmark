"""Compare benchmark runs side-by-side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    cases_path = run_dir / "cases.jsonl"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    cases: list[dict[str, Any]] = []
    if cases_path.is_file():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))

    return {
        "dir": run_dir,
        "name": run_dir.name,
        "manifest": manifest,
        "summary": summary,
        "cases": cases,
    }


def analyze_people_breakdown(cases: list[dict[str, Any]]) -> dict[str, int]:
    people_cases = [c for c in cases if c.get("method") == "check_people"]
    counts = {
        "total": len(people_cases),
        "valid": 0,
        "invalid": 0,
        "abstain": 0,
        "idle_to_idle": 0,
        "idle_to_interrupt": 0,
        "interrupt_to_interrupt": 0,
        "interrupt_to_idle": 0,
    }
    for c in people_cases:
        parsed = c.get("parsed")
        scores = c.get("scores", {})
        if not scores.get("raw_valid") or parsed is None:
            counts["invalid"] += 1
            continue
        counts["valid"] += 1
        if parsed.get("abstain"):
            counts["abstain"] += 1

        pred_tool = parsed.get("tool")
        # In methods cases, oracle / expected is compared:
        action_correct = scores.get("action_correct")
        if action_correct:
            if pred_tool == "idle":
                counts["idle_to_idle"] += 1
            elif pred_tool == "interrupt":
                counts["interrupt_to_interrupt"] += 1
        else:
            if pred_tool == "idle":
                counts["interrupt_to_idle"] += 1
            elif pred_tool == "interrupt":
                counts["idle_to_interrupt"] += 1
    return counts


def format_table(runs: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("| Metric | " + " | ".join(r["name"] for r in runs) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in runs) + " |")

    # Check if methods benchmark or v2
    has_methods = any("methods" in r["summary"] for r in runs)

    if has_methods:
        # decide metrics
        def _get_decide(r, key, default_key=None):
            d = r["summary"].get("methods", {}).get("decide", {})
            ov = d.get("overall", d)
            return ov.get(key, d.get(default_key or key))

        lines.append(
            "| **`decide` valid** | "
            + " | ".join(
                _fmt_ratio(
                    _get_decide(r, "raw_valid"),
                    _get_decide(r, "n", "n_cases"),
                )
                for r in runs
            )
            + " |"
        )
        lines.append(
            "| **`decide` correct** | "
            + " | ".join(
                _fmt_ratio(
                    _get_decide(r, "composite"),
                    _get_decide(r, "n", "n_cases"),
                )
                for r in runs
            )
            + " |"
        )

        # check_people metrics
        lines.append(
            "| **`check_people` valid** | "
            + " | ".join(
                _fmt_ratio(
                    r["summary"].get("methods", {}).get("check_people", {}).get("n_raw_valid"),
                    r["summary"].get("methods", {}).get("check_people", {}).get("n_cases"),
                )
                for r in runs
            )
            + " |"
        )
        lines.append(
            "| **`check_people` action correct** | "
            + " | ".join(
                _fmt_ratio(
                    r["summary"].get("methods", {}).get("check_people", {}).get("n_action_correct"),
                    r["summary"].get("methods", {}).get("check_people", {}).get("n_cases"),
                )
                for r in runs
            )
            + " |"
        )

        # describe_image metrics
        lines.append(
            "| **`describe_image` non-empty** | "
            + " | ".join(
                _fmt_ratio(
                    r["summary"].get("methods", {}).get("describe_image", {}).get("n_nonempty"),
                    r["summary"].get("methods", {}).get("describe_image", {}).get("n_cases"),
                )
                for r in runs
            )
            + " |"
        )
    else:
        # V2 benchmark metrics
        lines.append(
            "| **v2 valid** | "
            + " | ".join(_fmt_ratio(r["summary"].get("raw_valid"), r["summary"].get("n")) for r in runs)
            + " |"
        )
        lines.append(
            "| **v2 composite correct** | "
            + " | ".join(_fmt_ratio(r["summary"].get("composite"), r["summary"].get("n")) for r in runs)
            + " |"
        )

    # People confusion details if available
    people_breakdowns = [analyze_people_breakdown(r["cases"]) for r in runs]
    if any(b["total"] > 0 for b in people_breakdowns):
        lines.append("| *check_people invalid* | " + " | ".join(str(b["invalid"]) for b in people_breakdowns) + " |")
        lines.append(
            "| *idle $\\to$ idle* | " + " | ".join(str(b["idle_to_idle"]) for b in people_breakdowns) + " |"
        )
        lines.append(
            "| *interrupt $\\to$ interrupt* | "
            + " | ".join(str(b["interrupt_to_interrupt"]) for b in people_breakdowns)
            + " |"
        )
        lines.append(
            "| *idle $\\to$ interrupt* | " + " | ".join(str(b["idle_to_interrupt"]) for b in people_breakdowns) + " |"
        )
        lines.append(
            "| *interrupt $\\to$ idle* | " + " | ".join(str(b["interrupt_to_idle"]) for b in people_breakdowns) + " |"
        )

    return "\n".join(lines)


def _fmt_ratio(num: int | None, denom: int | None) -> str:
    if num is None or denom is None or denom == 0:
        return "N/A"
    pct = (num / denom) * 100
    return f"{num}/{denom} ({pct:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VLM benchmark runs")
    parser.add_argument("paths", nargs="+", help="Run directories or parent directory containing runs")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for p_str in args.paths:
        p = Path(p_str)
        if (p / "summary.json").is_file():
            run_dirs.append(p)
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if (child / "summary.json").is_file():
                    run_dirs.append(child)

    if not run_dirs:
        print("No valid run directories found (each must contain summary.json).")
        return

    runs = [load_run(d) for d in run_dirs]
    print("\n### Benchmark Comparison\n")
    print(format_table(runs))
    print()


if __name__ == "__main__":
    main()
