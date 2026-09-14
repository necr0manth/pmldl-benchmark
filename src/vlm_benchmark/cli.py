from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_datasets
from .reviews import export_reviews, import_reviews
from .runner import run_benchmark
from .v2 import load_v2_cases
from .v2runner import run_v2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="vlm-benchmark")
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("datasets", nargs="+")
    run = sub.add_parser("run")
    run.add_argument("datasets", nargs="+")
    run.add_argument("--config", action="append", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--case-id", action="append")
    review_export = sub.add_parser("review-export")
    review_export.add_argument("cases")
    review_export.add_argument("--output", required=True)
    review_import = sub.add_parser("review-import")
    review_import.add_argument("run_dir")
    review_import.add_argument("reviews")
    v2_validate = sub.add_parser("v2-validate")
    v2_validate.add_argument("dataset")
    v2_run = sub.add_parser("v2-run")
    v2_run.add_argument("dataset")
    v2_run.add_argument("--config", required=True)
    v2_run.add_argument("--output", required=True)
    v2_run.add_argument("--normalization", choices=("strict", "fence-only"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate":
        cases = load_datasets(args.datasets)
        print(json.dumps({"status": "valid", "cases": len(cases)}, ensure_ascii=False))
        return 0
    if args.command == "run":
        root = Path(args.output)
        outputs = []
        used_names: set[str] = set()
        for config_name in args.config:
            config_path = Path(config_name)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            name = config.get("name") or config_path.stem
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
            if safe_name in used_names:
                raise ValueError(f"duplicate config run name: {safe_name}")
            used_names.add(safe_name)
            outputs.append(str(run_benchmark(args.datasets, config_path, root / safe_name, args.case_id)))
        print(json.dumps({"status": "complete", "runs": outputs}, ensure_ascii=False))
        return 0
    if args.command == "review-export":
        count = export_reviews(args.cases, args.output)
        print(json.dumps({"status": "exported", "reviews": count}, ensure_ascii=False))
        return 0
    if args.command == "review-import":
        count = import_reviews(args.run_dir, args.reviews)
        print(json.dumps({"status": "imported", "reviews": count}, ensure_ascii=False))
        return 0
    if args.command == "v2-validate":
        cases = load_v2_cases(args.dataset)
        print(json.dumps({"status": "valid", "benchmark_version": "v2", "cases": len(cases)}, ensure_ascii=False))
        return 0
    if args.command == "v2-run":
        output = run_v2(args.dataset, args.config, args.output, args.normalization)
        print(json.dumps({"status": "complete", "run": str(output)}, ensure_ascii=False))
        return 0
    raise AssertionError("unreachable")
