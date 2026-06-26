#!/usr/bin/env python3
"""Run one analyzer shard against a prepared local asset dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trigger_strategy import analyze_trigger
from trigger_strategy.common import has_complete_analyzer_dataset
from trigger_strategy.replay import NUMBA_AVAILABLE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one tsv1 GitHub Actions shard")
    parser.add_argument("--args-json", required=True, help="JSON list of analyzer CLI args")
    parser.add_argument("--part", required=True, type=int)
    parser.add_argument("--parts", required=True, type=int)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--data-dir", default="preprocessed")
    parser.add_argument("--out-dir", default="results")
    return parser


def strip_option(argv: list[str], option: str, *, has_value: bool = True) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for value in argv:
        if skip_next:
            skip_next = False
            continue
        if value == option:
            skip_next = has_value
            continue
        if has_value and value.startswith(option + "="):
            continue
        if option.startswith("-") and not option.startswith("--") and has_value and value.startswith(option) and value != option:
            continue
        cleaned.append(value)
    return cleaned


def has_option(argv: list[str], option: str) -> bool:
    for value in argv:
        if value == option:
            return True
        if option.startswith("--") and value.startswith(option + "="):
            return True
        if option.startswith("-") and not option.startswith("--") and value.startswith(option) and value != option:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not NUMBA_AVAILABLE:
        print("ERROR: numba is mandatory for tsv1 runner shards", file=sys.stderr)
        return 2
    analyzer_args = json.loads(args.args_json)
    if not isinstance(analyzer_args, list) or not all(isinstance(x, str) for x in analyzer_args):
        print("ERROR: --args-json must be a JSON list of strings", file=sys.stderr)
        return 2
    if has_option(analyzer_args, "--data-dir"):
        print("ERROR: --data-dir is not supported on runner shards; data is downloaded from Hugging Face", file=sys.stderr)
        return 2
    parsed = analyze_trigger.parse_args(analyzer_args)
    clean_args = strip_option(strip_option(analyzer_args, "-j"), "--jobs")
    clean_args.extend(["-j", str(args.jobs)])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    if not has_complete_analyzer_dataset(data_dir / parsed.asset):
        print(f"ERROR: missing prepared dataset for {parsed.asset} under {data_dir}", file=sys.stderr)
        return 1
    shard_args = clean_args + [
        "--data-dir",
        str(data_dir),
        "--parts",
        str(args.parts),
        "--part",
        str(args.part),
        "--out-jsonl",
        str(out_dir / f"part-{args.part}.jsonl"),
        "--out-meta-json",
        str(out_dir / f"part-{args.part}.meta.json"),
    ]
    return analyze_trigger.main(shard_args)


if __name__ == "__main__":
    raise SystemExit(main())
