#!/usr/bin/env python3
"""Download one asset from Hugging Face and run one analyzer shard."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from trigger_strategy import analyze_trigger
from trigger_strategy.common import REQUIRED_ANALYZER_FILES
from trigger_strategy.replay import NUMBA_AVAILABLE


HF_BASE_URL = "https://huggingface.co/datasets/RooseveltHonaker/tsv1/resolve/main"


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
        cleaned.append(value)
    return cleaned


def has_jobs_arg(argv: list[str]) -> bool:
    for idx, value in enumerate(argv):
        if value == "-j" or value == "--jobs":
            return idx + 1 < len(argv)
        if value.startswith("--jobs="):
            return True
    return False


def download_asset(asset: str, data_dir: Path) -> None:
    asset_dir = data_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ANALYZER_FILES:
        dest = asset_dir / name
        url = f"{HF_BASE_URL}/preprocessed/{asset}/{name}"
        print(f"Downloading {asset}/{name}...", flush=True)
        urllib.request.urlretrieve(url, dest)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not NUMBA_AVAILABLE:
        print("ERROR: numba is mandatory for tsv1 runner shards", file=sys.stderr)
        return 2
    analyzer_args = json.loads(args.args_json)
    if not isinstance(analyzer_args, list) or not all(isinstance(x, str) for x in analyzer_args):
        print("ERROR: --args-json must be a JSON list of strings", file=sys.stderr)
        return 2
    parsed = analyze_trigger.parse_args(analyzer_args)
    clean_args = strip_option(analyzer_args, "--data-dir")
    if not has_jobs_arg(clean_args):
        clean_args.extend(["-j", str(args.jobs)])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    download_asset(parsed.asset, data_dir)
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
