#!/usr/bin/env python3
"""Download one analyzer asset from Hugging Face with bounded retries."""

from __future__ import annotations

import argparse
import email.utils
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
import urllib.request


HF_API_URL = "https://huggingface.co/api/datasets/RooseveltHonaker/tsv1"
HF_BASE_URL = "https://huggingface.co/datasets/RooseveltHonaker/tsv1/resolve/main"
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}
REQUIRED_ANALYZER_FILES = (
    "manifest.json",
    "window_event_start.npy",
    "window_outcome.npy",
    "event_side.npy",
    "event_price_1e4.npy",
    "event_count_centi.npy",
    "day_ord.npy",
    "day_window_start.npy",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download tsv1 Hugging Face analyzer data")
    parser.add_argument("--asset")
    parser.add_argument("--data-dir", default="preprocessed")
    parser.add_argument("--dataset-sha", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=8)
    return parser


def retry_after_seconds(exc: HTTPError) -> float | None:
    value = exc.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def sleep_before_retry(label: str, attempt: int, max_attempts: int, retry_after: float | None = None) -> None:
    delay = retry_after if retry_after is not None else min(120.0, (2 ** min(attempt, 7)) + random.uniform(0.0, 3.0))
    print(f"{label}; retry {attempt}/{max_attempts} after {delay:.1f}s", file=sys.stderr, flush=True)
    time.sleep(delay)


def open_with_retries(url: str, *, max_attempts: int):
    for attempt in range(1, max_attempts + 1):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP or attempt >= max_attempts:
                raise
            sleep_before_retry(f"HTTP {exc.code} for {url}", attempt, max_attempts, retry_after_seconds(exc))
        except (TimeoutError, URLError) as exc:
            if attempt >= max_attempts:
                raise
            sleep_before_retry(f"{type(exc).__name__} for {url}", attempt, max_attempts)


def fetch_dataset_sha(max_attempts: int) -> str:
    with open_with_retries(HF_API_URL, max_attempts=max_attempts) as resp:
        payload = json.load(resp)
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise RuntimeError("Hugging Face dataset API did not return a sha")
    return sha


def download_url(url: str, dest: Path, *, max_attempts: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open_with_retries(url, max_attempts=max_attempts) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def download_asset(asset: str, data_dir: Path, *, max_attempts: int) -> None:
    asset_dir = data_dir / asset
    for name in REQUIRED_ANALYZER_FILES:
        dest = asset_dir / name
        url = f"{HF_BASE_URL}/preprocessed/{asset}/{name}"
        print(f"Downloading {asset}/{name}...", flush=True)
        download_url(url, dest, max_attempts=max_attempts)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_attempts <= 0:
        print("ERROR: --max-attempts must be positive", file=sys.stderr)
        return 2
    try:
        if args.dataset_sha:
            print(fetch_dataset_sha(args.max_attempts))
            return 0
        if not args.asset:
            print("ERROR: --asset is required unless --dataset-sha is used", file=sys.stderr)
            return 2
        download_asset(args.asset, Path(args.data_dir), max_attempts=args.max_attempts)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
