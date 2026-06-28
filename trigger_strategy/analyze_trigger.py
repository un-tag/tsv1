#!/usr/bin/env python3
"""Analyzer for exact Kalshi trigger-strategy replay tapes."""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import suppress
import json
import multiprocessing as mp
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional locally, mandatory on the public runner.
    from numba import njit
except Exception:  # pragma: no cover - exercised only when numba is absent.
    njit = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from trigger_strategy.common import (  # noqa: E402
    MONEY_SCALE,
    PRICE_1E4,
    SUPPORTED_ASSETS,
    TriggerDataError,
    format_money,
    format_price_1e4,
    has_complete_analyzer_dataset,
    is_buy_bucket_candidate_rational,
    parse_cli_bound_to_1e4,
    parse_cli_price_to_1e4,
)
from trigger_strategy.replay import metric_row_to_dict, simulate_candidate_rows_for_buy_sell  # noqa: E402

DEFAULT_DATA_DIR = SCRIPT_DIR / "preprocessed"
TRIGGER_SCAN_PROGRESS_WINDOW_BLOCK = 1024
PROGRESS_LINE_CLEAR_WIDTH = 160


def parse_whole_dollars(text: str, flag: str) -> int:
    try:
        value = Decimal(str(text))
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{flag} must be a positive whole-dollar amount") from exc
    if value <= 0 or value != value.to_integral_value():
        raise argparse.ArgumentTypeError(f"{flag} must be a positive whole-dollar amount")
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact trigger-strategy analyzer")
    parser.add_argument("asset", choices=SUPPORTED_ASSETS, help="One asset to analyze")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Analyzer dataset root")
    parser.add_argument("--br", required=True, type=lambda x: parse_whole_dollars(x, "--br"), help="Whole-dollar bankroll / upper cap")
    parser.add_argument("-b", dest="buy", help="Fixed buy trigger in displayed cents")
    parser.add_argument("-s", dest="sell", help="Fixed sell trigger in displayed cents")
    parser.add_argument("-u", dest="use_amount", type=lambda x: parse_whole_dollars(x, "-u"), help="Fixed whole-dollar use amount")
    parser.add_argument("--u-step", type=lambda x: parse_whole_dollars(x, "--u-step"), default=1, help="Whole-dollar use sweep step")
    parser.add_argument("--b-min", help="Inclusive buy-trigger sweep min in displayed cents")
    parser.add_argument("--b-max", help="Inclusive buy-trigger sweep max in displayed cents")
    parser.add_argument("--s-min", help="Inclusive sell-trigger sweep min in displayed cents")
    parser.add_argument("--s-max", help="Inclusive sell-trigger sweep max in displayed cents")
    parser.add_argument("--last", type=int, metavar="DAYS", help="Analyze the last ET market-open calendar days")
    parser.add_argument("--top", type=int, default=10, help="Top rows to display in sweep mode (0 = all)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Analyzer worker count")
    parser.add_argument("--profit", action="store_true", help="Sort sweep rows by raw net dollars/window")
    parser.add_argument("--parts", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--part", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--out-jsonl", help=argparse.SUPPRESS)
    parser.add_argument("--out-meta-json", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.br <= 0:
        parser.error("--br must be positive")
    if args.use_amount is not None and args.use_amount > args.br:
        parser.error("-u must be less than or equal to --br")
    if args.use_amount is not None and args.u_step != 1:
        parser.error("--u-step is invalid with fixed -u")
    if args.last is not None and args.last <= 0:
        parser.error("--last must be positive")
    if args.top < 0:
        parser.error("--top must be non-negative")
    if args.jobs <= 0:
        parser.error("-j/--jobs must be positive")
    if args.parts <= 0:
        parser.error("--parts must be positive")
    if args.part < 0 or args.part >= args.parts:
        parser.error("--part must be greater than or equal to 0 and less than --parts")
    if args.buy is not None and (args.b_min is not None or args.b_max is not None):
        parser.error("--b-min/--b-max are invalid with fixed -b")
    if args.sell is not None and (args.s_min is not None or args.s_max is not None):
        parser.error("--s-min/--s-max are invalid with fixed -s")

    try:
        args.buy_price = parse_cli_price_to_1e4(args.buy) if args.buy is not None else None
        args.sell_price = parse_cli_price_to_1e4(args.sell) if args.sell is not None else None
        args.b_min_price = parse_cli_bound_to_1e4(args.b_min, "--b-min") if args.b_min is not None else None
        args.b_max_price = parse_cli_bound_to_1e4(args.b_max, "--b-max") if args.b_max is not None else None
        args.s_min_price = parse_cli_bound_to_1e4(args.s_min, "--s-min") if args.s_min is not None else None
        args.s_max_price = parse_cli_bound_to_1e4(args.s_max, "--s-max") if args.s_max is not None else None
    except ValueError as exc:
        parser.error(str(exc))

    if args.buy_price is not None:
        if args.buy_price <= 5_000:
            parser.error("-b must be greater than 50c")
        if not is_buy_bucket_candidate_rational(args.buy_price):
            parser.error("-b is not economically rational after taker fees")
    if args.buy_price is not None and args.sell_price is not None and args.sell_price >= args.buy_price:
        parser.error("-s must be less than -b")

    return args


def load_dataset(data_dir: str | Path, asset: str) -> dict[str, np.ndarray]:
    base = Path(data_dir) / asset
    if not has_complete_analyzer_dataset(base):
        raise TriggerDataError(f"missing complete analyzer dataset for {asset}: {base}")
    day_ord = np.load(base / "day_ord.npy", mmap_mode="r")
    day_window_start = np.load(base / "day_window_start.npy", mmap_mode="r")
    window_outcome = np.load(base / "window_outcome.npy", mmap_mode="r")
    window_day_ord = expand_window_day_ord(day_ord, day_window_start, len(window_outcome))
    return {
        "base": base,
        "window_event_start": np.load(base / "window_event_start.npy", mmap_mode="r"),
        "window_outcome": window_outcome,
        "event_side": np.load(base / "event_side.npy", mmap_mode="r"),
        "event_price_1e4": np.load(base / "event_price_1e4.npy", mmap_mode="r"),
        "event_count_centi": np.load(base / "event_count_centi.npy", mmap_mode="r"),
        "day_ord": day_ord,
        "day_window_start": day_window_start,
        "window_day_ord": window_day_ord,
    }


def expand_window_day_ord(day_ord: np.ndarray, day_window_start: np.ndarray, total_windows: int) -> np.ndarray:
    if len(day_ord) == total_windows:
        return np.asarray(day_ord, dtype=np.uint32)
    if len(day_ord) != len(day_window_start) - 1:
        raise TriggerDataError(
            f"day metadata mismatch: day_ord={len(day_ord)} day_window_start={len(day_window_start)} windows={total_windows}"
        )
    counts = np.diff(np.asarray(day_window_start, dtype=np.uint64)).astype(np.int64)
    return np.repeat(np.asarray(day_ord, dtype=np.uint32), counts)


def select_window_range(dataset: dict[str, np.ndarray], last_days: int | None) -> tuple[int, int, list[int]]:
    total_windows = len(dataset["window_outcome"])
    if total_windows <= 0:
        raise TriggerDataError("dataset has zero windows")
    day_window_start = dataset["day_window_start"]
    day_ord = dataset["day_ord"]
    if len(day_window_start) < 2:
        return 0, total_windows, sorted(set(int(x) for x in dataset["window_day_ord"]))
    if last_days is None:
        if len(day_ord) == len(day_window_start) - 1:
            return 0, total_windows, [int(x) for x in day_ord]
        return 0, total_windows, sorted(set(int(x) for x in dataset["window_day_ord"]))
    start_day_idx = max(0, len(day_window_start) - 1 - int(last_days))
    start = int(day_window_start[start_day_idx])
    end = int(day_window_start[-1])
    if len(day_ord) == len(day_window_start) - 1:
        selected_days = [int(day_ord[i]) for i in range(start_day_idx, len(day_window_start) - 1)]
    else:
        selected_days = sorted(set(int(x) for x in dataset["window_day_ord"][start:end]))
    return start, end, selected_days


def price_range_from_args(args: argparse.Namespace, fixed: int | None, min_price: int | None, max_price: int | None, *, is_buy: bool) -> list[int]:
    if fixed is not None:
        return [int(fixed)]
    legal = [int(p) for p in PRICE_1E4]
    if is_buy:
        default_min = 5_010
        default_max = max(legal)
    else:
        default_min = min(legal)
        default_max = max(legal)
    lo = int(min_price) if min_price is not None else default_min
    hi = int(max_price) if max_price is not None else default_max
    if hi < lo:
        return []
    return [p for p in legal if lo <= p <= hi]


def use_amounts_from_args(args: argparse.Namespace) -> list[int]:
    if args.use_amount is not None:
        return [int(args.use_amount)]
    amounts = list(range(1, int(args.br) + 1, int(args.u_step)))
    if amounts[-1] != int(args.br):
        amounts.append(int(args.br))
    return amounts


def build_candidates(args: argparse.Namespace) -> tuple[list[int], dict[int, list[int]], list[int], int, int]:
    buy_prices = price_range_from_args(args, args.buy_price, args.b_min_price, args.b_max_price, is_buy=True)
    sell_prices = price_range_from_args(args, args.sell_price, args.s_min_price, args.s_max_price, is_buy=False)
    use_amounts = use_amounts_from_args(args)
    raw_pairs = len(buy_prices) * len(sell_prices)
    filtered_by_buy: dict[int, list[int]] = {}
    rejected = 0
    for buy in buy_prices:
        valid_sells: list[int] = []
        buy_valid = buy > 5_000 and is_buy_bucket_candidate_rational(buy)
        for sell in sell_prices:
            if buy_valid and sell < buy:
                valid_sells.append(sell)
            else:
                rejected += 1
        if valid_sells:
            filtered_by_buy[buy] = valid_sells
    filtered_pairs = sum(len(v) for v in filtered_by_buy.values())
    if filtered_pairs <= 0 or not use_amounts:
        raise TriggerDataError("candidate set is empty after applying bounds and invariants")
    return sorted(filtered_by_buy), filtered_by_buy, use_amounts, raw_pairs, rejected


def partition_candidates(
    buy_prices: list[int],
    sell_by_buy: dict[int, list[int]],
    parts: int,
    part: int,
) -> tuple[list[int], dict[int, list[int]]]:
    if int(parts) == 1:
        return buy_prices, sell_by_buy
    chunks = build_worker_chunks(buy_prices, sell_by_buy, int(parts))
    if int(part) >= len(chunks):
        return [], {}
    return chunks[int(part)]


def _counter_add(progress_counter: np.ndarray | None, progress_slot: int, value: int) -> None:
    if progress_counter is None or progress_slot < 0:
        return
    progress_counter[int(progress_slot)] += int(value)


def print_progress_line(done: int, total: int, started_at: float, *, current: str = "") -> None:
    now = time.monotonic()
    rate = int(done) / max(now - started_at, 0.001)
    eta = (int(total) - int(done)) / rate if rate > 0 else 0
    suffix = f"  {current}" if current else ""
    pct_done = 100.0 * int(done) / int(total) if int(total) > 0 else 100.0
    print(f"\r  Work {done:,}/{total:,} ({pct_done:5.1f}%)  {rate:.1f}/s  ETA {eta:.0f}s{suffix}    ", end="", flush=True)


def clear_progress_line() -> None:
    print("\r" + (" " * PROGRESS_LINE_CLEAR_WIDTH) + "\r", end="", flush=True)


def _ignore_sigint_in_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _attach_existing_shared_memory(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _shutdown_executor_now(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    terminate = getattr(executor, "terminate_workers", None)
    if terminate is not None:
        terminate()
    else:
        executor.shutdown(wait=False, cancel_futures=True)


def progress_work_total(
    selected_start: int,
    selected_end: int,
    buy_prices: list[int],
    sell_by_buy: dict[int, list[int]],
    use_amounts: list[int],
) -> int:
    windows = max(0, int(selected_end) - int(selected_start))
    trigger_scan_work = windows * len(buy_prices)
    replay_work = windows * len(use_amounts) * sum(len(sell_by_buy[buy]) for buy in buy_prices)
    return trigger_scan_work + replay_work


def build_worker_chunks(
    buy_prices: list[int],
    sell_by_buy: dict[int, list[int]],
    requested_jobs: int,
) -> list[tuple[list[int], dict[int, list[int]]]]:
    pair_total = sum(len(sell_by_buy[buy]) for buy in buy_prices)
    if pair_total <= 0:
        return []
    jobs = min(max(1, int(requested_jobs)), pair_total)
    chunk_sells: list[dict[int, list[int]]] = [{} for _ in range(jobs)]
    chunk_buy_order: list[list[int]] = [[] for _ in range(jobs)]
    chunk_work = [0 for _ in range(jobs)]
    original_order = {buy: idx for idx, buy in enumerate(buy_prices)}
    ordered_buys = sorted(buy_prices, key=lambda buy: (-len(sell_by_buy[buy]), original_order[buy]))
    for buy in ordered_buys:
        sells = sell_by_buy[buy]
        if len(buy_prices) >= jobs:
            slot = min(range(jobs), key=lambda idx: chunk_work[idx])
            chunk_sells[slot][buy] = list(sells)
            chunk_buy_order[slot].append(buy)
            chunk_work[slot] += len(sells)
            continue
        for sell in sells:
            slot = min(range(jobs), key=lambda idx: chunk_work[idx])
            if buy not in chunk_sells[slot]:
                chunk_sells[slot][buy] = []
                chunk_buy_order[slot].append(buy)
            chunk_sells[slot][buy].append(sell)
            chunk_work[slot] += 1
    chunks: list[tuple[list[int], dict[int, list[int]]]] = []
    for buys, sells in zip(chunk_buy_order, chunk_sells):
        if not buys:
            continue
        ordered_chunk_buys = sorted(buys, key=lambda buy: original_order[buy])
        ordered_chunk_sells = {buy: sells[buy] for buy in ordered_chunk_buys}
        chunks.append((ordered_chunk_buys, ordered_chunk_sells))
    return chunks


def progress_work_total_for_chunks(
    selected_start: int,
    selected_end: int,
    chunks: list[tuple[list[int], dict[int, list[int]]]],
    use_amounts: list[int],
) -> int:
    return sum(progress_work_total(selected_start, selected_end, buys, sells, use_amounts) for buys, sells in chunks)


def _worker_evaluate(
    data_dir: str,
    asset: str,
    selected_start: int,
    selected_end: int,
    buy_prices: list[int],
    sell_by_buy: dict[int, list[int]],
    use_amounts: list[int],
    emit_all: bool,
    progress_name: str | None = None,
    progress_slots: int = 0,
    progress_slot: int = -1,
) -> list[tuple[float, ...]]:
    dataset = load_dataset(data_dir, asset)
    rows: list[tuple[float, ...]] = []
    shm: shared_memory.SharedMemory | None = None
    progress_counter: np.ndarray | None = None
    if progress_name is not None and progress_slots > 0:
        shm = _attach_existing_shared_memory(progress_name)
        progress_counter = np.ndarray((int(progress_slots),), dtype=np.int64, buffer=shm.buf)
    try:
        for buy in buy_prices:
            first_trigger = compute_first_trigger_starts(
                dataset["window_event_start"],
                dataset["event_price_1e4"],
                selected_start,
                selected_end,
                buy,
                progress_counter=progress_counter,
                progress_slot=progress_slot,
            )
            for sell in sell_by_buy[buy]:
                rows.extend(
                    simulate_candidate_rows_for_buy_sell(
                        dataset["window_event_start"],
                        dataset["window_outcome"],
                        dataset["event_side"],
                        dataset["event_price_1e4"],
                        dataset["event_count_centi"],
                        selected_start,
                        selected_end,
                        dataset["window_day_ord"],
                        first_trigger,
                        buy,
                        sell,
                        use_amounts,
                        emit_all=emit_all,
                        progress_counter=progress_counter,
                        progress_slot=progress_slot,
                    )
                )
    finally:
        if shm is not None:
            shm.close()
    return rows


if njit is not None:

    @njit(cache=True, nogil=True)
    def _compute_first_trigger_starts_numba(
        window_event_start: np.ndarray,
        event_price_1e4: np.ndarray,
        selected_start: int,
        selected_end: int,
        buy_price_1e4: int,
        progress_counter: np.ndarray,
        progress_slot: int,
    ) -> np.ndarray:
        starts = np.full(int(selected_end) - int(selected_start), -1, dtype=np.int64)
        since_progress = 0
        use_progress = progress_slot >= 0 and len(progress_counter) > progress_slot
        for offset in range(starts.size):
            window_idx = int(selected_start) + offset
            start = int(window_event_start[window_idx])
            end = int(window_event_start[window_idx + 1])
            first = -1
            for event_idx in range(start, end):
                if int(event_price_1e4[event_idx]) >= int(buy_price_1e4):
                    first = event_idx
                    break
            starts[offset] = first
            if use_progress:
                since_progress += 1
                if since_progress >= TRIGGER_SCAN_PROGRESS_WINDOW_BLOCK:
                    progress_counter[progress_slot] += since_progress
                    since_progress = 0
        if use_progress and since_progress > 0:
            progress_counter[progress_slot] += since_progress
        return starts


def compute_first_trigger_starts(
    window_event_start: np.ndarray,
    event_price_1e4: np.ndarray,
    selected_start: int,
    selected_end: int,
    buy_price_1e4: int,
    progress_counter: np.ndarray | None = None,
    progress_slot: int = -1,
) -> np.ndarray:
    if njit is not None:
        counter = progress_counter if progress_counter is not None else np.empty(0, dtype=np.int64)
        return _compute_first_trigger_starts_numba(
            window_event_start,
            event_price_1e4,
            int(selected_start),
            int(selected_end),
            int(buy_price_1e4),
            counter,
            int(progress_slot),
        )

    starts = np.full(int(selected_end) - int(selected_start), -1, dtype=np.int64)
    since_progress = 0
    for offset, window_idx in enumerate(range(int(selected_start), int(selected_end))):
        start = int(window_event_start[window_idx])
        end = int(window_event_start[window_idx + 1])
        if end <= start:
            since_progress += 1
            if since_progress >= TRIGGER_SCAN_PROGRESS_WINDOW_BLOCK:
                _counter_add(progress_counter, progress_slot, since_progress)
                since_progress = 0
            continue
        hits = np.flatnonzero(event_price_1e4[start:end] >= int(buy_price_1e4))
        if hits.size:
            starts[offset] = start + int(hits[0])
        since_progress += 1
        if since_progress >= TRIGGER_SCAN_PROGRESS_WINDOW_BLOCK:
            _counter_add(progress_counter, progress_slot, since_progress)
            since_progress = 0
    if since_progress:
        _counter_add(progress_counter, progress_slot, since_progress)
    return starts


def evaluate_candidates(
    args: argparse.Namespace,
    _dataset: dict[str, np.ndarray],
    selected_start: int,
    selected_end: int,
    buy_prices: list[int],
    sell_by_buy: dict[int, list[int]],
    use_amounts: list[int],
    *,
    fixed: bool,
) -> list[dict[str, Any]]:
    candidate_total = sum(len(sell_by_buy[buy]) for buy in buy_prices) * len(use_amounts)
    if candidate_total <= 0:
        return []
    chunks = build_worker_chunks(buy_prices, sell_by_buy, int(args.jobs))
    jobs = len(chunks)
    total_work = progress_work_total_for_chunks(selected_start, selected_end, chunks, use_amounts)
    print(f"  Evaluating {candidate_total:,} candidates with {jobs} worker{'s' if jobs != 1 else ''}...")
    ctx = mp.get_context("spawn")
    compact_rows: list[tuple[float, ...]] = []
    t0 = time.monotonic()
    shm = shared_memory.SharedMemory(create=True, size=jobs * np.dtype(np.int64).itemsize)
    progress = np.ndarray((jobs,), dtype=np.int64, buffer=shm.buf)
    progress.fill(0)
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    futures: list[concurrent.futures.Future[list[tuple[float, ...]]]] = []
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=ctx,
            initializer=_ignore_sigint_in_worker,
        )
        futures = [
            executor.submit(
                _worker_evaluate,
                args.data_dir,
                args.asset,
                selected_start,
                selected_end,
                chunk_buy_prices,
                chunk_sell_by_buy,
                use_amounts,
                fixed,
                shm.name,
                jobs,
                slot,
            )
            for slot, (chunk_buy_prices, chunk_sell_by_buy) in enumerate(chunks)
        ]
        while True:
            done_futures = sum(1 for future in futures if future.done())
            done = int(progress.sum())
            print_progress_line(done, total_work, t0, current=f"{done_futures}/{len(futures)} workers complete")
            for future in futures:
                if future.done():
                    exc = future.exception()
                    if exc is not None:
                        raise exc
            if done_futures == len(futures):
                break
            time.sleep(1.0)
        for future in futures:
            compact_rows.extend(future.result())
        executor.shutdown(wait=True)
        executor = None
        print(f"\r  Work {total_work:,}/{total_work:,} complete{' ' * 40}")
    except BaseException:
        for future in futures:
            future.cancel()
        if executor is not None:
            with suppress(Exception):
                _shutdown_executor_now(executor)
            executor = None
        clear_progress_line()
        raise
    finally:
        if executor is not None:
            with suppress(Exception):
                executor.shutdown(wait=False, cancel_futures=True)
        shm.close()
        with suppress(FileNotFoundError):
            shm.unlink()
    return [metric_row_to_dict(row) for row in compact_rows]


def dollars(value_1e4: int | float) -> float:
    return float(value_1e4) / MONEY_SCALE


def pct(num: int, den: int) -> str:
    if den <= 0:
        return "0.0%"
    return f"{100.0 * num / den:.1f}%"


def metric_sort_key(row: dict[str, Any], profit: bool) -> tuple[Any, ...]:
    primary = row["net_per_window_1e4"] if profit else row["survival_score"]
    return (
        primary,
        row["net_per_window_1e4"],
        row["total_pnl_1e4"],
        -row["survival_budget_1e4"],
        -row["buy_price_1e4"],
    )


def format_optional_money(value: int | None) -> str:
    if value is None:
        return "n/a"
    return format_money(int(value))


def rows_for_output(rows: list[dict[str, Any]], *, fixed: bool) -> list[dict[str, Any]]:
    if fixed:
        return list(rows)
    return [r for r in rows if r["net_per_window_1e4"] > 0]


def render_rows(rows: list[dict[str, Any]], *, fixed: bool, top: int, profit: bool) -> str:
    if not fixed:
        rows = rows_for_output(rows, fixed=fixed)
        rows.sort(key=lambda r: metric_sort_key(r, profit), reverse=True)
        if top:
            rows = rows[:top]
    else:
        rows.sort(key=lambda r: metric_sort_key(r, profit), reverse=True)

    if not rows:
        return "  No positive net/window rows after filtering.\n"

    header = (
        "rank  buy    sell   use  surv_score  net/window  total_pnl  surv_budget  "
        "trade  trig   pos_all  pos_trd  worst_win  max_dd  cycles  stops  flips  "
        "avg_cash  max_cash  worst1d  worst3d  worst7d  lose_streak"
    )
    lines = [header]
    for idx, row in enumerate(rows, 1):
        windows = int(row["total_windows"])
        traded = int(row["traded_windows"])
        lines.append(
            f"{idx:>4}  "
            f"{format_price_1e4(row['buy_price_1e4']):>5}  "
            f"{format_price_1e4(row['sell_price_1e4']):>5}  "
            f"${row['use_amount_dollars']:<3}  "
            f"{row['survival_score']:.8f}  "
            f"{format_money(round(row['net_per_window_1e4'])):>10}  "
            f"{format_money(row['total_pnl_1e4']):>10}  "
            f"{format_money(row['survival_budget_1e4']):>11}  "
            f"{pct(traded, windows):>6}  "
            f"{pct(int(row['trigger_windows']), windows):>6}  "
            f"{pct(int(row['positive_windows']), windows):>7}  "
            f"{pct(int(row['positive_traded_windows']), traded):>7}  "
            f"{format_money(row['worst_window_1e4']):>9}  "
            f"{format_money(row['max_drawdown_1e4']):>7}  "
            f"{int(row['cycles']):>6}  "
            f"{int(row['stop_count']):>5}  "
            f"{int(row['flip_count']):>5}  "
            f"{format_money(round(row['avg_cash_used_1e4'])):>8}  "
            f"{format_money(row['max_cash_used_1e4']):>8}  "
            f"{format_optional_money(row['worst_1d_1e4']):>8}  "
            f"{format_optional_money(row['worst_3d_1e4']):>8}  "
            f"{format_optional_money(row['worst_7d_1e4']):>8}  "
            f"{int(row['max_losing_streak']):>11}"
        )
    return "\n".join(lines) + "\n"


def print_rows(rows: list[dict[str, Any]], *, fixed: bool, top: int, profit: bool) -> None:
    print(render_rows(rows, fixed=fixed, top=top, profit=profit), end="")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = load_dataset(args.data_dir, args.asset)
        selected_start, selected_end, selected_days = select_window_range(dataset, args.last)
        buy_prices, sell_by_buy, use_amounts, raw_pairs, rejected = build_candidates(args)
        fixed = args.buy_price is not None and args.sell_price is not None and args.use_amount is not None
        total_buy_prices = len(buy_prices)
        total_pairs = sum(len(v) for v in sell_by_buy.values())
        buy_prices, sell_by_buy = partition_candidates(buy_prices, sell_by_buy, args.parts, args.part)
        shard_pairs = sum(len(v) for v in sell_by_buy.values())

        if rejected > 0:
            print(
                f"  Warning: discarded {rejected:,} bounded trigger pairs by candidate invariants "
                f"(buy > 50c, sell < buy, rational buy bucket)."
            )
        day_start = datetime.fromordinal(selected_days[0]).date().isoformat()
        day_end = datetime.fromordinal(selected_days[-1]).date().isoformat()
        print(
            f"  Asset: {args.asset.upper()}  ET days: {day_start}..{day_end}  "
            f"windows: {selected_end - selected_start:,}  raw trigger pairs: {raw_pairs:,}  "
            f"use amounts: {len(use_amounts)}"
        )
        if args.parts > 1:
            print(
                f"  Shard: {args.part + 1}/{args.parts}  buy triggers: {len(buy_prices):,}/{total_buy_prices:,}  "
                f"trigger pairs: {shard_pairs:,}/{total_pairs:,}"
            )
        rows = evaluate_candidates(args, dataset, selected_start, selected_end, buy_prices, sell_by_buy, use_amounts, fixed=fixed)
        jsonl_rows = rows_for_output(rows, fixed=fixed)
        if args.out_jsonl:
            write_jsonl(args.out_jsonl, jsonl_rows)
        if args.out_meta_json:
            write_json(
                args.out_meta_json,
                {
                    "asset": args.asset,
                    "b_max_price_1e4": args.b_max_price,
                    "b_min_price_1e4": args.b_min_price,
                    "bankroll_dollars": int(args.br),
                    "buy_price_1e4": args.buy_price,
                    "data_dir": str(args.data_dir),
                    "day_start": day_start,
                    "day_end": day_end,
                    "fixed": fixed,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "jobs": int(args.jobs),
                    "jsonl_rows": len(jsonl_rows),
                    "last_days": args.last,
                    "part": int(args.part),
                    "parts": int(args.parts),
                    "profit": bool(args.profit),
                    "raw_trigger_pairs": int(raw_pairs),
                    "rejected_trigger_pairs": int(rejected),
                    "s_max_price_1e4": args.s_max_price,
                    "s_min_price_1e4": args.s_min_price,
                    "selected_window_start": int(selected_start),
                    "selected_window_end": int(selected_end),
                    "sell_price_1e4": args.sell_price,
                    "shard_buy_triggers": len(buy_prices),
                    "shard_trigger_pairs": int(shard_pairs),
                    "top": int(args.top),
                    "total_buy_triggers": int(total_buy_prices),
                    "total_trigger_pairs": int(total_pairs),
                    "u_step_dollars": int(args.u_step),
                    "use_amount_dollars": args.use_amount,
                    "use_amounts": len(use_amounts),
                    "windows": int(selected_end - selected_start),
                },
            )
        print_rows(rows, fixed=fixed, top=args.top, profit=args.profit)
        return 0
    except KeyboardInterrupt:
        clear_progress_line()
        print("Interrupted by Ctrl+C.")
        return 130
    except TriggerDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
