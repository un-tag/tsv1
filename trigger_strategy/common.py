#!/usr/bin/env python3
"""Analyzer-only helpers for the public tsv1 runner."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np


YES = 0
NO = 1
SIDE_NAMES = ("YES", "NO")

ASSET_PREFIXES: dict[str, str] = {
    "btc": "KXBTC15M",
    "eth": "KXETH15M",
    "sol": "KXSOL15M",
    "xrp": "KXXRP15M",
    "doge": "KXDOGE15M",
    "bnb": "KXBNB15M",
    "hype": "KXHYPE15M",
}
SUPPORTED_ASSETS = tuple(ASSET_PREFIXES)

MONEY_SCALE = 10_000
QTY_SCALE = 100
MAX_PRICE_1E4 = 10_000

PRICE_BUCKETS = np.asarray(
    list(range(1, 100)) + list(range(100, 901, 10)) + list(range(901, 1000)),
    dtype=np.uint16,
)
PRICE_BUCKET_SET = frozenset(int(x) for x in PRICE_BUCKETS)
PRICE_1E4 = (PRICE_BUCKETS.astype(np.uint32) * 10).astype(np.uint16)
PRICE_1E4_SET = frozenset(int(x) for x in PRICE_1E4)

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


class TriggerDataError(RuntimeError):
    """Raised when trigger-strategy data is incomplete or inconsistent."""


def opposite_side(side: int) -> int:
    return NO if int(side) == YES else YES


def ceil_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("denominator must be positive")
    if num <= 0:
        return 0
    return (num + den - 1) // den


def floor_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("denominator must be positive")
    if num <= 0:
        return 0
    return num // den


def quantize_money_1e4(num: int, den: int, *, rounding: str) -> int:
    if rounding == "ceil":
        return ceil_div(num, den)
    if rounding == "floor":
        return floor_div(num, den)
    raise ValueError(f"unknown money rounding mode {rounding!r}")


def buy_notional_1e4(qty_centi: int, price_1e4: int) -> int:
    return quantize_money_1e4(int(qty_centi) * int(price_1e4), QTY_SCALE, rounding="ceil")


def sell_proceeds_1e4(qty_centi: int, sale_price_1e4: int) -> int:
    return quantize_money_1e4(int(qty_centi) * int(sale_price_1e4), QTY_SCALE, rounding="floor")


def payout_1e4(qty_centi: int) -> int:
    return int(qty_centi) * (MONEY_SCALE // QTY_SCALE)


def taker_fee_1e4(qty_centi: int, price_1e4: int) -> int:
    qty = int(qty_centi)
    price = int(price_1e4)
    if qty <= 0 or price <= 0 or price >= MAX_PRICE_1E4:
        return 0
    return ceil_div(7 * qty * price * (MAX_PRICE_1E4 - price), 100_000_000)


def buy_cost_1e4(qty_centi: int, price_1e4: int) -> int:
    return buy_notional_1e4(qty_centi, price_1e4) + taker_fee_1e4(qty_centi, price_1e4)


def sell_proceeds_after_fee_1e4(qty_centi: int, sale_price_1e4: int) -> int:
    return sell_proceeds_1e4(qty_centi, sale_price_1e4) - taker_fee_1e4(qty_centi, sale_price_1e4)


def is_buy_economically_rational(qty_centi: int, price_1e4: int) -> bool:
    qty = int(qty_centi)
    if qty <= 0:
        return False
    return buy_cost_1e4(qty, price_1e4) < payout_1e4(qty)


def is_buy_bucket_candidate_rational(price_1e4: int) -> bool:
    return is_buy_economically_rational(QTY_SCALE, price_1e4)


def max_affordable_buy_qty(cash_1e4: int, price_1e4: int, available_centi: int) -> int:
    cash = int(cash_1e4)
    available = int(available_centi)
    if cash <= 0 or available <= 0:
        return 0
    lo = 0
    hi = available
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if buy_cost_1e4(mid, price_1e4) <= cash:
            lo = mid
        else:
            hi = mid - 1
    if lo <= 0 or not is_buy_economically_rational(lo, price_1e4):
        return 0
    return lo


def sell_proceeds_1e4_after_fee(qty_centi: int, sale_price_1e4: int) -> int:
    return sell_proceeds_after_fee_1e4(qty_centi, sale_price_1e4)


def parse_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def parse_cli_price_to_1e4(text: str) -> int:
    raw = str(text).strip()
    dec = parse_decimal(raw)
    if dec is None:
        raise ValueError(f"{text!r} is not a number of cents")
    if dec < 1:
        raise ValueError(f"{text!r} looks like dollars; pass cents such as 80 or 98.4")
    if dec <= 0 or dec >= 100:
        raise ValueError("trigger price must be greater than 0c and less than 100c")
    bucket_dec = dec * Decimal("10")
    bucket_int = bucket_dec.to_integral_value()
    if bucket_dec != bucket_int:
        raise ValueError(f"{text!r} is not on the 0.1c trigger grid")
    bucket = int(bucket_int)
    if bucket not in PRICE_BUCKET_SET:
        raise ValueError(f"{text!r} is not on the 15m crypto executable lattice")
    return bucket * 10


def parse_cli_bound_to_1e4(text: str, flag_name: str) -> int:
    try:
        return parse_cli_price_to_1e4(text)
    except ValueError as exc:
        raise ValueError(f"{flag_name}: {exc}") from exc


def format_price_1e4(price: int) -> str:
    bucket = int(price) // 10
    if bucket % 10 == 0:
        return f"{bucket // 10}c"
    return f"{bucket / 10:.1f}c"


def format_money(value_1e4: int) -> str:
    sign = "-" if int(value_1e4) < 0 else ""
    abs_value = abs(int(value_1e4))
    return f"{sign}${abs_value // MONEY_SCALE}.{abs_value % MONEY_SCALE:04d}"


def has_complete_analyzer_dataset(base: str | Path) -> bool:
    path = Path(base)
    return all((path / name).exists() for name in REQUIRED_ANALYZER_FILES)


def sell_proceeds_1e4(qty_centi: int, sale_price_1e4: int) -> int:  # type: ignore[no-redef]
    return quantize_money_1e4(int(qty_centi) * int(sale_price_1e4), QTY_SCALE, rounding="floor")
