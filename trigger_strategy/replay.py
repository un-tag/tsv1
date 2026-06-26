#!/usr/bin/env python3
"""Exact replay state machine for trigger-strategy candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # Optional but used in the normal local environment for full-grid speed.
    from numba import njit
except Exception:  # pragma: no cover - exercised only when numba is absent.
    njit = None

from trigger_strategy.common import (
    MAX_PRICE_1E4,
    MONEY_SCALE,
    NO,
    YES,
    buy_cost_1e4,
    max_affordable_buy_qty,
    opposite_side,
    payout_1e4,
    sell_proceeds_1e4,
    taker_fee_1e4,
)

MODE_IDLE = 0
MODE_BUYING = 1
MODE_SELLING = 2
NUMBA_AVAILABLE = njit is not None
REPLAY_PROGRESS_FLUSH_WINDOWS = 256


@dataclass
class WindowReplayResult:
    pnl_1e4: int
    traded: bool
    triggered: bool
    positive: bool
    fills: int
    cycles: int
    stop_count: int
    flip_count: int
    cash_used_1e4: int


@dataclass
class CandidateMetrics:
    buy_price_1e4: int
    sell_price_1e4: int
    use_amount_dollars: int
    total_windows: int
    total_pnl_1e4: int
    net_per_window_1e4: float
    survival_budget_1e4: int
    survival_score: float
    max_drawdown_1e4: int
    traded_windows: int
    trigger_windows: int
    positive_windows: int
    positive_traded_windows: int
    worst_window_1e4: int
    cycles: int
    stop_count: int
    flip_count: int
    avg_cash_used_1e4: float
    max_cash_used_1e4: int
    worst_1d_1e4: int | None
    worst_3d_1e4: int | None
    worst_7d_1e4: int | None
    max_losing_streak: int


def _buy_fill(cash_1e4: int, price_1e4: int, available_centi: int) -> tuple[int, int, int]:
    qty = max_affordable_buy_qty(cash_1e4, price_1e4, available_centi)
    if qty <= 0:
        return 0, cash_1e4, available_centi
    return qty, cash_1e4 - buy_cost_1e4(qty, price_1e4), available_centi - qty


def _sell_fill(cash_1e4: int, held_centi: int, sale_price_1e4: int, available_centi: int) -> tuple[int, int, int, int]:
    qty = min(int(held_centi), int(available_centi))
    if qty <= 0:
        return 0, cash_1e4, held_centi, available_centi
    sale_price = int(sale_price_1e4)
    proceeds = sell_proceeds_1e4(qty, sale_price)
    fee = taker_fee_1e4(qty, sale_price)
    return qty, cash_1e4 + proceeds - fee, held_centi - qty, available_centi - qty


def simulate_window(
    event_side: Sequence[int] | np.ndarray,
    event_price_1e4: Sequence[int] | np.ndarray,
    event_count_centi: Sequence[int] | np.ndarray,
    outcome_side: int,
    buy_price_1e4: int,
    sell_price_1e4: int,
    use_amount_dollars: int,
) -> WindowReplayResult:
    """Replay one independent market window for one candidate."""
    start_cash = int(use_amount_dollars) * MONEY_SCALE
    cash = start_cash
    min_cash = start_cash
    held_side = -1
    held_qty = 0
    mode = MODE_IDLE
    flip_target = -1
    triggered = False
    traded = False
    fills = 0
    cycles = 0
    stop_count = 0
    flip_count = 0

    for raw_side, raw_price, raw_count in zip(event_side, event_price_1e4, event_count_centi):
        side = int(raw_side)
        price = int(raw_price)
        available = int(raw_count)
        if available <= 0:
            continue

        if held_side < 0:
            if mode != MODE_BUYING and price >= buy_price_1e4:
                held_side = side
                mode = MODE_BUYING
                flip_target = -1
                triggered = True
            if mode == MODE_BUYING and side == held_side:
                qty, cash, available = _buy_fill(cash, price, available)
                if qty > 0:
                    if held_qty <= 0:
                        cycles += 1
                    held_qty += qty
                    traded = True
                    fills += 1
                    min_cash = min(min_cash, cash)
            continue

        if held_qty <= 0 and mode == MODE_BUYING:
            if side == held_side:
                qty, cash, available = _buy_fill(cash, price, available)
                if qty > 0:
                    cycles += 1
                    held_qty += qty
                    traded = True
                    fills += 1
                    min_cash = min(min_cash, cash)
            elif price >= buy_price_1e4:
                held_side = side
                flip_target = -1
                triggered = True
                qty, cash, available = _buy_fill(cash, price, available)
                if qty > 0:
                    cycles += 1
                    held_qty += qty
                    traded = True
                    fills += 1
                    min_cash = min(min_cash, cash)
            continue

        if side == held_side:
            if price >= buy_price_1e4 and mode == MODE_SELLING:
                mode = MODE_BUYING
                flip_target = -1
                triggered = True
            elif price <= sell_price_1e4 and mode != MODE_SELLING:
                mode = MODE_SELLING
                stop_count += 1
                triggered = True

            if mode == MODE_SELLING and price <= MAX_PRICE_1E4 - buy_price_1e4:
                pending = opposite_side(held_side)
                if flip_target != pending:
                    flip_target = pending
                    flip_count += 1
                triggered = True

            if mode == MODE_BUYING:
                qty, cash, available = _buy_fill(cash, price, available)
                if qty > 0:
                    if held_qty <= 0:
                        cycles += 1
                    held_qty += qty
                    traded = True
                    fills += 1
                    min_cash = min(min_cash, cash)
            elif mode == MODE_SELLING:
                qty, cash, held_qty, available = _sell_fill(cash, held_qty, price, available)
                if qty > 0:
                    traded = True
                    fills += 1
                    min_cash = min(min_cash, cash)
                if held_qty <= 0:
                    pending = flip_target
                    flip_target = -1
                    if pending >= 0:
                        held_side = pending
                        mode = MODE_BUYING
                    else:
                        held_side = -1
                        mode = MODE_IDLE
            continue

        if price >= buy_price_1e4:
            if flip_target != side:
                flip_target = side
                flip_count += 1
            triggered = True

    if held_qty > 0 and held_side == int(outcome_side):
        cash += payout_1e4(held_qty)

    pnl = cash - start_cash
    return WindowReplayResult(
        pnl_1e4=pnl,
        traded=traded,
        triggered=triggered,
        positive=pnl > 0,
        fills=fills,
        cycles=cycles,
        stop_count=stop_count,
        flip_count=flip_count,
        cash_used_1e4=max(0, start_cash - min_cash),
    )


def _zero_window_result(use_amount_dollars: int) -> WindowReplayResult:
    return WindowReplayResult(
        pnl_1e4=0,
        traded=False,
        triggered=False,
        positive=False,
        fills=0,
        cycles=0,
        stop_count=0,
        flip_count=0,
        cash_used_1e4=0,
    )


def simulate_window_many_uses(
    event_side: Sequence[int] | np.ndarray,
    event_price_1e4: Sequence[int] | np.ndarray,
    event_count_centi: Sequence[int] | np.ndarray,
    outcome_side: int,
    buy_price_1e4: int,
    sell_price_1e4: int,
    use_amounts_dollars: Sequence[int],
) -> list[WindowReplayResult]:
    """Replay one window while sharing event traversal across use amounts."""
    n = len(use_amounts_dollars)
    start_cash = [int(u) * MONEY_SCALE for u in use_amounts_dollars]
    cash = start_cash.copy()
    min_cash = start_cash.copy()
    held_side = [-1] * n
    held_qty = [0] * n
    mode = [MODE_IDLE] * n
    flip_target = [-1] * n
    triggered = [False] * n
    traded = [False] * n
    fills = [0] * n
    cycles = [0] * n
    stop_count = [0] * n
    flip_count = [0] * n

    for raw_side, raw_price, raw_count in zip(event_side, event_price_1e4, event_count_centi):
        side = int(raw_side)
        price = int(raw_price)
        raw_available = int(raw_count)
        if raw_available <= 0:
            continue

        for idx in range(n):
            available = raw_available
            if held_side[idx] < 0:
                if mode[idx] != MODE_BUYING and price >= buy_price_1e4:
                    held_side[idx] = side
                    mode[idx] = MODE_BUYING
                    flip_target[idx] = -1
                    triggered[idx] = True
                if mode[idx] == MODE_BUYING and side == held_side[idx]:
                    qty, new_cash, available = _buy_fill(cash[idx], price, available)
                    if qty > 0:
                        if held_qty[idx] <= 0:
                            cycles[idx] += 1
                        cash[idx] = new_cash
                        held_qty[idx] += qty
                        traded[idx] = True
                        fills[idx] += 1
                        min_cash[idx] = min(min_cash[idx], cash[idx])
                continue

            if held_qty[idx] <= 0 and mode[idx] == MODE_BUYING:
                if side == held_side[idx]:
                    qty, new_cash, available = _buy_fill(cash[idx], price, available)
                    if qty > 0:
                        cash[idx] = new_cash
                        cycles[idx] += 1
                        held_qty[idx] += qty
                        traded[idx] = True
                        fills[idx] += 1
                        min_cash[idx] = min(min_cash[idx], cash[idx])
                elif price >= buy_price_1e4:
                    held_side[idx] = side
                    flip_target[idx] = -1
                    triggered[idx] = True
                    qty, new_cash, available = _buy_fill(cash[idx], price, available)
                    if qty > 0:
                        cash[idx] = new_cash
                        cycles[idx] += 1
                        held_qty[idx] += qty
                        traded[idx] = True
                        fills[idx] += 1
                        min_cash[idx] = min(min_cash[idx], cash[idx])
                continue

            if side == held_side[idx]:
                if price >= buy_price_1e4 and mode[idx] == MODE_SELLING:
                    mode[idx] = MODE_BUYING
                    flip_target[idx] = -1
                    triggered[idx] = True
                elif price <= sell_price_1e4 and mode[idx] != MODE_SELLING:
                    mode[idx] = MODE_SELLING
                    stop_count[idx] += 1
                    triggered[idx] = True

                if mode[idx] == MODE_SELLING and price <= MAX_PRICE_1E4 - buy_price_1e4:
                    pending = opposite_side(held_side[idx])
                    if flip_target[idx] != pending:
                        flip_target[idx] = pending
                        flip_count[idx] += 1
                    triggered[idx] = True

                if mode[idx] == MODE_BUYING:
                    qty, new_cash, available = _buy_fill(cash[idx], price, available)
                    if qty > 0:
                        if held_qty[idx] <= 0:
                            cycles[idx] += 1
                        cash[idx] = new_cash
                        held_qty[idx] += qty
                        traded[idx] = True
                        fills[idx] += 1
                        min_cash[idx] = min(min_cash[idx], cash[idx])
                elif mode[idx] == MODE_SELLING:
                    qty, new_cash, new_held_qty, available = _sell_fill(cash[idx], held_qty[idx], price, available)
                    if qty > 0:
                        cash[idx] = new_cash
                        held_qty[idx] = new_held_qty
                        traded[idx] = True
                        fills[idx] += 1
                        min_cash[idx] = min(min_cash[idx], cash[idx])
                    if held_qty[idx] <= 0:
                        pending = flip_target[idx]
                        flip_target[idx] = -1
                        if pending >= 0:
                            held_side[idx] = pending
                            mode[idx] = MODE_BUYING
                        else:
                            held_side[idx] = -1
                            mode[idx] = MODE_IDLE
                continue

            if price >= buy_price_1e4:
                if flip_target[idx] != side:
                    flip_target[idx] = side
                    flip_count[idx] += 1
                triggered[idx] = True

    results: list[WindowReplayResult] = []
    for idx in range(n):
        if held_qty[idx] > 0 and held_side[idx] == int(outcome_side):
            cash[idx] += payout_1e4(held_qty[idx])
        pnl = cash[idx] - start_cash[idx]
        results.append(
            WindowReplayResult(
                pnl_1e4=pnl,
                traded=traded[idx],
                triggered=triggered[idx],
                positive=pnl > 0,
                fills=fills[idx],
                cycles=cycles[idx],
                stop_count=stop_count[idx],
                flip_count=flip_count[idx],
                cash_used_1e4=max(0, start_cash[idx] - min_cash[idx]),
            )
        )
    return results


def _rolling_worst_by_day(window_pnl: list[int], day_ord: np.ndarray, days: int) -> int | None:
    if not window_pnl:
        return None
    daily: dict[int, int] = {}
    for pnl, day in zip(window_pnl, day_ord):
        daily[int(day)] = daily.get(int(day), 0) + int(pnl)
    ordered = sorted(daily)
    if len(ordered) < days:
        return None
    values = [daily[d] for d in ordered]
    best = None
    for idx in range(0, len(values) - days + 1):
        total = sum(values[idx:idx + days])
        if best is None or total < best:
            best = total
    return best


if njit is not None:

    @njit(cache=True, nogil=True)
    def _ceil_div_nb(num: int, den: int) -> int:
        if num <= 0:
            return 0
        return (num + den - 1) // den


    @njit(cache=True, nogil=True)
    def _buy_cost_1e4_nb(qty_centi: int, price_1e4: int) -> int:
        notional = _ceil_div_nb(qty_centi * price_1e4, 100)
        fee = _ceil_div_nb(7 * qty_centi * price_1e4 * (10_000 - price_1e4), 100_000_000)
        return notional + fee


    @njit(cache=True, nogil=True)
    def _sell_return_1e4_nb(qty_centi: int, sale_price_1e4: int) -> int:
        proceeds = (qty_centi * sale_price_1e4) // 100
        fee = _ceil_div_nb(7 * qty_centi * sale_price_1e4 * (10_000 - sale_price_1e4), 100_000_000)
        return proceeds - fee


    @njit(cache=True, nogil=True)
    def _is_buy_rational_nb(qty_centi: int, price_1e4: int) -> bool:
        if qty_centi <= 0:
            return False
        return _buy_cost_1e4_nb(qty_centi, price_1e4) < qty_centi * 100


    @njit(cache=True, nogil=True)
    def _max_affordable_buy_qty_nb(cash_1e4: int, price_1e4: int, available_centi: int) -> int:
        if cash_1e4 <= 0 or available_centi <= 0:
            return 0
        lo = 0
        hi = available_centi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _buy_cost_1e4_nb(mid, price_1e4) <= cash_1e4:
                lo = mid
            else:
                hi = mid - 1
        if lo <= 0 or not _is_buy_rational_nb(lo, price_1e4):
            return 0
        return lo


    @njit(cache=True, nogil=True)
    def _simulate_windows_many_uses_numba(
        window_event_start: np.ndarray,
        window_outcome: np.ndarray,
        event_side: np.ndarray,
        event_price_1e4: np.ndarray,
        event_count_centi: np.ndarray,
        selected_window_start: int,
        selected_window_end: int,
        first_trigger_event_idx: np.ndarray,
        buy_price_1e4: int,
        sell_price_1e4: int,
        use_amounts_dollars: np.ndarray,
        progress_counter: np.ndarray,
        progress_slot: int,
        progress_units_per_window: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_windows = selected_window_end - selected_window_start
        n_uses = len(use_amounts_dollars)
        pnl = np.zeros((n_uses, n_windows), dtype=np.int64)
        traded = np.zeros((n_uses, n_windows), dtype=np.uint8)
        triggered = np.zeros((n_uses, n_windows), dtype=np.uint8)
        cycles = np.zeros((n_uses, n_windows), dtype=np.int64)
        stops = np.zeros((n_uses, n_windows), dtype=np.int64)
        flips = np.zeros((n_uses, n_windows), dtype=np.int64)
        cash_used = np.zeros((n_uses, n_windows), dtype=np.int64)
        use_progress = progress_slot >= 0 and progress_units_per_window > 0 and len(progress_counter) > progress_slot
        progress_since = 0

        for window_offset in range(n_windows):
            first = int(first_trigger_event_idx[window_offset])
            if first < 0:
                if use_progress:
                    progress_since += progress_units_per_window
                    if progress_since >= progress_units_per_window * REPLAY_PROGRESS_FLUSH_WINDOWS:
                        progress_counter[progress_slot] += progress_since
                        progress_since = 0
                continue
            window_idx = selected_window_start + window_offset
            end = int(window_event_start[window_idx + 1])
            outcome_side = int(window_outcome[window_idx])

            cash = np.empty(n_uses, dtype=np.int64)
            start_cash = np.empty(n_uses, dtype=np.int64)
            min_cash = np.empty(n_uses, dtype=np.int64)
            held_side = np.empty(n_uses, dtype=np.int64)
            held_qty = np.empty(n_uses, dtype=np.int64)
            mode = np.empty(n_uses, dtype=np.int64)
            flip_target = np.empty(n_uses, dtype=np.int64)

            for idx in range(n_uses):
                start_value = int(use_amounts_dollars[idx]) * 10_000
                start_cash[idx] = start_value
                cash[idx] = start_value
                min_cash[idx] = start_value
                held_side[idx] = -1
                held_qty[idx] = 0
                mode[idx] = MODE_IDLE
                flip_target[idx] = -1

            for event_idx in range(first, end):
                side = int(event_side[event_idx])
                price = int(event_price_1e4[event_idx])
                raw_available = int(event_count_centi[event_idx])
                if raw_available <= 0:
                    continue

                for idx in range(n_uses):
                    available = raw_available

                    if held_side[idx] < 0:
                        if mode[idx] != MODE_BUYING and price >= buy_price_1e4:
                            held_side[idx] = side
                            mode[idx] = MODE_BUYING
                            flip_target[idx] = -1
                            triggered[idx, window_offset] = 1
                        if mode[idx] == MODE_BUYING and side == held_side[idx]:
                            qty = _max_affordable_buy_qty_nb(cash[idx], price, available)
                            if qty > 0:
                                if held_qty[idx] <= 0:
                                    cycles[idx, window_offset] += 1
                                cash[idx] -= _buy_cost_1e4_nb(qty, price)
                                held_qty[idx] += qty
                                traded[idx, window_offset] = 1
                                if cash[idx] < min_cash[idx]:
                                    min_cash[idx] = cash[idx]
                        continue

                    if held_qty[idx] <= 0 and mode[idx] == MODE_BUYING:
                        if side == held_side[idx]:
                            qty = _max_affordable_buy_qty_nb(cash[idx], price, available)
                            if qty > 0:
                                cash[idx] -= _buy_cost_1e4_nb(qty, price)
                                cycles[idx, window_offset] += 1
                                held_qty[idx] += qty
                                traded[idx, window_offset] = 1
                                if cash[idx] < min_cash[idx]:
                                    min_cash[idx] = cash[idx]
                        elif price >= buy_price_1e4:
                            held_side[idx] = side
                            flip_target[idx] = -1
                            triggered[idx, window_offset] = 1
                            qty = _max_affordable_buy_qty_nb(cash[idx], price, available)
                            if qty > 0:
                                cash[idx] -= _buy_cost_1e4_nb(qty, price)
                                cycles[idx, window_offset] += 1
                                held_qty[idx] += qty
                                traded[idx, window_offset] = 1
                                if cash[idx] < min_cash[idx]:
                                    min_cash[idx] = cash[idx]
                        continue

                    if side == held_side[idx]:
                        if price >= buy_price_1e4 and mode[idx] == MODE_SELLING:
                            mode[idx] = MODE_BUYING
                            flip_target[idx] = -1
                            triggered[idx, window_offset] = 1
                        elif price <= sell_price_1e4 and mode[idx] != MODE_SELLING:
                            mode[idx] = MODE_SELLING
                            stops[idx, window_offset] += 1
                            triggered[idx, window_offset] = 1

                        if mode[idx] == MODE_SELLING and price <= 10_000 - buy_price_1e4:
                            pending = 1
                            if held_side[idx] == 1:
                                pending = 0
                            if flip_target[idx] != pending:
                                flip_target[idx] = pending
                                flips[idx, window_offset] += 1
                            triggered[idx, window_offset] = 1

                        if mode[idx] == MODE_BUYING:
                            qty = _max_affordable_buy_qty_nb(cash[idx], price, available)
                            if qty > 0:
                                if held_qty[idx] <= 0:
                                    cycles[idx, window_offset] += 1
                                cash[idx] -= _buy_cost_1e4_nb(qty, price)
                                held_qty[idx] += qty
                                traded[idx, window_offset] = 1
                                if cash[idx] < min_cash[idx]:
                                    min_cash[idx] = cash[idx]
                        elif mode[idx] == MODE_SELLING:
                            qty = held_qty[idx]
                            if available < qty:
                                qty = available
                            if qty > 0:
                                cash[idx] += _sell_return_1e4_nb(qty, price)
                                held_qty[idx] -= qty
                                available -= qty
                                traded[idx, window_offset] = 1
                                if cash[idx] < min_cash[idx]:
                                    min_cash[idx] = cash[idx]
                            if held_qty[idx] <= 0:
                                pending = flip_target[idx]
                                flip_target[idx] = -1
                                if pending >= 0:
                                    held_side[idx] = pending
                                    mode[idx] = MODE_BUYING
                                else:
                                    held_side[idx] = -1
                                    mode[idx] = MODE_IDLE
                        continue

                    if price >= buy_price_1e4:
                        if flip_target[idx] != side:
                            flip_target[idx] = side
                            flips[idx, window_offset] += 1
                        triggered[idx, window_offset] = 1

            for idx in range(n_uses):
                if held_qty[idx] > 0 and held_side[idx] == outcome_side:
                    cash[idx] += held_qty[idx] * 100
                pnl[idx, window_offset] = cash[idx] - start_cash[idx]
                cash_used[idx, window_offset] = max(0, start_cash[idx] - min_cash[idx])
            if use_progress:
                progress_since += progress_units_per_window
                if progress_since >= progress_units_per_window * REPLAY_PROGRESS_FLUSH_WINDOWS:
                    progress_counter[progress_slot] += progress_since
                    progress_since = 0

        if use_progress and progress_since > 0:
            progress_counter[progress_slot] += progress_since

        return pnl, traded, triggered, cycles, stops, flips, cash_used


def aggregate_candidate_metrics(
    buy_price_1e4: int,
    sell_price_1e4: int,
    use_amount_dollars: int,
    window_results: list[WindowReplayResult],
    selected_day_ord: np.ndarray,
) -> CandidateMetrics:
    total_windows = len(window_results)
    pnl_values = [int(r.pnl_1e4) for r in window_results]
    total_pnl = sum(pnl_values)
    peak = 0
    cumulative = 0
    max_drawdown = 0
    losing = 0
    max_losing = 0
    for pnl in pnl_values:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if pnl < 0:
            losing += 1
            max_losing = max(max_losing, losing)
        else:
            losing = 0
    use_floor = int(use_amount_dollars) * MONEY_SCALE
    denom = max(max_drawdown, use_floor)
    net_per_window = total_pnl / total_windows if total_windows else 0.0
    survival_score = net_per_window / denom if denom > 0 else 0.0
    traded = sum(1 for r in window_results if r.traded)
    return CandidateMetrics(
        buy_price_1e4=int(buy_price_1e4),
        sell_price_1e4=int(sell_price_1e4),
        use_amount_dollars=int(use_amount_dollars),
        total_windows=total_windows,
        total_pnl_1e4=total_pnl,
        net_per_window_1e4=net_per_window,
        survival_budget_1e4=max_drawdown,
        survival_score=survival_score,
        max_drawdown_1e4=max_drawdown,
        traded_windows=traded,
        trigger_windows=sum(1 for r in window_results if r.triggered),
        positive_windows=sum(1 for r in window_results if r.positive),
        positive_traded_windows=sum(1 for r in window_results if r.traded and r.positive),
        worst_window_1e4=min(pnl_values) if pnl_values else 0,
        cycles=sum(r.cycles for r in window_results),
        stop_count=sum(r.stop_count for r in window_results),
        flip_count=sum(r.flip_count for r in window_results),
        avg_cash_used_1e4=(sum(r.cash_used_1e4 for r in window_results) / total_windows if total_windows else 0.0),
        max_cash_used_1e4=max((r.cash_used_1e4 for r in window_results), default=0),
        worst_1d_1e4=_rolling_worst_by_day(pnl_values, selected_day_ord, 1),
        worst_3d_1e4=_rolling_worst_by_day(pnl_values, selected_day_ord, 3),
        worst_7d_1e4=_rolling_worst_by_day(pnl_values, selected_day_ord, 7),
        max_losing_streak=max_losing,
    )


def aggregate_candidate_metrics_from_arrays(
    buy_price_1e4: int,
    sell_price_1e4: int,
    use_amount_dollars: int,
    pnl_values: np.ndarray,
    traded_values: np.ndarray,
    triggered_values: np.ndarray,
    cycles_values: np.ndarray,
    stop_values: np.ndarray,
    flip_values: np.ndarray,
    cash_used_values: np.ndarray,
    selected_day_ord: np.ndarray,
) -> CandidateMetrics:
    pnl_i64 = np.asarray(pnl_values, dtype=np.int64)
    total_windows = int(pnl_i64.size)
    total_pnl = int(pnl_i64.sum()) if total_windows else 0
    if total_windows:
        cumulative = np.cumsum(pnl_i64)
        peaks = np.maximum.accumulate(np.maximum(cumulative, 0))
        drawdowns = peaks - cumulative
        max_drawdown = int(drawdowns.max())
        worst_window = int(pnl_i64.min())
    else:
        max_drawdown = 0
        worst_window = 0

    losing = 0
    max_losing = 0
    for pnl in pnl_i64:
        if int(pnl) < 0:
            losing += 1
            max_losing = max(max_losing, losing)
        else:
            losing = 0

    use_floor = int(use_amount_dollars) * MONEY_SCALE
    denom = max(max_drawdown, use_floor)
    net_per_window = total_pnl / total_windows if total_windows else 0.0
    survival_score = net_per_window / denom if denom > 0 else 0.0
    traded = int(np.asarray(traded_values, dtype=np.uint8).sum())
    positive_mask = pnl_i64 > 0
    traded_mask = np.asarray(traded_values, dtype=np.uint8) > 0
    pnl_list = [int(x) for x in pnl_i64]
    return CandidateMetrics(
        buy_price_1e4=int(buy_price_1e4),
        sell_price_1e4=int(sell_price_1e4),
        use_amount_dollars=int(use_amount_dollars),
        total_windows=total_windows,
        total_pnl_1e4=total_pnl,
        net_per_window_1e4=net_per_window,
        survival_budget_1e4=max_drawdown,
        survival_score=survival_score,
        max_drawdown_1e4=max_drawdown,
        traded_windows=traded,
        trigger_windows=int(np.asarray(triggered_values, dtype=np.uint8).sum()),
        positive_windows=int(positive_mask.sum()),
        positive_traded_windows=int((positive_mask & traded_mask).sum()),
        worst_window_1e4=worst_window,
        cycles=int(np.asarray(cycles_values, dtype=np.int64).sum()),
        stop_count=int(np.asarray(stop_values, dtype=np.int64).sum()),
        flip_count=int(np.asarray(flip_values, dtype=np.int64).sum()),
        avg_cash_used_1e4=(float(np.asarray(cash_used_values, dtype=np.int64).sum()) / total_windows if total_windows else 0.0),
        max_cash_used_1e4=int(np.asarray(cash_used_values, dtype=np.int64).max()) if total_windows else 0,
        worst_1d_1e4=_rolling_worst_by_day(pnl_list, selected_day_ord, 1),
        worst_3d_1e4=_rolling_worst_by_day(pnl_list, selected_day_ord, 3),
        worst_7d_1e4=_rolling_worst_by_day(pnl_list, selected_day_ord, 7),
        max_losing_streak=max_losing,
    )


def simulate_candidate(
    window_event_start: np.ndarray,
    window_outcome: np.ndarray,
    event_side: np.ndarray,
    event_price_1e4: np.ndarray,
    event_count_centi: np.ndarray,
    selected_window_start: int,
    selected_window_end: int,
    day_ord: np.ndarray,
    buy_price_1e4: int,
    sell_price_1e4: int,
    use_amount_dollars: int,
) -> CandidateMetrics:
    results: list[WindowReplayResult] = []
    for window_idx in range(int(selected_window_start), int(selected_window_end)):
        start = int(window_event_start[window_idx])
        end = int(window_event_start[window_idx + 1])
        results.append(
            simulate_window(
                event_side[start:end],
                event_price_1e4[start:end],
                event_count_centi[start:end],
                int(window_outcome[window_idx]),
                buy_price_1e4,
                sell_price_1e4,
                use_amount_dollars,
            )
        )
    return aggregate_candidate_metrics(
        buy_price_1e4,
        sell_price_1e4,
        use_amount_dollars,
        results,
        day_ord[selected_window_start:selected_window_end],
    )


def simulate_candidates_for_buy_sell(
    window_event_start: np.ndarray,
    window_outcome: np.ndarray,
    event_side: np.ndarray,
    event_price_1e4: np.ndarray,
    event_count_centi: np.ndarray,
    selected_window_start: int,
    selected_window_end: int,
    window_day_ord: np.ndarray,
    first_trigger_event_idx: np.ndarray,
    buy_price_1e4: int,
    sell_price_1e4: int,
    use_amounts_dollars: Sequence[int],
    progress_counter: np.ndarray | None = None,
    progress_slot: int = -1,
) -> list[CandidateMetrics]:
    """Replay one buy/sell pair, batching all use amounts over shared events."""
    if NUMBA_AVAILABLE:
        use_array = np.asarray(use_amounts_dollars, dtype=np.int64)
        counter = progress_counter if progress_counter is not None else np.empty(0, dtype=np.int64)
        pnl, traded, triggered, cycles, stops, flips, cash_used = _simulate_windows_many_uses_numba(
            window_event_start,
            window_outcome,
            event_side,
            event_price_1e4,
            event_count_centi,
            int(selected_window_start),
            int(selected_window_end),
            first_trigger_event_idx,
            int(buy_price_1e4),
            int(sell_price_1e4),
            use_array,
            counter,
            int(progress_slot),
            int(len(use_array)),
        )
        selected_days = window_day_ord[int(selected_window_start):int(selected_window_end)]
        return [
            aggregate_candidate_metrics_from_arrays(
                buy_price_1e4,
                sell_price_1e4,
                int(use_amount),
                pnl[idx],
                traded[idx],
                triggered[idx],
                cycles[idx],
                stops[idx],
                flips[idx],
                cash_used[idx],
                selected_days,
            )
            for idx, use_amount in enumerate(use_amounts_dollars)
        ]

    per_use_results: list[list[WindowReplayResult]] = [[] for _ in use_amounts_dollars]
    for offset, window_idx in enumerate(range(int(selected_window_start), int(selected_window_end))):
        first = int(first_trigger_event_idx[offset])
        if first < 0:
            for idx, use_amount in enumerate(use_amounts_dollars):
                per_use_results[idx].append(_zero_window_result(int(use_amount)))
            if progress_counter is not None and progress_slot >= 0:
                progress_counter[progress_slot] += len(use_amounts_dollars)
            continue
        end = int(window_event_start[window_idx + 1])
        results = simulate_window_many_uses(
            event_side[first:end],
            event_price_1e4[first:end],
            event_count_centi[first:end],
            int(window_outcome[window_idx]),
            buy_price_1e4,
            sell_price_1e4,
            use_amounts_dollars,
        )
        for idx, result in enumerate(results):
            per_use_results[idx].append(result)
        if progress_counter is not None and progress_slot >= 0:
            progress_counter[progress_slot] += len(use_amounts_dollars)

    selected_days = window_day_ord[int(selected_window_start):int(selected_window_end)]
    return [
        aggregate_candidate_metrics(
            buy_price_1e4,
            sell_price_1e4,
            int(use_amount),
            per_use_results[idx],
            selected_days,
        )
        for idx, use_amount in enumerate(use_amounts_dollars)
    ]
