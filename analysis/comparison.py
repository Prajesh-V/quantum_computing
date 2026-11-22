"""Formula comparisons at equal steps, equal CNOT budgets and equal gate budgets.

All errors are ``operator_norm_error`` values measured by the Phase 1 pipeline.
Nothing here is extrapolated: when a comparison needs an error at a budget that
was not measured exactly, it is obtained by *log-log interpolation between two
measured points of the same curve*, and every such value is flagged
(``interpolated=True``).  Interpolation outside the measured range is refused.
"""

from __future__ import annotations

import numpy as np

from .resource_analysis import Row, Dataset, NUMERICAL_FLOOR

BUDGET_METRICS = {
    "cnot": "cnot",
    "total_gates": "total",
    "depth": "depth",
    "two_qubit_depth": "twoq",
}


def _x(row: Row, metric: str) -> float:
    return float(getattr(row, BUDGET_METRICS[metric]))


def equal_steps(dataset: Dataset, n: int) -> list[dict]:
    """Every ``r`` measured for both orders, with both resource vectors.

    This is the cleanest comparison available: ``t``, ``J``, ``h`` and ``n`` are
    identical, and only the formula changes.
    """
    by_r = {}
    for order in dataset.orders():
        for row in dataset.series(n, order):
            by_r.setdefault(row.r, {})[order] = row
    out = []
    for r in sorted(by_r):
        pair = by_r[r]
        if 1 not in pair or 2 not in pair:
            continue
        a, b = pair[1], pair[2]
        out.append({
            "n": n,
            "r": r,
            "error_order1": a.err_op,
            "error_order2": b.err_op,
            "error_ratio": (b.err_op / a.err_op) if a.err_op else float("nan"),
            "cnot_order1": a.cnot, "cnot_order2": b.cnot,
            "total_order1": a.total, "total_order2": b.total,
            "depth_order1": a.depth, "depth_order2": b.depth,
            "twoq_order1": a.twoq, "twoq_order2": b.twoq,
            "cnot_overhead_pct": 100.0 * (b.cnot - a.cnot) / a.cnot,
            "gate_overhead_pct": 100.0 * (b.total - a.total) / a.total,
            "depth_overhead_pct": 100.0 * (b.depth - a.depth) / a.depth,
        })
    return out


def interpolate_error_at(series: list[Row], target: float, metric: str = "cnot",
                         floor: float = 0.0):
    """Error of one curve at a given resource budget.

    Exact match returns the measured value with ``interpolated=False``.  A
    budget inside the measured range is interpolated linearly in log-log space
    between its two neighbouring measured points (justified by the power-law
    error decay established in Phase 1, and flagged in the result).  A budget
    outside the range returns ``(None, False)`` - never an extrapolation.

    Returns ``(error, interpolated)``.
    """
    if not series:
        return None, False
    xs = np.array([_x(row, metric) for row in series], dtype=float)
    ys = np.array([row.err_op for row in series], dtype=float)
    keep = ys > floor
    xs, ys = xs[keep], ys[keep]
    if len(xs) == 0:
        return None, False

    exact = np.isclose(xs, target, rtol=0, atol=0)
    if exact.any():
        return float(ys[exact][0]), False
    if target <= xs.min() or target >= xs.max():
        return None, False

    hi = int(np.searchsorted(xs, target))
    x0, x1 = xs[hi - 1], xs[hi]
    y0, y1 = ys[hi - 1], ys[hi]
    if x0 == x1:
        return float(y0), False
    weight = (np.log(target) - np.log(x0)) / (np.log(x1) - np.log(x0))
    return float(np.exp(np.log(y0) + weight * (np.log(y1) - np.log(y0)))), True


def equal_budget(dataset: Dataset, n: int, metric: str = "cnot") -> list[dict]:
    """Compare orders at the same resource budget.

    For each measured second-order configuration (budget ``B``) this reports:

    * the first-order error interpolated to exactly ``B`` (same-budget
      comparison, flagged ``mode="interpolated"``), and
    * the best first-order configuration that uses **no more** than ``B``
      (``mode="nearest_feasible"``), with its actual budget gap.

    Both are given so a reader can see whether the conclusion depends on the
    interpolation.
    """
    first = dataset.series(n, 1)
    second = dataset.series(n, 2)
    if not first or not second:
        return []

    out = []
    for row in second:
        budget = _x(row, metric)
        feasible = [other for other in first if _x(other, metric) <= budget]
        nearest = max(feasible, key=lambda other: _x(other, metric)) if feasible else None
        interp_err, interp_flag = interpolate_error_at(first, budget, metric)

        gap = (100.0 * (_x(nearest, metric) - budget) / budget) if nearest else float("nan")
        out.append({
            "n": n,
            "order2_r": row.r,
            "budget_metric": metric,
            "budget": budget,
            "error_order2": row.err_op,
            "error_order1_interpolated": interp_err,
            "interpolated": bool(interp_flag),
            "nearest_r_order1": nearest.r if nearest else None,
            "nearest_budget_order1": _x(nearest, metric) if nearest else float("nan"),
            "budget_gap_pct": gap,
            "error_order1_nearest": nearest.err_op if nearest else float("nan"),
            "error_reduction_pct_interpolated": (
                100.0 * (interp_err - row.err_op) / interp_err
                if interp_err else float("nan")),
            "error_reduction_pct_nearest": (
                100.0 * (nearest.err_op - row.err_op) / nearest.err_op
                if nearest else float("nan")),
            "mode": "interpolated" if interp_err is not None else "nearest_feasible",
        })
    return out


def comparison_table(dataset: Dataset, n: int) -> list[dict]:
    """The §7.4 table: measured error and all resources, both formulas."""
    rows = []
    for order in (1, 2):
        for row in dataset.series(n, order):
            rows.append({
                "Qubits": row.n, "Formula": f"order {row.order}", "Steps": row.r,
                "Error": row.err_op, "CNOT": row.cnot, "RZ": row.rz, "RX": row.rx,
                "Total gates": row.total,
                "Depth (conservative scheduler)": row.depth,
                "Two-qubit depth": row.twoq,
                "Source": row.source,
            })
    return rows


def accuracy_per_cnot(dataset: Dataset) -> dict[int, float]:
    """Local log-log slope of error vs CNOT count for each ``(n, order)``.

    A slope of ``-1`` means each extra decade of CNOTs buys one decade of
    accuracy; ``-2`` buys two.  This is the honest form of "accuracy per CNOT":
    a slope, not an invented score.
    """
    out = {}
    for n in dataset.n_values():
        for order in dataset.orders():
            series = dataset.series(n, order)
            x = np.array([row.cnot for row in series], dtype=float)
            y = np.array([row.err_op for row in series], dtype=float)
            mask = y > NUMERICAL_FLOOR
            if mask.sum() < 3:
                continue
            out[n, order] = float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])
    return out


def gate_savings_pct(gates_first: float, gates_second: float) -> float:
    """``[(G_first - G_second) / G_first] * 100``; >0 means order 2 is cheaper."""
    if not gates_first:
        return float("nan")
    return 100.0 * (gates_first - gates_second) / gates_first


def error_reduction_pct(error_first: float, error_second: float) -> float:
    """``[(E_first - E_second) / E_first] * 100``."""
    if not error_first:
        return float("nan")
    return 100.0 * (error_first - error_second) / error_first
