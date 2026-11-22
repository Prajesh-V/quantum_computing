"""Resource scaling with qubit count and with Trotter step count.

Analytical reference (open chain, ``nb = n - 1``), as derived and validated in
Phase 1 and re-checked here against every measured row:

    order 1:            CNOT = 2 r nb,          RZ = r nb,          RX = r n
                        total = r (4n - 3)
    order 2 (merged):   CNOT = 2 (r+1) nb,      RZ = (r+1) nb,      RX = r n
                        total = 3 (r+1) nb + r n

``circuit_cost.analytic_counts`` is imported rather than re-implemented.
"""

from __future__ import annotations

import numpy as np

import circuit_cost

from .resource_analysis import Dataset, NUMERICAL_FLOOR

FIELDS = ("cnot", "rz", "rx", "total")


def analytic_resources(n: int, r: int, order: int) -> dict[str, int]:
    """Closed-form gate counts for one configuration (reused from Phase 1)."""
    return circuit_cost.analytic_counts(n, r, order)


def verify_analytic(dataset: Dataset) -> dict:
    """Compare every measured row with the closed-form counts."""
    worst = 0
    mismatches = []
    for row in dataset.rows:
        ref = analytic_resources(row.n, row.r, row.order)
        for field in FIELDS:
            measured = getattr(row, field)
            diff = measured - ref[field]
            worst = max(worst, abs(diff))
            if diff:
                mismatches.append({"n": row.n, "r": row.r, "order": row.order,
                                   "field": field, "measured": measured,
                                   "analytic": ref[field]})
    return {"max_abs_difference": worst, "all_match": worst == 0,
            "n_mismatches": len(mismatches), "mismatches": mismatches[:10],
            "rows_checked": len(dataset.rows)}


def table_at_fixed_r(dataset: Dataset, r: int = 200) -> list[dict]:
    """Measured resources for every ``n`` at one fixed step count."""
    out = []
    for n in dataset.n_values():
        for order in dataset.orders():
            match = [row for row in dataset.series(n, order) if row.r == r]
            if not match:
                continue
            row = match[0]
            ref = analytic_resources(n, r, order)
            out.append({
                "n": n, "order": order, "r": r,
                "cnot": row.cnot, "total": row.total, "depth": row.depth,
                "twoq": row.twoq,
                "cnot_analytic": ref["cnot"], "total_analytic": ref["total"],
            })
    return out


def empirical_exponents(dataset: Dataset, r_ref: int = 200) -> dict:
    """Fit the exponents the analytical formulas predict.

    * ``cnot ~ (n-1)^1`` at fixed ``r`` (one bond per extra qubit)
    * ``total ~ r^1`` at fixed ``n`` (a step is repeated, not resized)
    * ``depth ~ r^1`` at fixed ``n``
    * ``depth ~ n^0`` at fixed ``r`` (only the ``3(n-1)`` boundary term grows)
    """
    out: dict[str, dict] = {}
    stats = {}
    for order in dataset.orders():
        # resources against qubit count at fixed r
        ns, cnots, depths = [], [], []
        for n in dataset.n_values():
            match = [row for row in dataset.series(n, order) if row.r == r_ref]
            if not match:
                continue
            ns.append(n - 1)
            cnots.append(match[0].cnot)
            depths.append(match[0].depth)
        if len(ns) >= 2:
            cnot_slope = float(np.polyfit(np.log(ns), np.log(cnots), 1)[0])
            depth_slope = float(np.polyfit(np.log(ns), np.log(depths), 1)[0])
        else:
            cnot_slope = depth_slope = float("nan")
        stats[order] = {"cnot_vs_bonds": cnot_slope,
                        "depth_vs_bonds": depth_slope,
                        "depth_at_r": dict(zip([n for n in dataset.n_values()], depths))}

        # resources against r at fixed n
        n0 = dataset.n_values()[0]
        series = dataset.series(n0, order)
        rs = np.array([row.r for row in series], dtype=float)
        totals = np.array([row.total for row in series], dtype=float)
        depth = np.array([row.depth for row in series], dtype=float)
        stats[order]["total_vs_r"] = float(np.polyfit(np.log(rs), np.log(totals), 1)[0])
        stats[order]["depth_vs_r"] = float(np.polyfit(np.log(rs), np.log(depth), 1)[0])
        # CNOT difference at equal r; the formulas predict 2(n-1), independent of r
        stats[order]["fixed_overhead_cnot"] = {
            n: sum(row2.cnot - row1.cnot
                   for row1, row2 in zip(dataset.series(n, 1), dataset.series(n, 2)))
            / max(1, len(dataset.series(n, 1)))
            for n in dataset.n_values()}
    out["by_order"] = stats
    out["r_ref"] = r_ref
    return out


def depth_vs_qubits_claim(dataset: Dataset, r_ref: int = 200) -> dict:
    """Quantify how weakly depth depends on ``n`` at fixed ``r``."""
    out = {}
    for order in dataset.orders():
        values = {}
        for n in dataset.n_values():
            match = [row for row in dataset.series(n, order) if row.r == r_ref]
            if match:
                values[n] = match[0].depth
        if len(values) >= 2:
            spread = (max(values.values()) - min(values.values())) / min(values.values())
            out[order] = {"depth": values, "spread_pct": 100.0 * spread}
    return out


def step_counts_for_error(dataset: Dataset, n: int, order: int,
                          threshold: float) -> int | None:
    """Largest measured ``r`` still above ``threshold``, if the target is met.

    Returns ``None`` when no measured configuration reaches the threshold, and
    the smallest such ``r`` otherwise.  Used only for reporting, never for
    extrapolation.
    """
    series = [row for row in dataset.series(n, order)
              if row.err_op <= threshold and row.err_op > NUMERICAL_FLOOR]
    if not series:
        return None
    return min(row.r for row in series)
