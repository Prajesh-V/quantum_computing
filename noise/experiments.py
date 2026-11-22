"""Phase 3 measurements: noisy sweeps, sensitivity grids and the optimal-r search.

Every number in the Phase 3 tables comes from one call to
``noisy_simulator.simulate`` on a circuit built by ``circuit_cost``.  Three error
quantities are kept separate for every configuration, because they answer
different questions and are **not** additive:

``ideal_trotter_error``  ``1 - F(rho_trotter, psi_exact)``  - formula truncation
``noise_induced_error``  ``1 - F(rho_noisy,   psi_trotter)`` - what the channels did
``total_error``          ``1 - F(rho_noisy,   psi_exact)``  - what is actually wrong

``psi_exact = exp(-iHt)|00...0>`` and ``psi_trotter = U_trotter|00...0>`` with
``U_trotter`` from ``trotter.trotter_unitary`` (unchanged since Phase 1).

Nothing here extrapolates: the optimal step count is the argmin over the step
counts that were actually simulated, and a minimum at either end of the tested
grid is reported as a boundary result rather than fitted into an interior one.
"""

from __future__ import annotations

import math

import numpy as np

from . import metrics
from .density_matrix import (check_density_matrix, magnetization_x, purity,
                             zz_correlation)
from .noisy_simulator import (NoiseModel, exact_state, ideal_trotter_state,
                              reference_states, simulate)

__all__ = [
    "DEFAULT_STEPS",
    "DENSE_STEPS",
    "NOISE_LEVELS",
    "measure",
    "sweep",
    "noise_sensitivity",
    "optimal_step_count",
    "pair_orders",
    "noise_key",
    "GLOBAL_COLUMNS",
    "GATE_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "OPTIMAL_COLUMNS",
    "CONFIG_COLUMNS",
]

#: Mixed grid: dense where the optimum of a flat low-noise curve lives, coarse
#: where only the noise tail matters.  Chosen after timing the simulator.
DEFAULT_STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512)

#: Dense integer grid up to 16 to eliminate discrete grid artefacts:
DENSE_STEPS = tuple(range(1, 17)) + (24, 32, 64, 128, 256, 512)

#: Illustrative simulation parameters.  These are NOT measured hardware error
#: rates and must not be read as device specifications.
NOISE_LEVELS = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)

GLOBAL_COLUMNS = [
    "system_size", "trotter_order", "trotter_steps", "total_time",
    "global_noise_probability", "cnot_count", "rx_count", "rz_count",
    "total_gate_count", "logical_depth", "fidelity", "infidelity", "purity",
    "total_error", "ideal_trotter_error", "noise_induced_error",
    "two_qubit_depth", "noise_channels_applied", "trace_distance",
    "noise_model",
]

GATE_COLUMNS = [
    "system_size", "trotter_order", "trotter_steps", "total_time",
    "single_qubit_error", "two_qubit_error", "cnot_count", "rx_count",
    "rz_count", "total_gate_count", "logical_depth", "ideal_trotter_error",
    "noise_induced_error", "total_error", "fidelity", "purity",
    "trace_distance", "magnetization", "zz_correlation", "two_qubit_depth",
    "noise_channels_applied",
]

SENSITIVITY_COLUMNS = [
    "system_size", "trotter_order", "trotter_steps", "single_qubit_error",
    "two_qubit_error", "global_noise_probability",     "noise_model",
    "cnot_count", "total_gate_count", "logical_depth", "fidelity",
    "total_error", "noise_induced_error", "purity", "grid",
]

OPTIMAL_COLUMNS = [
    "system_size", "trotter_order", "noise_model", "single_qubit_error",
    "two_qubit_error", "global_noise_probability", "total_time", "optimal_r",
    "optimal_fidelity", "minimum_total_error", "ideal_trotter_error_at_optimum",
    "noise_error_at_optimum", "cnot_count_at_optimum",
    "total_gate_count_at_optimum", "logical_depth_at_optimum", "optimum_status",
]

CONFIG_COLUMNS = [
    "system_size", "trotter_order", "trotter_steps", "single_qubit_error",
    "two_qubit_error", "global_noise_probability", "noise_model",
    "cnot_count", "rx_count", "rz_count", "total_gate_count", "logical_depth",
    "fidelity", "total_error", "ideal_trotter_error", "noise_induced_error",
]


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

def measure(n: int, r: int, order: int = 1, t: float = 1.0, J: float = 1.0,
            h: float = 1.0, model: NoiseModel | None = None, *,
            merge_half_steps: bool = True) -> dict:
    """Simulate one (n, r, order, noise) configuration and return every metric."""
    model = model or NoiseModel()
    out = simulate(n, int(r), order, t, J, h, model,
                   merge_half_steps=merge_half_steps)
    rho = out["rho"]
    resources = out["resources"]
    refs = reference_states(n, int(r), order, t, J, h)

    fidelity = metrics.fidelity_pure(rho, refs["psi_exact"])
    total_error = metrics.clip_unit_interval(1.0 - fidelity)
    ideal_error = metrics.infidelity(refs["rho_trotter"], refs["psi_exact"])
    noise_error = metrics.infidelity(rho, refs["psi_trotter"])

    row = {
        "system_size": n,
        "trotter_order": int(order),
        "trotter_steps": int(r),
        "total_time": float(t),
        "single_qubit_error": float(model.single_qubit_error),
        "two_qubit_error": float(model.two_qubit_error),
        "global_noise_probability": float(model.global_probability),
        "noise_model": model.kind,
        "cnot_count": resources.num_cnot,
        "rx_count": resources.num_rx,
        "rz_count": resources.num_rz,
        "total_gate_count": resources.total_gates,
        "logical_depth": resources.circuit_depth,
        "two_qubit_depth": resources.two_qubit_depth,
        "fidelity": fidelity,
        "infidelity": total_error,
        "purity": purity(rho),
        "total_error": total_error,
        "ideal_trotter_error": ideal_error,
        "noise_induced_error": noise_error,
        "trace_distance": metrics.trace_distance(rho, refs["rho_exact"]),
        "magnetization": magnetization_x(rho, n),
        "zz_correlation": zz_correlation(rho, n),
        "noise_channels_applied": out["num_channels"],
        "initial_state": "0" * n,
    }
    return row


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep(ns, orders, rs, models, t: float = 1.0, J: float = 1.0,
          h: float = 1.0, *, merge_half_steps: bool = True,
          progress=None) -> list[dict]:
    """Run the full (model, n, order, r) grid, caching references between rows."""
    rows: list[dict] = []
    total = len(list(models)) * len(list(ns)) * len(list(orders)) * len(list(rs))
    done = 0
    for model in models:
        for n in ns:
            for order in orders:
                for r in rs:
                    rows.append(measure(n, r, order, t, J, h, model,
                                        merge_half_steps=merge_half_steps))
                    done += 1
                    if progress is not None:
                        progress(done, total, model, n, order, r)
    return rows


def noise_sensitivity(n: int, r: int, order: int = 2, t: float = 1.0,
                      p1_values=NOISE_LEVELS, p2_values=NOISE_LEVELS,
                      *, include_univariate: bool = True, **kwargs) -> list[dict]:
    """Grid over ``(p_1, p_2)`` plus the two single-parameter slices.

    The slices (``p_2 = 0`` with ``p_1`` varying, and ``p_1 = 0`` with ``p_2``
    varying) isolate which gate class dominates the accumulated error.
    """
    rows = []
    for p1 in p1_values:
        for p2 in p2_values:
            model = NoiseModel(kind="gate", single_qubit_error=p1, two_qubit_error=p2)
            row = measure(n, r, order, t, model=model, **kwargs)
            row["grid"] = "full"
            rows.append(row)

    if include_univariate:
        for p1 in p1_values:
            model = NoiseModel(kind="gate", single_qubit_error=p1, two_qubit_error=0.0)
            row = measure(n, r, order, t, model=model, **kwargs)
            row["grid"] = "single_qubit_only"
            rows.append(row)
        for p2 in p2_values:
            model = NoiseModel(kind="gate", single_qubit_error=0.0, two_qubit_error=p2)
            row = measure(n, r, order, t, model=model, **kwargs)
            row["grid"] = "two_qubit_only"
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Optimal step count
# ---------------------------------------------------------------------------

OPTIMUM_INTERIOR = "interior"
OPTIMUM_SMALLEST = "at_smallest_tested_r"
OPTIMUM_LARGEST = "at_largest_tested_r"
OPTIMUM_SINGLE = "single_candidate"


def optimal_step_count(rows, r_values=None) -> list[dict]:
    """Argmin of the measured total error over the tested step counts.

    Rows are grouped by everything held fixed in the experiment: system size,
    Trotter order, noise model and noise strengths.  The minimum is taken over
    the step counts that appear in the group; whether it sits at an end of the
    tested range decides ``optimum_status``.  A boundary minimum is reported as
    a boundary minimum, never smoothed into an interior one.
    """
    keys = ("system_size", "trotter_order", "noise_model", "single_qubit_error",
            "two_qubit_error", "global_noise_probability")
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    out: list[dict] = []
    grid = sorted({int(r) for r in (r_values if r_values is not None else
                                    (row["trotter_steps"] for row in rows))})
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: row["trotter_steps"])
        best = min(members, key=lambda row: row["total_error"])
        steps = [row["trotter_steps"] for row in members]
        if len(members) == 1:
            status = OPTIMUM_SINGLE
        elif best["trotter_steps"] == min(steps):
            status = OPTIMUM_SMALLEST
        elif best["trotter_steps"] == max(steps):
            status = OPTIMUM_LARGEST
        else:
            status = OPTIMUM_INTERIOR
        out.append({
            "system_size": best["system_size"],
            "trotter_order": best["trotter_order"],
            "noise_model": best["noise_model"],
            "single_qubit_error": best["single_qubit_error"],
            "two_qubit_error": best["two_qubit_error"],
            "global_noise_probability": best["global_noise_probability"],
            "total_time": best["total_time"],
            "optimal_r": best["trotter_steps"],
            "optimal_fidelity": best["fidelity"],
            "minimum_total_error": best["total_error"],
            "ideal_trotter_error_at_optimum": best["ideal_trotter_error"],
            "noise_error_at_optimum": best["noise_induced_error"],
            "cnot_count_at_optimum": best["cnot_count"],
            "total_gate_count_at_optimum": best["total_gate_count"],
            "logical_depth_at_optimum": best["logical_depth"],
            "optimum_status": status,
            "tested_r_min": min(steps),
            "tested_r_max": max(steps),
            "candidates_evaluated": len(members),
            "tested_r_values": list(grid),
        })
    return out


# ---------------------------------------------------------------------------
# First-order vs second-order at matched resources
# ---------------------------------------------------------------------------

def noise_key(row) -> tuple:
    """The noise settings a row was simulated with.

    Two rows may only be compared if these match: pairing a noisy second-order
    circuit against a noise-free first-order one would report the noise as if it
    were a property of the formula.
    """
    return (row["noise_model"], float(row["single_qubit_error"]),
            float(row.get("two_qubit_error", 0.0)),
            float(row.get("global_noise_probability", 0.0)))


def pair_orders(rows, n: int, noise_model: str, metric: str = "cnot") -> list[dict]:
    """Pair each second-order point with a first-order point of comparable cost.

    For every measured second-order configuration, the first-order configuration
    **with identical noise settings** and the largest budget that does not exceed
    the second-order budget is selected.  If even the cheapest such first-order
    circuit is over budget, the cheapest one is used and ``within_budget`` is
    False.

    Both budgets come from ``circuit_cost``, so the comparison is on the same
    counts that Phases 1 and 2 reported, and the two circuits compared always
    carry the same noise.
    """
    if metric not in ("cnot", "total_gate_count"):
        raise ValueError(f"metric must be 'cnot' or 'total_gate_count', got {metric!r}")
    column = "cnot_count" if metric == "cnot" else "total_gate_count"

    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        if (row["system_size"] == n and row["trotter_order"] == 1
                and row["noise_model"] == noise_model):
            buckets.setdefault(noise_key(row), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row[column])
    second = [row for row in rows
              if row["system_size"] == n and row["trotter_order"] == 2
              and row["noise_model"] == noise_model]
    if not buckets or not second:
        return []

    pairs = []
    for row2 in sorted(second, key=lambda row: row["trotter_steps"]):
        bucket = buckets.get(noise_key(row2))
        if not bucket:
            continue
        budget = row2[column]
        feasible = [row for row in bucket if row[column] <= budget]
        within = bool(feasible)
        row1 = feasible[-1] if feasible else bucket[0]
        pairs.append({
            "system_size": n,
            "noise_model": noise_model,
            "single_qubit_error": row2["single_qubit_error"],
            "two_qubit_error": row2.get("two_qubit_error", 0.0),
            "global_noise_probability": row2.get("global_noise_probability", 0.0),
            "budget_metric": metric,
            "order2_r": row2["trotter_steps"],
            "order1_r": row1["trotter_steps"],
            "order2_budget": budget,
            "order1_budget": row1[column],
            "within_budget": within,
            "fidelity_order1": row1["fidelity"],
            "fidelity_order2": row2["fidelity"],
            "total_error_order1": row1["total_error"],
            "total_error_order2": row2["total_error"],
            "ideal_error_order1": row1["ideal_trotter_error"],
            "ideal_error_order2": row2["ideal_trotter_error"],
            "noise_error_order1": row1["noise_induced_error"],
            "noise_error_order2": row2["noise_induced_error"],
            "fidelity_gain": row2["fidelity"] - row1["fidelity"],
            "error_reduction_pct": (
                100.0 * (row1["total_error"] - row2["total_error"]) / row1["total_error"]
                if row1["total_error"] > 0 else float("nan")),
        })
    return pairs


def equal_steps_pairs(rows, n: int, noise_model: str) -> list[dict]:
    """Same ``r`` for both orders, with identical noise settings."""
    index1 = {(noise_key(row), row["trotter_steps"]): row for row in rows
              if row["system_size"] == n and row["trotter_order"] == 1
              and row["noise_model"] == noise_model}
    pairs = []
    for row2 in sorted((row for row in rows
                        if row["system_size"] == n and row["trotter_order"] == 2
                        and row["noise_model"] == noise_model),
                       key=lambda row: row["trotter_steps"]):
        row1 = index1.get((noise_key(row2), row2["trotter_steps"]))
        if row1 is None:
            continue
        pairs.append({
            "system_size": n,
            "noise_model": noise_model,
            "single_qubit_error": row2["single_qubit_error"],
            "two_qubit_error": row2.get("two_qubit_error", 0.0),
            "global_noise_probability": row2.get("global_noise_probability", 0.0),
            "r": row2["trotter_steps"],
            "cnot_order1": row1["cnot_count"],
            "cnot_order2": row2["cnot_count"],
            "fidelity_order1": row1["fidelity"],
            "fidelity_order2": row2["fidelity"],
            "total_error_order1": row1["total_error"],
            "total_error_order2": row2["total_error"],
            "ideal_error_order1": row1["ideal_trotter_error"],
            "ideal_error_order2": row2["ideal_trotter_error"],
            "noise_error_order1": row1["noise_induced_error"],
            "noise_error_order2": row2["noise_induced_error"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Small helpers used by the report
# ---------------------------------------------------------------------------

def geometric_mean(values) -> float:
    """Geometric mean of positive values (used for ratios across n)."""
    array = np.asarray([v for v in values if v > 0], dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(array))))


def is_monotone_decreasing(values) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(array) < 0))


def finite(value) -> float:
    """Replace non-finite numbers with NaN so the CSV stays strict JSON-friendly."""
    value = float(value)
    return value if math.isfinite(value) else float("nan")
