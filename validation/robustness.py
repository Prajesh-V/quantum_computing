"""Robustness experiments around the noise-dependent optimal step count.

All runs here use :mod:`validation.independent` (the from-scratch engine), not
the audited Phase 3 code, so these are genuinely new measurements rather than
re-runs of the original sweep.

Five experiments, as the audit brief requires:

1. initial-state dependence;
2. total-evolution-time dependence;
3. noise-strength trend (configuration by configuration, from the stored data);
4. single-qubit versus two-qubit noise contributions;
5. candidate-grid sensitivity around the reported optimum.

Nothing here extrapolates: every reported optimum is the best value among the
step counts that were actually simulated, and is labelled as such.
"""

from __future__ import annotations

import numpy as np

from . import independent as ind

__all__ = [
    "R_GRID",
    "INITIAL_STATES",
    "initial_states",
    "initial_state_experiment",
    "evolution_time_experiment",
    "noise_strength_check",
    "noise_components_experiment",
    "candidate_grid_experiment",
    "argmin_row",
    "with_optimal_flags",
    "classify",
]

# 1, 2, 3, 4, 6, 8, ... is the Phase 3 grid; trimming the top keeps the
# independent engine's runtime reasonable (its channels are explicit Pauli sums).
R_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64)

INITIAL_STATES = ("|0...0>", "|1...1>", "|+...+>", "random")

RANDOM_SEED = 20260522


def initial_states(n: int) -> dict[str, np.ndarray]:
    """The four initial states, built explicitly (qubit 0 is the MSB)."""
    zero = np.zeros(2 ** n, dtype=complex)
    zero[0] = 1.0
    one = np.zeros(2 ** n, dtype=complex)
    one[-1] = 1.0
    plus = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    random = ind.random_pure_state(n, RANDOM_SEED)
    return {"|0...0>": zero, "|1...1>": one, "|+...+>": plus, "random": random}


def classify(rows, best) -> str:
    steps = [row["trotter_steps"] for row in rows]
    if best["trotter_steps"] == min(steps):
        return "at_smallest_tested_r"
    if best["trotter_steps"] == max(steps):
        return "at_largest_tested_r"
    return "interior"


def argmin_row(rows):
    return min(rows, key=lambda row: row["total_error"])


def with_optimal_flags(rows, group_keys) -> list[dict]:
    """Mark, inside each group, the row whose total error is the group minimum."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    for members in groups.values():
        best = argmin_row(members)
        for row in members:
            row["is_optimal"] = row is best
            row["optimal_r"] = best["trotter_steps"]
            row["optimum_status"] = classify(members, best)
    return rows


def _summary(group_keys, rows, extra=None) -> dict:
    best = argmin_row(rows)
    out = dict(group_keys)
    out.update({
        "optimal_r": best["trotter_steps"],
        "status": classify(rows, best),
        "minimum_total_error": best["total_error"],
        "fidelity_at_optimum": best["fidelity"],
        "ideal_trotter_error_at_optimum": best["ideal_trotter_error"],
        "noise_error_at_optimum": best["noise_induced_error"],
        "purity_at_optimum": best["purity"],
        "cnot_at_optimum": best["cnot_count"],
        "total_gates_at_optimum": best["total_gate_count"],
        "candidates": len(rows),
        "tested_r_min": min(row["trotter_steps"] for row in rows),
        "tested_r_max": max(row["trotter_steps"] for row in rows),
    })
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------

def initial_state_experiment(ns=(4,), orders=(1, 2), levels=(0.0, 1e-3, 5e-3, 1e-2),
                             grid=R_GRID, states=INITIAL_STATES) -> dict:
    """Optimal step count for four initial states, one noise model at a time."""
    rows_out: list[dict] = []
    summary: list[dict] = []

    for n in ns:
        vectors = initial_states(n)
        for name in states:
            psi0 = vectors[name]
            for order in orders:
                for level in levels:
                    spec = ind.Spec("gate" if level > 0 else "none", p1=level, p2=level)
                    group = []
                    for r in grid:
                        row = ind.measure(n, r, order, 1.0, spec, psi0=psi0)
                        row["initial_state"] = name
                        group.append(row)
                        rows_out.append(row)
                    summary.append(_summary(
                        {"initial_state": name, "system_size": n,
                         "trotter_order": order, "single_qubit_error": level},
                        group))

    return {"rows": rows_out, "summary": summary}


# ---------------------------------------------------------------------------
# 2. Total evolution time
# ---------------------------------------------------------------------------

def evolution_time_experiment(ts=(0.5, 1.0, 2.0, 4.0), n: int = 4,
                              orders=(1, 2), levels=(1e-3,), grid=R_GRID) -> dict:
    """Does the optimal ``r`` move when the simulated time changes?"""
    rows_out: list[dict] = []
    summary: list[dict] = []
    for t in ts:
        for order in orders:
            for level in levels:
                spec = ind.Spec("gate", p1=level, p2=level)
                group = []
                for r in grid:
                    row = ind.measure(n, r, order, float(t), spec)
                    group.append(row)
                    rows_out.append(row)
                summary.append(_summary(
                    {"total_time": float(t), "system_size": n,
                     "trotter_order": order, "single_qubit_error": level},
                    group))
    return {"rows": rows_out, "summary": summary}


# ---------------------------------------------------------------------------
# 3. Noise-strength trend, from the stored optima
# ---------------------------------------------------------------------------

def noise_strength_check(optimal_rows) -> dict:
    """Is ``r_opt`` monotonically decreasing in the noise strength?

    Groups are formed only within a fixed system size, formula order and noise
    model, so different sizes and decompositions are never pooled.  For gate
    noise the running parameter is ``p1`` (with ``p2 = p1`` in these runs); for
    the global models it is ``p_g``.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in optimal_rows:
        key = (row["system_size"], row["trotter_order"], row["noise_model"])
        groups.setdefault(key, []).append(row)

    details = []
    for key in sorted(groups):
        members = groups[key]
        parameter = "global_noise_probability" if key[2].startswith("global") else "single_qubit_error"
        members = sorted(members, key=lambda row: row[parameter])
        levels = [row[parameter] for row in members]
        steps = [row["optimal_r"] for row in members]
        deltas = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
        strictly = all(delta < 0 for delta in deltas)
        non_increasing = all(delta <= 0 for delta in deltas)
        increases = [d for d in deltas if d > 0]
        repeated = [d for d in deltas if d == 0]
        details.append({
            "system_size": key[0],
            "trotter_order": key[1],
            "noise_model": key[2],
            "parameter": parameter,
            "levels": levels,
            "optimal_steps": steps,
            "strictly_decreasing": strictly,
            "non_increasing": non_increasing,
            "violations": len(increases),
            "max_increase": max(increases) if increases else 0,
            "flat_steps": len(repeated),
            "deltas": deltas,
            "zeros_excluded": bool(levels and levels[0] == 0.0),
        })

    usable = [group for group in details if not group["zeros_excluded"]]
    return {
        "groups": details,
        "groups_without_noise_free": len(usable),
        "strictly_decreasing": sum(1 for group in usable if group["strictly_decreasing"]),
        "non_increasing": sum(1 for group in usable if group["non_increasing"]),
        "violations": sum(group["violations"] for group in usable),
        "max_increase": max((group["max_increase"] for group in usable), default=0),
        "groups_including_noise_free": sum(1 for group in details if group["zeros_excluded"]),
        "noise_free_groups_monotone": sum(
            1 for group in details
            if group["zeros_excluded"] and group["non_increasing"]),
    }


# ---------------------------------------------------------------------------
# 4. Single-qubit versus two-qubit contributions
# ---------------------------------------------------------------------------

def noise_components_experiment(n: int = 4, orders=(1, 2),
                                levels=(1e-3, 5e-3, 1e-2),
                                grid=R_GRID) -> dict:
    """``p1`` alone, ``p2`` alone, and both together."""
    variants = (("single_qubit_only", 1.0, 0.0), ("two_qubit_only", 0.0, 1.0),
                ("both", 1.0, 1.0))
    rows_out: list[dict] = []
    summary: list[dict] = []

    for order in orders:
        for level in levels:
            for label, scale1, scale2 in variants:
                spec = ind.Spec("gate", p1=level * scale1, p2=level * scale2)
                group = []
                for r in grid:
                    row = ind.measure(n, r, order, 1.0, spec)
                    row["variant"] = label
                    group.append(row)
                    rows_out.append(row)
                entry = _summary({"variant": label, "system_size": n,
                                  "trotter_order": order,
                                  "nominal_level": level}, group)
                entry["single_qubit_error"] = level * scale1
                entry["two_qubit_error"] = level * scale2
                # error at a common step count, for a like-for-like comparison
                at_four = [row for row in group if row["trotter_steps"] == 4]
                entry["noise_error_at_r4"] = at_four[0]["noise_induced_error"] if at_four else None
                entry["total_error_at_r4"] = at_four[0]["total_error"] if at_four else None
                entry["rx_count"] = group[0]["rx_count"]
                entry["rz_count"] = group[0]["rz_count"]
                entry["cnot_count"] = group[0]["cnot_count"]
                summary.append(entry)

    # per-qubit * and per-CNOT exposure: error per gate class, normalised by count
    for entry in summary:
        level = entry["nominal_level"]
        if entry["variant"] == "single_qubit_only" and entry["noise_error_at_r4"]:
            entry["share_explained"] = "single-qubit gates only"
        elif entry["variant"] == "two_qubit_only" and entry["noise_error_at_r4"]:
            entry["share_explained"] = "two-qubit gates only"
        else:
            entry["share_explained"] = "both classes"
        del level
    return {"rows": rows_out, "summary": summary}


# ---------------------------------------------------------------------------
# 5. Candidate-grid sensitivity
# ---------------------------------------------------------------------------

def candidate_grid_experiment(optimal_rows, ns=(4, 5, 6), orders=(1, 2),
                              level: float = 1e-3, span: int = 8) -> dict:
    """Re-sweep integer step counts around the reported optimum.

    The Phase 3 grid skips 5, 7, 9, 10, 11, ... so an "optimum at 4" only means
    "best of the tested set".  Here every integer in ``[r_opt - span, r_opt + span]``
    (clipped at 1) is simulated and the best of *that* set is reported.
    """
    lookup = {}
    for row in optimal_rows:
        if (row["noise_model"] == "gate"
                and abs(row["single_qubit_error"] - level) < 1e-15
                and int(row["system_size"]) in ns):
            lookup[(int(row["system_size"]), int(row["trotter_order"]))] = int(row["optimal_r"])

    rows_out: list[dict] = []
    summary: list[dict] = []
    for (n, order), stored_r in sorted(lookup.items()):
        if order not in orders:
            continue
        spec = ind.Spec("gate", p1=level, p2=level)
        candidate_rs = sorted({r for r in range(max(1, stored_r - span),
                                               stored_r + span + 1)})
        group = []
        for r in candidate_rs:
            row = ind.measure(n, r, order, 1.0, spec)
            group.append(row)
            rows_out.append(row)
        best = argmin_row(group)
        summary.append({
            "system_size": n,
            "trotter_order": order,
            "single_qubit_error": level,
            "phase3_grid_optimal_r": stored_r,
            "dense_grid_optimal_r": best["trotter_steps"],
            "dense_grid_minimum_error": best["total_error"],
            "dense_grid_candidates": len(candidate_rs),
            "dense_grid_range": f"{candidate_rs[0]}-{candidate_rs[-1]}",
            "optimum_changed": best["trotter_steps"] != stored_r,
            "error_at_phase3_r": [row for row in group
                                  if row["trotter_steps"] == stored_r][0]["total_error"],
            "status": classify(group, best),
        })
    return {"rows": rows_out, "summary": summary}
