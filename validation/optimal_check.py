"""Recompute every stored optimal Trotter step count from the raw results.

The check is deliberately done twice over:

* from the stored CSV (the argmin over the candidates that were actually
  simulated), which verifies the *selection* logic for all 72 configurations;
* against the ``optimal_r.csv`` record, which verifies that the reported
  minimum, status and resources belong to the row that achieves it.

Independent recalculation of the total error itself lives in
:mod:`validation.reproduction`; this module checks that the stored sweep is
internally consistent and that no row from another system size, order or noise
level leaks into a group.
"""

from __future__ import annotations

__all__ = ["GROUP_KEYS", "verify_all", "classify"]

GROUP_KEYS = ("system_size", "trotter_order", "noise_model",
              "single_qubit_error", "two_qubit_error", "global_noise_probability")

INTERIOR = "interior"
SMALLEST = "at_smallest_tested_r"
LARGEST = "at_largest_tested_r"
SINGLE = "single_candidate"


def classify(best, members) -> str:
    """Where the minimum sits inside the tested range."""
    steps = [row["trotter_steps"] for row in members]
    if len(members) == 1:
        return SINGLE
    if best["trotter_steps"] == min(steps):
        return SMALLEST
    if best["trotter_steps"] == max(steps):
        return LARGEST
    return INTERIOR


def _group(rows) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[name] for name in GROUP_KEYS)
        groups.setdefault(key, []).append(row)
    for members in groups.values():
        members.sort(key=lambda row: row["trotter_steps"])
    return groups


def verify_all(rows, optimal_rows) -> dict:
    """Check every stored optimum against the sweep it came from."""
    groups = _group(rows)
    out: list[dict] = []
    problems: list[str] = []

    for optimal in optimal_rows:
        key = tuple(optimal[name] for name in GROUP_KEYS)
        members = groups.get(key)
        if not members:
            problems.append(f"no sweep rows for group {key}")
            continue

        best = min(members, key=lambda row: row["total_error"])
        status = classify(best, members)
        steps = [row["trotter_steps"] for row in members]
        in_grid = any(row["trotter_steps"] == optimal["optimal_r"] for row in members)

        passed = (
            best["trotter_steps"] == optimal["optimal_r"]
            and abs(best["total_error"] - optimal["minimum_total_error"]) < 1e-12
            and status == optimal["optimum_status"]
            and in_grid
        )
        if not passed:
            problems.append(
                f"group {key}: stored r={optimal['optimal_r']} "
                f"({optimal['optimum_status']}) vs recomputed "
                f"r={best['trotter_steps']} ({status})")

        # a group must not mix evolution times, and every group in one experiment
        # must share the same candidate grid
        times = {row["total_time"] for row in members}
        if len(times) != 1:
            problems.append(f"group {key} mixes total_time values {sorted(times)}")

        out.append({
            "system_size": optimal["system_size"],
            "trotter_order": optimal["trotter_order"],
            "noise_model": optimal["noise_model"],
            "single_qubit_error": optimal["single_qubit_error"],
            "two_qubit_error": optimal["two_qubit_error"],
            "global_noise_probability": optimal["global_noise_probability"],
            "stored_optimal_r": optimal["optimal_r"],
            "recomputed_optimal_r": best["trotter_steps"],
            "stored_status": optimal["optimum_status"],
            "recomputed_status": status,
            "stored_minimum_error": optimal["minimum_total_error"],
            "recomputed_minimum_error": best["total_error"],
            "stored_fidelity": optimal["optimal_fidelity"],
            "recomputed_fidelity": best["fidelity"],
            "candidates": len(members),
            "tested_r_min": min(steps),
            "tested_r_max": max(steps),
            "optimal_r_in_tested_grid": in_grid,
            "cnot_at_optimum": best["cnot_count"],
            "total_gates_at_optimum": best["total_gate_count"],
            "passed": passed,
        })

    # cross-checks on the grid itself
    grids = {tuple(sorted({row["trotter_steps"] for row in members}))
             for members in groups.values()}
    per_model = {}
    for row in out:
        bucket = per_model.setdefault(row["noise_model"], {"configs": 0, "interior": 0,
                                                           "boundary": 0,
                                                           "r_values": []})
        bucket["configs"] += 1
        if row["recomputed_status"] == INTERIOR:
            bucket["interior"] += 1
        else:
            bucket["boundary"] += 1
        bucket["r_values"].append(row["recomputed_optimal_r"])
    for bucket in per_model.values():
        bucket["r_min"] = min(bucket["r_values"])
        bucket["r_max"] = max(bucket["r_values"])
        bucket["r_values"] = sorted(bucket["r_values"])

    boundary_rows = [row for row in out if row["recomputed_status"] != INTERIOR]
    return {
        "rows": out,
        "configurations": len(out),
        "passed": sum(1 for row in out if row["passed"]),
        "failed": sum(1 for row in out if not row["passed"]),
        "problems": problems,
        "grids": len(grids),
        "grid_values": sorted(grids.pop()) if len(grids) == 1 else "differ",
        "per_noise_model": per_model,
        "boundary_configurations": boundary_rows,
    }
