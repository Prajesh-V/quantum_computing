"""Resource-formula verification and the resource/error correlation study.

Phase 1's closed-form counts are the independent reference here:

    order 1:              CNOT = 2r(n-1),        RZ = r(n-1),        RX = rn
    order 2 (merged):     CNOT = 2(r+1)(n-1),    RZ = (r+1)(n-1),    RX = rn

The circuits used by the noise experiments must follow them exactly, and the
stored CSV counts must equal what the circuit builder produces.  The
correlation study then asks which resource tracks the noise-induced error --
it is a correlation, not a causal claim, and the counts are almost collinear in
``r``, which limits what can be concluded.
"""

from __future__ import annotations

import numpy as np

import circuit_cost

__all__ = ["verify_formulas", "correlation_study"]


def verify_formulas(ns=(3, 4, 5, 6), rs=(1, 2, 3, 4, 6, 8, 12, 24),
                    orders=(1, 2)) -> dict:
    """Compare the analytical counts, the compiled counts and the stored CSV."""
    rows = []
    worst = 0
    for n in ns:
        for order in orders:
            for r in rs:
                gates = circuit_cost.build_circuit(n, r, 1.0, 1.0, 1.0, order)
                tally = {"cnot": 0, "rx": 0, "rz": 0}
                for gate in gates:
                    tally[gate.gate_type] += 1
                analytic = circuit_cost.analytic_counts(n, r, order)
                resources = circuit_cost.estimate_resources(n, r, 1.0, 1.0, 1.0, order)
                differences = {
                    "cnot_circuit": tally["cnot"] - analytic["cnot"],
                    "cnot_estimate": resources.num_cnot - analytic["cnot"],
                    "rz_circuit": tally["rz"] - analytic["rz"],
                    "rz_estimate": resources.num_rz - analytic["rz"],
                    "rx_circuit": tally["rx"] - analytic["rx"],
                    "rx_estimate": resources.num_rx - analytic["rx"],
                }
                worst = max(worst, max(abs(value) for value in differences.values()))
                rows.append({"system_size": n, "trotter_order": order,
                             "trotter_steps": r,
                             "cnot_formula": analytic["cnot"],
                             "cnot_circuit": tally["cnot"],
                             "rz_formula": analytic["rz"], "rz_circuit": tally["rz"],
                             "rx_formula": analytic["rx"], "rx_circuit": tally["rx"],
                             "depth": resources.circuit_depth,
                             "two_qubit_depth": resources.two_qubit_depth,
                             **{key: value for key, value in differences.items()}})
    return {"rows": rows, "max_abs_difference": int(worst),
            "cases": len(rows), "passed": worst == 0}


def verify_stored_counts(rows) -> dict:
    """Do the counts recorded in ``gate_noise_results.csv`` match the builder?"""
    mismatches = 0
    checked = 0
    for row in rows:
        n = int(row["system_size"])
        r = int(row["trotter_steps"])
        order = int(row["trotter_order"])
        resources = circuit_cost.estimate_resources(n, r, 1.0, 1.0, 1.0, order)
        checked += 1
        if (resources.num_cnot != int(row["cnot_count"])
                or resources.num_rx != int(row["rx_count"])
                or resources.num_rz != int(row["rz_count"])
                or resources.total_gates != int(row["total_gate_count"])
                or resources.circuit_depth != int(row["logical_depth"])
                or resources.two_qubit_depth != int(row["two_qubit_depth"])):
            mismatches += 1
    return {"checked": checked, "mismatches": mismatches, "passed": mismatches == 0}


def correlation_study(rows, metric="noise_induced_error") -> dict:
    """Pearson correlation of the noise error with each resource, per group.

    Groups are ``(system size, order, noise level)``; within a group the step
    count varies, which is what drives every resource at once.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["total_error"] <= 0:
            continue
        key = (int(row["system_size"]), int(row["trotter_order"]),
               row.get("single_qubit_error", 0.0), row.get("global_noise_probability", 0.0))
        groups.setdefault(key, []).append(row)

    resources = {"cnot_count": "CNOT count", "total_gate_count": "total gate count",
                 "logical_depth": "logical depth", "two_qubit_depth": "two-qubit depth",
                 "rx_count": "RX count", "rz_count": "RZ count"}
    detail = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 4:
            continue
        error = np.array([row[metric] for row in members], dtype=float)
        if np.allclose(error, error[0]):
            continue
        entry = {"system_size": key[0], "trotter_order": key[1],
                 "single_qubit_error": key[2], "global_noise_probability": key[3],
                 "rows": len(members)}
        for column, label in resources.items():
            values = np.array([row[column] for row in members], dtype=float)
            if np.allclose(values, values[0]):
                entry[column] = float("nan")
                continue
            entry[column] = float(np.corrcoef(np.log(values), error)[0, 1])
        detail.append(entry)

    summary = {}
    for column, label in resources.items():
        values = [entry[column] for entry in detail if not np.isnan(entry[column])]
        if values:
            summary[label] = {
                "mean_pearson_log_resource": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "groups": len(values),
            }
    return {"groups": detail, "summary": summary,
            "note": "counts are collinear in the step count, so these correlations "
                    "do not separate the resources from one another"}
