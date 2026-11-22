"""Independent reproduction of selected Phase 3 measurements.

The stored CSV values are compared with a full recalculation by
:mod:`validation.independent`, which shares no physics code with ``noise/``:
the circuit comes from ``circuit_cost.build_circuit`` (as it must, to be the
same circuit), but every channel, every metric and the evolution itself are
re-implemented.

Cases are chosen to cover both orders, all three system sizes, low / reference /
high noise, gate-level and both global variants, and both an optimal and a
clearly non-optimal step count.
"""

from __future__ import annotations

import numpy as np

from . import independent as ind

__all__ = ["CASES", "select_cases", "run"]

# (label, n, order, r, (noise kind, p1, p2, p_g))
CASES = [
    ("n=4 1st order, reference noise, optimal r", 4, 1, 4, ("gate", 1e-3, 1e-3, 0.0)),
    ("n=4 1st order, reference noise, r=64 (overshoot)", 4, 1, 64, ("gate", 1e-3, 1e-3, 0.0)),
    ("n=4 2nd order, reference noise, optimal r", 4, 2, 3, ("gate", 1e-3, 1e-3, 0.0)),
    ("n=4 2nd order, low noise, r=12", 4, 2, 12, ("gate", 1e-4, 1e-4, 0.0)),
    ("n=5 1st order, 5e-3, r=6", 5, 1, 6, ("gate", 5e-3, 5e-3, 0.0)),
    ("n=5 1st order, high noise, r=1", 5, 1, 1, ("gate", 1e-2, 1e-2, 0.0)),
    ("n=6 2nd order, high noise, r=2", 6, 2, 2, ("gate", 1e-2, 1e-2, 0.0)),
    ("n=6 1st order, low noise, r=24", 6, 1, 24, ("gate", 1e-4, 1e-4, 0.0)),
    ("n=4 2nd order, noiseless", 4, 2, 8, ("gate", 0.0, 0.0, 0.0)),
    ("n=4 1st order, global per step, r=8", 4, 1, 8, ("global_per_step", 0.0, 0.0, 1e-3)),
    ("n=6 2nd order, global per step, r=16", 6, 2, 16, ("global_per_step", 0.0, 0.0, 1e-2)),
    # the end-of-circuit reference was run at the reference level p_g = 1e-3 only
    ("n=5 1st order, global end of circuit, r=16", 5, 1, 16,
     ("global_end_of_circuit", 0.0, 0.0, 1e-3)),
]

FLOAT_QUANTITIES = ["fidelity", "total_error", "ideal_trotter_error",
                    "noise_induced_error", "purity", "trace_distance"]
INT_QUANTITIES = ["cnot_count", "rx_count", "rz_count", "total_gate_count",
                  "logical_depth", "noise_channels_applied"]

TOLERANCE = 1e-9


def _find(rows, n, order, r, spec):
    """Pick the stored row matching a case exactly."""
    kind, p1, p2, pg = spec
    for row in rows:
        if (int(row["system_size"]) == n and int(row["trotter_order"]) == order
                and int(row["trotter_steps"]) == r
                and row.get("noise_model", "gate") == kind
                and abs(row["single_qubit_error"] - p1) < 1e-15
                and abs(row["two_qubit_error"] - p2) < 1e-15
                and abs(row["global_noise_probability"] - pg) < 1e-15):
            return row
    raise KeyError(f"stored row missing for {(n, order, r, spec)}")


def select_cases(gate_rows, global_rows):
    """Resolve the case list against the stored results."""
    resolved = []
    for label, n, order, r, spec in CASES:
        source = gate_rows if spec[0] == "gate" else global_rows
        resolved.append((label, n, order, r, spec, _find(source, n, order, r, spec)))
    return resolved


def run(gate_rows, global_rows) -> dict:
    """Recalculate every case independently and compare with the stored row."""
    rows_out: list[dict] = []
    per_case: list[dict] = []

    for label, n, order, r, spec, stored in select_cases(gate_rows, global_rows):
        kind, p1, p2, pg = spec
        model = ind.Spec(kind=kind, p1=p1, p2=p2, pg=pg)
        fresh = ind.measure(n, r, order, 1.0, model)

        worst = 0.0
        failures = 0
        for quantity in FLOAT_QUANTITIES + INT_QUANTITIES:
            value_stored = float(stored[quantity])
            value_fresh = float(fresh[quantity])
            difference = abs(value_stored - value_fresh)
            scale = max(abs(value_stored), 1e-300)
            tolerance = 0.0 if quantity in INT_QUANTITIES else TOLERANCE
            passed = difference <= tolerance
            worst = max(worst, difference)
            failures += 0 if passed else 1
            rows_out.append({
                "case": label,
                "system_size": n,
                "trotter_order": order,
                "trotter_steps": r,
                "noise_model": kind,
                "single_qubit_error": p1,
                "two_qubit_error": p2,
                "global_noise_probability": pg,
                "quantity": quantity,
                "stored": value_stored,
                "independent": value_fresh,
                "abs_difference": difference,
                "rel_difference": difference / scale,
                "tolerance": tolerance,
                "passed": passed,
            })
        per_case.append({
            "case": label,
            "n": n,
            "order": order,
            "r": r,
            "noise_model": kind,
            "max_abs_difference": worst,
            "failures": failures,
            "stored_total_error": float(stored["total_error"]),
            "independent_total_error": float(fresh["total_error"]),
        })

    floats = [row for row in rows_out if row["tolerance"] > 0]
    return {
        "rows": rows_out,
        "cases": per_case,
        "case_count": len(per_case),
        "quantity_count": len(rows_out),
        "max_abs_difference": max(row["abs_difference"] for row in floats),
        "max_rel_difference": max(row["rel_difference"] for row in floats),
        "failures": sum(1 for row in rows_out if not row["passed"]),
        "tolerance": TOLERANCE,
    }


def error_separation(rows) -> dict:
    """``delta = total - (trotter + noise)`` over every stored noisy row."""
    from .audit import error_separation_from_rows
    noisy = [row for row in rows
             if not (row["single_qubit_error"] == 0.0
                     and row["two_qubit_error"] == 0.0
                     and row["global_noise_probability"] == 0.0)]
    return error_separation_from_rows(noisy)
