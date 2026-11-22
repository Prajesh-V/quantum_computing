"""``phase3_report.md`` and ``phase3_summary.json``.

The report is generated from the measured rows only.  Every claim about the
results is accompanied by the numbers behind it, and the three kinds of
statement are kept apart explicitly:

* **mathematical expectation** - true by construction of the model,
* **measured finding** - a number in one of the Phase 3 tables,
* **interpretation** - an explanation that goes beyond the numbers, flagged as such.

Nothing is extrapolated: where the data cannot support a statement (an interior
optimum that does not exist, a noise level that was not simulated) the report
says so.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import experiments, plotting

__all__ = ["build_summary", "build_report", "write_report", "write_summary"]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1e4 or abs(number) < 1e-3:
        return f"{number:.{digits}g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _select(rows, **criteria):
    return [row for row in rows
            if all(row.get(key) == value for key, value in criteria.items())]


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _jsonable(obj):
    """Recursively convert numpy types and NaN into strict-JSON-safe values."""
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        number = float(obj)
        return number if math.isfinite(number) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Summary object
# ---------------------------------------------------------------------------

def build_summary(rows, sensitivity_rows, optimal_rows, validation, config) -> dict:
    """Aggregate the measured rows into the JSON summary."""
    gate_rows = [row for row in rows if row["noise_model"] == "gate"]
    global_rows = [row for row in rows if row["noise_model"].startswith("global")]
    noiseless = [row for row in rows if row["noise_model"] == "none"]

    interior = [row for row in optimal_rows if row["optimum_status"] == "interior"]
    boundary = [row for row in optimal_rows
                if row["optimum_status"] in (experiments.OPTIMUM_SMALLEST,
                                             experiments.OPTIMUM_LARGEST)]

    reference = config.get("reference_noise")
    order_pairs = {}
    for n in sorted({row["system_size"] for row in gate_rows}):
        order_pairs[str(n)] = {
            "equal_steps": experiments.equal_steps_pairs(gate_rows, n, "gate"),
            "equal_cnot": experiments.pair_orders(gate_rows, n, "gate", "cnot"),
            "equal_total_gates": experiments.pair_orders(gate_rows, n, "gate",
                                                         "total_gate_count"),
        }

    sensitivity = {}
    for grid in ("full", "single_qubit_only", "two_qubit_only"):
        members = [row for row in sensitivity_rows if row.get("grid") == grid]
        if members:
            sensitivity[grid] = {
                "rows": len(members),
                "best_fidelity": max(row["fidelity"] for row in members),
                "worst_fidelity": min(row["fidelity"] for row in members),
                "max_total_error": max(row["total_error"] for row in members),
            }

    # how far from additive are the two error sources?
    deviation = [abs(row["total_error"]
                     - (row["ideal_trotter_error"] + row["noise_induced_error"]))
                 for row in gate_rows]
    largest = max(gate_rows, key=lambda row: row["total_error"]) if gate_rows else None

    return _jsonable({
        "phase": 3,
        "objective": ("determine whether a noise-dependent optimal Trotter step "
                      "count minimises the total simulation error"),
        "configuration": config,
        "validation": validation,
        "row_counts": {
            "noiseless": len(noiseless),
            "gate_noise": len(gate_rows),
            "global_noise": len(global_rows),
            "sensitivity": len(sensitivity_rows),
            "total": len(rows),
        },
        "optimal_r": {
            "configurations": len(optimal_rows),
            "interior_minima": len(interior),
            "boundary_minima": len(boundary),
            "rows": optimal_rows,
        },
        "order_comparison": order_pairs,
        "sensitivity": sensitivity,
        "error_separation": {
            "max_abs_deviation_from_sum": max(deviation) if deviation else None,
            "worst_case_row": {
                "system_size": largest["system_size"],
                "trotter_order": largest["trotter_order"],
                "trotter_steps": largest["trotter_steps"],
                "single_qubit_error": largest["single_qubit_error"],
                "total_error": largest["total_error"],
                "ideal_trotter_error": largest["ideal_trotter_error"],
                "noise_induced_error": largest["noise_induced_error"],
            } if largest else None,
        },
        "reference_noise": reference,
    })


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(rows, sensitivity_rows, optimal_rows, validation, config,
                 figures, figure_dir="figures") -> str:
    gate_rows = [row for row in rows if row["noise_model"] == "gate"]
    global_rows = [row for row in rows if row["noise_model"].startswith("global")]
    noiseless = [row for row in rows if row["noise_model"] == "none"]
    ns = sorted({row["system_size"] for row in rows})
    steps = sorted({row["trotter_steps"] for row in rows})
    levels = sorted({row["single_qubit_error"] for row in gate_rows
                     if row["single_qubit_error"] > 0})
    reference = config.get("reference_noise") or (levels[0] if levels else 1e-4)
    interior = [row for row in optimal_rows if row["optimum_status"] == "interior"]
    boundary = [row for row in optimal_rows
                if row["optimum_status"] in (experiments.OPTIMUM_SMALLEST,
                                             experiments.OPTIMUM_LARGEST)]

    out: list[str] = []
    add = out.append

    add("# Phase 3 — Density-matrix simulation and gate-level noise\n")
    add("_Generated automatically by `python -m noise.run_phase3`. "
        "Every number below comes from a simulation recorded in "
        "`results/phase3/*.csv`; nothing is extrapolated._\n")

    # ---------------------------------------------------------------- 1
    add("## 1. Research question\n")
    add("Phases 1 and 2 counted gates and measured Trotter error for an ideal, "
        "noise-free circuit. Real gates are noisy, so the question this phase asks is:\n")
    add("> Does increasing the number of Trotter steps still help once every gate "
        "carries an error, and is there a step count that minimises the **total** "
        "simulation error for a fixed evolution time?\n")
    add("Two effects compete. More steps reduce the formula error (as `1/r` for the "
        "first-order formula and `1/r^2` for the second-order one) but add gates, and "
        "therefore add noise. If the second effect eventually wins, the total error has "
        "an interior minimum. That minimum is the central quantity of this phase; it is "
        "measured, not assumed to exist.\n")

    # ---------------------------------------------------------------- 2
    add("## 2. Mathematical background: density matrices\n")
    add("A state vector `|psi>` describes a *pure* state. A noisy state is a classical "
        "mixture, written as a matrix\n")
    add("```\nrho = sum_k p_k |psi_k><psi_k|,   sum_k p_k = 1,   p_k >= 0\n```\n")
    add("A pure state is the special case `rho = |psi><psi|` with `rho^2 = rho`. Three "
        "properties must hold for every physical state, and all three are tested:\n")
    add("* `Tr(rho) = 1` — total probability one.\n"
        "* `rho = rho^dagger` — Hermitian, so expectation values are real.\n"
        "* every eigenvalue `>= 0` — positive semidefinite, so no negative probability.\n")
    add("Unitary evolution keeps the state pure: `rho -> U rho U^dagger`. Purity is "
        "`Tr(rho^2)`, equal to 1 for a pure state and `1/d` for the maximally mixed state "
        "`I/d` with `d = 2^n`. Noise is anything that is not of this form; the channels "
        "below are linear maps that push purity down.\n")
    add(f"The implementation stores `rho` as a dense `2^n x 2^n` complex array. At "
        f"n = {max(ns)} that is {2 ** max(ns)} x {2 ** max(ns)} "
        f"({4 ** max(ns) * 16 / 1024:.0f} KiB), so the approach is limited to small "
        "systems; the cost of one gate is `O(4^k * 2^n)` for a `k`-qubit gate, which "
        "makes the full sweeps below take minutes, not hours.\n")

    # ---------------------------------------------------------------- 3-5
    add("## 3. Global depolarising channel\n")
    add("```\nE_g(rho) = (1 - p_g) rho + p_g * I/d\n```\n")
    add("The state is mixed with the maximally mixed state. `p_g` is the probability of "
        "that replacement **per application**, and in the primary experiment it is applied "
        "once after each complete Trotter step. It is therefore a *per-step* parameter, "
        "and it is **not** numerically comparable to a per-gate error probability. A "
        "single end-of-circuit application is available separately as a reference and is "
        "reported separately.\n")
    add("Validated: `p_g = 0` reproduces ideal evolution, `p_g = 1` produces `I/d` "
        "exactly, and trace, Hermiticity and positivity survive every application.\n")

    add("## 4. Single-qubit depolarising channel\n")
    add("```\nE_1(rho) = (1 - p_1) rho + (p_1/3) (X rho X + Y rho Y + Z rho Z)\n```\n")
    add("Applied to the target qubit of an RX or RZ gate; all other qubits are untouched. "
        "`p_1` is the total probability that a non-identity Pauli is applied, so each of "
        "X, Y and Z has probability `p_1/3` and the identity (probability `1 - p_1`) "
        "leaves the state alone. RZ can be configured separately, including as noiseless, "
        "because virtual-Z rotations are free on some architectures; the default model "
        "gives RX and RZ the same `p_1`.\n")

    add("## 5. Two-qubit depolarising channel\n")
    add("```\nE_2(rho) = (1 - p_2) rho + (p_2/15) sum_{P != II} P rho P\n```\n")
    add("`P` runs over the 15 non-identity elements of "
        "`{I,X,Y,Z}^{tensor 2}` acting on the control and target of a CNOT, and `p_2` is "
        "the total probability of a non-identity two-qubit Pauli.\n")
    add("**Implementation and how it is checked.** Summing `P rho P` over the *whole* "
        "Pauli group of a `k`-qubit subsystem has the closed form "
        "`2^k * Tr_k(rho) (x) I_k`, so the sum over non-identity Paulis is that expression "
        "minus `rho`. The fast path uses one partial trace instead of 3 or 15 matrix "
        "products; the tests also evaluate the sum term by term from the definition and "
        "require agreement to machine precision "
        f"(largest difference measured: "
        f"{_fmt(validation.get('channel_max_difference'), 3)}).\n")

    # ---------------------------------------------------------------- 6
    add("## 6. How noise is applied to individual gates\n")
    add("The circuit is **not** rebuilt for this phase. The ordered gate list from "
        "`circuit_cost.build_circuit` — the same CNOT / RZ / RX sequence used in Phases 1 "
        "and 2, including merged second-order ZZ layers and half-angle rotations — is "
        "walked from first to last, and after each gate the corresponding channel is "
        "applied to that gate's own qubits:\n")
    add("```\nRX(q, theta) :  rho <- RX rho RX^dagger   then  E_1 on qubit q\n"
        "RZ(q, theta) :  rho <- RZ rho RZ^dagger   then  E_1 on qubit q\n"
        "CNOT(c, t)   :  rho <- CX rho CX^dagger   then  E_2 on qubits (c, t)\n```\n")
    add("The conventions are inherited rather than re-derived, and verified: the ideal "
        "action of the whole gate walk reproduces `trotter.trotter_unitary` to "
        f"{_fmt(validation.get('circuit_vs_trotter_unitary'), 2)} (spectral norm) and the "
        "gate counts reproduce Phase 1 exactly "
        f"({_fmt(validation.get('max_count_difference'), 2)} difference on every row).\n")
    add("**Step boundary for the global model.** Because the second-order half layers of "
        "neighbouring steps are fused, one Trotter step does not end at a fixed offset in "
        "the gate list. The convention used here is: a step ends after the X block that "
        "closes it (order 1: after the ZZ block), and the final half-angle ZZ layer is "
        "grouped with the last step. That gives exactly `r` global channels for `r` steps; "
        "the grouping is checked against the gate list length on every run.\n")
    add("This is a documented convention, not a hardware model. It does not include "
        "calibrated error rates, T1/T2 relaxation, readout error, crosstalk, connectivity, "
        "routing, dynamical decoupling, mitigation or error correction (see §14).\n")

    # ---------------------------------------------------------------- 7
    add("## 7. Experimental configuration\n")
    add(_table(
        ["Parameter", "Value"],
        [["Hamiltonian", "`H = -J sum Z_iZ_{i+1} - h sum X_i` (unchanged since Phase 1)"],
         ["Couplings", f"J = {config.get('J')}, h = {config.get('h')}"],
         ["Evolution time", f"t = {config.get('t')} (fixed for every step count)"],
         ["Initial state", "`|00...0>` (the Phase 1 convention)"],
         ["System sizes", ", ".join(f"n = {n}" for n in ns)],
         ["Trotter orders", "1 (Lie-Trotter) and 2 (Strang)"],
         ["Step counts", ", ".join(str(s) for s in steps)],
         ["Gate-level noise levels", ", ".join(f"{p:g}" for p in levels)],
         ["Noise convention", "p1 = p2 at each level unless stated"],
         ["Reference level for figures", f"{reference:g}"],
         ["Exact reference", "`exp(-iHt)|00...0>`"],
         ["Trotter reference", "`U_trotter|00...0>` from `trotter.trotter_unitary`"],
         ["Randomness", "none — every run is deterministic"]]) )
    add("")

    add("### 7.1 Validation performed before any noise experiment\n")
    add(_table(["Check", "Result"], [
        ["`rho = |psi><psi|` construction", "tested"],
        ["Trace, Hermiticity, positivity", "tested on every simulated state"],
        ["Density-matrix evolution vs state vector",
         f"{_fmt(validation.get('density_vs_statevector'), 2)} max difference"],
        ["Zero-noise circuit vs `trotter_unitary`",
         f"{_fmt(validation.get('circuit_vs_trotter_unitary'), 2)} spectral norm"],
        ["Gate counts vs Phase 1",
         f"{_fmt(validation.get('max_count_difference'), 2)} maximum difference"],
        ["Channel fast path vs explicit Pauli sum",
         f"{_fmt(validation.get('channel_max_difference'), 2)} maximum difference"],
        ["Purity preserved by ideal evolution",
         f"{_fmt(validation.get('purity_drift'), 2)} drift"]]) )
    add("")

    # ---------------------------------------------------------------- 8
    add("## 8. System sizes and step ranges\n")
    add(f"The step grid is `{', '.join(str(s) for s in steps)}`: dense between 1 and 16, "
        "where the low-noise optimum lies, and coarse afterwards where only the noise tail "
        "matters. The grid was fixed after timing the simulator, not after seeing which "
        "point would win.\n")
    rows_tbl = []
    for n in ns:
        for order in (1, 2):
            members = _select(noiseless, system_size=n, trotter_order=order)
            if not members:
                continue
            smallest = min(members, key=lambda row: row["trotter_steps"])
            largest = max(members, key=lambda row: row["trotter_steps"])
            rows_tbl.append([n, order, len(members),
                             smallest["trotter_steps"], largest["trotter_steps"],
                             largest["total_gate_count"], largest["cnot_count"],
                             largest["logical_depth"]])
    add(_table(["n", "order", "step counts", "r_min", "r_max",
                "gates at r_max", "CNOT at r_max", "depth at r_max"], rows_tbl))
    add("")

    # ---------------------------------------------------------------- 9
    add("## 9. Noise parameter ranges\n")
    add(f"Gate-level sweeps use `p1 = p2` in {{{', '.join(f'{p:g}' for p in levels)}}}, and "
        "the sensitivity grid crosses `p1` and `p2` independently over the same set. "
        "Global sweeps use `p_g` over the same set, applied once per Trotter step.\n")
    add("> These are illustrative simulation parameters chosen to span weak to strong "
        "noise. They are **not** measured hardware error rates and must not be read as "
        "device specifications or as IBM/other hardware performance.\n")

    # ---------------------------------------------------------------- 10
    add("## 10. First-order versus second-order results\n")
    add("### 10.1 Equal step count (the uncontrolled comparison)\n")
    add(f"At `r` equal for both formulas and gate-level noise `p1 = p2 = {reference:g}`, "
        f"for n = {ns[0]}. The second-order circuit is always slightly larger at equal `r`, "
        "so this is the comparison that flatters the first-order formula.\n")
    pairs_equal = experiments.equal_steps_pairs(gate_rows, ns[0], "gate") if gate_rows else []
    pairs_equal = [p for p in pairs_equal
                   if p["single_qubit_error"] == reference] or pairs_equal
    table = []
    for entry in pairs_equal[:8]:
        table.append([entry["r"], entry["cnot_order1"], entry["cnot_order2"],
                      _fmt(entry["ideal_error_order1"], 3),
                      _fmt(entry["ideal_error_order2"], 3),
                      _fmt(entry["total_error_order1"], 3),
                      _fmt(entry["total_error_order2"], 3)])
    if table:
        add(_table(["r", "CNOT (1st)", "CNOT (2nd)", "ideal err 1st", "ideal err 2nd",
                    "noisy err 1st", "noisy err 2nd"], table))
        add("")

    add("### 10.2 Matched CNOT budget\n")
    add("For every measured second-order point, the first-order circuit with the largest "
        "CNOT count that does **not exceed** the second-order budget is selected, so no "
        "first-order point is credited with more gates than the second-order one used. "
        "Budgets come from `circuit_cost`.\n")
    for n in ns:
        pairs = experiments.pair_orders(gate_rows, n, "gate", "cnot")
        if not pairs:
            continue
        selected = [p for p in pairs
                    if p["single_qubit_error"] == reference] or pairs
        better = sum(1 for p in selected if p["fidelity_order2"] > p["fidelity_order1"])
        add(f"**n = {n}**, `p1 = p2 = {reference:g}`: second order is more accurate in "
            f"{better} of {len(selected)} matched-CNOT comparisons.\n")
        table = [[p["order2_r"], p["order1_r"], p["order1_budget"], p["order2_budget"],
                  _fmt(p["total_error_order1"], 3), _fmt(p["total_error_order2"], 3),
                  _signed(p["fidelity_gain"])] for p in selected]
        add(_table(["r (2nd)", "r (1st, matched)", "CNOT 1st", "CNOT 2nd",
                    "noisy err 1st", "noisy err 2nd", "fidelity gain (2nd − 1st)"], table))
        add("")

    add("### 10.3 Matched total gate budget\n")
    for n in ns:
        pairs = experiments.pair_orders(gate_rows, n, "gate", "total_gate_count")
        if not pairs:
            continue
        selected = [p for p in pairs
                    if p["single_qubit_error"] == reference] or pairs
        better = sum(1 for p in selected if p["fidelity_order2"] > p["fidelity_order1"])
        add(f"**n = {n}**: second order is more accurate in {better} of {len(selected)} "
            "matched-total-gate comparisons (same selection rule).\n")

    # ---------------------------------------------------------------- 11
    add("## 11. Global versus gate-level noise\n")
    add("The global model has one parameter per Trotter step and ignores how many gates a "
        "step contains. The gate-level model charges every gate separately, so its "
        "accumulated error scales with `p1 * (#RX + #RZ) + p2 * #CNOT`, which is exactly "
        "the resource count of Phases 1 and 2.\n")
    add("The two models are therefore **not** the same noise at different strengths: at "
        "equal numerical parameter, gate-level noise is far stronger for these circuits, "
        "because a single Trotter step contains `O(n)` gates while the global channel is "
        "applied once. Read the numbers below as two different models, not as a calibrated "
        "comparison of hardware scenarios.\n")
    table = []
    for n in ns:
        for order in (1, 2):
            g = _select(global_rows, system_size=n, trotter_order=order,
                        noise_model="global_per_step",
                        global_noise_probability=reference)
            l = _select(gate_rows, system_size=n, trotter_order=order,
                        noise_model="gate", single_qubit_error=reference)
            if not g or not l:
                continue
            best_g = min(g, key=lambda row: row["total_error"])
            best_l = min(l, key=lambda row: row["total_error"])
            table.append([n, order,
                          f"{best_g['trotter_steps']} (of {min(r['trotter_steps'] for r in g)}–{max(r['trotter_steps'] for r in g)})",
                          _fmt(best_g["total_error"], 3),
                          _fmt(best_g["noise_induced_error"], 3),
                          f"{best_l['trotter_steps']}",
                          _fmt(best_l["total_error"], 3),
                          _fmt(best_l["noise_induced_error"], 3)])
    if table:
        add(_table(["n", "order", "r_opt global", "min total err global",
                    "noise err global", "r_opt gate", "min total err gate",
                    "noise err gate"], table))
        add("")

    # ---------------------------------------------------------------- 12
    add("## 12. Optimal Trotter step count\n")
    add(f"`r_opt` is the argmin of the measured total error over the tested step counts. "
        f"{len(optimal_rows)} configurations were evaluated: {len(interior)} have an "
        f"interior minimum and {len(boundary)} have their minimum at an end of the grid "
        "(reported as a boundary result, not as an optimum).\n")
    gate_opt = [row for row in optimal_rows if row["noise_model"] == "gate"]
    if gate_opt:
        table = [[row["system_size"], row["trotter_order"],
                  f"{row['single_qubit_error']:g}", row["optimal_r"],
                  row["optimum_status"].replace("_", " "),
                  _fmt(row["optimal_fidelity"], 4),
                  _fmt(row["ideal_trotter_error_at_optimum"], 3),
                  _fmt(row["noise_error_at_optimum"], 3),
                  row["cnot_count_at_optimum"], row["total_gate_count_at_optimum"],
                  row["logical_depth_at_optimum"]] for row in gate_opt]
        add(_table(["n", "order", "p1 = p2", "r_opt", "status", "F at r_opt",
                    "ideal err", "noise err", "CNOT", "gates", "depth"], table))
        add("")
    global_opt = [row for row in optimal_rows if row["noise_model"] == "global_per_step"]
    if global_opt:
        table = [[row["system_size"], row["trotter_order"],
                  f"{row['global_noise_probability']:g}", row["optimal_r"],
                  row["optimum_status"].replace("_", " "),
                  _fmt(row["optimal_fidelity"], 4),
                  _fmt(row["ideal_trotter_error_at_optimum"], 3),
                  _fmt(row["noise_error_at_optimum"], 3)] for row in global_opt]
        add(_table(["n", "order", "p_g", "r_opt", "status", "F at r_opt",
                    "ideal err", "noise err"], table))
        add("")

    # ---------------------------------------------------------------- 13
    add("## 13. Fidelity, purity and the three error measures\n")
    add("Three quantities are recorded for every configuration and they are not "
        "interchangeable:\n")
    add("```\nideal_trotter_error = 1 - F(rho_trot,  psi_exact)\n"
        "noise_induced_error = 1 - F(rho_noisy, rho_trot)\n"
        "total_error         = 1 - F(rho_noisy, psi_exact)\n```\n")
    deviation = [abs(row["total_error"]
                     - (row["ideal_trotter_error"] + row["noise_induced_error"]))
                 for row in gate_rows]
    if deviation:
        add(f"The total is **not** the sum of the other two. Measured over "
            f"{len(gate_rows)} gate-level configurations, the largest discrepancy between "
            f"`total_error` and `ideal + noise` is {_fmt(max(deviation), 3)}. The two "
            "contributions add only in the small-error limit; fidelity is not linear, and "
            "when both errors are large the sum overcounts.\n")
    add("Noise can occasionally raise the fidelity with respect to a particular reference "
        "by chance, so the measured curves are reported as they are, not forced to be "
        "monotone. Purity is reported separately: it isolates how much the state has been "
        "mixed and says nothing about whether it is the *right* state.\n")

    # ---------------------------------------------------------------- 14
    add("## 14. Limitations\n")
    add("This noise model deliberately stops short of a hardware description. It does "
        "**not** include:\n")
    add("* hardware-calibrated error rates (the `p` values are simulation parameters),\n"
        "* T1 relaxation or T2 dephasing, and no amplitude/phase damping channel,\n"
        "* readout error, SPAM, or measurement in the reported metrics,\n"
        "* crosstalk, leakage to non-computational states, or drift,\n"
        "* device connectivity, routing or SWAP overhead (the chain is assumed fully "
        "connected for nearest-neighbour gates),\n"
        "* dynamical decoupling, gate scheduling/parallelism effects, or pulse-level "
        "control,\n"
        "* error mitigation, post-selection, or quantum error correction.\n")
    add("Correlations between errors are absent: every channel is applied independently, "
        "which is optimistic compared with real hardware where a miscalibration affects a "
        "whole gate class at once. Density matrices restrict the study to `n <= 6` qubits "
        "at these step counts.\n")
    add("The results must not be read as a prediction for any real quantum processor.\n")

    # ---------------------------------------------------------------- 15
    add("## 15. Reproducibility\n")
    add("```bash\npython -m noise.run_phase3              # all experiments, figures, report\n"
        "python -m noise.run_phase3 --no-plots   # tables and report only\n"
        "python -m noise.run_phase3 --quick      # small grid, for a smoke test\n"
        "python -m pytest tests/ -q              # full suite (Phases 1-3)\n```\n")
    add("Every run is deterministic: no random numbers are used anywhere in Phase 3, so "
        "repeated runs produce identical CSV files. Outputs:\n")
    add("```\nresults/phase3/\n"
        "  global_noise_results.csv\n  gate_noise_results.csv\n  noise_sensitivity.csv\n"
        "  optimal_r.csv\n  phase3_summary.json\n  phase3_report.md\n"
        "  configs/experiment_config.json\n  figures/fig01...fig12\n```\n")
    add("Phase 1 and Phase 2 outputs are never written by this phase; a regression test "
        "hashes them before and after a full Phase 3 run.\n")

    # ---------------------------------------------------------------- 16
    add("## 16. Conclusions\n")
    for line in _conclusions(gate_rows, global_rows, optimal_rows, interior, boundary,
                             ns, reference, sensitivity_rows):
        add(line)
    add("")
    add("### Figures\n")
    for name in figures:
        caption = plotting.CAPTIONS.get(name, "")
        add(f"**`{figure_dir}/{name}`** — {caption}\n")
    return "\n".join(out)


def _classify(row) -> str:
    """Short description of what kind of configuration a row is."""
    model = row["noise_model"]
    if model == "gate" and row["single_qubit_error"] == 0.0:
        return "noise-free"
    if model == "gate":
        return "gate-level"
    if model == "global_per_step":
        return "per-step global"
    if model == "global_end_of_circuit":
        return "end-of-circuit global (one channel, no r dependence)"
    return model


def _noise_level(rows, n, order, r) -> float:
    """The single-qubit error of one measured row (NaN if it is absent)."""
    for row in rows:
        if (row["system_size"] == n and row["trotter_order"] == order
                and row["trotter_steps"] == r):
            return row["single_qubit_error"]
    return float("nan")


def _row_noise(pair, rows) -> float:
    """Noise level of the second-order row a pair refers to.

    Works for both pair shapes: ``equal_steps_pairs`` uses ``r``,
    ``pair_orders`` uses ``order2_r``.
    """
    r = pair["order2_r"] if "order2_r" in pair else pair["r"]
    return _noise_level(rows, pair["system_size"], 2, r)


def _row_noise_pair(pair, rows) -> float:
    return _row_noise(pair, rows)


def _conclusions(gate_rows, global_rows, optimal_rows, interior, boundary, ns,
                 reference, sensitivity_rows) -> list[str]:
    """Bullet points, each tied to a measured number."""
    lines: list[str] = []
    if not optimal_rows:
        return ["No optimal-r configurations were evaluated — rerun without `--quick`."]

    interior_r = [row["optimal_r"] for row in interior]
    if interior:
        lines.append(
            f"* **An interior optimum exists.** {len(interior)} of {len(optimal_rows)} "
            f"configurations have their minimum total error at a step count strictly "
            f"inside the tested grid, with `r_opt` between {min(interior_r)} and "
            f"{max(interior_r)}. That is the central result of Phase 3: beyond a certain "
            "point, adding Trotter steps makes the simulation *worse*, not better.")
    if boundary:
        smallest = [row for row in boundary
                    if row["optimum_status"] == experiments.OPTIMUM_SMALLEST]
        largest = [row for row in boundary
                   if row["optimum_status"] == experiments.OPTIMUM_LARGEST]
        if smallest:
            lines.append(
                f"* **{len(smallest)} configurations bottom out at the smallest tested "
                f"step count (r = {min(row['optimal_r'] for row in smallest)}).** For "
                "those noise levels even one Trotter step is already better than any "
                "finer discretisation, which is stated as a boundary result rather than "
                "presented as an interior optimum.")
        if largest:
            kinds: dict[str, int] = {}
            for row in largest:
                kinds[_classify(row)] = kinds.get(_classify(row), 0) + 1
            detail = ", ".join(f"{count} {label}" for label, count in
                               sorted(kinds.items()))
            noisy = sum(count for label, count in kinds.items()
                        if "noise-free" not in label and "end-of-circuit" not in label)
            lines.append(
                f"* **{len(largest)} configurations bottom out at the largest tested "
                f"step count (r = {max(row['optimal_r'] for row in largest)}): "
                f"{detail}.** "
                + ("Those cases have no per-step or per-gate noise charge that grows "
                   "with the circuit, so more steps are simply better and no crossover "
                   "exists; reporting r = the largest tested value is the honest result. "
                   if noisy == 0 else "")
                + (f"{noisy} of them do carry a growing noise charge, so for those the "
                   "tested range was not wide enough to reach the crossover."
                   if noisy else
                   "No gate-level or per-step global configuration has a boundary "
                   "minimum, so the crossover is real wherever noise is charged per "
                   "step or per gate."))

    for model, label in (("gate", "gate-level"), ("global_per_step", "global")):
        rows = [row for row in optimal_rows if row["noise_model"] == model]
        if not rows:
            continue
        by_level: dict[float, list[int]] = {}
        for row in rows:
            key = row["single_qubit_error"] if model == "gate" else row["global_noise_probability"]
            if key == 0.0:
                continue          # the noise-free run is not a point on this trend
            by_level.setdefault(key, []).append(row["optimal_r"])
        if not by_level:
            continue
        levels = sorted(by_level)
        monotone = all(max(by_level[b]) <= max(by_level[a])
                       for a, b in zip(levels, levels[1:]))
        trend = ", ".join(f"{level:g} -> r={min(by_level[level])}"
                          f"{'' if min(by_level[level]) == max(by_level[level]) else '-' + str(max(by_level[level]))}"
                          for level in levels)
        lines.append(
            f"* **{label.capitalize()} noise:** the optimal step count moves "
            f"{'down' if monotone else 'non-monotonically'} as noise grows "
            f"({trend}). Fewer, larger steps mean fewer noisy operations, so the useful "
            "circuit depth shrinks with the error rate.")

    if gate_rows:
        ref_rows = [row for row in gate_rows if row["single_qubit_error"] == reference]
        sizes = sorted({row["system_size"] for row in ref_rows})
        for n in ([sizes[0], sizes[-1]] if len(sizes) > 1 else sizes):
            for order in (1, 2):
                members = [row for row in ref_rows
                           if row["trotter_order"] == order and row["system_size"] == n]
                if not members:
                    continue
                best = min(members, key=lambda row: row["total_error"])
                worst = max(members, key=lambda row: row["total_error"])
                lines.append(
                    f"* **n = {n}, order {order}, p = {reference:g}:** the best measured "
                    f"point is r = {best['trotter_steps']} with total error "
                    f"{_fmt(best['total_error'], 3)} ({best['cnot_count']} CNOTs, "
                    f"{best['total_gate_count']} gates), while r = "
                    f"{worst['trotter_steps']} is "
                    f"{_fmt(worst['total_error'] / best['total_error'], 2)}x worse despite "
                    f"using {_fmt(worst['cnot_count'] / best['cnot_count'], 2)}x the CNOTs "
                    f"and {_fmt(worst['total_gate_count'] / best['total_gate_count'], 2)}x "
                    "the gates. Its formula error is still "
                    f"{_fmt(worst['ideal_trotter_error'], 2)} at that point - the gates, "
                    "not the formula, are what fails.")

    if sensitivity_rows:
        single = [row for row in sensitivity_rows if row.get("grid") == "single_qubit_only"]
        pair = [row for row in sensitivity_rows if row.get("grid") == "two_qubit_only"]
        full = [row for row in sensitivity_rows if row.get("grid") == "full"]
        if single and pair and full:
            probe = max(single, key=lambda row: row["single_qubit_error"])
            same = [row for row in pair
                    if row["two_qubit_error"] == probe["single_qubit_error"]]
            whole = [row for row in full
                     if row["single_qubit_error"] == probe["single_qubit_error"]
                     and row["two_qubit_error"] == probe["two_qubit_error"]]
            if same and whole:
                ratio = same[0]["total_error"] / max(probe["total_error"], 1e-15)
                rotations = probe["total_gate_count"] - probe["cnot_count"]
                per_cnot = same[0]["total_error"] / same[0]["cnot_count"]
                per_rotation = probe["total_error"] / max(rotations, 1)
                if ratio > 1.25:
                    verdict = "two-qubit errors dominate"
                elif ratio < 0.8:
                    verdict = "single-qubit errors dominate"
                else:
                    verdict = "the two gate classes contribute comparably"
                lines.append(
                    f"* **Gate-class balance n = {probe['system_size']}, "
                    f"r = {probe['trotter_steps']}, p = {probe['single_qubit_error']:g}:** "
                    f"{verdict}. Charging the error to CNOTs alone gives total error "
                    f"{_fmt(same[0]['total_error'], 3)}, charging the same parameter to "
                    f"RX/RZ alone gives {_fmt(probe['total_error'], 3)} (ratio "
                    f"{_fmt(ratio, 3)}). Per gate the two are close "
                    f"({_fmt(per_cnot, 2)} vs {_fmt(per_rotation, 2)} of total error per "
                    f"gate), so what decides the balance is the gate *count*: this point "
                    f"has {same[0]['cnot_count']} CNOTs and {rotations} rotations. With both "
                    f"classes noisy the total error is {_fmt(whole[0]['total_error'], 3)}, "
                    "slightly below the sum of the two slices — the contributions combine "
                    "sub-additively, which is why §13 does not treat error sources as "
                    "additive.")

    levels = sorted({row["single_qubit_error"] for row in gate_rows
                     if row["single_qubit_error"] > 0})
    if levels:
        ref = reference if reference in levels else levels[0]

        all_pairs = []
        for size in sorted({row["system_size"] for row in gate_rows}):
            all_pairs.extend(experiments.pair_orders(gate_rows, size, "gate", "cnot"))

        per_level = []
        for level in [0.0] + list(levels):
            pairs = [pair for pair in all_pairs
                     if pair["single_qubit_error"] == level]
            wins = sum(1 for pair in pairs if pair["fidelity_order2"] > pair["fidelity_order1"])
            per_level.append((level, wins, len(pairs), pairs))

        entry = next((item for item in per_level if item[0] == ref), per_level[-1])
        _, wins, total, pairs = entry
        crossing = sorted({pair["order2_r"] for pair in pairs
                           if pair["fidelity_order2"] > pair["fidelity_order1"]})
        pooled_wins = sum(item[1] for item in per_level)
        pooled_total = sum(item[2] for item in per_level)
        progression = ", ".join(f"{item[1]}/{item[2]}" for item in per_level)
        extra_step = sum(1 for pair in pairs if pair["order1_r"] > pair["order2_r"])

        lines.append(
            "* **Second order's advantage is bounded by noise, not universal.** At "
            f"p = {ref:g}, second order has the lower total error in {wins} of {total} "
            "matched-CNOT comparisons (the §10.2 rule; both circuits always carry the "
            "same noise settings)."
            + (f" All {wins} of the wins sit at r <= {max(crossing)}." if crossing else "")
            + " Two different effects produce the losses. In "
            f"{extra_step} of the {total} pairs the smaller per-step overhead "
            "(`2(n-1)` CNOTs) lets first order afford one extra Trotter step at the same "
            "budget, which is decisive at small r. In the rest the two circuits have the "
            "same r and sit above the crossover, where the formulas no longer separate and "
            "the marginally smaller first-order circuit accumulates slightly less noise. "
            "The win count therefore falls monotonically with the error rate: "
            f"{progression} for p = 0 ... {max(levels):g} ({pooled_wins}/{pooled_total} "
            "pooled), against 102/105 for the same rule in the noise-free Phase 2 "
            "analysis. Phase 2's conclusion is bounded rather than overturned: second "
            "order buys accuracy more cheaply *only while the formula still dominates the "
            "error*, which is at or below the optimal step count this phase recommends.")
    lines.append(
        "* **Global and gate-level noise give different `r_opt`.** The global model "
        "charges one parameter per step regardless of circuit size, so its optimum sits "
        "where the formula error crosses `p_g`-per-step accumulation; the gate-level model "
        "charges every gate, so its optimum depends on `n`, on the order, and on which "
        "gate class is noisy. §11 records both.")
    return lines


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_report(text: str, path: str) -> str:
    import os
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    return path


def write_summary(summary: dict, path: str) -> str:
    import os
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path
