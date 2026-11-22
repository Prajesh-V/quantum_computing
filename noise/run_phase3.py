#!/usr/bin/env python3
"""Phase 3 driver: density-matrix simulation under gate-level depolarising noise.

Runs, in order:

1. **Validation** - the density-matrix engine is checked against the unchanged
   state-vector simulator and against ``trotter.trotter_unitary``, the channels
   are checked against the explicit Pauli sum, and the gate counts against
   Phase 1's closed forms.  Nothing else runs until these pass.
2. **Sweeps** - gate-level noise for every (n, order, r, noise level), global
   noise per Trotter step, and one end-of-circuit global reference.
3. **Sensitivity** - the (p1, p2) grid plus the two single-parameter slices.
4. **Optimal r** - the measured argmin of the total error per configuration.
5. **Outputs** - CSVs, figures, ``phase3_summary.json`` and ``phase3_report.md``.

Usage
-----
    python -m noise.run_phase3                 # full study
    python -m noise.run_phase3 --quick         # small grid, smoke test
    python -m noise.run_phase3 --no-plots      # tables and report only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

import circuit_cost
import trotter

from . import channels, experiments, plotting, report
from .density_matrix import (apply_operator, basis_density_matrix, purity,
                             pure_density_matrix, check_density_matrix)
from .metrics import fidelity_pure
from .noisy_simulator import (NoiseModel, exact_state, ideal_trotter_state,
                              local_matrix, reset_cache, simulate, step_ranges)

OUTPUT_DIR = os.path.join("results", "phase3")


# ---------------------------------------------------------------------------
# Stage 1 - validation
# ---------------------------------------------------------------------------

def validate(t: float, J: float, h: float) -> dict:
    """Checks that must pass before any noise experiment is reported."""
    print("Stage 1 - validation")
    checks: dict[str, float] = {}

    # (a) density-matrix ideal walk vs the existing Trotter propagator
    worst = 0.0
    for n, r, order in ((4, 1, 1), (4, 3, 1), (4, 3, 2), (5, 2, 2), (6, 2, 1)):
        out = simulate(n, r, order, t, J, h, NoiseModel(kind="gate"))
        target = pure_density_matrix(ideal_trotter_state(n, r, order, t, J, h))
        worst = max(worst, float(np.abs(out["rho"] - target).max()))
    checks["density_vs_statevector"] = worst
    print(f"  density matrix vs state vector / Trotter unitary : {worst:.3e}")

    # (b) the circuit's ideal action vs trotter_unitary, in spectral norm
    worst = 0.0
    for n in (4, 5):
        A, B, _ = trotter.tfim_terms(n, J, h)
        for order in (1, 2):
            for r in (1, 2, 5):
                gates = circuit_cost.build_circuit(n, r, t, J, h, order)
                U_ref = trotter.trotter_unitary(A, B, t, r, order)
                U_circ = circuit_cost.circuit_unitary(n, gates)
                worst = max(worst, float(np.linalg.norm(U_circ - U_ref, 2)))
    checks["circuit_vs_trotter_unitary"] = worst
    print(f"  gate circuit vs trotter_unitary (spectral norm)   : {worst:.3e}")

    # (c) local gate embeddings vs circuit_cost.gate_matrix
    worst = 0.0
    for gate in circuit_cost.build_circuit(4, 2, t, J, h, 2):
        embedded = (circuit_cost._embed_two(4, *gate.qubits, local_matrix(gate))
                    if gate.gate_type == "cnot"
                    else trotter.single_qubit(4, gate.qubits[0], local_matrix(gate)))
        worst = max(worst, float(np.abs(embedded - circuit_cost.gate_matrix(4, gate)).max()))
    checks["local_matrix_embedding"] = worst
    print(f"  local gate matrices vs circuit_cost.gate_matrix   : {worst:.3e}")

    # (d) channels: fast partial-trace path vs the explicit Pauli sum
    rng = np.random.default_rng(20261122)
    worst = 0.0
    for n, qubits, p in ((3, (0,), 0.07), (4, (2,), 0.4), (4, (0, 3), 0.07),
                         (5, (1, 2), 0.5)):
        d = 2 ** n
        matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        state = matrix @ matrix.conj().T
        state /= np.trace(state)
        fast = channels.depolarize_qubits(state, qubits, p)
        slow = channels.explicit_pauli_depolarizing(state, qubits, p)
        worst = max(worst, float(np.abs(fast - slow).max()))
    checks["channel_max_difference"] = worst
    print(f"  channel fast path vs explicit Pauli sum           : {worst:.3e}")

    # (e) gate counts against Phase 1 closed forms, on the whole step grid
    worst_counts = 0.0
    for n in (4, 5, 6):
        for order in (1, 2):
            for r in experiments.DEFAULT_STEPS:
                res = circuit_cost.estimate_resources(n, r, t, J, h, order)
                ref = circuit_cost.analytic_counts(n, r, order)
                worst_counts = max(worst_counts, float(abs(res.num_cnot - ref["cnot"])),
                                   float(abs(res.total_gates - ref["total"])))
    checks["max_count_difference"] = worst_counts
    print(f"  compiled gate counts vs Phase 1 closed forms      : {worst_counts:.0f}")

    # (f) purity and positivity of ideal evolution
    drift = 0.0
    for n in (4, 6):
        for r in (1, 5, 32):
            for order in (1, 2):
                out = simulate(n, r, order, t, J, h, NoiseModel(kind="gate"))
                drift = max(drift, float(abs(purity(out["rho"]) - 1.0)))
                info = check_density_matrix(out["rho"])
                if not info["ok"]:
                    raise SystemExit(f"ideal state at n={n}, r={r} is not a valid density matrix")
    checks["purity_drift"] = drift
    print(f"  purity drift under ideal evolution                : {drift:.3e}")

    # (g) channel end points
    dim = 2 ** 4
    mixed = channels.apply_global_depolarizing(basis_density_matrix(4, [0] * 4), 1.0)
    checks["global_p1_to_maximally_mixed"] = float(
        np.abs(mixed - np.eye(dim) / dim).max())
    print(f"  global p_g = 1 vs maximally mixed I/d             : "
          f"{checks['global_p1_to_maximally_mixed']:.3e}")

    # (h) every *noisy* state is still a valid density matrix
    worst_trace = 0.0
    worst_negative = 0.0
    for n in (4, 6):
        for order in (1, 2):
            for p in (1e-3, 1e-2):
                for kind in ("gate", "global_per_step"):
                    model = NoiseModel(kind=kind, single_qubit_error=p,
                                       two_qubit_error=p, global_probability=p)
                    info = check_density_matrix(
                        simulate(n, 32, order, t, J, h, model)["rho"])
                    worst_trace = max(worst_trace, info["trace_error"])
                    worst_negative = max(worst_negative, -info["min_eigenvalue"])
    checks["noisy_trace_error"] = worst_trace
    checks["noisy_negativity"] = worst_negative
    print(f"  noisy states: max |Tr-1|, max negative eigenvalue: "
          f"{worst_trace:.2e}, {worst_negative:.2e}")

    # (i) zero noise reproduces the ideal circuit exactly
    out_none = simulate(4, 4, 2, t, J, h, NoiseModel(kind="none"))
    out_zero = simulate(4, 4, 2, t, J, h,
                        NoiseModel(kind="gate", single_qubit_error=0.0,
                                   two_qubit_error=0.0))
    checks["zero_noise_vs_ideal"] = float(np.abs(out_none["rho"] - out_zero["rho"]).max())
    print(f"  p1 = p2 = 0 vs no channel at all                  : "
          f"{checks['zero_noise_vs_ideal']:.3e}")

    tolerance_ok = (checks["density_vs_statevector"] < 1e-10
                    and checks["circuit_vs_trotter_unitary"] < 1e-10
                    and checks["local_matrix_embedding"] < 1e-12
                    and checks["channel_max_difference"] < 1e-12
                    and checks["purity_drift"] < 1e-10
                    and checks["noisy_trace_error"] < 1e-10
                    and checks["noisy_negativity"] < 1e-10)
    if not tolerance_ok:
        raise SystemExit("density-matrix / channel validation failed - not proceeding")
    checks["all_passed"] = True
    print("  -> all checks pass; safe to proceed\n")
    return checks


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return value
    return f"{float(value):.15g}"


def write_rows(rows, columns, path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_value(row[column]) for column in columns])
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(args, steps, levels) -> dict:
    reference = levels[len(levels) // 2] if levels else 1e-3
    return {
        "phase": 3,
        "model": "transverse-field Ising chain, open boundary",
        "hamiltonian": "H = -J sum Z_iZ_{i+1} - h sum X_i",
        "J": float(args.J),
        "h": float(args.h),
        "t": float(args.t),
        "initial_state": "0" * 4 + " (|00...0>, the Phase 1 convention)",
        "system_sizes": list(args.ns),
        "trotter_orders": [1, 2],
        "trotter_steps": list(steps),
        "gate_noise_levels": list(levels),
        "global_noise_levels": list(levels),
        "noise_convention": ("p_1 and p_2 are total non-identity Pauli "
                             "probabilities per gate; p_g is applied once per "
                             "Trotter step and is not comparable to p_1 or p_2"),
        "rz_noise": "same as RX (p_1); rz_noiseless=False",
        "reference_noise": reference,
        "sensitivity_point": {"system_size": 4, "trotter_steps": 16,
                              "trotter_order": 2},
        "deterministic": True,
        "random_seed": None,
        "note": ("noise parameters are illustrative simulation parameters, "
                 "not measured hardware error rates"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t", type=float, default=1.0, help="evolution time")
    parser.add_argument("--J", type=float, default=1.0, help="Ising coupling")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field")
    parser.add_argument("--ns", default="4,5,6", help="system sizes, e.g. 4,5,6")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    parser.add_argument("--quick", action="store_true",
                        help="small grid for a smoke test")
    parser.add_argument("--levels", default=None,
                        help="comma-separated noise levels (default: %s)"
                             % ",".join(f"{p:g}" for p in experiments.NOISE_LEVELS))
    parser.add_argument("--steps", default=None,
                        help="comma-separated Trotter step counts")
    args = parser.parse_args(argv)

    args.ns = [int(part) for part in args.ns.split(",") if part.strip()]
    steps = (tuple(int(part) for part in args.steps.split(",") if part.strip())
             if args.steps else
             ((1, 2, 4, 8, 16) if args.quick else experiments.DEFAULT_STEPS))
    if args.levels:
        levels = tuple(float(part) for part in args.levels.split(",") if part.strip())
    else:
        levels = ((1e-4, 1e-3) if args.quick
                  else tuple(experiments.NOISE_LEVELS))
    if args.quick and args.steps is None:
        args.ns = args.ns[:1]

    t, J, h = float(args.t), float(args.J), float(args.h)
    outdir = args.outdir
    figure_dir = os.path.join(outdir, "figures")
    config_dir = os.path.join(outdir, "configs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    started = time.perf_counter()
    reset_cache()
    validation = validate(t, J, h)

    config = build_config(args, steps, levels)
    with open(os.path.join(config_dir, "experiment_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(f"  configuration written to {config_dir}/experiment_config.json\n")

    # -- Stage 2: sweeps ---------------------------------------------------
    print("Stage 2 - noisy sweeps")
    models = [NoiseModel(kind="gate")]                      # p1 = p2 = 0: ideal
    models += [NoiseModel(kind="gate", single_qubit_error=p, two_qubit_error=p)
               for p in levels]
    models += [NoiseModel(kind="global_per_step", global_probability=p) for p in levels]
    reference = config["reference_noise"]
    models += [NoiseModel(kind="global_end_of_circuit", global_probability=reference)]

    rows: list[dict] = []
    orders = (1, 2)
    for model in models:
        print(f"  {model.label():<40s} (n={args.ns}, r<={max(steps)})", flush=True)
        rows.extend(experiments.sweep(args.ns, orders, steps, [model], t, J, h))
    gate_rows = [row for row in rows if row["noise_model"] == "gate"]
    global_rows = [row for row in rows if row["noise_model"].startswith("global")]
    print(f"  -> {len(rows)} configurations "
          f"({len(gate_rows)} gate-level, {len(global_rows)} global)\n")

    # -- Stage 3: sensitivity ---------------------------------------------
    print("Stage 3 - noise sensitivity grid")
    point = config["sensitivity_point"]
    sensitivity_rows = []
    for n in [point["system_size"]] + ([args.ns[-1]] if args.ns[-1] != point["system_size"] else []):
        sensitivity_rows.extend(experiments.noise_sensitivity(
            n, point["trotter_steps"], point["trotter_order"], t,
            p1_values=levels, p2_values=levels))
    print(f"  -> {len(sensitivity_rows)} configurations at r={point['trotter_steps']}\n")

    # -- Stage 4: optimal r ------------------------------------------------
    print("Stage 4 - optimal step count")
    optimal_rows = experiments.optimal_step_count(rows, r_values=steps)
    interior = [row for row in optimal_rows if row["optimum_status"] == "interior"]
    boundary = [row for row in optimal_rows
                if row["optimum_status"] in (experiments.OPTIMUM_SMALLEST,
                                             experiments.OPTIMUM_LARGEST)]
    print(f"  -> {len(optimal_rows)} configurations: {len(interior)} interior minima, "
          f"{len(boundary)} at a tested boundary\n")

    # -- Stage 5: outputs --------------------------------------------------
    print("Stage 5 - outputs")
    paths = {
        "global": write_rows(global_rows, experiments.GLOBAL_COLUMNS,
                             os.path.join(outdir, "global_noise_results.csv")),
        "gate": write_rows(gate_rows, experiments.GATE_COLUMNS,
                           os.path.join(outdir, "gate_noise_results.csv")),
        "sensitivity": write_rows(sensitivity_rows, experiments.SENSITIVITY_COLUMNS,
                                  os.path.join(outdir, "noise_sensitivity.csv")),
        "optimal": write_rows(optimal_rows, experiments.OPTIMAL_COLUMNS,
                              os.path.join(outdir, "optimal_r.csv")),
    }
    for name, path in paths.items():
        print(f"  wrote {path}")

    figures: list[str] = []
    if not args.no_plots:
        figures = plotting.make_all_figures(
            rows, sensitivity_rows, optimal_rows, figure_dir,
            reference_noise=reference)
        print(f"  wrote {len(figures)} figures to {figure_dir}")

    summary = report.build_summary(rows, sensitivity_rows, optimal_rows,
                                   validation, config)
    summary["figures"] = [os.path.basename(path) for path in figures]
    report.write_summary(summary, os.path.join(outdir, "phase3_summary.json"))
    print(f"  wrote {outdir}/phase3_summary.json")

    text = report.build_report(rows, sensitivity_rows, optimal_rows, validation,
                               config, [os.path.basename(path) for path in figures],
                               figure_dir="figures")
    report.write_report(text, os.path.join(outdir, "phase3_report.md"))
    print(f"  wrote {outdir}/phase3_report.md")

    # -- console summary ---------------------------------------------------
    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("Phase 3 summary")
    print("=" * 78)
    print(f"  {'n':>2} {'ord':>3} {'model':>10} {'p':>8} {'r_opt':>6} {'status':>18} "
          f"{'F':>8} {'total err':>10} {'CNOT':>6}")
    for row in optimal_rows:
        parameter = (row["single_qubit_error"] if row["noise_model"] == "gate"
                     else row["global_noise_probability"])
        print(f"  {row['system_size']:>2} {row['trotter_order']:>3} "
              f"{row['noise_model'][:10]:>10} {parameter:>8.4g} {row['optimal_r']:>6} "
              f"{row['optimum_status']:>18} {row['optimal_fidelity']:>8.4f} "
              f"{row['minimum_total_error']:>10.3e} {row['cnot_count_at_optimum']:>6}")
    print(f"\n  total run time {elapsed:.1f} s")
    print(f"  outputs in {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
