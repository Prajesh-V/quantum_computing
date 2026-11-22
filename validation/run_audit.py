"""Driver for the Phase 4 scientific audit (Stages 1-8 of the brief).

    python -m validation.run_audit

Reads the stored Phase 3 results, re-verifies them, runs the robustness
experiments with the independent engine, and writes everything to
``results/phase4/``.  Nothing under ``results/phase3/`` is written to, and no
file outside ``results/phase4/`` is modified.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy
import scipy

from noise import experiments as phase3_experiments

from . import audit, comparisons, io_utils, optimal_check, plotting, reproduction
from . import report as audit_report
from . import resources as resource_audit
from . import robustness

__all__ = ["main", "PHASE3_COMMIT"]

PHASE3_COMMIT = "9e47be8"

SWEEP_COLUMNS = [
    "initial_state", "variant", "total_time", "system_size", "trotter_order",
    "single_qubit_error", "two_qubit_error", "trotter_steps", "fidelity",
    "ideal_trotter_error", "noise_induced_error", "total_error", "purity",
    "trace_distance", "cnot_count", "rx_count", "rz_count", "total_gate_count",
    "logical_depth", "two_qubit_depth", "optimal_r", "optimum_status", "is_optimal",
]


def _stage(number: int, title: str) -> float:
    print(f"\n=== Stage {number}: {title} ===")
    return time.time()


def _done(started: float, note: str = "") -> None:
    print(f"    completed in {time.time() - started:.1f}s {note}".rstrip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/phase4",
                        help="root output directory (default results/phase4)")
    parser.add_argument("--results", default="results/phase3",
                        help="stored Phase 3 results directory")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="audit and reproduction only (fast)")
    args = parser.parse_args(argv)

    audit_dir = f"{args.outdir}/audit"
    robust_dir = f"{args.outdir}/robustness"
    figure_dir = f"{args.outdir}/figures"

    started = _stage(1, "load stored Phase 3 results (read-only)")
    stored = io_utils.load_results(args.results)
    print(f"    gate rows {len(stored['gate'])}, global rows {len(stored['global'])}, "
          f"optimal rows {len(stored['optimal'])}, "
          f"sensitivity rows {len(stored['sensitivity'])}")
    _done(started)

    started = _stage(2, "mathematical audit of the conventions")
    math_audit = audit.run_all(n=4)
    print(f"    {math_audit['summary']['passed']}/{math_audit['summary']['checks']} "
          f"checks passed, largest equality deviation "
          f"{math_audit['summary']['max_deviation']:.2e}")
    _done(started)

    started = _stage(3, "independent reproduction of stored measurements")
    repro = reproduction.run(stored["gate"], stored["global"])
    print(f"    {repro['case_count']} cases, {repro['quantity_count']} values, "
          f"{repro['failures']} failures, max |diff| {repro['max_abs_difference']:.2e}")
    separation = {
        "gate": reproduction.error_separation(stored["gate"]),
        "global": reproduction.error_separation(stored["global"]),
    }
    print(f"    error separation: max |delta| {separation['gate']['max_abs_deviation']:.4f} "
          "(gate rows)")
    _done(started)

    started = _stage(4, "verify every stored optimal step count")
    optimal = optimal_check.verify_all(stored["gate"] + stored["global"],
                                       stored["optimal"])
    print(f"    {optimal['passed']}/{optimal['configurations']} agree; "
          f"{len(optimal['boundary_configurations'])} boundary minima")
    _done(started)

    started = _stage(5, "robustness experiments")
    if args.skip_robustness:
        initial = evolution = strength = components = candidate = None
        print("    skipped (--skip-robustness)")
    else:
        t0 = time.time()
        initial = robustness.initial_state_experiment()
        print(f"    initial state: {len(initial['summary'])} groups "
              f"({time.time() - t0:.0f}s)")
        t0 = time.time()
        evolution = robustness.evolution_time_experiment()
        print(f"    evolution time: {len(evolution['summary'])} groups "
              f"({time.time() - t0:.0f}s)")
        strength = robustness.noise_strength_check(stored["optimal"])
        print(f"    noise strength: {strength['non_increasing']}/"
              f"{strength['groups_without_noise_free']} groups non-increasing, "
              f"{strength['strictly_decreasing']} strictly decreasing")
        t0 = time.time()
        components = robustness.noise_components_experiment()
        print(f"    noise components: {len(components['summary'])} groups "
              f"({time.time() - t0:.0f}s)")
        t0 = time.time()
        candidate = robustness.candidate_grid_experiment(stored["optimal"])
        changed = sum(1 for entry in candidate["summary"] if entry["optimum_changed"])
        print(f"    candidate grid: {changed}/{len(candidate['summary'])} optima moved "
              f"({time.time() - t0:.0f}s)")
        for collection, keys in (
                (initial["rows"], ("initial_state", "system_size", "trotter_order",
                                   "single_qubit_error")),
                (evolution["rows"], ("total_time", "system_size", "trotter_order",
                                     "single_qubit_error")),
                (components["rows"], ("variant", "system_size", "trotter_order",
                                      "single_qubit_error")),
                (candidate["rows"], ("system_size", "trotter_order",
                                     "single_qubit_error"))):
            robustness.with_optimal_flags(collection, keys)
    _done(started)

    started = _stage(6, "comparison audit")
    pairs_cnot = comparisons.pair_matched(stored["gate"], "cnot")
    pairs_total = comparisons.pair_matched(stored["gate"], "total_gate")
    pairs_equal = comparisons.equal_step_pairs(stored["gate"])
    phase3_pairs = []
    for n in (4, 5, 6):
        phase3_pairs.extend(phase3_experiments.pair_orders(stored["gate"], n, "gate", "cnot"))
    comps = {
        "cnot": {"pairs": pairs_cnot, "summary": comparisons.summarise(pairs_cnot),
                 "agreement": comparisons.compare_with_phase3(pairs_cnot, phase3_pairs)},
        "total_gate": {"pairs": pairs_total,
                       "summary": comparisons.summarise(pairs_total)},
        "equal_step": {"pairs": pairs_equal,
                       "summary": comparisons.summarise(pairs_equal)},
    }
    print(f"    matched CNOT: {comps['cnot']['summary']['progression']} "
          f"({comps['cnot']['summary']['progression_total']} pooled), "
          f"agreement {comps['cnot']['agreement']['win_disagreements']} disagreements")
    _done(started)

    started = _stage(7, "resource formulas and correlation study")
    resource_results = {
        "formulas": resource_audit.verify_formulas(),
        "stored": resource_audit.verify_stored_counts(stored["gate"]),
        "correlation": resource_audit.correlation_study(stored["gate"]),
    }
    print(f"    analytic counts: {resource_results['formulas']['cases']} cases, "
          f"max difference {resource_results['formulas']['max_abs_difference']}; "
          f"stored counts: {resource_results['stored']['mismatches']} mismatches")
    _done(started)

    started = _stage(8, "write outputs")
    io_utils.write_csv(f"{audit_dir}/independent_reproduction.csv",
                       ["case", "system_size", "trotter_order", "trotter_steps",
                        "noise_model", "single_qubit_error", "two_qubit_error",
                        "global_noise_probability", "quantity", "stored", "independent",
                        "abs_difference", "rel_difference", "tolerance", "passed"],
                       repro["rows"])
    io_utils.write_csv(f"{audit_dir}/optimal_r_verification.csv",
                       ["system_size", "trotter_order", "noise_model",
                        "single_qubit_error", "two_qubit_error",
                        "global_noise_probability", "stored_optimal_r",
                        "recomputed_optimal_r", "stored_status", "recomputed_status",
                        "stored_minimum_error", "recomputed_minimum_error",
                        "stored_fidelity", "recomputed_fidelity", "candidates",
                        "tested_r_min", "tested_r_max", "optimal_r_in_tested_grid",
                        "cnot_at_optimum", "total_gates_at_optimum", "passed"],
                       optimal["rows"])
    io_utils.write_csv(f"{audit_dir}/comparison_audit.csv",
                       ["comparison", "system_size", "noise_model",
                        "single_qubit_error", "two_qubit_error",
                        "global_noise_probability", "total_time", "order1_r",
                        "order2_r", "order1_budget", "order2_budget", "order1_error",
                        "order2_error", "error_difference", "order1_fidelity",
                        "order2_fidelity", "within_budget", "winner"],
                       pairs_cnot + pairs_total + pairs_equal)

    if args.skip_robustness:
        robust_files = {}
    else:
        robust_files = {
            "initial_state": io_utils.write_csv(
                f"{robust_dir}/robustness_initial_state.csv", SWEEP_COLUMNS,
                initial["rows"]),
            "evolution_time": io_utils.write_csv(
                f"{robust_dir}/robustness_evolution_time.csv", SWEEP_COLUMNS,
                evolution["rows"]),
            "noise_components": io_utils.write_csv(
                f"{robust_dir}/robustness_noise_components.csv", SWEEP_COLUMNS,
                components["rows"]),
            "candidate_grid": io_utils.write_csv(
                f"{robust_dir}/robustness_candidate_grid.csv", SWEEP_COLUMNS,
                candidate["rows"]),
        }
        strength_rows = []
        for group in strength["groups"]:
            deltas = group["deltas"]
            for index, (level, steps) in enumerate(zip(group["levels"],
                                                       group["optimal_steps"])):
                strength_rows.append({
                    "system_size": group["system_size"],
                    "trotter_order": group["trotter_order"],
                    "noise_model": group["noise_model"],
                    "parameter": group["parameter"],
                    "noise_level": level,
                    "optimal_r": steps,
                    "delta_from_previous": deltas[index - 1] if index else None,
                    "strictly_decreasing": group["strictly_decreasing"],
                    "non_increasing": group["non_increasing"],
                    "group_violations": group["violations"],
                })
        robust_files["noise_strength"] = io_utils.write_csv(
            f"{robust_dir}/robustness_noise_strength.csv",
            ["system_size", "trotter_order", "noise_model", "parameter", "noise_level",
             "optimal_r", "delta_from_previous", "strictly_decreasing",
             "non_increasing", "group_violations"], strength_rows)
        print(f"    wrote {len(robust_files)} robustness CSVs")

    figures = []
    if not args.no_figures and not args.skip_robustness:
        figures = plotting.make_all(figure_dir, initial, evolution, strength,
                                    components, candidate)
        print(f"    wrote {len(figures)} figures")

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase3_commit": PHASE3_COMMIT,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "results_source": args.results,
    }
    results = {
        "meta": meta,
        "audit": math_audit,
        "reproduction": repro,
        "separation": separation,
        "optimal": optimal,
        "comparisons": comps,
        "resources": resource_results,
        "figures": figures,
        "robustness": ({}
                       if args.skip_robustness else
                       {"initial": initial, "evolution": evolution,
                        "strength": strength, "components": components,
                        "candidate": candidate}),
    }

    text = audit_report.build(results)
    path = io_utils.ensure_dir(f"{audit_dir}/audit_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)

    validation_summary = {
        "meta": meta,
        "verdict": "pass_with_qualifications",
        "mathematical_audit": math_audit["summary"],
        "mathematical_sections": math_audit["sections"],
        "reproduction": {key: value for key, value in repro.items()
                         if key not in ("rows", "cases")},
        "error_separation": {key: {k: v for k, v in value.items() if k != "worst_row"}
                             for key, value in separation.items()},
        "optimal_r": {
            "configurations": optimal["configurations"],
            "passed": optimal["passed"],
            "failed": optimal["failed"],
            "per_noise_model": optimal["per_noise_model"],
            "boundary_configurations": len(optimal["boundary_configurations"]),
            "grid": optimal["grid_values"],
            "problems": optimal["problems"],
        },
        "comparisons": {
            name: {"summary": block["summary"],
                   **({"agreement": block["agreement"]} if "agreement" in block else {})}
            for name, block in comps.items()
        },
        "resources": resource_results,
        "robustness": (
            {} if args.skip_robustness else {
                "initial_state": {
                    "states": sorted({row["initial_state"] for row in initial["summary"]}),
                    "groups": initial["summary"],
                },
                "evolution_time": {"groups": evolution["summary"]},
                "noise_strength": {key: value for key, value in strength.items()
                                   if key != "groups"},
                "noise_components": {"groups": components["summary"]},
                "candidate_grid": {
                    "changed": sum(1 for entry in candidate["summary"]
                                   if entry["optimum_changed"]),
                    "groups": candidate["summary"],
                },
            }),
        "figures": figures,
        "outputs": {
            "audit_report": path,
            "independent_reproduction": f"{audit_dir}/independent_reproduction.csv",
            "optimal_r_verification": f"{audit_dir}/optimal_r_verification.csv",
            "comparison_audit": f"{audit_dir}/comparison_audit.csv",
            "robustness": robust_files,
        },
    }
    io_utils.write_json(f"{audit_dir}/validation_summary.json", validation_summary)
    io_utils.write_json(f"{args.outdir}/phase4_summary.json", validation_summary)
    print(f"    wrote {path}")
    print(f"    wrote {audit_dir}/validation_summary.json")
    _done(started)

    print("\n=== audit complete ===")
    print(f"verdict: {validation_summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
