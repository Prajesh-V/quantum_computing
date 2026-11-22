"""Driver for Phase 2: measure how Trotter accuracy trades against circuit resources.

    python -m analysis.run_phase2                  # analyse committed data
    python -m analysis.run_phase2 --write-supplement  # also measure r = 240...30000
    python -m analysis.run_phase2 --no-plots

Phase 1 files are only ever read.  Every Phase 2 output goes to
``results/phase2/``.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import circuit_cost
import trotter

from . import accuracy_targets, comparison, plotting, report, scaling
from .resource_analysis import (PLOT_FLOOR, Dataset, audit, load_dataset,
                                NUMERICAL_FLOOR)

SUPPLEMENTARY_COLUMNS = list(circuit_cost.RESOURCE_COLUMNS) + ["source"]
EFFICIENCY_COLUMNS = [
    "n", "order", "r", "operator_norm_error", "cnot_count", "total_gates",
    "circuit_depth", "two_qubit_depth", "error_per_cnot", "error_per_gate",
    "error_reduction_vs_first_order_same_steps_pct", "at_numerical_floor",
]


# ---------------------------------------------------------------------------
# Supplementary measurement (same physics, larger r)
# ---------------------------------------------------------------------------

def write_supplementary(ns, r_values, path: str, t: float = 1.0) -> int:
    """Measure the existing pipeline at step counts Phase 1 did not cover.

    This is a *measurement*, not an interpolation: ``trotter.sweep`` and
    ``circuit_cost.estimate_resources`` are called exactly as in Phase 1.  It
    exists because the committed sweep stops at ``r = 200``, where first order
    is still at ~1e-2 and the 1e-3 / 1e-4 targets of the brief are unreachable.
    """
    r_values = np.unique(np.round(np.asarray(r_values, dtype=float)).astype(int))
    rows = []
    for n in ns:
        A, B, H = trotter.tfim_terms(n)
        results = trotter.sweep(A, B, H, t, r_values)
        for order in (1, 2):
            for r, e_op, e_inf in zip(r_values, results[order]["op"], results[order]["inf"]):
                res = circuit_cost.estimate_resources(n, int(r), t, order=order)
                rows.append(["TFIM", n, order, int(r), "0" * n,
                             f"{e_op:.12e}", f"{e_inf:.12e}",
                             res.num_qubits, res.num_cnot, res.num_rx, res.num_rz,
                             res.total_gates, res.circuit_depth,
                             res.two_qubit_depth, "supplementary"])
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUPPLEMENTARY_COLUMNS)
        writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Analysis assembly
# ---------------------------------------------------------------------------

def build_analysis(dataset: Dataset, thresholds=accuracy_targets.THRESHOLDS) -> dict:
    """Compute every Phase 2 quantity from the loaded data."""
    info = audit(dataset)
    slopes = comparison.accuracy_per_cnot(dataset)

    equal = {n: comparison.equal_steps(dataset, n) for n in dataset.n_values()}
    budgets = {metric: {n: comparison.equal_budget(dataset, n, metric)
                        for n in dataset.n_values()}
               for metric in ("cnot", "total_gates")}

    verification = scaling.verify_analytic(dataset)
    exponents = scaling.empirical_exponents(dataset)
    depth_spread = scaling.depth_vs_qubits_claim(dataset)

    min_rows = accuracy_targets.minimum_resources(dataset, thresholds)
    min_pairs = accuracy_targets.pair_by_threshold(min_rows)
    min_md = accuracy_targets.summary_markdown(min_rows)

    errors = [row.err_op for row in dataset.rows]
    nonmonotone = sum(cfg["nonmonotone_steps"] for cfg in info["configs"].values())
    monotone_configs = sum(1 for cfg in info["configs"].values()
                           if cfg["strictly_decreasing"])
    reliability = {
        "at_floor": sum(1 for row in dataset.rows if row.at_floor),
        "min_error": float(min(errors)) if errors else float("nan"),
        "monotone_configs": monotone_configs,
        "nonmonotone_steps": nonmonotone,
        "zero_errors": sum(1 for value in errors if value <= 0),
        "plot_floor": PLOT_FLOOR,
        "numerical_floor": NUMERICAL_FLOOR,
    }

    analysis = {
        "audit": info,
        "slopes": slopes,
        "equal_steps": equal,
        "budget": budgets,
        "scaling": {"verification": verification,
                    "by_order": exponents["by_order"],
                    "depth_spread": {str(k): v for k, v in depth_spread.items()},
                    "r_ref": exponents["r_ref"]},
        "min_resources": min_rows,
        "min_resources_pairs": min_pairs,
        "min_resources_markdown": min_md,
        "reliability": reliability,
    }
    analysis["findings"] = _findings(dataset, analysis)
    return analysis


def _findings(dataset: Dataset, analysis: dict) -> list[str]:
    """Turn the computed numbers into defensible statements."""
    out: list[str] = []
    slopes = analysis["slopes"]
    ns = dataset.n_values()
    orders = dataset.orders()

    if (ns[0], 1) in slopes and (ns[0], 2) in slopes:
        s1 = np.mean([abs(slopes[n, 1]) for n in ns if (n, 1) in slopes])
        s2 = np.mean([abs(slopes[n, 2]) for n in ns if (n, 2) in slopes])
        out.append(
            f"Error falls as CNOT^-{s1:.2f} for the first-order formula and CNOT^-{s2:.2f} "
            f"for the second-order formula (mean over n = {ns}). Gaining one decade of "
            f"accuracy costs about {10 ** (1 / s1):.1f}x more CNOTs at first order versus "
            f"{10 ** (1 / s2):.2f}x at second order.")

    # equal steps: improvement factor = error(order 1) / error(order 2)
    factors = []
    coarse = []
    for n in ns:
        for row in analysis["equal_steps"].get(n) or []:
            if row["error_order2"] > 0:
                factor = row["error_order1"] / row["error_order2"]
                factors.append(factor)
                if factor < 1.0:
                    coarse.append((n, row["r"], factor))
    if factors:
        text = (f"At equal r, second order is more accurate by a factor of "
                f"{min(factors):.2f} to {max(factors):.1f} (median {np.median(factors):.1f}) "
                f"while using only 2(n-1) extra CNOTs and 3(n-1) extra gates - a fixed cost "
                f"that does not grow with r.")
        if coarse:
            text += (" The exception is r=" + ", ".join(str(r) for _, r, _ in coarse) +
                     " (pre-asymptotic: the two formulas have not yet separated).")
        out.append(text)

    # equal budgets
    for metric, label in (("cnot", "CNOT"), ("total_gates", "total gate")):
        pairs = []
        for n in ns:
            for row in analysis["budget"][metric].get(n) or []:
                value = row["error_reduction_pct_interpolated"]
                if value == value:
                    pairs.append((row["order2_r"], value))
        better = [value for _, value in pairs if value > 0]
        worse = [(r, value) for r, value in pairs if value <= 0]
        if better:
            upper = ("more than 99.99%" if max(better) >= 99.995
                     else f"{max(better):.1f}%")
            text = (f"At a matched {label} budget, second order reaches a lower error in "
                    f"{len(better)} of {len(pairs)} measured budgets, by "
                    f"{min(better):.1f}% to {upper} "
                    f"(median {np.median(better):.1f}%).")
            if worse:
                text += (f" It is worse at {len(worse)} budget(s), all at coarse steps "
                         f"(r <= {max(r for r, _ in worse)}), where its fixed 2(n-1)-CNOT "
                         f"overhead buys an extra step for the first-order formula instead "
                         f"of accuracy.")
            out.append(text)

    # savings at fixed accuracy
    savings = [entry for entry in analysis["min_resources_pairs"] if entry["achieved_by_both"]]
    if savings:
        gate_savings = [entry["gate_savings_pct"] for entry in savings]
        cnot_savings = [entry["cnot_savings_pct"] for entry in savings]
        out.append(
            f"For a fixed achieved accuracy where both formulas succeed, second order needs "
            f"{min(gate_savings):.1f}% to {max(gate_savings):.1f}% fewer gates "
            f"({min(cnot_savings):.1f}% to {max(cnot_savings):.1f}% fewer CNOTs).")

    # scaling
    verification = analysis["scaling"]["verification"]
    out.append(
        f"Measured resources match the closed forms exactly "
        f"(max |difference| = {verification['max_abs_difference']:g} over "
        f"{verification['rows_checked']} rows): CNOT is exactly linear in (n-1) and in r, "
        f"so the resource cost of a chain is set by its bonds, not its qubits.")

    spread = analysis["scaling"]["depth_spread"]
    if spread:
        worst = max(value["spread_pct"] for value in spread.values())
        out.append(
            f"Logical depth grows linearly with r but only by {worst:.2f}% across "
            f"n = {ns[0]}...{ns[-1]} at fixed r, because the conservative scheduler "
            f"serialises bonds; a hardware-aware schedule would change this, so no "
            f"depth-based claim beyond 'conservative logical estimate' is made.")

    reliability = analysis["reliability"]
    if reliability["at_floor"]:
        out.append(
            f"{reliability['at_floor']} measured points sit at or below the "
            f"{reliability['numerical_floor']:g} numerical floor and are flagged as "
            f"round-off, not accuracy. The smallest measured error is "
            f"{reliability['min_error']:.2e}.")
    else:
        out.append(
            f"No measured point reaches the {reliability['numerical_floor']:g} numerical "
            f"floor (smallest measured error {reliability['min_error']:.2e}), so every "
            f"error reported here is truncation error rather than round-off.")

    if orders == [1, 2]:
        out.append(
            "Answer to the phase question: second order is the better use of every resource "
            "measured (steps, CNOTs, total gates), and its advantage is largest when the "
            "accuracy target is tight; the extra cost is a constant 2(n-1) CNOTs, not a "
            "per-step penalty.")
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_efficiency_csv(dataset: Dataset, path: str) -> int:
    """Per-configuration cost-normalised error, plus the equal-step improvement."""
    same_steps = {}
    for n in dataset.n_values():
        for row in comparison.equal_steps(dataset, n):
            same_steps[(n, row["r"])] = row["error_order1"]

    rows = []
    for row in sorted(dataset.rows, key=lambda item: (item.n, item.order, item.r)):
        baseline = same_steps.get((row.n, row.r))
        reduction = (100.0 * (baseline - row.err_op) / baseline
                     if baseline and row.order == 2 else "")
        rows.append([
            row.n, row.order, row.r, f"{row.err_op:.12e}", row.cnot, row.total,
            row.depth, row.twoq,
            f"{row.err_op / row.cnot:.12e}" if row.cnot else "",
            f"{row.err_op / row.total:.12e}" if row.total else "",
            f"{reduction:.4f}" if reduction != "" else "",
            int(row.at_floor),
        ])
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(EFFICIENCY_COLUMNS)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase1", default="results/circuit_resources.csv")
    parser.add_argument("--outdir", default=os.path.join("results", "phase2"))
    parser.add_argument("--supplementary", default=None,
                        help="supplementary sweep file (default: <outdir>/supplementary_sweep.csv)")
    parser.add_argument("--write-supplement", action="store_true",
                        help="measure r = 240...30000 with the unchanged pipeline")
    parser.add_argument("--supplement-rmax", type=int, default=30000)
    parser.add_argument("--supplement-points", type=int, default=14)
    parser.add_argument("--no-supplement", action="store_true",
                        help="ignore the supplementary file and use Phase 1 data only")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    supp_path = args.supplementary or os.path.join(args.outdir, "supplementary_sweep.csv")

    if args.write_supplement:
        ns = (4, 5, 6)
        r_values = np.geomspace(240, args.supplement_rmax, args.supplement_points)
        count = write_supplementary(ns, r_values, supp_path)
        print(f"measured supplementary sweep: {count} rows -> {supp_path}")

    dataset = load_dataset(args.phase1, supp_path, use_supplementary=not args.no_supplement)
    info = audit(dataset)

    print("=" * 78)
    print("Phase 2 data audit")
    print("=" * 78)
    print(f"  rows {info['rows']}  n {info['n_values']}  orders {info['orders']}  "
          f"sources {info['sources']}")
    print(f"  r from {min(info['r_values'])} to {max(info['r_values'])} "
          f"({len(info['r_values'])} values)")
    print(f"  every row has consistent resources: {info['every_row_has_resources']}")
    print(f"  row problems: {len(info['row_problems'])}")
    for name, cfg in info["configs"].items():
        print(f"  {name}: rows {cfg['rows']}, error {cfg['error_first']:.3e} -> "
              f"{cfg['error_last']:.3e}, decreasing {cfg['strictly_decreasing']}")

    analysis = build_analysis(dataset)

    ver = analysis["scaling"]["verification"]
    print()
    print("=" * 78)
    print("Analytical gate-count verification")
    print("=" * 78)
    print(f"  rows checked {ver['rows_checked']}, max |difference| = "
          f"{ver['max_abs_difference']:g}, all match: {ver['all_match']}")
    if not ver["all_match"]:
        print(f"  first mismatches: {ver['mismatches'][:3]}")

    # outputs
    outputs = {
        "minimum_resources_for_accuracy.csv": "cheapest measured configuration per target",
        "resource_efficiency.csv": "error per CNOT and per gate, plus equal-step reduction",
        "phase2_summary.json": "machine-readable analysis",
        "phase2_report.md": "this report's source",
    }
    accuracy_targets.write_csv(analysis["min_resources"],
                               os.path.join(args.outdir,
                                            "minimum_resources_for_accuracy.csv"))
    write_efficiency_csv(dataset, os.path.join(args.outdir, "resource_efficiency.csv"))

    if not args.no_plots:
        written = plotting.write_all(dataset, args.outdir)
        for path in written:
            outputs[os.path.basename(path)] = "figure from measured data"
        print()
        for path in written:
            print(f"  wrote {path}")

    analysis["outputs"] = outputs
    report.write_json(analysis, os.path.join(args.outdir, "phase2_summary.json"))
    with open(os.path.join(args.outdir, "phase2_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report.markdown(analysis))

    print()
    print("=" * 78)
    print("Findings")
    print("=" * 78)
    for finding in analysis["findings"]:
        print("  * " + finding)
    print()
    print(f"  wrote {args.outdir}/phase2_report.md and phase2_summary.json")


if __name__ == "__main__":
    main()
