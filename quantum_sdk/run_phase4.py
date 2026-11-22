"""Driver for the Phase 4 SDK study (Stages 7-13 of the brief).

    python -m quantum_sdk.run_phase4

Requires the audit to have run first (``python -m validation.run_audit``),
because the Phase 4 report and summary merge both halves.  Writes only under
``results/phase4/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy
import qiskit

from validation import io_utils

from . import experiments, noise_check, plotting, report, resources, transpilation

__all__ = ["main"]

PHASE3_COMMIT = "9e47be8"


def _stage(number: int, title: str) -> float:
    print(f"\n=== Stage {number}: {title} ===")
    return time.time()


def _done(started: float, note: str = "") -> None:
    print(f"    completed in {time.time() - started:.1f}s {note}".rstrip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/phase4")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="one system size and two step counts")
    args = parser.parse_args(argv)

    sdk_dir = f"{args.outdir}/sdk"
    hardware_dir = f"{args.outdir}/hardware_aware"
    figure_dir = f"{args.outdir}/figures"
    audit_summary_path = f"{args.outdir}/audit/validation_summary.json"

    if not os.path.exists(audit_summary_path):
        print(f"missing {audit_summary_path}: run `python -m validation.run_audit` first")
        return 2
    with open(audit_summary_path, encoding="utf-8") as handle:
        audit_summary = json.load(handle)
    print(f"loaded audit summary from {audit_summary_path}")

    ns, rs = experiments.NS, experiments.RS
    if args.quick:
        ns, rs = (4,), (2, 4)

    started = _stage(8, "circuit equivalence (custom vs Qiskit)")
    equivalence = experiments.equivalence_study(ns=ns, rs=rs)
    worst = max(row["spectral_distance"] for row in equivalence)
    print(f"    {sum(1 for row in equivalence if row['passed'])}/{len(equivalence)} "
          f"equivalent, worst spectral distance {worst:.2e}")
    _done(started)

    started = _stage(9, "pre-transpilation resources")
    pre = experiments.pre_transpilation_study(ns=ns, rs=rs)
    print(f"    counts match Phase 1 in {sum(1 for row in pre if row['counts_match'])}"
          f"/{len(pre)}, depth matches in {sum(1 for row in pre if row['depth_match'])}"
          f"/{len(pre)}")
    _done(started)

    started = _stage(10, "transpilation")
    transpilation_results = experiments.transpilation_study(ns=ns, rs=rs)
    rows = transpilation_results["rows"]
    for topology in ("ideal", "line", "ring", "star"):
        subset = [row for row in rows if row["topology"] == topology]
        if subset:
            print(f"    {topology:<6} max extra CX "
                  f"{max(row['compiled_cx'] - row['logical_cx'] for row in subset):>3}, "
                  f"max depth inflation "
                  f"{max(row['depth_inflation'] for row in subset):.2f}")
    _done(started)

    started = _stage(11, "compiled resource analysis")
    connectivity = experiments.connectivity_study(ns=ns, rs=rs)
    compiled = experiments.compiled_comparison_study(ns=ns, rs=rs)
    print(f"    {len(connectivity)} connectivity rows, {len(compiled)} comparison rows")
    _done(started)

    started = _stage(12, "SDK noise: channel reconciliation and compiled noise")
    channel = noise_check.channel_convention()
    noise = experiments.noise_study(ns=ns, rs=rs)
    agreement = max(row["max_density_matrix_difference"]
                    for row in noise["sdk_simulation"])
    print(f"    native depolarizing_error differs by "
          f"{channel['native_max_difference']:.2e} at p = 0.2 (different convention); "
          f"project channel reproduced to "
          f"{channel['pauli_mixture_max_difference']:.2e}")
    print(f"    whole-circuit Aer vs custom: max difference {agreement:.2e}")
    penalties = [row["star_penalty"] for row in noise["compiled_noise"]]
    print(f"    star routing penalty: {min(penalties):+.4f} to {max(penalties):+.4f}")
    _done(started)

    started = _stage(13, "write outputs")
    io_utils.write_csv(f"{sdk_dir}/circuit_equivalence.csv",
                       list(equivalence[0].keys()), equivalence)
    io_utils.write_csv(f"{sdk_dir}/pre_transpilation_resources.csv",
                       list(pre[0].keys()), pre)
    io_utils.write_csv(f"{sdk_dir}/post_transpilation_resources.csv",
                       list(rows[0].keys()), rows)
    io_utils.write_json(f"{sdk_dir}/transpilation_metadata.json",
                        transpilation_results["metadata"])
    io_utils.write_csv(f"{sdk_dir}/sdk_simulation_results.csv",
                       list(noise["sdk_simulation"][0].keys()),
                       noise["sdk_simulation"])
    io_utils.write_csv(f"{hardware_dir}/connectivity_comparison.csv",
                       list(connectivity[0].keys()), connectivity)
    io_utils.write_csv(f"{hardware_dir}/compiled_resource_comparison.csv",
                       list(compiled[0].keys()), compiled)
    io_utils.write_csv(f"{hardware_dir}/compiled_noise_results.csv",
                       list(noise["compiled_noise"][0].keys()),
                       noise["compiled_noise"])
    print("    wrote 8 sdk/hardware-aware files")

    figures = []
    if not args.no_figures:
        figures = plotting.make_all(figure_dir, {
            "equivalence": equivalence,
            "transpilation": transpilation_results,
            "connectivity": connectivity,
            "noise": noise,
        })
        print(f"    wrote {len(figures)} SDK figures (fig06-fig12)")
    _done(started)

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "qiskit": qiskit.__version__,
        "aer": __import__("qiskit_aer").__version__,
        "phase3_commit": PHASE3_COMMIT,
    }
    results = {
        "meta": meta,
        "audit": audit_summary,
        "equivalence": equivalence,
        "pre": pre,
        "transpilation": transpilation_results,
        "connectivity": connectivity,
        "compiled": compiled,
        "noise": noise,
        "channel": channel,
        "figures": figures,
    }

    text = report.build(results)
    path = io_utils.ensure_dir(f"{args.outdir}/phase4_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"    wrote {path}")

    summary = dict(audit_summary)
    summary["meta"] = meta
    summary["sdk"] = {
        "qiskit": meta["qiskit"],
        "aer": meta["aer"],
        "equivalence": {
            "configurations": len(equivalence),
            "passed": sum(1 for row in equivalence if row["passed"]),
            "worst_spectral_distance": max(row["spectral_distance"]
                                           for row in equivalence),
            "worst_statevector_infidelity": max(row["statevector_infidelity"]
                                               for row in equivalence),
        },
        "pre_transpilation": {
            "configurations": len(pre),
            "counts_match": sum(1 for row in pre if row["counts_match"]),
            "depth_match": sum(1 for row in pre if row["depth_match"]),
        },
        "transpilation": {
            topology: {
                "max_extra_cx": max(row["compiled_cx"] - row["logical_cx"]
                                    for row in rows if row["topology"] == topology),
                "max_two_qubit_inflation": max(row["two_qubit_inflation"]
                                               for row in rows if row["topology"] == topology),
                "max_depth_inflation": max(row["depth_inflation"]
                                           for row in rows if row["topology"] == topology),
            }
            for topology in ("ideal", "line", "ring", "star")
            if any(row["topology"] == topology for row in rows)
        },
        "noise": {
            "channel": channel,
            "worst_aer_vs_custom": max(row["max_density_matrix_difference"]
                                       for row in noise["sdk_simulation"]),
            "star_penalty_min": min(row["star_penalty"] for row in noise["compiled_noise"]),
            "star_penalty_max": max(row["star_penalty"] for row in noise["compiled_noise"]),
            "line_penalty_max": max(abs(row["line_penalty"])
                                    for row in noise["compiled_noise"]),
            "compiled_noise": noise["compiled_noise"],
            "sdk_simulation": noise["sdk_simulation"],
        },
        "connectivity": connectivity,
        "compiled_resources": compiled,
        "figures": figures,
        "hardware_execution": False,
    }
    io_utils.write_json(f"{args.outdir}/phase4_summary.json", summary)
    print(f"    wrote {args.outdir}/phase4_summary.json")
    _done(started)

    print("\n=== phase 4 SDK study complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
