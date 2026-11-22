"""All Phase 2 figures.  Every point is a measured configuration.

Conventions used in every figure:

* colour = number of qubits, line style = formula order;
* filled markers = Phase 1 sweep (``r <= 200``), **hollow markers = the
  supplementary sweep at larger r** (still measured, same propagator);
* depth axes are labelled as the conservative logical estimate from Phase 1's
  scheduler; no hardware timing is implied anywhere.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

from . import comparison  # noqa: E402
from .resource_analysis import Dataset, safe_log  # noqa: E402

COLORS = {4: "#1b4965", 5: "#5fa8d3", 6: "#c1666b", 7: "#6a994e", 8: "#bc6c25"}
MARKERS = {1: "o", 2: "s"}
LINESTYLES = {1: "-", 2: "--"}
ORDER_LABELS = {1: "1st order", 2: "2nd order"}
YLABEL = r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$"


def _color(n: int) -> str:
    return COLORS.get(n, "#333333")


def _finish(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_curve(ax, dataset: Dataset, n: int, order: int, xattr: str,
                label: str | None = None, alpha: float = 1.0) -> None:
    """One measured error curve, with supplementary points as hollow markers."""
    series = dataset.series(n, order)
    if not series:
        return
    x = [getattr(row, xattr) for row in series]
    y = safe_log([row.err_op for row in series])
    ax.loglog(x, y, color=_color(n), ls=LINESTYLES[order], marker=MARKERS[order],
              ms=3.4, lw=1.35, alpha=alpha,
              label=label or f"n={n}, {ORDER_LABELS[order]}")
    extra = [row for row in series if row.source == "supplementary"]
    if extra:
        ax.loglog([getattr(row, xattr) for row in extra],
                  safe_log([row.err_op for row in extra]), ls="none",
                  marker=MARKERS[order], ms=5.5, mfc="none", mec=_color(n),
                  alpha=alpha)


def _note_supplementary(ax, dataset: Dataset) -> None:
    if "supplementary" in dataset.sources():
        ax.annotate("hollow markers: supplementary measurements at r > 200",
                    xy=(0.995, 0.02), xycoords="axes fraction", ha="right",
                    fontsize=7.5, color="0.35",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.85))


def error_vs_cnot(dataset: Dataset, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for n, order in dataset.configs():
        _draw_curve(ax, dataset, n, order, "cnot",
                    label=f"n={n}, {ORDER_LABELS[order]}")
    slopes = comparison.accuracy_per_cnot(dataset)
    text = "\n".join(f"n={n} {ORDER_LABELS[o]}: slope {s:+.2f}"
                     for (n, o), s in slopes.items())
    ax.annotate(text, xy=(0.015, 0.03), xycoords="axes fraction", va="bottom",
                fontsize=7.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))
    ax.set_xlabel("CNOT count")
    ax.set_ylabel(YLABEL)
    ax.set_title("Trotter error against entangling-gate count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    _note_supplementary(ax, dataset)
    _finish(fig, path)


def error_vs_total_gates(dataset: Dataset, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for n, order in dataset.configs():
        _draw_curve(ax, dataset, n, order, "total",
                    label=f"n={n}, {ORDER_LABELS[order]}")
    ax.set_xlabel("total elementary gates (CNOT + RX + RZ)")
    ax.set_ylabel(YLABEL)
    ax.set_title("Trotter error against total gate count\n"
                 "counting convention: CNOT + RX + RZ, no barriers or identities")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    _note_supplementary(ax, dataset)
    _finish(fig, path)


def error_vs_depth(dataset: Dataset, path: str, two_qubit: bool = False) -> None:
    attr = "twoq" if two_qubit else "depth"
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for n, order in dataset.configs():
        _draw_curve(ax, dataset, n, order, attr,
                    label=f"n={n}, {ORDER_LABELS[order]}")
    ax.set_xlabel("estimated logical circuit depth (conservative scheduler)"
                  if not two_qubit else
                  "estimated two-qubit gate depth (conservative scheduler)")
    ax.set_ylabel(YLABEL)
    ax.set_title("Trotter error against "
                 + ("two-qubit depth" if two_qubit else "circuit depth")
                 + "\nlogical estimate, not hardware-calibrated")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    _note_supplementary(ax, dataset)
    _finish(fig, path)


def order_comparison_equal_steps(dataset: Dataset, path: str) -> None:
    ns = dataset.n_values()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for n, order in dataset.configs():
        _draw_curve(ax, dataset, n, order, "r",
                    label=f"n={n}, {ORDER_LABELS[order]}")
    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel(YLABEL)
    ax.set_title("Equal step count: same r, same n, same t")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    _note_supplementary(ax, dataset)

    for n in ns:
        rows = comparison.equal_steps(dataset, n)
        if not rows:
            continue
        rs = [row["r"] for row in rows]
        ax2.semilogx(rs, [row["cnot_overhead_pct"] for row in rows],
                     marker=MARKERS[2], ms=3.4, lw=1.3, color=_color(n),
                     label=f"n={n}: CNOT")
        ax2.semilogx(rs, [row["gate_overhead_pct"] for row in rows],
                     marker=MARKERS[1], ms=3.4, lw=1.3, ls="--", color=_color(n),
                     label=f"n={n}: total gates")
    ax2.axhline(0.0, color="0.4", lw=1.0)
    ax2.set_xlabel("number of Trotter steps  r")
    ax2.set_ylabel("second-order overhead vs first order  [%]")
    ax2.set_title("Fixed cost of the Strang formula")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=7.0, ncol=2)
    fig.suptitle("Formula comparison at equal Trotter step count", fontsize=11)
    _finish(fig, path)


def _budget_panels(dataset: Dataset, path: str, metric: str) -> None:
    ns = dataset.n_values()
    fig, axes = plt.subplots(1, len(ns), figsize=(4.1 * len(ns), 4.4), squeeze=False)
    xattr = "cnot" if metric == "cnot" else "total"
    for ax, n in zip(axes[0], ns):
        _draw_curve(ax, dataset, n, 1, xattr, label="1st order (measured)")
        _draw_curve(ax, dataset, n, 2, xattr, label="2nd order (measured)")
        rows = comparison.equal_budget(dataset, n, metric)
        matched = [row for row in rows if row["error_order1_interpolated"] is not None]
        if matched:
            # vertical connector at selected budgets: how far below the first-order
            # curve the second-order circuit sits at the same budget
            pick = np.unique(np.round(np.geomspace(
                max(matched[0]["budget"], 10.0), matched[-1]["budget"], 5)).astype(int))
            for budget in pick:
                nearest = min(matched, key=lambda row: abs(row["budget"] - budget))
                if not nearest["error_order2"]:
                    continue
                factor = nearest["error_order1_interpolated"] / nearest["error_order2"]
                if factor < 1.05:
                    continue
                middle = float(np.sqrt(nearest["error_order2"]
                                       * nearest["error_order1_interpolated"]))
                ax.plot([nearest["budget"], nearest["budget"]],
                        [nearest["error_order2"], nearest["error_order1_interpolated"]],
                        ls=":", color="0.55", lw=0.9)
                label = f"x{factor:.0f}" if factor >= 10 else f"x{factor:.1f}"
                ax.annotate(label, xy=(nearest["budget"], middle),
                            xytext=(4, 0), textcoords="offset points", rotation=90,
                            va="center", fontsize=6.5, color="0.35")
            reductions = [row["error_reduction_pct_interpolated"] for row in matched]
            ax.set_title(f"n={n}: median error reduction "
                         f"{np.median(reductions):.1f}%", fontsize=9.5)
        ax.set_xlabel("CNOT budget" if metric == "cnot" else "total gate budget")
        if ax is axes[0][0]:
            ax.set_ylabel(YLABEL)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.0, loc="lower left")
    fig.suptitle("Formula comparison at an equal "
                 + ("CNOT" if metric == "cnot" else "total-gate")
                 + " budget (dotted: gap between the curves at that budget)",
                 fontsize=10.5)
    _finish(fig, path)


def order_comparison_equal_cnot(dataset: Dataset, path: str) -> None:
    _budget_panels(dataset, path, "cnot")


def order_comparison_equal_gates(dataset: Dataset, path: str) -> None:
    _budget_panels(dataset, path, "total_gates")


def _resource_vs_qubits(dataset: Dataset, path: str, attr: str,
                        ylabel: str, title: str,
                        show_analytic: bool = False, log_y: bool = True,
                        note: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    for order in dataset.orders():
        for r_fixed, alpha in ((45, 0.55), (200, 1.0)):
            matching = [row for n in dataset.n_values()
                        for row in dataset.series(n, order) if row.r == r_fixed]
            if not matching:
                continue
            ns = [row.n for row in matching]
            values = [getattr(row, attr) for row in matching]
            draw = ax.loglog if log_y else ax.semilogx
            draw(ns, values, marker=MARKERS[order], ms=4.4, lw=1.4,
                 color="#d1495b" if order == 1 else "#00798c", alpha=alpha,
                 label=f"{ORDER_LABELS[order]}, r={r_fixed}")
            if show_analytic:
                draw(ns, [circuit_cost_ref(dataset, row, attr) for row in matching],
                     ls=":", lw=1.1, color="#d1495b" if order == 1 else "#00798c",
                     alpha=alpha)
    ticks = dataset.n_values()
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(n) for n in ticks])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if note:
        ax.annotate(note, xy=(0.5, 0.48), xycoords="axes fraction",
                    ha="center", va="center", fontsize=7.5, color="0.25",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85"))
    ax.set_xlabel("number of qubits  n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.0)
    _finish(fig, path)


def circuit_cost_ref(dataset: Dataset, row, attr: str) -> int:
    """Closed-form value for one measured row (dotted comparison lines)."""
    from .scaling import analytic_resources
    ref = analytic_resources(row.n, row.r, row.order)
    return {"cnot": ref["cnot"], "total": ref["total"]}.get(attr, 0)


def cnot_vs_qubits(dataset: Dataset, path: str) -> None:
    _resource_vs_qubits(dataset, path, "cnot", "CNOT count",
                        "CNOT count vs chain length, two step counts\n"
                        "dotted: analytic 2r(n-1) and 2(r+1)(n-1)",
                        show_analytic=True, log_y=False,
                        note="CNOT = 2 x bonds x steps: exactly linear in (n-1)")


def total_gates_vs_qubits(dataset: Dataset, path: str) -> None:
    _resource_vs_qubits(dataset, path, "total", "total gates (CNOT + RX + RZ)",
                        "Total gate count vs chain length, two step counts\n"
                        "dotted: analytic r(4n-3) and 3(r+1)(n-1)+rn",
                        show_analytic=True, log_y=False,
                        note="order 2 sits only n gates from order 1 at the next step")


def depth_vs_qubits(dataset: Dataset, path: str) -> None:
    _resource_vs_qubits(
        dataset, path, "depth", "logical circuit depth (conservative scheduler)",
        "Logical depth vs chain length, two step counts\n"
        "depth is set by r, barely by n",
        log_y=False,
        note="r=45 and r=200 differ by ~4.5x; within one r the change is +3 layers\n"
             "per extra qubit (0.4% from n=4 to n=6)")


def write_all(dataset: Dataset, outdir: str) -> list[str]:
    """Generate every required figure and return the paths written."""
    os.makedirs(outdir, exist_ok=True)
    jobs = [
        ("error_vs_cnot.png", error_vs_cnot),
        ("error_vs_total_gates.png", error_vs_total_gates),
        ("error_vs_circuit_depth.png", lambda d, p: error_vs_depth(d, p)),
        ("error_vs_two_qubit_depth.png",
         lambda d, p: error_vs_depth(d, p, two_qubit=True)),
        ("order_comparison_equal_steps.png", order_comparison_equal_steps),
        ("order_comparison_equal_cnot.png", order_comparison_equal_cnot),
        ("order_comparison_equal_gates.png", order_comparison_equal_gates),
        ("cnot_vs_qubits.png", cnot_vs_qubits),
        ("total_gates_vs_qubits.png", total_gates_vs_qubits),
        ("depth_vs_qubits.png", depth_vs_qubits),
    ]
    written = []
    for name, fn in jobs:
        path = os.path.join(outdir, name)
        fn(dataset, path)
        written.append(path)
    return written
