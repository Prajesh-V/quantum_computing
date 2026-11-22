"""Figures for Phase 3.

Twelve figures, each drawing a small, named subset of the measured rows rather
than everything at once.  The visual grammar is fixed across all of them:

* **colour = system size** (n = 4, 5, 6),
* **line style + marker = Trotter order** (solid/o = 1st, dashed/s = 2nd),
* **hollow marker = a boundary result** (optimal r at an end of the grid),
* grey dotted = ideal / noise-free reference.

All values are the raw measured numbers; zero errors are clipped only for the
logarithmic axes and the tables keep the unmodified values.
"""

from __future__ import annotations

import math
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

__all__ = ["make_all_figures", "FIGURE_NAMES", "compute_layout"]

COLORS = {4: "#1b4965", 5: "#2a9d8f", 6: "#c1666b"}
MARKERS = {1: "o", 2: "s"}
LINESTYLES = {1: "-", 2: "--"}
ORDER_LABEL = {1: "1st order", 2: "2nd order"}

FLOOR = 1e-16          # only for log axes
GRID = dict(which="both", alpha=0.25)

FIGURE_NAMES = [
    "fig01_fidelity_vs_steps.png",
    "fig02_trotter_error_vs_steps.png",
    "fig03_noise_error_vs_steps.png",
    "fig04_total_error_vs_steps.png",
    "fig05_fidelity_vs_cnot.png",
    "fig06_fidelity_vs_total_gates.png",
    "fig07_order_comparison_gate_noise.png",
    "fig08_global_vs_gate_noise.png",
    "fig09_optimal_r_vs_noise.png",
    "fig10_sensitivity_heatmap.png",
    "fig11_purity_vs_steps.png",
    "fig12_total_error_vs_depth.png",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rows(rows, **criteria):
    out = []
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            out.append(row)
    return sorted(out, key=lambda row: row["trotter_steps"])


def _ideal(rows, n, order):
    """The zero-noise rows: gate-level noise with p1 = p2 = 0, i.e. the ideal circuit."""
    return _rows(rows, system_size=n, trotter_order=order, noise_model="gate",
                 single_qubit_error=0.0, two_qubit_error=0.0)


def _xy(rows, key, xkey="trotter_steps"):
    x = np.array([row[xkey] for row in rows], dtype=float)
    y = np.maximum(np.array([row[key] for row in rows], dtype=float), FLOOR)
    return x, y


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _error_axis(ax, xlabel="number of Trotter steps  r"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, **GRID)


def _fidelity_axis(ax, xlabel="number of Trotter steps  r"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, **GRID)


def compute_layout(rows):
    """The (system size, order, noise) subsets the figures actually use."""
    return {
        "n_values": sorted({row["system_size"] for row in rows}),
        "orders": sorted({row["trotter_order"] for row in rows}),
        "reference_noise": min({row["single_qubit_error"] for row in rows
                                if row["noise_model"] == "gate"
                                and row["single_qubit_error"] > 0}, default=None),
    }


# ---------------------------------------------------------------------------
# 1. ideal vs noisy fidelity
# ---------------------------------------------------------------------------

def fig_fidelity_vs_steps(gate_rows, ns, p, path):
    fig, axes = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 4.6), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for order in (1, 2):
            ideal = _ideal(gate_rows, n, order)
            x, y = _xy(ideal, "fidelity")
            ax.plot(x, y, ls=":", marker=MARKERS[order], ms=3.4, lw=1.3,
                    color="0.55",
                    label=f"ideal, {ORDER_LABEL[order]}")
            noisy = _rows(gate_rows, system_size=n, trotter_order=order,
                          noise_model="gate", single_qubit_error=p)
            x, y = _xy(noisy, "fidelity")
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=4.2,
                    lw=1.6, color=COLORS[n],
                    label=f"noisy, {ORDER_LABEL[order]}")
        _fidelity_axis(ax)
        ax.set_ylabel("fidelity vs exact evolution  F")
        ax.set_title(f"n = {n} qubits")
        ax.legend(fontsize=7.6, loc="lower left")
    fig.suptitle(f"Ideal vs gate-level depolarising noise (p1 = p2 = {p:g}), "
                 f"t = 1, |00...0> start", fontsize=11)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 2. ideal Trotter error
# ---------------------------------------------------------------------------

def fig_trotter_error(gate_rows, ns, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for n in ns:
        for order in (1, 2):
            rows = _ideal(gate_rows, n, order)
            x, y = _xy(rows, "ideal_trotter_error")
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=3.6,
                    lw=1.4, color=COLORS[n], label=f"n = {n}, {ORDER_LABEL[order]}")
    _error_axis(ax)
    ax.set_ylabel(r"ideal Trotter error  $1 - F(\rho_\mathrm{trot}, \psi_\mathrm{exact})$")
    ax.set_title("Formula error alone: noise-free Trotter error vs step count")
    ax.legend(fontsize=7.8, ncol=2)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 3. noise-induced error
# ---------------------------------------------------------------------------

def fig_noise_error(gate_rows, ns, p, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for ax, order in zip(axes, (1, 2)):
        for n in ns:
            rows = _rows(gate_rows, system_size=n, trotter_order=order,
                         noise_model="gate", single_qubit_error=p)
            x, y = _xy(rows, "noise_induced_error")
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=3.8,
                    lw=1.5, color=COLORS[n], label=f"n = {n}")
        _error_axis(ax)
        ax.set_ylabel(r"noise-induced error  $1 - F(\rho_\mathrm{noisy}, \rho_\mathrm{trot})$")
        ax.set_title(f"{ORDER_LABEL[order]}: error from the channels alone")
        ax.legend(fontsize=8)
    fig.suptitle(f"Gate-level noise accumulates with circuit length "
                 f"(p1 = p2 = {p:g})", fontsize=11)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 4. total error
# ---------------------------------------------------------------------------

def fig_total_error(gate_rows, ns, p_values, path):
    fig, axes = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 4.6), squeeze=False)
    shades = ["#6a4c93", "#1982c4", "#8ac926", "#ff924c"]
    for ax, n in zip(axes[0], ns):
        for index, p in enumerate(p_values):
            color = shades[index % len(shades)]
            for order in (1, 2):
                rows = _rows(gate_rows, system_size=n, trotter_order=order,
                             noise_model="gate", single_qubit_error=p)
                x, y = _xy(rows, "total_error")
                ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=3.4,
                        lw=1.5, color=color,
                        label=f"p1=p2={p:g}, {ORDER_LABEL[order]}")
        for order in (1, 2):
            ideal = _ideal(gate_rows, n, order)
            x, y = _xy(ideal, "total_error")
            ax.plot(x, y, ls=":", marker=MARKERS[order], ms=3.0, lw=1.2,
                    color="0.5", label=f"ideal, {ORDER_LABEL[order]}")
        _error_axis(ax)
        ax.set_ylabel(r"total error  $1 - F(\rho_\mathrm{noisy}, \psi_\mathrm{exact})$")
        ax.set_title(f"n = {n} qubits")
        ax.legend(fontsize=6.6, ncol=1, loc="lower left")
    fig.suptitle("Total simulation error: the two competing effects meet", fontsize=11)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 5-6. fidelity against resource counts
# ---------------------------------------------------------------------------

def fig_fidelity_vs_budget(gate_rows, ns, p, key, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    for n in ns:
        for order in (1, 2):
            rows = _rows(gate_rows, system_size=n, trotter_order=order,
                         noise_model="gate", single_qubit_error=p)
            x, y = _xy(rows, "fidelity", xkey=key)
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=3.8,
                    lw=1.5, color=COLORS[n], label=f"n = {n}, {ORDER_LABEL[order]}")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fidelity vs exact evolution  F")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, **GRID)
    ax.set_title(title + f"\n(gate-level noise, p1 = p2 = {p:g})")
    ax.legend(fontsize=7.8, ncol=2)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 7. order comparison under gate noise
# ---------------------------------------------------------------------------

def fig_order_comparison(gate_rows, ns, p, path):
    fig, axes = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 4.6), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for order in (1, 2):
            ideal = _ideal(gate_rows, n, order)
            ideal_x, _ = _xy(ideal, "total_error", xkey="cnot_count")
            _, ideal_y = _xy(ideal, "total_error")
            ax.plot(ideal_x, ideal_y, ls=":", marker=MARKERS[order],
                    ms=3.2, lw=1.2, color=COLORS[n],
                    label=f"ideal, {ORDER_LABEL[order]}")
            noisy = _rows(gate_rows, system_size=n, trotter_order=order,
                          noise_model="gate", single_qubit_error=p)
            x, y = _xy(noisy, "total_error", xkey="cnot_count")
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=4.0,
                    lw=1.6, color=COLORS[n],
                    label=f"noisy, {ORDER_LABEL[order]}")
        _error_axis(ax, xlabel="CNOT count (same horizontal cost for both orders)")
        ax.set_ylabel("total error")
        ax.set_title(f"n = {n} qubits")
        ax.legend(fontsize=7.4)
    fig.suptitle(f"First vs second order at matched CNOT cost, gate-level noise "
                 f"(p1 = p2 = {p:g})\ndotted: noise-free reference for the same order",
                 fontsize=10.5)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 8. global vs gate-level
# ---------------------------------------------------------------------------

def fig_global_vs_gate(gate_rows, global_rows, ns, path, reference_noise=None):
    shared = {row["single_qubit_error"] for row in gate_rows
              if row["noise_model"] == "gate" and row["single_qubit_error"] > 0}
    shared &= {row["global_noise_probability"] for row in global_rows
               if row["noise_model"] == "global_per_step"}
    pair = reference_noise if reference_noise in shared else (
        min(shared) if shared else 1e-3)
    fig, axes = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 4.6), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for order in (1, 2):
            rows = _rows(global_rows, system_size=n, trotter_order=order,
                         noise_model="global_per_step", global_noise_probability=pair)
            x, y = _xy(rows, "total_error")
            ax.plot(x, y, ls="-", marker=MARKERS[order], ms=4.0, lw=1.6,
                    color="#e07a5f", label=f"global, {ORDER_LABEL[order]}")
            rows = _rows(gate_rows, system_size=n, trotter_order=order,
                         noise_model="gate", single_qubit_error=pair)
            x, y = _xy(rows, "total_error")
            ax.plot(x, y, ls="--", marker=MARKERS[order], ms=4.0, lw=1.6,
                    color=COLORS[n], label=f"gate-level, {ORDER_LABEL[order]}")
        _error_axis(ax)
        ax.set_ylabel("total error")
        ax.set_title(f"n = {n} qubits")
        ax.legend(fontsize=7.4, loc="lower left")
    fig.suptitle("Simplified global model vs gate-level model "
                 f"(p_g = {pair:g} per step, p1 = p2 = {pair:g} per gate)", fontsize=10.5)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 9. optimal r vs noise strength
# ---------------------------------------------------------------------------

def fig_optimal_r(optimal_rows, path):
    panels = [("gate", "gate-level noise (p1 = p2)", "error probability p1 = p2"),
              ("global_per_step", "global noise per Trotter step", "p_g per step")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, (model, title, xlabel) in zip(axes, panels):
        xkey = ("single_qubit_error" if model == "gate"
                else "global_noise_probability")
        plotted_steps: set[int] = set()
        for n in sorted({row["system_size"] for row in optimal_rows}):
            for order in (1, 2):
                # only genuine noise levels: the p = 0 row has its minimum at the
                # largest tested r and is not a noise-dependent optimum
                chosen = [row for row in optimal_rows
                          if row["noise_model"] == model and row["system_size"] == n
                          and row["trotter_order"] == order
                          and row["optimum_status"] != "single_candidate"
                          and row[xkey] > 0]
                if not chosen:
                    continue
                chosen.sort(key=lambda row: row[xkey])
                x = [row[xkey] for row in chosen]
                y = [row["optimal_r"] for row in chosen]
                plotted_steps.update(y)
                interior = [row["optimum_status"] == "interior" for row in chosen]
                ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=6,
                        lw=1.5, color=COLORS[n], label=f"n = {n}, {ORDER_LABEL[order]}")
                boundary_x = [xx for xx, flag in zip(x, interior) if not flag]
                boundary_y = [yy for yy, flag in zip(y, interior) if not flag]
                if boundary_x:
                    ax.plot(boundary_x, boundary_y, ls="none", marker=MARKERS[order],
                            ms=11, mfc="none", mec=COLORS[n], mew=1.4)
        ax.set_xscale("log")
        # r_opt takes a handful of small integer values; a linear axis with one
        # tick per measured value is far easier to read than a log axis.
        ax.set_yscale("linear")
        if plotted_steps:
            ticks = sorted(plotted_steps)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(value) for value in ticks])
            ax.set_ylim(min(ticks) * 0.82, max(ticks) * 1.18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("measured optimal Trotter steps  r_opt")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(fontsize=7.6)
    fig.suptitle("Optimal step count falls as noise grows\n"
                 "hollow markers: the minimum sits at an end of the tested r grid",
                 fontsize=10.5)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 10. sensitivity heatmap
# ---------------------------------------------------------------------------

def fig_sensitivity(sensitivity_rows, path):
    grid_rows = [row for row in sensitivity_rows if row.get("grid") == "full"]
    # the sensitivity grid may be run at several system sizes; keep one panel's worth
    used_n = grid_rows[0]["system_size"] if grid_rows else 4
    grid_rows = [row for row in grid_rows if row["system_size"] == used_n]
    p1 = sorted({row["single_qubit_error"] for row in grid_rows})
    p2 = sorted({row["two_qubit_error"] for row in grid_rows})
    error = np.full((len(p2), len(p1)), np.nan)
    for row in grid_rows:
        i = p2.index(row["two_qubit_error"])
        j = p1.index(row["single_qubit_error"])
        error[i, j] = row["total_error"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    ax = axes[0]
    mesh = ax.pcolormesh(np.arange(len(p1) + 1), np.arange(len(p2) + 1),
                         np.clip(error, 1e-12, None), shading="flat",
                         cmap="viridis", norm=LogNorm(vmin=np.nanmin(error[error > 0]),
                                                      vmax=np.nanmax(error)))
    for i in range(len(p2)):
        for j in range(len(p1)):
            value = error[i, j]
            ax.text(j + 0.5, i + 0.5, f"{value:.1e}", ha="center", va="center",
                    fontsize=6.2,
                    color="white" if value < np.nanmax(error) * 0.35 else "black")
    ax.set_xticks(np.arange(len(p1)) + 0.5, [f"{v:g}" for v in p1], fontsize=8)
    ax.set_yticks(np.arange(len(p2)) + 0.5, [f"{v:g}" for v in p2], fontsize=8)
    ax.set_xlabel("single-qubit error  p1")
    ax.set_ylabel("two-qubit error  p2")
    ax.set_title("total error (annotated)")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="total error")

    ax = axes[1]
    for label, key, marker, color in (
            ("p1 only (p2 = 0)", "single_qubit_only", "o", "#1b4965"),
            ("p2 only (p1 = 0)", "two_qubit_only", "s", "#c1666b")):
        rows = [row for row in sensitivity_rows
                if row.get("grid") == key and row["system_size"] == used_n]
        if not rows:
            continue
        pf = "single_qubit_error" if key == "single_qubit_only" else "two_qubit_error"
        rows.sort(key=lambda row: row[pf])
        ax.loglog([row[pf] for row in rows], [max(row["total_error"], FLOOR) for row in rows],
                  marker=marker, ms=5, lw=1.6, color=color, label=label)
    ax.set_xlabel("error probability applied to one gate class")
    ax.set_ylabel("total error")
    ax.set_title("which gate class dominates?")
    ax.grid(True, **GRID)
    ax.legend(fontsize=8)

    r = grid_rows[0]["trotter_steps"] if grid_rows else 0
    order = grid_rows[0]["trotter_order"] if grid_rows else 1
    fig.suptitle(f"Noise sensitivity, n = {used_n}, r = {r}, "
                 f"{ORDER_LABEL.get(order, '')}", fontsize=10.5)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 11. purity
# ---------------------------------------------------------------------------

def fig_purity(gate_rows, ns, p, path):
    fig, axes = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 4.5), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for order in (1, 2):
            rows = _rows(gate_rows, system_size=n, trotter_order=order,
                         noise_model="gate", single_qubit_error=p)
            x, y = _xy(rows, "purity")
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=3.8,
                    lw=1.5, color=COLORS[n], label=f"noisy, {ORDER_LABEL[order]}")
            ideal = _ideal(gate_rows, n, order)
            x, y = _xy(ideal, "purity")
            ax.plot(x, y, ls=":", marker=MARKERS[order], ms=3.0, lw=1.2,
                    color="0.55", label=f"ideal, {ORDER_LABEL[order]}")
        ax.axhline(1.0 / 2 ** n, color="0.4", ls="-.", lw=1.0)
        ax.annotate(r"maximally mixed  $1/2^n$", xy=(0.98, 1.0 / 2 ** n),
                    xycoords=blended_transform_factory(ax.transAxes, ax.transData),
                    ha="right", va="bottom", fontsize=7.2, color="0.35")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of Trotter steps  r")
        ax.set_ylabel(r"purity  $\mathrm{Tr}(\rho^2)$")
        ax.set_title(f"n = {n} qubits")
        ax.grid(True, **GRID)
        ax.legend(fontsize=7.2, loc="lower left")
    fig.suptitle(f"Purity loss with circuit length (p1 = p2 = {p:g})", fontsize=10.5)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 12. total error vs depth
# ---------------------------------------------------------------------------

def fig_error_vs_depth(gate_rows, ns, p, path):
    fig, axes = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 4.5), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for order in (1, 2):
            rows = _rows(gate_rows, system_size=n, trotter_order=order,
                         noise_model="gate", single_qubit_error=p)
            x, y = _xy(rows, "total_error", xkey="logical_depth")
            ax.plot(x, y, ls=LINESTYLES[order], marker=MARKERS[order], ms=3.8,
                    lw=1.5, color=COLORS[n], label=f"noisy, {ORDER_LABEL[order]}")
        _error_axis(ax, xlabel="logical circuit depth (ASAP schedule)")
        ax.set_ylabel("total error")
        ax.set_title(f"n = {n} qubits")
        ax.legend(fontsize=7.6)
    fig.suptitle(f"Depth is not free: total error vs circuit depth "
                 f"(p1 = p2 = {p:g})", fontsize=10.5)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def make_all_figures(rows, sensitivity_rows, optimal_rows, figure_dir,
                     reference_noise=None) -> list[str]:
    """Draw all twelve figures into ``figure_dir`` and return their paths."""
    os.makedirs(figure_dir, exist_ok=True)
    gate_rows = [row for row in rows if row["noise_model"] == "gate"]
    global_rows = [row for row in rows if row["noise_model"].startswith("global")]
    ns = sorted({row["system_size"] for row in rows})

    if reference_noise is None:
        candidates = sorted({row["single_qubit_error"] for row in gate_rows
                             if row["noise_model"] == "gate"
                             and row["single_qubit_error"] > 0})
        reference_noise = candidates[min(1, len(candidates) - 1)] if candidates else 1e-3
    p_values = sorted({row["single_qubit_error"] for row in gate_rows
                       if row["noise_model"] == "gate" and row["single_qubit_error"] > 0})
    draw = p_values[:3] if len(p_values) > 3 else p_values

    paths = []
    target = lambda name: os.path.join(figure_dir, name)  # noqa: E731

    fig_fidelity_vs_steps(gate_rows, ns, reference_noise,
                          target("fig01_fidelity_vs_steps.png"))
    paths.append(target("fig01_fidelity_vs_steps.png"))

    fig_trotter_error(gate_rows, ns, target("fig02_trotter_error_vs_steps.png"))
    paths.append(target("fig02_trotter_error_vs_steps.png"))

    fig_noise_error(gate_rows, ns, reference_noise,
                    target("fig03_noise_error_vs_steps.png"))
    paths.append(target("fig03_noise_error_vs_steps.png"))

    fig_total_error(gate_rows, ns, draw, target("fig04_total_error_vs_steps.png"))
    paths.append(target("fig04_total_error_vs_steps.png"))

    fig_fidelity_vs_budget(gate_rows, ns, reference_noise, "cnot_count",
                           "CNOT count",
                           "Fidelity against CNOT budget",
                           target("fig05_fidelity_vs_cnot.png"))
    paths.append(target("fig05_fidelity_vs_cnot.png"))

    fig_fidelity_vs_budget(gate_rows, ns, reference_noise, "total_gate_count",
                           "total gate count (CNOT + RX + RZ)",
                           "Fidelity against total gate budget",
                           target("fig06_fidelity_vs_total_gates.png"))
    paths.append(target("fig06_fidelity_vs_total_gates.png"))

    fig_order_comparison(gate_rows, ns, reference_noise,
                         target("fig07_order_comparison_gate_noise.png"))
    paths.append(target("fig07_order_comparison_gate_noise.png"))

    fig_global_vs_gate(gate_rows, global_rows, ns,
                       target("fig08_global_vs_gate_noise.png"),
                       reference_noise=reference_noise)
    paths.append(target("fig08_global_vs_gate_noise.png"))

    fig_optimal_r(optimal_rows, target("fig09_optimal_r_vs_noise.png"))
    paths.append(target("fig09_optimal_r_vs_noise.png"))

    fig_sensitivity(sensitivity_rows, target("fig10_sensitivity_heatmap.png"))
    paths.append(target("fig10_sensitivity_heatmap.png"))

    fig_purity(gate_rows, ns, reference_noise, target("fig11_purity_vs_steps.png"))
    paths.append(target("fig11_purity_vs_steps.png"))

    fig_error_vs_depth(gate_rows, ns, reference_noise,
                       target("fig12_total_error_vs_depth.png"))
    paths.append(target("fig12_total_error_vs_depth.png"))
    return paths


CAPTIONS = {
    "fig01_fidelity_vs_steps.png":
        "Fidelity against exact evolution for the ideal and gate-level noisy "
        "circuit, both Trotter orders, at the reference noise level.",
    "fig02_trotter_error_vs_steps.png":
        "Noise-free Trotter error only: 1st order falls as r^-1, 2nd order as r^-2.",
    "fig03_noise_error_vs_steps.png":
        "Error contributed by the channels alone (noisy vs ideal Trotter). "
        "Grows with circuit length, roughly linearly in the gate count.",
    "fig04_total_error_vs_steps.png":
        "Total error at several noise strengths: the descending formula error "
        "and the ascending noise error cross, producing a minimum.",
    "fig05_fidelity_vs_cnot.png":
        "The same data against CNOT budget, the resource Phase 1 counted.",
    "fig06_fidelity_vs_total_gates.png":
        "Fidelity against total elementary gate count.",
    "fig07_order_comparison_gate_noise.png":
        "First vs second order at matched CNOT cost, with the noise-free "
        "reference of each order shown dotted.",
    "fig08_global_vs_gate_noise.png":
        "Global (one channel per step) vs gate-level (one channel per gate) "
        "noise at comparable strength parameters.",
    "fig09_optimal_r_vs_noise.png":
        "Measured argmin of the total error against noise strength; hollow "
        "markers are boundary minima.",
    "fig10_sensitivity_heatmap.png":
        "Total error over the (p1, p2) grid, plus the two single-parameter "
        "slices that separate single- and two-qubit gate contributions.",
    "fig11_purity_vs_steps.png":
        "Purity against circuit length, with the maximally mixed value 1/2^n.",
    "fig12_total_error_vs_depth.png":
        "Total error against the conservative logical depth of Phase 1.",
}
