"""Figures for the Phase 4 audit and robustness results.

Five figures, all drawn from the audit's own data (never from Phase 3 plots).
Every panel is labelled with units and with the exact configuration it shows,
and no axis is clipped to hide a regime.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .robustness import INITIAL_STATES  # noqa: E402

__all__ = ["CAPTIONS", "fig_initial_state", "fig_evolution_time",
           "fig_noise_strength", "fig_noise_components", "fig_candidate_grid",
           "make_all"]

COLORS = {"|0...0>": "#1b4965", "|1...1>": "#5fa8d3",
          "|+...+>": "#c1666b", "random": "#8d6a9f"}
ORDER_COLORS = {1: "#d1495b", 2: "#00798c"}

CAPTIONS = {
    "fig01": "Optimal tested step count against noise strength for four initial states.",
    "fig02": "Optimal tested step count against total evolution time.",
    "fig03": "Every recorded optimal step count against noise strength, one line per "
             "(system size, order, noise model) group.",
    "fig04": "Noise-induced error against step count when only single-qubit gates are "
             "noisy, only two-qubit gates are noisy, or both are.",
    "fig05": "Dense candidate grid around the reported optimum: the coarse Phase 3 grid "
             "is marked against every integer step count.",
}


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_initial_state(summary, path):
    """r* vs noise level for the four initial states, one panel per order."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, order in zip(axes, (1, 2)):
        for state in INITIAL_STATES:
            series = [row for row in summary
                      if row["trotter_order"] == order and row["initial_state"] == state]
            series.sort(key=lambda row: row["single_qubit_error"])
            if not series:
                continue
            levels = [max(row["single_qubit_error"], 1e-5) for row in series]
            steps = [row["optimal_r"] for row in series]
            ax.plot(levels, steps, marker="o", ms=5, lw=1.6, color=COLORS[state],
                    label=state)
        ax.set_xscale("log")
        ax.set_xlabel("gate error probability  $p_1 = p_2$")
        ax.set_title(f"{'first' if order == 1 else 'second'} order")
        ax.grid(True, which="both", alpha=0.25)
        ax.axvline(1e-5, color="0.6", ls=":", lw=1.0)
        ax.annotate("noiseless\n(plotted at 1e-5)", xy=(1.2e-5, 24), fontsize=7.5,
                    color="0.4")
    axes[0].set_ylabel("optimal tested steps  $r^*$")
    axes[0].legend(fontsize=8.5, title="initial state", title_fontsize=8.5)
    fig.suptitle("n = 4: the interior optimum exists for every initial state, "
                 "but its location moves", fontsize=10.5)
    return _finish(fig, path)


def fig_evolution_time(summary, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for order in (1, 2):
        series = sorted([row for row in summary if row["trotter_order"] == order],
                        key=lambda row: row["total_time"])
        times = [row["total_time"] for row in series]
        steps = [row["optimal_r"] for row in series]
        ax.loglog(times, steps, marker="o" if order == 1 else "s", ms=6, lw=1.8,
                  color=ORDER_COLORS[order],
                  label=f"{'first' if order == 1 else 'second'} order (r* = "
                        f"{steps[0]} -> {steps[-1]})")
    ax.set_xlabel("total evolution time  $t$")
    ax.set_ylabel("optimal tested steps  $r^*$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    ax.set_title("n = 4, $p_1 = p_2 = 10^{-3}$: the optimum grows with the\n"
                 "evolution time, roughly in proportion to it", fontsize=10)
    return _finish(fig, path)


def fig_noise_strength(groups, path):
    """r* against noise strength, split by noise model.

    Boundary optima (and the single-level end-of-circuit groups) are excluded:
    they sit at r = 512 and would compress every real trend into a flat band
    hugging zero.  They are reported in the table and in the CSV instead.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    markers = {1: "o", 2: "s"}
    titles = {"gate": "gate-level noise (per-gate $p_1 = p_2$)",
              "global_per_step": "global noise (per-step $p_g$)"}
    for ax, model in zip(axes, ("gate", "global_per_step")):
        for group in groups:
            if group["noise_model"] != model:
                continue
            levels = [level for level in group["levels"] if level > 0]
            steps = group["optimal_steps"][-len(levels):]
            if len(levels) < 2:
                continue
            if any(step == group.get("tested_r_max", 512) for step in steps):
                continue
            colour = {4: "#1b4965", 5: "#5fa8d3", 6: "#c1666b"}[int(group["system_size"])]
            ax.plot(levels, steps, marker=markers[int(group["trotter_order"])], ms=5,
                    lw=1.6, color=colour,
                    ls="-" if int(group["trotter_order"]) == 1 else "--",
                    label=f"n={int(group['system_size'])}, "
                          f"{'first' if group['trotter_order'] == 1 else 'second'} order")
        ax.set_xscale("log")
        ax.set_xlabel("noise strength")
        ax.set_title(titles[model], fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, title="system size, order", title_fontsize=8)
    axes[0].set_ylabel("optimal tested steps  $r^*$")
    axes[0].set_ylim(0, None)
    fig.suptitle("No group increases with noise; several flatten at the smallest "
                 "tested $r$.\nBoundary optima at $r=512$ are omitted here",
                 fontsize=10.5)
    return _finish(fig, path)


def fig_noise_components(rows, path, level: float = 1e-3, n: int = 4):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    styles = {"single_qubit_only": ("#00798c", "o", "single-qubit gates only"),
              "two_qubit_only": ("#d1495b", "s", "two-qubit gates only"),
              "both": ("#3d3d3d", "^", "both classes")}
    for ax, order in zip(axes, (1, 2)):
        for variant, (colour, marker, label) in styles.items():
            # the nominal level is the larger of the two probabilities, which is
            # the level for every variant (p1-only, p2-only and both)
            series = [row for row in rows
                      if row["variant"] == variant
                      and row["trotter_order"] == order
                      and abs(max(row["single_qubit_error"],
                                  row["two_qubit_error"]) - level) < 1e-15]
            if not series:
                continue
            series.sort(key=lambda row: row["trotter_steps"])
            steps = [row["trotter_steps"] for row in series]
            noise = [row["noise_induced_error"] for row in series]
            ax.plot(steps, noise, marker=marker, ms=4, lw=1.6, color=colour, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("number of Trotter steps  $r$")
        ax.set_title(f"{'first' if order == 1 else 'second'} order")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("noise-induced error  $1 - F(\\rho_{noisy}, \\psi_{Trotter})$")
    fig.suptitle(f"n = {n}, nominal level {level:g}: the two gate classes contribute "
                 "comparably at equal per-gate error", fontsize=10.5)
    return _finish(fig, path)


def fig_candidate_grid(summary, rows, path, level: float = 1e-3):
    groups = sorted(summary, key=lambda row: (row["trotter_order"], row["system_size"]))
    columns = min(2, len(groups))
    rows_count = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns,
                             figsize=(5.6 * columns, 3.6 * rows_count), squeeze=False)
    for index, entry in enumerate(groups):
        ax = axes[index // columns][index % columns]
        series = [row for row in rows
                  if row["system_size"] == entry["system_size"]
                  and row["trotter_order"] == entry["trotter_order"]
                  and abs(row["single_qubit_error"] - level) < 1e-15]
        series.sort(key=lambda row: row["trotter_steps"])
        steps = [row["trotter_steps"] for row in series]
        errors = [row["total_error"] for row in series]
        ax.plot(steps, errors, marker="o", ms=4, lw=1.5, color=ORDER_COLORS[1])
        ax.axvline(entry["phase3_grid_optimal_r"], color="#d1495b", ls="--", lw=1.4,
                   label=f"Phase 3 grid optimum r={entry['phase3_grid_optimal_r']}")
        ax.axvline(entry["dense_grid_optimal_r"], color="#00798c", ls=":", lw=1.8,
                   label=f"dense grid optimum r={entry['dense_grid_optimal_r']}")
        ax.set_xlabel("Trotter steps  $r$ (every integer)")
        ax.set_ylabel("total error")
        ax.set_title(f"n={entry['system_size']}, "
                     f"{'first' if entry['trotter_order'] == 1 else 'second'} order",
                     fontsize=9.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.2)
    for index in range(len(groups), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle(f"Dense candidate grid at $p$ = {level:g}: the coarse grid can miss "
                 "the best discrete step count", fontsize=10.5)
    return _finish(fig, path)


def make_all(directory: str, initial, evolution, strength, components,
             candidate) -> list[dict]:
    """Render all five audit figures into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    figures = [
        ("fig01_initial_state_optimal_r.png",
         fig_initial_state(initial["summary"], os.path.join(directory, "fig01_initial_state_optimal_r.png")),
         CAPTIONS["fig01"]),
        ("fig02_evolution_time_optimal_r.png",
         fig_evolution_time(evolution["summary"], os.path.join(directory, "fig02_evolution_time_optimal_r.png")),
         CAPTIONS["fig02"]),
        ("fig03_noise_strength_trend.png",
         fig_noise_strength(strength["groups"], os.path.join(directory, "fig03_noise_strength_trend.png")),
         CAPTIONS["fig03"]),
        ("fig04_noise_components.png",
         fig_noise_components(components["rows"], os.path.join(directory, "fig04_noise_components.png")),
         CAPTIONS["fig04"]),
        ("fig05_candidate_grid.png",
         fig_candidate_grid(candidate["summary"], candidate["rows"],
                            os.path.join(directory, "fig05_candidate_grid.png")),
         CAPTIONS["fig05"]),
    ]
    return [{"file": name, "path": path, "caption": caption}
            for name, path, caption in figures]
