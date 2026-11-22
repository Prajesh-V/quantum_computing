"""Seven figures for the SDK, transpilation and hardware-aware results."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import experiments  # noqa: E402

__all__ = ["CAPTIONS", "make_all"]

ORDER_COLORS = {1: "#d1495b", 2: "#00798c"}
TOPOLOGY_COLORS = {"ideal": "#1b4965", "line": "#2a9d8f", "ring": "#8d6a9f",
                   "star": "#e07a5f"}
TOPOLOGY_MARKERS = {"ideal": "o", "line": "s", "ring": "^", "star": "D"}
TOPOLOGY_STYLES = {"ideal": "-", "line": "--", "ring": "-.", "star": ":"}

CAPTIONS = {
    "fig06": "Custom circuit unitary against the Qiskit circuit unitary, up to a global phase.",
    "fig07": "Logical against transpiled CNOT count, per connectivity.",
    "fig08": "Logical against transpiled circuit depth, per connectivity.",
    "fig09": "Extra CNOTs introduced by routing, under each connectivity map.",
    "fig10": "First versus second order compiled resources on the star topology.",
    "fig11": "Fidelity against compiled two-qubit gate count under gate-level noise.",
    "fig12": "Total error against compiled two-qubit gates: logical versus hardware-aware.",
}


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_equivalence(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for order in (1, 2):
        subset = [row for row in rows if row["trotter_order"] == order]
        sizes = sorted({row["system_size"] for row in subset})
        for n in sizes:
            points = sorted([row for row in subset if row["system_size"] == n],
                            key=lambda row: row["trotter_steps"])
            ax.semilogy([row["trotter_steps"] for row in points],
                        [max(row["spectral_distance"], 1e-18) for row in points],
                        marker="o" if order == 1 else "s", ms=5, lw=1.4,
                        color=ORDER_COLORS[order], alpha={3: 0.55, 4: 0.8, 5: 1.0}[n],
                        label=f"n={n}, {'first' if order == 1 else 'second'} order")
    ax.axhline(1e-14, color="0.5", ls=":", lw=1.2)
    ax.annotate("machine precision", xy=(0.02, 1.25e-14), xycoords=("axes fraction", "data"),
                fontsize=8, color="0.4")
    ax.set_xlabel("Trotter steps  r")
    ax.set_ylabel(r"$\min_\phi \Vert U_\mathrm{custom} - e^{i\phi} U_\mathrm{Qiskit}\Vert_2$")
    ax.set_title("Custom and Qiskit circuits are the same unitary\n"
                 "(no phase offset beyond round-off)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    return _finish(fig, path)


def fig_counts(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=False)
    n_show = max(row["system_size"] for row in rows)
    for ax, order in zip(axes, (1, 2)):
        for topology in ("ideal", "line", "star"):
            subset = sorted([row for row in rows
                             if row["system_size"] == n_show
                             and row["trotter_order"] == order
                             and row["topology"] == topology],
                            key=lambda row: row["trotter_steps"])
            if not subset:
                continue
            ax.plot([row["trotter_steps"] for row in subset],
                    [row["compiled_cx"] for row in subset],
                    marker=TOPOLOGY_MARKERS[topology], ms=5, lw=1.5,
                    ls=TOPOLOGY_STYLES[topology], color=TOPOLOGY_COLORS[topology],
                    label=f"{topology}")
        ax.set_xlabel("Trotter steps  r")
        ax.set_ylabel("CNOT count")
        ax.set_title(f"n={n_show}, {'first' if order == 1 else 'second'} order",
                     fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, title="connectivity", title_fontsize=8)
    fig.suptitle("Transpilation leaves the CNOT count unchanged on a line "
                 "and inflates it on a star", fontsize=10.5)
    return _finish(fig, path)


def fig_depth(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    n_show = max(row["system_size"] for row in rows)
    for ax, order in zip(axes, (1, 2)):
        for topology in ("ideal", "line", "star"):
            subset = sorted([row for row in rows
                             if row["system_size"] == n_show
                             and row["trotter_order"] == order
                             and row["topology"] == topology],
                            key=lambda row: row["trotter_steps"])
            if not subset:
                continue
            ax.plot([row["trotter_steps"] for row in subset],
                    [row["compiled_depth"] for row in subset],
                    marker=TOPOLOGY_MARKERS[topology], ms=5, lw=1.5,
                    ls=TOPOLOGY_STYLES[topology], color=TOPOLOGY_COLORS[topology],
                    label=topology)
        ax.set_xlabel("Trotter steps  r")
        ax.set_ylabel("circuit depth")
        ax.set_title(f"n={n_show}, {'first' if order == 1 else 'second'} order",
                     fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, title="connectivity", title_fontsize=8)
    fig.suptitle("Depth is the resource the star topology really costs",
                 fontsize=10.5)
    return _finish(fig, path)


def fig_swap(rows, path):
    sizes = sorted({row["system_size"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    width = 0.36
    for index, order in enumerate((1, 2)):
        values = []
        for n in sizes:
            subset = [row for row in rows
                      if row["system_size"] == n and row["trotter_order"] == order]
            values.append(max((row["star_extra_cx"] for row in subset), default=0))
        positions = [position + (index - 0.5) * width for position in range(len(sizes))]
        bars = ax.bar(positions, values, width=width,
                      color=ORDER_COLORS[order],
                      label=f"{'first' if order == 1 else 'second'} order")
        for bar, value in zip(bars, values):
            ax.annotate(f"{int(value)}", (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"n={n}" for n in sizes])
    ax.set_ylabel("extra CNOTs introduced by routing (r=8)")
    ax.set_title("Line and ring connectivity add no CNOTs; the star does", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)
    return _finish(fig, path)


def fig_order_resources(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    n_show = max(row["system_size"] for row in rows)
    metrics = (("compiled_cx", "compiled CNOTs"), ("compiled_depth", "compiled depth"))
    for ax, (key, label) in zip(axes, metrics):
        for order in (1, 2):
            subset = sorted([row for row in rows
                             if row["system_size"] == n_show
                             and row["trotter_order"] == order
                             and row["topology"] == "star"],
                            key=lambda row: row["trotter_steps"])
            if not subset:
                continue
            ax.plot([row["trotter_steps"] for row in subset],
                    [row[key] for row in subset],
                    marker="o" if order == 1 else "s", ms=5, lw=1.6,
                    color=ORDER_COLORS[order],
                    label=f"{'first' if order == 1 else 'second'} order")
        ax.set_xlabel("Trotter steps  r")
        ax.set_ylabel(label)
        ax.set_title(f"n={n_show}, star topology", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.suptitle("At equal r the second-order circuit costs more compiled two-qubit "
                 "gates and more depth on the star", fontsize=10.5)
    return _finish(fig, path)


def fig_fidelity_vs_cx(rows, path):
    """Error against compiled two-qubit gates on a log axis.

    A log axis is used because the r = 1 points sit two orders of magnitude above
    the rest; on a linear axis they flatten everything that matters.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for topology in ("ideal", "line", "star"):
        for order in (1, 2):
            subset = [row for row in rows if row["trotter_order"] == order]
            if not subset:
                continue
            big = topology == "ideal"
            ax.scatter([max(row[f"{topology}_cx"], 0.5) for row in subset],
                       [max(row[f"{topology}_total_error"], 1e-3) for row in subset],
                       s=90 if big else 40, marker="o" if big else TOPOLOGY_MARKERS[topology],
                       facecolors="none",
                       edgecolors=TOPOLOGY_COLORS[topology],
                       linewidths=1.6 if big else 1.3,
                       label=f"{topology}, {'first' if order == 1 else 'second'}")
    ax.set_yscale("log")
    ax.set_xlabel("compiled two-qubit gate count (CNOTs)")
    ax.set_ylabel("total error  $1-F$  at the exact state")
    ax.set_title("Total error under gate-level depolarising noise\n"
                 r"($p_1 = p_2 = 10^{-3}$; large hollow circles: logical circuit)",
                 fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=7.8, ncol=2)
    return _finish(fig, path)


def fig_efficiency(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, order in zip(axes, (1, 2)):
        for topology in ("ideal", "line", "star"):
            subset = [row for row in rows if row["trotter_order"] == order]
            if not subset:
                continue
            big = topology == "ideal"
            ax.scatter([max(row[f"{topology}_cx"], 0.5) for row in subset],
                       [max(row[f"{topology}_total_error"], 1e-3) for row in subset],
                       s=80 if big else 34, marker="o" if big else TOPOLOGY_MARKERS[topology],
                       facecolors="none", edgecolors=TOPOLOGY_COLORS[topology],
                       linewidths=1.6 if big else 1.1, alpha=0.9,
                       label=topology)
        ax.set_yscale("log")
        ax.set_xlabel("compiled two-qubit gates")
        ax.set_ylabel("total error  $1-F$")
        ax.set_title(f"{'first' if order == 1 else 'second'} order", fontsize=10)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8, title="connectivity", title_fontsize=8)
    fig.suptitle("For the same error the star topology needs many more compiled "
                 "two-qubit gates\n(large hollow circles: logical circuit, hidden "
                 "under the line markers where they coincide)", fontsize=10)
    return _finish(fig, path)


def make_all(directory: str, results: dict) -> list[dict]:
    os.makedirs(directory, exist_ok=True)
    rows = results["transpilation"]["rows"]
    figures = [
        ("fig06_custom_vs_sdk_unitary.png",
         fig_equivalence(results["equivalence"], os.path.join(directory, "fig06_custom_vs_sdk_unitary.png")),
         CAPTIONS["fig06"]),
        ("fig07_logical_vs_transpiled_cnot.png",
         fig_counts(rows, os.path.join(directory, "fig07_logical_vs_transpiled_cnot.png")),
         CAPTIONS["fig07"]),
        ("fig08_logical_vs_transpiled_depth.png",
         fig_depth(rows, os.path.join(directory, "fig08_logical_vs_transpiled_depth.png")),
         CAPTIONS["fig08"]),
        ("fig09_swap_overhead.png",
         fig_swap(results["connectivity"], os.path.join(directory, "fig09_swap_overhead.png")),
         CAPTIONS["fig09"]),
        ("fig10_order_compiled_resources.png",
         fig_order_resources(rows, os.path.join(directory, "fig10_order_compiled_resources.png")),
         CAPTIONS["fig10"]),
        ("fig11_fidelity_vs_compiled_cx.png",
         fig_fidelity_vs_cx(results["noise"]["compiled_noise"],
                            os.path.join(directory, "fig11_fidelity_vs_compiled_cx.png")),
         CAPTIONS["fig11"]),
        ("fig12_logical_vs_hardware_efficiency.png",
         fig_efficiency(results["noise"]["compiled_noise"],
                        os.path.join(directory, "fig12_logical_vs_hardware_efficiency.png")),
         CAPTIONS["fig12"]),
    ]
    return [{"file": name, "path": path, "caption": caption}
            for name, path, caption in figures]
