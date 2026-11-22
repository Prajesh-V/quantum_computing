"""Minimum measured resources needed to reach a target accuracy.

Rules, all of them deliberate:

* a target is reached only by a **measured** configuration whose operator error
  is ``<= threshold``; nothing is extrapolated to reach a threshold that the
  data never reaches (those rows say ``Not achieved within tested range``);
* configurations sitting at the numerical floor (``err <= 1e-10``) are excluded
  from being "the cheapest way to reach a target" at looser thresholds, because
  their error is round-off rather than truncation;
* if several configurations reach a target, the cheapest by the relevant
  resource wins, and ties break towards the smaller step count;
* no monotonicity of error in the resource is assumed - the search scans every
  measured row, so a non-monotone curve cannot produce a false minimum.
"""

from __future__ import annotations

import csv
import os

from .resource_analysis import Dataset, NUMERICAL_FLOOR

THRESHOLDS = (1e-1, 1e-2, 1e-3, 1e-4)
NOT_ACHIEVED = "Not achieved within tested range"

COLUMNS = [
    "n", "order", "target_error", "achieved", "selected_by",
    "r", "cnot_count", "total_gates", "circuit_depth", "two_qubit_depth",
    "operator_norm_error", "note",
]


def _candidates(dataset: Dataset, n: int, order: int, threshold: float):
    return [row for row in dataset.series(n, order)
            if row.err_op <= threshold and row.err_op > NUMERICAL_FLOOR]


def minimum_resources(dataset: Dataset, thresholds=THRESHOLDS) -> list[dict]:
    """Two rows per ``(n, order, threshold)``: cheapest by CNOT, cheapest by gates."""
    out: list[dict] = []
    for n in dataset.n_values():
        for order in dataset.orders():
            for threshold in thresholds:
                candidates = _candidates(dataset, n, order, threshold)
                if not candidates:
                    out.append({
                        "n": n, "order": order, "target_error": threshold,
                        "achieved": 0, "selected_by": "none", "r": None,
                        "cnot_count": None, "total_gates": None,
                        "circuit_depth": None, "two_qubit_depth": None,
                        "operator_norm_error": None, "note": NOT_ACHIEVED,
                    })
                    continue
                largest = max(row.r for row in dataset.series(n, order))
                for label, key in (("min_cnot", "cnot"), ("min_total_gates", "total")):
                    best = min(candidates, key=lambda row: (getattr(row, key), row.r))
                    out.append({
                        "n": n, "order": order, "target_error": threshold,
                        "achieved": 1, "selected_by": label, "r": best.r,
                        "cnot_count": best.cnot, "total_gates": best.total,
                        "circuit_depth": best.depth, "two_qubit_depth": best.twoq,
                        "operator_norm_error": best.err_op,
                        "note": "; ".join(filter(None, [
                            "at numerical floor" if best.at_floor else "",
                            "at largest measured r" if best.r == largest else ""])),
                    })
    return out


def write_csv(rows: list[dict], path: str) -> None:
    """Write the minimum-resource table exactly as computed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pair_by_threshold(rows: list[dict]) -> list[dict]:
    """Join the two selection modes into one comparison row per configuration.

    Returns ``n, order, threshold, cnot1/cnot2 (first/second order), gate
    savings, error reduction`` only where **both** formulas reach the target;
    otherwise ``achieved_by_both=False`` with the reason.
    """
    index = {}
    for row in rows:
        index.setdefault((row["n"], row["order"], row["target_error"]), {})[
            row["selected_by"]] = row

    out = []
    for n in sorted({row["n"] for row in rows}):
        for threshold in sorted({row["target_error"] for row in rows}, reverse=True):
            first = index.get((n, 1, threshold), {})
            second = index.get((n, 2, threshold), {})
            f_cnot = first.get("min_cnot")
            s_cnot = second.get("min_cnot")
            both = bool(f_cnot and s_cnot and f_cnot.get("achieved") and s_cnot.get("achieved"))
            entry = {
                "n": n, "target_error": threshold, "achieved_by_both": int(both),
                "cnot_order1": f_cnot["cnot_count"] if f_cnot and f_cnot["achieved"] else None,
                "cnot_order2": s_cnot["cnot_count"] if s_cnot and s_cnot["achieved"] else None,
                "gates_order1": f_cnot["total_gates"] if f_cnot and f_cnot["achieved"] else None,
                "gates_order2": s_cnot["total_gates"] if s_cnot and s_cnot["achieved"] else None,
                "r_order1": f_cnot["r"] if f_cnot and f_cnot["achieved"] else None,
                "r_order2": s_cnot["r"] if s_cnot and s_cnot["achieved"] else None,
                "cnot_savings_pct": None, "gate_savings_pct": None,
                "note": "",
            }
            if both:
                entry["cnot_savings_pct"] = 100.0 * (entry["cnot_order1"] - entry["cnot_order2"]) / entry["cnot_order1"]
                entry["gate_savings_pct"] = 100.0 * (entry["gates_order1"] - entry["gates_order2"]) / entry["gates_order1"]
            else:
                missing = [f"order {o}" for o, c in ((1, f_cnot), (2, s_cnot))
                           if not (c and c.get("achieved"))]
                entry["note"] = "not reached by " + ", ".join(missing)
            out.append(entry)
    return out


def summary_markdown(rows: list[dict]) -> str:
    """Human-readable table for the report."""
    lines = ["| n | formula | target | achieved | cheapest by | r | CNOT | total gates | depth | two-qubit depth | note |",
             "|---|---------|--------|----------|-------------|---|------|-------------|-------|-----------------|------|"]
    for row in rows:
        if not row["achieved"]:
            lines.append(f"| {row['n']} | {row['order']} | {row['target_error']:g} | "
                         f"no - {NOT_ACHIEVED} | - | - | - | - | - | - | |")
            continue
        lines.append(
            f"| {row['n']} | {row['order']} | {row['target_error']:g} | yes | "
            f"{row['selected_by']} | {row['r']} | "
            f"{row['cnot_count']} | {row['total_gates']} | {row['circuit_depth']} | "
            f"{row['two_qubit_depth']} | {row['note'] or ''} |")
    return "\n".join(lines)
