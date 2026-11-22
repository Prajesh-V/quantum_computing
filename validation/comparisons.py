"""Audit of the first-order / second-order comparisons.

Three comparisons are kept strictly apart:

* **equal step count** -- same ``r``, so the second-order circuit is larger;
  this comparison flatters first order and is reported as such;
* **equal CNOT budget** -- each second-order point is matched to the most
  expensive first-order point whose CNOT count does not exceed it;
* **equal total-gate budget** -- the same rule on the total gate count.

Every pair must agree on system size, noise model, both noise probabilities,
the global probability and the total evolution time.  The pairing is
re-implemented here from scratch and then compared with
``noise.experiments.pair_orders``, which is how the corrected Phase 3 logic is
independently verified.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NOISE_KEYS",
    "pair_matched",
    "equal_step_pairs",
    "summarise",
    "compare_with_phase3",
]

NOISE_KEYS = ("noise_model", "single_qubit_error", "two_qubit_error",
              "global_noise_probability")

WIN_ORDER2 = "second_order"
WIN_ORDER1 = "first_order"
WIN_TIE = "tie"


def _key(row):
    return (row["system_size"], row["trotter_order"], row["total_time"]) + tuple(
        row[name] for name in NOISE_KEYS)


def _noise_key(row):
    return tuple(row[name] for name in NOISE_KEYS)


def _winner(row1, row2) -> str:
    """Second order wins only with a strictly smaller total error."""
    if row2["total_error"] < row1["total_error"]:
        return WIN_ORDER2
    if row2["total_error"] > row1["total_error"]:
        return WIN_ORDER1
    return WIN_TIE


def pair_matched(rows, metric: str = "cnot") -> list[dict]:
    """Independent reimplementation of the matched-budget pairing.

    ``metric`` is ``"cnot"`` or ``"total_gate"``.  For each second-order row the
    first-order row with the largest budget not exceeding the second-order
    budget is selected, always within the same system size and noise settings.
    """
    column = {"cnot": "cnot_count", "total_gate": "total_gate_count"}[metric]
    first: dict[tuple, list[dict]] = {}
    for row in rows:
        if int(row["trotter_order"]) == 1:
            first.setdefault((int(row["system_size"]),) + _noise_key(row), []).append(row)
    for bucket in first.values():
        bucket.sort(key=lambda row: (row[column], row["trotter_steps"]))

    pairs: list[dict] = []
    second = [row for row in rows if int(row["trotter_order"]) == 2]
    for row2 in sorted(second, key=lambda row: (row["system_size"],
                                                row["trotter_steps"],
                                                _noise_key(row))):
        bucket = first.get((int(row2["system_size"]),) + _noise_key(row2))
        if not bucket:
            continue
        budget = row2[column]
        feasible = [row for row in bucket if row[column] <= budget]
        row1 = feasible[-1] if feasible else bucket[0]
        pairs.append({
            "comparison": f"matched_{metric}",
            "system_size": int(row2["system_size"]),
            "noise_model": row2["noise_model"],
            "single_qubit_error": row2["single_qubit_error"],
            "two_qubit_error": row2["two_qubit_error"],
            "global_noise_probability": row2["global_noise_probability"],
            "total_time": row2["total_time"],
            "order1_r": int(row1["trotter_steps"]),
            "order2_r": int(row2["trotter_steps"]),
            "order1_budget": row1[column],
            "order2_budget": row2[column],
            "order1_error": row1["total_error"],
            "order2_error": row2["total_error"],
            "error_difference": row2["total_error"] - row1["total_error"],
            "order1_fidelity": row1["fidelity"],
            "order2_fidelity": row2["fidelity"],
            "within_budget": bool(feasible),
            "winner": _winner(row1, row2),
        })
    return pairs


def equal_step_pairs(rows) -> list[dict]:
    """Same ``r``, same noise, same total time: the comparison that flatters order 1."""
    index = {}
    for row in rows:
        if int(row["trotter_order"]) == 1:
            index[(int(row["system_size"]), row["trotter_steps"]) + _noise_key(row)] = row
    pairs = []
    for row2 in rows:
        if int(row2["trotter_order"]) != 2:
            continue
        row1 = index.get((int(row2["system_size"]), row2["trotter_steps"])
                         + _noise_key(row2))
        if row1 is None:
            continue
        pairs.append({
            "comparison": "equal_step",
            "system_size": int(row2["system_size"]),
            "noise_model": row2["noise_model"],
            "single_qubit_error": row2["single_qubit_error"],
            "two_qubit_error": row2["two_qubit_error"],
            "global_noise_probability": row2["global_noise_probability"],
            "total_time": row2["total_time"],
            "order1_r": int(row1["trotter_steps"]),
            "order2_r": int(row2["trotter_steps"]),
            "order1_budget": row1["cnot_count"],
            "order2_budget": row2["cnot_count"],
            "order1_error": row1["total_error"],
            "order2_error": row2["total_error"],
            "error_difference": row2["total_error"] - row1["total_error"],
            "order1_fidelity": row1["fidelity"],
            "order2_fidelity": row2["fidelity"],
            "within_budget": True,
            "winner": _winner(row1, row2),
        })
    return pairs


def summarise(pairs) -> dict:
    """Win counts overall, per noise level and per system size."""
    def tally(subset):
        wins2 = sum(1 for pair in subset if pair["winner"] == WIN_ORDER2)
        wins1 = sum(1 for pair in subset if pair["winner"] == WIN_ORDER1)
        ties = sum(1 for pair in subset if pair["winner"] == WIN_TIE)
        return {"pairs": len(subset), "second_order": wins2, "first_order": wins1,
                "ties": ties}

    levels = sorted({round(pair["single_qubit_error"], 12) for pair in pairs})
    per_level = {}
    for level in levels:
        subset = [pair for pair in pairs if round(pair["single_qubit_error"], 12) == level]
        per_level[level] = tally(subset)
    per_size = {}
    for size in sorted({pair["system_size"] for pair in pairs}):
        subset = [pair for pair in pairs if pair["system_size"] == size]
        per_size[size] = tally(subset)

    extras = [pair for pair in pairs if pair["order1_r"] > pair["order2_r"]]
    return {
        "total": tally(pairs),
        "per_noise_level": per_level,
        "per_system_size": per_size,
        "extra_step_pairs": len(extras),
        "wins_at_r_le_8": sum(1 for pair in pairs
                              if pair["winner"] == WIN_ORDER2 and pair["order2_r"] <= 8),
        "max_r_of_win": max((pair["order2_r"] for pair in pairs
                             if pair["winner"] == WIN_ORDER2), default=None),
        "progression": [per_level[level]["second_order"] for level in levels],
        "progression_total": sum(per_level[level]["second_order"] for level in levels),
    }


def compare_with_phase3(pairs, phase3_pairs) -> dict:
    """Do my independently paired rows agree with ``noise.experiments.pair_orders``?"""
    def index(items, keys):
        out = {}
        for item in items:
            out[tuple(item[key] for key in keys)] = item
        return out

    keys = ("system_size", "order1_r", "order2_r", "single_qubit_error")
    mine = index(pairs, keys)
    theirs = index(phase3_pairs, keys)
    shared = sorted(set(mine) & set(theirs))

    def phase3_errors(pair):
        """``noise.experiments.pair_orders`` stores the two errors separately."""
        return (pair["total_error_order1"], pair["total_error_order2"])

    def phase3_winner(pair):
        error1, error2 = phase3_errors(pair)
        return WIN_ORDER2 if error2 < error1 else (WIN_ORDER1 if error2 > error1 else WIN_TIE)

    differences = [abs(mine[key]["error_difference"]
                       - (phase3_errors(theirs[key])[1] - phase3_errors(theirs[key])[0]))
                   for key in shared]
    win_disagreements = [key for key in shared
                         if mine[key]["winner"] != phase3_winner(theirs[key])]
    return {
        "my_pairs": len(pairs),
        "phase3_pairs": len(phase3_pairs),
        "shared": len(shared),
        "only_mine": len(set(mine) - set(theirs)),
        "only_phase3": len(set(theirs) - set(mine)),
        "max_error_difference": float(max(differences)) if differences else 0.0,
        "win_disagreements": len(win_disagreements),
        "keys_only_mine": sorted(set(mine) - set(theirs))[:10],
        "keys_only_phase3": sorted(set(theirs) - set(mine))[:10],
    }
