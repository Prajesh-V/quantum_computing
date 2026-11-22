"""Small file helpers shared by the audit modules."""

from __future__ import annotations

import csv
import json
import os

__all__ = ["write_csv", "write_json", "read_csv", "ensure_dir", "load_results"]


def ensure_dir(path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def write_csv(path: str, columns, rows) -> str:
    """Write ``rows`` (dicts) with the given column order."""
    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _value(row.get(key)) for key in columns})
    return path


def _value(value):
    if isinstance(value, float):
        return f"{value:.12e}"
    if isinstance(value, bool):
        return int(value)
    return value


def read_csv(path: str) -> list[dict]:
    """Read a CSV, converting every field to float when it looks numeric."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                converted[key] = value
                continue
            converted[key] = number
        out.append(converted)
    return out


def load_results(directory: str = "results/phase3") -> dict:
    """Load the stored Phase 3 results and normalise the noise-model column.

    ``gate_noise_results.csv`` has no ``noise_model`` column (every row in it is
    gate-level noise), while the global file does.  Adding the missing label
    here keeps every downstream grouping key uniform.
    """
    gate = read_csv(f"{directory}/gate_noise_results.csv")
    for row in gate:
        row.setdefault("noise_model", "gate")
        row.setdefault("global_noise_probability", 0.0)
    glob = read_csv(f"{directory}/global_noise_results.csv")
    for row in glob:
        row.setdefault("noise_model", "global_per_step")
        row.setdefault("single_qubit_error", 0.0)
        row.setdefault("two_qubit_error", 0.0)
    return {
        "gate": gate,
        "global": glob,
        "optimal": read_csv(f"{directory}/optimal_r.csv"),
        "sensitivity": read_csv(f"{directory}/noise_sensitivity.csv"),
    }


def write_json(path: str, payload) -> str:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path
