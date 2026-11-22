"""Loading, validation and description of Phase 1 circuit-resource results.

Phase 2 never recomputes the Trotter errors it analyses: every number comes from
``results/circuit_resources.csv`` (written by ``circuit_cost.py``) or, when that
file does not reach a required accuracy, from the *supplementary sweep* in
``results/phase2/supplementary_sweep.csv``, which is produced by the same
unchanged ``trotter.sweep`` call at larger ``r``.  No value is ever invented or
extrapolated into a table.

Schema (14 columns, as written by ``circuit_cost.write_csv``)::

    model, n, order, r, initial_state, operator_norm_error, state_infidelity,
    qubits, cnot_count, rx_count, rz_count, total_gates, circuit_depth,
    two_qubit_depth
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

REQUIRED_COLUMNS = (
    "model", "n", "order", "r", "initial_state",
    "operator_norm_error", "state_infidelity",
    "qubits", "cnot_count", "rx_count", "rz_count", "total_gates",
    "circuit_depth", "two_qubit_depth",
)

#: Phase 1 established that the numerical floor of this pipeline is ~1e-11
#: (plot F: minimum 2.50e-11 at r = 2.2e5).  Errors below this are round-off,
#: not accuracy, and are flagged rather than celebrated.
NUMERICAL_FLOOR = 1e-10

#: Floor used only for logarithmic axes; raw values are never modified.
PLOT_FLOOR = 1e-15

PHASE1_CSV = os.path.join("results", "circuit_resources.csv")
SUPPLEMENTARY_CSV = os.path.join("results", "phase2", "supplementary_sweep.csv")


class SchemaError(ValueError):
    """The CSV does not have the columns Phase 2 needs."""


@dataclass(frozen=True)
class Row:
    """One measured configuration: numerical error plus its circuit resources."""

    model: str
    n: int
    order: int
    r: int
    initial_state: str
    err_op: float
    err_inf: float
    qubits: int
    cnot: int
    rx: int
    rz: int
    total: int
    depth: int
    twoq: int
    source: str = "phase1"

    @property
    def at_floor(self) -> bool:
        return self.err_op <= NUMERICAL_FLOOR

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.n, self.order, self.r)


@dataclass
class Dataset:
    """All loaded rows plus any row-level problems found while parsing."""

    rows: list[Row] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    # -- selection ---------------------------------------------------------
    def n_values(self) -> list[int]:
        return sorted({row.n for row in self.rows})

    def orders(self) -> list[int]:
        return sorted({row.order for row in self.rows})

    def sources(self) -> list[str]:
        return sorted({row.source for row in self.rows})

    def subset(self, n: int | None = None, order: int | None = None,
               source: str | None = None) -> list[Row]:
        return [row for row in self.rows
                if (n is None or row.n == n)
                and (order is None or row.order == order)
                and (source is None or row.source == source)]

    def series(self, n: int, order: int) -> list[Row]:
        """Rows for one (n, order), sorted by r, duplicates removed by r."""
        seen: dict[int, Row] = {}
        for row in self.subset(n=n, order=order):
            seen.setdefault(row.r, row)
        return [seen[r] for r in sorted(seen)]

    def array(self, n: int, order: int, attribute: str) -> np.ndarray:
        return np.array([getattr(row, attribute) for row in self.series(n, order)],
                        dtype=float)

    def configs(self) -> list[tuple[int, int]]:
        return [(n, order) for n in self.n_values() for order in self.orders()]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def check_schema(path: str, required=REQUIRED_COLUMNS) -> list[str]:
    """Return the list of required columns missing from ``path`` (empty if fine)."""
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    return [name for name in required if name not in header]


def _to_int(value, name, line, problems):
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append({"line": line, "field": name, "value": value,
                         "reason": "not a number"})
        return None
    if not math.isfinite(number) or number != int(number):
        problems.append({"line": line, "field": name, "value": value,
                         "reason": "not an integer"})
        return None
    return int(number)


def _to_float(value, name, line, problems, minimum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append({"line": line, "field": name, "value": value,
                         "reason": "not a number"})
        return None
    if not math.isfinite(number):
        problems.append({"line": line, "field": name, "value": value,
                         "reason": "not finite"})
        return None
    if minimum is not None and number < minimum:
        problems.append({"line": line, "field": name, "value": value,
                         "reason": f"below {minimum}"})
        return None
    return number


def load_csv(path: str, source: str = "phase1") -> Dataset:
    """Read one results CSV, skipping and reporting malformed rows.

    A malformed row is never silently repaired: it is dropped and recorded in
    ``Dataset.problems`` with its line number, so the audit can report it.
    """
    missing = check_schema(path)
    if missing:
        raise SchemaError(f"{path} is missing required columns: {missing}")

    dataset = Dataset(files=[path])
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for line, raw in enumerate(reader, start=2):
            n = _to_int(raw["n"], "n", line, dataset.problems)
            order = _to_int(raw["order"], "order", line, dataset.problems)
            r = _to_int(raw["r"], "r", line, dataset.problems)
            err_op = _to_float(raw["operator_norm_error"], "operator_norm_error",
                               line, dataset.problems, minimum=0.0)
            err_inf = _to_float(raw["state_infidelity"], "state_infidelity",
                                line, dataset.problems, minimum=0.0)
            if None in (n, order, r, err_op, err_inf):
                continue
            if order not in (1, 2):
                dataset.problems.append({"line": line, "field": "order",
                                         "value": order, "reason": "not 1 or 2"})
                continue
            if n is None or n < 2 or r < 1:
                dataset.problems.append({"line": line, "field": "n/r",
                                         "value": f"{n}/{r}",
                                         "reason": "out of range"})
                continue

            ints = {}
            for name in ("qubits", "cnot_count", "rx_count", "rz_count",
                         "total_gates", "circuit_depth", "two_qubit_depth"):
                ints[name] = _to_int(raw[name], name, line, dataset.problems)
            if any(value is None for value in ints.values()):
                continue
            if ints["qubits"] != n:
                dataset.problems.append({"line": line, "field": "qubits",
                                         "value": ints["qubits"],
                                         "reason": f"does not match n={n}"})
                continue
            total = ints["cnot_count"] + ints["rx_count"] + ints["rz_count"]
            if total != ints["total_gates"]:
                dataset.problems.append({"line": line, "field": "total_gates",
                                         "value": ints["total_gates"],
                                         "reason": f"CNOT+RX+RZ = {total}"})
                continue

            dataset.rows.append(Row(
                model=raw["model"], n=n, order=order, r=r,
                initial_state=raw["initial_state"], err_op=err_op, err_inf=err_inf,
                qubits=ints["qubits"], cnot=ints["cnot_count"], rx=ints["rx_count"],
                rz=ints["rz_count"], total=ints["total_gates"],
                depth=ints["circuit_depth"], twoq=ints["two_qubit_depth"],
                source=source))
    return dataset


def load_dataset(phase1: str = PHASE1_CSV,
                 supplementary: str | None = SUPPLEMENTARY_CSV,
                 use_supplementary: bool = True) -> Dataset:
    """Load Phase 1 results and, if present, the supplementary sweep.

    The supplementary file only ever *adds* rows at larger ``r``; it is a real
    measurement with the unchanged propagator, not an interpolation.
    """
    dataset = load_csv(phase1, source="phase1")
    if use_supplementary and supplementary and os.path.exists(supplementary):
        extra = load_csv(supplementary, source="supplementary")
        known = {row.key for row in dataset.rows}
        for row in extra.rows:
            if row.key in known:          # never let the two files fight
                dataset.problems.append({"line": -1, "field": "duplicate",
                                         "value": str(row.key),
                                         "reason": "already in Phase 1 data"})
                continue
            dataset.rows.append(row)
        dataset.problems.extend(extra.problems)
        dataset.files.extend(extra.files)
    return dataset


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(dataset: Dataset) -> dict:
    """Describe what the data can and cannot support, without assuming anything."""
    configs = {}
    for n, order in dataset.configs():
        series = dataset.series(n, order)
        if not series:
            continue
        err = np.array([row.err_op for row in series])
        steps = np.array([row.r for row in series])
        increments = np.diff(err)
        configs[f"n={n},order={order}"] = {
            "rows": len(series),
            "r_min": int(steps.min()),
            "r_max": int(steps.max()),
            "error_first": float(err[0]),
            "error_last": float(err[-1]),
            "error_min": float(err.min()),
            "strictly_decreasing": bool(np.all(increments < 0)),
            "nonmonotone_steps": int(np.sum(increments >= 0)),
            "at_numerical_floor": int(np.sum(err <= NUMERICAL_FLOOR)),
            "sources": sorted({row.source for row in series}),
        }
    return {
        "phase1_file": PHASE1_CSV,
        "rows": len(dataset.rows),
        "files": dataset.files,
        "n_values": dataset.n_values(),
        "orders": dataset.orders(),
        "sources": dataset.sources(),
        "r_values": sorted({row.r for row in dataset.rows}),
        "row_problems": dataset.problems,
        "every_row_has_resources": all(
            row.cnot + row.rx + row.rz == row.total for row in dataset.rows),
        "error_metric": "operator_norm_error",
        "depth_definition": "estimated logical circuit depth (conservative ASAP scheduler)",
        "counting_convention": "CNOT + RX + RZ (no barriers, identity or measurement)",
        "numerical_floor": NUMERICAL_FLOOR,
        "configs": configs,
        "thresholds": [1e-1, 1e-2, 1e-3, 1e-4],
        "threshold_reachable": {
            f"{thr:g}": [n for n in dataset.n_values()
                        if any(row.err_op <= thr for row in dataset.subset(n=n))]
            for thr in (1e-1, 1e-2, 1e-3, 1e-4)
        },
    }


# ---------------------------------------------------------------------------
# Small numerical helpers shared by every experiment
# ---------------------------------------------------------------------------

def safe_log(values, floor: float = PLOT_FLOOR) -> np.ndarray:
    """Copy of ``values`` with zeros/non-positives clipped, for plotting only.

    The input array is never modified; the caller keeps the raw numbers.
    """
    array = np.asarray(values, dtype=float)
    return np.maximum(array, floor)


def fit_loglog(x, y, drop_first: bool = True, floor: float = 1e-13):
    """Least-squares slope of log(y) against log(x), with a mask.

    Mirrors the Phase 1 convention: the smallest x is dropped (pre-asymptotic)
    and points at or below the floating-point floor are excluded.  Returns
    ``(slope, mask)``; slope is NaN when fewer than three points survive.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (y > floor) & (x > 0)
    if drop_first and len(x) > 1:
        mask &= x > x.min()
    if mask.sum() < 3:
        return float("nan"), mask
    slope = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]
    return float(slope), mask
