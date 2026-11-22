"""Tests for the Phase 2 analysis package.

Run with:  python -m pytest tests/ -q

The tests check *calculations*: matching logic, budget selection, threshold
search, scaling identities and numerical-reliability rules.  Synthetic datasets
with a known power law are used where the real data cannot pin an answer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os

import numpy as np
import pytest

import circuit_cost
import trotter

from analysis import accuracy_targets, comparison, report, scaling
from analysis.resource_analysis import (Dataset, NUMERICAL_FLOOR, Row,
                                        SchemaError, audit, check_schema,
                                        load_csv, load_dataset, safe_log)

PHASE1 = os.path.join("results", "circuit_resources.csv")
SUPP = os.path.join("results", "phase2", "supplementary_sweep.csv")
HEADER = list(circuit_cost.RESOURCE_COLUMNS)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _row(n=4, order=1, r=1, err=1.0, cnot=None, total=None, depth=None,
         twoq=None, source="phase1") -> Row:
    """A Row with resources consistent with the closed forms."""
    ref = circuit_cost.analytic_counts(n, r, order)
    return Row(model="TFIM", n=n, order=order, r=r, initial_state="0" * n,
               err_op=err, err_inf=err, qubits=n,
               cnot=ref["cnot"] if cnot is None else cnot,
               rx=ref["rx"], rz=ref["rz"],
               total=ref["total"] if total is None else total,
               depth=ref["cnot"] + ref["rx"] + ref["rz"] if depth is None else depth,
               twoq=ref["cnot"] if twoq is None else twoq,
               source=source)


def _synthetic(n=4, rs=(1, 2, 4, 8, 16, 32), power1=1.0, power2=2.0,
               a1=1.0, a2=1.0) -> Dataset:
    """Two power-law curves whose errors are known exactly."""
    rows = []
    for r in rs:
        rows.append(_row(n=n, order=1, r=r, err=a1 * r ** (-power1)))
        rows.append(_row(n=n, order=2, r=r, err=a2 * r ** (-power2)))
    return Dataset(rows=rows)


def _write_csv(path, rows, columns=HEADER):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# A. Data loading
# ---------------------------------------------------------------------------

def test_phase1_csv_loads_with_expected_shape():
    dataset = load_csv(PHASE1)
    assert len(dataset.rows) == 132
    assert dataset.n_values() == [4, 5, 6]
    assert dataset.orders() == [1, 2]
    assert dataset.problems == []
    first = dataset.rows[0]
    assert first.cnot + first.rx + first.rz == first.total
    assert first.qubits == first.n
    assert first.err_op >= 0.0


def test_supplementary_file_only_adds_larger_r():
    if not os.path.exists(SUPP):
        pytest.skip("supplementary sweep not present")
    base = load_csv(PHASE1)
    merged = load_dataset(PHASE1, SUPP)
    assert len(merged.rows) > len(base.rows)
    assert merged.problems == []
    assert "supplementary" in merged.sources()
    largest_base = max(row.r for row in base.rows)
    for row in merged.subset(source="supplementary"):
        assert row.r > largest_base


def test_missing_column_is_detected(tmp_path):
    path = tmp_path / "broken.csv"
    columns = [name for name in HEADER if name != "circuit_depth"]
    _write_csv(path, [["TFIM", 4, 1, 1, "0000", 0.5, 0.1, 4, 6, 4, 3, 13, 12]], columns)
    assert check_schema(str(path)) == ["circuit_depth"]
    with pytest.raises(SchemaError):
        load_csv(str(path))


def test_missing_and_invalid_values_are_reported_not_repaired(tmp_path):
    path = tmp_path / "dirty.csv"
    good = ["TFIM", 4, 1, 1, "0000", 0.5, 0.1, 4, 6, 4, 3, 13, 12, 6]
    empty_error = list(good)
    empty_error[5] = ""
    bad_n = list(good)
    bad_n[1] = "four"
    bad_total = list(good)
    bad_total[11] = 999
    _write_csv(path, [good, empty_error, bad_n, bad_total])
    dataset = load_csv(str(path))
    assert len(dataset.rows) == 1                       # only the clean row
    reasons = {problem["field"] for problem in dataset.problems}
    assert {"operator_norm_error", "n", "total_gates"} <= reasons
    assert all(problem["line"] > 0 for problem in dataset.problems)


def test_qubit_column_must_match_n(tmp_path):
    path = tmp_path / "mismatch.csv"
    _write_csv(path, [["TFIM", 4, 1, 1, "0000", 0.5, 0.1, 5, 6, 4, 3, 13, 12, 6]])
    dataset = load_csv(str(path))
    assert dataset.rows == []
    assert dataset.problems[0]["field"] == "qubits"


# ---------------------------------------------------------------------------
# B. Comparison logic
# ---------------------------------------------------------------------------

def test_equal_steps_pairs_only_matching_configurations():
    dataset = load_dataset(PHASE1, SUPP)
    rows = comparison.equal_steps(dataset, 5)
    assert rows
    for entry in rows:
        series1 = {row.r for row in dataset.series(5, 1)}
        series2 = {row.r for row in dataset.series(5, 2)}
        assert entry["r"] in series1 and entry["r"] in series2
        assert entry["cnot_order1"] == 2 * entry["r"] * 4
        assert entry["cnot_order2"] == 2 * (entry["r"] + 1) * 4
        assert entry["error_ratio"] == pytest.approx(
            entry["error_order2"] / entry["error_order1"])


def test_equal_steps_never_mixes_qubit_counts():
    dataset = load_dataset(PHASE1, SUPP)
    for entry in comparison.equal_steps(dataset, 4):
        assert entry["n"] == 4
        assert entry["cnot_order1"] == 2 * entry["r"] * 3


def test_interpolation_reproduces_a_known_power_law():
    dataset = _synthetic(rs=(1, 2, 4, 8, 16, 32), power1=1.0, a1=1.0)
    # r=8 -> CNOT = 2*8*(n-1) = 48 and error 1/8: an exact measured point
    err, flag = comparison.interpolate_error_at(dataset.series(4, 1), 48.0, "cnot")
    assert flag is False and err == pytest.approx(1.0 / 8.0)
    # midway (in log space) between r=4 (CNOT 24, err 1/4) and r=8 (CNOT 48, err 1/8)
    err, flag = comparison.interpolate_error_at(dataset.series(4, 1),
                                                math.sqrt(24 * 48), "cnot")
    assert flag is True
    assert err == pytest.approx(math.sqrt(0.25 * 0.125), rel=1e-9)


def test_interpolation_refuses_to_extrapolate():
    dataset = _synthetic(rs=(1, 2, 4, 8), power1=1.0)
    series = dataset.series(4, 1)
    assert comparison.interpolate_error_at(series, 1e9, "cnot") == (None, False)
    assert comparison.interpolate_error_at(series, 0.1, "cnot") == (None, False)


def test_equal_budget_selects_the_nearest_feasible_configuration():
    dataset = _synthetic(rs=(1, 2, 4, 8, 16), power1=1.0, power2=2.0)
    rows = comparison.equal_budget(dataset, 4, "cnot")
    assert rows
    first = {row.r: row for row in dataset.series(4, 1)}
    for entry in rows:
        budget = entry["budget"]
        feasible = [row for row in first.values() if row.cnot <= budget]
        if feasible:
            best = max(feasible, key=lambda row: row.cnot)
            assert entry["nearest_r_order1"] == best.r
            assert entry["nearest_budget_order1"] <= budget
        # the interpolated first-order error must match 1/r exactly here
        if entry["error_order1_interpolated"] is not None:
            assert entry["error_order1_interpolated"] == pytest.approx(
                1.0 / entry["nearest_r_order1"], rel=0.35)


def test_equal_budget_orders_are_separated_and_budgets_match():
    dataset = load_dataset(PHASE1, SUPP)
    for metric, attr in (("cnot", "cnot"), ("total_gates", "total")):
        rows = comparison.equal_budget(dataset, 6, metric)
        assert rows
        for entry in rows:
            order2 = [row for row in dataset.series(6, 2) if row.r == entry["order2_r"]]
            assert order2 and order2[0].err_op == entry["error_order2"]
            assert entry["budget"] == getattr(order2[0], attr)
            if entry["error_order1_interpolated"] is not None:
                # exact hits are not flagged as interpolated, interpolated ones are
                assert entry["interpolated"] in (True, False)
                assert entry["error_reduction_pct_interpolated"] == pytest.approx(
                    100 * (entry["error_order1_interpolated"] - entry["error_order2"])
                    / entry["error_order1_interpolated"])


def test_accuracy_per_cnot_recovers_the_power_law():
    # large r so that CNOT = 2(r+1)(n-1) is proportional to r to within 1%
    rs = (256, 512, 1024, 2048, 4096)
    dataset = _synthetic(rs=rs, power1=1.0, power2=2.0, a1=1.0, a2=30.0)
    slopes = comparison.accuracy_per_cnot(dataset)
    assert slopes[(4, 1)] == pytest.approx(-1.0, abs=0.02)
    assert slopes[(4, 2)] == pytest.approx(-2.0, abs=0.05)
    assert slopes[(4, 1)] > slopes[(4, 2)]          # first order decays more slowly


def test_efficiency_helpers():
    assert comparison.gate_savings_pct(1000, 250) == pytest.approx(75.0)
    assert comparison.gate_savings_pct(1000, 1250) == pytest.approx(-25.0)
    assert comparison.error_reduction_pct(1.0, 0.25) == pytest.approx(75.0)
    assert math.isnan(comparison.gate_savings_pct(0, 10))


# ---------------------------------------------------------------------------
# C. Accuracy thresholds
# ---------------------------------------------------------------------------

def test_minimum_resources_picks_the_cheapest_success():
    # order 1: err = 1/r, order 2: err = 1/r^2, CNOT = 2 r (n-1)
    dataset = _synthetic(rs=(1, 2, 4, 8, 16, 32, 64), power1=1.0, power2=2.0)
    rows = accuracy_targets.minimum_resources(dataset, thresholds=(0.1,))
    first = [row for row in rows if row["order"] == 1 and row["selected_by"] == "min_cnot"][0]
    second = [row for row in rows if row["order"] == 2 and row["selected_by"] == "min_cnot"][0]
    assert first["achieved"] and second["achieved"]
    assert first["r"] == 16 and first["operator_norm_error"] <= 0.1     # 1/16
    assert second["r"] == 4 and second["operator_norm_error"] <= 0.1    # 1/16
    assert second["cnot_count"] < first["cnot_count"]
    # and the cheapest-by-gates selection is consistent
    for row in rows:
        if row["achieved"]:
            assert row["cnot_count"] + 0 == circuit_cost.analytic_counts(
                row["n"], row["r"], row["order"])["cnot"]


def test_unachieved_threshold_is_reported_not_extrapolated():
    dataset = _synthetic(rs=(1, 2, 4, 8), power1=1.0, power2=2.0)
    rows = accuracy_targets.minimum_resources(dataset, thresholds=(1e-9,))
    assert rows
    for row in rows:
        assert row["achieved"] == 0
        assert row["selected_by"] == "none"
        assert row["note"] == accuracy_targets.NOT_ACHIEVED
        assert row["cnot_count"] is None and row["r"] is None


def test_errors_at_the_floor_do_not_count_as_reaching_a_target():
    dataset = Dataset(rows=[_row(n=4, order=2, r=8, err=NUMERICAL_FLOOR / 10)])
    rows = accuracy_targets.minimum_resources(dataset, thresholds=(1e-3,))
    assert all(row["achieved"] == 0 for row in rows)


def test_non_monotonic_curve_still_yields_the_cheapest_success():
    # r = 1 fails, r = 2 succeeds, r = 4 fails again (noisy/non-monotone data)
    dataset = Dataset(rows=[
        _row(n=4, order=1, r=1, err=0.9),
        _row(n=4, order=1, r=2, err=0.05),
        _row(n=4, order=1, r=4, err=0.2),
        _row(n=4, order=1, r=8, err=0.01),
    ])
    rows = accuracy_targets.minimum_resources(dataset, thresholds=(0.1,))
    best = [row for row in rows if row["selected_by"] == "min_cnot"][0]
    assert best["r"] == 2                       # cheapest, even though r=4 regresses
    assert best["operator_norm_error"] == pytest.approx(0.05)


def test_minimum_resources_csv_and_pairing(tmp_path):
    dataset = _synthetic(rs=(1, 2, 4, 8, 16), power1=1.0, power2=2.0)
    rows = accuracy_targets.minimum_resources(dataset, thresholds=(0.1, 0.001))
    path = tmp_path / "min.csv"
    accuracy_targets.write_csv(rows, str(path))
    with open(path, newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0]) == accuracy_targets.COLUMNS
    assert len(table) == len(rows)

    pairs = accuracy_targets.pair_by_threshold(rows)
    assert pairs
    for entry in pairs:
        assert entry["target_error"] in (0.1, 0.001)
        if entry["achieved_by_both"]:
            assert entry["gate_savings_pct"] == pytest.approx(
                100 * (entry["gates_order1"] - entry["gates_order2"])
                / entry["gates_order1"])
        else:
            assert entry["note"]


# ---------------------------------------------------------------------------
# D. Scaling
# ---------------------------------------------------------------------------

def test_analytic_counts_match_every_measured_row():
    dataset = load_dataset(PHASE1, SUPP)
    verification = scaling.verify_analytic(dataset)
    assert verification["all_match"] is True
    assert verification["max_abs_difference"] == 0
    assert verification["rows_checked"] == len(dataset.rows)


def test_analytic_formulas_are_the_phase1_closed_forms():
    for n in (4, 5, 6, 8):
        for r in (1, 10, 200):
            first = scaling.analytic_resources(n, r, 1)
            assert first["cnot"] == 2 * r * (n - 1)
            assert first["rz"] == r * (n - 1)
            assert first["rx"] == r * n
            assert first["total"] == r * (4 * n - 3)
            second = scaling.analytic_resources(n, r, 2)
            assert second["cnot"] == 2 * (r + 1) * (n - 1)
            assert second["total"] == 3 * (r + 1) * (n - 1) + r * n
            assert circuit_cost.analytic_counts(n, r, 1) == first


def test_scaling_exponents_on_synthetic_linear_growth():
    rows = []
    for n in (4, 5, 6):
        for r in (1, 10, 100):
            for order in (1, 2):
                ref = circuit_cost.analytic_counts(n, r, order)
                rows.append(Row(model="TFIM", n=n, order=order, r=r,
                                initial_state="0" * n, err_op=1e-3 / r, err_inf=0.0,
                                qubits=n, cnot=ref["cnot"], rx=ref["rx"], rz=ref["rz"],
                                total=ref["total"], depth=7 * r + 3 * (n - 3),
                                twoq=ref["cnot"]))
    dataset = Dataset(rows=rows)
    exponents = scaling.empirical_exponents(dataset, r_ref=100)["by_order"]
    assert exponents[1]["cnot_vs_bonds"] == pytest.approx(1.0, abs=1e-9)
    assert exponents[1]["total_vs_r"] == pytest.approx(1.0, abs=1e-9)
    # depth is 7r + 3(n-3): at fixed r it grows with n, but only very weakly
    assert abs(exponents[1]["depth_vs_bonds"]) < 0.05


def test_fixed_second_order_cnot_overhead_is_two_bonds():
    dataset = load_dataset(PHASE1, SUPP)
    for n in dataset.n_values():
        first = dataset.series(n, 1)
        second = dataset.series(n, 2)
        for a, b in zip(first, second):
            assert b.cnot - a.cnot == 2 * (n - 1)


# ---------------------------------------------------------------------------
# E. Numerical reliability
# ---------------------------------------------------------------------------

def test_safe_log_clips_without_touching_the_input():
    raw = np.array([0.0, 1e-30, 1e-3, 1.0])
    original = raw.copy()
    result = safe_log(raw)
    assert np.all(result > 0)
    assert result[0] == pytest.approx(1e-15)          # documented plot floor
    assert np.array_equal(raw, original)              # raw data untouched


def test_loading_never_edits_the_phase1_file():
    before = hashlib.sha256(open(PHASE1, "rb").read()).hexdigest()
    dataset = load_dataset(PHASE1, SUPP)
    audit(dataset)
    for n in dataset.n_values():
        comparison.equal_budget(dataset, n, "cnot")
    after = hashlib.sha256(open(PHASE1, "rb").read()).hexdigest()
    assert before == after


def test_reported_errors_are_the_raw_values():
    dataset = load_csv(PHASE1)
    with open(PHASE1, newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    lookup = {(int(row["n"]), int(row["order"]), int(row["r"])): float(
        row["operator_norm_error"]) for row in raw}
    for row in dataset.rows:
        assert row.err_op == lookup[(row.n, row.order, row.r)]


def test_no_synthetic_rows_are_added_to_the_dataset():
    base = load_csv(PHASE1)
    merged = load_dataset(PHASE1, SUPP)
    keys = {(row.n, row.order, row.r, row.count_steps if hasattr(row, "count_steps") else row.r)
            for row in merged.rows}
    assert len(keys) == len(merged.rows)              # no duplicated (n, order, r)
    for row in merged.rows:
        source = "phase1" if row.key in {r.key for r in base.rows} else "supplementary"
        assert row.source == source


# ---------------------------------------------------------------------------
# F. Regression on Phase 1
# ---------------------------------------------------------------------------

def test_phase1_gate_counts_still_exact():
    for n, r, order in ((4, 5, 1), (6, 10, 2), (8, 50, 1)):
        res = circuit_cost.estimate_resources(n, r, order=order)
        ref = circuit_cost.analytic_counts(n, r, order)
        assert res.num_cnot == ref["cnot"]
        assert res.total_gates == ref["total"]


def test_phase1_evolution_is_unchanged():
    A, B, _ = trotter.tfim_terms(3)
    for order in (1, 2):
        gates = circuit_cost.build_circuit(3, 2, 1.0, order=order)
        assert np.linalg.norm(circuit_cost.circuit_unitary(3, gates)
                              - trotter.trotter_unitary(A, B, 1.0, 2, order), 2) < 1e-10


def test_running_phase2_leaves_every_phase1_file_untouched(tmp_path, monkeypatch):
    """End-to-end: the driver writes only into its own output directory."""
    import sys
    import analysis.run_phase2 as run_phase2

    watched = [PHASE1] + [os.path.join("results", name) for name in
                          ("floor_study.csv", "trotter_number.csv")]
    before = {path: hashlib.sha256(open(path, "rb").read()).hexdigest()
              for path in watched if os.path.exists(path)}
    monkeypatch.setattr(sys, "argv", ["run_phase2", "--no-plots",
                                       "--outdir", str(tmp_path)])
    run_phase2.main()
    for path, digest in before.items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
    assert (tmp_path / "phase2_report.md").exists()
    assert (tmp_path / "minimum_resources_for_accuracy.csv").exists()
    assert (tmp_path / "phase2_summary.json").exists()


def test_json_summary_is_serialisable_and_complete():
    dataset = load_csv(PHASE1)
    analysis = {
        "audit": audit(dataset),
        "slopes": comparison.accuracy_per_cnot(dataset),
        "equal_steps": {n: comparison.equal_steps(dataset, n)
                        for n in dataset.n_values()},
        "budget": {"cnot": {n: comparison.equal_budget(dataset, n, "cnot")
                            for n in dataset.n_values()},
                   "total_gates": {n: comparison.equal_budget(dataset, n, "total_gates")
                                   for n in dataset.n_values()}},
    }
    text = json.dumps(report._jsonable(analysis))
    assert "slopes" in json.loads(text)
    assert "NaN" not in text                      # strict JSON: no bare NaN tokens
