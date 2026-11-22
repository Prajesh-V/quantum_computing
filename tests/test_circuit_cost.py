"""Tests for circuit_cost.py (Phase 1: gate-level resource estimation).

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

import circuit_cost as cc
import trotter


# ---------------------------------------------------------------------------
# A. Analytical gate counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,r", [(2, 1), (4, 5), (6, 10), (8, 50)])
def test_first_order_counts_match_literal_formulas(n, r):
    """CNOT = 2r(n-1), RZ = r(n-1), RX = rn, total = r(4n-3)."""
    res = cc.estimate_resources(n, r, order=1)
    assert res.num_cnot == 2 * r * (n - 1)
    assert res.num_rz == r * (n - 1)
    assert res.num_rx == r * n
    assert res.total_gates == r * (4 * n - 3)
    assert res.total_gates == res.num_cnot + res.num_rz + res.num_rx


@pytest.mark.parametrize("n,r", [(2, 1), (4, 5), (6, 10)])
def test_second_order_counts_with_merging(n, r):
    """Merged: CNOT = 2(r+1)(n-1), RZ = (r+1)(n-1), RX = rn."""
    res = cc.estimate_resources(n, r, order=2, merge_half_steps=True)
    assert res.num_zz_layers == r + 1          # not 2r
    assert res.num_cnot == 2 * (r + 1) * (n - 1)
    assert res.num_rz == (r + 1) * (n - 1)
    assert res.num_rx == r * n
    assert res.total_gates == 3 * (r + 1) * (n - 1) + r * n


@pytest.mark.parametrize("n,r", [(2, 1), (4, 5), (6, 10)])
def test_second_order_counts_without_merging(n, r):
    """Unmerged: 2r ZZ layers -> CNOT = 4r(n-1), RZ = 2r(n-1), total = r(7n-6)."""
    res = cc.estimate_resources(n, r, order=2, merge_half_steps=False)
    assert res.num_zz_layers == 2 * r
    assert res.num_cnot == 4 * r * (n - 1)
    assert res.num_rz == 2 * r * (n - 1)
    assert res.num_rx == r * n
    assert res.total_gates == r * (7 * n - 6)


def test_merging_removes_exactly_r_minus_one_zz_layers():
    n, r = 6, 17
    merged = cc.estimate_resources(n, r, order=2, merge_half_steps=True)
    split = cc.estimate_resources(n, r, order=2, merge_half_steps=False)
    assert split.num_zz_layers - merged.num_zz_layers == r - 1
    assert split.num_cnot - merged.num_cnot == 2 * (r - 1) * (n - 1)
    # and the merged circuit is cheaper, never equal, for r > 1
    assert merged.total_gates < split.total_gates


def test_per_step_counts():
    n, r = 7, 12
    res = cc.estimate_resources(n, r, order=1)
    assert res.cnot_per_step == 2 * (n - 1)
    assert res.rx_per_step == n
    assert res.rz_per_step == n - 1
    assert res.gates_per_step == 4 * n - 3


def test_both_orders_compile_the_same_number_of_x_layers():
    n, r = 5, 9
    assert (cc.estimate_resources(n, r, order=1).num_x_layers
            == cc.estimate_resources(n, r, order=2).num_x_layers == r)


def test_zero_couplings_follow_the_documented_policy():
    n, r = 4, 6
    # J = 0: no ZZ block at all (angle is exactly zero -> block dropped)
    no_j = cc.estimate_resources(n, r, J=0.0, h=1.0, order=1)
    assert (no_j.num_cnot, no_j.num_rz) == (0, 0)
    assert no_j.num_rx == r * n
    # ... and keeping the symbolic zero rotations restores the structural counts
    kept = cc.estimate_resources(n, r, J=0.0, h=1.0, order=1, keep_zero_angles=True)
    assert kept.num_cnot == 2 * r * (n - 1)
    assert kept.num_rz == r * (n - 1)
    # h = 0: no RX gates, ZZ part unchanged
    no_h = cc.estimate_resources(n, r, J=1.0, h=0.0, order=1)
    assert no_h.num_rx == 0
    assert no_h.num_cnot == 2 * r * (n - 1)
    # both zero: an empty circuit
    empty = cc.estimate_resources(n, r, J=0.0, h=0.0, order=1)
    assert empty.total_gates == 0 and empty.circuit_depth == 0


def test_negative_time_is_supported_and_mirrors_the_angles():
    n, r, t = 4, 5, 1.0
    plus = cc.build_circuit(n, r, t, order=1)
    minus = cc.build_circuit(n, r, -t, order=1)
    assert len(plus) == len(minus)
    for a, b in zip(plus, minus):
        assert a.gate_type == b.gate_type and a.qubits == b.qubits
        assert b.angle == pytest.approx(-a.angle)


# ---------------------------------------------------------------------------
# B. Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(n=1, r=1), dict(n=0, r=1), dict(n=4, r=0), dict(n=4, r=-3),
    dict(n=4, r=1, order=3), dict(n=4, r=1, order=0),
    dict(n=4, r=1, boundary="periodic"),
    dict(n=4, r=1, J=float("nan")), dict(n=4, r=1, h=float("inf")),
    dict(n=4, r=1, t=float("nan")),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        cc.estimate_resources(kwargs.pop("n"), **kwargs)


@pytest.mark.parametrize("n", [4.0, "4", None, True])
def test_non_integer_qubit_counts_are_rejected(n):
    with pytest.raises(TypeError):
        cc.estimate_resources(n, 5)


def test_zero_couplings_are_allowed():
    cc.estimate_resources(4, 3, J=0.0)
    cc.estimate_resources(4, 3, h=0.0)


# ---------------------------------------------------------------------------
# C. Scheduling and depth
# ---------------------------------------------------------------------------

def _depth(gates):
    cc.schedule(gates)
    return cc.circuit_depth(gates)


def test_disjoint_gates_share_a_layer():
    gates = [cc.Gate("rx", (0,), 0.3), cc.Gate("rx", (1,), 0.3), cc.Gate("rx", (2,), 0.3)]
    assert _depth(gates) == 1


def test_gates_sharing_a_qubit_are_sequential():
    gates = [cc.Gate("rx", (1,), 0.3), cc.Gate("rz", (1,), 0.4)]
    assert _depth(gates) == 2


def test_cnot_occupies_both_qubits():
    gates = [cc.Gate("cnot", (0, 1)),
             cc.Gate("rx", (1,), 0.2),      # blocked: target busy until layer 0 ends
             cc.Gate("rx", (2,), 0.2)]      # qubit 2 is free from the start
    cc.schedule(gates)
    assert [g.layer for g in gates] == [0, 1, 0]
    assert cc.circuit_depth(gates) == 2


def test_two_qubit_depth_counts_only_cnot_layers():
    assert cc.two_qubit_depth([cc.Gate("rx", (0,), 0.1)]) == 0
    single = [cc.Gate("cnot", (0, 1))]
    cc.schedule(single)
    assert cc.two_qubit_depth(single) == 1
    chain = [cc.Gate("cnot", (0, 1)), cc.Gate("cnot", (1, 2))]
    cc.schedule(chain)
    assert cc.two_qubit_depth(chain) == 2      # share qubit 1 -> cannot overlap
    pair = [cc.Gate("cnot", (0, 1)), cc.Gate("cnot", (2, 3))]
    cc.schedule(pair)
    assert cc.two_qubit_depth(pair) == 1       # disjoint -> one layer


def test_depth_is_not_gate_count():
    n, r = 4, 5
    res = cc.estimate_resources(n, r, order=1)
    assert res.circuit_depth < res.total_gates
    assert res.two_qubit_depth <= res.circuit_depth


def test_single_zz_layer_depth_is_three():
    # one first-order step with h = 0 is exactly one CNOT/RZ/CNOT layer
    gates = cc.build_circuit(2, 1, 1.0, J=1.0, h=0.0, order=1)
    assert _depth(gates) == 3


def test_first_order_step_depth_formula():
    # per step: 3 layers per bond (sequentially scheduled) + 1 RX layer
    for n in (2, 3, 4, 6):
        res = cc.estimate_resources(n, 1, order=1)
        assert res.circuit_depth == 3 * (n - 1) + 1


# ---------------------------------------------------------------------------
# D. Reports
# ---------------------------------------------------------------------------

def test_report_structure_and_values():
    n, r, t, J, h, order = 6, 50, 1.0, 1.0, 1.0, 1
    report = cc.resource_report(n, r, t, J, h, order)
    assert set(report) == {"system", "trotter", "resources", "assumptions"}
    assert report["system"]["num_qubits"] == n
    assert report["trotter"]["formula_order"] == order
    assert report["trotter"]["steps"] == r
    assert report["trotter"]["step_size"] == pytest.approx(t / r)

    res = cc.estimate_resources(n, r, t, J, h, order)
    got = report["resources"]
    assert got["cnot_count"] == 2 * r * (n - 1)
    assert got["rz_count"] == r * (n - 1)
    assert got["rx_count"] == r * n
    assert got["total_gates"] == r * (4 * n - 3)
    assert got["circuit_depth"] == res.circuit_depth
    assert got["two_qubit_depth"] == res.two_qubit_depth
    json.dumps(report)                      # serialisable as written


def test_json_export_round_trip(tmp_path):
    path = tmp_path / "reports.json"
    reports = [cc.resource_report(4, 5, order=1), cc.resource_report(4, 5, order=2)]
    cc.write_json(reports, str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert loaded[1]["resources"]["cnot_count"] == 2 * 6 * 3     # 2(r+1)(n-1)


def test_csv_export_header_and_values(tmp_path):
    path = tmp_path / "resources.csv"
    rows = [["TFIM", 4, 1, 5, "0000", "1.0e-02", "1.0e-04", 4, 30, 20, 15, 65, 50, 30]]
    cc.write_csv(rows, str(path))
    with open(path, newline="", encoding="utf-8") as fh:
        table = list(csv.reader(fh))
    assert table[0] == cc.RESOURCE_COLUMNS
    assert table[1] == [str(v) for v in rows[0]]
    assert "cnot_count" in table[0] and "two_qubit_depth" in table[0]


# ---------------------------------------------------------------------------
# E. Mathematical verification and regression of the existing simulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("merge", [True, False])
def test_compiled_circuit_equals_matrix_product_formula(order, merge):
    """The gate list must reproduce trotter.trotter_unitary exactly."""
    n, r, t = 4, 3, 1.0
    A, B, _ = trotter.tfim_terms(n)
    gates = cc.build_circuit(n, r, t, order=order, merge_half_steps=merge)
    U_circ = cc.circuit_unitary(n, gates)
    U_ref = trotter.trotter_unitary(A, B, t, r, order)
    assert np.linalg.norm(U_circ - U_ref, 2) < 1e-10


def test_verification_helper_passes():
    out = cc.verify_gate_decomposition(n=4, r=3)
    assert out["ok"], out["checks"]
    assert out["max_error"] < 1e-10


def test_merged_and_unmerged_second_order_are_the_same_operator():
    n, r, t = 5, 4, 1.0
    a = cc.circuit_unitary(n, cc.build_circuit(n, r, t, order=2, merge_half_steps=True))
    b = cc.circuit_unitary(n, cc.build_circuit(n, r, t, order=2, merge_half_steps=False))
    assert np.linalg.norm(a - b, 2) < 1e-12


def test_matrix_product_formula_still_matches_scipy_expm():
    """Regression: the existing evolution definition is untouched."""
    for n, r in ((3, 4), (4, 2)):
        A, B, H = trotter.tfim_terms(n)
        U_exact = expm(-1j * H * 1.0)
        for order in (1, 2):
            U_trot = trotter.trotter_unitary(A, B, 1.0, r, order)
            unit = np.linalg.norm(U_trot.conj().T @ U_trot - np.eye(2 ** n), 2)
            assert unit < 1e-12                       # still unitary
            err = np.linalg.norm(U_exact - U_trot, 2)
            assert math.isfinite(err) and 0.0 <= err < 1.0


def test_existing_results_csv_reproduced(tmp_path):
    """A row of the committed results.csv must still be computed identically."""
    import os
    if not os.path.exists("results.csv"):
        pytest.skip("results.csv not present")
    with open("results.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    target = [row for row in rows
              if row["model"] == "TFIM" and int(row["n"]) == 4
              and int(row["order"]) == 2 and int(row["r"]) == 10][0]

    A, B, H = trotter.tfim_terms(4)
    U_exact = expm(-1j * H * 1.0)
    U_trot = trotter.trotter_unitary(A, B, 1.0, 10, 2)
    err = trotter.operator_error(U_exact, U_trot)
    assert err == pytest.approx(float(target["operator_norm_error"]), abs=1e-12)
