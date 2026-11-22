"""Tests for the Phase 4 independent scientific audit (``validation/``).

These tests do not re-run the long experiments; they check the mathematical
claims the audit makes, using small systems and analytically known values.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import trotter
from validation import audit, comparisons, independent as ind, io_utils, robustness

TOL = 1e-12


# ---------------------------------------------------------------------------
# A. The audit itself
# ---------------------------------------------------------------------------

def test_independent_audit_passes_every_mathematical_check():
    result = audit.run_all(n=3)
    summary = result["summary"]
    assert summary["failed"] == 0, [
        check for check in result["checks"] if not check["passed"]
    ][:5]
    assert summary["checks"] == 187
    assert summary["max_deviation"] < 1e-12


# ---------------------------------------------------------------------------
# B. Hamiltonian and product formulas
# ---------------------------------------------------------------------------

def _pauli(letter):
    if letter == "I":
        return np.eye(2, dtype=complex)
    if letter == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if letter == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(letter)


def _open_chain_hamiltonian(n, J=1.0, h=1.0):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        term = np.array([[1.0 + 0j]])
        for position in range(n):
            term = np.kron(term, _pauli("X") if position == i else _pauli("I"))
        H -= h * term
    for i in range(n - 1):
        term = np.array([[1.0 + 0j]])
        for position in range(n):
            letter = "Z" if position in (i, i + 1) else "I"
            term = np.kron(term, _pauli(letter))
        H -= J * term
    return H


def test_hamiltonian_is_the_open_chain_tfim():
    n = 3
    A, B, H = trotter.tfim_terms(n)
    assert np.max(np.abs(H - _open_chain_hamiltonian(n))) < TOL
    assert np.max(np.abs((A + B) - H)) < TOL
    assert np.max(np.abs(H - H.conj().T)) < TOL           # Hermitian
    # exactly n - 1 ZZ bonds: a periodic chain would differ measurably
    periodic = _open_chain_hamiltonian(n) - np.kron(np.kron(_pauli("Z"), _pauli("I")),
                                                    _pauli("Z"))
    assert np.max(np.abs(periodic - H)) > 0.1


def test_exact_evolution_of_the_field_only_hamiltonian():
    """J = 0 leaves only -h sum X_i, whose exponential is a product of RX."""
    n, t = 3, 0.7
    got = ind.exact_state(n, t=t, J=0.0, h=1.0)
    single = np.array([[np.cos(t), -1j * np.sin(t)], [-1j * np.sin(t), np.cos(t)]])
    expected = single
    for _ in range(n - 1):
        expected = np.kron(expected, single)
    assert np.max(np.abs(np.abs(got) - np.abs(expected[:, 0]))) < TOL


def test_first_order_circuit_is_the_documented_product():
    n, r, t = 3, 2, 0.9
    dt = t / r
    A, B, _ = trotter.tfim_terms(n)
    expected = np.eye(2 ** n, dtype=complex)
    for _ in range(r):
        expected = expm(-1j * A * dt) @ expm(-1j * B * dt) @ expected
    got = ind.circuit_unitary(n, ind.gate_list(n, r, 1, t))
    assert np.max(np.abs(got - expected)) < 1e-10
    # the opposite factor order must differ, or the test proves nothing
    swapped = np.eye(2 ** n, dtype=complex)
    for _ in range(r):
        swapped = expm(-1j * B * dt) @ expm(-1j * A * dt) @ swapped
    assert np.max(np.abs(swapped - expected)) > 1e-3


def test_second_order_circuit_uses_half_angle_layers():
    n, r, t = 3, 2, 0.9
    dt = t / r
    A, B, _ = trotter.tfim_terms(n)
    expected = np.eye(2 ** n, dtype=complex)
    for _ in range(r):
        expected = (expm(-1j * A * dt / 2) @ expm(-1j * B * dt)
                    @ expm(-1j * A * dt / 2) @ expected)
    got = ind.circuit_unitary(n, ind.gate_list(n, r, 2, t))
    assert np.max(np.abs(got - expected)) < 1e-10
    full_angle = np.eye(2 ** n, dtype=complex)
    for _ in range(r):
        full_angle = (expm(-1j * A * dt) @ expm(-1j * B * dt)
                      @ expm(-1j * A * dt) @ full_angle)
    assert np.max(np.abs(full_angle - expected)) > 1e-3


# ---------------------------------------------------------------------------
# C. Noise channels
# ---------------------------------------------------------------------------

def test_global_channel_limits():
    dim = 8
    psi = ind.exact_state(3)
    rho = np.outer(psi, psi.conj())
    assert np.max(np.abs(ind.global_channel(rho, 0.0) - rho)) < TOL
    assert np.max(np.abs(ind.global_channel(rho, 1.0) - np.eye(dim) / dim)) < TOL


@pytest.mark.parametrize("bit", [0, 1])
def test_single_qubit_channel_depolarises_fully_at_three_quarters(bit):
    psi = np.zeros(2, dtype=complex)
    psi[bit] = 1.0
    rho = np.outer(psi, psi.conj())
    got = ind.depolarize_explicit(rho, (0,), 0.75)
    assert np.max(np.abs(got - np.eye(2) / 2)) < TOL
    # p = 1 is NOT the fully depolarising value under this convention
    assert np.max(np.abs(ind.depolarize_explicit(rho, (0,), 1.0) - np.eye(2) / 2)) > 0.1


def test_two_qubit_channel_averages_over_fifteen_paulis():
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    rho = np.outer(psi, psi.conj())
    operators = [matrix for _, matrix in ind.pauli_operators(1)]
    explicit = 0.0
    for labels in np.ndindex(4, 4):
        if labels == (0, 0):
            continue
        op = np.kron(operators[labels[0]], operators[labels[1]])
        explicit = explicit + op @ rho @ op.conj().T
    got = ind.depolarize_explicit(rho, (0, 1), 0.3)
    expected = 0.7 * rho + 0.3 / 15 * explicit
    assert np.max(np.abs(got - expected)) < TOL
    assert len(ind.non_identity_paulis(2)) == 15


@pytest.mark.parametrize("p", [0.1, 0.5])
def test_noise_channels_are_completely_positive(p):
    for k in (1, 2):
        def channel(rho, qubits, _p=p):
            return ind.depolarize_explicit(rho, qubits, _p)
        # the Choi matrix is built from a *normalised* maximally entangled state,
        # so complete positivity shows up as PSD with unit trace
        choi = ind.choi_matrix(channel, k)
        assert np.min(np.linalg.eigvalsh((choi + choi.conj().T) / 2)) > -1e-12
        assert abs(np.trace(choi) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# D. Metrics
# ---------------------------------------------------------------------------

def test_fidelity_and_purity_on_known_states():
    zero = np.zeros(4, dtype=complex)
    zero[0] = 1.0
    pure = np.outer(zero, zero.conj())
    mixed = np.eye(4) / 4
    assert ind.fidelity(pure, zero) == pytest.approx(1.0, abs=TOL)
    assert ind.fidelity(mixed, zero) == pytest.approx(0.25, abs=TOL)
    assert ind.purity(pure) == pytest.approx(1.0, abs=TOL)
    assert ind.purity(mixed) == pytest.approx(0.25, abs=TOL)
    other = np.zeros(4, dtype=complex)
    other[3] = 1.0
    assert ind.fidelity(pure, other) == pytest.approx(0.0, abs=TOL)


def test_trace_distance_uses_the_trace_norm():
    rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
    sigma = np.array([[0.4, 0.0], [0.0, 0.6]], dtype=complex)
    expected = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho - sigma)))
    assert ind.trace_distance(rho, sigma) == pytest.approx(expected, abs=1e-14)
    frobenius = np.linalg.norm(rho - sigma)
    assert abs(ind.trace_distance(rho, sigma) - frobenius) > 1e-3


def test_error_separation_reproduces_the_reported_deviation():
    rows = io_utils.load_results("results/phase3")["gate"]
    separation = audit.error_separation_from_rows(rows)
    assert separation["max_abs_deviation"] == pytest.approx(0.24248777390141396, abs=1e-12)
    worst = separation["worst_row"]
    delta = worst["total_error"] - (worst["ideal_trotter_error"]
                                    + worst["noise_induced_error"])
    assert delta == pytest.approx(worst["delta"], abs=1e-12)
    assert (worst["system_size"], worst["trotter_order"], worst["trotter_steps"]) == (6, 2, 1)


# ---------------------------------------------------------------------------
# E. Optimal-r selection and comparison pairing
# ---------------------------------------------------------------------------

def _synthetic_rows():
    rows = []
    for r, total in ((2, 0.30), (3, 0.10), (4, 0.02), (6, 0.05), (8, 0.20)):
        rows.append({"system_size": 4, "trotter_order": 1, "trotter_steps": r,
                     "single_qubit_error": 1e-3, "two_qubit_error": 1e-3,
                     "total_time": 1.0, "total_error": total,
                     "ideal_trotter_error": 0.01, "noise_induced_error": total - 0.01})
    return rows


def test_optimal_r_is_the_argmin_of_total_error():
    flagged = robustness.with_optimal_flags(_synthetic_rows(), ("system_size",))
    best = [row for row in flagged if row["is_optimal"]]
    assert len(best) == 1
    assert best[0]["trotter_steps"] == 4
    assert best[0]["optimum_status"] == "interior"


def test_boundary_optimum_is_labelled():
    rows = _synthetic_rows()
    rows[-1]["total_error"] = 0.0
    flagged = robustness.with_optimal_flags(rows, ("system_size",))
    best = [row for row in flagged if row["is_optimal"]][0]
    assert best["trotter_steps"] == 8
    assert best["optimum_status"] == "at_largest_tested_r"


def test_pairing_never_mixes_noise_levels():
    rows = []
    for order in (1, 2):
        for p in (1e-3, 1e-2):
            rows.append({"system_size": 4, "trotter_order": order, "trotter_steps": 4,
                         "noise_model": "gate", "single_qubit_error": p,
                         "two_qubit_error": p, "global_noise_probability": 0.0,
                         "total_time": 1.0, "cnot_count": 24, "total_gate_count": 52,
                         "fidelity": 0.9 if p == 1e-3 else 0.5,
                         "total_error": 0.1 if order == 1 else 0.2})
    pairs = comparisons.pair_matched(rows, metric="cnot")
    assert len(pairs) == 2                       # one per noise level, never mixed
    for pair in pairs:
        assert pair["single_qubit_error"] in (1e-3, 1e-2)
        assert pair["order1_error"] == 0.1 and pair["order2_error"] == 0.2
        assert pair["winner"] == comparisons.WIN_ORDER1
        assert pair["order1_budget"] == pair["order2_budget"] == 24
        # the fidelity tag encodes the noise level: a 1e-2 row can never meet a
        # 1e-3 row, which is exactly the bug the corrected pairing removes
        assert pair["order1_fidelity"] == pair["order2_fidelity"]


def test_equal_step_pairs_require_the_same_step_count():
    rows = []
    for order in (1, 2):
        for r in (2, 4):
            rows.append({"system_size": 3, "trotter_order": order, "trotter_steps": r,
                         "noise_model": "gate", "single_qubit_error": 1e-3,
                         "two_qubit_error": 1e-3, "global_noise_probability": 0.0,
                         "total_time": 1.0, "cnot_count": 8 * r,
                         "total_gate_count": 18 * r, "fidelity": 0.9,
                         "total_error": 0.1 + 0.01 * r})
    pairs = comparisons.equal_step_pairs(rows)
    assert len(pairs) == 2
    for pair in pairs:
        assert pair["order1_r"] == pair["order2_r"]


# ---------------------------------------------------------------------------
# F. Reproducibility of the robustness experiments
# ---------------------------------------------------------------------------

def test_random_initial_state_is_reproducible():
    first = ind.random_pure_state(4, robustness.RANDOM_SEED)
    second = ind.random_pure_state(4, robustness.RANDOM_SEED)
    assert np.array_equal(first, second)
    assert abs(np.vdot(first, first) - 1.0) < TOL
    assert not np.array_equal(first, ind.random_pure_state(4, robustness.RANDOM_SEED + 1))


def test_initial_state_set_has_the_documented_four_states():
    states = robustness.initial_states(3)
    assert set(states) == {"|0...0>", "|1...1>", "|+...+>", "random"}
    for state in states.values():
        assert abs(np.vdot(state, state) - 1.0) < TOL
