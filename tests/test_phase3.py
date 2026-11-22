"""Tests for the Phase 3 noise package.

Run with:  python -m pytest tests/ -q

The tests deliberately avoid "function A equals function B" checks wherever an
independent calculation is possible: the density-matrix engine is compared with
the state-vector simulator, the fast channel with the explicit Pauli sum and
with hand-built product states, and the metrics with closed forms.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

import numpy as np
import pytest
from scipy.linalg import expm

import circuit_cost
import trotter

from noise import channels, experiments, metrics, plotting, report
from noise import density_matrix as dm
from noise.density_matrix import (apply_operator, basis_density_matrix,
                                 check_density_matrix, min_eigenvalue,
                                 pure_density_matrix, purity, trace)
from noise.noisy_simulator import (CNOT, NoiseModel, exact_state,
                                   ideal_trotter_state, local_matrix,
                                   reference_states, reset_cache, simulate,
                                   step_ranges)

X = trotter.X
Y = trotter.Y
Z = trotter.Z
I2 = trotter.I2

PHASE1_FILES = [
    os.path.join("results", "circuit_resources.csv"),
    os.path.join("results", "circuit_reports.json"),
    os.path.join("results", "trotter_number.csv"),
    os.path.join("results", "floor_study.csv"),
    os.path.join("results.csv"),
]
PHASE2_FILES = [os.path.join("results", "phase2", name) for name in (
    "minimum_resources_for_accuracy.csv", "resource_efficiency.csv",
    "supplementary_sweep.csv", "phase2_summary.json", "phase2_report.md")]


def _random_density_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = 2 ** n
    matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    state = matrix @ matrix.conj().T
    return state / np.trace(state)


def _embedded_cnot(n: int, control: int, target: int) -> np.ndarray:
    """CNOT on an arbitrary (not necessarily adjacent) pair, built from trotter.py.

    ``circuit_cost._embed_two`` assumes the two qubits are adjacent, which is all
    Phase 1 ever needs; this helper is independent of it.
    """
    projector0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    projector1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return (trotter.two_qubit(n, control, target, projector0, I2)
            + trotter.two_qubit(n, control, target, projector1, X))


def _embedded(op, qubits, n):
    """Embed an arbitrary ``k``-qubit operator on non-adjacent qubits."""
    factors = [I2] * n
    assignment = dict(zip(qubits, op.reshape((2,) * (2 * len(qubits)))))
    for index, qubit in enumerate(qubits):
        factors[qubit] = op.reshape((2,) * (2 * len(qubits)))[
            tuple((slice(None) if axis in (index, index + len(qubits)) else 0)
                  for axis in range(2 * len(qubits)))]
    return trotter.kron_all(factors)


# ---------------------------------------------------------------------------
# A. Density matrix: construction and properties
# ---------------------------------------------------------------------------

def test_pure_state_conversion_is_the_outer_product():
    psi = np.array([1, 1j, -1, 0], dtype=complex) / math.sqrt(3)
    rho = pure_density_matrix(psi)
    assert np.allclose(rho, np.outer(psi, psi.conj()))
    assert rho.shape == (4, 4)


def test_pure_state_conversion_rejects_unnormalised_vectors():
    with pytest.raises(ValueError):
        pure_density_matrix(np.array([1.0, 1.0]))


def test_basis_state_convention_matches_trotter():
    # |10> has qubit 0 = 1 (most significant), matching trotter.basis_state
    rho = basis_density_matrix(2, [1, 0])
    psi = trotter.basis_state(2, [1, 0])
    assert np.allclose(rho, np.outer(psi, psi.conj()))
    assert rho[2, 2] == pytest.approx(1.0)


def test_trace_and_purity_of_a_pure_state():
    rho = pure_density_matrix(np.array([0, 1, 0, 0], dtype=complex))
    assert trace(rho) == pytest.approx(1.0)
    assert purity(rho) == pytest.approx(1.0)


def test_purity_of_the_maximally_mixed_state():
    rho = channels.maximally_mixed(8)
    assert purity(rho) == pytest.approx(1.0 / 8.0)
    assert trace(rho) == pytest.approx(1.0)


def test_check_density_matrix_accepts_valid_and_rejects_invalid():
    rho = _random_density_matrix(2)
    info = check_density_matrix(rho)
    assert info["ok"] is True
    assert info["num_qubits"] == 2
    assert abs(info["trace"] - 1.0) < 1e-12

    bad = np.array([[1.0, 0.0], [0.0, -0.3]], dtype=complex)   # negative eigenvalue
    report_bad = check_density_matrix(bad)
    assert report_bad["ok"] is False
    assert report_bad["min_eigenvalue"] < 0

    non_hermitian = np.array([[0.5, 1.0], [0.0, 0.5]], dtype=complex)
    assert check_density_matrix(non_hermitian)["ok"] is False


def test_hermiticity_and_positivity_of_random_states():
    for seed in range(3):
        rho = _random_density_matrix(3, seed=seed)
        assert dm.hermiticity_error(rho) < 1e-12
        assert min_eigenvalue(rho) >= -1e-12


# ---------------------------------------------------------------------------
# B. Local operator algebra
# ---------------------------------------------------------------------------

def test_apply_operator_reproduces_the_full_matrix_product():
    n = 3
    rho = _random_density_matrix(n, seed=4)
    for q in range(n):
        theta = 0.37 + q
        op = np.array([[math.cos(theta / 2), -1j * math.sin(theta / 2)],
                       [-1j * math.sin(theta / 2), math.cos(theta / 2)]], dtype=complex)
        full = trotter.single_qubit(n, q, op)
        assert np.allclose(apply_operator(rho, op, (q,)), full @ rho @ full.conj().T)


def test_apply_operator_two_qubit_matches_the_full_matrix_product():
    n = 3
    rho = _random_density_matrix(n, seed=5)
    full = _embedded_cnot(n, 0, 2)
    assert np.allclose(apply_operator(rho, CNOT, (0, 2)), full @ rho @ full.conj().T)
    # and the adjacent case through Phase 1's own embedding
    full = circuit_cost._embed_two(n, 1, 2, CNOT)
    assert np.allclose(apply_operator(rho, CNOT, (1, 2)), full @ rho @ full.conj().T)


def test_apply_operator_rejects_wrong_shapes_and_repeated_qubits():
    rho = _random_density_matrix(2)
    with pytest.raises(ValueError):
        apply_operator(rho, CNOT, (0,))            # 4x4 op on one qubit
    with pytest.raises(ValueError):
        apply_operator(rho, X, (0, 0))
    with pytest.raises(ValueError):
        apply_operator(rho, X, (0, 2))             # qubit out of range


def test_local_matrix_embedding_matches_the_phase1_gate_matrix():
    for gate in circuit_cost.build_circuit(4, 2, 1.0, order=2)[:12]:
        if gate.gate_type == "cnot":
            embedded = circuit_cost._embed_two(4, *gate.qubits, local_matrix(gate))
        else:
            embedded = trotter.single_qubit(4, gate.qubits[0], local_matrix(gate))
        assert np.allclose(embedded, circuit_cost.gate_matrix(4, gate))


def test_partial_trace_of_a_product_state():
    rho_a = _random_density_matrix(1, seed=7)
    rho_b = _random_density_matrix(1, seed=8)
    product = np.kron(rho_a, rho_b)
    assert np.allclose(dm.partial_trace(product, (1,)), rho_a)   # keep qubit 0
    assert np.allclose(dm.partial_trace(product, (0,)), rho_b)


# ---------------------------------------------------------------------------
# C. Unitary evolution
# ---------------------------------------------------------------------------

def test_ideal_unitary_evolution_preserves_purity():
    rho = _random_density_matrix(2, seed=9)
    out = apply_operator(rho, CNOT, (0, 1))
    assert purity(out) == pytest.approx(purity(rho), rel=1e-12)


def test_density_matrix_evolution_agrees_with_the_state_vector():
    for n, r, order in ((4, 1, 1), (4, 3, 1), (4, 3, 2), (6, 2, 2)):
        out = simulate(n, r, order, model=NoiseModel(kind="gate"))
        psi = ideal_trotter_state(n, r, order)
        assert np.allclose(out["rho"], np.outer(psi, psi.conj()), atol=1e-12)


def test_ideal_circuit_matches_trotter_unitary_for_both_orders():
    A, B, _ = trotter.tfim_terms(4)
    for order in (1, 2):
        gates = circuit_cost.build_circuit(4, 3, 1.0, order=order)
        assert np.linalg.norm(circuit_cost.circuit_unitary(4, gates)
                              - trotter.trotter_unitary(A, B, 1.0, 3, order), 2) < 1e-10


def test_density_matrix_matches_trotter_sweep_infidelity():
    """The rebuilt state must reproduce the Phase 1 infidelity numbers."""
    A, B, H = trotter.tfim_terms(4)
    reference = trotter.sweep(A, B, H, 1.0, [1, 2, 5])
    psi_exact = exact_state(4, 1.0)
    for order in (1, 2):
        for index, r in enumerate((1, 2, 5)):
            row = experiments.measure(4, r, order, model=NoiseModel(kind="gate"))
            assert row["ideal_trotter_error"] == pytest.approx(
                reference[order]["inf"][index], abs=1e-12)
            assert metrics.infidelity(reference_states(4, r, order)["rho_trotter"],
                                      psi_exact) == pytest.approx(
                reference[order]["inf"][index], abs=1e-12)


# ---------------------------------------------------------------------------
# D. Channels: validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_probabilities_outside_the_unit_interval_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        channels.validate_probability(bad)


def test_non_numeric_probabilities_are_rejected():
    with pytest.raises(TypeError):
        channels.validate_probability("0.1")
    with pytest.raises(TypeError):
        channels.validate_probability(True)


def test_noise_model_validation():
    with pytest.raises(ValueError):
        NoiseModel(kind="depolarising")
    with pytest.raises(ValueError):
        NoiseModel(kind="gate", single_qubit_error=2.0)
    model = NoiseModel(kind="gate", single_qubit_error=0.1, two_qubit_error=0.2,
                       rz_error=0.0)
    assert model.error_for("rx") == pytest.approx(0.1)
    assert model.error_for("cnot") == pytest.approx(0.2)
    assert model.error_for("rz") == pytest.approx(0.0)
    assert NoiseModel(kind="gate", single_qubit_error=0.1).error_for("rz") == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# E. Global channel
# ---------------------------------------------------------------------------

def test_global_channel_end_points_and_trace():
    rho = basis_density_matrix(3, [1, 0, 1])
    assert np.allclose(channels.apply_global_depolarizing(rho, 0.0), rho)
    mixed = channels.apply_global_depolarizing(rho, 1.0)
    assert np.allclose(mixed, np.eye(8) / 8)
    assert trace(mixed) == pytest.approx(1.0)
    assert min_eigenvalue(mixed) >= -1e-14


def test_global_channel_is_the_documented_linear_mixture():
    rho = _random_density_matrix(2, seed=11)
    p = 0.23
    expected = (1 - p) * rho + p * np.eye(4) / 4
    assert np.allclose(channels.apply_global_depolarizing(rho, p), expected)
    out = channels.apply_global_depolarizing(rho, p)
    assert trace(out) == pytest.approx(1.0)
    assert dm.hermiticity_error(out) < 1e-13


# ---------------------------------------------------------------------------
# F. Pauli channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_pauli_group_has_the_right_size_and_normalisation(k):
    operators = channels.pauli_operators(k)
    assert len(operators) == 4 ** k
    labels = [label for label, _ in operators]
    assert len(set(labels)) == 4 ** k
    identity = sum(1 for label in labels if set(label) == {"I"})
    assert identity == 1
    assert len(operators) - identity == (channels.SINGLE_QUBIT_NON_IDENTITY if k == 1
                                        else channels.TWO_QUBIT_NON_IDENTITY)
    for _, matrix in operators:
        assert np.allclose(matrix @ matrix.conj().T, np.eye(2 ** k))


def test_explicit_pauli_channel_matches_the_definition_by_hand():
    n = 2
    rho = _random_density_matrix(n, seed=12)
    p = 0.31
    manual = (1 - p) * rho
    for op in (X, Y, Z):
        manual = manual + (p / 3.0) * apply_operator(rho, op, (1,))
    assert np.allclose(channels.explicit_pauli_depolarizing(rho, (1,), p), manual)


def test_fast_channel_matches_the_explicit_pauli_sum():
    for n, qubits, p in ((3, (0,), 0.05), (4, (2,), 0.6), (4, (0, 2), 0.05),
                         (5, (1, 3), 0.9)):
        rho = _random_density_matrix(n, seed=13)
        fast = channels.depolarize_qubits(rho, qubits, p)
        slow = channels.explicit_pauli_depolarizing(rho, qubits, p)
        assert np.abs(fast - slow).max() < 1e-14


def test_single_qubit_channel_preserves_trace_and_positivity():
    rho = _random_density_matrix(3, seed=14)
    for p in (0.0, 0.2, 1.0):
        out = channels.depolarize_qubits(rho, (1,), p)
        assert trace(out) == pytest.approx(1.0)
        info = check_density_matrix(out)
        assert info["ok"], info
        assert purity(out) <= purity(rho) + 1e-12


def test_two_qubit_channel_preserves_trace_and_positivity():
    rho = _random_density_matrix(4, seed=15)
    for p in (0.0, 0.2, 1.0):
        out = channels.depolarize_qubits(rho, (0, 3), p)
        assert trace(out) == pytest.approx(1.0)
        assert check_density_matrix(out)["ok"]
        assert purity(out) <= purity(rho) + 1e-12


def test_single_qubit_channel_shrinks_the_bloch_vector_as_documented():
    """Analytic check of the p/3 normalisation: lambda = 1 - 4p/3."""
    vector = np.array([0.3, -0.2, 0.5])
    rho = (np.eye(2) + vector[0] * X + vector[1] * Y + vector[2] * Z) / 2
    for p in (0.0, 0.1, 0.5, 0.75, 1.0):
        out = channels.depolarize_qubits(rho, (0,), p)
        lam = 1.0 - 4.0 * p / 3.0
        for op, expected in ((X, lam * vector[0]), (Y, lam * vector[1]),
                             (Z, lam * vector[2])):
            assert dm.expectation(out, op, (0,)) == pytest.approx(expected, abs=1e-12)
        # p = 3/4 is complete depolarisation of the qubit, not p = 1
        if p == 0.75:
            assert np.allclose(out, np.eye(2) / 2)


def test_channel_matches_the_full_pauli_group_sum():
    """E at p = 1 equals (sum over all Paulis - rho)/(4**k - 1), summed explicitly."""
    rho = _random_density_matrix(3, seed=16)
    for k, qubits in ((1, (1,)), (2, (0, 2))):
        total = sum(apply_operator(rho, matrix, qubits)
                    for _, matrix in channels.pauli_operators(k))
        assert np.trace(total) == pytest.approx(4 ** k)     # every P is unitary
        got = channels.depolarize_qubits(rho, qubits, 1.0)
        assert np.allclose(got, (total - rho) / (4 ** k - 1))


def test_channel_leaves_the_rest_of_the_system_untouched():
    """The marginal on the qubits outside the channel is unchanged at any p."""
    rho = _random_density_matrix(3, seed=17)
    for p in (0.0, 0.3, 1.0):
        out = channels.depolarize_qubits(rho, (1,), p)
        assert np.allclose(dm.partial_trace(out, (1,)),
                           dm.partial_trace(rho, (1,)))


def test_local_noise_acts_on_the_named_qubit_only():
    """Product state: the channel on qubit 1 must leave qubits 0 and 2 alone."""
    rho0 = _random_density_matrix(1, seed=21)
    rho1 = _random_density_matrix(1, seed=22)
    rho2 = _random_density_matrix(1, seed=23)
    product = np.kron(np.kron(rho0, rho1), rho2)
    p = 0.4
    out = channels.depolarize_qubits(product, (1,), p)
    expected = np.kron(np.kron(rho0, channels.depolarize_qubits(rho1, (0,), p)), rho2)
    assert np.allclose(out, expected)


def test_two_qubit_noise_acts_on_the_named_pair_only():
    """Adjacent pair: an independent product-state construction."""
    rho0 = _random_density_matrix(1, seed=31)
    rho1 = _random_density_matrix(1, seed=32)
    rho2 = _random_density_matrix(1, seed=33)
    product = np.kron(np.kron(rho0, rho1), rho2)
    p = 0.4
    expected = np.kron(channels.depolarize_qubits(np.kron(rho0, rho1), (0, 1), p), rho2)
    assert np.allclose(channels.depolarize_qubits(product, (0, 1), p), expected)


def test_two_qubit_noise_is_covariant_under_qubit_permutation():
    """Non-adjacent pair: swapping a spectator qubit and the channel commute."""
    rho = _random_density_matrix(3, seed=34)
    swap2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                      [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    perm = circuit_cost._embed_two(3, 1, 2, swap2)        # exchanges qubits 1 and 2
    p = 0.37
    left = channels.depolarize_qubits(rho, (0, 2), p)
    right = perm @ channels.depolarize_qubits(perm @ rho @ perm, (0, 1), p) @ perm
    assert np.allclose(left, right)


def test_channel_rejects_bad_qubit_indices():
    rho = _random_density_matrix(2)
    with pytest.raises(ValueError):
        channels.depolarize_qubits(rho, (0, 0), 0.1)
    with pytest.raises(ValueError):
        channels.depolarize_qubits(rho, (0, 5), 0.1)


# ---------------------------------------------------------------------------
# G. Gate-level integration
# ---------------------------------------------------------------------------

def test_zero_noise_reproduces_the_ideal_circuit():
    for order in (1, 2):
        for r in (1, 4):
            ideal = simulate(4, r, order, model=NoiseModel(kind="none"))
            zero = simulate(4, r, order,
                            model=NoiseModel(kind="gate", single_qubit_error=0.0,
                                             two_qubit_error=0.0))
            assert np.allclose(ideal["rho"], zero["rho"], atol=1e-14)


def test_gate_counts_match_phase1_exactly():
    for n in (4, 5, 6):
        for order in (1, 2):
            for r in (1, 3, 12):
                out = simulate(n, r, order, model=NoiseModel(kind="gate"))
                reference = circuit_cost.estimate_resources(n, r, order=order)
                assert out["resources"].num_cnot == reference.num_cnot
                assert out["resources"].total_gates == reference.total_gates
                assert out["gates"] == reference.total_gates
                analytic = circuit_cost.analytic_counts(n, r, order)
                assert out["resources"].num_cnot == analytic["cnot"]


def test_noise_channels_are_applied_after_every_nonzero_gate():
    model = NoiseModel(kind="gate", single_qubit_error=0.01, two_qubit_error=0.02)
    out = simulate(4, 3, 2, model=model)
    assert out["num_channels"] == out["gates"]


def test_rz_can_be_declared_noiseless():
    noisy = NoiseModel(kind="gate", single_qubit_error=0.01, two_qubit_error=0.01)
    quiet_rz = NoiseModel(kind="gate", single_qubit_error=0.01, two_qubit_error=0.01,
                          rz_error=0.0)
    a = simulate(4, 4, 2, model=noisy)
    b = simulate(4, 4, 2, model=quiet_rz)
    assert np.abs(a["rho"] - b["rho"]).max() > 1e-6      # RZ noise really was removed
    assert b["num_channels"] == a["num_channels"] - b["resources"].num_rz


def test_merged_second_order_layers_still_reproduce_the_unitary():
    A, B, _ = trotter.tfim_terms(4)
    merged = circuit_cost.build_circuit(4, 3, 1.0, order=2, merge_half_steps=True)
    unmerged = circuit_cost.build_circuit(4, 3, 1.0, order=2, merge_half_steps=False)
    target = trotter.trotter_unitary(A, B, 1.0, 3, 2)
    assert np.linalg.norm(circuit_cost.circuit_unitary(4, merged) - target, 2) < 1e-10
    assert np.linalg.norm(circuit_cost.circuit_unitary(4, unmerged) - target, 2) < 1e-10
    assert len(merged) < len(unmerged)


def test_zero_angle_blocks_follow_the_phase1_convention():
    gates = circuit_cost.build_circuit(4, 3, 1.0, J=0.0, order=1)
    assert all(gate.gate_type == "rx" for gate in gates)          # no ZZ block
    gates = circuit_cost.build_circuit(4, 3, 1.0, h=0.0, order=1)
    assert all(gate.gate_type in ("cnot", "rz") for gate in gates)


def test_gate_noise_is_stronger_than_the_global_model_at_equal_parameters():
    p = 1e-2
    gate = simulate(4, 8, 1, model=NoiseModel(kind="gate", single_qubit_error=p,
                                             two_qubit_error=p))
    glob = simulate(4, 8, 1, model=NoiseModel(kind="global_per_step",
                                              global_probability=p))
    psi = ideal_trotter_state(4, 8, 1)
    assert metrics.infidelity(gate["rho"], psi) > metrics.infidelity(glob["rho"], psi)


# ---------------------------------------------------------------------------
# H. Trotter-step boundaries for the global model
# ---------------------------------------------------------------------------

def test_step_ranges_cover_the_gate_list_exactly():
    for n in (4, 6):
        for order in (1, 2):
            for r in (1, 2, 7):
                gates = circuit_cost.build_circuit(n, r, 1.0, order=order)
                ranges = step_ranges(n, r, order=order)
                assert len(ranges) == r
                assert sum(stop - start for start, stop in ranges) == len(gates)
                assert ranges[0][0] == 0 and ranges[-1][1] == len(gates)
                for (_, stop), (start, _) in zip(ranges, ranges[1:]):
                    assert stop == start                 # contiguous, no overlap


def test_global_model_applies_exactly_one_channel_per_step():
    for order in (1, 2):
        out = simulate(5, 6, order, model=NoiseModel(kind="global_per_step",
                                                    global_probability=0.01))
        assert out["num_channels"] == 6


def test_end_of_circuit_global_model_applies_one_channel():
    out = simulate(4, 6, 2, model=NoiseModel(kind="global_end_of_circuit",
                                            global_probability=0.01))
    assert out["num_channels"] == 1


# ---------------------------------------------------------------------------
# I. Metrics
# ---------------------------------------------------------------------------

def test_fidelity_of_identical_and_orthogonal_pure_states():
    psi = trotter.basis_state(2, [0, 1])
    rho = pure_density_matrix(psi)
    assert metrics.fidelity_pure(rho, psi) == pytest.approx(1.0)
    assert metrics.infidelity(rho, psi) == pytest.approx(0.0)
    other = trotter.basis_state(2, [1, 0])
    assert metrics.fidelity_pure(rho, other) == pytest.approx(0.0)
    assert metrics.infidelity(rho, other) == pytest.approx(1.0)


def test_fidelity_stays_in_the_unit_interval_and_matches_a_closed_form():
    psi = np.ones(4, dtype=complex) / 2
    rho = channels.apply_global_depolarizing(pure_density_matrix(psi), 0.3)
    # <psi|rho|psi> = 0.7 * 1 + 0.3/4
    assert metrics.fidelity_pure(rho, psi) == pytest.approx(0.775)
    for p in (0.0, 0.5, 1.0):
        value = metrics.fidelity_pure(
            channels.apply_global_depolarizing(pure_density_matrix(psi), p), psi)
        assert 0.0 <= value <= 1.0


def test_infidelity_is_one_minus_fidelity():
    psi = np.array([1, 1, 1, 1], dtype=complex) / 2
    rho = channels.apply_global_depolarizing(pure_density_matrix(psi), 0.42)
    assert metrics.infidelity(rho, psi) == pytest.approx(
        1.0 - metrics.fidelity_pure(rho, psi))


def test_general_fidelity_reduces_to_the_pure_form():
    psi = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    rho = pure_density_matrix(psi)
    assert metrics.fidelity_general(rho, rho) == pytest.approx(1.0)
    sigma = channels.maximally_mixed(4)          # <psi|I/4|psi> = 1/4 for a pure state
    assert metrics.fidelity_general(rho, sigma) == pytest.approx(0.25, rel=1e-9)


def test_purity_decreases_for_a_genuinely_mixed_state():
    psi = trotter.basis_state(3, [0, 0, 0])
    rho = pure_density_matrix(psi)
    mixed = channels.apply_global_depolarizing(rho, 0.2)
    assert purity(mixed) < purity(rho)
    assert purity(mixed) == pytest.approx(0.8 ** 2 + 0.2 ** 2 / 8 + 2 * 0.8 * 0.2 / 8)


def test_trace_distance_of_identical_states_is_zero():
    rho = _random_density_matrix(2, seed=41)
    assert metrics.trace_distance(rho, rho) == pytest.approx(0.0, abs=1e-12)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    orth = pure_density_matrix(np.array([0, 0, 0, 1], dtype=complex))
    assert metrics.trace_distance(pure_density_matrix(psi), orth) == pytest.approx(1.0)


def test_observables_on_states_with_known_values():
    up = basis_density_matrix(4, [0, 0, 0, 0])
    assert dm.magnetization_x(up, 4) == pytest.approx(0.0)
    assert dm.zz_correlation(up, 4) == pytest.approx(1.0)

    plus = np.ones(16, dtype=complex) / 4
    assert dm.magnetization_x(pure_density_matrix(plus), 4) == pytest.approx(1.0)
    assert dm.zz_correlation(pure_density_matrix(plus), 4) == pytest.approx(0.0,
                                                                          abs=1e-12)

    neel = basis_density_matrix(4, [0, 1, 0, 1])
    assert dm.zz_correlation(neel, 4) == pytest.approx(-1.0)


def test_expectation_matches_an_explicit_matrix_trace():
    n = 3
    rho = _random_density_matrix(n, seed=42)
    for q in range(n):
        full = trotter.single_qubit(n, q, X)
        assert dm.expectation(rho, X, (q,)) == pytest.approx(
            np.real(np.trace(rho @ full)))
    pair = trotter.two_qubit(n, 0, 2, Z, Z)      # non-adjacent, built independently
    assert dm.expectation(rho, np.kron(Z, Z), (0, 2)) == pytest.approx(
        np.real(np.trace(rho @ pair)))


def test_measure_reports_the_three_errors_separately():
    row = experiments.measure(4, 6, 2, model=NoiseModel(kind="gate",
                                                       single_qubit_error=1e-3,
                                                       two_qubit_error=1e-3))
    assert row["total_error"] == pytest.approx(1.0 - row["fidelity"])
    assert row["ideal_trotter_error"] > 0
    assert row["noise_induced_error"] > 0
    assert row["total_error"] == pytest.approx(1.0 - row["fidelity"])
    # the three are related but not additive in general
    assert row["total_error"] >= 0.0
    assert 0.0 <= row["fidelity"] <= 1.0


# ---------------------------------------------------------------------------
# J. Experiments and the optimal-r search
# ---------------------------------------------------------------------------

def _toy_rows():
    """Synthetic rows with a known interior minimum at r = 4."""
    rows = []
    for r, error in ((1, 0.5), (2, 0.1), (4, 0.01), (8, 0.2)):
        rows.append({
            "system_size": 4, "trotter_order": 1, "trotter_steps": r,
            "noise_model": "gate", "single_qubit_error": 1e-3,
            "two_qubit_error": 1e-3, "global_noise_probability": 0.0,
            "total_time": 1.0, "total_error": error, "fidelity": 1 - error,
            "ideal_trotter_error": 0.5 / r, "noise_induced_error": 0.001 * r,
            "cnot_count": 12 * r, "total_gate_count": 26 * r, "logical_depth": 14 * r,
        })
    return rows


def test_optimal_r_is_taken_from_evaluated_candidates_only():
    rows = _toy_rows()
    best = experiments.optimal_step_count(rows)[0]
    assert best["optimal_r"] == 4
    assert best["optimum_status"] == experiments.OPTIMUM_INTERIOR
    assert best["candidates_evaluated"] == 4
    assert best["minimum_total_error"] == pytest.approx(0.01)
    assert best["optimal_r"] in {row["trotter_steps"] for row in rows}


def test_boundary_minima_are_labelled():
    rows = _toy_rows()
    for row in rows:
        row["total_error"] = 1.0 / row["trotter_steps"]      # monotone: minimum at max r
    best = experiments.optimal_step_count(rows)[0]
    assert best["optimal_r"] == 8
    assert best["optimum_status"] == experiments.OPTIMUM_LARGEST

    for row in rows:
        row["total_error"] = row["trotter_steps"] * 1e-3     # minimum at smallest r
    best = experiments.optimal_step_count(rows)[0]
    assert best["optimal_r"] == 1
    assert best["optimum_status"] == experiments.OPTIMUM_SMALLEST


def test_optimal_r_separates_different_noise_levels():
    rows = _toy_rows()
    rows += [{**row, "single_qubit_error": 1e-2, "two_qubit_error": 1e-2,
              "total_error": row["total_error"] * 3} for row in rows]
    groups = experiments.optimal_step_count(rows)
    assert len(groups) == 2
    assert {group["single_qubit_error"] for group in groups} == {1e-3, 1e-2}


def test_pair_orders_respects_the_budget():
    rows = []
    for order in (1, 2):
        scale = 2 if order == 1 else 2.2
        for r in (1, 2, 4, 8):
            rows.append({
                "system_size": 4, "trotter_order": order, "trotter_steps": r,
                "noise_model": "gate", "single_qubit_error": 1e-3,
                "cnot_count": int(scale * 12 * r),
                "total_gate_count": int(scale * 26 * r),
                "fidelity": 1 - 1.0 / r ** order,
                "total_error": 1.0 / r ** order,
                "ideal_trotter_error": 1.0 / r ** order,
                "noise_induced_error": 1e-3 * r,
            })
    for metric, column in (("cnot", "cnot_count"), ("total_gate_count", "total_gate_count")):
        pairs = experiments.pair_orders(rows, 4, "gate", metric)
        assert pairs
        for pair in pairs:
            assert pair["order1_budget"] <= pair["order2_budget"]
            assert pair["within_budget"] is True


def _tagged_rows(levels=(0.0, 1e-3, 1e-2), steps=(1, 2, 4, 8)):
    """Rows whose total_error encodes the noise level and the order as p + order."""
    rows = []
    for p in levels:
        for order in (1, 2):
            for r in steps:
                rows.append({
                    'system_size': 4,
                    'trotter_order': order,
                    'trotter_steps': r,
                    'noise_model': 'gate',
                    'single_qubit_error': p,
                    'two_qubit_error': p,
                    'global_noise_probability': 0.0,
                    'cnot_count': 12 * r if order == 1 else 12 * (r + 1),
                    'total_gate_count': 26 * r if order == 1 else 29 * (r + 1),
                    'fidelity': 0.5,
                    'total_error': p + order,
                    'ideal_trotter_error': 0.0,
                    'noise_induced_error': 0.0,
                    'logical_depth': 10 * r,
                })
    return rows


def test_pair_orders_never_matches_across_noise_levels():
    """A noisy circuit must never be compared with a noise-free one."""
    pairs = experiments.pair_orders(_tagged_rows(), 4, 'gate', 'cnot')
    assert len(pairs) == 12                      # 3 noise levels x 4 step counts
    for pair in pairs:
        level = pair['single_qubit_error']
        assert pair['total_error_order1'] == pytest.approx(level + 1)
        assert pair['total_error_order2'] == pytest.approx(level + 2)
        assert pair['order1_budget'] <= pair['order2_budget']


def test_equal_steps_pairs_never_matches_across_noise_levels():
    pairs = experiments.equal_steps_pairs(_tagged_rows(), 4, 'gate')
    assert len(pairs) == 12
    for pair in pairs:
        level = pair['single_qubit_error']
        assert pair['total_error_order1'] == pytest.approx(level + 1)
        assert pair['total_error_order2'] == pytest.approx(level + 2)


def test_measure_is_reproducible():
    model = NoiseModel(kind="gate", single_qubit_error=3e-3, two_qubit_error=3e-3)
    first = experiments.measure(4, 5, 2, model=model)
    second = experiments.measure(4, 5, 2, model=model)
    assert first == second


def test_invalid_experiment_configuration_fails_clearly():
    with pytest.raises(ValueError):
        experiments.measure(4, 0, 1)                    # r must be >= 1
    with pytest.raises(ValueError):
        experiments.measure(4, 2, 3)                    # unknown order
    with pytest.raises(ValueError):
        experiments.measure(4, 2, 1, model=NoiseModel(kind="gate",
                                                     single_qubit_error=-1.0))


def test_sensitivity_grid_has_the_expected_shape():
    rows = experiments.noise_sensitivity(4, 4, 2, p1_values=(1e-4, 1e-2),
                                         p2_values=(1e-4, 1e-2))
    full = [row for row in rows if row["grid"] == "full"]
    assert len(full) == 4
    assert len([row for row in rows if row["grid"] == "single_qubit_only"]) == 2
    assert len([row for row in rows if row["grid"] == "two_qubit_only"]) == 2
    assert all(0.0 <= row["fidelity"] <= 1.0 for row in rows)


def test_both_gate_classes_contribute_to_the_accumulated_error():
    """At equal p, neither gate class can be neglected - and they combine.

    The measured balance depends on the gate *counts*, not only on the channel:
    these circuits contain a number of single-qubit gates comparable to the CNOT
    count, so the two slices give comparable total errors.
    """
    rows = experiments.noise_sensitivity(4, 8, 2, p1_values=(1e-3,),
                                         p2_values=(1e-3,))
    single = [row for row in rows if row["grid"] == "single_qubit_only"][0]
    pair = [row for row in rows if row["grid"] == "two_qubit_only"][0]
    combined = [row for row in rows if row["grid"] == "full"][0]
    assert pair["total_error"] > 0.0 and single["total_error"] > 0.0
    assert combined["total_error"] >= max(pair["total_error"], single["total_error"])
    ratio = pair["total_error"] / single["total_error"]
    assert 0.5 < ratio < 3.0                       # comparable, not one-sided
    # per gate, a noisy CNOT costs more error than a noisy rotation
    single_gates = single["total_gate_count"] - single["cnot_count"]
    assert pair["total_error"] / pair["cnot_count"] > single["total_error"] / single_gates


# ---------------------------------------------------------------------------
# K. Outputs
# ---------------------------------------------------------------------------

def test_report_contains_every_required_section():
    rows = [experiments.measure(4, r, order, model=NoiseModel(kind="gate"))
            for order in (1, 2) for r in (1, 2, 4)]
    optimal = experiments.optimal_step_count(rows)
    text = report.build_report(rows, [], optimal, {"all_passed": True},
                               {"t": 1.0, "J": 1.0, "h": 1.0,
                                "reference_noise": 1e-3}, [], "figures")
    for heading in ("## 1. Research question", "## 3. Global depolarising channel",
                    "## 4. Single-qubit depolarising channel",
                    "## 5. Two-qubit depolarising channel",
                    "## 6. How noise is applied to individual gates",
                    "## 12. Optimal Trotter step count", "## 14. Limitations",
                    "## 15. Reproducibility", "## 16. Conclusions"):
        assert heading in text


def test_summary_json_is_strict_and_serialisable(tmp_path):
    rows = [experiments.measure(4, r, order, model=NoiseModel(kind="gate"))
            for order in (1, 2) for r in (1, 2)]
    summary = report.build_summary(rows, [], experiments.optimal_step_count(rows),
                                   {"all_passed": True}, {"t": 1.0})
    path = tmp_path / "summary.json"
    report.write_summary(summary, str(path))
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    import json
    assert json.loads(text)["phase"] == 3


def test_figure_names_are_all_produced(tmp_path):
    rows = []
    for n in (4,):
        for order in (1, 2):
            for r in (1, 2, 4, 8):
                for p in (0.0, 1e-4, 1e-3):
                    rows.append(experiments.measure(
                        n, r, order,
                        model=NoiseModel(kind="gate", single_qubit_error=p,
                                         two_qubit_error=p)))
                rows.append(experiments.measure(
                    n, r, order,
                    model=NoiseModel(kind="global_per_step", global_probability=1e-3)))
    sensitivity = experiments.noise_sensitivity(4, 4, 2, p1_values=(1e-4, 1e-3),
                                               p2_values=(1e-4, 1e-3))
    optimal = experiments.optimal_step_count(rows)
    paths = plotting.make_all_figures(rows, sensitivity, optimal, str(tmp_path),
                                      reference_noise=1e-4)
    assert len(paths) == 12
    for path in paths:
        assert os.path.getsize(path) > 5000


# ---------------------------------------------------------------------------
# L. Regression: earlier phases are untouched
# ---------------------------------------------------------------------------

def test_phase1_and_phase2_outputs_are_unchanged_by_a_phase3_run(tmp_path):
    watched = [path for path in PHASE1_FILES + PHASE2_FILES if os.path.exists(path)]
    before = {path: hashlib.sha256(open(path, "rb").read()).hexdigest()
              for path in watched}
    from noise import run_phase3
    exit_code = run_phase3.main([
        "--quick", "--no-plots", "--outdir", str(tmp_path),
        "--steps", "1,2,4", "--levels", "1e-3", "--ns", "4"])
    assert exit_code == 0
    for path, digest in before.items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
    assert (tmp_path / "phase3_report.md").exists()
    assert (tmp_path / "optimal_r.csv").exists()
    assert (tmp_path / "gate_noise_results.csv").exists()
    assert (tmp_path / "configs" / "experiment_config.json").exists()


def test_phase1_gate_identity_still_holds():
    """The Phase 1 decomposition is the foundation of Phase 3 - keep it exact."""
    result = circuit_cost.verify_gate_decomposition(n=4, r=3)
    assert result["ok"] is True
    assert result["max_error"] < 1e-10
