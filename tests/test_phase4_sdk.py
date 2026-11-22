"""Tests for the Phase 4 Qiskit integration (``quantum_sdk/``).

The point of these tests is mathematical equivalence, not that Qiskit can build
a circuit: the custom gate list must become the *same unitary*, the pre-
transpilation resources must equal Phase 1, and Aer's noise must reproduce the
project's channels once the conventions are reconciled.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

import circuit_cost as cc
from quantum_sdk import (circuit_builder, equivalence, layout, noise_check,
                         resources, transpilation)

TOL = 1e-10


# ---------------------------------------------------------------------------
# A. Circuit conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5])
def test_qubit_index_convention(n):
    """Repository qubit 0 is the most significant bit; Qiskit's is the least."""
    report = circuit_builder.verify_index_convention(n)
    assert report["matches_most_significant_convention"], report
    assert circuit_builder.to_qiskit_qubit(n, 0) == n - 1
    assert circuit_builder.to_qiskit_qubit(n, n - 1) == 0


def test_gate_order_angles_and_targets_are_preserved():
    gates = cc.build_circuit(3, 2, 1)
    circuit = circuit_builder.build_from_gates(3, gates)
    instructions = [(item.operation.name, tuple(circuit.find_bit(q).index
                                                for q in item.qubits),
                     float(getattr(item.operation, "params", [0.0])[0])
                     if getattr(item.operation, "params", None) else None)
                    for item in circuit.data]
    expected = [(gate.gate_type.lower(),
                 tuple(circuit_builder.to_qiskit_qubit(3, q) for q in gate.qubits),
                 gate.angle) for gate in gates]
    assert len(instructions) == len(expected)
    for got, want in zip(instructions, expected):
        assert got[0] == ("cx" if want[0] == "cnot" else want[0])
        assert got[1] == want[1]                      # control and target order
        if got[0] in ("rx", "rz"):
            assert got[2] == pytest.approx(want[2], abs=1e-15)
        else:
            assert got[2] is None


@pytest.mark.parametrize("n,r,order", [(3, 1, 1), (3, 3, 2), (4, 2, 1), (4, 2, 2), (5, 3, 2)])
def test_custom_and_qiskit_unitaries_agree(n, r, order):
    result = equivalence.compare_unitary(n, r, order)
    assert result["spectral"] < 1e-12, result
    assert abs(result["phase"]) < 1e-12


@pytest.mark.parametrize("n,r,order", [(3, 2, 1), (4, 1, 2)])
def test_statevectors_agree(n, r, order):
    result = equivalence.compare_statevector(n, r, order)
    assert result["infidelity"] < 1e-12, result


def test_phase_insensitive_comparison_ignores_a_global_phase():
    matrix = np.eye(4, dtype=complex)
    rotated = np.exp(1j * 0.7) * matrix
    assert equivalence.phase_insensitive(matrix, rotated)["spectral"] < 1e-12
    assert equivalence.phase_insensitive(matrix, matrix * 1.1)["spectral"] > 0.1


# ---------------------------------------------------------------------------
# B. Pre-transpilation resources
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,r,order", [(3, 2, 1), (4, 3, 1), (4, 3, 2), (5, 4, 2)])
def test_pre_transpilation_resources_match_phase1(n, r, order):
    circuit = circuit_builder.build(n, r, order)
    got = resources.circuit_resources(circuit)
    want = resources.logical_resources(n, r, order)
    assert got["cx"] == want["cx"]
    assert got["rz"] == want["rz"]
    assert got["rx"] == want["rx"]
    assert got["total_ops"] == want["total_ops"]
    assert got["depth"] == want["depth"]
    counted = cc.estimate_resources(n, r, order=order)
    assert got["cx"] == counted.num_cnot
    assert got["total_ops"] == counted.num_cnot + counted.num_rz + counted.num_rx


# ---------------------------------------------------------------------------
# C. Transpilation
# ---------------------------------------------------------------------------

def test_coupling_maps_are_symmetric_and_connected():
    n = 5
    maps = transpilation.topologies(n)
    assert set(maps) == {"ideal", "line", "ring", "star"}
    assert maps["ideal"] is None
    for name, coupling in maps.items():
        if coupling is None:
            continue
        raw = list(coupling.get_edges())
        assert all((b, a) in raw for a, b in raw), name   # symmetric coupling
        assert all(a != b for a, b in raw), name          # no self loops
        edges = {tuple(sorted(edge)) for edge in raw}
        assert (0, n - 1) in edges or name != "ring"
    assert (0, n - 1) in {tuple(sorted(edge)) for edge in maps["ring"].get_edges()}
    assert (0, 1) in {tuple(sorted(edge)) for edge in maps["star"].get_edges()}
    assert (1, 2) not in {tuple(sorted(edge)) for edge in maps["star"].get_edges()}


@pytest.mark.parametrize("n", [4, 5])
def test_line_routing_adds_no_two_qubit_gates(n):
    for order in (1, 2):
        circuit = circuit_builder.build(n, 3, order)
        for config in transpilation.configurations(n):
            if config["topology"] == "star":
                continue
            compiled = transpilation.run_transpilation(circuit, config)["circuit"]
            logical = resources.circuit_resources(circuit)
            got = resources.circuit_resources(compiled)
            assert got["cx"] == logical["cx"], config["name"]
            assert got["depth"] == logical["depth"], config["name"]


def test_star_routing_adds_two_qubit_gates():
    n = 4
    circuit = circuit_builder.build(n, 3, 1)
    star = [config for config in transpilation.configurations(n)
            if config["topology"] == "star"][0]
    compiled = transpilation.run_transpilation(circuit, star)["circuit"]
    assert resources.circuit_resources(compiled)["cx"] > resources.circuit_resources(circuit)["cx"]


def test_transpilation_is_deterministic_with_a_fixed_seed():
    n = 4
    circuit = circuit_builder.build(n, 3, 2)
    star = [config for config in transpilation.configurations(n)
            if config["topology"] == "star"][0]
    first = resources.circuit_resources(
        transpilation.run_transpilation(circuit, star)["circuit"])
    second = resources.circuit_resources(
        transpilation.run_transpilation(circuit, star)["circuit"])
    assert first == second


def test_transpiled_circuit_only_uses_the_basis():
    n = 4
    circuit = circuit_builder.build(n, 2, 2)
    for config in transpilation.configurations(n):
        compiled = transpilation.run_transpilation(circuit, config)["circuit"]
        assert {item.operation.name for item in compiled.data} <= set(
            circuit_builder.BASIS)


def test_layout_restoration_recovers_the_logical_state():
    n = 4
    circuit = circuit_builder.build(n, 2, 1)
    reference = noise_check.aer_density_matrix(circuit)
    for config in transpilation.configurations(n):
        compiled = transpilation.run_transpilation(circuit, config)["circuit"]
        noiseless = noise_check.aer_density_matrix(compiled)
        restored = layout.restore_order(noiseless, noiseless, reference, n)
        assert restored["residual_after_restore"] < 1e-10, config["name"]
        assert restored["rho"].shape == reference.shape
        assert sorted(restored["permutation"]) == list(range(n))
        if config["topology"] != "star":
            assert restored["identity"], config["name"]


# ---------------------------------------------------------------------------
# D. Simulator execution and noise conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,r,order", [(3, 2, 1), (4, 2, 2)])
def test_aer_density_matrix_is_physical(n, r, order):
    circuit = circuit_builder.build(n, r, order)
    rho = noise_check.aer_density_matrix(circuit)
    assert np.max(np.abs(rho - rho.conj().T)) < 1e-12
    assert abs(np.trace(rho) - 1.0) < 1e-12
    assert np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2)) > -1e-12
    assert abs(np.trace(rho @ rho) - 1.0) < 1e-10        # ideal evolution is pure


@pytest.mark.parametrize("n,r,order", [(3, 2, 1), (3, 3, 2), (4, 2, 1)])
def test_aer_reproduces_the_custom_noisy_density_matrix(n, r, order):
    result = noise_check.logical_circuit_agreement(n, r, order, 1e-3, 1e-3)
    assert result["max_difference"] < 1e-12


def test_project_channel_is_reproduced_exactly_as_a_pauli_mixture():
    for p in (1e-3, 0.05):
        assert noise_check.single_qubit_agreement(p)["max_difference"] < 1e-12
        assert noise_check.two_qubit_agreement(p)["max_difference"] < 1e-12


def test_native_depolarizing_error_is_a_different_channel():
    """Aer's ``depolarizing_error`` is not the project's mixture; the measured
    conversion factors are p_aer = 4p/3 (one qubit) and 16p/15 (two qubits)."""
    report = noise_check.channel_convention(levels=(0.05,))
    assert report["native_max_difference"] > 1e-3
    assert report["pauli_mixture_is_identical"]
    assert report["pauli_mixture_max_difference"] < 1e-12


# ---------------------------------------------------------------------------
# E. Conversion factors, checked against the channels themselves
# ---------------------------------------------------------------------------

def _aer_single(p_aer, angle=0.37):
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    circuit = QuantumCircuit(1)
    circuit.rx(angle, 0)
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p_aer, 1), ["rx"])
    return noise_check.aer_density_matrix(circuit, model)


def _custom_single(p, angle=0.37):
    from validation import independent as ind
    psi = ind._rx(angle) @ np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    return ind.depolarize_explicit(rho, (0,), p)


@pytest.mark.parametrize("p", [0.01, 0.1])
def test_documented_conversion_factor_for_one_qubit_noise(p):
    assert np.max(np.abs(_aer_single(4 * p / 3) - _custom_single(p))) < 1e-12
    assert np.max(np.abs(_aer_single(p) - _custom_single(p))) > 1e-6
