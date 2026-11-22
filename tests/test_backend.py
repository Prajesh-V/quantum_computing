"""Tests for quantum execution backends, mock devices, error mitigation, and observables."""

import math
import numpy as np
import pytest

import circuit_cost
from backend import (
    AerBackend,
    JobStatus,
    LocalDensityMatrixBackend,
    LocalStatevectorBackend,
    MockHardwareBackend,
    calibrate_readout_matrix,
    create_mock_falcon,
    exact_tfim_energy,
    fold_circuit_cnot,
    measure_tfim_energy,
    mitigate_readout_counts,
    richardson_extrapolate,
)
from noise.noisy_simulator import NoiseModel


def test_local_statevector_backend():
    backend = LocalStatevectorBackend(num_qubits=4)
    gates = circuit_cost.build_circuit(4, 2, 1.0, order=2)
    job = backend.submit(gates, shots=1000, seed=42)
    assert job.status() == JobStatus.COMPLETED
    res = job.result()
    assert res.shots == 1000
    assert sum(res.counts.values()) == 1000
    assert all(len(b) == 4 for b in res.counts.keys())


def test_local_density_matrix_backend():
    model = NoiseModel(kind="gate", single_qubit_error=1e-3, two_qubit_error=1e-3)
    backend = LocalDensityMatrixBackend(num_qubits=4, noise_model=model)
    gates = circuit_cost.build_circuit(4, 2, 1.0, order=1)
    res = backend.run(gates, shots=500, seed=7)
    assert res.shots == 500
    assert sum(res.counts.values()) == 500


def test_aer_backend():
    backend = AerBackend(num_qubits=4)
    gates = circuit_cost.build_circuit(3, 2, 1.0, order=1)
    res = backend.run(gates, shots=500, seed=42)
    assert res.shots == 500
    assert sum(res.counts.values()) == 500


def test_mock_falcon_device_coupling_and_readout():
    falcon = create_mock_falcon()
    assert falcon.properties.num_qubits == 7
    assert falcon.properties.coupling_map is not None

    # Valid linear chain on Falcon edges: (0, 1), (1, 3), (3, 5)
    # A 3-qubit circuit with line connectivity 0-1-2
    gates = [
        circuit_cost.Gate("rx", (0,), 0.5),
        circuit_cost.Gate("cnot", (0, 1), 0.0),
    ]
    res = falcon.run(gates, shots=200, seed=42)
    assert res.shots == 200

    # Invalid CNOT violating coupling map
    invalid_gates = [circuit_cost.Gate("cnot", (0, 6), 0.0)]
    with pytest.raises(ValueError, match="violates device coupling map"):
        falcon.run(invalid_gates, shots=100)


def test_readout_mitigation_unfolding():
    falcon = create_mock_falcon()
    calib = calibrate_readout_matrix(falcon, num_qubits=2, shots=2000, seed=42)
    assert calib.shape == (4, 4)
    # Diagonal dominant
    for i in range(4):
        assert calib[i, i] > 0.85

    # Mitigate counts
    raw_counts = {"00": 850, "01": 50, "10": 70, "11": 30}
    mit_probs = mitigate_readout_counts(raw_counts, calib, shots=1000)
    assert abs(sum(mit_probs.values()) - 1.0) < 1e-4
    assert all(p >= 0.0 for p in mit_probs.values())


def test_cnot_folding_and_richardson():
    gates = [
        circuit_cost.Gate("rx", (0,), 0.5),
        circuit_cost.Gate("cnot", (0, 1), 0.0),
        circuit_cost.Gate("rz", (1,), 0.2),
    ]
    folded3 = fold_circuit_cnot(gates, scale_factor=3)
    cnot_count = sum(1 for g in folded3 if g.gate_type == "cnot")
    assert cnot_count == 3

    # Extrapolation
    scales = [1.0, 3.0, 5.0]
    # Linear model: y = 2.0 + 0.1 * scale
    vals = [2.0 + 0.1 * s for s in scales]
    extrap = richardson_extrapolate(scales, vals, order=1)
    assert extrap == pytest.approx(2.0, abs=1e-8)


def test_measure_tfim_energy():
    backend = LocalStatevectorBackend(num_qubits=4)
    res = measure_tfim_energy(backend, n=4, r=4, order=2, shots=4000, seed=42)
    exact = exact_tfim_energy(4, 1.0, 1.0, 1.0)
    assert abs(res["measured_energy"] - exact) < 0.2
