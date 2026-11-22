"""Tests for quantum trajectory (Monte Carlo Wavefunction) simulation."""

import math
import numpy as np
import pytest

from noise import noisy_simulator
from noise.channels import PAULI_1Q
from noise.trajectories import (
    apply_gate_statevector,
    compare_trajectory_with_density_matrix,
    depolarizing_kraus_1q,
    depolarizing_kraus_2q,
    run_trajectory,
    sample_kraus_branch,
    simulate_trajectories,
    statevector_magnetization_x,
    statevector_zz_correlation,
)


def test_apply_gate_statevector():
    n = 3
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1.0  # |000>

    # Apply X on qubit 1 -> |010> (index 2)
    X = PAULI_1Q["X"]
    psi_out = apply_gate_statevector(psi, X, (1,), n)
    assert abs(psi_out[2] - 1.0) < 1e-12
    assert abs(np.linalg.norm(psi_out) - 1.0) < 1e-12


def test_depolarizing_kraus_completeness():
    k1 = depolarizing_kraus_1q(0.01)
    sum_1q = sum(k.conj().T @ k for k in k1)
    assert np.allclose(sum_1q, np.eye(2), atol=1e-12)

    k2 = depolarizing_kraus_2q(0.01)
    sum_2q = sum(k.conj().T @ k for k in k2)
    assert np.allclose(sum_2q, np.eye(4), atol=1e-12)


def test_sample_kraus_branch_normalization():
    rng = np.random.default_rng(42)
    psi = np.array([0.6, 0.8], dtype=complex)
    k1 = depolarizing_kraus_1q(0.1)
    new_psi, branch = sample_kraus_branch(psi, k1, (0,), 1, rng)
    assert abs(np.linalg.norm(new_psi) - 1.0) < 1e-12
    assert 0 <= branch < len(k1)


def test_trajectory_execution_noiseless():
    # Ideal trajectory must match ideal Trotter state exactly
    n = 4
    r = 2
    order = 2
    psi_traj = run_trajectory(n, r, order, model=noisy_simulator.NoiseModel(kind="none"))
    psi_ref = noisy_simulator.ideal_trotter_state(n, r, order)
    overlap = abs(np.vdot(psi_ref, psi_traj))
    assert abs(overlap - 1.0) < 1e-10


def test_trajectory_convergence_to_density_matrix():
    # In the limit of trajectories, MCWF ensemble matches density matrix
    res = compare_trajectory_with_density_matrix(
        n=3,
        r=2,
        order=2,
        model=noisy_simulator.NoiseModel(kind="gate", single_qubit_error=1e-3, two_qubit_error=1e-3),
        num_trajectories=200,
        seed=123,
    )
    assert res["passed"], f"Trajectory did not match density matrix: diff={res['difference']:.4f}, sem={res['sem']:.4f}"


def test_trajectory_scaling_large_system():
    # n = 8 spins, 256 statevector dimension
    res = simulate_trajectories(
        n=8,
        r=2,
        order=1,
        model=noisy_simulator.NoiseModel(kind="gate", single_qubit_error=1e-3, two_qubit_error=1e-3),
        num_trajectories=5,
        seed=42,
    )
    assert res["system_size"] == 8
    assert 0.0 <= res["fidelity"] <= 1.0
    assert -1.0 <= res["magnetization"] <= 1.0
