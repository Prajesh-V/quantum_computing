"""Tests for thermal relaxation (T1, T2) channels and Lindblad dynamics."""

import math
import numpy as np
import pytest

from noise import density_matrix, noisy_simulator
from noise.relaxation import (
    amplitude_damping_kraus,
    apply_kraus_channel,
    bloch_vector,
    pure_dephasing_kraus,
    thermal_relaxation_kraus,
    two_qubit_relaxation_channel,
    verify_cptp,
)


def test_amplitude_damping_cptp():
    for gamma in (0.0, 0.05, 0.5, 0.99, 1.0):
        kraus = amplitude_damping_kraus(gamma)
        ok, dev = verify_cptp(kraus)
        assert ok, f"gamma={gamma} failed CPTP with max dev {dev}"


def test_pure_dephasing_cptp():
    for lambd in (0.0, 0.05, 0.5, 0.99, 1.0):
        kraus = pure_dephasing_kraus(lambd)
        ok, dev = verify_cptp(kraus)
        assert ok, f"lambda={lambd} failed CPTP with max dev {dev}"


def test_thermal_relaxation_cptp():
    t1 = 100e-6
    t2 = 80e-6
    for t_gate in (0.0, 10e-9, 35e-9, 300e-9, 1e-6):
        kraus = thermal_relaxation_kraus(t_gate, t1, t2)
        ok, dev = verify_cptp(kraus)
        assert ok, f"t_gate={t_gate} failed CPTP with max dev {dev}"


def test_thermal_relaxation_unphysical_rejection():
    with pytest.raises(ValueError, match="cannot exceed 2 \\* T1"):
        thermal_relaxation_kraus(35e-9, t1=50e-6, t2=110e-6)  # T2 > 2*T1


def test_bloch_decay_dynamics():
    """Verify that repeated relaxation channels follow exact exponential decay laws."""
    t1 = 100e-6
    t2 = 80e-6
    t_gate = 100e-9
    kraus = thermal_relaxation_kraus(t_gate, t1, t2)

    # Prepare |+> state: rx = 1, ry = 0, rz = 0
    psi_plus = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    rho = np.outer(psi_plus, psi_plus.conj())

    # Apply 100 steps of relaxation
    n_steps = 100
    for _ in range(n_steps):
        rho = apply_kraus_channel(rho, kraus, (0,))

    rx, ry, rz = bloch_vector(rho)
    total_time = n_steps * t_gate

    expected_rx = math.exp(-total_time / t2)
    expected_rz = 1.0 - math.exp(-total_time / t1)  # decay towards |0> (rz = +1)

    assert rx == pytest.approx(expected_rx, abs=1e-4)
    assert rz == pytest.approx(expected_rz, abs=1e-4)
    assert abs(ry) < 1e-10


def test_two_qubit_relaxation():
    # 2-qubit density matrix
    rho = np.zeros((4, 4), dtype=complex)
    rho[3, 3] = 1.0  # |11>
    t1 = 100e-6
    t2 = 80e-6
    rho_rel = two_qubit_relaxation_channel(rho, (0, 1), 300e-9, t1, t2)

    check = density_matrix.check_density_matrix(rho_rel)
    assert check["ok"]
    assert abs(density_matrix.trace(rho_rel) - 1.0) < 1e-12


def test_noisy_simulator_with_relaxation():
    model = noisy_simulator.NoiseModel(
        kind="relaxation",
        t1=100e-6,
        t2=80e-6,
        gate_time_1q=35e-9,
        gate_time_2q=300e-9,
    )
    res = noisy_simulator.simulate(4, 2, order=2, model=model)
    assert res["num_channels"] > 0
    check = density_matrix.check_density_matrix(res["rho"])
    assert check["ok"]
