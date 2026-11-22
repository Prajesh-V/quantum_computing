"""Thermal relaxation and dephasing channels (T1, T2, pure dephasing).

Extends the Phase 3 noise models with realistic continuous-time decoherence
processes parameterized by physical relaxation time T1, coherence time T2,
and gate durations t_gate.

Physical constraints:
--------------------
- T1: longitudinal relaxation time (decay to ground state |0>).
- T2: transverse dephasing time, satisfying T2 <= 2 * T1.
- T_phi: pure dephasing time, 1 / T_phi = 1 / T2 - 1 / (2 * T1).
- Single-qubit Kraus representation:
    gamma = 1 - exp(-t_gate / T1)
    lambda = 1 - exp(-t_gate / T_phi)
  Composed amplitude damping and pure dephasing yields a completely positive
  trace-preserving (CPTP) map with 4 Kraus operators:
    K_0 = [[1,               0              ], [0, sqrt(1 - gamma) * sqrt(1 - lambda)]]
    K_1 = [[0,               0              ], [0, sqrt(1 - gamma) * sqrt(lambda)    ]]
    K_2 = [[0, sqrt(gamma)                  ], [0, 0                                  ]]
    K_3 = [[0, 0                            ], [0, 0                                  ]]  # zero / omitted

All Kraus sets are verified to satisfy sum_k K_k^dagger K_k = I to machine precision.
"""

from __future__ import annotations

import math
import numpy as np

from .density_matrix import apply_operator, num_qubits, _check_qubits

__all__ = [
    "validate_probability",
    "amplitude_damping_kraus",
    "pure_dephasing_kraus",
    "thermal_relaxation_kraus",
    "apply_kraus_channel",
    "two_qubit_relaxation_channel",
    "verify_cptp",
    "bloch_vector",
]


def validate_probability(p, name: str = "p") -> float:
    """Return ``p`` as a float, rejecting anything outside ``[0, 1]``."""
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(p).__name__}")
    value = float(p)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {p!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return value


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Kraus operators for amplitude damping with probability gamma = 1 - exp(-t/T1).

    K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
    K_1 = [[0, sqrt(gamma)], [0, 0]]
    """
    gamma = validate_probability(gamma, "gamma")
    k0 = np.array([[1.0, 0.0],
                   [0.0, math.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(gamma)],
                   [0.0, 0.0]], dtype=complex)
    return [k0, k1]


def pure_dephasing_kraus(lambd: float) -> list[np.ndarray]:
    """Kraus operators for pure phase damping with probability lambda = 1 - exp(-t/T_phi).

    K_0 = [[1, 0], [0, sqrt(1 - lambda)]]
    K_1 = [[0, 0], [0, math.sqrt(lambda)]]
    """
    lambd = validate_probability(lambd, "lambda")
    k0 = np.array([[1.0, 0.0],
                   [0.0, math.sqrt(1.0 - lambd)]], dtype=complex)
    k1 = np.array([[0.0, 0.0],
                   [0.0, math.sqrt(lambd)]], dtype=complex)
    return [k0, k1]


def thermal_relaxation_kraus(t_gate: float, t1: float, t2: float) -> list[np.ndarray]:
    """Kraus operators for combined T1 relaxation and T2 dephasing during gate time t_gate.

    Requires t_gate >= 0, t1 > 0, t2 > 0, and t2 <= 2 * t1 (physical Lindblad constraint).
    """
    if t_gate < 0:
        raise ValueError(f"gate duration t_gate must be >= 0, got {t_gate}")
    if t1 <= 0:
        raise ValueError(f"T1 must be positive, got {t1}")
    if t2 <= 0:
        raise ValueError(f"T2 must be positive, got {t2}")
    if t2 > 2.0 * t1 + 1e-14:
        raise ValueError(f"Unphysical parameters: T2 ({t2:g}) cannot exceed 2 * T1 ({2*t1:g})")

    if t_gate == 0.0:
        return [np.eye(2, dtype=complex)]

    gamma = 1.0 - math.exp(-t_gate / t1)
    inv_tphi = max(0.0, 1.0 / t2 - 1.0 / (2.0 * t1))
    lambd = 1.0 - math.exp(-2.0 * t_gate * inv_tphi)

    # Composition of amplitude damping followed by pure dephasing:
    # E(rho) = E_deph(E_amp(rho)) = sum_{i,j} K_deph,j K_amp,i rho K_amp,i^dagger K_deph,j^dagger
    k_amp = amplitude_damping_kraus(gamma)
    k_deph = pure_dephasing_kraus(lambd)

    kraus_ops = []
    for kd in k_deph:
        for ka in k_amp:
            op = kd @ ka
            # Keep non-zero operators
            if np.linalg.norm(op) > 1e-15:
                kraus_ops.append(op)

    return kraus_ops


def apply_kraus_channel(rho: np.ndarray, kraus_ops: list[np.ndarray], qubits: tuple[int, ...]) -> np.ndarray:
    """Apply a CPTP channel defined by Kraus operators {K_k} to specified qubits of rho.

    E(rho) = sum_k K_k rho K_k^dagger
    """
    array = np.asarray(rho, dtype=complex)
    out = np.zeros_like(array)
    for k_op in kraus_ops:
        out += apply_operator(array, k_op, qubits)
    return out


def two_qubit_relaxation_channel(rho: np.ndarray, qubits: tuple[int, int],
                                 t_gate: float, t1: float, t2: float) -> np.ndarray:
    """Apply independent thermal relaxation to both qubits of a two-qubit gate over duration t_gate."""
    kraus_1q = thermal_relaxation_kraus(t_gate, t1, t2)
    # Apply to qubit 0 then qubit 1
    rho1 = apply_kraus_channel(rho, kraus_1q, (qubits[0],))
    rho2 = apply_kraus_channel(rho1, kraus_1q, (qubits[1],))
    return rho2


def verify_cptp(kraus_ops: list[np.ndarray], tol: float = 1e-12) -> tuple[bool, float]:
    """Verify trace preservation sum_k K_k^dagger K_k = I."""
    dim = kraus_ops[0].shape[0]
    total = np.zeros((dim, dim), dtype=complex)
    for op in kraus_ops:
        total += op.conj().T @ op
    identity = np.eye(dim, dtype=complex)
    max_dev = float(np.max(np.abs(total - identity)))
    return (max_dev <= tol, max_dev)


def bloch_vector(rho_1q: np.ndarray) -> tuple[float, float, float]:
    """Extract (rx, ry, rz) Bloch coordinates for a 1-qubit density matrix."""
    x = float(np.real(rho_1q[0, 1] + rho_1q[1, 0]))
    y = float(np.real(1j * (rho_1q[0, 1] - rho_1q[1, 0])))
    z = float(np.real(rho_1q[0, 0] - rho_1q[1, 1]))
    return (x, y, z)
