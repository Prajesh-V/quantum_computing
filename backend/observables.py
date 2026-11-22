"""Physical observables and TFIM energy estimation from measurement counts.

Evaluates spin chain observables from finite-shot bitstrings:
- Nearest-neighbour ZZ correlations <Z_i Z_{i+1}> (direct computational basis shots).
- Transverse-field magnetization <X_i> (via basis-rotation shots).
- Total Hamiltonian energy <H> = -J sum <Z_i Z_{i+1}> - h sum <X_i>.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import circuit_cost
from .interface import QuantumBackend
from .mitigation import calibrate_readout_matrix, mitigate_readout_counts

__all__ = [
    "expectation_zz_from_counts",
    "expectation_z_from_counts",
    "add_x_basis_rotation",
    "measure_tfim_energy",
    "exact_tfim_energy",
]


def expectation_z_from_counts(counts_or_probs: Dict[str, Union[int, float]], qubit: int) -> float:
    """<Z_q> = sum_x P(x) * (-1)^{x_q}."""
    total_val = 0.0
    total_weight = 0.0
    for bitstr, weight in counts_or_probs.items():
        val = 1.0 if bitstr[qubit] == "0" else -1.0
        total_val += weight * val
        total_weight += weight
    return total_val / total_weight if total_weight > 0 else 0.0


def expectation_zz_from_counts(counts_or_probs: Dict[str, Union[int, float]], q0: int, q1: int) -> float:
    """<Z_q0 Z_q1> = sum_x P(x) * (-1)^{x_q0 ^ x_q1}."""
    total_val = 0.0
    total_weight = 0.0
    for bitstr, weight in counts_or_probs.items():
        bit0 = int(bitstr[q0])
        bit1 = int(bitstr[q1])
        val = 1.0 if (bit0 ^ bit1) == 0 else -1.0
        total_val += weight * val
        total_weight += weight
    return total_val / total_weight if total_weight > 0 else 0.0


def add_x_basis_rotation(gates: List[circuit_cost.Gate], n: int) -> List[circuit_cost.Gate]:
    """Append basis rotation on all qubits so that measuring in Z measures X.

    Uses RZ(pi/2) RX(pi/2) RZ(pi/2) (Hadamard-equivalent basis transform).
    """
    rotated = list(gates)
    for q in range(n):
        rotated.append(circuit_cost.Gate("rz", (q,), math.pi / 2))
        rotated.append(circuit_cost.Gate("rx", (q,), math.pi / 2))
        rotated.append(circuit_cost.Gate("rz", (q,), math.pi / 2))
    return rotated


def exact_tfim_energy(n: int, t: float, J: float = 1.0, h: float = 1.0) -> float:
    """Theoretical energy <psi(t)| H |psi(t)> = <psi0| H |psi0> (conserved under exact evolution)."""
    import trotter
    A, B, H = trotter.tfim_terms(n, J, h)
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    return float(np.real(np.vdot(psi0, H @ psi0)))


def measure_tfim_energy(backend: QuantumBackend, n: int, r: int, order: int = 1,
                        t: float = 1.0, J: float = 1.0, h: float = 1.0,
                        shots: int = 4096, seed: int = 42,
                        mitigate_readout: bool = False,
                        assignment_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Measure <H> = -J sum <Z_i Z_{i+1}> - h sum <X_i> on the given backend."""
    base_gates = circuit_cost.build_circuit(n, r, t, J, h, order, merge_half_steps=True)

    # 1. Measure Z-basis (ZZ correlations)
    job_z = backend.submit(base_gates, shots=shots, seed=seed)
    res_z = job_z.result()
    counts_z = res_z.counts

    # 2. Measure X-basis (transverse field terms)
    x_gates = add_x_basis_rotation(base_gates, n)
    job_x = backend.submit(x_gates, shots=shots, seed=seed + 1)
    res_x = job_x.result()
    counts_x = res_x.counts

    # Mitigation if requested
    if mitigate_readout:
        if assignment_matrix is None:
            assignment_matrix = calibrate_readout_matrix(backend, n, shots=shots, seed=seed + 10)
        eval_z = mitigate_readout_counts(counts_z, assignment_matrix, shots)
        eval_x = mitigate_readout_counts(counts_x, assignment_matrix, shots)
    else:
        eval_z = counts_z
        eval_x = counts_x

    # Calculate ZZ terms
    zz_terms = []
    for q in range(n - 1):
        val = expectation_zz_from_counts(eval_z, q, q + 1)
        zz_terms.append(val)
    mean_zz = float(np.mean(zz_terms)) if zz_terms else 0.0

    # Calculate X terms
    x_terms = []
    for q in range(n):
        val = expectation_z_from_counts(eval_x, q)
        x_terms.append(val)
    mean_x = float(np.mean(x_terms)) if x_terms else 0.0

    # Total energy H = -J sum ZZ - h sum X
    measured_energy = float(-J * np.sum(zz_terms) - h * np.sum(x_terms))
    exact_energy = exact_tfim_energy(n, t, J, h)
    energy_error = abs(measured_energy - exact_energy)

    return {
        "system_size": n,
        "trotter_steps": r,
        "trotter_order": order,
        "measured_energy": measured_energy,
        "exact_energy": exact_energy,
        "energy_error": energy_error,
        "mean_zz": mean_zz,
        "mean_x": mean_x,
        "shots": shots,
        "mitigated": mitigate_readout,
    }
