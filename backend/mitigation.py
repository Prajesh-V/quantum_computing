"""Quantum error mitigation: readout error unfolding and Zero-Noise Extrapolation (ZNE).

Includes:
1. Readout error calibration and matrix inversion unfolding (with physical simplex projection).
2. Zero-Noise Extrapolation (ZNE) via deterministic CNOT gate folding (CNOT -> CNOT CNOT CNOT).
3. Richardson polynomial extrapolation to zero noise limit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import circuit_cost
from .interface import QuantumBackend

__all__ = [
    "calibrate_readout_matrix",
    "mitigate_readout_counts",
    "fold_circuit_cnot",
    "richardson_extrapolate",
]


def calibrate_readout_matrix(backend: QuantumBackend, num_qubits: int,
                             shots: int = 4096, seed: int = 42) -> np.ndarray:
    """Measure the 2**n x 2**n assignment matrix M_{i,j} = P(measured i | prepared j)."""
    dim = 2 ** num_qubits
    matrix = np.zeros((dim, dim), dtype=float)

    for prep_j in range(dim):
        # Prepare state |prep_j>
        bitstr = f"{prep_j:0{num_qubits}b}"
        gates: List[circuit_cost.Gate] = []
        for q_idx, bit in enumerate(bitstr):
            if bit == "1":
                # Apply RX(pi) as an X flip
                gates.append(circuit_cost.Gate("rx", (q_idx,), math.pi))

        job = backend.submit(gates, shots=shots, seed=seed + prep_j, num_qubits=num_qubits)
        res = job.result()

        for meas_str, count in res.counts.items():
            meas_i = int(meas_str, 2)
            if meas_i < dim:
                matrix[meas_i, prep_j] = count / shots

    return matrix


def project_to_probability_simplex(v: np.ndarray) -> np.ndarray:
    """Project a real vector onto the probability simplex sum_i p_i = 1, p_i >= 0 (Duchi et al.)."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def mitigate_readout_counts(counts: Dict[str, int], assignment_matrix: np.ndarray,
                            shots: int) -> Dict[str, float]:
    """Invert the readout assignment matrix and project onto the probability simplex."""
    dim = assignment_matrix.shape[0]
    n = int(math.log2(dim))
    p_meas = np.zeros(dim, dtype=float)

    for bitstr, count in counts.items():
        idx = int(bitstr, 2)
        if idx < dim:
            p_meas[idx] = count / shots

    # Invert assignment matrix with regularisation
    inv_m = np.linalg.pinv(assignment_matrix, rcond=1e-6)
    p_raw_mit = inv_m @ p_meas

    # Project onto physical probability simplex
    p_mit = project_to_probability_simplex(p_raw_mit)

    mitigated_counts: Dict[str, float] = {}
    for idx, p in enumerate(p_mit):
        if p > 1e-7:
            bitstr = f"{idx:0{n}b}"
            mitigated_counts[bitstr] = float(p)

    return mitigated_counts


def fold_circuit_cnot(gates: Sequence[circuit_cost.Gate], scale_factor: int = 3) -> List[circuit_cost.Gate]:
    """Unitary gate folding for Zero-Noise Extrapolation: CNOT -> (CNOT CNOT CNOT ...).

    Since CNOT * CNOT = Identity, replacing each CNOT with (2k + 1) CNOTs preserves the exact
    ideal unitary while scaling the effective two-qubit gate noise by scale_factor = 2k + 1.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(f"scale_factor must be an odd positive integer (1, 3, 5,...), got {scale_factor}")

    folded: List[circuit_cost.Gate] = []
    for g in gates:
        if g.gate_type == "cnot":
            for _ in range(scale_factor):
                folded.append(g)
        else:
            folded.append(g)
    return folded


def richardson_extrapolate(scale_factors: Sequence[float], values: Sequence[float],
                           order: int = 1) -> float:
    """Polynomial Richardson extrapolation to the zero-noise limit c -> 0."""
    scales = np.asarray(scale_factors, dtype=float)
    vals = np.asarray(values, dtype=float)
    if len(scales) != len(vals) or len(scales) < 2:
        raise ValueError("At least 2 points needed for extrapolation")

    deg = min(order, len(scales) - 1)
    coeffs = np.polyfit(scales, vals, deg=deg)
    # The zero-noise value is the constant term coeffs[-1]
    return float(coeffs[-1])
