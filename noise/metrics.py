"""Fidelity, infidelity and trace distance for Phase 3.

The primary research metric is the fidelity against a **pure** reference state,
which is what ``fidelity_pure`` computes.  Three different references are used
and are never confused:

* exact evolution ``exp(-iHt)|psi0>``           -> total / Trotter error
* ideal Trotter evolution (the gate circuit)     -> noise-induced error
* the noisy density matrix itself                -> reported fidelity

A noisy density matrix and an ideal pure reference are enough for all three,
so the cheap ``<psi|rho|psi>`` form is used everywhere in the experiments;
``fidelity_general`` (Uhlmann) is provided for density-matrix pairs and is used
only in tests and diagnostics.
"""

from __future__ import annotations

import numpy as np

from .density_matrix import num_qubits, purity  # noqa: F401  (purity re-exported)

__all__ = [
    "fidelity_pure",
    "infidelity",
    "fidelity_general",
    "trace_distance",
    "clip_unit_interval",
    "purity",
]


def clip_unit_interval(value: float) -> float:
    """Clip to ``[0, 1]``; round-off can push a fidelity a hair outside."""
    return float(min(1.0, max(0.0, value)))


def _reference_vector(psi) -> np.ndarray:
    vector = np.asarray(psi, dtype=complex).reshape(-1)
    num_qubits(vector.size)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("reference state vector must not be zero")
    return vector / norm


def fidelity_pure(rho, psi) -> float:
    """``F(rho, |psi>) = <psi| rho |psi>`` for a pure reference state."""
    array = np.asarray(rho, dtype=complex)
    vector = _reference_vector(psi)
    if array.shape != (vector.size, vector.size):
        raise ValueError(
            f"rho shape {array.shape} does not match a {vector.size}-dimensional state"
        )
    value = np.vdot(vector, array @ vector)          # vdot conjugates its first argument
    return clip_unit_interval(float(np.real(value)))


def infidelity(rho, psi) -> float:
    """``1 - F(rho, |psi>)``."""
    return clip_unit_interval(1.0 - fidelity_pure(rho, psi))


def fidelity_general(rho, sigma) -> float:
    """Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2`` for two states."""
    a = np.asarray(rho, dtype=complex)
    b = np.asarray(sigma, dtype=complex)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")

    def _sqrt(matrix):
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
        values = np.clip(values, 0.0, None)
        return (vectors * np.sqrt(values)) @ vectors.conj().T

    root = _sqrt(a)
    inner = root @ b @ root
    values = np.linalg.eigvalsh(0.5 * (inner + inner.conj().T))
    return clip_unit_interval(float(np.sum(np.sqrt(np.clip(values, 0.0, None)))) ** 2)


def trace_distance(rho, sigma) -> float:
    """``D(rho, sigma) = 0.5 * ||rho - sigma||_1`` via the eigenvalues of the difference."""
    a = np.asarray(rho, dtype=complex)
    b = np.asarray(sigma, dtype=complex)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    difference = a - b
    herm = 0.5 * (difference + difference.conj().T)
    values = np.linalg.eigvalsh(herm)
    return float(0.5 * np.sum(np.abs(values)))
