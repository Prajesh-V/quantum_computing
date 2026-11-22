"""Depolarising noise channels and the partial-trace identity behind them.

Three models are implemented, exactly as specified for Phase 3.

Global depolarising channel
---------------------------
    E_g(rho) = (1 - p_g) rho + p_g I/d,      d = 2**n

``p_g`` is the probability, *per application*, that the state is replaced by the
maximally mixed state.  In the primary experiment it is applied once after every
complete Trotter step, so it is a **per-Trotter-step** mixing probability and is
**not** comparable to a per-gate error rate.  A single end-of-circuit
application is also supported, and is reported separately.

Single-qubit Pauli depolarising channel (after one 1-qubit gate)
---------------------------------------------------------------
    E_1(rho) = (1 - p_1) rho + (p_1/3) (X rho X + Y rho Y + Z rho Z)

``p_1`` is the total probability that a non-identity Pauli is applied to the
target qubit after the gate; each of X, Y, Z therefore has probability p_1/3.
Identity errors (probability 1 - p_1) leave the state unchanged.

Note what the parameter means at its extremes: ``p_1 = 1`` says "one of X, Y, Z
for certain", which shrinks the Bloch vector to ``1 - 4/3 = -1/3`` of its
length rather than depolarising the qubit completely. Complete depolarisation
of a single qubit happens at ``p_1 = 3/4``, where the shrink factor
``1 - 4 p_1 / 3`` reaches zero. Both are valid channels; the experiments in
this phase use ``p_1 <= 1e-2``, well inside the small-perturbation regime, and
the tests check the ``1 - 4 p_1 / 3`` shrink law analytically.

Two-qubit Pauli depolarising channel (after a CNOT)
--------------------------------------------------
    E_2(rho) = (1 - p_2) rho + (p_2/15) sum_{P != II} P rho P

with ``P`` running over the 15 non-identity elements of ``{I,X,Y,Z}^{tensor 2}``
acting on the control and target qubits of the CNOT.

Implementation note: the partial-trace identity
----------------------------------------------
Summing a Pauli conjugation over the *whole* Pauli group of a ``k``-qubit
subsystem has a closed form,

    sum_{P in Pauli_k} P rho P^dagger = 2**k * Tr_k(rho) (x) I_k,

so the sum over the non-identity Paulis is that expression minus ``rho`` itself.
``depolarize_qubits`` uses this form (one partial trace instead of 3 or 15 tensor
contractions), and ``explicit_pauli_depolarizing`` evaluates the sum term by term.
The test suite checks that the two agree to machine precision, so the fast path
is validated against a direct transcription of the definition.
"""

from __future__ import annotations

import math

import numpy as np

from .density_matrix import (apply_operator, num_qubits, partial_trace)

__all__ = [
    "PAULI_1Q",
    "validate_probability",
    "maximally_mixed",
    "apply_global_depolarizing",
    "pauli_operators",
    "explicit_pauli_depolarizing",
    "depolarize_qubits",
    "SINGLE_QUBIT_NON_IDENTITY",
    "TWO_QUBIT_NON_IDENTITY",
]

PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

#: 3 non-identity single-qubit Paulis: the denominator of the p_1/3 weighting.
SINGLE_QUBIT_NON_IDENTITY = 3

#: 15 non-identity two-qubit Paulis: the denominator of the p_2/15 weighting.
TWO_QUBIT_NON_IDENTITY = 15


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_probability(p, name: str = "p") -> float:
    """Return ``p`` as a float, rejecting anything outside ``[0, 1]``.

    Non-numeric, NaN and infinite values are rejected explicitly rather than
    allowed to propagate into a simulation and produce a silent NaN state.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(p).__name__}")
    value = float(p)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {p!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return value


def maximally_mixed(dim: int) -> np.ndarray:
    """``I / dim``, the fixed point of every depolarising channel."""
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"dimension must be positive, got {dim}")
    return np.eye(dim, dtype=complex) / dim


# ---------------------------------------------------------------------------
# Global channel
# ---------------------------------------------------------------------------

def apply_global_depolarizing(rho, p: float) -> np.ndarray:
    """``(1 - p) rho + p I/d``."""
    p = validate_probability(p, "p_g")
    array = np.asarray(rho, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"rho must be square, got shape {array.shape}")
    d = array.shape[0]
    num_qubits(d)
    if p == 0.0:
        return array.copy()
    if p == 1.0:
        return maximally_mixed(d)
    return (1.0 - p) * array + p * maximally_mixed(d)


# ---------------------------------------------------------------------------
# Pauli-group helpers
# ---------------------------------------------------------------------------

def pauli_operators(k: int):
    """All ``4**k`` Pauli operators on ``k`` qubits, as ``(label, matrix)`` pairs.

    The Kronecker order follows the repository convention: the first letter is
    the most significant qubit of the operator's own 2**k-dimensional space.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    letters = ("I", "X", "Y", "Z")
    ops = [(letter, PAULI_1Q[letter]) for letter in letters]
    for _ in range(k - 1):
        ops = [(label + letter, np.kron(matrix, single))
               for label, matrix in ops
               for letter, single in ((l, PAULI_1Q[l]) for l in letters)]
    return ops


def explicit_pauli_depolarizing(rho, qubits, p: float) -> np.ndarray:
    """The channel evaluated literally as a sum over non-identity Paulis.

    Slow (3 tensor contractions for one qubit, 15 for two), but a direct
    transcription of the definition, so it is the reference the fast path is
    tested against.
    """
    p = validate_probability(p)
    array = np.asarray(rho, dtype=complex)
    n = num_qubits(array.shape[0])
    qs = tuple(int(q) for q in qubits)
    k = len(qs)
    denominator = 4 ** k - 1
    out = (1.0 - p) * array
    for label, matrix in pauli_operators(k):
        if label == "I" * k:
            continue
        out = out + (p / denominator) * apply_operator(array, matrix, qs)
    return out


def depolarize_qubits(rho, qubits, p: float) -> np.ndarray:
    """Pauli depolarising channel on a ``k``-qubit subsystem.

    Uses ``sum_{P != I} P rho P = 2**k Tr_k(rho) (x) I_k - rho``, so the cost is
    one partial trace and a tensor reshape rather than ``4**k - 1`` matrix
    products.  ``k = 1`` gives the ``p/3`` single-qubit channel, ``k = 2`` the
    ``p/15`` two-qubit channel.
    """
    p = validate_probability(p, "p")
    array = np.asarray(rho, dtype=complex)
    d = array.shape[0]
    n = num_qubits(d)
    qs = tuple(int(q) for q in qubits)
    if len(set(qs)) != len(qs):
        raise ValueError(f"qubit indices must be distinct, got {qs}")
    if any(q < 0 or q >= n for q in qs):
        raise ValueError(f"qubit indices {qs} out of range for n={n}")
    if p == 0.0:
        return array.copy()

    k = len(qs)
    dk = 2 ** k
    rest = tuple(q for q in range(n) if q not in qs)
    drest = 2 ** len(rest)
    perm = list(qs) + list(rest) + [n + q for q in qs] + [n + o for o in rest]

    tensor = array.reshape((2,) * (2 * n)).transpose(perm).reshape(dk, drest, dk, drest)
    reduced = np.einsum("aiaj->ij", tensor)          # Tr_k(rho), shape (drest, drest)
    mixed = np.zeros_like(tensor)
    for a in range(dk):
        mixed[a, :, a, :] = dk * reduced             # 2**k Tr_k(rho) (x) I_k

    denominator = 4 ** k - 1
    channel = (1.0 - p) * tensor + (p / denominator) * (mixed - tensor)
    channel = channel.reshape((2,) * (2 * n)).transpose(np.argsort(perm)).reshape(d, d)
    return channel
