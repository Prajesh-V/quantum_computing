"""Phase-insensitive circuit equivalence.

A global phase is physically irrelevant, so the comparison is

    min_phi  || U_repo - exp(i phi) U_sdk ||_2

which is minimised by ``exp(i phi) = <U_repo, U_sdk> / |<U_repo, U_sdk>|`` with
``<A, B> = sum_ij conj(A_ij) B_ij``.  Both the Frobenius and the spectral norm of
the residual are reported; the phase itself is reported too, because a phase that
is *not* near one of the simple roots of unity would be a red flag.

Matching gate counts is not accepted as evidence of equivalence anywhere here.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator, Statevector

import circuit_cost

from . import circuit_builder

__all__ = ["unitary", "phase_insensitive", "compare_unitary",
           "compare_statevector", "max_entry_difference"]


def unitary(circuit) -> np.ndarray:
    """Full unitary of a (classical-bit free) Qiskit circuit."""
    return np.asarray(Operator(circuit).data, dtype=complex)


def phase_insensitive(a, b) -> dict:
    """Smallest distance between two matrices over all global phases."""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    inner = np.vdot(a, b)
    if inner == 0.0:
        return {"phase": 0.0, "frobenius": float(np.linalg.norm(a - b)),
                "spectral": float(np.linalg.norm(a - b, 2)),
                "max_entry": float(np.max(np.abs(a - b)))}
    # ||a - e^{i phi} b||^2 = |a|^2 + |b|^2 - 2 Re(e^{i phi} <a,b>), minimised at
    # phi = -arg(<a,b>).  Using +arg here would report the distance at twice the
    # offset instead of the minimum.
    phase = -float(np.angle(inner))
    rotated = np.exp(1j * phase) * b
    residual = a - rotated
    return {
        "phase": phase,
        "phase_in_turns": phase / (2 * np.pi),
        "frobenius": float(np.linalg.norm(residual)),
        "spectral": float(np.linalg.norm(residual, 2)),
        "max_entry": float(np.max(np.abs(residual))),
    }


def max_entry_difference(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def compare_unitary(n: int, r: int, order: int, t: float = 1.0,
                    J: float = 1.0, h: float = 1.0) -> dict:
    """Repository circuit unitary vs the Qiskit circuit unitary."""
    gates = circuit_cost.build_circuit(n, r, t, J, h, order)
    u_repo = circuit_cost.circuit_unitary(n, gates)
    circuit = circuit_builder.build_from_gates(n, gates)
    u_sdk = unitary(circuit)
    result = phase_insensitive(u_repo, u_sdk)
    result.update({
        "system_size": n, "trotter_steps": r, "trotter_order": order,
        "total_time": t, "gates": len(gates),
    })
    return result


def compare_statevector(n: int, r: int, order: int, t: float = 1.0,
                        J: float = 1.0, h: float = 1.0) -> dict:
    """Output state for ``|00...0>`` from both representations."""
    gates = circuit_cost.build_circuit(n, r, t, J, h, order)
    u_repo = circuit_cost.circuit_unitary(n, gates)
    psi_repo = u_repo @ np.eye(2 ** n, dtype=complex)[:, 0]
    psi_sdk = np.asarray(Statevector(circuit_builder.build_from_gates(n, gates)).data)
    overlap = np.vdot(psi_repo, psi_sdk)
    phase = float(np.angle(overlap)) if abs(overlap) else 0.0
    return {
        "system_size": n, "trotter_steps": r, "trotter_order": order,
        "phase": phase,
        "max_entry": max_entry_difference(psi_repo, np.exp(1j * phase) * psi_sdk),
        "infidelity": float(max(0.0, 1.0 - abs(overlap) ** 2)),
    }
