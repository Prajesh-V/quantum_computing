"""Translate the Phase 1 gate list into a Qiskit circuit.

The decomposition is *not* re-derived here: the gates come from
``circuit_cost.build_circuit`` in chronological order, which is the same list
the density-matrix simulator executes and the same one Phase 1 counted.

Qubit indexing
--------------
The repository uses qubit 0 as the **most significant** bit of the basis-state
index (``kron_all`` puts the first factor first).  Qiskit uses little-endian
convention: qubit 0 is the **least significant** bit.  The translation is
therefore

    qiskit_qubit = n - 1 - repo_qubit

With that mapping the two basis indices agree exactly -- the repository's
``|b_0 b_1 ... b_{n-1}>`` and Qiskit's ``|q_{n-1} ... q_0>`` become the same
integer -- so unitaries and density matrices can be compared without any
permutation.  ``verify_index_convention`` checks this numerically rather than
assuming it.

Rotations
---------
Qiskit's ``rz(theta)`` is ``diag(exp(-i theta/2), exp(+i theta/2))`` and
``rx(theta)`` is ``cos(theta/2) I - i sin(theta/2) X``, which are exactly the
matrices used by ``circuit_cost.gate_matrix`` and
``noise.noisy_simulator.local_matrix``.  Angles are passed through unchanged --
no sign flip and no factor of two.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import circuit_cost

__all__ = ["qubit_map", "to_qiskit_qubit", "build", "build_from_gates",
           "gate_counts", "verify_index_convention", "BASIS"]

BASIS = ("rz", "rx", "cx")


def qubit_map(n: int) -> dict[int, int]:
    """Repository qubit -> Qiskit qubit (a reversal)."""
    return {qubit: n - 1 - qubit for qubit in range(n)}


def to_qiskit_qubit(n: int, qubit: int) -> int:
    return n - 1 - qubit


def build_from_gates(n: int, gates, name: str | None = None) -> QuantumCircuit:
    """One Qiskit circuit per Phase 1 gate, in the same order."""
    circuit = QuantumCircuit(n, name=name)
    for gate in gates:
        qubits = [to_qiskit_qubit(n, qubit) for qubit in gate.qubits]
        if gate.gate_type == "cnot":
            circuit.cx(qubits[0], qubits[1])
        elif gate.gate_type == "rx":
            circuit.rx(gate.angle, qubits[0])
        elif gate.gate_type == "rz":
            circuit.rz(gate.angle, qubits[0])
        else:                                   # pragma: no cover - Gate validates this
            raise ValueError(f"unknown gate type {gate.gate_type!r}")
    return circuit


def build(n: int, r: int, order: int = 1, t: float = 1.0, J: float = 1.0,
          h: float = 1.0, *, merge_half_steps: bool = True,
          name: str | None = None) -> QuantumCircuit:
    """The Phase 1 circuit for one configuration, as a Qiskit circuit."""
    gates = circuit_cost.build_circuit(n, r, t, J, h, order,
                                       merge_half_steps=merge_half_steps)
    if name is None:
        name = f"tfim_n{n}_r{r}_o{order}"
    return build_from_gates(n, gates, name=name)


def gate_counts(circuit: QuantumCircuit) -> dict:
    """Gate-type histogram of a circuit."""
    return {str(key): int(value) for key, value in circuit.count_ops().items()}


def verify_index_convention(n: int = 3) -> dict:
    """Check that ``qubit 0 = MSB`` translates to ``qubit n-1`` in Qiskit.

    ``X`` on repository qubit 0 must move the population from index 0 to
    ``2**(n-1)`` in *both* representations; if the mapping were the wrong way
    round, Qiskit would put it at index 1 instead.
    """
    gates = [circuit_cost.Gate("rx", (0,), np.pi)]      # pi pulse = X up to phase
    circuit = build_from_gates(n, gates)
    state = np.asarray(Operator(circuit).data[:, 0])
    index_qiskit = int(np.argmax(np.abs(state)))

    # repository-side reference: X on qubit 0 with qubit 0 as the MSB
    reference = np.zeros(2 ** n, dtype=complex)
    reference[2 ** (n - 1)] = 1.0
    return {
        "qiskit_index_of_excited_population": index_qiskit,
        "expected_index": 2 ** (n - 1),
        "matches_most_significant_convention": index_qiskit == 2 ** (n - 1),
        "state": state,
        "reference": reference,
    }
