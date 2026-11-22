"""Local classical backends for circuit execution and bitstring sampling.

Provides:
- ``LocalStatevectorBackend``: exact statevector evolution + multinomial shot sampling.
- ``LocalDensityMatrixBackend``: noisy density matrix evolution + multinomial shot sampling.

Both operate directly on either Phase 1 gate lists or Qiskit circuits (translated
via repository conventions) with zero external cloud dependencies.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import circuit_cost
from noise.density_matrix import basis_density_matrix, num_qubits
from noise.noisy_simulator import NoiseModel, simulate
from noise.trajectories import apply_gate_statevector

from .interface import BackendProperties, Job, JobResult, JobStatus, QuantumBackend

__all__ = ["LocalStatevectorBackend", "LocalDensityMatrixBackend"]


def _extract_gates(circuit: Any, n: int) -> List[circuit_cost.Gate]:
    """Accept either a list of circuit_cost.Gate or a Qiskit QuantumCircuit."""
    if isinstance(circuit, list) and all(isinstance(g, circuit_cost.Gate) for g in circuit):
        return circuit
    # If it's a Qiskit circuit, extract gates
    from quantum_sdk.circuit_builder import to_qiskit_qubit
    gates = []
    for item in circuit.data:
        name = item.operation.name.lower()
        # Qiskit qubit -> repository qubit: repo_qubit = n - 1 - qiskit_qubit
        q_indices = [n - 1 - circuit.find_bit(q).index for q in item.qubits]
        params = getattr(item.operation, "params", [])
        angle = float(params[0]) if params else 0.0
        if name in ("cx", "cnot"):
            gates.append(circuit_cost.Gate("cnot", tuple(q_indices), 0.0))
        elif name == "rx":
            gates.append(circuit_cost.Gate("rx", tuple(q_indices), angle))
        elif name == "rz":
            gates.append(circuit_cost.Gate("rz", tuple(q_indices), angle))
    return gates


class LocalStatevectorBackend(QuantumBackend):
    """Exact ideal statevector simulation with shot sampling."""

    def __init__(self, num_qubits: int = 12, name: str = "local_statevector_simulator") -> None:
        props = BackendProperties(
            name=name,
            num_qubits=num_qubits,
            basis_gates=("rz", "rx", "cx"),
            is_simulator=True,
            metadata={"type": "statevector", "method": "exact_tensor"},
        )
        super().__init__(props)

    def submit(self, circuit: Any, shots: int = 1024, seed: Optional[int] = None, **kwargs) -> Job:
        job_id = f"local_sv_{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()

        # Determine n
        if kwargs.get("num_qubits") is not None:
            n = int(kwargs["num_qubits"])
        elif hasattr(circuit, "num_qubits"):
            n = circuit.num_qubits
        else:
            # list of gates: find max qubit index
            n = max(max(g.qubits) for g in circuit) + 1 if circuit else 1

        gates = _extract_gates(circuit, n)
        from noise.noisy_simulator import local_matrix

        # Evolve |00...0>
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0

        for gate in gates:
            mat = local_matrix(gate)
            psi = apply_gate_statevector(psi, mat, gate.qubits, n)

        # Probabilities P(x) = |<x|psi>|^2
        probs = np.real(psi * psi.conj())
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.sum(probs)

        # Sample shots
        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(2 ** n, size=shots, p=probs)
        unique, counts_arr = np.unique(sampled_indices, return_counts=True)

        counts: Dict[str, int] = {}
        for idx, count in zip(unique, counts_arr):
            bitstr = f"{idx:0{n}b}"
            counts[bitstr] = int(count)

        prob_dict = {f"{i:0{n}b}": float(p) for i, p in enumerate(probs) if p > 1e-12}

        elapsed = time.perf_counter() - t0
        res = JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            counts=counts,
            probabilities=prob_dict,
            shots=shots,
            execution_time=elapsed,
            backend_name=self.name,
            metadata={"num_qubits": n, "num_gates": len(gates)},
        )

        self._jobs[job_id] = {
            "status": JobStatus.COMPLETED,
            "result": res,
        }
        return Job(job_id, self)


class LocalDensityMatrixBackend(QuantumBackend):
    """Noisy density-matrix simulation with shot sampling."""

    def __init__(self, num_qubits: int = 6, name: str = "local_density_matrix_simulator",
                 noise_model: Optional[NoiseModel] = None) -> None:
        props = BackendProperties(
            name=name,
            num_qubits=num_qubits,
            basis_gates=("rz", "rx", "cx"),
            is_simulator=True,
            metadata={"type": "density_matrix", "noise_model": noise_model.kind if noise_model else "none"},
        )
        super().__init__(props)
        self.noise_model = noise_model or NoiseModel()

    def submit(self, circuit: Any, shots: int = 1024, seed: Optional[int] = None, **kwargs) -> Job:
        job_id = f"local_dm_{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()

        if kwargs.get("num_qubits") is not None:
            n = int(kwargs["num_qubits"])
        elif hasattr(circuit, "num_qubits"):
            n = circuit.num_qubits
        else:
            n = max(max(g.qubits) for g in circuit) + 1 if circuit else 1

        gates = _extract_gates(circuit, n)
        from noise.noisy_simulator import apply_gate, gate_noise_channels
        from noise.channels import depolarize_qubits

        rho = basis_density_matrix(n, [0] * n)
        for gate in gates:
            rho = apply_gate(rho, gate)
            ch = gate_noise_channels(gate, self.noise_model)
            if ch is not None:
                rho = depolarize_qubits(rho, ch[0], ch[1])

        probs = np.real(np.diag(rho))
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.sum(probs)

        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(2 ** n, size=shots, p=probs)
        unique, counts_arr = np.unique(sampled_indices, return_counts=True)

        counts: Dict[str, int] = {}
        for idx, count in zip(unique, counts_arr):
            bitstr = f"{idx:0{n}b}"
            counts[bitstr] = int(count)

        prob_dict = {f"{i:0{n}b}": float(p) for i, p in enumerate(probs) if p > 1e-12}

        elapsed = time.perf_counter() - t0
        res = JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            counts=counts,
            probabilities=prob_dict,
            shots=shots,
            execution_time=elapsed,
            backend_name=self.name,
            metadata={"num_qubits": n, "num_gates": len(gates)},
        )

        self._jobs[job_id] = {
            "status": JobStatus.COMPLETED,
            "result": res,
        }
        return Job(job_id, self)
