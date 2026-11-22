"""Qiskit Aer simulator backend integration.

Executes circuits using ``qiskit_aer.AerSimulator`` with configurable noise models,
device coupling topologies, and shot counts.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoiseModel

import circuit_cost
from quantum_sdk import circuit_builder

from .interface import BackendProperties, Job, JobResult, JobStatus, QuantumBackend

__all__ = ["AerBackend"]


class AerBackend(QuantumBackend):
    """AerSimulator backend with support for custom noise models and coupling maps."""

    def __init__(self, name: str = "aer_simulator",
                 noise_model: Optional[AerNoiseModel] = None,
                 coupling_map: Optional[Any] = None,
                 basis_gates: Optional[Tuple[str, ...]] = None,
                 num_qubits: int = 16) -> None:
        basis = basis_gates or ("rz", "rx", "cx")
        props = BackendProperties(
            name=name,
            num_qubits=num_qubits,
            basis_gates=basis,
            coupling_map=coupling_map.get_edges() if coupling_map is not None else None,
            is_simulator=True,
            metadata={"framework": "qiskit-aer"},
        )
        super().__init__(props)
        self.noise_model = noise_model
        self.coupling_map = coupling_map
        self.sim = AerSimulator(
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=list(basis),
        )

    def submit(self, circuit: Any, shots: int = 1024, seed: Optional[int] = None, **kwargs) -> Job:
        job_id = f"aer_{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()

        if isinstance(circuit, list) and all(isinstance(g, circuit_cost.Gate) for g in circuit):
            n = max(max(g.qubits) for g in circuit) + 1 if circuit else 1
            qc = circuit_builder.build_from_gates(n, circuit)
        elif isinstance(circuit, QuantumCircuit):
            qc = circuit
            n = circuit.num_qubits
        else:
            raise TypeError(f"Unsupported circuit type {type(circuit)}")

        # Ensure measurements are present
        if not any(item.operation.name == "measure" for item in qc.data):
            qc_measured = qc.copy()
            qc_measured.measure_all()
        else:
            qc_measured = qc

        aer_job = self.sim.run(qc_measured, shots=shots, seed_simulator=seed)
        aer_result = aer_job.result()
        raw_counts = aer_result.get_counts(qc_measured)

        # Standardize bitstring format
        counts: Dict[str, int] = {}
        prob_dict: Dict[str, float] = {}
        for bitstr, count in raw_counts.items():
            clean_str = bitstr.replace(" ", "")
            counts[clean_str] = int(count)
            prob_dict[clean_str] = float(count / shots)

        elapsed = time.perf_counter() - t0
        res = JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            counts=counts,
            probabilities=prob_dict,
            shots=shots,
            execution_time=elapsed,
            backend_name=self.name,
            metadata={"num_qubits": n, "aer_time_taken": aer_result.time_taken},
        )

        self._jobs[job_id] = {
            "status": JobStatus.COMPLETED,
            "result": res,
        }
        return Job(job_id, self)
