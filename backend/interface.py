"""Quantum backend interface: job lifecycle, execution abstractions, and hardware metadata.

Provides a unified interface for executing Trotter circuits across:
1. Local exact statevector/density-matrix samplers (pure repo code).
2. Qiskit Aer simulation with realistic noise and shot sampling.
3. Mock hardware processors emulating real device constraints (coupling map,
   latency, gate error rates, asymmetric readout assignment errors).
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class JobStatus(str, Enum):
    """Execution status of a backend job."""
    INITIALIZING = "INITIALIZING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class BackendProperties:
    """Hardware specifications and calibration characteristics."""
    name: str
    num_qubits: int
    coupling_map: Optional[List[Tuple[int, int]]] = None
    basis_gates: Tuple[str, ...] = ("rz", "rx", "cx")
    t1: Optional[Dict[int, float]] = None          # qubit -> T1 (seconds)
    t2: Optional[Dict[int, float]] = None          # qubit -> T2 (seconds)
    gate_times: Optional[Dict[str, float]] = None  # gate -> duration (seconds)
    single_qubit_error: float = 1e-4
    two_qubit_error: float = 1e-3
    readout_error: float = 0.015                   # symmetric or baseline readout error
    is_simulator: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Outcome of an executed quantum job."""
    job_id: str
    status: JobStatus
    counts: Dict[str, int]
    probabilities: Dict[str, float]
    shots: int
    execution_time: float
    backend_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def most_frequent(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Return the top_k most frequent bitstrings."""
        return sorted(self.counts.items(), key=lambda item: item[1], reverse=True)[:top_k]


class Job:
    """Handle to a submitted execution job."""

    def __init__(self, job_id: str, backend: "QuantumBackend") -> None:
        self.job_id = job_id
        self.backend = backend
        self._created_at = time.time()

    def status(self) -> JobStatus:
        """Query current execution status."""
        return self.backend.status(self.job_id)

    def result(self, timeout: Optional[float] = None) -> JobResult:
        """Wait for completion and return the JobResult."""
        t0 = time.time()
        while True:
            st = self.status()
            if st in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return self.backend.result(self.job_id)
            if timeout is not None and (time.time() - t0) > timeout:
                raise TimeoutError(f"Job {self.job_id} did not finish within {timeout}s (status: {st})")
            time.sleep(0.01)

    def cancel(self) -> bool:
        """Cancel the job."""
        return self.backend.cancel(self.job_id)


class QuantumBackend(ABC):
    """Abstract base class for quantum execution backends."""

    def __init__(self, properties: BackendProperties) -> None:
        self.properties = properties
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return self.properties.name

    @abstractmethod
    def submit(self, circuit: Any, shots: int = 1024, **kwargs) -> Job:
        """Submit a circuit for execution, returning a Job handle."""
        raise NotImplementedError

    def run(self, circuit: Any, shots: int = 1024, **kwargs) -> JobResult:
        """Synchronously submit and wait for result."""
        job = self.submit(circuit, shots=shots, **kwargs)
        return job.result()

    def status(self, job_id: str) -> JobStatus:
        """Return current status of job_id."""
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job_id {job_id!r}")
        return self._jobs[job_id]["status"]

    def result(self, job_id: str) -> JobResult:
        """Retrieve stored result for job_id."""
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job_id {job_id!r}")
        job_data = self._jobs[job_id]
        if job_data["status"] != JobStatus.COMPLETED:
            if job_data["status"] == JobStatus.FAILED:
                raise RuntimeError(f"Job {job_id} failed: {job_data.get('error')}")
            raise RuntimeError(f"Job {job_id} is not completed (current: {job_data['status']})")
        return job_data["result"]

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        if job_id not in self._jobs:
            return False
        st = self._jobs[job_id]["status"]
        if st in (JobStatus.QUEUED, JobStatus.INITIALIZING):
            self._jobs[job_id]["status"] = JobStatus.CANCELLED
            return True
        return False
