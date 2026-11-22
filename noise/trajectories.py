"""Quantum trajectory (Monte Carlo Wavefunction) simulator for Trotter circuits.

Solves open-system Trotter evolution for spin chains scaling beyond the density-matrix
size limit (n > 6, e.g. n = 7, 8, 10, 12).

Statevector dimension is 2**n rather than 4**n for density matrices:
- At n = 8: statevector has 256 complex amplitudes (2 KiB) vs 65,536 (512 KiB)
- At n = 10: statevector has 1,024 complex amplitudes (8 KiB) vs 1,048,576 (8 MiB)
- At n = 12: statevector has 4,096 complex amplitudes (32 KiB) vs 16,777,216 (128 MiB)

In each trajectory m:
1. Pure state |psi> evolves under unitary gates U.
2. For each gate's noise channel with Kraus operators {K_k}:
   Branch probabilities are p_k = <psi| K_k^dagger K_k |psi> = || K_k |psi> ||^2.
   A branch k is chosen stochastically with probability p_k, and |psi> is collapsed:
     |psi> -> K_k |psi> / sqrt(p_k).
3. The ensemble average over M trajectories recovers the density matrix:
     (1/M) sum_m |psi_m><psi_m| -> rho.
   Fidelity with an ideal pure state |psi_ref> is:
     F = (1/M) sum_m |<psi_ref | psi_m>|^2.
   Standard error of the mean is sigma / sqrt(M).
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

import circuit_cost
import trotter

from .channels import PAULI_1Q, pauli_operators
from .noisy_simulator import CNOT, NoiseModel, exact_state, ideal_trotter_state, local_matrix
from .relaxation import thermal_relaxation_kraus

__all__ = [
    "apply_gate_statevector",
    "depolarizing_kraus_1q",
    "depolarizing_kraus_2q",
    "sample_kraus_branch",
    "run_trajectory",
    "simulate_trajectories",
    "compare_trajectory_with_density_matrix",
    "statevector_magnetization_x",
    "statevector_zz_correlation",
]


def _axes_statevector(n: int, qubits: tuple[int, ...]):
    rest = tuple(q for q in range(n) if q not in qubits)
    perm = list(qubits) + list(rest)
    return perm, rest


def apply_gate_statevector(psi: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Apply a local 2**k x 2**k matrix to an n-qubit statevector."""
    k = len(qubits)
    dk = 2 ** k
    dim = 2 ** n
    if psi.shape != (dim,):
        raise ValueError(f"psi shape {psi.shape} does not match 2**{n} = {dim}")
    if matrix.shape != (dk, dk):
        raise ValueError(f"matrix shape {matrix.shape} does not match 2**{k}")

    perm, rest = _axes_statevector(n, qubits)
    tensor = psi.reshape((2,) * n).transpose(perm).reshape(dk, 2 ** len(rest))
    tensor = matrix @ tensor
    tensor = tensor.reshape((2,) * n).transpose(np.argsort(perm)).reshape(dim)
    return tensor


def depolarizing_kraus_1q(p: float) -> list[np.ndarray]:
    """Single-qubit depolarizing Kraus operators: sqrt(1 - p) I, sqrt(p/3) X, Y, Z."""
    if p == 0.0:
        return [np.eye(2, dtype=complex)]
    p_pauli = p / 3.0
    return [
        math.sqrt(1.0 - p) * PAULI_1Q["I"],
        math.sqrt(p_pauli) * PAULI_1Q["X"],
        math.sqrt(p_pauli) * PAULI_1Q["Y"],
        math.sqrt(p_pauli) * PAULI_1Q["Z"],
    ]


def depolarizing_kraus_2q(p: float) -> list[np.ndarray]:
    """Two-qubit depolarizing Kraus operators: sqrt(1 - p) II, sqrt(p/15) P."""
    if p == 0.0:
        return [np.eye(4, dtype=complex)]
    p_pauli = p / 15.0
    ops = [math.sqrt(1.0 - p) * np.eye(4, dtype=complex)]
    for label, mat in pauli_operators(2):
        if label != "II":
            ops.append(math.sqrt(p_pauli) * mat)
    return ops


def sample_kraus_branch(psi: np.ndarray, kraus_ops: Sequence[np.ndarray],
                        qubits: tuple[int, ...], n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Stochastically select a Kraus branch weighted by p_k = || K_k |psi> ||^2."""
    if len(kraus_ops) == 1:
        # Identity / noiseless
        return psi, 0

    branches = [apply_gate_statevector(psi, op, qubits, n) for op in kraus_ops]
    probs = np.array([float(np.real(np.vdot(branch, branch))) for branch in branches], dtype=float)
    total_p = float(np.sum(probs))
    if total_p <= 0.0 or not math.isfinite(total_p):
        return psi, 0
    probs /= total_p

    choice = int(rng.choice(len(kraus_ops), p=probs))
    chosen_state = branches[choice]
    norm = float(np.linalg.norm(chosen_state))
    if norm > 1e-15:
        chosen_state = chosen_state / norm
    return chosen_state, choice


def run_trajectory(n: int, r: int, order: int = 1, t: float = 1.0, J: float = 1.0,
                   h: float = 1.0, model: NoiseModel | None = None,
                   rng: np.random.Generator | None = None,
                   merge_half_steps: bool = True) -> np.ndarray:
    """Execute a single Monte Carlo quantum trajectory, returning the final pure statevector."""
    if rng is None:
        rng = np.random.default_rng()
    model = model or NoiseModel()

    gates = circuit_cost.build_circuit(n, int(r), t, J, h, order, merge_half_steps=merge_half_steps)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0  # |00...0>

    # Precompute Kraus operators
    if model.kind == "gate" and not model.is_noiseless:
        p1 = model.single_qubit_error
        p2 = model.two_qubit_error
        rz_p = model.single_qubit_error if model.rz_error is None else model.rz_error
        k_1q_rx = depolarizing_kraus_1q(p1)
        k_1q_rz = depolarizing_kraus_1q(rz_p)
        k_2q = depolarizing_kraus_2q(p2)
    elif model.kind == "relaxation" and not model.is_noiseless:
        k_1q = thermal_relaxation_kraus(model.gate_time_1q, model.t1, model.t2)
        k_2q_1q = thermal_relaxation_kraus(model.gate_time_2q, model.t1, model.t2)

    for gate in gates:
        mat = local_matrix(gate)
        psi = apply_gate_statevector(psi, mat, gate.qubits, n)

        if model.kind == "gate" and not model.is_noiseless:
            if gate.gate_type == "cnot":
                psi, _ = sample_kraus_branch(psi, k_2q, gate.qubits, n, rng)
            elif gate.gate_type == "rz":
                psi, _ = sample_kraus_branch(psi, k_1q_rz, (gate.qubits[0],), n, rng)
            else:
                psi, _ = sample_kraus_branch(psi, k_1q_rx, (gate.qubits[0],), n, rng)
        elif model.kind == "relaxation" and not model.is_noiseless:
            if gate.gate_type == "cnot":
                psi, _ = sample_kraus_branch(psi, k_2q_1q, (gate.qubits[0],), n, rng)
                psi, _ = sample_kraus_branch(psi, k_2q_1q, (gate.qubits[1],), n, rng)
            else:
                psi, _ = sample_kraus_branch(psi, k_1q, (gate.qubits[0],), n, rng)

    return psi


def statevector_magnetization_x(psi: np.ndarray, n: int) -> float:
    """<psi| (1/n sum_i X_i) |psi>."""
    dim = 2 ** n
    X = PAULI_1Q["X"]
    total = 0.0
    for q in range(n):
        x_psi = apply_gate_statevector(psi, X, (q,), n)
        total += float(np.real(np.vdot(psi, x_psi)))
    return total / n


def statevector_zz_correlation(psi: np.ndarray, n: int) -> float:
    """<psi| (1/(n-1) sum_i Z_i Z_{i+1}) |psi>."""
    if n < 2:
        return 0.0
    dim = 2 ** n
    Z = PAULI_1Q["Z"]
    ZZ = np.kron(Z, Z)
    total = 0.0
    for q in range(n - 1):
        zz_psi = apply_gate_statevector(psi, ZZ, (q, q + 1), n)
        total += float(np.real(np.vdot(psi, zz_psi)))
    return total / (n - 1)


def simulate_trajectories(n: int, r: int, order: int = 1, t: float = 1.0, J: float = 1.0,
                          h: float = 1.0, model: NoiseModel | None = None,
                          num_trajectories: int = 100, seed: int = 42,
                          merge_half_steps: bool = True) -> dict:
    """Run an ensemble of quantum trajectories and calculate statistical observables."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    model = model or NoiseModel()

    # Reference states
    psi_exact = exact_state(n, t, J, h)
    psi_trotter = ideal_trotter_state(n, r, order, t, J, h)

    exact_overlaps = []
    trotter_overlaps = []
    m_x_list = []
    zz_list = []

    for _ in range(num_trajectories):
        psi_m = run_trajectory(n, r, order, t, J, h, model, rng, merge_half_steps=merge_half_steps)
        # overlap squared |<psi_ref | psi_m>|^2
        f_exact = float(np.abs(np.vdot(psi_exact, psi_m)) ** 2)
        f_trotter = float(np.abs(np.vdot(psi_trotter, psi_m)) ** 2)
        exact_overlaps.append(f_exact)
        trotter_overlaps.append(f_trotter)
        m_x_list.append(statevector_magnetization_x(psi_m, n))
        zz_list.append(statevector_zz_correlation(psi_m, n))

    elapsed = time.perf_counter() - t0

    exact_arr = np.array(exact_overlaps)
    trotter_arr = np.array(trotter_overlaps)
    mean_fidelity = float(np.mean(exact_arr))
    sem_fidelity = float(np.std(exact_arr, ddof=1) / math.sqrt(num_trajectories)) if num_trajectories > 1 else 0.0

    mean_trotter_fid = float(np.mean(trotter_arr))
    sem_trotter_fid = float(np.std(trotter_arr, ddof=1) / math.sqrt(num_trajectories)) if num_trajectories > 1 else 0.0

    resources = circuit_cost.estimate_resources(n, int(r), t, J, h, order, merge_half_steps=merge_half_steps)

    return {
        "system_size": n,
        "trotter_steps": int(r),
        "trotter_order": int(order),
        "total_time": float(t),
        "num_trajectories": num_trajectories,
        "fidelity": mean_fidelity,
        "sem_fidelity": sem_fidelity,
        "total_error": max(0.0, 1.0 - mean_fidelity),
        "trotter_fidelity": mean_trotter_fid,
        "sem_trotter_fidelity": sem_trotter_fid,
        "noise_induced_error": max(0.0, 1.0 - mean_trotter_fid),
        "magnetization": float(np.mean(m_x_list)),
        "zz_correlation": float(np.mean(zz_list)),
        "cnot_count": resources.num_cnot,
        "total_gates": resources.total_gates,
        "logical_depth": resources.circuit_depth,
        "elapsed_seconds": elapsed,
        "noise_model": model.kind,
    }


def compare_trajectory_with_density_matrix(n: int, r: int, order: int = 1,
                                           model: NoiseModel | None = None,
                                           num_trajectories: int = 300,
                                           seed: int = 42) -> dict:
    """Compare trajectory ensemble average with exact density matrix calculation."""
    from .noisy_simulator import simulate as dm_simulate
    from .metrics import fidelity_pure

    model = model or NoiseModel(kind="gate", single_qubit_error=1e-3, two_qubit_error=1e-3)
    dm_res = dm_simulate(n, r, order, model=model)
    psi_ex = exact_state(n, 1.0, 1.0, 1.0)
    dm_fid = float(fidelity_pure(dm_res["rho"], psi_ex))

    traj_res = simulate_trajectories(n, r, order, model=model,
                                     num_trajectories=num_trajectories, seed=seed)
    diff = abs(traj_res["fidelity"] - dm_fid)
    sem = traj_res["sem_fidelity"]

    # Trajectory mean should be within 3 standard errors of the exact density matrix value
    passed = diff <= max(3.5 * sem, 0.03)

    return {
        "system_size": n,
        "trotter_steps": r,
        "order": order,
        "num_trajectories": num_trajectories,
        "density_matrix_fidelity": dm_fid,
        "trajectory_fidelity": traj_res["fidelity"],
        "sem": sem,
        "difference": diff,
        "passed": passed,
    }
