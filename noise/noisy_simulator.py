"""Chronological noisy execution of the Phase 1 gate circuit.

The circuit is **not** re-derived here.  ``circuit_cost.build_circuit`` produces
the ordered gate list (CNOT / RZ / RX) used by Phases 1 and 2, and this module
walks that list in order, applying

    rho -> U_gate rho U_gate^dagger  ->  E_gate(rho)

for every gate, where ``E_gate`` is the single-qubit channel after RX/RZ and the
two-qubit channel after CNOT.  The ideal action of the whole walk is tested
against ``trotter.trotter_unitary`` and against ``trotter.sweep``, so the gate
order, CNOT control/target ordering, qubit indexing, merged second-order ZZ
layers and half-angle rotations are all inherited, not re-invented.

Gate application uses local ``2 x 2`` / ``4 x 4`` matrices via
``density_matrix.apply_operator``; the embedding is tested against
``circuit_cost.gate_matrix``.  Cost is ``O(gates * 4**k * d)`` with ``k <= 2``,
i.e. linear in the number of gates and quadratic in ``d = 2**n``; memory is one
``d x d`` complex matrix (16 KiB at n = 6, 256 KiB at n = 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import circuit_cost
import trotter

from .channels import (apply_global_depolarizing, depolarize_qubits,
                       validate_probability)
from .relaxation import (apply_kraus_channel, thermal_relaxation_kraus,
                         two_qubit_relaxation_channel)
from .density_matrix import (apply_operator, basis_density_matrix, num_qubits,
                             pure_density_matrix)

__all__ = [
    "NoiseModel",
    "CNOT",
    "local_matrix",
    "apply_gate",
    "gate_noise_channels",
    "step_ranges",
    "simulate",
    "exact_state",
    "ideal_trotter_state",
    "reference_states",
    "reset_cache",
]

#: CNOT with ``qubits[0]`` as control (most significant of the pair), matching
#: ``circuit_cost._CNOT4`` and therefore ``trotter.two_qubit``.
CNOT = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0]], dtype=complex)

NOISE_KINDS = ("none", "global_per_step", "global_end_of_circuit", "gate", "relaxation")


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseModel:
    """One noise configuration.

    ``kind``
        ``"none"``                  ideal evolution (no channel at all)
        ``"global_per_step"``       global depolarising channel after every Trotter step
        ``"global_end_of_circuit"`` one global channel after the last gate
        ``"gate"``                  single-qubit channel after RX/RZ, two-qubit after CNOT
        ``"relaxation"``            thermal relaxation (T1, T2) after each gate

    ``single_qubit_error`` / ``two_qubit_error``
        ``p_1`` and ``p_2``: the total probability of a non-identity Pauli after
        the corresponding gate.  These are abstract simulation parameters, not
        measured hardware error rates.

    ``rz_error``
        Optional separate error for RZ (virtual-Z architectures).  ``None``
        means "use ``single_qubit_error``"; ``0.0`` makes RZ noiseless while
        leaving RX noisy.

    ``t1`` / ``t2`` / ``gate_time_1q`` / ``gate_time_2q``
        Physical parameters for ``"relaxation"`` models.
    """

    kind: str = "none"
    single_qubit_error: float = 0.0
    two_qubit_error: float = 0.0
    global_probability: float = 0.0
    rz_error: float | None = None
    t1: float = 50e-6
    t2: float = 70e-6
    gate_time_1q: float = 35e-9
    gate_time_2q: float = 300e-9

    def __post_init__(self) -> None:
        if self.kind not in NOISE_KINDS:
            raise ValueError(f"unknown noise kind {self.kind!r}; choose from {NOISE_KINDS}")
        validate_probability(self.single_qubit_error, "p_1")
        validate_probability(self.two_qubit_error, "p_2")
        validate_probability(self.global_probability, "p_g")
        if self.rz_error is not None:
            validate_probability(self.rz_error, "rz_error")
        if self.kind == "relaxation":
            if self.t1 <= 0 or self.t2 <= 0:
                raise ValueError("T1 and T2 must be positive")
            if self.t2 > 2.0 * self.t1 + 1e-14:
                raise ValueError(f"T2 ({self.t2}) cannot exceed 2 * T1 ({2*self.t1})")
            if self.gate_time_1q < 0 or self.gate_time_2q < 0:
                raise ValueError("gate times must be non-negative")

    # -- convenience -------------------------------------------------------
    @property
    def is_noiseless(self) -> bool:
        if self.kind == "none":
            return True
        if self.kind == "gate":
            return (self.single_qubit_error == 0.0 and self.two_qubit_error == 0.0
                    and (self.rz_error or 0.0) == 0.0)
        if self.kind == "relaxation":
            return self.gate_time_1q == 0.0 and self.gate_time_2q == 0.0
        return self.global_probability == 0.0

    def error_for(self, gate_type: str) -> float:
        """Error probability applied after ``gate_type``."""
        if gate_type == "cnot":
            return self.two_qubit_error
        if gate_type == "rz":
            return self.single_qubit_error if self.rz_error is None else self.rz_error
        return self.single_qubit_error

    def label(self) -> str:
        if self.kind == "gate":
            return (f"gate-level noise p1={self.single_qubit_error:g}, "
                    f"p2={self.two_qubit_error:g}")
        if self.kind == "global_per_step":
            return f"global noise p_g={self.global_probability:g} per step"
        if self.kind == "global_end_of_circuit":
            return (f"global noise p_g={self.global_probability:g} "
                    "(one channel, end of circuit)")
        if self.kind == "relaxation":
            return f"thermal relaxation T1={self.t1*1e6:.1f}us, T2={self.t2*1e6:.1f}us"
        return "ideal (no noise)"


# ---------------------------------------------------------------------------
# Local gate action
# ---------------------------------------------------------------------------

def local_matrix(gate: circuit_cost.Gate) -> np.ndarray:
    """The small matrix a gate applies on its own qubits.

    RX/RZ are the same rotations used by ``circuit_cost.gate_matrix``:
    ``RZ(theta) = diag(exp(-i theta/2), exp(+i theta/2))`` and
    ``RX(theta) = cos(theta/2) I - i sin(theta/2) X``.
    """
    theta = gate.angle
    if gate.gate_type == "cnot":
        return CNOT
    if gate.gate_type == "rz":
        return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(complex)
    if gate.gate_type == "rx":
        return np.array(
            [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
             [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    raise ValueError(f"unknown gate type {gate.gate_type!r}")


def apply_gate(rho, gate: circuit_cost.Gate) -> np.ndarray:
    """``U rho U^dagger`` for one gate of the Phase 1 circuit."""
    return apply_operator(rho, local_matrix(gate), gate.qubits)


def gate_noise_channels(gate: circuit_cost.Gate, model: NoiseModel):
    """The channel tuple ``(qubits, p)`` applied after ``gate``, or ``None``."""
    p = model.error_for(gate.gate_type)
    if p == 0.0:
        return None
    return (tuple(gate.qubits), p)


# ---------------------------------------------------------------------------
# Trotter-step boundaries
# ---------------------------------------------------------------------------

def step_ranges(n: int, r: int, t: float = 1.0, J: float = 1.0, h: float = 1.0,
                order: int = 1, *, merge_half_steps: bool = True) -> list[tuple[int, int]]:
    """Index ranges ``[start, stop)`` of the gate list, one per Trotter step.

    The ranges come from the *same* ``circuit_cost.block_sequence`` that
    ``build_circuit`` expands, so they line up with the gate list exactly.

    A first-order step is ``[X, ZZ]`` and ends after its ZZ block; a
    second-order step is ``[ZZ(dt/2), X, ZZ(dt/2)]`` and ends after its X block.
    With ``merge_half_steps`` the two touching half-ZZ layers of neighbouring
    steps are fused into one full ZZ layer, so the final half layer is grouped
    with the last step.  Either way there are exactly ``r`` global channels.
    """
    bonds = circuit_cost.bond_list(n)
    nb = len(bonds)
    blocks = [b for b in circuit_cost.block_sequence(
        n, r, t, J, h, order, merge_half_steps) if b.active]
    if not blocks:
        return [(0, 0)]

    end_kind = "zz" if order == 1 else "x"
    sizes = [3 * nb if block.kind == "zz" else n for block in blocks]
    ranges: list[tuple[int, int]] = []
    start = acc = 0
    for block, size in zip(blocks, sizes):
        acc += size
        if block.kind == end_kind:
            ranges.append((start, start + acc))
            start += acc
            acc = 0
    if acc:
        if ranges:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + acc)
        else:
            ranges.append((0, acc))
    return ranges


# ---------------------------------------------------------------------------
# Reference states (exact and ideal Trotter)
# ---------------------------------------------------------------------------

_CACHE: dict[tuple, object] = {}


def reset_cache() -> None:
    """Drop the cached Hamiltonians, propagators and reference states."""
    _CACHE.clear()


def _psi0(n: int) -> np.ndarray:
    """``|00...0>``: the initial state of ``trotter.sweep`` and of Phases 1-2."""
    vector = np.zeros(2 ** n, dtype=complex)
    vector[0] = 1.0
    return vector


def _hamiltonian(n: int, J: float, h: float):
    key = ("terms", n, float(J), float(h))
    if key not in _CACHE:
        A, B, H = trotter.tfim_terms(n, J, h)
        _CACHE[key] = (A, B, H, trotter.Spectral(H))
    return _CACHE[key]


def exact_state(n: int, t: float = 1.0, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """``exp(-iHt)|00...0>``, the exact reference state."""
    key = ("exact", n, float(t), float(J), float(h))
    if key not in _CACHE:
        _, _, _, spectral = _hamiltonian(n, J, h)
        _CACHE[key] = spectral.exp(t) @ _psi0(n)
    return _CACHE[key]


def ideal_trotter_state(n: int, r: int, order: int = 1, t: float = 1.0,
                        J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """``U_trotter |00...0>`` from ``trotter.trotter_unitary`` (unchanged)."""
    key = ("trotter", n, int(r), int(order), float(t), float(J), float(h))
    if key not in _CACHE:
        A, B, _, _ = _hamiltonian(n, J, h)
        _CACHE[key] = trotter.trotter_unitary(A, B, t, int(r), int(order)) @ _psi0(n)
    return _CACHE[key]


def reference_states(n: int, r: int, order: int = 1, t: float = 1.0,
                     J: float = 1.0, h: float = 1.0) -> dict:
    """Exact and ideal-Trotter states (with their density matrices) for one config."""
    psi_exact = exact_state(n, t, J, h)
    psi_trotter = ideal_trotter_state(n, r, order, t, J, h)
    return {
        "psi_exact": psi_exact,
        "psi_trotter": psi_trotter,
        "rho_exact": pure_density_matrix(psi_exact),
        "rho_trotter": pure_density_matrix(psi_trotter),
    }


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------

def simulate(n: int, r: int, order: int = 1, t: float = 1.0, J: float = 1.0,
             h: float = 1.0, model: NoiseModel | None = None, *,
             merge_half_steps: bool = True) -> dict:
    """Walk the Phase 1 circuit once, with the requested noise model.

    Returns ``{"rho", "resources", "gates", "ranges", "num_channels"}`` where
    ``resources`` is the unchanged ``circuit_cost.CircuitResources`` record, so
    every metric in the CSV can be tied to the Phase 1 gate counts.
    """
    model = model or NoiseModel()
    gates = circuit_cost.build_circuit(n, int(r), t, J, h, order,
                                       merge_half_steps=merge_half_steps)
    resources = circuit_cost.estimate_resources(n, int(r), t, J, h, order,
                                                merge_half_steps=merge_half_steps)
    rho = basis_density_matrix(n, [0] * n)
    channels = 0

    if model.kind in ("none", "gate") or model.is_noiseless:
        if model.kind == "gate":
            for gate in gates:
                rho = apply_gate(rho, gate)
                channel = gate_noise_channels(gate, model)
                if channel is not None:
                    rho = depolarize_qubits(rho, channel[0], channel[1])
                    channels += 1
        else:
            for gate in gates:
                rho = apply_gate(rho, gate)
        ranges: list[tuple[int, int]] = []
    elif model.kind == "relaxation":
        kraus_1q = thermal_relaxation_kraus(model.gate_time_1q, model.t1, model.t2)
        for gate in gates:
            rho = apply_gate(rho, gate)
            if gate.gate_type == "cnot":
                rho = two_qubit_relaxation_channel(rho, gate.qubits, model.gate_time_2q, model.t1, model.t2)
                channels += 1
            else:
                rho = apply_kraus_channel(rho, kraus_1q, (gate.qubits[0],))
                channels += 1
        ranges = []
    elif model.kind == "global_end_of_circuit":
        for gate in gates:
            rho = apply_gate(rho, gate)
        rho = apply_global_depolarizing(rho, model.global_probability)
        channels = 1
        ranges = []
    else:  # global_per_step
        ranges = step_ranges(n, int(r), t, J, h, order,
                             merge_half_steps=merge_half_steps)
        for start, stop in ranges:
            for gate in gates[start:stop]:
                rho = apply_gate(rho, gate)
            rho = apply_global_depolarizing(rho, model.global_probability)
            channels += 1
        if sum(stop - start for start, stop in ranges) != len(gates):
            raise RuntimeError(
                "Trotter-step ranges do not cover the gate list; the block "
                "sequence and build_circuit have diverged")

    return {
        "rho": rho,
        "resources": resources,
        "gates": len(gates),
        "ranges": ranges,
        "num_channels": channels,
        "num_qubits": n,
        "trotter_steps": int(r),
        "trotter_order": int(order),
        "total_time": float(t),
        "initial_state": "0" * n,
    }
