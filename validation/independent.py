"""An independent density-matrix engine, written to audit Phase 3.

Nothing here imports the ``noise`` package.  The circuit gate list still comes
from ``circuit_cost.build_circuit`` -- the decomposition must be the one the
project actually uses -- but the density-matrix algebra, the three depolarising
channels and the metrics are re-implemented from first principles so that a bug
inside ``noise/`` cannot reproduce itself through a shared helper.

Conventions (copied from the repository, then verified against it by
:mod:`validation.audit`):

* qubit 0 is the most significant bit of the basis-state index;
* ``RX(theta) = cos(theta/2) I - i sin(theta/2) X``, i.e. ``exp(-i theta X / 2)``;
* ``RZ(theta) = diag(exp(-i theta/2), exp(+i theta/2))``, i.e. ``exp(-i theta Z / 2)``;
* ``CNOT(c, t)`` flips the target when the control is 1, control = first qubit;
* a gate acts on the state as ``rho -> U rho U^dagger``;
* the default initial state is ``|00...0>``.

Two deliberate differences from the audited code, both for independence:

* local operators are embedded by explicit index arithmetic and applied with a
  single ``einsum``, rather than by reshape/transpose;
* a ``k``-qubit depolarising channel is applied as the explicit
  ``4**k - 1``-term Pauli sum, not through the partial-trace shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

import circuit_cost

__all__ = [
    "Spec",
    "pauli_operators",
    "non_identity_paulis",
    "num_qubits",
    "embed_one",
    "embed_two",
    "apply_local",
    "local_gate_operator",
    "gate_operator",
    "circuit_unitary",
    "global_channel",
    "depolarize_explicit",
    "choi_matrix",
    "fidelity",
    "purity",
    "trace_distance",
    "trace_distance_svd",
    "random_pure_state",
    "gate_list",
    "step_ranges_independent",
    "noisy_evolution",
    "exact_state",
    "trotter_state",
    "reference_states",
]

# ---------------------------------------------------------------------------
# Pauli algebra
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

_SINGLE = [("I", I2), ("X", X), ("Y", Y), ("Z", Z)]


def pauli_operators(k: int):
    """Every ``k``-qubit Pauli string as ``(name, matrix)``, identity included."""
    if k < 1:
        raise ValueError("k must be >= 1")
    out = list(_SINGLE)
    for _ in range(k - 1):
        out = [(name + suffix, np.kron(matrix, other))
               for name, matrix in out for suffix, other in _SINGLE]
    return out


def non_identity_paulis(k: int):
    """All ``4**k - 1`` non-identity Pauli strings."""
    return [(name, matrix) for name, matrix in pauli_operators(k)
            if name != "I" * k]


# ---------------------------------------------------------------------------
# Operator embedding
# ---------------------------------------------------------------------------

def num_qubits(dim: int) -> int:
    """``log2(dim)``, checked to be an exact power of two."""
    n = int(round(np.log2(dim)))
    if 2 ** n != dim:
        raise ValueError(f"{dim} is not a power of two")
    return n


def embed_one(n: int, qubit: int, op) -> np.ndarray:
    """Full ``2**n`` matrix of a 2x2 operator acting on one qubit."""
    factors = [I2] * n
    factors[qubit] = np.asarray(op, dtype=complex)
    out = np.array([[1.0 + 0.0j]])
    for factor in factors:
        out = np.kron(out, factor)
    return out


def embed_two(n: int, i: int, j: int, matrix4) -> np.ndarray:
    """Full ``2**n`` matrix of a 4x4 operator on qubits ``i`` (more significant) and ``j``.

    Written as explicit index arithmetic (not a Kronecker product) so that it
    is an independent check of ``circuit_cost._embed_two``.
    """
    m4 = np.asarray(matrix4, dtype=complex)
    dim = 2 ** n
    out = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        bits = [(column >> (n - 1 - q)) & 1 for q in range(n)]
        local_column = bits[i] * 2 + bits[j]
        for a in range(2):
            for b in range(2):
                amplitude = m4[a * 2 + b, local_column]
                if amplitude == 0.0:
                    continue
                new_bits = list(bits)
                new_bits[i] = a
                new_bits[j] = b
                row = 0
                for bit in new_bits:
                    row = (row << 1) | bit
                out[row, column] += amplitude
    return out


def apply_local(rho, op, qubits) -> np.ndarray:
    """``rho -> (op on qubits) rho (op on qubits)^dagger``.

    The density matrix is viewed as a tensor with ``n`` bra and ``n`` ket
    indices, the target indices are moved to the front, one ``einsum`` applies
    the operator and its conjugate transpose, and the indices are moved back.
    """
    rho = np.asarray(rho, dtype=complex)
    n = num_qubits(rho.shape[0])
    qubits = tuple(int(q) for q in qubits)
    k = len(qubits)
    if k == 0:
        return rho.copy()
    if len(set(qubits)) != k or any(not 0 <= q < n for q in qubits):
        raise ValueError(f"bad qubit list {qubits} for {n} qubits")
    op = np.asarray(op, dtype=complex)
    if op.shape != (2 ** k, 2 ** k):
        raise ValueError(f"operator shape {op.shape} does not fit {k} qubits")

    rest = [q for q in range(n) if q not in qubits]
    order = list(qubits) + rest
    tensor = rho.reshape((2,) * n + (2,) * n)
    tensor = tensor.transpose(order + [n + q for q in order])
    tensor = tensor.reshape(2 ** k, 2 ** (n - k), 2 ** k, 2 ** (n - k))
    # (a, r, c, s) -> (a', r, c', s) with the operator on the bra index and its
    # conjugate transpose on the ket index: op[a, b] rho[b, c] conj(op[d, c]).
    # The third factor is indexed "dc" (not "cd"): transposing it is what makes
    # this rho -> op rho op^dagger.  For real symmetric operators the two are
    # identical, but for Y (antisymmetric) only "dc" is correct.
    tensor = np.einsum("ab,brcs,dc->ards", op, tensor, op.conj())
    tensor = tensor.reshape((2,) * n + (2,) * n)
    back = list(np.argsort(order))
    tensor = tensor.transpose(back + [n + q for q in back])
    return tensor.reshape(2 ** n, 2 ** n)


# ---------------------------------------------------------------------------
# Gates of the Phase 1 circuit
# ---------------------------------------------------------------------------

_CNOT4 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(complex)


def _rx(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def local_gate_operator(gate) -> np.ndarray:
    """The 2x2 or 4x4 matrix a gate applies on its own qubits."""
    if gate.gate_type == "cnot":
        return _CNOT4
    if gate.gate_type == "rz":
        return _rz(gate.angle)
    if gate.gate_type == "rx":
        return _rx(gate.angle)
    raise ValueError(f"unknown gate type {gate.gate_type!r}")


def gate_operator(n: int, gate) -> np.ndarray:
    """Full ``2**n`` matrix of one ``circuit_cost.Gate``."""
    local = local_gate_operator(gate)
    if gate.gate_type == "cnot":
        return embed_two(n, gate.qubits[0], gate.qubits[1], local)
    return embed_one(n, gate.qubits[0], local)


def circuit_unitary(n: int, gates) -> np.ndarray:
    """Product of the gate matrices in application order (first gate first)."""
    unitary = np.eye(2 ** n, dtype=complex)
    for gate in gates:
        unitary = gate_operator(n, gate) @ unitary
    return unitary


def gate_list(n: int, r: int, order: int, t: float = 1.0,
              J: float = 1.0, h: float = 1.0):
    """The chronological gate list of Phase 1 (unchanged decomposition)."""
    return circuit_cost.build_circuit(n, r, t, J, h, order)


def step_ranges_independent(n: int, r: int, order: int, t: float = 1.0,
                            J: float = 1.0, h: float = 1.0):
    """Trotter-step boundaries derived from the block sequence, from scratch.

    Order 1: a step ends after its ZZ block.  Order 2: after its X block, and
    the trailing half ZZ layer is folded into the last step.  Gate sizes are
    counted directly (``3 * bonds`` per ZZ layer, ``n`` per X layer), so this
    does not reuse ``noise.noisy_simulator.step_ranges``.
    """
    bonds = len(circuit_cost.bond_list(n))
    blocks = [block for block in
              circuit_cost.block_sequence(n, r, t, J, h, order) if block.active]
    ranges = []
    start = 0
    pending = 0
    end_kind = "zz" if order == 1 else "x"
    for block in blocks:
        pending += 3 * bonds if block.kind == "zz" else n
        if block.kind == end_kind:
            ranges.append((start, start + pending))
            start += pending
            pending = 0
    if pending:
        if ranges:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + pending)
        else:
            ranges = [(0, pending)]
    return ranges


# ---------------------------------------------------------------------------
# Noise channels, as explicit sums
# ---------------------------------------------------------------------------

def _maximally_mixed(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=complex) / dim


def global_channel(rho, p: float) -> np.ndarray:
    """``(1 - p) rho + p I/d``."""
    rho = np.asarray(rho, dtype=complex)
    _check_probability(p)
    dim = rho.shape[0]
    return (1.0 - p) * rho + p * _maximally_mixed(dim)


def depolarize_explicit(rho, qubits, p: float) -> np.ndarray:
    """``(1 - p) rho + p/(4**k - 1) sum_{P != I} P rho P`` on ``qubits``.

    This is the explicit Pauli sum: no partial-trace identity is used, so it is
    an independent check of ``noise.channels.depolarize_qubits``.
    """
    rho = np.asarray(rho, dtype=complex)
    _check_probability(p)
    qubits = tuple(int(q) for q in qubits)
    k = len(qubits)
    if p == 0.0:
        return rho.copy()
    total = (1.0 - p) * rho
    weight = p / (4 ** k - 1)
    for _, matrix in non_identity_paulis(k):
        total = total + weight * apply_local(rho, matrix, qubits)
    return total


def _check_probability(p: float) -> None:
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"probability must lie in [0, 1], got {p!r}")


def choi_matrix(channel, k: int, **kwargs) -> np.ndarray:
    """Choi matrix of a ``k``-qubit channel: ``(id (x) E)(|Omega><Omega|)``.

    ``channel`` has the ``(rho, qubits, p)`` signature used above.  A channel
    is completely positive exactly when this matrix is positive semidefinite.
    """
    dim = 2 ** k
    omega = np.zeros(dim * dim, dtype=complex)
    omega[:: dim + 1] = 1.0 / np.sqrt(dim)      # sum_i |i>|i> / sqrt(d)
    rho = np.outer(omega, omega.conj())
    return channel(rho, tuple(range(k, 2 * k)), **kwargs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def fidelity(rho, psi) -> float:
    """``<psi| rho |psi>`` for a normalised pure reference state."""
    rho = np.asarray(rho, dtype=complex)
    vector = np.asarray(psi, dtype=complex).reshape(-1)
    vector = vector / np.linalg.norm(vector)
    value = np.vdot(vector, rho @ vector)
    return float(min(1.0, max(0.0, np.real(value))))


def purity(rho) -> float:
    """``Tr(rho**2)``."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))


def trace_distance(rho, sigma) -> float:
    """``0.5 * ||rho - sigma||_1`` from the eigenvalues of a Hermitian difference."""
    difference = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    hermitian = 0.5 * (difference + difference.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def trace_distance_svd(rho, sigma) -> float:
    """The same quantity from singular values -- the textbook definition.

    Used by the audit to confirm that ``trace_distance`` really is the trace
    norm and not, say, the Frobenius norm.
    """
    difference = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    return float(0.5 * np.sum(np.linalg.svd(difference, compute_uv=False)))


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

def random_pure_state(n: int, seed: int) -> np.ndarray:
    """A reproducibly generated Haar-ish random pure state."""
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    return vector / np.linalg.norm(vector)


def _default_state(n: int) -> np.ndarray:
    vector = np.zeros(2 ** n, dtype=complex)
    vector[0] = 1.0
    return vector


def exact_state(n: int, t: float = 1.0, J: float = 1.0, h: float = 1.0,
                psi0=None) -> np.ndarray:
    """``exp(-i H t)|psi0>`` from a dense matrix exponential."""
    import trotter
    _, _, hamiltonian = trotter.tfim_terms(n, J, h)
    psi0 = _default_state(n) if psi0 is None else np.asarray(psi0, dtype=complex)
    return expm(-1j * hamiltonian * t) @ psi0


def trotter_state(n: int, r: int, order: int = 1, t: float = 1.0,
                  J: float = 1.0, h: float = 1.0, psi0=None) -> np.ndarray:
    """The product formula applied to ``psi0`` through the **circuit**, not the formula.

    Building the unitary from the gate list is a second, independent path to
    the ideal Trotter state: ``trotter.trotter_unitary`` composes matrix
    exponentials, this composes the compiled circuit.
    """
    psi0 = _default_state(n) if psi0 is None else np.asarray(psi0, dtype=complex)
    return circuit_unitary(n, gate_list(n, r, order, t, J, h)) @ psi0


def reference_states(n: int, r: int, order: int = 1, t: float = 1.0,
                     J: float = 1.0, h: float = 1.0, psi0=None) -> dict:
    """Exact and ideal-Trotter states for one configuration."""
    psi_exact = exact_state(n, t, J, h, psi0)
    psi_trotter = trotter_state(n, r, order, t, J, h, psi0)
    return {"psi_exact": psi_exact, "psi_trotter": psi_trotter}


# ---------------------------------------------------------------------------
# Noisy evolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    """A noise configuration, in the same four variants Phase 3 used."""

    # none | gate | global_per_step | global_end_of_circuit  (as in Phase 3)
    kind: str = "none"
    p1: float = 0.0
    p2: float = 0.0
    pg: float = 0.0
    rz_error: float | None = None

    def error_for(self, gate_type: str) -> float:
        if gate_type == "cnot":
            return self.p2
        if gate_type == "rz":
            return self.p1 if self.rz_error is None else self.rz_error
        return self.p1

    def label(self) -> str:
        if self.kind == "gate":
            return f"gate-level p1={self.p1:g} p2={self.p2:g}"
        if self.kind == "global_per_step":
            return f"global per step p_g={self.pg:g}"
        if self.kind == "global_end_of_circuit":
            return f"global end of circuit p_g={self.pg:g}"
        return "ideal"


def noisy_evolution(n: int, r: int, order: int = 1, t: float = 1.0,
                    spec: Spec | None = None, psi0=None,
                    J: float = 1.0, h: float = 1.0) -> dict:
    """Walk the Phase 1 gate list once with the requested noise.

    Returns ``{"rho", "psi0", "gates", "num_channels", "ranges", "resources"}``.
    """
    spec = spec or Spec()
    gates = gate_list(n, r, order, t, J, h)
    psi0 = _default_state(n) if psi0 is None else np.asarray(psi0, dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)
    rho = np.outer(psi0, psi0.conj())
    channels = 0
    ranges: list[tuple[int, int]] = []

    if spec.kind in ("none", "gate"):
        for gate in gates:
            rho = apply_local(rho, local_gate_operator(gate), gate.qubits)
            if spec.kind == "gate":
                p = spec.error_for(gate.gate_type)
                if p:
                    rho = depolarize_explicit(rho, gate.qubits, p)
                    channels += 1
    elif spec.kind == "global_end_of_circuit":
        for gate in gates:
            rho = apply_local(rho, local_gate_operator(gate), gate.qubits)
        rho = global_channel(rho, spec.pg)
        channels = 1
    elif spec.kind == "global_per_step":
        ranges = step_ranges_independent(n, r, order, t, J, h)
        for start, stop in ranges:
            for gate in gates[start:stop]:
                rho = apply_local(rho, local_gate_operator(gate), gate.qubits)
            rho = global_channel(rho, spec.pg)
            channels += 1
        if sum(stop - start for start, stop in ranges) != len(gates):
            raise RuntimeError("step ranges do not cover the gate list")
    else:
        raise ValueError(f"unknown noise kind {spec.kind!r}")

    return {
        "rho": rho,
        "psi0": psi0,
        "gates": len(gates),
        "num_channels": channels,
        "ranges": ranges,
        "resources": circuit_cost.estimate_resources(n, r, t, J, h, order),
    }


def measure(n: int, r: int, order: int = 1, t: float = 1.0,
            spec: Spec | None = None, psi0=None, J: float = 1.0,
            h: float = 1.0) -> dict:
    """Every Phase 3 metric, computed independently for one configuration."""
    spec = spec or Spec()
    out = noisy_evolution(n, r, order, t, spec, psi0, J, h)
    rho = out["rho"]
    refs = reference_states(n, r, order, t, J, h, out["psi0"])
    rho_exact = np.outer(refs["psi_exact"], refs["psi_exact"].conj())
    value = fidelity(rho, refs["psi_exact"])
    resources = out["resources"]
    return {
        "system_size": n,
        "trotter_order": int(order),
        "trotter_steps": int(r),
        "total_time": float(t),
        "single_qubit_error": float(spec.p1),
        "two_qubit_error": float(spec.p2),
        "global_noise_probability": float(spec.pg),
        "noise_model": spec.kind,
        "cnot_count": resources.num_cnot,
        "rx_count": resources.num_rx,
        "rz_count": resources.num_rz,
        "total_gate_count": resources.total_gates,
        "logical_depth": resources.circuit_depth,
        "two_qubit_depth": resources.two_qubit_depth,
        "fidelity": value,
        "total_error": 1.0 - value,
        "ideal_trotter_error": 1.0 - fidelity(rho_exact, refs["psi_trotter"]),
        "noise_induced_error": 1.0 - fidelity(rho, refs["psi_trotter"]),
        "purity": purity(rho),
        "trace_distance": trace_distance(rho, rho_exact),
        "noise_channels_applied": out["num_channels"],
    }
