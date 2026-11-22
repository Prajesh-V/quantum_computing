"""Mathematical audit of the Phase 3 conventions.

Every check here compares the repository implementation with an independent
construction (see :mod:`validation.independent`), and records the measured
difference.  A check fails only if its measured quantity is outside a stated
tolerance; nothing is asserted by inspection.

The checks are grouped exactly as the audit brief asks: Hamiltonian, product
formulas, density-matrix evolution, the three channels, the metrics, and the
separation of the three error definitions.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import circuit_cost
import trotter
from noise import channels, density_matrix, metrics, noisy_simulator

from . import independent as ind

__all__ = ["run_all", "error_separation_from_rows"]

TOL = 1e-10


class Recorder:
    """Collects check outcomes and the largest observed deviation."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, section: str, name: str, measured: float, tolerance: float,
            detail: str = "", passed: bool | None = None,
            kind: str = "equality") -> dict:
        """Record one check.

        ``kind`` is ``"equality"`` when the two sides must agree (pass means
        ``measured <= tolerance``) and ``"difference"`` when a *large* value is
        the point -- e.g. showing that reversing the factor order really does
        change the unitary, so that the agreement test above has teeth.
        """
        if passed is None:
            passed = (bool(measured <= tolerance) if kind == "equality"
                      else bool(measured >= tolerance))
        record = {
            "section": section,
            "check": name,
            "measured": float(measured),
            "tolerance": float(tolerance),
            "kind": kind,
            "passed": passed,
            "detail": detail,
        }
        self.checks.append(record)
        return record

    def flag(self, section: str, name: str, value, expected, detail: str = "") -> dict:
        """A boolean/symbolic check (no numeric tolerance)."""
        record = {
            "section": section,
            "check": name,
            "measured": float(value) if isinstance(value, (int, float, np.floating)) else None,
            "tolerance": None,
            "passed": bool(value == expected),
            "detail": detail,
        }
        self.checks.append(record)
        return record

    def summary(self) -> dict:
        passed = sum(1 for check in self.checks if check["passed"])
        equality = [c for c in self.checks if c.get("kind") == "equality"]
        difference = [c for c in self.checks if c.get("kind") == "difference"]
        return {
            "checks": len(self.checks),
            "passed": passed,
            "failed": len(self.checks) - passed,
            "equality_checks": len(equality),
            "difference_checks": len(difference),
            "max_deviation": max((c["measured"] for c in equality), default=0.0),
            "min_difference": min((c["measured"] for c in difference), default=0.0),
        }


def _maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def partial_trace_keep(rho, keep, n: int):
    """Keep only the ``keep`` qubits of an ``n``-qubit density matrix.

    Written with a reshape/transpose/tensordot trace so that it is independent
    of ``noise.density_matrix.partial_trace``.
    """
    traced = [q for q in range(n) if q not in keep]
    tensor = np.asarray(rho, dtype=complex).reshape((2,) * n + (2,) * n)
    order = list(keep) + traced
    tensor = tensor.transpose(order + [n + q for q in order])
    dk, dr = 2 ** len(keep), 2 ** len(traced)
    tensor = tensor.reshape(dk, dr, dk, dr)
    return np.tensordot(tensor, np.eye(dr), axes=([1, 3], [0, 1]))


# ---------------------------------------------------------------------------
# A. Hamiltonian and conventions
# ---------------------------------------------------------------------------

def check_hamiltonian(rec: Recorder, n: int = 4) -> dict:
    J = h = 1.0
    A, B, H = trotter.tfim_terms(n, J, h)

    # independent construction: -J sum_{i<n-1} Z_i Z_{i+1} - h sum_i X_i
    A_ref = np.zeros_like(A)
    for i in range(n - 1):
        A_ref -= J * ind.embed_two(n, i, i + 1, np.kron(ind.Z, ind.Z))
    B_ref = np.zeros_like(B)
    for i in range(n):
        B_ref -= h * ind.embed_one(n, i, ind.X)

    rec.add("hamiltonian", "A matches -J sum_{i<n-1} Z_i Z_{i+1}", _maxdiff(A, A_ref), TOL)
    rec.add("hamiltonian", "B matches -h sum_i X_i", _maxdiff(B, B_ref), TOL)
    rec.add("hamiltonian", "H = A + B", _maxdiff(H, A + B), TOL)
    rec.add("hamiltonian", "H is Hermitian", _maxdiff(H, H.conj().T), TOL)
    rec.add("hamiltonian", "H has the expected dimension",
            abs(H.shape[0] - 2 ** n), 0, f"dim {H.shape[0]} for n={n}")

    # open boundary: n-1 bonds.  A ring would add the (n-1, 0) bond.
    ring = A_ref - J * ind.embed_two(n, n - 1, 0, np.kron(ind.Z, ind.Z))
    rec.add("hamiltonian", "boundary is open, not periodic", _maxdiff(ring, A), 1e-6,
            "a ring Hamiltonian would add the (n-1, 0) bond and differ from A",
            kind="difference")

    # qubit 0 is the most significant bit
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    flipped = ind.embed_one(n, 0, ind.X) @ state
    index0 = int(np.argmax(np.abs(flipped)))
    index_last = int(np.argmax(np.abs(ind.embed_one(n, n - 1, ind.X) @ state)))
    rec.flag("hamiltonian", "qubit 0 is the most significant bit",
             index0, 2 ** (n - 1), f"X on qubit 0 maps |0..0> to index {index0}")
    rec.flag("hamiltonian", "qubit n-1 is the least significant bit",
             index_last, 1, f"X on the last qubit maps |0..0> to index {index_last}")

    # the exact propagator is unitary
    U = expm(-1j * H * 1.0)
    rec.add("hamiltonian", "exp(-iHt) is unitary", _maxdiff(U.conj().T @ U, np.eye(2 ** n)), TOL)
    return {"A": A, "B": B, "H": H}


# ---------------------------------------------------------------------------
# B. Product formulas
# ---------------------------------------------------------------------------

def _u1(A, B, t, r):
    """Independent Lie-Trotter product: exp(-iA dt) exp(-iB dt), r times."""
    dt = t / r
    step = expm(-1j * A * dt) @ expm(-1j * B * dt)
    return np.linalg.matrix_power(step, r)


def _u2(A, B, t, r):
    """Independent Strang product: exp(-iA dt/2) exp(-iB dt) exp(-iA dt/2)."""
    dt = t / r
    half = expm(-1j * A * (dt / 2.0))
    step = half @ expm(-1j * B * dt) @ half
    return np.linalg.matrix_power(step, r)


def check_trotter(rec: Recorder, n: int = 4,
                  cases=((1, 1.0), (2, 1.0), (3, 0.7), (5, 2.0))) -> dict:
    A, B, H = trotter.tfim_terms(n)
    worst = {}
    for order in (1, 2):
        for r, t in cases:
            ref = _u1(A, B, t, r) if order == 1 else _u2(A, B, t, r)
            mine = trotter.trotter_unitary(A, B, t, r, order)
            gates = circuit_cost.build_circuit(n, r, t, 1.0, 1.0, order)
            by_circuit = circuit_cost.circuit_unitary(n, gates)
            by_circuit_ind = ind.circuit_unitary(n, gates)
            tag = f"order {order}, r={r}, t={t}"
            rec.add("trotter", f"trotter_unitary vs independent product ({tag})",
                    _maxdiff(mine, ref), TOL)
            rec.add("trotter", f"circuit unitary vs independent product ({tag})",
                    _maxdiff(by_circuit, ref), TOL)
            rec.add("trotter", f"independent circuit builder agrees ({tag})",
                    _maxdiff(by_circuit_ind, by_circuit), TOL)
            worst[(order, r, t)] = _maxdiff(by_circuit, ref)

    # the product order is load-bearing: reversing it must change the answer,
    # otherwise comparing against it would prove nothing.
    t, r = 1.0, 3
    dt = t / r
    forward = np.linalg.matrix_power(expm(-1j * A * dt) @ expm(-1j * B * dt), r)
    reverse = np.linalg.matrix_power(expm(-1j * B * dt) @ expm(-1j * A * dt), r)
    rec.add("trotter", "reversing the factor order changes U (test is sensitive)",
            _maxdiff(forward, reverse), 1e-3,
            "a difference above the tolerance means the convention matters",
            kind="difference")

    # half-angle factor: the second-order circuit must NOT equal the first-order
    # one at the same r, and must equal the Strang formula (checked above).
    gates1 = circuit_cost.build_circuit(n, r, t, 1.0, 1.0, 1)
    gates2 = circuit_cost.build_circuit(n, r, t, 1.0, 1.0, 2)
    rec.add("trotter", "second-order circuit differs from first-order",
            _maxdiff(circuit_cost.circuit_unitary(n, gates1),
                     circuit_cost.circuit_unitary(n, gates2)), 1e-6,
            kind="difference")
    return {"max_error": max(worst.values())}


# ---------------------------------------------------------------------------
# C. Density-matrix evolution
# ---------------------------------------------------------------------------

def check_density_matrix(rec: Recorder, n: int = 4) -> dict:
    psi = ind.random_pure_state(n, 7)
    rho = np.outer(psi, psi.conj())
    gates = circuit_cost.build_circuit(n, 3, 0.8, 1.0, 1.0, 2)
    pure = noisy_simulator.simulate(n, 3, 2, 0.8, model=noisy_simulator.NoiseModel())
    rho_sim = pure["rho"]

    rec.add("density_matrix", "ideal trace is 1", abs(np.trace(rho_sim) - 1.0), TOL)
    rec.add("density_matrix", "ideal state stays Hermitian",
            _maxdiff(rho_sim, rho_sim.conj().T), TOL)
    rec.add("density_matrix", "ideal state stays positive semidefinite",
            abs(min(0.0, float(np.min(np.linalg.eigvalsh(0.5 * (rho_sim + rho_sim.conj().T)))))),
            TOL)
    rec.add("density_matrix", "ideal evolution preserves purity",
            abs(float(np.real(np.trace(rho_sim @ rho_sim))) - 1.0), 1e-9)

    # the simulator must equal the state-vector circuit evolution
    psi_circuit = ind.circuit_unitary(n, gates) @ np.eye(2 ** n, dtype=complex)[:, 0]
    rho_circuit = np.outer(psi_circuit, psi_circuit.conj())
    rec.add("density_matrix", "simulator equals |psi><psi| from the circuit",
            _maxdiff(rho_sim, rho_circuit), TOL)

    # independent einsum engine vs kron reference
    for qubit in (0, n - 1):
        op = ind._rz(0.31)
        got = ind.apply_local(rho, op, (qubit,))
        full = ind.embed_one(n, qubit, op)
        rec.add("density_matrix", f"independent apply_local on qubit {qubit}",
                _maxdiff(got, full @ rho @ full.conj().T), 1e-14)

    # the audited apply_operator, same comparison
    rec.add("density_matrix", "noise.density_matrix.apply_operator agrees",
            _maxdiff(density_matrix.apply_operator(rho, ind._rz(0.31), (1,)),
                     ind.apply_local(rho, ind._rz(0.31), (1,))), TOL)
    return {"rho": rho_sim}


# ---------------------------------------------------------------------------
# D. Noise channels
# ---------------------------------------------------------------------------

def check_global_channel(rec: Recorder, n: int = 3) -> dict:
    psi = ind.random_pure_state(n, 11)
    rho = np.outer(psi, psi.conj())
    d = 2 ** n

    rec.add("global_channel", "p = 0 is the identity channel",
            _maxdiff(channels.apply_global_depolarizing(rho, 0.0), rho), 0.0)
    rec.add("global_channel", "p = 1 gives I/d",
            _maxdiff(channels.apply_global_depolarizing(rho, 1.0), np.eye(d) / d), TOL)
    for p in (0.01, 0.37, 0.9):
        rec.add("global_channel", f"matches (1-p)rho + p I/d at p={p}",
                _maxdiff(channels.apply_global_depolarizing(rho, p), ind.global_channel(rho, p)),
                TOL)
        out = channels.apply_global_depolarizing(rho, p)
        rec.add("global_channel", f"trace preserved at p={p}", abs(np.trace(out) - 1.0), TOL)
        rec.add("global_channel", f"Hermiticity preserved at p={p}",
                _maxdiff(out, out.conj().T), TOL)
        rec.add("global_channel", f"purity stays in [0,1] at p={p}",
                abs(max(0.0, ind.purity(out) - 1.0)), 0.0)

    # choi_matrix expects the (rho, qubits, **kwargs) signature; the global
    # channel acts on the whole state, so the qubit argument is ignored.
    choi = ind.choi_matrix(lambda rho, qubits, p: ind.global_channel(rho, p), 1, p=0.4)
    rec.add("global_channel", "Choi matrix is positive semidefinite (CP)",
            abs(min(0.0, float(np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)))))), 1e-12)
    rec.add("global_channel", "trace preservation of the Choi matrix",
            abs(np.trace(choi) - 1.0), TOL)
    return {}


def check_single_qubit_channel(rec: Recorder) -> dict:
    """The p/3 channel, checked analytically on four reference states.

    For one qubit, ``sum_{P in {X,Y,Z}} P rho P = 2 Tr(rho) I - rho``, so the
    channel is ``(1 - 4p/3) rho + (2p/3) I`` on a normalised state -- which is
    why only ``p = 3/4`` reaches ``I/2``, not ``p = 1``.
    """
    states = {
        "|0>": np.array([1, 0], dtype=complex),
        "|1>": np.array([0, 1], dtype=complex),
        "|+>": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "|->": np.array([1, -1], dtype=complex) / np.sqrt(2),
    }
    for label, vector in states.items():
        rho = np.outer(vector, vector.conj())
        pauli_sum = sum(density_matrix.apply_operator(rho, matrix, (0,))
                        for matrix in (ind.X, ind.Y, ind.Z))
        # the analytic identity  sum P rho P = 2 Tr(rho) I - rho
        rec.add("single_qubit_channel",
                f"sum over X,Y,Z of P rho P equals 2I - rho for {label}",
                _maxdiff(pauli_sum, 2 * np.eye(2) - rho), 1e-12)
        for p in (0.0, 0.1, 0.75, 1.0):
            got = channels.depolarize_qubits(rho, (0,), p)
            want = (1.0 - p) * rho + (p / 3.0) * (2 * np.eye(2) - rho)
            rec.add("single_qubit_channel", f"{label}, p={p} matches the analytic form",
                    _maxdiff(got, want), TOL)
        rec.add("single_qubit_channel", f"{label} at p = 3/4 gives I/2",
                _maxdiff(channels.depolarize_qubits(rho, (0,), 0.75), np.eye(2) / 2), TOL)
        rec.add("single_qubit_channel", f"{label} at p = 1 does NOT give I/2",
                _maxdiff(channels.depolarize_qubits(rho, (0,), 1.0), np.eye(2) / 2), 1e-6,
                "p=1 leaves (1/3)(2I - rho); the p/3 convention is not uniform-Pauli",
                kind="difference")

    e0 = np.zeros((2, 2), dtype=complex)
    e0[0, 0] = 1.0
    rec.add("single_qubit_channel", "p = 1 on |0> does NOT give I/2 (documented)",
            _maxdiff(channels.depolarize_qubits(e0, (0,), 1.0), np.eye(2) / 2), 1e-6,
            "p=1 leaves diag(1/3, 2/3): the p/3 convention is not the uniform-Pauli one",
            kind="difference")

    # target qubit: the channel on qubit 1 must move the state of qubit 1 only
    two = np.zeros((4, 4), dtype=complex)
    two[0, 0] = 1.0                                   # |00>
    out = channels.depolarize_qubits(two, (1,), 1.0)
    rec.add("single_qubit_channel", "channel acts on the requested qubit",
            _maxdiff(partial_trace_keep(out, [0], 2), np.array([[1, 0], [0, 0]], dtype=complex)),
            TOL)
    rec.add("single_qubit_channel", "untouched qubit stays pure",
            abs(ind.purity(partial_trace_keep(out, [0], 2)) - 1.0), TOL)

    for p in (1e-3, 0.2, 1.0):
        rho = ind.random_pure_state(3, 13)
        rho = np.outer(rho, rho.conj())
        rec.add("single_qubit_channel", f"explicit 3-term sum agrees at p={p}",
                _maxdiff(channels.depolarize_qubits(rho, (2,), p),
                         ind.depolarize_explicit(rho, (2,), p)), 1e-14)
    choi = ind.choi_matrix(ind.depolarize_explicit, 1, p=0.6)
    rec.add("single_qubit_channel", "Choi matrix is positive semidefinite (CP)",
            abs(min(0.0, float(np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)))))), 1e-12)
    return {}


def check_two_qubit_channel(rec: Recorder) -> dict:
    paulis = channels.pauli_operators(2)
    rec.flag("two_qubit_channel", "all 16 two-qubit Paulis are present", len(paulis), 16,
             f"{len(paulis)} operators, labels "
             f"{''.join(sorted(label for label, _ in paulis))}")
    rec.flag("two_qubit_channel", "labels are the 16 distinct strings",
             len({label for label, _ in paulis}), 16)
    rec.flag("two_qubit_channel", "15 non-identity Paulis",
             len([1 for label, _ in paulis if label != "II"]), 15)

    states = {
        "|00>": np.array([1, 0, 0, 0], dtype=complex),
        "|01>": np.array([0, 1, 0, 0], dtype=complex),
        "|++>": np.array([0.5, 0.5, 0.5, 0.5], dtype=complex),
        "Bell": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    }
    for label, vector in states.items():
        rho = np.outer(vector, vector.conj())
        for p in (0.0, 1e-3, 0.3, 1.0):
            rec.add("two_qubit_channel", f"{label}, p={p} matches the 15-term sum",
                    _maxdiff(channels.depolarize_qubits(rho, (0, 1), p),
                             ind.depolarize_explicit(rho, (0, 1), p)), 1e-14)
            out = channels.depolarize_qubits(rho, (0, 1), p)
            rec.add("two_qubit_channel", f"{label}, p={p} preserves trace",
                    abs(np.trace(out) - 1.0), TOL)
            rec.add("two_qubit_channel", f"{label}, p={p} preserves Hermiticity",
                    _maxdiff(out, out.conj().T), TOL)
            rec.add("two_qubit_channel", f"{label}, p={p} stays positive",
                    abs(min(0.0, float(np.min(np.linalg.eigvalsh(0.5 * (out + out.conj().T)))))),
                    TOL)

    # p = 1 is the uniform mixture over the 15 non-identity Paulis, which is
    # NOT I/4; the exact result for |00> is diag(3,4,4,4)/15.
    rho00 = np.zeros((4, 4), dtype=complex)
    rho00[0, 0] = 1.0
    got = channels.depolarize_qubits(rho00, (0, 1), 1.0)
    want = np.diag([3, 4, 4, 4]).astype(complex) / 15.0
    rec.add("two_qubit_channel", "p = 1 equals the exact 15-Pauli average for |00>",
            _maxdiff(got, want), TOL)
    rec.add("two_qubit_channel", "p = 1 is not I/4 (documented)",
            _maxdiff(got, np.eye(4) / 4), 1e-6,
            "diag(3,4,4,4)/15 differs from I/4, as the p/15 convention implies",
            kind="difference")

    # sum over ALL 16 Paulis = 4 Tr_2(rho) (x) I_2, the identity the fast path uses
    rho = ind.random_pure_state(3, 17)
    rho = np.outer(rho, rho.conj())
    total = np.zeros_like(rho)
    for _, matrix in channels.pauli_operators(2):
        total += density_matrix.apply_operator(rho, matrix, (1, 2))
    # sum over Pauli_2 on qubits (1,2) = 2**2 * Tr_{1,2}(rho) (x) I_2
    reduced = partial_trace_keep(rho, [0], 3)
    predicted = 4.0 * np.kron(reduced, np.eye(4))
    rec.add("two_qubit_channel",
            "sum over all 16 Paulis equals 2^k Tr_k(rho) (x) I_k",
            _maxdiff(total, predicted), 1e-10)

    choi = ind.choi_matrix(ind.depolarize_explicit, 2, p=0.5)
    rec.add("two_qubit_channel", "Choi matrix is positive semidefinite (CP)",
            abs(min(0.0, float(np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)))))), 1e-12)
    return {}


# ---------------------------------------------------------------------------
# E. Metrics
# ---------------------------------------------------------------------------

def check_metrics(rec: Recorder) -> dict:
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    e0 = np.outer(zero, zero.conj())
    e1 = np.outer(one, one.conj())
    i2 = np.eye(2) / 2

    rec.add("metrics", "fidelity of identical pure states is 1",
            abs(metrics.fidelity_pure(e0, zero) - 1.0), TOL)
    rec.add("metrics", "fidelity of orthogonal pure states is 0",
            abs(metrics.fidelity_pure(e0, one)), TOL)
    rec.add("metrics", "fidelity with I/2 is 1/2",
            abs(metrics.fidelity_pure(i2, zero) - 0.5), TOL)
    rec.add("metrics", "fidelity with I/2 on |+> is 1/2",
            abs(metrics.fidelity_pure(i2, plus) - 0.5), TOL)
    rec.add("metrics", "infidelity is 1 - fidelity",
            abs(metrics.infidelity(i2, zero) - 0.5), TOL)
    rec.add("metrics", "independent fidelity agrees",
            abs(ind.fidelity(i2, zero) - metrics.fidelity_pure(i2, zero)), 1e-15)

    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho_bell = np.outer(bell, bell.conj())
    rec.add("metrics", "Bell state fidelity with itself is 1",
            abs(metrics.fidelity_pure(rho_bell, bell) - 1.0), TOL)
    rec.add("metrics", "purity of a pure state is 1", abs(ind.purity(rho_bell) - 1.0), TOL)
    rec.add("metrics", "purity of I/2 is 1/2", abs(ind.purity(i2) - 0.5), TOL)

    # the general (Uhlmann) fidelity returns the SQUARED fidelity
    rec.add("metrics", "fidelity_general(|0><0|, I/2) = 1/2 (squared convention)",
            abs(metrics.fidelity_general(e0, i2) - 0.5), TOL)
    rec.add("metrics", "fidelity_general(|0><0|, |1><1|) = 0",
            abs(metrics.fidelity_general(e0, e1)), TOL)

    # trace distance is the trace norm, not the Frobenius norm
    rec.add("metrics", "trace distance(I/2, |0><0|) = 1/2",
            abs(metrics.trace_distance(i2, e0) - 0.5), TOL)
    rec.add("metrics", "trace distance equals the SVD definition",
            abs(metrics.trace_distance(i2, e0) - ind.trace_distance_svd(i2, e0)), TOL)
    frobenius = 0.5 * float(np.linalg.norm(i2 - e0))
    rec.add("metrics", "trace distance is not the Frobenius norm",
            abs(metrics.trace_distance(i2, e0) - frobenius), 1e-6,
            f"trace norm 0.5 vs Frobenius/2 {frobenius:.3f}",
            kind="difference")
    rec.add("metrics", "trace distance of a state with itself is 0",
            abs(metrics.trace_distance(e0, e0)), TOL)
    rec.add("metrics", "trace distance saturates at 1 for |0> vs |1>",
            abs(metrics.trace_distance(e0, e1) - 1.0), TOL)
    return {}


# ---------------------------------------------------------------------------
# F. Error separation
# ---------------------------------------------------------------------------

def error_separation_from_rows(rows) -> dict:
    """Largest deviation of total error from the naive sum of the two parts."""
    deviations = []
    for row in rows:
        deviations.append(row["total_error"]
                          - (row["ideal_trotter_error"] + row["noise_induced_error"]))
    deviations = np.asarray(deviations, dtype=float)
    index = int(np.argmax(np.abs(deviations)))
    worst = rows[index]
    return {
        "rows": len(rows),
        "max_abs_deviation": float(np.max(np.abs(deviations))),
        "min_delta": float(np.min(deviations)),
        "max_delta": float(np.max(deviations)),
        "mean_abs_deviation": float(np.mean(np.abs(deviations))),
        "worst_row": {
            "system_size": worst["system_size"],
            "trotter_order": worst["trotter_order"],
            "trotter_steps": worst["trotter_steps"],
            "total_error": worst["total_error"],
            "ideal_trotter_error": worst["ideal_trotter_error"],
            "noise_induced_error": worst["noise_induced_error"],
            "delta": float(deviations[index]),
        },
    }


def check_metrics_against_stored_metric(rec: Recorder) -> None:
    """``noise.metrics`` and ``validation.independent`` on the same random states."""
    for seed in (1, 2, 3):
        psi = ind.random_pure_state(3, seed)
        rho = np.outer(psi, psi.conj())
        other = ind.random_pure_state(3, seed + 100)
        nim = np.outer(other, other.conj())
        rec.add("metrics", f"independent fidelity agrees with noise.metrics (seed {seed})",
                abs(ind.fidelity(rho, other) - metrics.fidelity_pure(rho, other)), 1e-15)
        rec.add("metrics", f"independent trace distance agrees (seed {seed})",
                abs(ind.trace_distance(rho, nim) - metrics.trace_distance(rho, nim)), 1e-15)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(n: int = 4) -> dict:
    """Run every mathematical check and return a serialisable summary."""
    rec = Recorder()
    check_hamiltonian(rec, n)
    check_trotter(rec, n)
    check_density_matrix(rec, n)
    check_global_channel(rec, n=3)
    check_single_qubit_channel(rec)
    check_two_qubit_channel(rec)
    check_metrics(rec)
    check_metrics_against_stored_metric(rec)
    summary = rec.summary()
    sections: dict[str, dict] = {}
    for check in rec.checks:
        bucket = sections.setdefault(check["section"],
                                     {"checks": 0, "passed": 0, "failed": 0})
        bucket["checks"] += 1
        bucket["passed" if check["passed"] else "failed"] += 1
    return {"summary": summary, "sections": sections, "checks": rec.checks}
