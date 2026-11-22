#!/usr/bin/env python3
"""
Circuit-level resource estimation for Trotterised spin-chain evolution
======================================================================

Phase 1 of the project: turn the state-vector Trotter simulation in
``trotter.py`` into a *gate-level resource estimate*, and verify that the
gate decomposition really reproduces the matrix evolution.

What is counted
---------------
For the transverse-field Ising chain of ``trotter.py`` (open boundary)::

    H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}  -  h * sum_{i=0}^{n-1} X_i  =  A + B

the two product formulas are

    1st order :  [ exp(-i A dt) exp(-i B dt) ]^r
    2nd order :  [ exp(-i A dt/2) exp(-i B dt) exp(-i A dt/2) ]^r,   dt = t/r

and each block is compiled to elementary gates:

    exp(-i A dt)  =  prod_bonds CNOT(i,i+1) . RZ(-2 J dt) . CNOT(i,i+1)
    exp(-i B dt)  =  prod_i     RX(-2 h dt)

using  CNOT(c,t) . RZ(theta)_t . CNOT(c,t) = exp(-i theta/2 Z_c Z_t)  and
RX(theta) = exp(-i theta/2 X).  Every rotation angle is therefore
``-2 * (coefficient of the Pauli term in H) * (time span of the block)``.

SIGN CONVENTION: the minus signs in H above are the ones used by trotter.py,
so the Ising angles are negative.  A Hamiltonian written as +J sum ZZ + h sum X
would give the same *gate counts* with the opposite angles; the numbers below
are counts, which do not depend on that sign.

Zero-angle policy
-----------------
A block whose rotation angle is exactly zero (J = 0 or h = 0) is dropped
completely, including the pair of CNOTs that would implement it, because
CNOT(i,j).CNOT(i,j) = I.  ``keep_zero_angles=True`` keeps the symbolic
operations instead, which is useful for structural comparisons.

Circuit depth
-------------
Depth is *not* the gate count.  The gate list keeps the order in which the
product formula applies its blocks, and ``schedule`` assigns each gate the
earliest layer in which all of its qubits are free (ASAP list scheduling).
Gates on disjoint qubits therefore run in parallel.  This is a conservative,
logical depth: it does not exploit that the ZZ bonds of one block commute and
could be re-ordered into even/odd rounds, and it says nothing about hardware
timings, connectivity or pulse calibration.

Usage
-----
    python circuit_cost.py                 # sweep, tables, 6 plots
    python circuit_cost.py --rmax 50 --no-plots

Outputs (default): results/circuit_resources.csv, results/circuit_reports.json
and plots H-M in results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import expm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

import trotter  # noqa: E402  (existing physics: Hamiltonians, exact evolution)


# ---------------------------------------------------------------------------
# 1. Resource representation
# ---------------------------------------------------------------------------

GATE_TYPES = ("cnot", "rx", "rz")


@dataclass
class Gate:
    """One elementary operation in the compiled circuit.

    Attributes
    ----------
    gate_type : {"cnot", "rx", "rz"}
    qubits    : tuple of qubit indices the gate touches (control first for CNOT)
    angle     : rotation angle in radians (0.0 for CNOT)
    layer     : time step assigned by the scheduler (-1 before scheduling)
    """

    gate_type: str
    qubits: tuple[int, ...]
    angle: float = 0.0
    layer: int = -1

    def __post_init__(self) -> None:
        if self.gate_type not in GATE_TYPES:
            raise ValueError(f"unknown gate type {self.gate_type!r}")
        expected = 2 if self.gate_type == "cnot" else 1
        if len(self.qubits) != expected:
            raise ValueError(
                f"{self.gate_type} takes {expected} qubit(s), got {len(self.qubits)}"
            )
        if not all(isinstance(q, int) and q >= 0 for q in self.qubits):
            raise ValueError(f"qubit indices must be non-negative ints: {self.qubits}")

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)


@dataclass
class CircuitResources:
    """Aggregated logical resources of one compiled Trotter circuit."""

    num_qubits: int
    num_cnot: int
    num_rx: int
    num_rz: int
    total_gates: int
    circuit_depth: int
    two_qubit_depth: int
    trotter_steps: int
    formula_order: int
    # Extra bookkeeping, kept explicit so the second-order merging is auditable.
    num_zz_layers: int = 0
    num_x_layers: int = 0
    half_step_merging: bool = True
    zero_angles_kept: bool = False

    @property
    def cnot_per_step(self) -> float:
        return self.num_cnot / self.trotter_steps

    @property
    def rx_per_step(self) -> float:
        return self.num_rx / self.trotter_steps

    @property
    def rz_per_step(self) -> float:
        return self.num_rz / self.trotter_steps

    @property
    def gates_per_step(self) -> float:
        return self.total_gates / self.trotter_steps

    def to_dict(self) -> dict:
        out = asdict(self)
        out["per_step"] = {
            "cnot": self.cnot_per_step,
            "rx": self.rx_per_step,
            "rz": self.rz_per_step,
            "total": self.gates_per_step,
        }
        return out


# ---------------------------------------------------------------------------
# 2. Parameters and block sequence
# ---------------------------------------------------------------------------

def bond_list(n: int, boundary: str = "open") -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of the chain.

    Only open boundaries are supported in Phase 1 (``n - 1`` bonds); a periodic
    ring would add the bond (n-1, 0) and is rejected rather than silently
    approximated.
    """
    if boundary == "open":
        return [(i, i + 1) for i in range(n - 1)]
    raise ValueError(
        f"unsupported boundary condition {boundary!r}: Phase 1 implements 'open' only"
    )


def validate_parameters(
    n: int, r: int, t: float, J: float, h: float, order: int, boundary: str = "open"
) -> None:
    """Reject inputs the estimator cannot or should not represent.

    ``J`` or ``h`` may be exactly zero (the corresponding block is then dropped,
    see the module docstring).  ``t`` may be negative: every block angle simply
    changes sign and the circuit is the inverse evolution, which the existing
    ``trotter.py`` supports.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if isinstance(r, bool) or not isinstance(r, (int, np.integer)):
        raise TypeError(f"r must be an integer, got {type(r).__name__}")
    if n < 2:
        raise ValueError(f"n must be >= 2 (need at least one bond), got {n}")
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    if order not in (1, 2):
        raise ValueError(f"unsupported formula order {order!r}: use 1 or 2")
    for name, value in (("t", t), ("J", J), ("h", h)):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    bond_list(n, boundary)  # validates the boundary condition


@dataclass
class Block:
    """One exponentiated block of the product formula, before gate expansion."""

    kind: str          # "zz" or "x"
    span: float        # simulated time covered by this block
    rz_angle: float    # angle of the RZ gate implementing a zz block
    rx_angle: float    # angle of the RX gate implementing an x block
    active: bool = True

    @property
    def angle(self) -> float:
        return self.rz_angle if self.kind == "zz" else self.rx_angle


def block_sequence(
    n: int,
    r: int,
    t: float,
    J: float,
    h: float,
    order: int,
    merge_half_steps: bool = True,
    keep_zero_angles: bool = False,
    boundary: str = "open",
) -> list[Block]:
    """Blocks of the product formula, in application order.

    Order 1: ``r`` repetitions of [X(dt), ZZ(dt)].  The order matters: trotter.py
    forms the step as ``exp(-i A dt) exp(-i B dt)``, and a matrix product applies
    its rightmost factor first, so the X block is emitted before the ZZ block.
    Order 2: ``r`` repetitions of [ZZ(dt/2), X(dt), ZZ(dt/2)] (symmetric, so the
    application order is unambiguous).

    With ``merge_half_steps`` an adjacent ZZ/half-ZZ pair is fused into a single
    ZZ block (the two rotations act on the same qubits with nothing in between,
    so RZ(a).RZ(b) = RZ(a+b)).  For r steps that turns 2r half-layers into
    r + 1 layers: two half-angle layers at the two ends and r - 1 full layers.
    """
    validate_parameters(n, r, t, J, h, order, boundary)
    dt = t / r
    raw: list[tuple[str, float]] = []
    if order == 1:
        for _ in range(r):
            raw.append(("x", dt))
            raw.append(("zz", dt))
    else:
        for _ in range(r):
            raw.append(("zz", dt / 2.0))
            raw.append(("x", dt))
            raw.append(("zz", dt / 2.0))

    if merge_half_steps:
        merged: list[list] = []
        for kind, span in raw:
            if merged and merged[-1][0] == kind == "zz":
                merged[-1][1] += span          # two adjacent half steps -> one
            else:
                merged.append([kind, span])
        raw = [(kind, span) for kind, span in merged]

    blocks = []
    for kind, span in raw:
        if kind == "zz":
            angle = -2.0 * J * span            # exp(-i A dt) with A = -J sum ZZ
        else:
            angle = -2.0 * h * span            # exp(-i B dt) with B = -h sum X
        active = keep_zero_angles or angle != 0.0
        blocks.append(Block(kind=kind, span=span, rz_angle=angle if kind == "zz" else 0.0,
                            rx_angle=angle if kind == "x" else 0.0, active=active))
    return blocks


def build_circuit(
    n: int,
    r: int,
    t: float,
    J: float = 1.0,
    h: float = 1.0,
    order: int = 1,
    *,
    merge_half_steps: bool = True,
    keep_zero_angles: bool = False,
    boundary: str = "open",
) -> list[Gate]:
    """Expand the product formula into an ordered list of elementary gates."""
    bonds = bond_list(n, boundary)
    gates: list[Gate] = []
    for block in block_sequence(n, r, t, J, h, order, merge_half_steps,
                                keep_zero_angles, boundary):
        if not block.active:
            continue                       # policy: whole zero-angle block dropped
        if block.kind == "zz":
            for i, j in bonds:
                gates.append(Gate("cnot", (i, j)))
                gates.append(Gate("rz", (j,), block.rz_angle))
                gates.append(Gate("cnot", (i, j)))
        else:
            for i in range(n):
                gates.append(Gate("rx", (i,), block.rx_angle))
    return gates


# ---------------------------------------------------------------------------
# 3. Scheduling and depth
# ---------------------------------------------------------------------------

def schedule(gates: list[Gate]) -> list[Gate]:
    """ASAP list scheduling: earliest feasible layer for every gate.

    A gate may enter layer L only if every qubit it touches was last used in a
    layer < L.  Gates keep their relative order, so non-commuting operations are
    never reordered.  Gates on disjoint qubits therefore share a layer.
    """
    last: dict[int, int] = {}
    for gate in gates:
        start = max((last.get(q, -1) for q in gate.qubits), default=-1) + 1
        gate.layer = start
        for q in gate.qubits:
            last[q] = start
    return gates


def circuit_depth(gates: list[Gate]) -> int:
    """Number of layers occupied by the scheduled circuit."""
    if not gates:
        return 0
    return max(g.layer for g in gates) + 1


def two_qubit_depth(gates: list[Gate]) -> int:
    """Layers of the full schedule that contain at least one two-qubit gate.

    Conservative: a layer holding a CNOT and a parallel RX counts as a
    two-qubit layer, and CNOTs separated by an RZ on the same qubit are never
    put in the same layer.
    """
    return len({g.layer for g in gates if g.gate_type == "cnot"})


# ---------------------------------------------------------------------------
# 4. Resource estimation and analytical reference
# ---------------------------------------------------------------------------

def estimate_resources(
    n: int,
    r: int,
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    order: int = 1,
    *,
    merge_half_steps: bool = True,
    keep_zero_angles: bool = False,
    boundary: str = "open",
) -> CircuitResources:
    """Compile the circuit and count gates and depth."""
    bonds = bond_list(n, boundary)
    gates = build_circuit(n, r, t, J, h, order,
                          merge_half_steps=merge_half_steps,
                          keep_zero_angles=keep_zero_angles,
                          boundary=boundary)
    schedule(gates)
    blocks = block_sequence(n, r, t, J, h, order, merge_half_steps,
                            keep_zero_angles, boundary)
    counts = {name: 0 for name in GATE_TYPES}
    for gate in gates:
        counts[gate.gate_type] += 1
    return CircuitResources(
        num_qubits=n,
        num_cnot=counts["cnot"],
        num_rx=counts["rx"],
        num_rz=counts["rz"],
        total_gates=len(gates),
        circuit_depth=circuit_depth(gates),
        two_qubit_depth=two_qubit_depth(gates),
        trotter_steps=r,
        formula_order=order,
        num_zz_layers=sum(1 for b in blocks if b.kind == "zz" and b.active),
        num_x_layers=sum(1 for b in blocks if b.kind == "x" and b.active),
        half_step_merging=merge_half_steps,
        zero_angles_kept=keep_zero_angles,
    )


def analytic_counts(
    n: int,
    r: int,
    order: int,
    J: float = 1.0,
    h: float = 1.0,
    *,
    merge_half_steps: bool = True,
    keep_zero_angles: bool = False,
    boundary: str = "open",
) -> dict[str, int]:
    """Closed-form gate counts, derived from the block sequence by hand.

    Open chain, ``nb = n - 1`` bonds, none of the angles zero:

        order 1:  CNOT = 2 r nb,        RZ = r nb,          RX = r n
                  total = r (4 n - 3)

        order 2, half steps merged (r+1 ZZ layers, r X layers):
                  CNOT = 2 (r+1) nb,    RZ = (r+1) nb,      RX = r n
                  total = 3 (r+1) nb + r n

        order 2, half steps NOT merged (2r ZZ layers, r X layers):
                  CNOT = 4 r nb,        RZ = 2 r nb,        RX = r n
                  total = 4 r nb + 2 r nb + r n = r (7 n - 6)

    A zero angle removes its whole block (J = 0: no ZZ gates, h = 0: no RX).
    """
    validate_parameters(n, r, 1.0, J, h, order, boundary)
    nb = len(bond_list(n, boundary))
    if order == 1:
        zz_layers, x_layers = r, r
    elif merge_half_steps:
        zz_layers, x_layers = r + 1, r
    else:
        zz_layers, x_layers = 2 * r, r

    zz_on = keep_zero_angles or J != 0.0
    x_on = keep_zero_angles or h != 0.0
    cnot = 2 * nb * zz_layers if zz_on else 0
    rz = nb * zz_layers if zz_on else 0
    rx = n * x_layers if x_on else 0
    return {"cnot": cnot, "rz": rz, "rx": rx, "total": cnot + rz + rx,
            "zz_layers": zz_layers if zz_on else 0,
            "x_layers": x_layers if x_on else 0}


def validate_counts(
    n: int,
    r: int,
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    order: int = 1,
    *,
    merge_half_steps: bool = True,
    keep_zero_angles: bool = False,
    boundary: str = "open",
) -> dict:
    """Compare compiled counts with the closed-form reference.

    Returns ``{"ok": bool, "compiled": {...}, "analytic": {...}, "differences": {...}}``.
    """
    res = estimate_resources(n, r, t, J, h, order,
                             merge_half_steps=merge_half_steps,
                             keep_zero_angles=keep_zero_angles, boundary=boundary)
    ref = analytic_counts(n, r, order, J, h, merge_half_steps=merge_half_steps,
                          keep_zero_angles=keep_zero_angles, boundary=boundary)
    compiled = {"cnot": res.num_cnot, "rz": res.num_rz, "rx": res.num_rx,
                "total": res.total_gates}
    diffs = {k: compiled[k] - ref[k] for k in compiled}
    return {"ok": all(v == 0 for v in diffs.values()),
            "compiled": compiled, "analytic": ref, "differences": diffs}


# ---------------------------------------------------------------------------
# 5. Independent verification of the gate decomposition
# ---------------------------------------------------------------------------

_CNOT4 = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=complex)


def _embed_two(n: int, i: int, j: int, matrix4: np.ndarray) -> np.ndarray:
    """Embed a 4x4 operator acting on qubits i < j (i is the more significant)."""
    if i > j:
        i, j = j, i
    left = np.eye(2 ** i, dtype=complex)
    right = np.eye(2 ** (n - j - 1), dtype=complex)
    return np.kron(np.kron(left, matrix4), right)


def gate_matrix(n: int, gate: Gate) -> np.ndarray:
    """Full 2^n x 2^n matrix of one gate."""
    theta = gate.angle
    if gate.gate_type == "cnot":
        return _embed_two(n, gate.qubits[0], gate.qubits[1], _CNOT4)
    if gate.gate_type == "rz":
        op = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    else:  # rx
        op = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                       [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    return trotter.single_qubit(n, gate.qubits[0], op)


def circuit_unitary(n: int, gates: list[Gate]) -> np.ndarray:
    """Product of the gates in application order (leftmost applied first)."""
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for gate in gates:
        U = gate_matrix(n, gate) @ U
    return U


def verify_gate_decomposition(
    n: int = 4,
    r: int = 3,
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    tol: float = 1e-10,
) -> dict:
    """Check that the compiled circuit reproduces ``trotter.trotter_unitary``.

    For both formula orders and both merging policies, build the gate list,
    multiply the 2^n x 2^n gate matrices and compare with the matrix product
    formula.  Also checks the two gate identities used for the decomposition.
    """
    A, B, H = trotter.tfim_terms(n, J, h)
    out: dict = {"n": n, "r": r, "t": t, "checks": {}, "max_error": 0.0}

    # identity 1: CNOT.RZ(theta).CNOT = exp(-i theta/2 Z Z)
    for theta in (-2.0 * J * t, 0.7, 2.5):
        rz = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
        circ = (_embed_two(n, 0, 1, _CNOT4)
                @ trotter.single_qubit(n, 1, rz)
                @ _embed_two(n, 0, 1, _CNOT4))
        target = expm(-1j * (theta / 2.0) * trotter.two_qubit(n, 0, 1, trotter.Z, trotter.Z))
        err = float(np.linalg.norm(circ - target, 2))
        out["checks"][f"zz_identity_theta={theta:+.3f}"] = err
        out["max_error"] = max(out["max_error"], err)

    # identity 2: RX(theta) = exp(-i theta/2 X)
    for theta in (-2.0 * h * t, 0.7, 2.5):
        got = gate_matrix(n, Gate("rx", (1,), theta))
        target = expm(-1j * (theta / 2.0) * trotter.single_qubit(n, 1, trotter.X))
        err = float(np.linalg.norm(got - target, 2))
        out["checks"][f"rx_identity_theta={theta:+.3f}"] = err
        out["max_error"] = max(out["max_error"], err)

    # full circuit vs matrix product formula
    for order in (1, 2):
        for merge in (True, False):
            gates = build_circuit(n, r, t, J, h, order, merge_half_steps=merge)
            U_circ = circuit_unitary(n, gates)
            U_ref = trotter.trotter_unitary(A, B, t, r, order)
            err = float(np.linalg.norm(U_circ - U_ref, 2))
            out["checks"][f"order{order}_merged={merge}"] = err
            out["max_error"] = max(out["max_error"], err)

    out["ok"] = out["max_error"] <= tol
    return out


# ---------------------------------------------------------------------------
# 6. Reports
# ---------------------------------------------------------------------------

def resource_report(
    n: int,
    r: int,
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    order: int = 1,
    *,
    merge_half_steps: bool = True,
    keep_zero_angles: bool = False,
    boundary: str = "open",
) -> dict:
    """Structured resource report for one experiment configuration."""
    res = estimate_resources(n, r, t, J, h, order,
                             merge_half_steps=merge_half_steps,
                             keep_zero_angles=keep_zero_angles, boundary=boundary)
    return {
        "system": {
            "model": "TFIM",
            "num_qubits": n,
            "interaction_strength": float(J),
            "transverse_field": float(h),
            "simulation_time": float(t),
            "boundary_condition": boundary,
        },
        "trotter": {
            "formula_order": order,
            "steps": r,
            "step_size": float(t / r),
            "half_step_merging": merge_half_steps,
        },
        "resources": {
            "cnot_count": res.num_cnot,
            "rx_count": res.num_rx,
            "rz_count": res.num_rz,
            "total_gates": res.total_gates,
            "circuit_depth": res.circuit_depth,
            "two_qubit_depth": res.two_qubit_depth,
            "cnot_per_step": res.cnot_per_step,
            "rx_per_step": res.rx_per_step,
            "rz_per_step": res.rz_per_step,
            "a_layers": res.num_zz_layers,
            "x_layers": res.num_x_layers,
        },
        "assumptions": {
            "zero_angles_kept": keep_zero_angles,
            "scheduling": "ASAP list scheduling, conservative, logical depth only",
            "hamiltonian_sign": "H = -J sum ZZ - h sum X (as in trotter.py)",
            "gate_set": "CNOT, RX, RZ",
        },
    }


def write_json(reports, path: str) -> None:
    """Write one report or a list of reports as JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
        fh.write("\n")


RESOURCE_COLUMNS = [
    "model", "n", "order", "r", "initial_state",
    "operator_norm_error", "state_infidelity",
    "qubits", "cnot_count", "rx_count", "rz_count", "total_gates",
    "circuit_depth", "two_qubit_depth",
]


def write_csv(rows, path: str) -> None:
    """Write resource rows (lists aligned with RESOURCE_COLUMNS) as CSV."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESOURCE_COLUMNS)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 7. Study: join numerical error with circuit resources
# ---------------------------------------------------------------------------

def study(
    ns=(4, 5, 6),
    t: float = 1.0,
    rs=None,
    J: float = 1.0,
    h: float = 1.0,
    orders=(1, 2),
    *,
    merge_half_steps: bool = True,
    boundary: str = "open",
) -> dict:
    """Run the Trotter sweep and attach gate resources to every (n, r, order).

    The numerical errors come from ``trotter.sweep`` unchanged: this function
    only reads them.
    """
    if rs is None:
        rs = np.unique(np.round(np.geomspace(1, 200, 26)).astype(int))
    rs = np.asarray(rs, dtype=int)

    data: dict[int, dict[int, dict]] = {}
    rows: list[list] = []

    for n in ns:
        A, B, H = trotter.tfim_terms(n, J, h)
        res = trotter.sweep(A, B, H, t, rs, orders=orders)
        data[n] = {}
        for order in orders:
            resources = [estimate_resources(n, int(r), t, J, h, order,
                                           merge_half_steps=merge_half_steps,
                                           boundary=boundary) for r in rs]
            data[n][order] = {
                "err_op": res[order]["op"],
                "err_inf": res[order]["inf"],
                "resources": resources,
                "cnot": np.array([q.num_cnot for q in resources]),
                "total": np.array([q.total_gates for q in resources]),
                "depth": np.array([q.circuit_depth for q in resources]),
            }
            for r, e_op, e_inf, q in zip(rs, res[order]["op"], res[order]["inf"],
                                         resources):
                rows.append(["TFIM", n, order, int(r), "0" * n,
                             f"{e_op:.12e}", f"{e_inf:.12e}",
                             q.num_qubits, q.num_cnot, q.num_rx, q.num_rz,
                             q.total_gates, q.circuit_depth, q.two_qubit_depth])

    return {"rs": rs, "ns": tuple(ns), "data": data, "rows": rows,
            "t": t, "J": J, "h": h}


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

_COLORS = {4: "#1b4965", 5: "#5fa8d3", 6: "#c1666b"}
_MARKERS = {1: "o", 2: "s"}
_LABELS = {1: "1st order", 2: "2nd order"}


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_vs_cnot(result: dict, path: str, n: int = 4) -> None:
    """Plot 1: Trotter error against the number of CNOT gates."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for order in (1, 2):
        d = result["data"][n][order]
        slope = np.polyfit(np.log(d["cnot"]), np.log(d["err_op"]), 1)[0]
        ax.loglog(d["cnot"], d["err_op"], marker=_MARKERS[order], ms=4, lw=1.5,
                  color="#d1495b" if order == 1 else "#00798c",
                  label=f"{_LABELS[order]}: error ~ CNOT^{slope:+.2f}")
    ax.set_xlabel("CNOT count")
    ax.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
    ax.set_title(f"Trotter error vs CNOT count, Ising chain n={n}, t={result['t']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    _finish(fig, path)


def plot_error_vs_gates(result: dict, path: str) -> None:
    """Plot 2: Trotter error against total elementary gate count."""
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for n in result["ns"]:
        for order in (1, 2):
            d = result["data"][n][order]
            ax.loglog(d["total"], d["err_op"], marker=_MARKERS[order], ms=3.6, lw=1.4,
                      color=_COLORS[n], ls="-" if order == 1 else "--",
                      label=f"n={n}, {_LABELS[order]}")
    ax.set_xlabel("total gate count (CNOT + RX + RZ)")
    ax.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
    ax.set_title(f"Trotter error vs total gates, t={result['t']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    _finish(fig, path)


def plot_error_vs_depth(result: dict, path: str) -> None:
    """Plot 3: Trotter error against logical circuit depth."""
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for n in result["ns"]:
        for order in (1, 2):
            d = result["data"][n][order]
            ax.loglog(d["depth"], d["err_op"], marker=_MARKERS[order], ms=3.6, lw=1.4,
                      color=_COLORS[n], ls="-" if order == 1 else "--",
                      label=f"n={n}, {_LABELS[order]}")
    ax.set_xlabel("logical circuit depth (ASAP schedule)")
    ax.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
    ax.set_title(f"Trotter error vs circuit depth, t={result['t']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    _finish(fig, path)


def plot_cnot_vs_qubits(result: dict, path: str) -> None:
    """Plot 4: CNOT count against chain length at fixed r."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ns = np.asarray(result["ns"], dtype=float)
    r_fixed = int(result["rs"][-1])
    for order in (1, 2):
        counts = [estimate_resources(n, r_fixed, result["t"], result["J"],
                                     result["h"], order).num_cnot
                  for n in result["ns"]]
        g = np.polyfit(np.log(ns), np.log(counts), 1)[0]
        ax.loglog(ns, counts, marker=_MARKERS[order], ms=5, lw=1.6,
                  color="#d1495b" if order == 1 else "#00798c",
                  label=f"{_LABELS[order]}: CNOT ~ n^{g:+.2f}")
    ax.set_xlabel("number of qubits  n")
    ax.set_ylabel("CNOT count")
    ax.set_title(f"CNOT count vs chain length at r={r_fixed}, t={result['t']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    _finish(fig, path)


def plot_gates_vs_steps(result: dict, path: str) -> None:
    """Plot 5: total gate count against r."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for n in result["ns"]:
        for order in (1, 2):
            d = result["data"][n][order]
            ax.loglog(result["rs"], d["total"], marker=_MARKERS[order], ms=3.6, lw=1.4,
                      color=_COLORS[n], ls="-" if order == 1 else "--",
                      label=f"n={n}, {_LABELS[order]}")
    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel("total gate count")
    ax.set_title("Gate count grows linearly with r")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    _finish(fig, path)


def plot_depth_vs_steps(result: dict, path: str) -> None:
    """Plot 6: circuit depth against r."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for n in result["ns"]:
        for order in (1, 2):
            d = result["data"][n][order]
            ax.loglog(result["rs"], d["depth"], marker=_MARKERS[order], ms=3.6, lw=1.4,
                      color=_COLORS[n], ls="-" if order == 1 else "--",
                      label=f"n={n}, {_LABELS[order]}")
    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel("logical circuit depth")
    ax.set_title("Circuit depth grows linearly with r")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    _finish(fig, path)


# ---------------------------------------------------------------------------
# 9. Command line
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t", type=float, default=1.0, help="evolution time")
    parser.add_argument("--J", type=float, default=1.0, help="Ising coupling (H = -J ZZ)")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field (H = -h X)")
    parser.add_argument("--rmax", type=int, default=200, help="largest r in the sweep")
    parser.add_argument("--npoints", type=int, default=26, help="log-spaced r values")
    parser.add_argument("--nlist", default="4,5,6", help="chain lengths, e.g. 4,5,6")
    parser.add_argument("--outdir", default="results", help="plot directory")
    parser.add_argument("--csv", default="results/circuit_resources.csv",
                        help="error + resource table")
    parser.add_argument("--json", default="results/circuit_reports.json",
                        help="structured resource reports")
    parser.add_argument("--no-plots", action="store_true", help="skip the six plots")
    parser.add_argument("--no-merge", action="store_true",
                        help="do not fuse the adjacent half steps of order 2")
    args = parser.parse_args()

    ns = tuple(int(x) for x in args.nlist.split(",") if x.strip())
    rs = np.unique(np.round(np.geomspace(1, args.rmax, args.npoints)).astype(int))
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("Gate decomposition verification (circuit vs matrix product formula)")
    print("=" * 78)
    ver = verify_gate_decomposition(n=4, r=3, t=args.t, J=args.J, h=args.h)
    for name, err in ver["checks"].items():
        print(f"  {name:<28} max |difference| = {err:.3e}")
    print(f"  -> {'PASS' if ver['ok'] else 'FAIL'} (tolerance 1e-10)")

    print()
    print("=" * 78)
    print("Analytical gate-count validation")
    print("=" * 78)
    all_ok = True
    for n, r in ((2, 1), (4, 5), (6, 10), (8, 50)):
        for order in (1, 2):
            v = validate_counts(n, r, args.t, args.J, args.h, order,
                                merge_half_steps=not args.no_merge)
            all_ok &= v["ok"]
            c, a = v["compiled"], v["analytic"]
            print(f"  n={n} r={r} order {order}: compiled CNOT={c['cnot']} RZ={c['rz']} "
                  f"RX={c['rx']} total={c['total']} | analytic CNOT={a['cnot']} "
                  f"RZ={a['rz']} RX={a['rx']} total={a['total']} "
                  f"-> {'match' if v['ok'] else 'MISMATCH ' + str(v['differences'])}")
    if not all_ok:
        raise SystemExit("analytical gate-count validation failed")

    print()
    print("=" * 78)
    print("Trotter error and circuit resources")
    print("=" * 78)
    result = study(ns=ns, t=args.t, rs=rs, J=args.J, h=args.h,
                   merge_half_steps=not args.no_merge)
    for n in ns:
        for order in (1, 2):
            q = result["data"][n][order]["resources"][-1]
            print(f"  n={n} order {order}: r={q.trotter_steps} -> "
                  f"CNOT {q.num_cnot}, RZ {q.num_rz}, RX {q.num_rx}, "
                  f"total {q.total_gates}, depth {q.circuit_depth}, "
                  f"2q-depth {q.two_qubit_depth}")

    write_csv(result["rows"], args.csv)
    print(f"\n  wrote {args.csv} ({len(result['rows'])} rows)")

    reports = [resource_report(n, r, args.t, args.J, args.h, order,
                               merge_half_steps=not args.no_merge)
               for n in (ns[0], ns[-1]) for r in (1, 5, 50, 200) for order in (1, 2)]
    write_json(reports, args.json)
    print(f"  wrote {args.json} ({len(reports)} reports)")

    if not args.no_plots:
        plots = {
            "plot_H_error_vs_cnot.png": lambda p: plot_error_vs_cnot(result, p),
            "plot_I_error_vs_gates.png": lambda p: plot_error_vs_gates(result, p),
            "plot_J_error_vs_depth.png": lambda p: plot_error_vs_depth(result, p),
            "plot_K_cnot_vs_qubits.png": lambda p: plot_cnot_vs_qubits(result, p),
            "plot_L_gates_vs_steps.png": lambda p: plot_gates_vs_steps(result, p),
            "plot_M_depth_vs_steps.png": lambda p: plot_depth_vs_steps(result, p),
        }
        print()
        for name, fn in plots.items():
            path = os.path.join(args.outdir, name)
            fn(path)
            print(f"  wrote {path}")

    print()
    print("Sample report (n={}, r=50, order 2):".format(ns[-1]))
    print(json.dumps(resource_report(ns[-1], 50, args.t, args.J, args.h, 2,
                                     merge_half_steps=not args.no_merge), indent=2))


if __name__ == "__main__":
    main()
