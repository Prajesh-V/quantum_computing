"""Density-matrix representation and local-operator algebra.

Phase 3 of the project: the state-vector simulator in ``trotter.py`` is replaced
by a density matrix so that non-unitary (noisy) evolution can be represented.

Conventions inherited from the rest of the repository
-----------------------------------------------------
* ``n`` qubits, Hilbert-space dimension ``d = 2**n``.
* Qubit ``0`` is the most significant bit of the basis index (``trotter.kron_all``).
* ``rho`` is a dense ``(d, d)`` complex array laid out as ``rho[row, col]`` in the
  computational basis, i.e. ``rho[i, j] = <i| rho |j>``.

Internally a local operator is applied with tensor reshapes and permutations of
the ``2n``-axis tensor ``(row_0, ..., row_{n-1}, col_0, ..., col_{n-1})``.  That
costs ``O(4**k * d)`` work for a ``k``-qubit operator instead of the ``O(d**3)``
of building full ``d x d`` matrices, which is what makes gate-level noise
tractable for ``n = 4, 5, 6``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "num_qubits",
    "pure_density_matrix",
    "basis_density_matrix",
    "trace",
    "purity",
    "hermiticity_error",
    "min_eigenvalue",
    "is_positive_semidefinite",
    "check_density_matrix",
    "apply_operator",
    "expectation",
    "magnetization_x",
    "zz_correlation",
    "partial_trace",
]


# ---------------------------------------------------------------------------
# Basic validation helpers
# ---------------------------------------------------------------------------

def num_qubits(dim: int) -> int:
    """``n`` such that ``dim == 2**n``, with a clear error otherwise."""
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"dimension must be positive, got {dim}")
    n = int(round(math.log2(dim)))
    if 2 ** n != dim:
        raise ValueError(f"dimension {dim} is not a power of two")
    return n


def _as_square(rho, name: str = "rho") -> np.ndarray:
    array = np.asarray(rho, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")
    num_qubits(array.shape[0])          # validates the dimension
    return array


def _check_qubits(qubits, n: int, k: int | None = None) -> tuple[int, ...]:
    qs = tuple(int(q) for q in qubits)
    if len(set(qs)) != len(qs):
        raise ValueError(f"qubit indices must be distinct, got {qs}")
    if any(q < 0 or q >= n for q in qs):
        raise ValueError(f"qubit indices {qs} out of range for n={n}")
    if k is not None and len(qs) != k:
        raise ValueError(f"operator needs {k} qubit(s), got {len(qs)}")
    return qs


def _axes(n: int, qubits: tuple[int, ...]):
    """Permutation putting ``qubits`` first on the row and on the column block."""
    rest = tuple(q for q in range(n) if q not in qubits)
    perm = list(qubits) + list(rest) + [n + q for q in qubits] + [n + o for o in rest]
    return perm, rest


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def pure_density_matrix(psi) -> np.ndarray:
    """``rho = |psi><psi|`` for a normalised state vector."""
    vector = np.asarray(psi, dtype=complex).reshape(-1)
    num_qubits(vector.size)
    norm = np.linalg.norm(vector)
    if abs(norm - 1.0) > 1e-8:
        raise ValueError(f"state vector must be normalised, |psi| = {norm:.12f}")
    return np.outer(vector, vector.conj())


def basis_density_matrix(n: int, bits) -> np.ndarray:
    """Pure density matrix of the computational basis state ``|b_0 ... b_{n-1}>``."""
    bits = [int(b) for b in bits]
    if len(bits) != n:
        raise ValueError(f"expected {n} bits, got {len(bits)}")
    if any(b not in (0, 1) for b in bits):
        raise ValueError(f"bits must be 0 or 1, got {bits}")
    index = 0
    for b in bits:
        index = (index << 1) | b
    psi = np.zeros(2 ** n, dtype=complex)
    psi[index] = 1.0
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def trace(rho) -> complex:
    """``Tr(rho)``."""
    return complex(np.trace(_as_square(rho)))


def purity(rho) -> float:
    """``Tr(rho**2)``; equals 1 for a pure state and ``1/d`` for the maximally mixed one."""
    array = _as_square(rho)
    return float(np.real(np.trace(array @ array)))


def hermiticity_error(rho) -> float:
    """Spectral norm of ``rho - rho^dagger``."""
    array = _as_square(rho)
    return float(np.linalg.norm(array - array.conj().T, 2))


def min_eigenvalue(rho) -> float:
    """Smallest eigenvalue of the Hermitian part (used for the PSD check)."""
    array = _as_square(rho)
    herm = 0.5 * (array + array.conj().T)
    return float(np.linalg.eigvalsh(herm).min())


def is_positive_semidefinite(rho, tol: float = 1e-10) -> bool:
    """True when no eigenvalue is negative beyond ``tol``."""
    return min_eigenvalue(rho) >= -abs(tol)


def check_density_matrix(rho, tol: float = 1e-10) -> dict:
    """Report the three defining properties of a density matrix.

    Returns a dict with the measured ``trace``, ``purity``, ``hermiticity_error``
    and ``min_eigenvalue`` plus an overall ``ok`` flag.  Small floating-point
    violations are tolerated, larger ones are reported as failures.
    """
    array = _as_square(rho)
    tr = trace(array)
    herm = hermiticity_error(array)
    lam_min = min_eigenvalue(array)
    return {
        "dimension": int(array.shape[0]),
        "num_qubits": num_qubits(array.shape[0]),
        "trace": tr,
        "trace_error": abs(tr - 1.0),
        "purity": purity(array),
        "hermiticity_error": herm,
        "min_eigenvalue": lam_min,
        "ok": bool(abs(tr - 1.0) <= tol and herm <= tol and lam_min >= -abs(tol)),
        "tolerance": tol,
    }


# ---------------------------------------------------------------------------
# Local operators
# ---------------------------------------------------------------------------

def apply_operator(rho, op, qubits) -> np.ndarray:
    """ ``op @ rho @ op^dagger`` with ``op`` acting on ``qubits``.

    ``qubits`` lists the qubit indices that the axes of ``op`` refer to, in the
    same order: for a two-qubit ``op`` the first axis belongs to ``qubits[0]``.
    Qubits outside the list are untouched.
    """
    array = _as_square(rho)
    d = array.shape[0]
    n = num_qubits(d)
    qs = _check_qubits(qubits, n)
    k = len(qs)
    dk = 2 ** k
    matrix = np.asarray(op, dtype=complex)
    if matrix.shape != (dk, dk):
        raise ValueError(
            f"operator shape {matrix.shape} does not match {k} qubit(s), expected {(dk, dk)}"
        )

    perm, rest = _axes(n, qs)
    drest = 2 ** len(rest)
    tensor = array.reshape((2,) * (2 * n)).transpose(perm).reshape(dk, drest, dk, drest)
    # out[r, i, c, j] = sum_{a,b} op[r, a] rho[a, i, b, j] op*[c, b]
    tensor = np.tensordot(matrix, tensor, axes=([1], [0]))       # (r, i, b, j)
    tensor = np.tensordot(tensor, matrix.conj(), axes=([2], [1]))  # (r, i, j, c)
    tensor = tensor.transpose(0, 1, 3, 2)                         # (r, i, c, j)
    tensor = tensor.reshape((2,) * (2 * n)).transpose(np.argsort(perm))
    return tensor.reshape(d, d)


def expectation(rho, op, qubits) -> float:
    """``Tr(rho * op_embedded)`` for a Hermitian ``op`` acting on ``qubits``."""
    array = _as_square(rho)
    d = array.shape[0]
    n = num_qubits(d)
    qs = _check_qubits(qubits, n)
    k = len(qs)
    dk = 2 ** k
    matrix = np.asarray(op, dtype=complex)
    if matrix.shape != (dk, dk):
        raise ValueError(f"operator shape {matrix.shape} does not match {k} qubit(s)")

    perm, rest = _axes(n, qs)
    drest = 2 ** len(rest)
    tensor = array.reshape((2,) * (2 * n)).transpose(perm).reshape(dk, drest, dk, drest)
    # Tr(rho * O) = sum_{i,j} rho[i,j] O[j,i]; row index i = (a, i_rest),
    # column index j = (b, j_rest), so j == i forces b == a and j_rest == i_rest.
    # axes are (a, i, b, j): the trace forces the column label b == row qubit
    # index a, and one output label b is contracted with O[b, a].
    value = np.einsum("aibi,ba->", tensor, matrix)
    return float(np.real(value))


def magnetization_x(rho, n: int) -> float:
    """``<M_x> = (1/n) sum_i Tr(rho X_i)``."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return sum(expectation(rho, x, (q,)) for q in range(n)) / n


ZZ = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]], dtype=complex)


def zz_correlation(rho, n: int) -> float:
    """``<C_zz> = 1/(n-1) sum_i Tr(rho Z_i Z_{i+1})`` over the open chain."""
    if n < 2:
        raise ValueError(f"need at least two qubits, got {n}")
    total = sum(expectation(rho, ZZ, (i, i + 1)) for i in range(n - 1))
    return total / (n - 1)


def partial_trace(rho, qubits, n: int | None = None) -> np.ndarray:
    """Trace out ``qubits`` and return the reduced density matrix.

    Used by the depolarising channels and by the tests that check them.
    """
    array = _as_square(rho)
    d = array.shape[0]
    n = num_qubits(d) if n is None else int(n)
    qs = _check_qubits(qubits, n)
    keep = tuple(q for q in range(n) if q not in qs)
    perm, _ = _axes(n, qs)
    dk, drest = 2 ** len(qs), 2 ** len(keep)
    tensor = array.reshape((2,) * (2 * n)).transpose(perm).reshape(dk, drest, dk, drest)
    # sum over the traced-out qubit label: reduced[i, j] = sum_a rho[(a,i),(a,j)]
    return np.einsum("aiaj->ij", tensor).reshape(drest, drest)
