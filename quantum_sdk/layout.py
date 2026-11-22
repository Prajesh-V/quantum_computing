"""Restore the logical qubit order after transpilation.

When a circuit is routed onto a connectivity graph, the transpiler may finish
with the state in a *permuted* qubit order: the swaps plus the final layout
relabel qubits, and Qiskit accounts for that in the measurement bit mapping.  A
statevector or density-matrix simulator applies no measurement, so the caller
sees the permuted state and must undo the permutation itself.

Ignoring this makes a routed circuit look catastrophically worse than it is --
the fidelity is then computed against a differently ordered state, and the
"penalty" is mostly relabelling.  This module therefore

1. reads the permutation from ``circuit.layout`` when it is available,
2. otherwise searches the ``n!`` permutations (cheap for n <= 6),
3. and **verifies** the chosen permutation by checking that the *noiseless*
   routed state becomes the logical state to 1e-10.

The verified permutation is then applied to the noisy state before any fidelity
is computed.  Nothing is assumed: a permutation that does not pass the check
raises.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["permute_basis", "permutation_from_layout", "find_permutation",
           "restore_order"]


def permute_basis(rho, order, n: int) -> np.ndarray:
    """Relabel basis states so that output position ``k`` carries logical qubit ``order[k]``."""
    dim = 2 ** n
    indices = np.arange(dim)
    bits = np.array([[(index >> (n - 1 - position)) & 1 for position in range(n)]
                     for index in indices])
    new_bits = bits[:, list(order)]
    new_index = new_bits.dot(1 << np.arange(n - 1, -1, -1))
    return rho[np.ix_(new_index, new_index)]


def permutation_from_layout(circuit, n: int):
    """The final layout as a list, if Qiskit exposes a usable one."""
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return None
    for attribute in ("final_index_layout", "final_virtual_layout"):
        method = getattr(layout, attribute, None)
        if method is None:
            continue
        try:
            candidate = list(method())
        except Exception:                      # pragma: no cover - API variation
            continue
        if (len(candidate) == n and sorted(int(value) for value in candidate) == list(range(n))):
            return [int(value) for value in candidate]
    return None


def find_permutation(signal, reference, n: int, tolerance: float = 1e-10):
    """The permutation that turns ``signal`` into ``reference``, or ``None``."""
    for order in itertools.permutations(range(n)):
        candidate = permute_basis(signal, order, n)
        if float(np.max(np.abs(candidate - reference))) < tolerance:
            return list(order)
    return None


def restore_order(noisy_compiled, noiseless_compiled, reference, n: int) -> dict:
    """Find and apply the qubit permutation, verified on the noiseless state.

    ``reference`` is the logical noiseless density matrix.  The permutation is
    accepted only if it maps ``noiseless_compiled`` onto ``reference``.
    """
    order = permutation_from_layout(noiseless_compiled, n)
    verified = False
    if order is not None:
        candidate = permute_basis(noiseless_compiled, order, n)
        verified = float(np.max(np.abs(candidate - reference))) < 1e-10
    if not verified:
        order = find_permutation(noiseless_compiled, reference, n)
        verified = order is not None
    if not verified:
        raise RuntimeError(
            "could not identify the qubit permutation of the transpiled circuit; "
            "the noiseless routed state does not match the logical state under any "
            "relabelling")
    return {
        "permutation": order,
        "identity": order == list(range(n)),
        "residual_after_restore": float(np.max(
            np.abs(permute_basis(noiseless_compiled, order, n) - reference))),
        "rho": permute_basis(noisy_compiled, order, n),
    }
