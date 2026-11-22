"""Reconcile Aer's depolarising channels with the custom ones.

Different depolarising conventions do **not** agree at the same numerical
parameter, so nothing here assumes a correspondence.  Instead the channels are
measured on small states and the measured relationship is reported:

* one gate, one qubit: Aer's ``depolarizing_error(p, 1)`` against
  ``(1-p) rho + (p/3) sum_{X,Y,Z} P rho P``;
* one CNOT: ``depolarizing_error(p, 2)`` against the 15-term two-qubit sum;
* the whole Trotter circuit: Aer's density matrix against
  ``validation.independent.noisy_evolution`` for the same probabilities.

Only after the single-gate agreement is established is it meaningful to compare
whole circuits, which is what :mod:`quantum_sdk.experiments` does.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

from validation import independent as ind

from . import circuit_builder

__all__ = ["aer_density_matrix", "single_qubit_agreement", "two_qubit_agreement",
           "channel_convention", "logical_circuit_agreement", "make_noise_model",
           "aer_fidelity"]


def aer_density_matrix(circuit, noise_model=None) -> np.ndarray:
    """Exact density matrix from Aer's ``density_matrix`` method."""
    prepared = circuit.copy()
    prepared.save_density_matrix()
    simulator = AerSimulator(method="density_matrix")
    result = simulator.run(prepared, noise_model=noise_model, shots=1).result()
    return np.asarray(result.data(0)["density_matrix"], dtype=complex)


def pauli_mixture(k: int, p: float):
    """The project's channel as an explicit Pauli error: p spread over 4**k - 1 terms.

    This is ``(1-p) rho + p/(4**k-1) sum_{P != I} P rho P`` written in the form
    Aer accepts, so building the model this way makes the two implementations
    the *same* channel -- unlike ``depolarizing_error``, which uses the
    full-depolarisation convention and only agrees on product states.
    """
    labels = ["X", "Y", "Z"] if k == 1 else \
        [first + second for first in "IXYZ" for second in "IXYZ"
         if first + second != "II"]
    weight = p / (4 ** k - 1)
    return pauli_error([(label, weight) for label in labels] + [("I" * k, 1.0 - p)])


def make_noise_model(p1: float, p2: float, gates=("rx", "rz", "cx")) -> NoiseModel:
    """Phase 3's channel, gate by gate, expressed as an Aer noise model."""
    model = NoiseModel()
    single = [gate for gate in gates if gate != "cx"]
    if p1 > 0 and single:
        model.add_all_qubit_quantum_error(pauli_mixture(1, p1), single)
    if p2 > 0:
        model.add_all_qubit_quantum_error(pauli_mixture(2, p2), ["cx"])
    return model


def single_qubit_agreement(p: float, angle: float = 0.37) -> dict:
    circuit = QuantumCircuit(1)
    circuit.rx(angle, 0)
    model = NoiseModel()
    model.add_all_qubit_quantum_error(pauli_mixture(1, p), ["rx"])
    got = aer_density_matrix(circuit, model)

    rho = np.outer(ind._rx(angle) @ np.array([1, 0], dtype=complex),
                   (ind._rx(angle) @ np.array([1, 0], dtype=complex)).conj())
    expected = ind.depolarize_explicit(rho, (0,), p)
    return {
        "qubits": 1, "p": p,
        "max_difference": float(np.max(np.abs(got - expected))),
        "aer": got, "custom": expected,
    }


def two_qubit_agreement(p: float) -> dict:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    model = NoiseModel()
    model.add_all_qubit_quantum_error(pauli_mixture(2, p), ["cx"])
    got = aer_density_matrix(circuit, model)

    zero = np.zeros(4, dtype=complex)
    zero[0] = 1.0
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)
    psi = cnot @ zero
    rho = np.outer(psi, psi.conj())
    expected = ind.depolarize_explicit(rho, (0, 1), p)
    return {
        "qubits": 2, "p": p,
        "max_difference": float(np.max(np.abs(got - expected))),
        "aer": got, "custom": expected,
    }


def _native_agreement(k: int, p: float) -> dict:
    """``depolarizing_error(p, k)`` against the project's channel."""
    if k == 1:
        circuit = QuantumCircuit(1)
        circuit.rx(0.37, 0)
        gate = "rx"
        psi = ind._rx(0.37) @ np.array([1, 0], dtype=complex)
    else:
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        gate = "cx"
        zero = np.zeros(4, dtype=complex)
        zero[0] = 1.0
        psi = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                       dtype=complex) @ zero
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p, k), [gate])
    got = aer_density_matrix(circuit, model)
    rho = np.outer(psi, psi.conj())
    expected = ind.depolarize_explicit(rho, tuple(range(k)), p)
    return {"aer": got, "custom": expected,
            "max_difference": float(np.max(np.abs(got - expected)))}


def channel_convention(levels=(1e-3, 0.05, 0.2)) -> dict:
    """Measure how Aer's channels relate to the project's, instead of assuming.

    The default ``depolarizing_error(p, k)`` is the *full depolarisation* channel
    ``(1-p) rho + p I/d``; the project uses the Pauli mixture ``p/3`` (one qubit)
    and ``p/15`` (two qubits).  On a pure product state the two are related by

        p_aer = (4/3) p_project   (one qubit)
        p_aer = (16/15) p_project (two qubits)

    and those factors are measured here, not assumed.  A whole-circuit comparison
    uses :func:`make_noise_model`, which rebuilds the project's channel exactly as
    an explicit Pauli mixture, so no conversion is involved there.
    """
    rows = []
    for p in levels:
        single = _native_agreement(1, p)
        double = _native_agreement(2, p)
        rows.append({
            "p": p,
            "single_qubit_max_difference": single["max_difference"],
            "two_qubit_max_difference": double["max_difference"],
            "single_qubit_ratio_to_p": single["max_difference"] / p,
            "two_qubit_ratio_to_p": double["max_difference"] / p,
        })
    exact = []
    for p in levels:
        single = single_qubit_agreement(p)
        double = two_qubit_agreement(p)
        exact.append({"p": p,
                      "single_qubit_max_difference": single["max_difference"],
                      "two_qubit_max_difference": double["max_difference"]})
    return {
        "native_convention_rows": rows,
        "native_max_difference": max(max(row["single_qubit_max_difference"],
                                         row["two_qubit_max_difference"])
                                     for row in rows),
        "pauli_mixture_rows": exact,
        "pauli_mixture_max_difference": max(max(row["single_qubit_max_difference"],
                                                row["two_qubit_max_difference"])
                                            for row in exact),
        "pauli_mixture_is_identical": all(
            max(row["single_qubit_max_difference"], row["two_qubit_max_difference"]) < 1e-12
            for row in exact),
        "native_conversion": {"one_qubit": "p_aer = 4 p / 3",
                              "two_qubit": "p_aer = 16 p / 15"},
        "note": ("depolarizing_error() is the full-depolarisation channel and is NOT "
                 "the project's Pauli mixture; make_noise_model() rebuilds the "
                 "project's channel exactly as a Pauli mixture, and that is what "
                 "every whole-circuit comparison uses"),
    }


def logical_circuit_agreement(n: int, r: int, order: int, p1: float, p2: float,
                              t: float = 1.0) -> dict:
    """Aer with noise vs the custom noisy density matrix, same probabilities."""
    circuit = circuit_builder.build(n, r, order, t)
    model = make_noise_model(p1, p2)
    got = aer_density_matrix(circuit, model)
    custom = ind.noisy_evolution(n, r, order, t, ind.Spec("gate", p1=p1, p2=p2))
    return {
        "system_size": n, "trotter_steps": r, "trotter_order": order,
        "single_qubit_error": p1, "two_qubit_error": p2,
        "max_difference": float(np.max(np.abs(got - custom["rho"]))),
    }


def aer_fidelity(n: int, r: int, order: int, p1: float, p2: float,
                 t: float = 1.0, circuit=None) -> dict:
    """Fidelity of an Aer noisy run against the exact state ``exp(-iHt)|0...0>``.

    ``circuit`` lets the caller pass a *transpiled* circuit, which is how the
    compiled noise exposure is measured.
    """
    if circuit is None:
        circuit = circuit_builder.build(n, r, order, t)
    model = make_noise_model(p1, p2)
    rho = aer_density_matrix(circuit, model)
    psi_exact = ind.exact_state(n, t)
    fidelity = float(np.real(np.vdot(psi_exact, rho @ psi_exact)))
    return {
        "fidelity": float(min(1.0, max(0.0, fidelity))),
        "total_error": float(1.0 - min(1.0, max(0.0, fidelity))),
        "purity": float(np.real(np.trace(rho @ rho))),
    }
