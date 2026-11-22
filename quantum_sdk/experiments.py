"""The Phase 4 SDK study.

Three questions, kept separate:

1. **Is the SDK circuit the same circuit?**  Full unitary comparison up to a
   global phase, plus the output state for ``|00...0>``.
2. **What does compilation cost?**  Logical counts against transpiled counts,
   for a generic basis and for three explicit connectivities.
3. **Does the resource picture survive compilation and noise?**  Aer's
   density-matrix simulator, driven by the *same* channel as the custom engine
   (an explicit Pauli mixture, see :mod:`quantum_sdk.noise_check`), run on the
   logical and on the hardware-aware circuits.

The custom simulator is never replaced: it is the reference the SDK runs are
checked against.
"""

from __future__ import annotations

import numpy as np

from validation import independent as ind

from . import circuit_builder, equivalence, layout, noise_check, resources, transpilation

__all__ = ["NS", "RS", "ORDERS", "REFERENCE_NOISE", "equivalence_study",
           "pre_transpilation_study", "transpilation_study", "connectivity_study",
           "compiled_comparison_study", "noise_study"]

NS = (3, 4, 5)
RS = (1, 2, 3, 4, 6, 8)
ORDERS = (1, 2)
REFERENCE_NOISE = 1e-3


def _grid(ns=NS, rs=RS, orders=ORDERS):
    for n in ns:
        for order in orders:
            for r in rs:
                yield n, order, r


def equivalence_study(ns=NS, rs=RS, orders=ORDERS) -> list[dict]:
    """Custom unitary vs Qiskit unitary, up to a global phase."""
    rows = []
    for n, order, r in _grid(ns, rs, orders):
        unit = equivalence.compare_unitary(n, r, order)
        state = equivalence.compare_statevector(n, r, order)
        rows.append({
            "system_size": n,
            "trotter_order": order,
            "trotter_steps": r,
            "gates": unit["gates"],
            "spectral_distance": unit["spectral"],
            "frobenius_distance": unit["frobenius"],
            "max_entry_distance": unit["max_entry"],
            "global_phase_radians": unit["phase"],
            "statevector_infidelity": state["infidelity"],
            "passed": unit["spectral"] < 1e-10 and state["infidelity"] < 1e-12,
        })
    return rows


def pre_transpilation_study(ns=NS, rs=RS, orders=ORDERS) -> list[dict]:
    """Phase 1 counts against the Qiskit circuit before any transpilation."""
    rows = []
    for n, order, r in _grid(ns, rs, orders):
        circuit = circuit_builder.build(n, r, order)
        qiskit_counts = resources.circuit_resources(circuit)
        phase1 = resources.logical_resources(n, r, order)
        rows.append({
            "system_size": n,
            "trotter_order": order,
            "trotter_steps": r,
            "phase1_cnot": phase1["cx"],
            "qiskit_cx": qiskit_counts["cx"],
            "phase1_rx": phase1["rx"],
            "qiskit_rx": qiskit_counts["rx"],
            "phase1_rz": phase1["rz"],
            "qiskit_rz": qiskit_counts["rz"],
            "phase1_total": phase1["total_ops"],
            "qiskit_total": qiskit_counts["total_ops"],
            "phase1_depth": phase1["depth"],
            "qiskit_depth": qiskit_counts["depth"],
            "phase1_two_qubit_depth": phase1["two_qubit_depth"],
            "qiskit_two_qubit_depth": qiskit_counts["two_qubit_depth"],
            "counts_match": (phase1["cx"] == qiskit_counts["cx"]
                             and phase1["rx"] == qiskit_counts["rx"]
                             and phase1["rz"] == qiskit_counts["rz"]),
            "depth_match": phase1["depth"] == qiskit_counts["depth"],
        })
    return rows


def transpilation_study(ns=NS, rs=RS, orders=ORDERS) -> dict:
    """Logical plus every transpilation configuration, with resources."""
    rows = []
    metadata = []
    for n, order, r in _grid(ns, rs, orders):
        circuit = circuit_builder.build(n, r, order)
        logical = resources.logical_resources(n, r, order)
        for config in transpilation.configurations(n):
            outcome = transpilation.run_transpilation(circuit, config)
            compiled = resources.circuit_resources(outcome["circuit"])
            overhead = resources.swap_overhead(logical, compiled)
            rows.append({
                "system_size": n,
                "trotter_order": order,
                "trotter_steps": r,
                "configuration": config["name"],
                "topology": config["topology"],
                "optimization_level": config["optimization_level"],
                "logical_cx": logical["cx"],
                "compiled_cx": compiled["cx"],
                "compiled_rz": compiled["rz"],
                "compiled_rx": compiled["rx"],
                "compiled_total_ops": compiled["total_ops"],
                "logical_depth": logical["depth"],
                "compiled_depth": compiled["depth"],
                "logical_two_qubit_depth": logical["two_qubit_depth"],
                "compiled_two_qubit_depth": compiled["two_qubit_depth"],
                "extra_cx": overhead["extra_cx"],
                "swaps_estimated": overhead["swaps_estimated"],
                "two_qubit_inflation": overhead["two_qubit_inflation"],
                "depth_inflation": overhead["depth_inflation"],
                "transpilation_seconds": outcome["metadata"]["transpilation_seconds"],
            })
            entry = dict(outcome["metadata"])
            entry.update({"system_size": n, "trotter_order": order,
                          "trotter_steps": r})
            metadata.append(entry)
    return {"rows": rows, "metadata": metadata}


def connectivity_study(ns=NS, rs=(2, 4, 8), orders=ORDERS) -> list[dict]:
    """Side-by-side CX cost and depth under ideal / line / ring / star."""
    rows = []
    for n, order, r in _grid(ns, rs, orders):
        circuit = circuit_builder.build(n, r, order)
        logical = resources.logical_resources(n, r, order)
        entry = {"system_size": n, "trotter_order": order, "trotter_steps": r,
                 "logical_cx": logical["cx"], "logical_depth": logical["depth"]}
        for config in transpilation.configurations(n):
            compiled = resources.circuit_resources(
                transpilation.run_transpilation(circuit, config)["circuit"])
            topology = config["topology"]
            entry[f"{topology}_cx"] = compiled["cx"]
            entry[f"{topology}_depth"] = compiled["depth"]
            entry[f"{topology}_two_qubit_depth"] = compiled["two_qubit_depth"]
        star_overhead = resources.swap_overhead(
            logical, {"cx": entry["star_cx"], "depth": entry["star_depth"]})
        entry["star_extra_cx"] = star_overhead["extra_cx"]
        entry["star_swaps_estimated"] = star_overhead["swaps_estimated"]
        entry["star_cx_inflation"] = star_overhead["two_qubit_inflation"]
        entry["star_depth_inflation"] = star_overhead["depth_inflation"]
        entry["line_is_free"] = entry["line_cx"] == logical["cx"]
        rows.append(entry)
    return rows


def compiled_comparison_study(ns=NS, rs=(2, 4, 8), orders=ORDERS) -> list[dict]:
    """Logical, generic-compiled and hardware-aware counts in one row."""
    rows = []
    for n, order, r in _grid(ns, rs, orders):
        circuit = circuit_builder.build(n, r, order)
        logical = resources.logical_resources(n, r, order)
        entry = {"system_size": n, "trotter_order": order, "trotter_steps": r,
                 "logical_cx": logical["cx"], "logical_total_ops": logical["total_ops"],
                 "logical_depth": logical["depth"],
                 "logical_two_qubit_depth": logical["two_qubit_depth"]}
        for config in transpilation.configurations(n):
            compiled = resources.circuit_resources(
                transpilation.run_transpilation(circuit, config)["circuit"])
            tag = config["topology"]
            entry[f"{tag}_cx"] = compiled["cx"]
            entry[f"{tag}_total_ops"] = compiled["total_ops"]
            entry[f"{tag}_depth"] = compiled["depth"]
            entry[f"{tag}_two_qubit_depth"] = compiled["two_qubit_depth"]
        rows.append(entry)
    return rows


def noise_study(ns=NS, rs=(2, 4), orders=ORDERS,
                p: float = REFERENCE_NOISE) -> dict:
    """Aer noise on the logical and the compiled circuits, against the exact state.

    Two separate things are recorded:

    * ``sdk_simulation`` -- the custom noisy density matrix and Aer's, for the
      logical circuit at the same probabilities (a channel-level cross-check);
    * ``compiled_noise`` -- the fidelity of the logical, line-routed and
      star-routed circuits under the same noise, which is what decides whether
      the logical advantage survives compilation.
    """
    sdk_rows = []
    compiled_rows = []

    for n, order, r in _grid(ns, rs, orders):
        logical_circuit = circuit_builder.build(n, r, order)
        custom = ind.noisy_evolution(n, r, order, 1.0,
                                     ind.Spec("gate", p1=p, p2=p))
        agreement = noise_check.logical_circuit_agreement(n, r, order, p, p)
        sdk_rows.append({
            "system_size": n,
            "trotter_order": order,
            "trotter_steps": r,
            "single_qubit_error": p,
            "max_density_matrix_difference": agreement["max_difference"],
            "custom_fidelity": agreement and float(
                np.real(np.vdot(ind.exact_state(n), custom["rho"] @ ind.exact_state(n)))),
            "agrees": agreement["max_difference"] < 1e-10,
        })

        entry = {"system_size": n, "trotter_order": order, "trotter_steps": r,
                 "single_qubit_error": p, "two_qubit_error": p}
        reference = noise_check.aer_density_matrix(logical_circuit)
        psi_exact = ind.exact_state(n)
        model = noise_check.make_noise_model(p, p)
        for config in transpilation.configurations(n):
            outcome = transpilation.run_transpilation(logical_circuit, config)
            compiled = outcome["circuit"]
            tag = config["topology"]
            ideal = noise_check.aer_density_matrix(compiled)
            # A routed circuit can return its state in a permuted qubit order;
            # the permutation is identified and verified on the noiseless state
            # before any noisy fidelity is computed.
            restored = layout.restore_order(
                noise_check.aer_density_matrix(compiled, model), ideal, reference, n)
            fidelity = float(np.real(np.vdot(psi_exact, restored["rho"] @ psi_exact)))
            entry[f"{tag}_cx"] = resources.circuit_resources(compiled)["cx"]
            entry[f"{tag}_fidelity"] = float(min(1.0, max(0.0, fidelity)))
            entry[f"{tag}_purity"] = float(np.real(np.trace(restored["rho"] @ restored["rho"])))
            entry[f"{tag}_layout_permutation"] = "".join(str(q) for q in restored["permutation"])
            entry[f"{tag}_layout_identity"] = bool(restored["identity"])
            entry[f"{tag}_layout_residual"] = restored["residual_after_restore"]
            entry[f"{tag}_ideal_difference"] = float(np.max(np.abs(ideal - reference)))
        for tag in ("ideal", "line", "ring", "star"):
            if f"{tag}_fidelity" not in entry:
                continue
            entry[f"{tag}_total_error"] = 1.0 - entry[f"{tag}_fidelity"]
            entry[f"{tag}_penalty"] = (entry[f"{tag}_total_error"]
                                       - entry["ideal_total_error"])
        # kept for readability in the report and the older column names
        entry["logical_total_error"] = entry["ideal_total_error"]
        entry["line_penalty"] = entry["line_penalty"]
        entry["star_penalty"] = entry["star_penalty"]
        compiled_rows.append(entry)

    return {"sdk_simulation": sdk_rows, "compiled_noise": compiled_rows}
