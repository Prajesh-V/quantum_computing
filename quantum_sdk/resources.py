"""Resource extraction before and after transpilation.

Logical counts (Phase 1) and compiled counts are reported in separate columns
and never added together: they describe different gate sets.  A compiled circuit
in ``rz``/``rx``/``cx`` with routing swaps is compared with the logical circuit
by counting two-qubit operations, because a routed ``cx`` on non-adjacent qubits
becomes three ``cx`` plus single-qubit fixups.
"""

from __future__ import annotations

import circuit_cost

__all__ = ["circuit_resources", "logical_resources", "compiled_comparison",
           "swap_overhead"]


def circuit_resources(circuit) -> dict:
    """Gate counts, depth and width of a Qiskit circuit."""
    ops = {str(key): int(value) for key, value in circuit.count_ops().items()}
    two_qubit_ops = sum(1 for instruction in circuit.data
                        if len(instruction.qubits) == 2)
    return {
        "num_qubits": int(circuit.num_qubits),
        "rz": ops.get("rz", 0),
        "rx": ops.get("rx", 0),
        "cx": ops.get("cx", 0),
        "swap": ops.get("swap", 0),
        "other": sum(value for key, value in ops.items()
                     if key not in ("rz", "rx", "cx", "swap")),
        "total_ops": int(sum(ops.values())),
        "single_qubit_ops": ops.get("rz", 0) + ops.get("rx", 0),
        "two_qubit_ops": int(two_qubit_ops),
        "depth": int(circuit.depth()),
        "two_qubit_depth": int(circuit.depth(
            lambda instruction: len(instruction.qubits) > 1)),
        "histogram": ops,
    }


def logical_resources(n: int, r: int, order: int, t: float = 1.0) -> dict:
    """Phase 1's own accounting, for the same configuration."""
    resources = circuit_cost.estimate_resources(n, r, t, 1.0, 1.0, order)
    return {
        "num_qubits": resources.num_qubits,
        "rz": resources.num_rz,
        "rx": resources.num_rx,
        "cx": resources.num_cnot,
        "swap": 0,
        "total_ops": resources.total_gates,
        "single_qubit_ops": resources.num_rx + resources.num_rz,
        "two_qubit_ops": resources.num_cnot,
        "depth": resources.circuit_depth,
        "two_qubit_depth": resources.two_qubit_depth,
    }


def swap_overhead(logical: dict, compiled: dict) -> dict:
    """How many extra two-qubit operations routing introduced."""
    extra = compiled["cx"] - logical["cx"]
    return {
        "extra_cx": int(extra),
        # with no swap in the basis, each routed swap decomposes into 3 CX
        "swaps_estimated": int(max(0, extra) // 3),
        "symbolic_swaps": int(compiled.get("swap", 0)),
        "two_qubit_inflation": (compiled["cx"] / logical["cx"]
                                if logical["cx"] else None),
        "depth_inflation": (compiled["depth"] / logical["depth"]
                            if logical["depth"] else None),
    }


def compiled_comparison(logical: dict, compiled: dict) -> dict:
    """Side-by-side counts, with the differences made explicit."""
    out = {}
    for key in ("rz", "rx", "cx", "total_ops", "depth", "two_qubit_depth"):
        out[f"logical_{key}"] = logical.get(key, 0)
        out[f"compiled_{key}"] = compiled.get(key, 0)
        out[f"delta_{key}"] = compiled.get(key, 0) - logical.get(key, 0)
    out.update(swap_overhead(logical, compiled))
    return out
