"""Advanced routing and layout study: Sabre vs Basic routing on mismatched topologies.

Investigates whether intelligent layout (Sabre layout) and routing (SabreSwap with
lookahead and decay heuristics) can recover the second-order Trotter circuit depth
advantage on non-line topologies (star, ring, heavy-hex/falcon).

Phase 4 demonstrated that while second-order Trotter retains its CNOT gate advantage
on star topologies, its depth advantage was degraded due to bottleneck routing
through the central hub. This module evaluates if Sabre routing recovers that advantage.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import qiskit
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from . import circuit_builder, resources, transpilation
from .transpilation import BASIS, SEED, topologies

__all__ = [
    "ROUTING_METHODS",
    "run_routing_benchmark",
    "compare_routing_strategies",
    "evaluate_second_order_recovery",
]

ROUTING_METHODS = [
    {
        "name": "basic_opt1",
        "description": "Trivial layout + Basic swap router, Opt Level 1 (Phase 4 baseline)",
        "layout_method": "trivial",
        "routing_method": "basic",
        "optimization_level": 1,
    },
    {
        "name": "sabre_opt1",
        "description": "Sabre layout + Sabre swap router, Opt Level 1",
        "layout_method": "sabre",
        "routing_method": "sabre",
        "optimization_level": 1,
    },
    {
        "name": "sabre_opt3",
        "description": "Sabre layout + Sabre swap router + Commutation Analysis, Opt Level 3",
        "layout_method": "sabre",
        "routing_method": "sabre",
        "optimization_level": 3,
    },
]


def run_routing_benchmark(n: int, r: int, order: int, topology_name: str,
                          method_config: Dict[str, Any], seed: int = SEED) -> Dict[str, Any]:
    """Transpile a Trotter circuit using a specific layout/routing method."""
    qc = circuit_builder.build(n, r, order)
    maps = topologies(n)
    cmap = maps.get(topology_name)

    t0 = time.perf_counter()
    transpiled = transpile(
        qc,
        basis_gates=list(BASIS),
        coupling_map=cmap,
        layout_method=method_config.get("layout_method"),
        routing_method=method_config.get("routing_method"),
        optimization_level=method_config.get("optimization_level", 1),
        seed_transpiler=seed,
    )
    elapsed = time.perf_counter() - t0

    op_counts = transpiled.count_ops()
    cx_count = int(op_counts.get("cx", 0))
    swap_count = int(op_counts.get("swap", 0))

    # In our basis ("rz", "rx", "cx"), swaps are decomposed into 3 CNOTs
    # Total effective 2Q operations is cx_count
    total_depth = int(transpiled.depth())
    two_qubit_depth = int(transpiled.depth(lambda instruction: len(instruction.qubits) > 1))

    return {
        "system_size": n,
        "trotter_steps": r,
        "trotter_order": order,
        "topology": topology_name,
        "method": method_config["name"],
        "cnot_count": cx_count,
        "total_gates": int(sum(op_counts.values())),
        "total_depth": total_depth,
        "two_qubit_depth": two_qubit_depth,
        "transpilation_time": elapsed,
    }


def compare_routing_strategies(ns: Sequence[int] = (4, 5),
                               rs: Sequence[int] = (1, 2, 4, 8),
                               topologies_to_test: Sequence[str] = ("line", "star"),
                               methods: Sequence[Dict[str, Any]] = ROUTING_METHODS) -> List[Dict[str, Any]]:
    """Benchmark routing methods across system sizes, step counts, and orders."""
    results = []
    for n in ns:
        for topology in topologies_to_test:
            for r in rs:
                for order in (1, 2):
                    for method in methods:
                        res = run_routing_benchmark(n, r, order, topology, method)
                        results.append(res)
    return results


def evaluate_second_order_recovery(benchmark_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze whether Sabre routing recovers the 2nd-order depth advantage on the star topology."""
    # Group by (n, r, topology, method)
    groups: Dict[Tuple[int, int, str, str], Dict[int, Dict[str, Any]]] = {}
    for row in benchmark_rows:
        key = (row["system_size"], row["trotter_steps"], row["topology"], row["method"])
        if key not in groups:
            groups[key] = {}
        groups[key][row["trotter_order"]] = row

    comparison_rows = []
    for (n, r, topo, method), order_dict in groups.items():
        if 1 in order_dict and 2 in order_dict:
            r1 = order_dict[1]
            r2 = order_dict[2]

            depth_ratio = r2["total_depth"] / max(1, r1["total_depth"])
            cnot_ratio = r2["cnot_count"] / max(1, r1["cnot_count"])
            two_q_depth_ratio = r2["two_qubit_depth"] / max(1, r1["two_qubit_depth"])

            comparison_rows.append({
                "system_size": n,
                "trotter_steps": r,
                "topology": topo,
                "method": method,
                "order1_cnot": r1["cnot_count"],
                "order2_cnot": r2["cnot_count"],
                "cnot_ratio_o2_o1": cnot_ratio,
                "order1_depth": r1["total_depth"],
                "order2_depth": r2["total_depth"],
                "depth_ratio_o2_o1": depth_ratio,
                "order1_2q_depth": r1["two_qubit_depth"],
                "order2_2q_depth": r2["two_qubit_depth"],
                "two_q_depth_ratio_o2_o1": two_q_depth_ratio,
                "recovered_depth_advantage": depth_ratio < 1.0,
            })
    return comparison_rows
