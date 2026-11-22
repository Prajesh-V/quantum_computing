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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qiskit import transpile

from . import circuit_builder, resources
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
    if topology_name not in maps:
        raise ValueError(f"Unknown topology {topology_name!r}; choose from {list(maps.keys())}")
    cmap = maps[topology_name]

    logical_res = resources.logical_resources(n, r, order)

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

    # In our basis ("rz", "rx", "cx"), swaps are decomposed into 3 CNOTs
    routing_cnot_overhead = max(0, cx_count - logical_res["cx"])
    swaps_inserted = routing_cnot_overhead // 3

    total_depth = int(transpiled.depth())
    two_qubit_depth = int(transpiled.depth(lambda instruction: len(instruction.qubits) > 1))
    depth_inflation = total_depth / max(1, logical_res["depth"])

    return {
        "system_size": n,
        "trotter_steps": r,
        "trotter_order": order,
        "topology": topology_name,
        "method": method_config["name"],
        "cnot_count": cx_count,
        "logical_cnot": logical_res["cx"],
        "routing_cnot_overhead": routing_cnot_overhead,
        "swaps_inserted": swaps_inserted,
        "total_gates": int(sum(op_counts.values())),
        "total_depth": total_depth,
        "logical_depth": logical_res["depth"],
        "depth_inflation": depth_inflation,
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
    # Group by (n, r, topology) -> method -> order -> row
    nested: Dict[Tuple[int, int, str], Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for row in benchmark_rows:
        key = (row["system_size"], row["trotter_steps"], row["topology"])
        method = row["method"]
        order = row["trotter_order"]
        if key not in nested:
            nested[key] = {}
        if method not in nested[key]:
            nested[key][method] = {}
        nested[key][method][order] = row

    comparison_rows = []
    for (n, r, topo), method_dict in nested.items():
        basic_o2_depth = None
        if "basic_opt1" in method_dict and 2 in method_dict["basic_opt1"]:
            basic_o2_depth = method_dict["basic_opt1"][2]["total_depth"]

        for method, order_dict in method_dict.items():
            if 1 in order_dict and 2 in order_dict:
                r1 = order_dict[1]
                r2 = order_dict[2]

                depth_ratio = r2["total_depth"] / max(1, r1["total_depth"])
                cnot_ratio = r2["cnot_count"] / max(1, r1["cnot_count"])
                two_q_depth_ratio = r2["two_qubit_depth"] / max(1, r1["two_qubit_depth"])

                # Depth reduction of Order 2 compared to basic router
                depth_reduction_pct = 0.0
                if basic_o2_depth and basic_o2_depth > 0:
                    depth_reduction_pct = max(0.0, 100.0 * (basic_o2_depth - r2["total_depth"]) / basic_o2_depth)

                recovered = (depth_reduction_pct > 0.0) or (r2["total_depth"] <= r1["total_depth"])

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
                    "sabre_depth_reduction_pct": depth_reduction_pct,
                    "recovered_depth_advantage": recovered,
                })
    return comparison_rows
