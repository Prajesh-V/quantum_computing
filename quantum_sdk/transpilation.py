"""Transpilation configurations: a generic baseline and explicit connectivities.

Two things are deliberately kept apart:

* **Configuration A** -- a logical baseline.  The circuit is already written in
  ``rz``/``rx``/``cx``, so transpiling into that basis with optimiser level 0 and
  no coupling map should reproduce the logical circuit almost exactly.  It is the
  control for the hardware-aware runs.
* **Configuration B** -- hardware-aware.  The same circuit is routed onto an
  explicitly defined connectivity graph (line, ring, star) with optimiser level
  1.  These are *illustrative topologies*, not a real device: no calibration
  data, no error map and no backend is involved.

Every run records the versions, the basis, the coupling map, the seed, the
optimisation level and both layouts, so a compiled number can always be traced
back to the configuration that produced it.
"""

from __future__ import annotations

import re
import time

import qiskit
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from . import circuit_builder

__all__ = ["BASIS", "SEED", "topologies", "configurations", "run_transpilation",
           "describe_topology"]

BASIS = ["rz", "rx", "cx"]
SEED = 7

DESCRIPTIONS = {
    "line": "linear chain 0-1-2-...-(n-1): exactly the TFIM nearest-neighbour graph",
    "ring": "closed ring, the line plus an (n-1, 0) edge",
    "star": "star centred on qubit 0: the TFIM bonds are two hops apart",
}


def topologies(n: int) -> dict:
    """Explicit coupling maps for ``n`` qubits (``None`` = no connectivity limit)."""
    maps: dict[str, CouplingMap | None] = {"ideal": None}
    if n >= 2:
        maps["line"] = CouplingMap.from_line(n)
        maps["ring"] = CouplingMap.from_ring(n, bidirectional=True)
        star = CouplingMap([[0, qubit] for qubit in range(1, n)])
        # make_symmetric() mutates in place and returns None in Qiskit 2.x;
        # keep the object either way.
        star = star.make_symmetric() or star
        maps["star"] = star
    return maps


def _virtual_index(qubit) -> int:
    """The integer index of a circuit qubit, without relying on private API."""
    index = getattr(qubit, "_index", None)
    if isinstance(index, int):
        return index
    match = re.search(r"index=(\d+)", str(qubit))
    return int(match.group(1)) if match else -1


def describe_topology(coupling_map: CouplingMap | None) -> dict:
    if coupling_map is None:
        return {"name": "ideal", "edges": None, "description": "no connectivity constraint"}
    return {
        "edges": [list(edge) for edge in coupling_map.get_edges()],
        "size": coupling_map.size(),
        "distance_matrix_available": True,
    }


def configurations(n: int) -> list[dict]:
    """The transpilation configurations used for ``n`` qubits."""
    maps = topologies(n)
    configs = [{
        "name": "A_logical_generic",
        "description": "generic basis rz/rx/cx, no connectivity constraint, "
                       "optimiser level 0 (minimal rewrite)",
        "basis_gates": BASIS,
        "coupling_map": None,
        "topology": "ideal",
        "optimization_level": 0,
        "seed_transpiler": SEED,
    }]
    for topology in ("line", "ring", "star"):
        if topology not in maps or maps[topology] is None:
            continue
        configs.append({
            "name": f"B_{topology}",
            "description": f"routed onto the {topology} topology: {DESCRIPTIONS[topology]}",
            "basis_gates": BASIS,
            "coupling_map": maps[topology],
            "topology": topology,
            "optimization_level": 1,
            "seed_transpiler": SEED,
        })
    return configs


def run_transpilation(circuit, config: dict) -> dict:
    """Transpile one circuit and return it together with its metadata."""
    started = time.perf_counter()
    compiled = transpile(
        circuit,
        basis_gates=config["basis_gates"],
        coupling_map=config["coupling_map"],
        optimization_level=config["optimization_level"],
        seed_transpiler=config["seed_transpiler"],
    )
    elapsed = time.perf_counter() - started

    layout = getattr(compiled, "layout", None)
    initial = final = None
    if layout is not None:
        try:
            initial = {_virtual_index(virtual): int(physical) for virtual, physical
                       in layout.initial_layout.get_virtual_bits().items()}
        except Exception:                      # pragma: no cover - API variation
            initial = None
        for attribute in ("final_index_layout", "final_virtual_layout"):
            method = getattr(layout, attribute, None)
            if method is None:
                continue
            try:
                final = [int(index) for index in method()]
                break
            except Exception:                  # pragma: no cover - API variation
                continue

    metadata = {
        "qiskit_version": qiskit.__version__,
        "configuration": config["name"],
        "description": config["description"],
        "basis_gates": list(config["basis_gates"]),
        "topology": config["topology"],
        "topology_description": DESCRIPTIONS.get(config["topology"], "no constraint"),
        "coupling_map_edges": describe_topology(config["coupling_map"])["edges"],
        "optimization_level": config["optimization_level"],
        "seed_transpiler": config["seed_transpiler"],
        "initial_layout": initial,
        "final_layout": final,
        "transpilation_seconds": round(elapsed, 4),
        "num_qubits": compiled.num_qubits,
    }
    return {"circuit": compiled, "metadata": metadata}


def transpile_all(n: int, r: int, order: int, t: float = 1.0) -> list[dict]:
    """Logical circuit plus every configuration, for one (n, r, order)."""
    logical = circuit_builder.build(n, r, order, t)
    results = []
    for config in configurations(n):
        outcome = run_transpilation(logical, config)
        outcome["configuration"] = config["name"]
        results.append(outcome)
    return results
