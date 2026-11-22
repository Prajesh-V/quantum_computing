"""Phase 4 Qiskit layer: independent circuit validation and compiled resources.

The custom simulator stays the reference implementation.  This package

* translates the **existing** Phase 1 gate list into a Qiskit circuit (it does
  not re-derive the Trotter decomposition),
* proves the two circuits are the same unitary up to global phase,
* transpiles them with a generic basis and with explicit connectivity maps,
* measures the compiled resources, and
* replays the logic circuits through Aer's density-matrix simulator to check the
  custom noise model, never to replace it.

Modules
-------
``circuit_builder``  gate list -> ``QuantumCircuit``, with the qubit mapping
``equivalence``      phase-insensitive unitary and statevector comparison
``transpilation``    generic and connectivity-aware configurations
``resources``        logical and compiled gate counts, depth, swap overhead
``noise_check``      Aer depolarising channels against the custom channels
``experiments``      the Phase 4 SDK study
``plotting``         compiled-resource figures
``report``           ``phase4_report.md``
``run_phase4``       driver: ``python -m quantum_sdk.run_phase4``
"""

__all__ = ["circuit_builder", "equivalence", "transpilation", "resources"]
