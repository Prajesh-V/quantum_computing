"""Phase 3: density-matrix simulation and gate-level depolarising noise.

Modules
-------
``density_matrix``  representation, local-operator algebra, diagnostics
``channels``        global / single-qubit / two-qubit depolarising channels
``metrics``         fidelity, infidelity, trace distance
``noisy_simulator`` chronological noisy execution of the Phase 1 gate circuit
``experiments``     the Phase 3 measurement sweeps and the optimal-r search
``plotting``        figures
``report``          ``phase3_report.md`` and ``phase3_summary.json``
``run_phase3``      command-line driver
"""
