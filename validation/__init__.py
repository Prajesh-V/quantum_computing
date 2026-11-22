"""Phase 4: independent scientific audit and robustness validation.

This package deliberately does **not** import the ``noise`` package for its
physics.  The circuit gate list comes from ``circuit_cost.build_circuit`` (the
same decomposition Phases 1-3 used), but the density-matrix algebra, the three
depolarising channels and the error metrics are re-implemented from scratch in
:mod:`validation.independent`, so a mistake in ``noise/`` cannot hide behind a
shared helper.

Modules
-------
``independent``   from-scratch density matrices, channels, metrics, evolution
``audit``         mathematical verification of Hamiltonian, formulas, channels
``reproduction``  re-run selected Phase 3 rows independently and compare
``optimal_check`` recompute every stored optimal step count
``robustness``    initial state, evolution time, noise strength, grid density
``comparisons``   equal-step, equal-CNOT and equal-total-gate audits
``plotting``      robustness figures
``report``        the generated audit report
``run_audit``     driver: ``python -m validation.run_audit``
"""

__all__ = ["independent"]
