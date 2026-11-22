# PROGRESS — engineering handover

The concise, public-facing overview is `README.md`. This file is the detailed engineering log: what
each phase implemented, how it was validated, which configuration produced which artifact, what is
still open, and exactly what to do next.

Last updated with Phase 4.

---

## 1. Status at a glance

| Phase | Scope | Commit | Status |
|---|---|---|---|
| 1 | Gate-level circuit resource estimation | `abd5c54` | complete, validated |
| 2 | Trotter accuracy versus circuit resources | `9905a7f` | complete, validated |
| 3 | Density-matrix simulation and noise models | `9e47be8` | complete, audited in Phase 4 |
| 4 | Independent audit, robustness, Qiskit validation, transpilation | the commit that added this file | complete - **PASS WITH QUALIFICATIONS** |

Test suite: **201 tests, all passing** (`python -m pytest tests/ -q`, ~20 s).

| File | Tests | Covers |
|---|---|---|
| `tests/test_circuit_cost.py` | 48 | gate counts, parameter validation, scheduling, reports, regression |
| `tests/test_phase2.py` | 30 | data loading, comparisons, accuracy targets, scaling, round-off floor |
| `tests/test_phase3.py` | 71 | density matrices, the three channels, gate-level execution, metrics, optimal `r` |
| `tests/test_phase4_audit.py` | 20 | Hamiltonian, both product formulas, channels, Choi positivity, metrics, error separation, optimal-`r` selection, pairing, seeded reproducibility |
| `tests/test_phase4_sdk.py` | 32 | qubit mapping, gate order/angles/targets, unitary and statevector equivalence, pre-transpilation resource agreement, coupling-map validity, transpilation determinism, layout restoration, Aer-vs-custom noise, channel conversion factors |

---

## 2. Phase 1 - circuit resources

* `circuit_cost.py` builds the chronological gate list for both orders from the same Hamiltonian used in
  `trotter.py`, and estimates CNOT/RX/RZ counts, total gates, logical depth and two-qubit depth.
* Formulas: first order `CNOT = 2r(n-1)`, `RZ = r(n-1)`, `RX = rn`; second order with merged half-steps
  `CNOT = 2(r+1)(n-1)`, `RZ = (r+1)(n-1)`, `RX = rn`.
* Outputs: `results/circuit_resources.csv` (132 rows), `results/circuit_reports.json`, figures
  `plot_H`-`plot_M`.
* Phase 4 re-verified the analytic counts against the compiled circuits in 64 cases with zero
  difference, and against the stored CSV with zero mismatches.

## 3. Phase 2 - accuracy versus resources

* `analysis/` reads only Phase 1 output and produces equal-step, equal-CNOT and equal-total-gate
  comparisons, accuracy targets, scaling fits and a supplementary sweep above `r = 200`.
* Outputs: `results/phase2/` (10 figures, 4 data files, `phase2_report.md`, `phase2_summary.json`).

## 4. Phase 3 - noise and the optimal step count

Implementation (`noise/`): density-matrix state, three channels applied after every gate in the Phase 1
gate list, fidelity/purity/trace-distance metrics, sensitivity grid, optimal-`r` search, first-vs-second
order pairing, 12 figures, report.

Configuration actually used:

* System sizes `n = 4, 5, 6`; both orders; `t = 1`; `J = h = 1`.
* Step grid `r = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512` (note: no 5, 7, 9, 10, 11).
* Gate-level noise at `p = 0, 1e-4, 1e-3, 1e-2` with `p1 = p2`.
* Global (per-step) noise at `p_g = 1e-4, 1e-3, 1e-2`, plus an end-of-circuit reference at `p_g = 1e-3`.
* Initial state `|0...0>`; 504 gate rows and 504 global rows; 72 optimal-`r` configurations.

Outputs: `results/phase3/` (CSVs, 12 figures, `phase3_report.md`, `phase3_summary.json`,
`configs/experiment_config.json`).

## 5. Phase 4 - audit, robustness, SDK

### 5.1 Audit (`validation/`)

Written from scratch and deliberately shares no code with `noise/`, `trotter.py` or `circuit_cost.py`
except `circuit_cost.build_circuit` (the circuit being checked). `independent.py` re-derives everything:
Pauli embedding by explicit index arithmetic, product formulas from `scipy.linalg.expm`, channels as
explicit Pauli sums, metrics from their definitions.

Results: **187/187** mathematical checks pass (largest equality deviation `2.1e-15`); 12 configurations /
144 stored values recalculated with 0 failures (largest absolute difference `5.8e-15`); **72/72** stored
optima confirmed as argmin of total error, 12 of them boundary cases; independent pairing agrees with
`noise.experiments.pair_orders` on **252/252** pairs.

### 5.2 Robustness configuration

* Reference noise `p1 = p2 = 1e-3`, `n = 4` (extended to 5 where cheap), `t = 1`.
* Step grid for robustness: `r = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64`.
* Initial states: `|0...0>`, `|1...1>`, `|+...+>`, one seeded random pure state
  (`RANDOM_SEED = 20260522`) - 32 groups.
* Evolution times `t = 0.5, 1, 2, 4` - 8 groups.
* Noise strength: the four Phase 3 levels per model - 12 groups, all non-increasing, 6 strictly, 0
  violations.
* Gate classes: single-qubit only, two-qubit only, both, at each level - 18 groups.
* Candidate grid: dense integer grids around the reported optimum at `n = 4, 5, 6` - 6 groups, 3 optima
  move by one step.

### 5.3 SDK layer (`quantum_sdk/`)

* Qiskit `2.6.2`, `qiskit-aer 0.18.4`; installed with `pip install qiskit qiskit-aer`.
* Circuits are built from the Phase 1 gate list; qubit mapping `qiskit_qubit = n - 1 - repo_qubit`.
* Study grid: `n = 3, 4, 5`, `r = 1, 2, 3, 4, 6, 8`, both orders (36 configurations).
* Transpilation: `A_logical_generic` (basis `rz`/`rx`/`cx`, optimisation 0, no coupling map) and
  `B_line` / `B_ring` / `B_star` (optimisation 1, explicit `CouplingMap`, `seed_transpiler = 7`).
* Noise: the project's channels rebuilt as explicit Pauli mixtures (`qiskit_aer.noise.pauli_error`) so
  no convention conversion enters the comparison; Aer's own `depolarizing_error` was measured separately
  and differs (`p_aer = 4p/3` one qubit, `16p/15` two qubits; equal only on pure product states).

Key numbers: 36/36 unitaries equivalent (worst spectral distance `8.9e-16`); pre-transpilation counts and
depth identical to Phase 1 in 36/36; line and ring add **0** CNOTs and 0 depth, the star adds up to
**+39** CNOTs and `2.40x` depth; Aer and the custom engine agree on the whole noisy density matrix to
`1.6e-15`; star routing costs `-5.5e-5` to `+2.9e-2` in total error, line routing exactly `0`.

### 5.4 Defects found and fixed during Phase 4

| Where | Defect | Fix |
|---|---|---|
| `quantum_sdk/equivalence.phase_insensitive` | rotated by `+arg(<a,b>)` instead of `-arg(<a,b>)`, so it reported the distance at twice the phase offset instead of the minimum | sign corrected, semantics documented, regression test added (`tests/test_phase4_sdk.py`) |
| `validation/plotting.fig_noise_strength` | a single boundary point at `r = 512` squashed the axis; a 12-entry legend was unreadable | rebuilt as two panels (gate-level / global per-step), boundary and single-level groups excluded |
| `validation/plotting.fig_noise_components` | filter precedence bug mixed three noise levels into a sawtooth and hid one series | filter corrected to `abs(max(p1,p2) - level) < 1e-15` |
| `quantum_sdk/plotting` (fig10/11/12) | a title contradicted its own data; the `r = 1` points squashed the interesting region | title corrected; log error axis and hollow logical markers |
| `quantum_sdk/experiments.noise_study` | star/line fidelity compared a *relabelled* state, overstating the star penalty | `layout.restore_order` identifies and verifies the permutation before any fidelity is computed |

No Phase 1-3 source file was modified. Three Phase 3 *claims* were re-worded in the README (non-increasing
instead of strictly decreasing; "best tested discrete step count"; near-independence of `r*` on `n` holds
only within a noise level).

---

## 6. Artifacts and where they live

```
results/                                  Phases 1-3 (all committed on purpose)
  circuit_resources.csv, circuit_reports.json, plot_*.png
  phase2/{figures,CSVs,phase2_report.md,phase2_summary.json}
  phase3/{figures,CSVs,configs,phase3_report.md,phase3_summary.json}
results/phase4/
  audit/audit_report.md                    the 10-section audit
  audit/validation_summary.json            187 checks, reproduction, optima, comparisons, robustness
  audit/independent_reproduction.csv       stored vs independent value per quantity
  audit/optimal_r_verification.csv         every stored optimum re-derived
  audit/comparison_audit.csv               all 252+252+252 pairwise comparisons
  robustness/*.csv                         5 experiments
  sdk/circuit_equivalence.csv              36 configurations
  sdk/pre_transpilation_resources.csv      36 configurations
  sdk/post_transpilation_resources.csv     144 rows (4 topologies x 36)
  sdk/transpilation_metadata.json          version, basis, coupling maps, layouts, seeds
  sdk/sdk_simulation_results.csv           Aer vs custom noisy density matrices
  hardware_aware/connectivity_comparison.csv
  hardware_aware/compiled_resource_comparison.csv
  hardware_aware/compiled_noise_results.csv
  figures/fig01 ... fig12                  all 12 inspected visually
  phase4_summary.json                      merged audit + SDK summary
  phase4_report.md                         20-section report
```

`results/` is **not** git-ignored and must not be: the CSVs and figures are the deliverable.

---

## 7. How to reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                    # 201 passed
python circuit_cost.py                        # Phase 1
python -m analysis.run_phase2                 # Phase 2
python -m noise.run_phase3                    # Phase 3 (~4 min)
python -m validation.run_audit                # Phase 4 audit + robustness (~5 min)
python -m quantum_sdk.run_phase4              # Phase 4 SDK (~30 s, needs the audit summary)
```

`validation.run_audit` accepts `--skip-robustness` and `--no-figures`; `quantum_sdk.run_phase4` accepts
`--quick` and `--no-figures`. Both write only under `results/phase4/`.

Determinism: every run is seeded (`RANDOM_SEED = 20260522` for the random state, `seed_transpiler = 7`
for Qiskit), so repeated runs produce identical CSVs.

---

## 8. Known issues, limitations, and what is deliberately absent

* **No hardware, no credentials, no backend interface.** Nothing in this repository talks to a device.
  The optional backend interface from the Phase 4 brief was not built; adding it is the first step
  towards hardware work, not a finished feature.
* **Independent per-gate noise only.** No crosstalk, `T1/T2`, leakage, readout error or drift.
* **Density matrices cap the system size** at `n = 6` in Phases 3-4 and `n = 5` for the SDK noise study
  (Aer's density-matrix method).
* **Compiled results are transpiler-specific.** All swap counts and depth numbers belong to Qiskit
  `2.6.2`, optimisation level 1, seed 7.
* **The optimum is discrete.** `r*` is the argmin over simulated candidates on the documented grid.
* **Relative-difference column in `independent_reproduction.csv`** is large (up to `4.3e-2`) for a couple
  of quantities whose absolute value is `~1e-13`; the pass criterion is the absolute tolerance `1e-9`.
* **Phase 3 grid skips `r = 5, 7, 9, 10, 11`.** Phase 4 showed this matters for first order. Re-running
  Phase 3 on the dense grid would remove the caveat.

---

## 9. Exact next steps

1. **Dense Phase 3 grid.** Re-run `noise.run_phase3` on an integer grid up to `r = 16` (or make the grid
   configurable) so the first-order optimum is not a grid artefact. Update the README numbers and the
   candidate-grid caveat in §19.6.
2. **Larger systems.** Add a matrix-product-operator or quantum-trajectory backend behind the same
   interface as `noise/density_matrix.py`, keeping `validation/independent.py` as the small-`n`
   reference for cross-checks.
3. **Correlated noise.** Extend `noise/channels.py` with a `T1/T2` amplitude-and-phase-damping pair and
   re-run the audit's channel section; the audit would need the matching analytic checks.
4. **Backend interface.** `backend/interface.py` with submit / status / result / counts / metadata,
   implemented first against a local simulator and tested without credentials.
5. **Routing study.** The second-order advantage survives in compiled two-qubit gates but not in depth
   on a mismatched topology. Try a different routing/layout strategy and record whether depth recovers.
6. **Publication pass.** Only the claims in README §19 (verified) plus §18 (now qualified) are safe to
   quote. Re-read §19.6 before writing any of them down.
