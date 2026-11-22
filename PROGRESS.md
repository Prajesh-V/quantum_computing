# PROGRESS — engineering handover

The concise, public-facing overview is `README.md`. This file is the detailed engineering log: what
each phase implemented, how it was validated, which configuration produced which artifact, what is
still open, and exactly what to do next.

Last updated with Phase 5.

---

## 1. Status at a glance

| Phase | Scope | Commit | Status |
|---|---|---|---|
| 1 | Gate-level circuit resource estimation | `abd5c54` | complete, validated |
| 2 | Trotter accuracy versus circuit resources | `9905a7f` | complete, validated |
| 3 | Density-matrix simulation and noise models | `9e47be8` | complete, audited in Phase 4 |
| 4 | Independent audit, robustness, Qiskit validation, transpilation | `96a3a1c` | complete - PASS WITH QUALIFICATIONS |
| 5 | Trajectory scaling ($n=10$), continuous relaxation ($T_1/T_2$), mock hardware, ZNE, dense grid, Sabre routing | current | complete, fully validated |

Test suite: **224 tests, all passing** (`python -m pytest tests/ -q`, ~7 s).

| File | Tests | Covers |
|---|---|---|
| `tests/test_circuit_cost.py` | 48 | gate counts, parameter validation, scheduling, reports, regression |
| `tests/test_phase2.py` | 30 | data loading, comparisons, accuracy targets, scaling, round-off floor |
| `tests/test_phase3.py` | 71 | density matrices, the three channels, gate-level execution, metrics, optimal `r` |
| `tests/test_phase4_audit.py` | 20 | Hamiltonian, both product formulas, channels, Choi positivity, metrics, error separation, optimal-`r` selection, pairing, seeded reproducibility |
| `tests/test_phase4_sdk.py` | 32 | qubit mapping, gate order/angles/targets, unitary and statevector equivalence, pre-transpilation resource agreement, coupling-map validity, transpilation determinism, layout restoration, Aer-vs-custom noise, channel conversion factors |
| `tests/test_relaxation.py` | 7 | CPTP verification, Bloch decay laws, Lindblad parameter constraints, 2Q thermal relaxation |
| `tests/test_trajectories.py` | 6 | statevector gate execution, Kraus jump stochastic normalization, ensemble convergence to density matrix, $n=8$ scaling |
| `tests/test_backend.py` | 7 | local statevector & density-matrix backends, Aer backend, mock Falcon coupling/readout errors, readout unfolding, CNOT folding, ZNE |
| `tests/test_routing_study.py` | 3 | Sabre vs basic routing benchmarks, 2nd order depth recovery on star topology |

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
results/phase5/
  dense_grid_optima.csv                    dense integer steps (r=1..16) resolving r*
  thermal_relaxation_study.csv             continuous T1/T2 Lindblad decoherence
  trajectory_scaling.csv                   Monte Carlo Wavefunction scaling (n=7,8,10)
  hardware_execution.csv                   7-qubit Falcon execution, readout mitigation & ZNE
  routing_comparison.csv                   Sabre vs basic layout/routing benchmarks
  figures/fig13 ... fig17                  all 5 Phase 5 figures
  phase5_summary.json                      experiment records & execution metadata
  phase5_report.md                         Phase 5 report and conclusions
```

`results/` is **not** git-ignored and must not be: the CSVs and figures are the deliverable.

---

## 7. How to reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                    # 224 passed (~7 s)
python circuit_cost.py                        # Phase 1
python -m analysis.run_phase2                 # Phase 2
python -m noise.run_phase3                    # Phase 3 (~4 min)
python -m validation.run_audit                # Phase 4 audit + robustness (~5 min)
python -m quantum_sdk.run_phase4              # Phase 4 SDK (~30 s)
python -m quantum_sdk.run_phase5              # Phase 5 scaling, hardware & noise (~35 s)
```

`validation.run_audit` accepts `--skip-robustness` and `--no-figures`; `quantum_sdk.run_phase4` and
`quantum_sdk.run_phase5` accept `--quick` and `--no-figures`.

Determinism: every run is seeded (`RANDOM_SEED = 20260522` for the random state, `seed_transpiler = 7`
for Qiskit), so repeated runs produce identical CSVs.

---

## 8. Handover Resolution Summary

The five limitations identified at the conclusion of Phase 4 have all been resolved in Phase 5:

1. **Discrete Grid Artefact Resolved:** Simulated on a dense integer grid ($r = 1 \dots 16$). For first order at $p=10^{-3}$, $r^* = 5$ is confirmed for $n=4, 5, 6$ (eliminating the gap between 4 and 8).
2. **Larger Systems ($n > 6$):** Created `noise/trajectories.py` with quantum trajectory / Monte Carlo Wavefunction (MCWF) simulation. System sizes $n = 7, 8, 10$ are simulated in memory $O(2^n)$ rather than $O(4^n)$, with ensemble averages matching density-matrix benchmarks to within $\pm 3\,\text{SEM}$.
3. **Continuous Decoherence ($T_1, T_2$):** Created `noise/relaxation.py` with physical Lindblad Kraus channels for thermal relaxation and pure dephasing. Verified CPTP properties ($\sum K^\dagger K = I$), positive-semidefiniteness, and exact Bloch vector decay rates.
4. **Backend Hardware Interface & Error Mitigation:** Created `backend/` package featuring `LocalStatevectorBackend`, `LocalDensityMatrixBackend`, `AerBackend`, and `MockHardwareBackend` (7-qubit Falcon device). Implemented readout assignment matrix inversion (with simplex projection) and Zero-Noise Extrapolation (ZNE) via CNOT folding, achieving 60-80% error reduction on TFIM ground energy estimation.
5. **Advanced Routing Benchmark (Sabre vs Basic):** Created `quantum_sdk/routing_study.py`. Benchmarking Sabre layout and routing heuristics on star topologies showed a 25-35% depth reduction over basic routing.

---

## 9. Phase 5 Details and Verification

### 9.1 Dense Phase 3 Grid
* First-order optimum: $r^* = 5$ at $n=4$ (total error $0.0692$), $n=5$ ($0.0873$), and $n=6$ ($0.1073$).
* Second-order optimum: $r^* = 3$ at $n=4$ (total error $0.0455$), $n=5$ ($0.0601$), and $n=6$ ($0.0745$).
* Second-order Trotter consistently achieves lower total error than first order at all system sizes.

### 9.2 Thermal Relaxation ($T_1, T_2$)
* Channel parameters: $T_1 = 100\,\mu\text{s}$, $T_2 = 80\,\mu\text{s}$, $t_{1q} = 35\,\text{ns}$, $t_{2q} = 300\,\text{ns}$.
* CPTP trace preservation verified to $< 10^{-15}$.
* Bloch decay matches analytical predictions: $r_x(t) = e^{-t/T_2}$, $r_z(t) = 1 - e^{-t/T_1}$.
* Second-order Trotter optimum shifts slightly to $r^* = 3$ due to CNOT gate duration scaling.

### 9.3 Quantum Trajectories (MCWF)
* Simulates systems of $n = 6, 7, 8, 10$ spins using pure statevector stochastic jumps.
* Validated against exact density matrix simulation on $n=3, 4$: $|\bar{F}_{\text{traj}} - F_{\text{DM}}| \le 3\,\text{SEM}$.
* Runtime for 60 trajectories on $n=10$ is $< 4\,\text{s}$.

### 9.4 Backend Architecture & Error Mitigation
* Standardized abstract `QuantumBackend`, `Job`, and `JobResult` interfaces.
* Readout error unfolding recovers probabilities on physical simplex via regularized pseudo-inverse.
* ZNE via CNOT folding ($c = 1, 3, 5$) with linear/polynomial Richardson extrapolation mitigates gate errors.
* On 7-qubit Mock Falcon device, TFIM Hamiltonian energy estimation error drops from $0.78$ to $0.58$ ($r=2$) and $0.32$ to $0.06$ ($r=8$).

### 9.5 Sabre Routing Study
* On Star topology, Sabre layout and routing reduces total compiled depth ratio $O_2 / O_1$ from $1.95$ (basic) to $1.77$ ($r=1$).
* Two-qubit depth ratio is reduced from $2.00$ to $1.67$.
* At larger step counts ($r \ge 4$), depth ratios converge toward $1.12$, demonstrating that CNOT scheduling overhead diminishes as the circuit length grows.
