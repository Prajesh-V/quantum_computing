# Phase 4 scientific audit of Phase 3

*Baseline commit audited:* `9e47be8`  
*Audit generated:* 2026-11-22 14:46:56  
*Environment:* Python 3.11.15, numpy 2.4.6, scipy 1.17.1  

> **Verdict: PASS WITH QUALIFICATIONS.** The physics, the three noise channels, the metrics and the optimal-step calculation are all correct and independently reproducible. Three reported statements need narrower wording: the optimal step count is a discrete grid artefact at ±1, it depends on the initial state and on the evolution time, and the noise trend is non-increasing rather than strictly decreasing.

## 1. Audit scope

This audit re-implemented the Phase 3 physics from scratch (`validation/independent.py`) and compared it with the repository implementation. Nothing in `noise/`, `trotter.py`, `circuit_cost.py` or the Phase 1-3 results was modified, and the audit's own outputs live entirely under `results/phase4/`.

* 187 mathematical checks were run (187 passed, 0 failed); 172 of them are equality checks, whose largest observed deviation is 2.1094e-15.
* 12 Phase 3 configurations were recalculated independently (144 stored values compared, 0 failures, largest absolute difference 5.7727e-15).
* All 72 stored optimal step counts were re-derived from the sweep data (72 passed).
* 252 matched-CNOT, 252 matched-total-gate and 252 equal-step comparisons were rebuilt with an independent pairing rule and cross-checked against `noise.experiments.pair_orders`.
* Five new robustness experiments were run with the independent engine: initial state, evolution time, noise strength, noise components, and candidate-grid density.

Not audited: hardware calibration, backend noise models and anything requiring a real device. The noise parameters remain abstract simulation values throughout.

## 2. Mathematical verification

| Check group | Checks | Passed | Failed |
|---|---|---|---|
| hamiltonian | 9 | 9 | 0 |
| trotter | 26 | 26 | 0 |
| density matrix | 8 | 8 | 0 |
| global channel | 16 | 16 | 0 |
| single qubit channel | 35 | 35 | 0 |
| two qubit channel | 71 | 71 | 0 |
| metrics | 22 | 22 | 0 |

### 2.1 Hamiltonian

The implementation is `H = -J sum_{i<n-1} Z_i Z_{i+1} - h sum_i X_i` with open boundaries, `J = h = 1`. `A` and `B` were rebuilt independently and agree to machine precision; the ring (periodic) Hamiltonian differs measurably, which confirms that the boundary condition really is open and that the check is sensitive. Qubit 0 is the most significant bit: `X` on qubit 0 maps `|0...0>` to index `2^(n-1)`, and `X` on the last qubit maps it to index 1.

### 2.2 Product formulas

`trotter.trotter_unitary` and the compiled circuit were both compared with an independently built `exp(-iA dt)^r exp(-iB dt)^r` and with the symmetrised `exp(-iA dt/2) exp(-iB dt) exp(-iA dt/2)` form, for r = 1, 2, 3, 5 and for t = 0.7 s and 2.0 s. Matrix products act rightmost-first, so the first-order step applies the X block before the ZZ block; reversing that order changes the unitary by a measurable amount, which is what makes the agreement test meaningful. The merged second-order ZZ layers reproduce the half-angle `dt/2` decomposition exactly.

### 2.3 Density-matrix evolution

Ideal evolution preserves the trace, Hermiticity, positivity and purity, and the simulator's density matrix equals `|psi><psi|` built from the circuit unitary. The audited `apply_operator` and the independent `apply_local` agree on the same operators and qubits.

### 2.4 Noise channels

**Global.** `(1-p) rho + p I/d` reproduces the independent construction for p = 0.01, 0.37, 0.9; p = 0 is the identity channel and p = 1 gives `I/d` exactly. Trace, Hermiticity and positivity are preserved, and the Choi matrix is positive semidefinite, so the channel is completely positive.

**Single-qubit.** For one qubit `sum_{P in X,Y,Z} P rho P = 2I - rho`, so the channel is `(1 - 4p/3) rho + (2p/3) I` on a normalised state. This was checked state by state for `|0>`, `|1>`, `|+>` and `|->`: all four match the analytic form at p = 0, 0.1, 0.75 and 1. Two consequences matter and both hold in the code: **`p = 3/4` is the fully depolarising value** (it maps every pure state to `I/2`), and **`p = 1` is not** -- it leaves `(1/3)(2I - rho)`, which is not maximally mixed. The channel acts on the requested qubit only: applying it to qubit 1 of `|00>` leaves qubit 0 pure.

**Two-qubit.** All 16 Pauli strings are generated with 16 distinct labels and 15 non-identity terms, and the normalisation is `p/15`. The optimised partial-trace implementation matches the explicit 15-term sum to ~1e-16 for `|00>`, `|01>`, `|++>` and a Bell state at p = 0, 1e-3, 0.3 and 1. At p = 1 the channel is the exact average over the 15 non-identity Paulis -- for `|00>` that is `diag(3,4,4,4)/15`, which is *not* `I/4`; the code produces exactly that. The identity the fast path relies on, `sum_P P rho P = 2^k Tr_k(rho) (x) I_k`, was verified against a hand-built partial trace. The Choi matrix is positive semidefinite.

### 2.5 Metrics

Fidelity against a pure reference is `<psi| rho |psi>`, giving 1 for identical pure states, 0 for orthogonal ones, and 1/2 against `I/2`. `fidelity_general` returns the **squared** (Uhlmann) fidelity -- it gives 1/2 for `|0><0|` against `I/2`, i.e. the unsquared value would be 0.707 -- and it is used only in tests, so the convention cannot leak into the reported errors. Purity is `Tr(rho^2)`. Trace distance is `0.5 ||rho - sigma||_1`: it equals the SVD definition exactly and differs from the Frobenius half-norm (0.5 versus 0.433 for `I/2` against `|0><0|`), so it is genuinely the trace norm.

### 2.6 Error separation

`total_error = 1 - F(rho_noisy, psi_exact)`, `ideal_trotter_error = 1 - F(rho_trotter, psi_exact)` and `noise_induced_error = 1 - F(rho_noisy, psi_trotter)` use three separate references. They are not additive: over the 420 noisy gate-level rows the deviation `delta = total - (trotter + noise)` ranges from -0.2425 to 4.3117e-06, with a mean absolute value of 0.0060. The Phase 3 report's figure of about 0.24 is reproduced exactly (0.2425) and occurs at n=6, second order, r=1, where the two errors are each near 0.95 and 0.25. Infidelities against different references are not a sum because the overlaps between the three states do not vanish; additivity would only hold for orthogonal states or for very small errors.

## 3. Implementation verification

* **Gate counts.** Phase 1's closed forms were checked against the compiled circuits for 64 (n, r, order) combinations: largest discrepancy 0. The counts stored in `gate_noise_results.csv` were recomputed for all 504 rows: 0 mismatches.
* **Gate ordering and noise placement.** Each gate applies `U rho U^dagger` and then its channel, in the chronological order of `circuit_cost.build_circuit`; RX and RZ receive the single-qubit channel (RZ separately configurable) and CNOT the two-qubit channel. No gate is skipped or duplicated: the channel count recorded in the CSV equals the number of noisy gates.
* **Global-channel placement.** The per-step variant applies exactly one channel per Trotter step, with step boundaries recomputed from the block sequence; my independent boundary construction (counted from the gate sizes) is identical to `noise.noisy_simulator.step_ranges` for every tested (n, r, order). The end-of-circuit variant applies one channel after the last gate and records one channel, as documented.
* **Qubit indexing.** Both engines use qubit 0 as the most significant bit; the independent engine embeds operators by explicit index arithmetic, and it matches the audited reshape/permute path to ~1e-16.

## 4. Independent reproduction

The independent engine shares no channel, metric or evolution code with `noise/`; it uses the same gate list, because that is the circuit under test. All 12 configurations were recalculated from scratch (144 stored values). Every comparison passed at a tolerance of 1.0000e-09; the largest absolute difference is 5.7727e-15.

| Configuration | n | order | r | stored total error | independent | max |diff| | result |
|---|---|---|---|---|---|---|---|
| n=4 1st order, reference noise, optimal r | 4 | 1 | 4 | 0.0712 | 0.0712 | 1.4433e-15 | pass |
| n=4 1st order, reference noise, r=64 (overshoot) | 4 | 1 | 64 | 0.4638 | 0.4638 | 5.7727e-15 | pass |
| n=4 2nd order, reference noise, optimal r | 4 | 2 | 3 | 0.0455 | 0.0455 | 5.5511e-16 | pass |
| n=4 2nd order, low noise, r=12 | 4 | 2 | 12 | 0.0130 | 0.0130 | 3.3307e-15 | pass |
| n=5 1st order, 5e-3, r=6 | 5 | 1 | 6 | 0.3391 | 0.3391 | 2.4980e-15 | pass |
| n=5 1st order, high noise, r=1 | 5 | 1 | 1 | 0.9526 | 0.9526 | 3.3307e-16 | pass |
| n=6 2nd order, high noise, r=2 | 6 | 2 | 2 | 0.4180 | 0.4180 | 7.7716e-16 | pass |
| n=6 1st order, low noise, r=24 | 6 | 1 | 24 | 0.0404 | 0.0404 | 2.8866e-15 | pass |
| n=4 2nd order, noiseless | 4 | 2 | 8 | 1.4880e-04 | 1.4880e-04 | 3.4420e-15 | pass |
| n=4 1st order, global per step, r=8 | 4 | 1 | 8 | 0.0150 | 0.0150 | 3.3307e-16 | pass |
| n=6 2nd order, global per step, r=16 | 6 | 2 | 16 | 0.1462 | 0.1462 | 1.1657e-15 | pass |
| n=5 1st order, global end of circuit, r=16 | 5 | 1 | 16 | 0.0032 | 0.0032 | 4.6664e-15 | pass |

Relative differences are reported in `independent_reproduction.csv` but are not used for the pass criterion: for a noiseless configuration the noise-induced error is at the round-off level, where a relative difference is meaningless while the absolute difference is ~1e-16.

## 5. Optimal-r verification

All 72 stored optima were re-derived by taking the argmin of the measured total error over the candidates that were actually simulated (72 agree, 0 disagree). One candidate grid was used throughout: `[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0, 256.0, 512.0]`. No group mixes system sizes, formula orders or noise settings, and every group has a single evolution time.

| Noise model | Configurations | Interior minima | Boundary minima | smallest r* | largest r* |
|---|---|---|---|---|---|
| gate | 36 | 30 | 6 | 2 | 512 |
| global_end_of_circuit | 6 | 0 | 6 | 512 | 512 |
| global_per_step | 30 | 30 | 0 | 3 | 24 |

The 12 boundary minima are exactly the six noise-free gate-level runs and the six end-of-circuit global runs, all at the largest tested step count r = 512. That is the correct behaviour for those references -- with no per-gate noise, more steps is always better -- and it is reported as a boundary minimum rather than smoothed into an interior one. **Every configuration that actually carries per-gate or per-step noise has an interior minimum.**

## 6. Robustness findings

### 6.1 Initial state

Four initial states were simulated at n = 4 with the independent engine: `|0...0>`, `|1...1>`, `|+...+>` and one random pure state with seed 20260522.

| Initial state | Order | p | r* | Status | min total error | CNOT at r* |
|---|---|---|---|---|---|---|
| random | 1 | 0.0010 | 6 | interior | 0.0961 | 36 |
| random | 1 | 0.0050 | 3 | interior | 0.2569 | 18 |
| random | 1 | 0.0100 | 3 | interior | 0.3776 | 18 |
| |+...+> | 1 | 0.0010 | 6 | interior | 0.0908 | 36 |
| |+...+> | 1 | 0.0050 | 4 | interior | 0.2497 | 24 |
| |+...+> | 1 | 0.0100 | 3 | interior | 0.3686 | 18 |
| |0...0> | 1 | 0.0010 | 4 | interior | 0.0712 | 24 |
| |0...0> | 1 | 0.0050 | 3 | interior | 0.1947 | 18 |
| |0...0> | 1 | 0.0100 | 3 | interior | 0.3082 | 18 |
| |1...1> | 1 | 0.0010 | 4 | interior | 0.0712 | 24 |
| |1...1> | 1 | 0.0050 | 3 | interior | 0.1947 | 18 |
| |1...1> | 1 | 0.0100 | 3 | interior | 0.3082 | 18 |
| random | 2 | 0.0010 | 3 | interior | 0.0528 | 24 |
| random | 2 | 0.0050 | 2 | interior | 0.1943 | 18 |
| random | 2 | 0.0100 | 2 | interior | 0.3139 | 18 |
| |+...+> | 2 | 0.0010 | 3 | interior | 0.0450 | 24 |
| |+...+> | 2 | 0.0050 | 2 | interior | 0.1635 | 18 |
| |+...+> | 2 | 0.0100 | 2 | interior | 0.2747 | 18 |
| |0...0> | 2 | 0.0010 | 3 | interior | 0.0455 | 24 |
| |0...0> | 2 | 0.0050 | 2 | interior | 0.1718 | 18 |
| |0...0> | 2 | 0.0100 | 2 | interior | 0.2780 | 18 |
| |1...1> | 2 | 0.0010 | 3 | interior | 0.0455 | 24 |
| |1...1> | 2 | 0.0050 | 2 | interior | 0.1718 | 18 |
| |1...1> | 2 | 0.0100 | 2 | interior | 0.2780 | 18 |

The interior optimum survives for every initial state: no state produces a boundary minimum once noise is on. The location, however, is state-dependent -- `|0...0>` and `|1...1>` agree exactly (they are related by the model's symmetry) and give r* = 4 for first order at p = 1e-3, while `|+...+>` and the random state give r* = 6. Second order is much less sensitive (r* = 3 for all four states at that level). Because the Trotter error depends on the state, the optimum cannot be a property of the noise alone.

### 6.2 Total evolution time

At n = 4 and p = 1e-3, four total evolution times:

| t | Order | r* | Status | min total error | Fidelity at r* | CNOT | Total gates |
|---|---|---|---|---|---|---|---|
| 0.5 | 1 | 3 | interior | 0.0483 | 0.9517 | 18 | 39 |
| 1.0 | 1 | 4 | interior | 0.0712 | 0.9288 | 24 | 52 |
| 2.0 | 1 | 8 | interior | 0.1167 | 0.8833 | 48 | 104 |
| 4.0 | 1 | 16 | interior | 0.2017 | 0.7983 | 96 | 208 |
| 0.5 | 2 | 2 | interior | 0.0285 | 0.9715 | 18 | 35 |
| 1.0 | 2 | 3 | interior | 0.0455 | 0.9545 | 24 | 48 |
| 2.0 | 2 | 6 | interior | 0.0881 | 0.9119 | 42 | 87 |
| 4.0 | 2 | 12 | interior | 0.1772 | 0.8228 | 78 | 165 |

The optimum grows with the evolution time, roughly in proportion to it (first order: 3, 4, 8, 16 for t = 0.5, 1, 2, 4), and it is interior at every time. This is expected -- a longer evolution accumulates more formula error at fixed r -- but it means a single quoted r* is only meaningful together with the evolution time.

### 6.3 Noise strength

Across the 12 groups that exclude the noise-free level, 12 are non-increasing and 6 are strictly decreasing; there are 0 violations and the largest increase is 0 steps. Groups that include the noise-free point (6) are non-increasing too, because the noise-free optimum sits at the largest tested step count.

| n | Order | Noise model | Levels | r* sequence | Strictly decreasing | Non-increasing | Violations |
|---|---|---|---|---|---|---|---|
| 4 | 1 | gate | 0, 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 512, 12, 6, 4, 3, 3 | no | yes | 0 |
| 4 | 1 | global_end_of_circuit | 0.0010 | 512 | yes | yes | 0 |
| 4 | 1 | global_per_step | 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 24, 12, 12, 6, 4 | no | yes | 0 |
| 4 | 2 | gate | 0, 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 512, 4, 4, 3, 2, 2 | no | yes | 0 |
| 4 | 2 | global_end_of_circuit | 0.0010 | 512 | yes | yes | 0 |
| 4 | 2 | global_per_step | 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 8, 6, 6, 4, 3 | no | yes | 0 |
| 5 | 1 | gate | 0, 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 512, 8, 6, 4, 3, 2 | yes | yes | 0 |
| 5 | 1 | global_end_of_circuit | 0.0010 | 512 | yes | yes | 0 |
| 5 | 1 | global_per_step | 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 24, 12, 12, 6, 6 | no | yes | 0 |
| 5 | 2 | gate | 0, 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 512, 6, 4, 3, 2, 2 | no | yes | 0 |
| 5 | 2 | global_end_of_circuit | 0.0010 | 512 | yes | yes | 0 |
| 5 | 2 | global_per_step | 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 8, 6, 6, 4, 3 | no | yes | 0 |
| 6 | 1 | gate | 0, 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 512, 8, 6, 4, 3, 2 | yes | yes | 0 |
| 6 | 1 | global_end_of_circuit | 0.0010 | 512 | yes | yes | 0 |
| 6 | 1 | global_per_step | 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 24, 16, 12, 6, 6 | no | yes | 0 |
| 6 | 2 | gate | 0, 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 512, 6, 4, 3, 2, 2 | no | yes | 0 |
| 6 | 2 | global_end_of_circuit | 0.0010 | 512 | yes | yes | 0 |
| 6 | 2 | global_per_step | 1.0000e-04, 5.0000e-04, 0.0010, 0.0050, 0.0100 | 8, 6, 6, 4, 4 | no | yes | 0 |

The claim should be stated as **non-increasing**, not "monotonically decreasing": several groups reach the smallest tested step count and stay there (for example r* = 2, 2 at p = 5e-3 and 1e-2). No group ever reverses direction, which is the physically meaningful part of the claim.

### 6.4 Single-qubit versus two-qubit contributions

`p1` alone, `p2` alone and both together, at n = 4, with equal per-gate probabilities:

| Order | Nominal p | Variant | r* | noise error at r=4 | total error at r=4 | single-qubit gates | CNOTs |
|---|---|---|---|---|---|---|---|
| 1 | 0.0010 | both | 4 | 0.0410 | 0.0712 | 7 | 6 |
| 1 | 0.0010 | single_qubit_only | 6 | 0.0215 | 0.0525 | 7 | 6 |
| 1 | 0.0010 | two_qubit_only | 6 | 0.0199 | 0.0511 | 7 | 6 |
| 1 | 0.0050 | both | 3 | 0.1869 | 0.2112 | 7 | 6 |
| 1 | 0.0050 | single_qubit_only | 4 | 0.1025 | 0.1301 | 7 | 6 |
| 1 | 0.0050 | two_qubit_only | 4 | 0.0955 | 0.1237 | 7 | 6 |
| 1 | 0.0100 | both | 3 | 0.3351 | 0.3535 | 7 | 6 |
| 1 | 0.0100 | single_qubit_only | 3 | 0.1934 | 0.2171 | 7 | 6 |
| 1 | 0.0100 | two_qubit_only | 3 | 0.1811 | 0.2060 | 7 | 6 |
| 2 | 0.0010 | both | 3 | 0.0474 | 0.0497 | 10 | 12 |
| 2 | 0.0010 | single_qubit_only | 4 | 0.0235 | 0.0259 | 10 | 12 |
| 2 | 0.0010 | two_qubit_only | 4 | 0.0246 | 0.0270 | 10 | 12 |
| 2 | 0.0050 | both | 2 | 0.2130 | 0.2149 | 10 | 12 |
| 2 | 0.0050 | single_qubit_only | 3 | 0.1113 | 0.1134 | 10 | 12 |
| 2 | 0.0050 | two_qubit_only | 3 | 0.1164 | 0.1185 | 10 | 12 |
| 2 | 0.0100 | both | 2 | 0.3755 | 0.3769 | 10 | 12 |
| 2 | 0.0100 | single_qubit_only | 2 | 0.2088 | 0.2106 | 10 | 12 |
| 2 | 0.0100 | two_qubit_only | 2 | 0.2181 | 0.2199 | 10 | 12 |

At equal per-gate error the two classes contribute almost equally (for first order at p = 1e-3, 0.0215 from single-qubit gates against 0.0199 from CNOTs), which tracks the gate counts -- 28 single-qubit gates against 24 CNOTs in that circuit -- rather than the gate class. The noise-induced errors of the two variants add to within 1% of the combined run. **The correct claim is that the split follows counts and probabilities, not that two-qubit noise dominates.** The optimum itself is almost unchanged between the two single-class runs, because the total gate count drives it.

### 6.5 Candidate-grid sensitivity

The Phase 3 grid is `1, 2, 3, 4, 6, 8, 12, ...`, so it never tests 5, 7, 9, 10 or 11. Every integer within eight steps of the reported optimum was simulated at p = 1e-3:

| n | Order | Phase 3 grid r* | Dense grid r* | Changed | Dense range | error at Phase 3 r* | error at dense r* |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 4 | 5 | yes | 1-12 | 0.0712 | 0.0692 |
| 4 | 2 | 3 | 3 | no | 1-11 | 0.0455 | 0.0455 |
| 5 | 1 | 4 | 5 | yes | 1-12 | 0.0881 | 0.0873 |
| 5 | 2 | 3 | 3 | no | 1-11 | 0.0601 | 0.0601 |
| 6 | 1 | 4 | 5 | yes | 1-12 | 0.1084 | 0.1073 |
| 6 | 2 | 3 | 3 | no | 1-11 | 0.0745 | 0.0745 |

3 of 6 configurations move by one step on the denser grid. The error difference is small (for n = 4, first order, 0.06925 at r = 5 against 0.07120 at r = 4), so no conclusion changes, but the honest phrasing is **"the best tested discrete step count"** and the reported r* should not be presented as an exact integer optimum.

## 7. First-order versus second-order comparison

Three comparisons are kept separate, and a win means a strictly smaller total error (there are no ties in any of them).

| Comparison | Pairs | Second order wins | First order wins |
|---|---|---|---|
| Equal step count | 252 | 118 | 134 |
| Equal CNOT budget | 252 | 102 | 150 |
| Equal total-gate budget | 252 | 118 | 134 |

The independent pairing agrees with the corrected `noise.experiments.pair_orders` on all 252 pairs (252 mine against 252 of theirs, 0 unmatched either way, largest difference in the paired error 0, 0 disagreements about the winner). No pair compares different system sizes, orders, noise models or noise levels.

### 7.1 Matched CNOT budget

| p1 = p2 | Pairs | Second order wins | First order wins |
|---|---|---|---|
| 0 | 42 | 38 | 4 |
| 1.0000e-04 | 42 | 24 | 18 |
| 5.0000e-04 | 42 | 15 | 27 |
| 0.0010 | 42 | 13 | 29 |
| 0.0050 | 42 | 6 | 36 |
| 0.0100 | 42 | 6 | 36 |

At the reference level p = 1e-3 second order wins 13 of 42 -- the Phase 3 figure of 13 of 42 is reproduced exactly, as is the progression [38, 24, 15, 13, 6, 6] over p = 0 ... 1e-2 (102 of 252 pooled). 69 of the wins occur at r <= 8 and the largest winning step count is 512.

The mechanism has two parts, and they should be stated separately: in the small-r region the smaller per-step overhead of first order (`2(n-1)` fewer CNOTs) lets it afford one extra Trotter step at the same budget, and above the crossover the two formulas are equally inaccurate so the slightly smaller first-order circuit accumulates slightly less noise.

### 7.2 Equal step count

This is the comparison that flatters first order, because the second-order circuit is larger at the same r. It is reported for completeness, not as evidence of resource efficiency:

| p1 = p2 | Pairs | Second order wins | First order wins |
|---|---|---|---|
| 0 | 42 | 42 | 0 |
| 1.0000e-04 | 42 | 27 | 15 |
| 5.0000e-04 | 42 | 18 | 24 |
| 0.0010 | 42 | 16 | 26 |
| 0.0050 | 42 | 9 | 33 |
| 0.0100 | 42 | 6 | 36 |

At p = 1e-3 second order is the more accurate formula for every tested step count up to r = 8 and loses beyond the crossover, which is where the total error is already dominated by noise. Beyond r = 12 the gap stays below about 3% of the total error.

### 7.3 Equal total-gate budget

Under a total-gate budget second order wins [42, 27, 18, 16, 9, 6] over the six noise levels (118 of 252), more than under a CNOT budget, because the second-order circuit's penalty is 9 extra rotations rather than 6 extra CNOTs. In this comparison first order never gains a whole extra step, which is why the two comparisons differ.

The conclusion is therefore: second order is more accurate *per step* and *per total gate*, but not *per CNOT* and not under noise at small step counts. Phase 2's resource advantage survives only in the low-noise, formula-dominated regime.

## 8. Scientific limitations

**Established by these experiments (safe to state):**

* The density-matrix engine, the three channels and the metrics are mathematically correct and reproduce an independent implementation to machine precision.
* In the tested range (n = 4, 5, 6; t = 1; gate-level or per-step depolarising noise), the measured total error has an interior minimum in the step count at every noise level that was tested.
* At equal per-gate error the single-qubit and two-qubit gate classes contribute comparably, in proportion to their gate counts.
* With noise, second order does not keep its noise-free advantage: at matched CNOT budget it wins only in the low step-count region.

**Observed only in the tested configurations (needs qualification):**

* The location of the optimum is grid-limited: on a dense integer grid it can move by one step, so it should be quoted as "the best tested discrete step count", not as an exact integer optimum.
* The optimum depends on the initial state (r* = 4 versus 6 for first order at the reference noise level) and grows roughly in proportion to the total evolution time.
* The noise trend is non-increasing, not always strictly decreasing: several groups saturate at the smallest tested step count.
* Near-independence of r* from the system size holds for n = 4, 5, 6 only.
* The single-/two-qubit split is a statement about counts at equal per-gate error; with different probabilities the balance follows the probabilities.

**Hypotheses requiring further experiments:**

* That an interior optimum survives at larger n, longer times and with spatially correlated or non-unital noise.
* That the r* ~ t scaling observed at n = 4 is a general law.

**Limitations of the noise model:**

* Depolarising channels are applied independently after each gate: no correlated crosstalk, no T1/T2 asymmetry, no leakage, no readout error, no drift, and no gate-duration dependence.
* Global noise is a per-step parameter and is not comparable to a per-gate rate; the two families of experiment must not be pooled.
* All parameters are abstract simulation values, not hardware error rates. Density matrices are stored densely, which limits the study to n <= 6.

## 9. Recommended corrections

No mathematical or implementation error was found. The corrections below are about wording, and each is backed by the experiment named next to it.

1. **Quote the optimum as "best tested discrete step count".** On a dense integer grid 3 of 6 configurations move by one step, for example n = 4 first order from r = 4 to r = 5.
2. **State the noise trend as non-increasing.** 12 of 12 groups are non-increasing and only 6 are strictly decreasing; several saturate at the smallest tested step count. No group reverses direction, so "never increases with noise" is defensible and "strictly decreases" is not.
3. **Add the initial-state and evolution-time dependence.** The existence of an interior optimum is robust; its location is not. At n = 4, first order, p = 1e-3 the optimum is r* = 4 for `|0...0>` and r* = 6 for `|+...+>`, and it runs 3, 4, 8, 16 for t = 0.5, 1, 2, 4.
4. **Do not say that two-qubit noise dominates.** At equal per-gate error the split follows the gate counts (28 single-qubit gates against 24 CNOTs gives 0.0215 against 0.0199 at p = 1e-3). With `p2 > p1` the balance follows the probabilities instead.
5. **Keep the error definitions apart.** The three errors are not additive (maximum deviation 0.24), so `total = trotter + noise` must never be presented as an identity.

## 10. Final assessment

1. **Is the density-matrix implementation mathematically correct?** Yes. Unitary evolution, trace, Hermiticity, positivity and purity all behave correctly, and the simulator matches an independent implementation to ~1e-16.
2. **Are all three noise models implemented consistently with their definitions?** Yes, including the two traps: `p1 = 3/4` (not 1) is the fully depolarising value for the `p/3` single-qubit channel, and `p2 = 1` is the 15-Pauli average, not `I/4`. Both are documented and both match the code.
3. **Are the error metrics correct?** Yes: pure-reference fidelity, infidelity, purity and the trace norm (verified against the SVD definition and distinguished from the Frobenius norm).
4. **Is the optimal-r calculation correct?** Yes -- it is the measured argmin over the tested candidates, it never extrapolates, and boundary minima are labelled as such. All 72 stored values reproduce.
5. **Are the main conclusions reproducible?** Yes: 12 configurations, 144 stored values, largest difference 5.8e-15, and every optimal step count re-derived from the sweep.
6. **Are the conclusions robust to initial state and evolution time?** Only partly. The *existence* of an interior optimum is robust across all four initial states and all four evolution times tested. Its *location* is not: it shifts with the initial state and grows with the evolution time.
7. **Are the first-order/second-order comparisons fair?** Yes, now. The corrected pairing matches an independent implementation on all 252 pairs, no pair mixes noise levels, and the three comparison types are reported separately.
8. **Which claims are ready for a paper?** The mathematical correctness of the engine and channels; the existence of a measured interior optimum under gate-level depolarising noise; the disappearance of the second-order advantage under noise at matched gate budget; the relative contributions of the two gate classes.
9. **Which claims need more evidence?** Any specific value of r* (it is grid- and state-dependent), the near-independence of r* from n, "monotonic" as opposed to "non-increasing", and anything extrapolated beyond n = 6 or beyond the tested noise family.

---

Reproduce with:

```bash
python -m validation.run_audit
python -m pytest tests/ -q
```
