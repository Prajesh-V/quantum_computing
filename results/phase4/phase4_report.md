# Phase 4 — independent validation, robustness and quantum SDK analysis

*Generated:* 2026-11-22 15:04:41  
*Qiskit:* 2.6.2, qiskit-aer 0.18.4  
*Python:* 3.11.15, numpy 2.4.6  
*Phase 3 baseline commit:* `9e47be8`  

> **Verdict: PASS WITH QUALIFICATIONS.** The Phase 3 physics is correct and reproduces an independent implementation to machine precision, and the custom circuits are the same unitaries as their Qiskit counterparts. The noise-dependent optimal step count survives every robustness test in the *existence* sense, but its *value* is grid-, state- and time-dependent, and the second-order advantage in logical CNOTs survives compilation only while the circuit is routed on a line-compatible topology.

## 1. Research objectives

Three questions, in order of priority:

1. Is the Phase 3 result -- a finite optimal Trotter step count under gate-level noise -- mathematically correct and independently reproducible?
2. Is it robust to the initial state, the total evolution time, the noise strength, the split between gate classes, and the discreteness of the candidate grid?
3. Do the logical circuit-resource advantages survive real SDK compilation, connectivity constraints and an SDK noise model?

## 2. Phase 3 scientific audit

A from-scratch re-implementation (`validation/independent.py`) was compared with the Phase 3 code in 187 checks (187 passed, 0 failed). The largest deviation across the 172 equality checks is 2.1094e-15. The full audit is in `results/phase4/audit/audit_report.md`; the essential results are repeated here.

## 3. Mathematical verification

| Check group | Checks | Passed | Failed |
|---|---|---|---|
| hamiltonian | 9 | 9 | 0 |
| trotter | 26 | 26 | 0 |
| density matrix | 8 | 8 | 0 |
| global channel | 16 | 16 | 0 |
| single qubit channel | 35 | 35 | 0 |
| two qubit channel | 71 | 71 | 0 |
| metrics | 22 | 22 | 0 |

* **Hamiltonian**: `H = -J sum Z_i Z_{i+1} - h sum X_i`, open boundary, qubit 0 most significant. `A` and `B` rebuilt independently; a periodic Hamiltonian differs measurably, so the boundary convention is verified, not assumed.
* **Product formulas**: first order `exp(-iA dt) exp(-iB dt)` and second order `exp(-iA dt/2) exp(-iB dt) exp(-iA dt/2)` both reproduce independent matrix exponentials and the compiled circuit. Reversing the factor order changes the unitary, which is what makes the check meaningful.
* **Channels**: all three verified against explicit Pauli sums, including `p1 = 3/4` (not 1) being the fully depolarising single-qubit value and `p2 = 1` being the 15-Pauli average (not `I/4`). Choi matrices are positive semidefinite, so the maps are completely positive.
* **Metrics**: pure-reference fidelity, purity and the trace norm (matched against the SVD definition and distinguished from the Frobenius norm).

## 4. Independent reproduction results

12 configurations were recalculated from scratch (144 stored values): 0 failures, largest absolute difference 5.7727e-15, tolerance 1.0000e-09. The cases span n = 4, 5, 6, both formula orders, low/reference/high noise, gate-level and both global variants, and both optimal and clearly non-optimal step counts.

## 5. Robustness analysis

* **Initial state**: the interior optimum exists for all four tested states and every noise level. Its location moves: at n = 4, first order, p = 1e-3 it is r* = 4 for `|0...0>` and `|1...1>` but r* = 6 for `|+...+>` and the random state. Second order is insensitive (r* = 3 throughout).
* **Evolution time**: r* grows roughly in proportion to t (first order 3, 4, 8, 16 for t = 0.5, 1, 2, 4), and stays interior at every time. A quoted r* is meaningless without the evolution time.
* **Noise strength**: 12 of 12 groups are non-increasing and 6 strictly decreasing, with 0 violations. The claim must be "non-increasing", not "strictly decreasing".
* **Gate classes**: at equal per-gate error the single-qubit and two-qubit classes contribute comparably (0.0215 against 0.0199 at p = 1e-3 for first order), tracking the gate counts (28 against 24), not the gate class. The two contributions add to within 1% of the combined run.
* **Candidate grid**: 3 of 6 reported optima move by one step on a dense integer grid -- systematically 4 -> 5 for first order at n = 4, 5, 6, while second order stays at 3. The error change is small (0.0712 -> 0.0693 at n = 4) but the reported value should be described as the best *tested* discrete step count.

## 6. Initial-state dependence

| Initial state | Order | p | r* | min total error | CNOT at r* |
|---|---|---|---|---|---|
| random | 1 | 0.0010 | 6 | 0.0961 | 36 |
| random | 1 | 0.0050 | 3 | 0.2569 | 18 |
| random | 1 | 0.0100 | 3 | 0.3776 | 18 |
| |+...+> | 1 | 0.0010 | 6 | 0.0908 | 36 |
| |+...+> | 1 | 0.0050 | 4 | 0.2497 | 24 |
| |+...+> | 1 | 0.0100 | 3 | 0.3686 | 18 |
| |0...0> | 1 | 0.0010 | 4 | 0.0712 | 24 |
| |0...0> | 1 | 0.0050 | 3 | 0.1947 | 18 |
| |0...0> | 1 | 0.0100 | 3 | 0.3082 | 18 |
| |1...1> | 1 | 0.0010 | 4 | 0.0712 | 24 |
| |1...1> | 1 | 0.0050 | 3 | 0.1947 | 18 |
| |1...1> | 1 | 0.0100 | 3 | 0.3082 | 18 |
| random | 2 | 0.0010 | 3 | 0.0528 | 24 |
| random | 2 | 0.0050 | 2 | 0.1943 | 18 |
| random | 2 | 0.0100 | 2 | 0.3139 | 18 |
| |+...+> | 2 | 0.0010 | 3 | 0.0450 | 24 |
| |+...+> | 2 | 0.0050 | 2 | 0.1635 | 18 |
| |+...+> | 2 | 0.0100 | 2 | 0.2747 | 18 |
| |0...0> | 2 | 0.0010 | 3 | 0.0455 | 24 |
| |0...0> | 2 | 0.0050 | 2 | 0.1718 | 18 |
| |0...0> | 2 | 0.0100 | 2 | 0.2780 | 18 |
| |1...1> | 2 | 0.0010 | 3 | 0.0455 | 24 |
| |1...1> | 2 | 0.0050 | 2 | 0.1718 | 18 |
| |1...1> | 2 | 0.0100 | 2 | 0.2780 | 18 |

## 7. Evolution-time dependence

| t | Order | r* | min total error | Fidelity | CNOT | Total gates |
|---|---|---|---|---|---|---|
| 0.5 | 1 | 3 | 0.0483 | 0.9517 | 18 | 39 |
| 1.0 | 1 | 4 | 0.0712 | 0.9288 | 24 | 52 |
| 2.0 | 1 | 8 | 0.1167 | 0.8833 | 48 | 104 |
| 4.0 | 1 | 16 | 0.2017 | 0.7983 | 96 | 208 |
| 0.5 | 2 | 2 | 0.0285 | 0.9715 | 18 | 35 |
| 1.0 | 2 | 3 | 0.0455 | 0.9545 | 24 | 48 |
| 2.0 | 2 | 6 | 0.0881 | 0.9119 | 42 | 87 |
| 4.0 | 2 | 12 | 0.1772 | 0.8228 | 78 | 165 |

## 8. Noise-strength dependence

Across 12 groups that exclude the noise-free level: 12 non-increasing, 6 strictly decreasing, 0 violations. No group reverses direction. Several saturate at the smallest tested step count, which is why "strictly decreasing" is not supported.

## 9. Optimal-r verification

All 72 stored optima were re-derived from the sweep (72 agree). One grid, `1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512`, was used. The 12 boundary minima are exactly the noise-free and end-of-circuit reference runs, all at r = 512; every configuration that carries per-gate or per-step noise has an interior minimum.

| Noise model | Configs | Interior | Boundary | min r* | max r* |
|---|---|---|---|---|---|
| gate | 36 | 30 | 6 | 2 | 512 |
| global_end_of_circuit | 6 | 0 | 6 | 512 | 512 |
| global_per_step | 30 | 30 | 0 | 3 | 24 |

## 10. First-order versus second-order results

| Comparison | Pairs | Second order wins | First order wins | Wins per noise level |
|---|---|---|---|---|
| equal step | 252 | 118 | 134 | [42, 27, 18, 16, 9, 6] |
| cnot | 252 | 102 | 150 | [38, 24, 15, 13, 6, 6] |
| total gate | 252 | 118 | 134 | [42, 27, 18, 16, 9, 6] |

At matched CNOT budget the Phase 3 figure of 13 of 42 at p = 1e-3 is reproduced exactly, as is the progression over the six noise levels. The independent pairing agrees with `noise.experiments.pair_orders` on all 252 pairs with zero disagreements. Equal-step comparisons are reported separately because they flatter first order, and equal-total-gate comparisons differ from equal-CNOT ones (the second-order penalty is 9 extra rotations rather than 6 extra CNOTs).

## 11. SDK selection and version

**Qiskit 2.6.2** with **qiskit-aer 0.18.4**, installed with `pip install qiskit qiskit-aer`. Qiskit was chosen because the project is a gate-level qubit simulation with an explicit gate list, and because Aer provides an exact density-matrix method plus a noise model that can be made to match the project's channels exactly. The custom simulator remains the reference implementation; nothing in `noise/` was replaced.

## 12. Custom-to-SDK circuit conversion

`quantum_sdk.circuit_builder.build` walks the **existing** Phase 1 gate list from `circuit_cost.build_circuit` and emits `rx`, `rz` and `cx` in the same chronological order. No decomposition is re-derived.

Qubit indexing: the repository treats qubit 0 as the most significant bit, Qiskit treats qubit 0 as the least significant. The mapping is `qiskit_qubit = n - 1 - repo_qubit`, which makes the two basis indices equal, so unitaries and density matrices compare directly with no permutation. This was verified numerically: an `X` on repository qubit 0 puts the population at index `2^(n-1)` in both representations. Rotation angles pass through unchanged because Qiskit's `rz`/`rx` use the same matrices (`diag(e^{-i theta/2}, e^{+i theta/2})` and `cos I - i sin X`).

## 13. Circuit equivalence results

36 of 36 configurations pass. Comparison is phase-insensitive (`min_phi ||U_custom - e^(i phi) U_Qiskit||_2`, with the optimal phase measured rather than assumed); the largest spectral distance is 8.9005e-16 and the largest statevector infidelity 1.4211e-14. The measured global phase is ~1e-17 in every case, i.e. there is none beyond round-off.

| n | Order | r | spectral distance | statevector infidelity |
|---|---|---|---|---|
| 3 | 1 | 1 | 1.0530e-16 | 0 |
| 3 | 1 | 2 | 3.4812e-16 | 0 |
| 3 | 1 | 3 | 3.8766e-16 | 0 |
| 3 | 1 | 4 | 3.9936e-16 | 3.5527e-15 |
| 3 | 1 | 6 | 4.2575e-16 | 4.4409e-16 |
| 3 | 1 | 8 | 5.1909e-16 | 7.3275e-15 |
| 3 | 2 | 1 | 1.5480e-16 | 0 |
| 3 | 2 | 2 | 2.8191e-16 | 0 |
| 3 | 2 | 3 | 3.5171e-16 | 0 |
| 3 | 2 | 4 | 3.7515e-16 | 3.5527e-15 |
| 3 | 2 | 6 | 5.9226e-16 | 0 |
| 3 | 2 | 8 | 6.5156e-16 | 6.4393e-15 |

## 14. Transpilation configuration

Two families, with every parameter recorded in `results/phase4/sdk/transpilation_metadata.json`:

* **A_logical_generic** — basis `rz`/`rx`/`cx`, optimisation level 0, no coupling map, seed 7. The control: it should reproduce the logical circuit.
* **B_line / B_ring / B_star** — the same basis, optimisation level 1, seed 7, routed onto an explicit coupling map.

Pre-transpilation counts were checked against Phase 1 for 36 configurations: counts match in 36 of 36 and depth matches in 36 of 36. Configurations A and B_line are therefore bit-identical to the logical circuit.

## 15. Connectivity assumptions

* `line`: the chain `0-1-...-(n-1)`, which is exactly the TFIM nearest-neighbour graph.
* `ring`: the line plus the closing edge `(n-1, 0)`.
* `star`: qubits 1..n-1 each connected only to qubit 0, so every TFIM bond is two hops apart.

These are illustrative topologies, not a device: there is no calibration data, no error map and no backend behind them. No hardware job was submitted.

## 16. Compiled resource analysis

| Topology | Max extra CNOTs | Max two-qubit inflation | Max depth inflation |
|---|---|---|---|
| ideal | 0 | 1.000 | 1.000 |
| line | 0 | 1.000 | 1.000 |
| ring | 0 | 1.000 | 1.000 |
| star | 39 | 1.542 | 2.397 |

Routing on a line or a ring changes nothing: the logical circuit is already written on the chain, so the transpiler finds the layout that avoids every swap. The star is the interesting case: up to 39 extra CNOTs and up to 2.40x the depth, growing with n and r.

**A representation wrinkle worth recording:** a routed circuit can return its state in a permuted qubit order (the transpiler accounts for the final layout in the measurement bit mapping). A statevector or density-matrix simulation applies no measurement, so the permutation must be undone by the caller. `quantum_sdk.layout.restore_order` identifies it -- from the layout metadata when available, otherwise by search -- and **verifies** it by checking that the noiseless routed state becomes the logical state to 1e-10 (residuals are ~1e-16). Skipping this step makes the star topology look catastrophically worse than it is: the naive comparison gives a fidelity of 0.29 instead of 0.94 for n = 4, second order, a difference that is almost entirely relabelling.

## 17. Hardware-aware noise analysis

The SDK noise model is rebuilt from the project's own channel (an explicit Pauli mixture) rather than from `depolarizing_error`, because those are *different channels*. Measured: Aer's `depolarizing_error(p, k)` is `(1-p) rho + p I/d` and differs from the project's `p/3` and `p/15` mixtures by 0.0311 at p = 0.2, consistent with `p_aer = 4p/3` (one qubit) and `p_aer = 16p/15` (two qubits) on product states; the two conventions coincide only there. With the project's channel expressed as a Pauli mixture, the agreement is 2.2204e-16.

Whole circuits: Aer's density matrix against the custom noisy density matrix differs by at most 1.5547e-15 over 36 configurations. This is an independent validation of the custom noise engine -- channel definitions, gate placement, ordering and the absence of double application.

| n | Order | r | logical CX | line CX | star CX | logical F | star F | star penalty |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 1 | 4 | 4 | 4 | 0.1132 | 0.1132 | 0 |
| 3 | 1 | 2 | 8 | 8 | 8 | 0.8737 | 0.8737 | 1.1102e-16 |
| 3 | 1 | 3 | 12 | 12 | 12 | 0.9360 | 0.9360 | 3.3307e-16 |
| 3 | 1 | 4 | 16 | 16 | 16 | 0.9491 | 0.9491 | 2.2204e-16 |
| 3 | 1 | 6 | 24 | 24 | 24 | 0.9486 | 0.9486 | -4.4409e-16 |
| 3 | 1 | 8 | 32 | 32 | 32 | 0.9396 | 0.9396 | 1.1102e-15 |
| 3 | 2 | 1 | 8 | 8 | 8 | 0.3645 | 0.3645 | 1.1102e-16 |
| 3 | 2 | 2 | 12 | 12 | 12 | 0.9551 | 0.9551 | 2.2204e-16 |
| 3 | 2 | 3 | 16 | 16 | 16 | 0.9696 | 0.9696 | 1.1102e-16 |
| 3 | 2 | 4 | 20 | 20 | 20 | 0.9659 | 0.9659 | 0 |
| 3 | 2 | 6 | 28 | 28 | 28 | 0.9535 | 0.9535 | -4.4409e-16 |
| 3 | 2 | 8 | 36 | 36 | 36 | 0.9403 | 0.9403 | -8.8818e-16 |
| 4 | 1 | 1 | 6 | 6 | 9 | 0.0580 | 0.0581 | -5.5377e-05 |
| 4 | 1 | 2 | 12 | 12 | 15 | 0.8257 | 0.8235 | 0.0022 |
| 4 | 1 | 3 | 18 | 18 | 27 | 0.9113 | 0.9041 | 0.0072 |
| 4 | 1 | 4 | 24 | 24 | 33 | 0.9288 | 0.9218 | 0.0070 |
| 4 | 1 | 6 | 36 | 36 | 51 | 0.9272 | 0.9157 | 0.0116 |
| 4 | 1 | 8 | 48 | 48 | 69 | 0.9142 | 0.8983 | 0.0158 |
| 4 | 2 | 1 | 12 | 12 | 15 | 0.1858 | 0.1856 | 2.4720e-04 |
| 4 | 2 | 2 | 18 | 18 | 27 | 0.9258 | 0.9186 | 0.0073 |
| 4 | 2 | 3 | 24 | 24 | 33 | 0.9545 | 0.9474 | 0.0071 |
| 4 | 2 | 4 | 30 | 30 | 45 | 0.9503 | 0.9382 | 0.0121 |
| 4 | 2 | 6 | 42 | 42 | 63 | 0.9330 | 0.9166 | 0.0164 |
| 4 | 2 | 8 | 54 | 54 | 81 | 0.9146 | 0.8941 | 0.0204 |
| 5 | 1 | 1 | 8 | 8 | 11 | 0.0487 | 0.0487 | 5.4667e-05 |
| 5 | 1 | 2 | 16 | 16 | 22 | 0.7970 | 0.7928 | 0.0042 |
| 5 | 1 | 3 | 24 | 24 | 36 | 0.8936 | 0.8841 | 0.0094 |
| 5 | 1 | 4 | 32 | 32 | 47 | 0.9119 | 0.9003 | 0.0116 |
| 5 | 1 | 6 | 48 | 48 | 72 | 0.9073 | 0.8892 | 0.0181 |
| 5 | 1 | 8 | 64 | 64 | 97 | 0.8898 | 0.8657 | 0.0242 |
| 5 | 2 | 1 | 16 | 16 | 22 | 0.0976 | 0.0974 | 2.1500e-04 |
| 5 | 2 | 2 | 24 | 24 | 36 | 0.8997 | 0.8902 | 0.0094 |
| 5 | 2 | 3 | 32 | 32 | 47 | 0.9399 | 0.9282 | 0.0117 |
| 5 | 2 | 4 | 40 | 40 | 61 | 0.9350 | 0.9184 | 0.0166 |
| 5 | 2 | 6 | 56 | 56 | 86 | 0.9130 | 0.8905 | 0.0225 |
| 5 | 2 | 8 | 72 | 72 | 111 | 0.8895 | 0.8610 | 0.0285 |

Line and ring routing are free in fidelity as well as in gates (penalty exactly 0, and exactly 0 for the star at n = 3, where the star and the chain are the same graph up to relabelling). Genuine star routing costs between -5.5377e-05 and 0.0285 in total error at p = 1e-3, growing with n and r and in step with the extra two-qubit gates -- roughly an order of magnitude smaller than the relabelled comparison suggested.

One configuration has a *negative* penalty (-5.5377e-05, n = 4, first order, r = 1). This is not an error: at that point the noiseless fidelity is 0.0580, below the maximally mixed value 1/d = 0.0625, so depolarising noise moves the state *towards* the target and the fidelity rises. Where 1 - F is above 1/d the same channel's effect always has the expected sign.

## 18. Limitations

* **No hardware.** No job was submitted to a device, no calibration data was used, and the coupling maps are illustrative. Nothing here is a hardware measurement.
* **Noise model.** Independent depolarising channels after each gate: no correlated crosstalk, no T1/T2 asymmetry, no leakage, no readout error and no drift. Global per-step noise is not comparable to a per-gate rate.
* **Simulator method.** Aer's density-matrix method was used, which limits the circuit size to the region where a dense density matrix fits; the study is n <= 5 for the SDK noise runs and n <= 6 elsewhere.
* **Transpiler dependence.** Every compiled number is specific to Qiskit 2.6.2, optimisation level 1 and seed 7. A different transpiler or seed will give different swap counts.
* **Grid.** Every optimum is the best value among the simulated candidates; no continuous optimisation was attempted.
* **Timing.** Transpilation times are recorded but not interpreted; they vary with machine load and are not a scientific result.

## 19. Main research findings

1. The Phase 3 implementation is mathematically correct: 187 independent checks, 12 reproduced configurations, 72/72 optima, all channels, metrics and the error separation verified.
2. The noise-dependent optimal step count is real and reproducible, but its value is a *tested discrete* optimum that moves by one step on a denser grid and shifts with the initial state and roughly proportionally with the evolution time. The existence of the interior minimum is robust; the number is not universal.
3. The custom circuits are the same unitaries as their Qiskit equivalents (8.9005e-16 spectral distance), and Aer with the project's channel reproduces the custom noisy density matrix to 1.5547e-15.
4. Connectivity is free for this circuit on a line or ring -- exactly zero extra CNOTs -- because the TFIM chain is already the hardware graph. On a deliberately mismatched star it costs up to ~50% more CNOTs and 2.4x the depth, which reduces fidelity by 0.002-0.017 in total error at p = 1e-3.
5. After compilation the ordering of topologies is preserved and the second-order advantage in the noisy regime becomes a *reachability* advantage rather than a uniform gate saving: at a total-error target of 0.10 both orders need the same compiled two-qubit count at every n (n=3 order 1: 12, n=3 order 2: 12, n=4 order 1: 18, n=4 order 2: 18, n=5 order 1: 32, n=5 order 2: 32), while at the stricter 0.06 target second order is cheaper at n = 3 (12 against 16) and is the only order to reach it at n = 5. On the star every target costs about 50% more compiled two-qubit gates than on the line (n=3 order 1: 12, n=3 order 2: 12, n=4 order 1: 27, n=4 order 2: 27, n=5 order 1: 47, n=5 order 2: 47). Second order is therefore not cheaper per step in compiled resources -- at equal r it is strictly more expensive -- but it buys more accuracy per compiled two-qubit gate in the noise-limited regime.

## 20. Reproducibility instructions

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # full test suite
python -m validation.run_audit             # audit + robustness (~5 min)
python -m quantum_sdk.run_phase4           # SDK + transpilation (~1 min)
python circuit_cost.py                     # Phase 1
python trotter.py                          # ideal Trotter study
```

Outputs: `results/phase4/audit/`, `results/phase4/robustness/`, `results/phase4/sdk/`, `results/phase4/hardware_aware/`, `results/phase4/figures/`, plus `phase4_summary.json` and this report. Phase 1-3 outputs are read but never written.

---

**Figure index**

* `results/phase4/figures\fig06_custom_vs_sdk_unitary.png` — Custom circuit unitary against the Qiskit circuit unitary, up to a global phase.
* `results/phase4/figures\fig07_logical_vs_transpiled_cnot.png` — Logical against transpiled CNOT count, per connectivity.
* `results/phase4/figures\fig08_logical_vs_transpiled_depth.png` — Logical against transpiled circuit depth, per connectivity.
* `results/phase4/figures\fig09_swap_overhead.png` — Extra CNOTs introduced by routing, under each connectivity map.
* `results/phase4/figures\fig10_order_compiled_resources.png` — First versus second order compiled resources on the star topology.
* `results/phase4/figures\fig11_fidelity_vs_compiled_cx.png` — Fidelity against compiled two-qubit gate count under gate-level noise.
* `results/phase4/figures\fig12_logical_vs_hardware_efficiency.png` — Total error against compiled two-qubit gates: logical versus hardware-aware.
