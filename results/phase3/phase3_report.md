# Phase 3 — Density-matrix simulation and gate-level noise

_Generated automatically by `python -m noise.run_phase3`. Every number below comes from a simulation recorded in `results/phase3/*.csv`; nothing is extrapolated._

## 1. Research question

Phases 1 and 2 counted gates and measured Trotter error for an ideal, noise-free circuit. Real gates are noisy, so the question this phase asks is:

> Does increasing the number of Trotter steps still help once every gate carries an error, and is there a step count that minimises the **total** simulation error for a fixed evolution time?

Two effects compete. More steps reduce the formula error (as `1/r` for the first-order formula and `1/r^2` for the second-order one) but add gates, and therefore add noise. If the second effect eventually wins, the total error has an interior minimum. That minimum is the central quantity of this phase; it is measured, not assumed to exist.

## 2. Mathematical background: density matrices

A state vector `|psi>` describes a *pure* state. A noisy state is a classical mixture, written as a matrix

```
rho = sum_k p_k |psi_k><psi_k|,   sum_k p_k = 1,   p_k >= 0
```

A pure state is the special case `rho = |psi><psi|` with `rho^2 = rho`. Three properties must hold for every physical state, and all three are tested:

* `Tr(rho) = 1` — total probability one.
* `rho = rho^dagger` — Hermitian, so expectation values are real.
* every eigenvalue `>= 0` — positive semidefinite, so no negative probability.

Unitary evolution keeps the state pure: `rho -> U rho U^dagger`. Purity is `Tr(rho^2)`, equal to 1 for a pure state and `1/d` for the maximally mixed state `I/d` with `d = 2^n`. Noise is anything that is not of this form; the channels below are linear maps that push purity down.

The implementation stores `rho` as a dense `2^n x 2^n` complex array. At n = 6 that is 64 x 64 (64 KiB), so the approach is limited to small systems; the cost of one gate is `O(4^k * 2^n)` for a `k`-qubit gate, which makes the full sweeps below take minutes, not hours.

## 3. Global depolarising channel

```
E_g(rho) = (1 - p_g) rho + p_g * I/d
```

The state is mixed with the maximally mixed state. `p_g` is the probability of that replacement **per application**, and in the primary experiment it is applied once after each complete Trotter step. It is therefore a *per-step* parameter, and it is **not** numerically comparable to a per-gate error probability. A single end-of-circuit application is available separately as a reference and is reported separately.

Validated: `p_g = 0` reproduces ideal evolution, `p_g = 1` produces `I/d` exactly, and trace, Hermiticity and positivity survive every application.

## 4. Single-qubit depolarising channel

```
E_1(rho) = (1 - p_1) rho + (p_1/3) (X rho X + Y rho Y + Z rho Z)
```

Applied to the target qubit of an RX or RZ gate; all other qubits are untouched. `p_1` is the total probability that a non-identity Pauli is applied, so each of X, Y and Z has probability `p_1/3` and the identity (probability `1 - p_1`) leaves the state alone. RZ can be configured separately, including as noiseless, because virtual-Z rotations are free on some architectures; the default model gives RX and RZ the same `p_1`.

## 5. Two-qubit depolarising channel

```
E_2(rho) = (1 - p_2) rho + (p_2/15) sum_{P != II} P rho P
```

`P` runs over the 15 non-identity elements of `{I,X,Y,Z}^{tensor 2}` acting on the control and target of a CNOT, and `p_2` is the total probability of a non-identity two-qubit Pauli.

**Implementation and how it is checked.** Summing `P rho P` over the *whole* Pauli group of a `k`-qubit subsystem has the closed form `2^k * Tr_k(rho) (x) I_k`, so the sum over non-identity Paulis is that expression minus `rho`. The fast path uses one partial trace instead of 3 or 15 matrix products; the tests also evaluate the sum term by term from the definition and require agreement to machine precision (largest difference measured: 5.55e-17).

## 6. How noise is applied to individual gates

The circuit is **not** rebuilt for this phase. The ordered gate list from `circuit_cost.build_circuit` — the same CNOT / RZ / RX sequence used in Phases 1 and 2, including merged second-order ZZ layers and half-angle rotations — is walked from first to last, and after each gate the corresponding channel is applied to that gate's own qubits:

```
RX(q, theta) :  rho <- RX rho RX^dagger   then  E_1 on qubit q
RZ(q, theta) :  rho <- RZ rho RZ^dagger   then  E_1 on qubit q
CNOT(c, t)   :  rho <- CX rho CX^dagger   then  E_2 on qubits (c, t)
```

The conventions are inherited rather than re-derived, and verified: the ideal action of the whole gate walk reproduces `trotter.trotter_unitary` to 1.8e-15 (spectral norm) and the gate counts reproduce Phase 1 exactly (0 difference on every row).

**Step boundary for the global model.** Because the second-order half layers of neighbouring steps are fused, one Trotter step does not end at a fixed offset in the gate list. The convention used here is: a step ends after the X block that closes it (order 1: after the ZZ block), and the final half-angle ZZ layer is grouped with the last step. That gives exactly `r` global channels for `r` steps; the grouping is checked against the gate list length on every run.

This is a documented convention, not a hardware model. It does not include calibrated error rates, T1/T2 relaxation, readout error, crosstalk, connectivity, routing, dynamical decoupling, mitigation or error correction (see §14).

## 7. Experimental configuration

| Parameter | Value |
|---|---|
| Hamiltonian | `H = -J sum Z_iZ_{i+1} - h sum X_i` (unchanged since Phase 1) |
| Couplings | J = 1.0, h = 1.0 |
| Evolution time | t = 1.0 (fixed for every step count) |
| Initial state | `|00...0>` (the Phase 1 convention) |
| System sizes | n = 4, n = 5, n = 6 |
| Trotter orders | 1 (Lie-Trotter) and 2 (Strang) |
| Step counts | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512 |
| Gate-level noise levels | 0.0001, 0.0005, 0.001, 0.005, 0.01 |
| Noise convention | p1 = p2 at each level unless stated |
| Reference level for figures | 0.001 |
| Exact reference | `exp(-iHt)|00...0>` |
| Trotter reference | `U_trotter|00...0>` from `trotter.trotter_unitary` |
| Randomness | none — every run is deterministic |

### 7.1 Validation performed before any noise experiment

| Check | Result |
|---|---|
| `rho = |psi><psi|` construction | tested |
| Trace, Hermiticity, positivity | tested on every simulated state |
| Density-matrix evolution vs state vector | 3.6e-16 max difference |
| Zero-noise circuit vs `trotter_unitary` | 1.8e-15 spectral norm |
| Gate counts vs Phase 1 | 0 maximum difference |
| Channel fast path vs explicit Pauli sum | 5.6e-17 maximum difference |
| Purity preserved by ideal evolution | 4.8e-14 drift |

## 8. System sizes and step ranges

The step grid is `1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512`: dense between 1 and 16, where the low-noise optimum lies, and coarse afterwards where only the noise tail matters. The grid was fixed after timing the simulator, not after seeing which point would win.

| n | order | step counts | r_min | r_max | gates at r_max | CNOT at r_max | depth at r_max |
|---|---|---|---|---|---|---|---|

## 9. Noise parameter ranges

Gate-level sweeps use `p1 = p2` in {0.0001, 0.0005, 0.001, 0.005, 0.01}, and the sensitivity grid crosses `p1` and `p2` independently over the same set. Global sweeps use `p_g` over the same set, applied once per Trotter step.

> These are illustrative simulation parameters chosen to span weak to strong noise. They are **not** measured hardware error rates and must not be read as device specifications or as IBM/other hardware performance.

## 10. First-order versus second-order results

### 10.1 Equal step count (the uncontrolled comparison)

At `r` equal for both formulas and gate-level noise `p1 = p2 = 0.001`, for n = 4. The second-order circuit is always slightly larger at equal `r`, so this is the comparison that flatters the first-order formula.

| r | CNOT (1st) | CNOT (2nd) | ideal err 1st | ideal err 2nd | noisy err 1st | noisy err 2nd |
|---|---|---|---|---|---|---|
| 1 | 6 | 12 | 0.942 | 0.813 | 0.942 | 0.814 |
| 2 | 12 | 18 | 0.157 | 0.048 | 0.174 | 0.074 |
| 3 | 18 | 24 | 0.06 | 0.008 | 0.089 | 0.046 |
| 4 | 24 | 30 | 0.032 | 0.002 | 0.071 | 0.05 |
| 6 | 36 | 42 | 0.014 | 0.000476 | 0.073 | 0.067 |
| 8 | 48 | 54 | 0.008 | 0.000149 | 0.086 | 0.085 |
| 12 | 72 | 78 | 0.003 | 2.91e-05 | 0.118 | 0.122 |
| 16 | 96 | 102 | 0.002 | 9.19e-06 | 0.152 | 0.156 |

### 10.2 Matched CNOT budget

For every measured second-order point, the first-order circuit with the largest CNOT count that does **not exceed** the second-order budget is selected, so no first-order point is credited with more gates than the second-order one used. Budgets come from `circuit_cost`.

**n = 4**, `p1 = p2 = 0.001`: second order is more accurate in 5 of 14 matched-CNOT comparisons.

| r (2nd) | r (1st, matched) | CNOT 1st | CNOT 2nd | noisy err 1st | noisy err 2nd | fidelity gain (2nd − 1st) |
|---|---|---|---|---|---|---|
| 1 | 2 | 12 | 12 | 0.174 | 0.814 | -0.6398 |
| 2 | 3 | 18 | 18 | 0.089 | 0.074 | +0.0146 |
| 3 | 4 | 24 | 24 | 0.071 | 0.046 | +0.0257 |
| 4 | 4 | 24 | 30 | 0.071 | 0.05 | +0.0215 |
| 6 | 6 | 36 | 42 | 0.073 | 0.067 | +0.0058 |
| 8 | 8 | 48 | 54 | 0.086 | 0.085 | +0.0004 |
| 12 | 12 | 72 | 78 | 0.118 | 0.122 | -0.0032 |
| 16 | 16 | 96 | 102 | 0.152 | 0.156 | -0.0043 |
| 24 | 24 | 144 | 150 | 0.216 | 0.221 | -0.0047 |
| 32 | 32 | 192 | 198 | 0.275 | 0.279 | -0.0045 |
| 64 | 64 | 384 | 390 | 0.464 | 0.467 | -0.0033 |
| 128 | 128 | 768 | 774 | 0.688 | 0.689 | -0.0017 |
| 256 | 256 | 1536 | 1542 | 0.858 | 0.859 | -0.0005 |
| 512 | 512 | 3072 | 3078 | 0.925 | 0.925 | -0.0001 |

**n = 5**, `p1 = p2 = 0.001`: second order is more accurate in 4 of 14 matched-CNOT comparisons.

| r (2nd) | r (1st, matched) | CNOT 1st | CNOT 2nd | noisy err 1st | noisy err 2nd | fidelity gain (2nd − 1st) |
|---|---|---|---|---|---|---|
| 1 | 2 | 16 | 16 | 0.203 | 0.902 | -0.6994 |
| 2 | 3 | 24 | 24 | 0.106 | 0.1 | +0.0061 |
| 3 | 4 | 32 | 32 | 0.088 | 0.06 | +0.0280 |
| 4 | 4 | 32 | 40 | 0.088 | 0.065 | +0.0231 |
| 6 | 6 | 48 | 56 | 0.093 | 0.087 | +0.0057 |
| 8 | 8 | 64 | 72 | 0.11 | 0.11 | -0.0003 |
| 12 | 12 | 96 | 104 | 0.152 | 0.156 | -0.0044 |
| 16 | 16 | 128 | 136 | 0.194 | 0.199 | -0.0055 |
| 24 | 24 | 192 | 200 | 0.273 | 0.279 | -0.0058 |
| 32 | 32 | 256 | 264 | 0.343 | 0.349 | -0.0054 |
| 64 | 64 | 512 | 520 | 0.557 | 0.561 | -0.0036 |
| 128 | 128 | 1024 | 1032 | 0.781 | 0.782 | -0.0015 |
| 256 | 256 | 2048 | 2056 | 0.92 | 0.92 | -0.0003 |
| 512 | 512 | 4096 | 4104 | 0.962 | 0.962 | -0.0000 |

**n = 6**, `p1 = p2 = 0.001`: second order is more accurate in 4 of 14 matched-CNOT comparisons.

| r (2nd) | r (1st, matched) | CNOT 1st | CNOT 2nd | noisy err 1st | noisy err 2nd | fidelity gain (2nd − 1st) |
|---|---|---|---|---|---|---|
| 1 | 2 | 20 | 20 | 0.244 | 0.947 | -0.7032 |
| 2 | 3 | 30 | 30 | 0.13 | 0.126 | +0.0042 |
| 3 | 4 | 40 | 40 | 0.108 | 0.075 | +0.0338 |
| 4 | 4 | 40 | 50 | 0.108 | 0.08 | +0.0283 |
| 6 | 6 | 60 | 70 | 0.114 | 0.107 | +0.0072 |
| 8 | 8 | 80 | 90 | 0.135 | 0.135 | -0.0002 |
| 12 | 12 | 120 | 130 | 0.184 | 0.189 | -0.0052 |
| 16 | 16 | 160 | 170 | 0.234 | 0.24 | -0.0065 |
| 24 | 24 | 240 | 250 | 0.325 | 0.332 | -0.0067 |
| 32 | 32 | 320 | 330 | 0.405 | 0.412 | -0.0061 |
| 64 | 64 | 640 | 650 | 0.635 | 0.638 | -0.0037 |
| 128 | 128 | 1280 | 1290 | 0.846 | 0.847 | -0.0013 |
| 256 | 256 | 2560 | 2570 | 0.954 | 0.955 | -0.0002 |
| 512 | 512 | 5120 | 5130 | 0.981 | 0.981 | -0.0000 |

### 10.3 Matched total gate budget

**n = 4**: second order is more accurate in 6 of 14 matched-total-gate comparisons (same selection rule).

**n = 5**: second order is more accurate in 5 of 14 matched-total-gate comparisons (same selection rule).

**n = 6**: second order is more accurate in 5 of 14 matched-total-gate comparisons (same selection rule).

## 11. Global versus gate-level noise

The global model has one parameter per Trotter step and ignores how many gates a step contains. The gate-level model charges every gate separately, so its accumulated error scales with `p1 * (#RX + #RZ) + p2 * #CNOT`, which is exactly the resource count of Phases 1 and 2.

The two models are therefore **not** the same noise at different strengths: at equal numerical parameter, gate-level noise is far stronger for these circuits, because a single Trotter step contains `O(n)` gates while the global channel is applied once. Read the numbers below as two different models, not as a calibrated comparison of hardware scenarios.

| n | order | r_opt global | min total err global | noise err global | r_opt gate | min total err gate | noise err gate |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 12 (of 1–512) | 0.015 | 0.011 | 4 | 0.071 | 0.041 |
| 4 | 2 | 6 (of 1–512) | 0.006 | 0.006 | 3 | 0.046 | 0.038 |
| 5 | 1 | 12 (of 1–512) | 0.016 | 0.012 | 4 | 0.088 | 0.053 |
| 5 | 2 | 6 (of 1–512) | 0.006 | 0.006 | 3 | 0.06 | 0.049 |
| 6 | 1 | 12 (of 1–512) | 0.017 | 0.012 | 4 | 0.108 | 0.066 |
| 6 | 2 | 6 (of 1–512) | 0.007 | 0.006 | 3 | 0.075 | 0.061 |

## 12. Optimal Trotter step count

`r_opt` is the argmin of the measured total error over the tested step counts. 72 configurations were evaluated: 60 have an interior minimum and 12 have their minimum at an end of the grid (reported as a boundary result, not as an optimum).

| n | order | p1 = p2 | r_opt | status | F at r_opt | ideal err | noise err | CNOT | gates | depth |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 0 | 512 | at largest tested r | 1 | 1.84e-06 | 2.59e-13 | 3072 | 6656 | 3587 |
| 4 | 1 | 0.0001 | 12 | interior | 0.9844 | 0.003 | 0.012 | 72 | 156 | 87 |
| 4 | 1 | 0.0005 | 6 | interior | 0.9562 | 0.014 | 0.031 | 36 | 78 | 45 |
| 4 | 1 | 0.001 | 4 | interior | 0.9288 | 0.032 | 0.041 | 24 | 52 | 31 |
| 4 | 1 | 0.005 | 3 | interior | 0.8053 | 0.06 | 0.146 | 18 | 39 | 24 |
| 4 | 1 | 0.01 | 3 | interior | 0.6918 | 0.06 | 0.268 | 18 | 39 | 24 |
| 4 | 2 | 0 | 512 | at largest tested r | 1 | 8.55e-12 | 2.6e-13 | 3078 | 6665 | 3593 |
| 4 | 2 | 0.0001 | 4 | interior | 0.9927 | 0.002 | 0.005 | 30 | 61 | 37 |
| 4 | 2 | 0.0005 | 4 | interior | 0.9736 | 0.002 | 0.024 | 30 | 61 | 37 |
| 4 | 2 | 0.001 | 3 | interior | 0.9545 | 0.008 | 0.038 | 24 | 48 | 30 |
| 4 | 2 | 0.005 | 2 | interior | 0.8282 | 0.048 | 0.131 | 18 | 35 | 23 |
| 4 | 2 | 0.01 | 2 | interior | 0.722 | 0.048 | 0.244 | 18 | 35 | 23 |
| 5 | 1 | 0 | 512 | at largest tested r | 1 | 2.26e-06 | 3.4e-13 | 4096 | 8704 | 3590 |
| 5 | 1 | 0.0001 | 8 | interior | 0.9803 | 0.009 | 0.011 | 64 | 136 | 62 |
| 5 | 1 | 0.0005 | 6 | interior | 0.9447 | 0.016 | 0.04 | 48 | 102 | 48 |
| 5 | 1 | 0.001 | 4 | interior | 0.9119 | 0.037 | 0.053 | 32 | 68 | 34 |
| 5 | 1 | 0.005 | 3 | interior | 0.7595 | 0.069 | 0.186 | 24 | 51 | 27 |
| 5 | 1 | 0.01 | 2 | interior | 0.6224 | 0.181 | 0.246 | 16 | 34 | 20 |
| 5 | 2 | 0 | 512 | at largest tested r | 1 | 1.18e-11 | 3.4e-13 | 4104 | 8716 | 3596 |
| 5 | 2 | 0.0001 | 6 | interior | 0.9903 | 0.00067 | 0.009 | 56 | 114 | 54 |
| 5 | 2 | 0.0005 | 4 | interior | 0.9652 | 0.004 | 0.031 | 40 | 80 | 40 |
| 5 | 2 | 0.001 | 3 | interior | 0.9399 | 0.012 | 0.049 | 32 | 63 | 33 |
| 5 | 2 | 0.005 | 2 | interior | 0.7768 | 0.066 | 0.17 | 24 | 46 | 26 |
| 5 | 2 | 0.01 | 2 | interior | 0.6484 | 0.066 | 0.308 | 24 | 46 | 26 |
| 6 | 1 | 0 | 512 | at largest tested r | 1 | 2.92e-06 | 4.15e-13 | 5120 | 10752 | 3593 |
| 6 | 1 | 0.0001 | 8 | interior | 0.9754 | 0.012 | 0.013 | 80 | 168 | 65 |
| 6 | 1 | 0.0005 | 6 | interior | 0.9317 | 0.02 | 0.049 | 60 | 126 | 51 |
| 6 | 1 | 0.001 | 4 | interior | 0.8916 | 0.046 | 0.066 | 40 | 84 | 37 |
| 6 | 1 | 0.005 | 3 | interior | 0.711 | 0.085 | 0.225 | 30 | 63 | 30 |
| 6 | 1 | 0.01 | 2 | interior | 0.5562 | 0.217 | 0.296 | 20 | 42 | 23 |
| 6 | 2 | 0 | 512 | at largest tested r | 1 | 1.55e-11 | 4.16e-13 | 5130 | 10767 | 3599 |
| 6 | 2 | 0.0001 | 6 | interior | 0.988 | 0.00086 | 0.011 | 70 | 141 | 57 |
| 6 | 2 | 0.0005 | 4 | interior | 0.9569 | 0.005 | 0.039 | 50 | 99 | 43 |
| 6 | 2 | 0.001 | 3 | interior | 0.9255 | 0.015 | 0.061 | 40 | 78 | 36 |
| 6 | 2 | 0.005 | 2 | interior | 0.7283 | 0.085 | 0.206 | 30 | 57 | 29 |
| 6 | 2 | 0.01 | 2 | interior | 0.582 | 0.085 | 0.367 | 30 | 57 | 29 |

| n | order | p_g | r_opt | status | F at r_opt | ideal err | noise err |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.0001 | 24 | interior | 0.9969 | 0.000838 | 0.002 |
| 4 | 1 | 0.0005 | 12 | interior | 0.991 | 0.003 | 0.006 |
| 4 | 1 | 0.001 | 12 | interior | 0.9855 | 0.003 | 0.011 |
| 4 | 1 | 0.005 | 6 | interior | 0.9589 | 0.014 | 0.028 |
| 4 | 1 | 0.01 | 4 | interior | 0.9324 | 0.032 | 0.037 |
| 4 | 2 | 0.0001 | 8 | interior | 0.9991 | 0.000149 | 0.00075 |
| 4 | 2 | 0.0005 | 6 | interior | 0.9967 | 0.000476 | 0.003 |
| 4 | 2 | 0.001 | 6 | interior | 0.9939 | 0.000476 | 0.006 |
| 4 | 2 | 0.005 | 4 | interior | 0.9789 | 0.002 | 0.019 |
| 4 | 2 | 0.01 | 3 | interior | 0.9641 | 0.008 | 0.028 |
| 5 | 1 | 0.0001 | 24 | interior | 0.9967 | 0.001 | 0.002 |
| 5 | 1 | 0.0005 | 12 | interior | 0.9902 | 0.004 | 0.006 |
| 5 | 1 | 0.001 | 12 | interior | 0.9844 | 0.004 | 0.012 |
| 5 | 1 | 0.005 | 6 | interior | 0.9556 | 0.016 | 0.029 |
| 5 | 1 | 0.01 | 6 | interior | 0.9281 | 0.016 | 0.057 |
| 5 | 2 | 0.0001 | 8 | interior | 0.999 | 0.00021 | 0.000775 |
| 5 | 2 | 0.0005 | 6 | interior | 0.9964 | 0.00067 | 0.003 |
| 5 | 2 | 0.001 | 6 | interior | 0.9935 | 0.00067 | 0.006 |
| 5 | 2 | 0.005 | 4 | interior | 0.9773 | 0.004 | 0.019 |
| 5 | 2 | 0.01 | 3 | interior | 0.96 | 0.012 | 0.029 |
| 6 | 1 | 0.0001 | 24 | interior | 0.9963 | 0.001 | 0.002 |
| 6 | 1 | 0.0005 | 16 | interior | 0.9893 | 0.003 | 0.008 |
| 6 | 1 | 0.001 | 12 | interior | 0.9832 | 0.005 | 0.012 |
| 6 | 1 | 0.005 | 6 | interior | 0.9511 | 0.02 | 0.029 |
| 6 | 1 | 0.01 | 6 | interior | 0.9232 | 0.02 | 0.058 |
| 6 | 2 | 0.0001 | 8 | interior | 0.9989 | 0.000269 | 0.000787 |
| 6 | 2 | 0.0005 | 6 | interior | 0.9962 | 0.00086 | 0.003 |
| 6 | 2 | 0.001 | 6 | interior | 0.9933 | 0.00086 | 0.006 |
| 6 | 2 | 0.005 | 4 | interior | 0.976 | 0.005 | 0.02 |
| 6 | 2 | 0.01 | 4 | interior | 0.9569 | 0.005 | 0.039 |

## 13. Fidelity, purity and the three error measures

Three quantities are recorded for every configuration and they are not interchangeable:

```
ideal_trotter_error = 1 - F(rho_trot,  psi_exact)
noise_induced_error = 1 - F(rho_noisy, rho_trot)
total_error         = 1 - F(rho_noisy, psi_exact)
```

The total is **not** the sum of the other two. Measured over 504 gate-level configurations, the largest discrepancy between `total_error` and `ideal + noise` is 0.242. The two contributions add only in the small-error limit; fidelity is not linear, and when both errors are large the sum overcounts.

Noise can occasionally raise the fidelity with respect to a particular reference by chance, so the measured curves are reported as they are, not forced to be monotone. Purity is reported separately: it isolates how much the state has been mixed and says nothing about whether it is the *right* state.

## 14. Limitations

This noise model deliberately stops short of a hardware description. It does **not** include:

* hardware-calibrated error rates (the `p` values are simulation parameters),
* T1 relaxation or T2 dephasing, and no amplitude/phase damping channel,
* readout error, SPAM, or measurement in the reported metrics,
* crosstalk, leakage to non-computational states, or drift,
* device connectivity, routing or SWAP overhead (the chain is assumed fully connected for nearest-neighbour gates),
* dynamical decoupling, gate scheduling/parallelism effects, or pulse-level control,
* error mitigation, post-selection, or quantum error correction.

Correlations between errors are absent: every channel is applied independently, which is optimistic compared with real hardware where a miscalibration affects a whole gate class at once. Density matrices restrict the study to `n <= 6` qubits at these step counts.

The results must not be read as a prediction for any real quantum processor.

## 15. Reproducibility

```bash
python -m noise.run_phase3              # all experiments, figures, report
python -m noise.run_phase3 --no-plots   # tables and report only
python -m noise.run_phase3 --quick      # small grid, for a smoke test
python -m pytest tests/ -q              # full suite (Phases 1-3)
```

Every run is deterministic: no random numbers are used anywhere in Phase 3, so repeated runs produce identical CSV files. Outputs:

```
results/phase3/
  global_noise_results.csv
  gate_noise_results.csv
  noise_sensitivity.csv
  optimal_r.csv
  phase3_summary.json
  phase3_report.md
  configs/experiment_config.json
  figures/fig01...fig12
```

Phase 1 and Phase 2 outputs are never written by this phase; a regression test hashes them before and after a full Phase 3 run.

## 16. Conclusions

* **An interior optimum exists.** 60 of 72 configurations have their minimum total error at a step count strictly inside the tested grid, with `r_opt` between 2 and 24. That is the central result of Phase 3: beyond a certain point, adding Trotter steps makes the simulation *worse*, not better.
* **12 configurations bottom out at the largest tested step count (r = 512): 6 end-of-circuit global (one channel, no r dependence), 6 noise-free.** Those cases have no per-step or per-gate noise charge that grows with the circuit, so more steps are simply better and no crossover exists; reporting r = the largest tested value is the honest result. No gate-level or per-step global configuration has a boundary minimum, so the crossover is real wherever noise is charged per step or per gate.
* **Gate-level noise:** the optimal step count moves down as noise grows (0.0001 -> r=4-12, 0.0005 -> r=4-6, 0.001 -> r=3-4, 0.005 -> r=2-3, 0.01 -> r=2-3). Fewer, larger steps mean fewer noisy operations, so the useful circuit depth shrinks with the error rate.
* **Global noise:** the optimal step count moves down as noise grows (0.0001 -> r=8-24, 0.0005 -> r=6-16, 0.001 -> r=6-12, 0.005 -> r=4-6, 0.01 -> r=3-6). Fewer, larger steps mean fewer noisy operations, so the useful circuit depth shrinks with the error rate.
* **n = 4, order 1, p = 0.001:** the best measured point is r = 4 with total error 0.071 (24 CNOTs, 52 gates), while r = 1 is 13.23x worse despite using 0.25x the CNOTs and 0.25x the gates. Its formula error is still 0.94 at that point - the gates, not the formula, are what fails.
* **n = 4, order 2, p = 0.001:** the best measured point is r = 3 with total error 0.046 (24 CNOTs, 48 gates), while r = 512 is 20.31x worse despite using 128.25x the CNOTs and 138.85x the gates. Its formula error is still 8.6e-12 at that point - the gates, not the formula, are what fails.
* **n = 6, order 1, p = 0.001:** the best measured point is r = 4 with total error 0.108 (40 CNOTs, 84 gates), while r = 512 is 9.05x worse despite using 128x the CNOTs and 128x the gates. Its formula error is still 2.9e-06 at that point - the gates, not the formula, are what fails.
* **n = 6, order 2, p = 0.001:** the best measured point is r = 3 with total error 0.075 (40 CNOTs, 78 gates), while r = 512 is 13.16x worse despite using 128.25x the CNOTs and 138.04x the gates. Its formula error is still 1.6e-11 at that point - the gates, not the formula, are what fails.
* **Second order's advantage is bounded by noise, not universal.** At p = 0.001, second order has the lower total error in 13 of 42 matched-CNOT comparisons (the §10.2 rule; both circuits always carry the same noise settings). All 13 of the wins sit at r <= 8. Two different effects produce the losses. In 9 of the 42 pairs the smaller per-step overhead (`2(n-1)` CNOTs) lets first order afford one extra Trotter step at the same budget, which is decisive at small r. In the rest the two circuits have the same r and sit above the crossover, where the formulas no longer separate and the marginally smaller first-order circuit accumulates slightly less noise. The win count therefore falls monotonically with the error rate: 38/42, 24/42, 15/42, 13/42, 6/42, 6/42 for p = 0 ... 0.01 (102/252 pooled), against 102/105 for the same rule in the noise-free Phase 2 analysis. Phase 2's conclusion is bounded rather than overturned: second order buys accuracy more cheaply *only while the formula still dominates the error*, which is at or below the optimal step count this phase recommends.
* **Global and gate-level noise give different `r_opt`.** The global model charges one parameter per step regardless of circuit size, so its optimum sits where the formula error crosses `p_g`-per-step accumulation; the gate-level model charges every gate, so its optimum depends on `n`, on the order, and on which gate class is noisy. §11 records both.

### Figures

**`figures/fig01_fidelity_vs_steps.png`** — Fidelity against exact evolution for the ideal and gate-level noisy circuit, both Trotter orders, at the reference noise level.

**`figures/fig02_trotter_error_vs_steps.png`** — Noise-free Trotter error only: 1st order falls as r^-1, 2nd order as r^-2.

**`figures/fig03_noise_error_vs_steps.png`** — Error contributed by the channels alone (noisy vs ideal Trotter). Grows with circuit length, roughly linearly in the gate count.

**`figures/fig04_total_error_vs_steps.png`** — Total error at several noise strengths: the descending formula error and the ascending noise error cross, producing a minimum.

**`figures/fig05_fidelity_vs_cnot.png`** — The same data against CNOT budget, the resource Phase 1 counted.

**`figures/fig06_fidelity_vs_total_gates.png`** — Fidelity against total elementary gate count.

**`figures/fig07_order_comparison_gate_noise.png`** — First vs second order at matched CNOT cost, with the noise-free reference of each order shown dotted.

**`figures/fig08_global_vs_gate_noise.png`** — Global (one channel per step) vs gate-level (one channel per gate) noise at comparable strength parameters.

**`figures/fig09_optimal_r_vs_noise.png`** — Measured argmin of the total error against noise strength; hollow markers are boundary minima.

**`figures/fig10_sensitivity_heatmap.png`** — Total error over the (p1, p2) grid, plus the two single-parameter slices that separate single- and two-qubit gate contributions.

**`figures/fig11_purity_vs_steps.png`** — Purity against circuit length, with the maximally mixed value 1/2^n.

**`figures/fig12_total_error_vs_depth.png`** — Total error against the conservative logical depth of Phase 1.
