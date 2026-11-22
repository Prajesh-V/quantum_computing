# Trotterisation of a Spin Chain

**How much error does a Trotterised (split-operator) quantum simulation make, and how does that error shrink as you take more, smaller steps?**

This repository contains a self-contained numerical study of product-formula ("Trotter") errors for a
transverse-field Ising chain and a Heisenberg chain. Two independent error measures are used: the
operator-norm distance between the exact and Trotterised propagators, and the infidelity of an evolved
state. Textbooks predict a first-order error falling like `1/r` and a second-order error falling like
`1/r^2` in the number of steps `r` (and `1/r^2`, `1/r^4` for infidelity). This project measures whether
that actually happens, how tight the standard commutator bound is, how the error depends on chain length,
what finally stops the error from falling, and how many steps a fixed accuracy costs.

Everything is written from scratch with `numpy`, `scipy` and `matplotlib`. There are no datasets, no
downloads, no quantum-computing SDK, and no hidden dependencies. One command reproduces every number and
every figure in this file.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Objectives](#2-objectives)
3. [Requirements](#3-requirements)
4. [How to download the project](#4-how-to-download-the-project)
5. [How to run it](#5-how-to-run-it)
6. [Repository layout](#6-repository-layout)
7. [Methodology](#7-methodology)
8. [Command-line reference](#8-command-line-reference)
9. [Outputs](#9-outputs)
10. [Results](#10-results)
11. [Verification performed](#11-verification-performed)
12. [Honest caveats](#12-honest-caveats)
13. [Troubleshooting and FAQ](#13-troubleshooting-and-faq)
14. [Limitations and possible extensions](#14-limitations-and-possible-extensions)
15. [Where to change the code](#15-where-to-change-the-code)
16. [Circuit-level resource estimation (Phase 1)](#16-circuit-level-resource-estimation-phase-1)
17. [Accuracy versus circuit resources (Phase 2)](#17-accuracy-versus-circuit-resources-phase-2)

---

## 1. What this project is

### 1.1 The physics, in one paragraph

A quantum system is described by a **Hamiltonian** `H`, the rulebook that says how it evolves. To
simulate the system you need `U(t) = exp(-i H t)`, the "fast-forward" operator that takes a state to
time `t`. For a real material `H` is too complicated to exponentiate directly, so the standard trick is
to split it into pieces that *are* easy to exponentiate, apply them one after another, and repeat that
little cycle `r` times. That is a **Trotter / product formula**. Each cycle is a small approximation;
over the whole simulation those little mistakes accumulate, and the question is how fast the total
error goes to zero as `r` grows.

### 1.2 Why bother

This is the core of every digital quantum simulation algorithm, from chemistry to condensed matter: you
tune the number of steps until the error is acceptable. Knowing the exponent tells you how the cost
grows; knowing the prefactor tells you the actual step count; knowing what breaks the trend tells you
where the whole approach stops paying off. In this project the last point turns out to be the most
interesting result (a floating-point floor, see [§10.5](#105-plot-f--the-round-off-floor)).

### 1.3 The two models

**Transverse-field Ising chain (TFIM)**, open boundaries, `J = 1`, `h = 1`, `n` spins:

```
H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i
  = A + B
A = -J * sum_i Z_i Z_{i+1}      all terms commute with each other
B = -h * sum_i X_i              all terms commute with each other
```

`A` and `B` each consist of commuting terms, so `exp(-iAt)` and `exp(-iBt)` are cheap, but `A` does not
commute with `B`, so the split is not exact.

**Heisenberg chain** (optional second model, `n = 4`):

```
H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
A = sum_i (X_i X_{i+1} + Y_i Y_{i+1})     B = sum_i Z_i Z_{i+1}
```

Arrows `X, Y, Z` are the Pauli matrices; `Z_i Z_{i+1}` is a two-spin interaction, `X_i` a transverse
field. Each spin is a qubit, so `n` spins means a `2^n`-dimensional vector and a `2^n x 2^n` matrix.

---

## 2. Objectives

Each objective below is implemented in `trotter.py`, has an explicit pass condition, and is checked
automatically or reported in the console. Status is what the committed run produced.

| # | Objective | Pass condition | Result obtained | Status |
|---|-----------|----------------|-----------------|--------|
| O1 | Build the TFIM Hamiltonian from Pauli matrices with Kronecker products, no library shortcuts | `H` Hermitian to machine precision | `||H - H^dag||_2 = 0` exactly | met |
| O2 | Implement the exact propagator `exp(-iHt)` | unitary to machine precision | `||U^dag U - 1||_2 = 8.8e-16` | met |
| O3 | Implement first-order (Lie-Trotter) and second-order (Strang) product formulas | both run for every `n` | both implemented and swept | met |
| O4 | Measure error two independent ways: operator norm and state infidelity | both recorded per `r` | 220 rows, 2 measures | met |
| O5 | Recover the theoretical exponents `-1, -2` (operator) and `-2, -4` (infidelity) | fitted slope within 0.05 of theory | `-1.01, -2.01, -2.03, -4.03` | met |
| O6 | Compare the measured error against the rigorous commutator bound `(t^2/2r)||[A,B]||` | bound never violated | ratio 2.42 to 3.24, bound always above | met |
| O7 | Find what limits the error at very large `r` | locate a minimum | `2.50e-11` at `r = 2.2e5`, then rises | met |
| O8 | Measure how the step count for fixed accuracy grows with chain length | fit an exponent in `n` | `r ~ n^1.03` (1st), `n^0.55` (2nd) | met |
| O9 | Repeat the whole study on a second, non-commuting model | same exponents recovered | Heisenberg: `-1.04, -2.05, -2.16, -4.09` | met |
| O10 | Write raw numbers to CSV and figures to PNG, reproducibly | files appear on a clean run | 3 CSVs, 7 PNGs | met |

Scope explicitly **not** attempted: real quantum hardware, chains beyond 8 spins, noisy channels, or
advanced (recursive / random-product) error bounds. See [§14](#14-limitations-and-possible-extensions).

---

## 3. Requirements

* **Python 3.9 or newer** (developed and tested on Python 3.11).
* Three libraries, nothing else:

```bash
pip install numpy scipy matplotlib
```

Developed against `numpy 2.4.6`, `scipy 1.17.1`, `matplotlib 3.11.2`.

No GPU, no network access during a run, no display needed (matplotlib is forced onto the non-interactive
`Agg` backend before `pyplot` is imported, so it works over SSH and in CI).

Memory: the largest object is a `256 x 256` complex matrix (`n = 8`), i.e. about 1 MB. Any laptop works.

---

## 4. How to download the project

### Option A - clone with git (recommended, easiest to update)

```bash
git clone https://github.com/anoushka24dev/quantum_computing.git
cd quantum_computing
```

### Option B - download a ZIP

Open <https://github.com/anoushka24dev/quantum_computing>, click the green **Code** button, choose
**Download ZIP**, then unzip it and open a terminal in the unzipped folder.

### Option C - if you already have this folder

Just open a terminal in it. Nothing has to be installed except the three libraries in §3.

### Setting up a clean environment (optional but recommended)

```bash
python -m venv .venv
# Windows (PowerShell):      .venv\Scripts\Activate.ps1
# Windows (cmd):             .venv\Scripts\activate.bat
# macOS / Linux:             source .venv/bin/activate
pip install numpy scipy matplotlib
```

Everything is then run as `python trotter.py` from the repository root. The script creates `results/`
itself, so a fresh clone needs no directory setup.

---

## 5. How to run it

The whole project is one command:

```bash
python trotter.py
```

It will:

1. run the sanity checks and **abort with a clear message** if they fail (§11);
2. sweep `r` for `n = 4, 5, 6` with both orders (plots A, B, C);
3. compare against the commutator bound (plot D);
4. repeat the study on the Heisenberg chain (plot E);
5. push `r` to `1e8` to find the round-off floor (plot F);
6. measure the step count needed for a target accuracy for `n = 2...7` (plot G);
7. write the raw numbers to `results.csv`, `results/floor_study.csv`, `results/trotter_number.csv`;
8. print the slope table and a verification summary to the console.

To add the **circuit-level cost** of the same evolution (Phase 1, §16):

```bash
python circuit_cost.py        # verification, gate-count table, 6 more plots, ~15 s
python -m pytest tests/ -q    # unit tests for the estimator (~2 s, 48 tests)
```

`circuit_cost.py` reuses `trotter.tfim_terms` and `trotter.sweep` and does not modify the numerical
error calculation in any way.

Phase 2 analyses those resources against the measured errors:

```bash
python -m analysis.run_phase2                  # 10 figures, 4 data files, report (~5 s)
python -m analysis.run_phase2 --write-supplement  # re-measure r = 240...30000 first
```

It only reads `results/circuit_resources.csv` and writes everything into `results/phase2/`.

**Runtime: about 30 seconds** on a normal laptop (up to a minute on a slow machine). The `n`-scaling
study and the round-off study dominate. Nothing is parallelised; results are deterministic, so two runs
produce identical CSVs.

Expected console output ends with a table like the one in [§10.1](#101-the-slope-table).

---

## 6. Repository layout

```
quantum_computing/
├── trotter.py                  # the physics study: model, methods, experiments, plots (794 lines)
├── circuit_cost.py             # Phase 1: gate decomposition + resource estimation (~700 lines)
├── analysis/                   # Phase 2: resource-versus-accuracy analysis
│   ├── resource_analysis.py    # loading, schema/value validation, audit
│   ├── comparison.py           # equal steps, equal CNOT/gate budgets, efficiency
│   ├── scaling.py              # analytic formulas vs measured, exponents
│   ├── accuracy_targets.py     # minimum resources for 1e-1 ... 1e-4
│   ├── plotting.py             # the ten Phase 2 figures
│   ├── report.py               # phase2_report.md + phase2_summary.json
│   └── run_phase2.py           # driver
├── conftest.py                 # makes the project importable for pytest
├── tests/
│   ├── test_circuit_cost.py    # 48 tests: counts, validation, scheduling, reports, regression
│   └── test_phase2.py          # 30 tests: loading, comparisons, targets, scaling, floor
├── README.md                   # this file
├── results.csv                 # main data table, 220 rows (+ header), 7 columns
└── results/
    ├── plot_A_operator_error.png
    ├── plot_B_state_infidelity.png
    ├── plot_C_chain_length.png
    ├── plot_D_commutator_bound.png
    ├── plot_E_heisenberg.png
    ├── plot_F_roundoff_floor.png
    ├── plot_G_trotter_number.png
    ├── plot_H_error_vs_cnot.png        # circuit-cost figures, see #16
    ├── plot_I_error_vs_gates.png
    ├── plot_J_error_vs_depth.png
    ├── plot_K_cnot_vs_qubits.png
    ├── plot_L_gates_vs_steps.png
    ├── plot_M_depth_vs_steps.png
    ├── floor_study.csv         # 108 rows: error vs r up to 1e8
    ├── trotter_number.csv      # 12 rows: r(eps) for n = 2...7, both orders
    ├── circuit_resources.csv   # 132 rows: error + CNOT/RX/RZ/depth per (n, r, order)
    ├── circuit_reports.json    # 16 structured resource reports
    └── phase2/                 # Phase 2 outputs, never mixed with Phase 1
        ├── error_vs_cnot.png ... depth_vs_qubits.png   # 10 figures
        ├── minimum_resources_for_accuracy.csv          # 48 rows
        ├── resource_efficiency.csv                     # 216 rows
        ├── supplementary_sweep.csv                     # 84 extra measurements, r>200
        ├── phase2_summary.json
        └── phase2_report.md
```

`trotter.py` is deliberately a single file (no package, no imports beyond the three libraries) so that it
can be read top to bottom. Its sections are marked with banner comments: constants and Pauli matrices,
Kronecker helpers, model builders, exact evolution, error measures, the sweep and the fit, the three
studies, plotting, and `main()`.

The committed PNGs and CSVs are the output of the committed code, so a collaborator can read every result
without running anything.

---

## 7. Methodology

### 7.1 Discretising time

For `r` steps of size `dt = t/r`, with `A` and `B` the two commute-inside / do-not-commute-across halves
of `H`:

```
1st order (Lie-Trotter):   step = exp(-i A dt) @ exp(-i B dt)
2nd order (Strang):        step = exp(-i A dt/2) @ exp(-i B dt) @ exp(-i A dt/2)

U_trot = matrix_power(step, r)          U_exact = expm(-i H t)
```

Each exponential is computed with `scipy.linalg.expm`, i.e. exactly as a matrix exponential - no
approximation inside the building block. The error measured is therefore purely the splitting error.

The **second order costs the same matrices as the first order** but is one order more accurate, which is
why it is the default choice in real simulations and why plot A shows roughly an order of magnitude
improvement at every `r`.

### 7.2 Why the expected exponents are `1, 2, 2, 4`

`A` and `B` fail to commute (rotating a phone about two different axes in sequence does not land where
you expect), so one step is wrong by `O(dt^2)` for Lie-Trotter and `O(dt^3)` for Strang. `r` steps of
size `t/r` accumulate to `O(dt) = O(1/r)` and `O(dt^2) = O(1/r^2)` globally.

The state infidelity is `1 - |<psi_exact|psi_trot>|^2`, which is *quadratic* in the state-vector error,
so the same errors appear as `1/r^2` and `1/r^4`. Measuring both is a consistency check on the whole
implementation: if the operator exponents are right but the infidelity ones are not, the state evolution
or the inner product is wrong.

### 7.3 The two error measures

1. **Operator-norm error** `||U_exact - U_trot||_2` - the largest singular value, i.e. the worst-case
   error over *all* states at once. This is the harsh measure: it is what a "worst student" theorem uses.
2. **State infidelity** `1 - |<psi_exact|psi_trot>|^2` with `|psi> = U|00...0>`. This is what you would
   actually measure on one prepared state. It is smaller and reaches the noise floor much sooner.

Using both is deliberate: they must agree on the scaling but they disagree on the prefactor, and that
difference is itself informative.

### 7.4 Sweeping and fitting

`r` values are **log-spaced integers** from 1 to 200, generated with `np.geomspace(1, 200, 26)` and
rounded; because neighbouring values collide after rounding (1, 2, 3 all appear twice), the 26 requests
produce **22 distinct step counts**:

```
1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 19, 24, 30, 37, 45, 56, 69, 86, 106, 131, 162, 200
```

The slope is a least-squares fit of `log(error)` against `log(r)` on the asymptotic part of the curve:
`r = 1` is dropped (pre-asymptotic), and points with `error <= 1e-13` are dropped (round-off dominated).
Points used in the fit are drawn as **hollow** markers; points shown but excluded are filled.

The sweep stops at `r = 200` on purpose: plot F shows the curve turns over around `r ~ 2e5`, so the whole
A-E range is safely on the clean descending branch. The `1e-13` cut therefore only ever removes `r = 1`
(see [§12](#12-honest-caveats), caveat 2).

### 7.5 The three studies

* **Main sweep (plots A-E, `results.csv`).** `n = 4, 5, 6` TFIM plus `n = 4` Heisenberg, both orders, two
  measures. 220 rows that everything else is checked against.
* **Round-off study (plot F, `results/floor_study.csv`).** `r` from 1 to `1e8`, step-by-step application
  of the product formula (no `matrix_power` shortcut) so that round-off accumulates as it would in a real
  sequential simulation. 108 points. This answers "what stops the error falling?".
* **Trotter-number study (plot G, `results/trotter_number.csv`).** For each `n = 2...7`, the smallest `r`
  whose operator error is `<= 1e-3`, then a power-law fit in `n`. This is the practically useful question:
  not "what is the exponent" but "how many steps do I actually have to take?".

`n = 8` is supported (`--scaling-nmax 8`) but costs about 45 s extra because `expm` at `dim = 256` takes
around 10 s per call. For that reason plot G exponentiates through a **cached eigendecomposition**
(`exp(-iMt) = V exp(-i lambda t) V^dag`, exact for Hermitian `M`). All other results, including every row
of `results.csv`, still use `scipy.linalg.expm`.

### 7.6 Implementation notes

* **Kronecker products by hand** (`kron_all`) so the spin ordering and boundary conditions are explicit
  and auditable, instead of hidden in a library convention.
* **Caching.** The exact propagator per `(n, model)` and the cached eigen-decomposition are computed once
  and reused, which is what makes a 220-row sweep plus a 108-point study run in 30 s.
* **Determinism.** No randomness anywhere, so every run reproduces the committed CSVs exactly.
* **Console output is ASCII-only** so it prints on a Windows `cp1252` terminal without `UnicodeEncodeError`.
* **`matplotlib.use("Agg")` is set before importing `pyplot`**, so no window is ever opened.

---

## 8. Command-line reference

All flags are optional; the defaults reproduce the committed results.

| Flag | Default | Effect |
|------|---------|--------|
| `--t` | `1.0` | Evolution time `t` |
| `--rmax` | `200` | Largest `r` in the main sweep |
| `--npoints` | `26` | Number of log-spaced `r` values requested (rounds to 22 distinct) |
| `--outdir` | `results` | Directory for the PNGs and the two study CSVs |
| `--csv` | `results.csv` | Path of the main results table |
| `--no-heisenberg` | off | Skip the optional Heisenberg variant (plot E) |
| `--floor-rmax` | `1e8` | Largest `r` in the round-off study |
| `--no-floor-study` | off | Skip the round-off study (plot F); big runtime saving |
| `--eps` | `1e-3` | Target accuracy for the Trotter-number study |
| `--scaling-nmax` | `7` | Largest chain length in the Trotter-number study (`8` adds ~45 s) |
| `--scaling-rmax` | `200` | Largest `r` in the Trotter-number study |
| `--scaling-npoints` | `14` | Number of log-spaced `r` values in that study |
| `--no-scaling` | off | Skip the Trotter-number study (plot G) |

Typical uses:

```bash
python trotter.py                              # full run, ~30 s
python trotter.py --no-heisenberg              # Ising only
python trotter.py --rmax 400 --npoints 40       # finer, wider sweep of r
python trotter.py --t 2.0                       # longer evolution time
python trotter.py --outdir figures --csv out/table.csv
python trotter.py --no-floor-study --no-scaling # fast smoke test
python trotter.py --eps 1e-4                    # stricter target for plot G
```

Note that changing `--rmax` changes the fitted slopes slightly, because the fit window changes.

---

## 9. Outputs

### 9.1 `results.csv` (project root)

| Column | Meaning |
|--------|---------|
| `model` | `TFIM` or `Heisenberg` |
| `n` | Number of spins |
| `order` | `1` (Lie-Trotter) or `2` (Strang) |
| `r` | Number of steps |
| `initial_state` | `0000` = `|00...0>`, `0101` = Neel state `|0101...>` (Heisenberg only) |
| `operator_norm_error` | `||U_exact - U_trot||_2` |
| `state_infidelity` | `1 - |<psi_exact|psi_trot>|^2` |

220 data rows = `n = 4, 5, 6` x 2 orders x 22 step counts (TFIM) + `n = 4` x 2 orders x 22 (Heisenberg).

### 9.2 `results/floor_study.csv`

108 rows with columns `order, r, operator_norm_error, state_infidelity`, from the step-by-step run used for
plot F (`n = 4`, `r` up to `1e8`). Note that this file carries no `n` column: the study is fixed at `n = 4`.

### 9.3 `results/trotter_number.csv`

12 rows with columns `n, order, error_exponent, trotter_number, extrapolated` for `n = 2...7`.
`trotter_number` is the smallest `r` whose operator error is `<= 1e-3` (e.g. `703.6` for `n = 2`, first
order), and `extrapolated = 1` flags a value obtained by extrapolating past the measured range instead of
interpolating inside it.

### 9.4 The seven figures

**Plot A - `plot_A_operator_error.png`.** Operator-norm error vs `r`, `n = 4`, both orders, log-log.
Slopes `-1.01` and `-2.01`: exactly the textbook `1/r` and `1/r^2`, and Strang buys about an order of
magnitude at the same `r`.

**Plot B - `plot_B_state_infidelity.png`.** Same sweep for infidelity from `|0000>`. Slopes `-2.03` and
`-4.03`, twice the operator exponents, confirming that infidelity is quadratic in the state error. This
curve is noisier and hits its floor sooner because it probes a single state.

**Plot C - `plot_C_chain_length.png`.** Operator error for `n = 4, 5, 6` (solid = 1st order, dashed =
2nd). Six parallel lines: the **exponent does not depend on `n`, only the prefactor does**, and the error
grows with chain length (at `r = 200`, first order: `6.9e-3`, `8.8e-3`, `1.1e-2`; second order:
`2.8e-5`, `3.8e-5`, `4.6e-5`). The growth is sublinear because the Hamiltonian is a sum of local terms.

**Plot D - `plot_D_commutator_bound.png`.** Left: measured first-order error against the rigorous bound
`(t^2 / 2r) ||[A,B]||` with `||[A,B]||_2 = 8.944` for `n = 4`. Both fall like `1/r` and the bound sits
above the data at every `r`. Right: the ratio, settling at about `3`. So the bound is **valid but not
tight**: right exponent, prefactor roughly 3x too pessimistic, which is expected because it uses the norm
of a single commutator rather than the actual leading error term.

**Plot E - `plot_E_heisenberg.png`.** Heisenberg chain, operator error (`-1.04`, `-2.05`) and infidelity
(`-2.16`, `-4.09`) from the Neel state `|0101...>`. The exponents are model-independent, which is the point
of including a second model.

**Plot F - `plot_F_roundoff_floor.png`.** `r` pushed to `1e8`. Truncation error falls like `1/r^2` while
round-off accumulates like `eps_mach * r`; where they cross, the curve turns back up. Second order bottoms
out at **`2.50e-11` near `r = 2.2e5`**. First order is still falling at `r = 1e8` (`1.5e-8`). The `1e-13`
threshold is marked and is **never reached at any `r`** - the honest answer to "when does floating-point
round-off matter here?" is "three decades before your cutoff does". Slope on the clean branch `-2.00`,
versus `-1.22` if every point is fitted.

**Plot H - `plot_H_error_vs_cnot.png`.** Operator error vs CNOT count, `n = 4`, both orders. On this axis
the first-order curve falls like `CNOT^-1` and the second-order curve like `CNOT^-2`, so second order is
the better accuracy-per-CNOT purchase at every size measured.

**Plot I - `plot_I_error_vs_gates.png`.** Same data against total elementary gates (CNOT + RX + RZ) for
`n = 4, 5, 6`. Error falls faster than cost grows, so spending gates on `r` always pays.

**Plot J - `plot_J_error_vs_depth.png`.** Same data against logical circuit depth. Depth is smaller than
the gate count by the amount of parallelism the scheduler finds, so this axis reaches the same error with
fewer "time slots" than gates.

**Plot K - `plot_K_cnot_vs_qubits.png`.** CNOT count vs chain length at fixed `r = 200`, log-log: the count
is proportional to `n - 1` (one bond per extra qubit), i.e. asymptotically linear in `n`.

**Plot L - `plot_L_gates_vs_steps.png`.** Total gate count vs `r`: straight lines of slope 1, because the
circuit is `r` repetitions of a fixed step. The second-order line is parallel and only `3(n-1)` gates above
the first-order one.

**Plot M - `plot_M_depth_vs_steps.png`.** Circuit depth vs `r`, also linear; second order sits one extra
ZZ layer above first order.

**Plot G - `plot_G_trotter_number.png`.** Left: smallest `r` with operator error `<= 1e-3` vs `n`, log-log.
First order needs `r = 704, 1099, 1381, 1746, 2254, 2554` (`r ~ n^1.03`); second order needs
`r = 23.6, 29.2, 33.7, 39.0, 43.1, 46.9` (`r ~ n^0.55`). Hollow markers are extrapolated past `r = 200`
(all first-order points, since `1e-3` needs `r > 10^3`); second-order points are interpolated inside the
measured range. Right: the fitted exponent per `n`, flat at `-1.02` and `-2.02`.

### 9.5 The six circuit-resource figures

Produced by `python circuit_cost.py` from the same measured errors (see §16 for the formulas).

---

## 10. Results

### 10.1 The slope table

Printed by the script on every run:

```
        model   n  order  op-norm  infidel.
         TFIM   4      1    -1.01     -2.03
         TFIM   4      2    -2.01     -4.03
         TFIM   5      1    -1.02     -2.01
         TFIM   5      2    -2.01     -4.03
         TFIM   6      1    -1.01     -2.00
         TFIM   6      2    -2.01     -4.03
   Heisenberg   4      1    -1.04     -2.16
   Heisenberg   4      2    -2.05     -4.09
```

Theory: `-1, -2` for the operator norm and `-2, -4` for infidelity. Agreement is within `0.05` everywhere
except the Heisenberg infidelity, which is `0.09` off (see caveat 6).

### 10.2 The five headline results

1. **The textbook exponents are real.** `-1.01 / -2.01 / -2.03 / -4.03` for `n = 4`, and the same for
   `n = 5, 6` and for a completely different model.
2. **The rigorous bound is safe but ~3x pessimistic**, with the correct `1/r` scaling.
3. **The exponent is independent of system size**; only the prefactor grows, sublinearly in `n`.
4. **Floating-point round-off, not truncation, is the real wall**: `2.5e-11` at `r ~ 2.2e5`, and the
   `1e-13` "trusted" threshold is never reached.
5. **Cost for `1e-3` accuracy grows as `n^1.03` (first order) and `n^0.55` (second order)** - the practical
   selling point of Strang splitting.

### 10.3 What this is *not*

It is a verification and characterisation study: every exponent reported here was already known. The
contribution is a clean measurement, an honest bound comparison, and the round-off floor. No new physics
is claimed.

---

## 11. Verification performed

The script performs these checks and **aborts with a message** if one fails:

* `||H - H^dag||_2 = 0` exactly, and `||U_exact^dag U_exact - 1||_2 = 8.8e-16` for `n = 4`, `t = 1`
  (tolerance `1e-12`). A broken Hamiltonian or a bad `expm` would show up here immediately.
* The error decreases **strictly monotonically** in `r` for every `n` and both orders, by factors of
  `1.8e2` to `5.4e4` between `r = 1` and `r = 200`; the run aborts if the final error is not at least 10x
  smaller than the initial one.

Additional consistency checks reported in the console and visible in the figures:

* Fitted exponents agree with theory to `0.05`.
* The commutator bound is never violated at any measured `r`.
* Operator and infidelity exponents differ by exactly the factor of 2 predicted by the quadratic relation.
* The same exponents appear for a second, unrelated model.

For the circuit layer (§16), `python circuit_cost.py` additionally checks, and aborts on failure:

* **Gate decomposition.** The compiled gate list is multiplied out into a full `2^n x 2^n` matrix and
  compared with `trotter.trotter_unitary`: agreement to `1.1e-15` for both orders, with and without
  half-step merging. The two identities used for the decomposition are checked separately against
  `scipy.linalg.expm` to `1.8e-16`.
* **Gate counts.** Compiled counts are compared with closed-form expressions for `(n, r) =
  (2,1), (4,5), (6,10), (8,50)`; they match exactly.
* **Unit tests.** `python -m pytest tests/ -q` runs 48 tests (analytic counts, parameter rejection,
  scheduling, JSON/CSV export, decomposition, and a regression that recomputes a committed `results.csv`
  entry).

A useful way to read a failure: a slope of `0` means the sweep is not varying `r`; a slope of `-0.5`
usually means the fit window includes pre-asymptotic points; a *positive* slope at large `r` means you have
crossed the round-off floor and are fitting noise (plot F).

---

## 12. Honest caveats

1. **The `|00...0>` infidelity is useless for the Heisenberg chain.** `A = sum (XX + YY)` annihilates every
   all-up bond, so `exp(-iAt)|00...0> = |00...0>` and the Trotterised state differs by a phase only. The
   infidelity from `|0000>` is therefore `~1e-15` for every `r` and carries no information. Those rows are
   kept in `results.csv` (`initial_state = 0000`) to document the effect; the informative curve uses the
   **Neel state** `|0101...>`. This is a property of the initial state, not a bug.
2. **The `1e-13` cut never triggers.** An earlier version of this README claimed it protected finer sweeps.
   Plot F disproves it: the error stops falling at `~1e-11`, three decades above `1e-13`, because round-off
   grows like `eps_mach * r`. In practice the cut removes only `r = 1`; what protects the fits in plots A-E
   is that the sweep stops at `r = 200`. For a wide sweep the fit must be restricted to the descending
   branch instead (hence the hollow markers in plot F).
3. **Infidelity has no usable floor range in a log plot.** Being quadratic in the state error, it is
   clipped at exactly `0` for most `r > 1e5`, so it cannot be plotted that far out. Plot F uses the
   operator norm for this reason.
4. **Plot G does not use `scipy.linalg.expm`.** `expm` costs ~10 s per call at `dim = 256` (`n = 8`), which
   makes an `n`-scaling sweep impossible. Plot G therefore uses a cached eigendecomposition,
   `exp(-iMt) = V exp(-i lambda t) V^dag`, exact for Hermitian `M` and costing two matrix products.
   Every row of `results.csv` and plots A-E still come from `scipy.linalg.expm`.
5. **The first-order fit gives `-1.01`, not exactly `-1`.** The deviation is genuine pre-asymptotic
   curvature, not a fitting artefact; it is stable across `n` and across models.
6. **The Heisenberg infidelity slope (`-2.16`) is less clean** than its Ising counterpart (`-2.03`)
   because the Neel state has larger overlap with low-lying excitations, so the asymptotic regime is only
   entered at larger `r`.
7. **Only the operator-norm error is used** for plot C and the commutator-bound check; the infidelity is
   shown only for `n = 4` (plots B, E).
8. **Round-off scaling is a heuristic, not a theorem.** The `eps_mach * r` guide line in plot F is the
   expected order of magnitude, not a rigorous bound.
9. **The reported `n`-exponents are for fixed total time `t = 1`.** In an application where `t` grows with
   system size, the step-count scaling changes. See §14.

---

## 13. Troubleshooting and FAQ

**`ModuleNotFoundError: No module named 'numpy'`**
The libraries are not installed in the interpreter you are using: `pip install numpy scipy matplotlib`, or
activate the virtual environment first.

**Nothing happens / no PNGs appear**
Check the working directory: outputs go to `results/` relative to where you launched the script, and
`results.csv` to the repository root (override with `--outdir` / `--csv`).

**`UnicodeEncodeError` on Windows**
Only a problem if the code is edited to print non-ASCII characters; the console output is deliberately
plain ASCII.

**The run feels slow**
Skip the two expensive studies: `python trotter.py --no-floor-study --no-scaling`. This removes plots F
and G and brings the runtime down to a few seconds.

**The slopes changed after I edited a flag**
Expected: `--rmax` and `--npoints` change the fit window. Changing `--t` changes the prefactor and can move
the point where the fit window begins.

**`RuntimeWarning: invalid value encountered in log`**
A zero or negative error entered the fit, which only happens if the sweep is made so coarse that
`error <= 0` after clipping. Widen `--npoints` or reduce `--rmax`.

**Can I use this on a cluster?**
Yes. It is pure Python, single-threaded, no I/O beyond the outputs, and deterministic.

---

## 14. Limitations and possible extensions

Not implemented, each for a stated reason:

* **Real quantum hardware / a quantum SDK.** Everything here is exact classical linear algebra; simulating
  `8` spins classically is cheaper and more informative than running on a device.
* **Chains beyond `n = 8`.** Dense `2^n` matrices grow as `4^n`; beyond `n = 12` you need sparse or tensor
  network methods, which is a different project.
* **Noise and decoherence.** The only "error" besides truncation here is floating-point round-off, which
  does not exist on real hardware - hence a floor of `1e-11`, which no device will ever reach.
* **Hardware-aware scheduling and noise.** Circuit *gate counts* now exist (§16), but the depth is a
  conservative logical estimate and there is no connectivity, calibration or decoherence model. A real
  device would need the even/odd bond-colouring schedule, a native gate set, and a noise model to say
  anything about whether a circuit is runnable.
* **More models** (XY, long-range, dimerised) - the machinery is model-agnostic but each needs its own `A/B`
  split and a meaningful initial state.
* **Advanced bounds** (recursive commutator bounds, random product formulas, commuting-term refinements).
  Only the standard first-order commutator bound is compared.

Cheap extensions, in order of payoff: add random-state averaging; add a depolarising-noise run; scale `t`
with `n`; compare one number against a published figure.

---

## 15. Where to change the code

| To change... | Look at |
|--------------|---------|
| The physical model | `tfim_terms`, `heisenberg_terms` |
| Pauli matrices, spin ordering | `single_qubit`, `two_qubit`, `kron_all` |
| The splitting scheme | `trotter_unitary` (the `order` branch) |
| The error measures | `operator_error`, `infidelity` |
| Initial states | `basis_state`, `neel_state` |
| The sweep and the fit | `sweep`, `fit_slope` |
| The round-off study | `floor_study` |
| The step-count study | `trotter_number`, `Spectral` |
| Figures | `plot_errors` |
| Flags and run order | `main` |
| Gate decomposition, circuit counts | `circuit_cost.build_circuit`, `estimate_resources` |
| Circuit depth and scheduling | `circuit_cost.schedule`, `circuit_depth`, `two_qubit_depth` |
| Resource reports (JSON/CSV) | `circuit_cost.resource_report`, `write_json`, `write_csv` |

The code is organised so each of these is one small function; adding a third model means writing one
`*_terms` builder and a two-line entry in `main`.

---

## 16. Circuit-level resource estimation (Phase 1)

Sections 1-15 count cost in **Trotter steps `r`**. That is a mathematical cost, not a quantum-computing
one: two circuits with the same `r` can differ in gate count, depth and two-qubit count by orders of
magnitude, and on real hardware those are what decide whether the circuit runs. This section compiles the
same product formulas to a concrete gate list (CNOT, RX, RZ), counts it, schedules it, and checks the
result against the matrix evolution.

`circuit_cost.py` **reuses** `trotter.tfim_terms` and `trotter.sweep`; no Hamiltonian, propagator or
error formula is re-implemented, and `trotter.py` is unmodified.

### 16.1 Conventions (and one deliberate conflict with the original spec)

The Hamiltonian used everywhere in this repository, including `trotter.py`, is

```
H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}  -  h * sum_{i=0}^{n-1} X_i      J = h = 1
  = A + B,   A = -J sum ZZ,   B = -h sum X
```

with the **minus signs**. The Phase 1 brief wrote `H = +J sum ZZ + h sum X`. The implementation follows the
**code**, not the brief, because silently changing the sign convention would break the comparison with
`results.csv`. This matters only for the sign of the rotation angles, not for any count:

* each elementary rotation is a matrix exponential `exp(-i theta/2 P)` with `P` a Pauli;
* the angle is therefore `-2 x (coefficient of that Pauli term in H) x (time span)`;
* with the minus signs above every Ising angle is negative for `t > 0` (e.g. `theta = -2J dt`), and every
  RX angle is `-2h dt`.

Gate set: `{CNOT, RX, RZ}`. That is the standard fault-tolerant-ish gate-count model used in the
Trotter-cost literature; it ignores single-qubit Clifford gates, which are cheap, and it assumes
all-to-all connectivity.

### 16.2 The two decompositions

The circuit uses exactly two identities, both verified numerically against `expm` in
`verify_gate_decomposition`:

```
exp(-i theta/2 Z_i Z_j) = CNOT(i,j) . RZ_j(theta) . CNOT(i,j)
exp(-i theta/2 X_i)     = RX_i(theta)
```

so one nearest-neighbour ZZ term costs **2 CNOT + 1 RZ**, and one field term costs **1 RX**. With an open
chain of `n` qubits there are `n - 1` bonds, hence `2(n-1)` CNOT and `n-1` RZ per ZZ layer, and `n` RX
per X layer.

### 16.3 Gate counts

| quantity | 1st order | 2nd order, half steps merged | 2nd order, not merged |
|----------|-----------|------------------------------|-----------------------|
| CNOT | `2r(n-1)` | `2(r+1)(n-1)` | `4r(n-1)` |
| RZ | `r(n-1)` | `(r+1)(n-1)` | `2r(n-1)` |
| RX | `rn` | `rn` | `rn` |
| **total** | **`r(4n-3)`** | **`3(r+1)(n-1) + rn`** | **`r(7n-6)`** |

Numerical check, `n = 6`, `r = 10`:

| | CNOT | RZ | RX | total |
|---|---|---|---|---|
| 1st order | 100 | 50 | 60 | 210 |
| 2nd order, merged | 110 | 55 | 60 | 225 |
| 2nd order, unmerged | 200 | 100 | 60 | 360 |

Note that `3(r+1)(n-1) + rn = r(4n-3) + 3(n-1)`: **second order with merging costs only `3(n-1)` gates
more than first order at the same `r`** - a constant, independent of `r` and of the simulation length. It
buys an error smaller by a factor of order `r`. That is the single most useful number in this section, and
it is the obvious starting point for Phase 2 (accuracy per CNOT).

### 16.4 Why the half steps merge

Order 2 applies `ZZ(dt/2) X(dt) ZZ(dt/2)` `r` times. Written out, the end of step `k` and the start of step
`k+1` are `... ZZ(dt/2) ZZ(dt/2) ...` with nothing in between, acting on the same qubits, and
`RZ(a).RZ(b) = RZ(a+b)`. So `r` steps contain `2r` half-layers that fuse into `r + 1` layers: one
half-angle layer at the start, one at the end, and `r - 1` full layers inside. Counting them as `2r` full
layers (the unmerged column above) overestimates the CNOT count by `2(r-1)(n-1)` gates - for `n = 6`,
`r = 200` that is 1990 CNOTs of pure fiction. `--no-merge` produces that naive count on purpose, so the two
can be compared, and both circuits are checked to give the same unitary.

Two subtleties worth stating:

* merging is exact, not an approximation - it merges two rotations that are already adjacent, and the test
  `test_merged_and_unmerged_second_order_are_the_same_operator` confirms it to `1e-12`;
* the merged circuit is **not** the same as a first-order circuit: the merged blocks at the two ends have
  half the angle, and the middle layers differ from Lie-Trotter's by construction.

### 16.5 Depth is not gate count

The gate list is scheduled with ASAP list scheduling (`schedule`): a gate enters the earliest layer in
which all of its qubits are free, and the relative order of gates is preserved, so nothing non-commuting
is reordered. `circuit_depth` is the number of layers; `two_qubit_depth` is the number of layers that
contain at least one CNOT.

Measured at `r = 200` (from the committed run):

| | n=4, 1st | n=4, 2nd | n=6, 1st | n=6, 2nd |
|---|---|---|---|---|
| total gates | 2600 | 2609 | 4200 | 4215 |
| circuit depth | 1403 | 1409 | 1409 | 1415 |
| two-qubit depth | 1200 | 1206 | 1404 | 1411 |
| CNOT count | 1200 | 1206 | 2000 | 2010 |

So depth is about **half** the gate count at `n = 4` and a **third** of it at `n = 6`: at `n = 6` there are
2000 CNOTs but only 1404 CNOT layers, because bond `i+1` can start while bond `i` finishes its second CNOT.
Equating cost with gate count would therefore overstate the time cost by up to 3x.

Assumptions, stated because they are *not* hardware claims: the schedule is conservative and logical; it
does **not** re-order the commuting ZZ bonds of one layer into even/odd rounds (which would cut the per-layer
depth from about `3(n-1)` to about 6); it assumes every qubit can talk to every other; it reports no
wall-clock, no pulse duration, no fidelity.

### 16.6 Zero-angle policy

If `J = 0` or `h = 0`, the corresponding rotation angle is exactly zero. Default policy: **drop the whole
block**, CNOTs included, because `CNOT(i,j).CNOT(i,j) = I` and `RZ(0) = RX(0) = I`. Counting those gates
would inflate a resource estimate with operations that no compiler would emit. `keep_zero_angles=True`
(and the report field `assumptions.zero_angles_kept`) keeps them symbolically for structural comparisons,
which is why the module has both numbers available.

### 16.7 How to run it

```bash
python circuit_cost.py                        # full run, ~15 s
python circuit_cost.py --no-plots             # skip the six figures
python circuit_cost.py --no-merge             # naive unmerged second-order counts
python circuit_cost.py --nlist 4,5 --rmax 50  # smaller sweep
python -m pytest tests/ -q                    # 48 unit tests, ~2 s
```

Flags: `--t`, `--J`, `--h`, `--rmax`, `--npoints`, `--nlist`, `--outdir`, `--csv`, `--json`, `--no-plots`,
`--no-merge`. Defaults match `trotter.py` (`t = 1`, `J = h = 1`, `r` up to 200, 26 log-spaced points).

Outputs:

* `results/circuit_resources.csv` - 132 rows, one per `(n, r, order)`: the measured error from `trotter.sweep`
  plus `qubits, cnot_count, rx_count, rz_count, total_gates, circuit_depth, two_qubit_depth`;
* `results/circuit_reports.json` - 16 structured reports (the §16.9 schema) for `n = 4, 6`, `r = 1, 5, 50, 200`,
  both orders;
* plots H-M (§9.5).

### 16.8 What was verified

```
zz_identity_theta=-2.000     max |difference| = 0.000e+00
rx_identity_theta=-2.000     max |difference| = 1.796e-16
order1_merged=True           max |difference| = 1.103e-15
order1_merged=False          max |difference| = 1.103e-15
order2_merged=True           max |difference| = 1.132e-15
order2_merged=False          max |difference| = 9.015e-16
-> PASS (tolerance 1e-10)
```

(The `order*` lines compare the multiplied-out gate list with `trotter.trotter_unitary`, `n = 4`, `r = 3`.)

Analytical validation, printed by every run (`n=2 r=1`, `n=4 r=5`, `n=6 r=10`, `n=8 r=50`, both orders):
compiled counts equal the closed forms exactly, e.g. `n=8, r=50, order 1` gives `CNOT=700, RZ=350,
RX=400, total=1450 = 50*(4*8-3)`, and `order 2` gives `CNOT=714 = 2*51*7`.

The 48 tests cover: analytic counts against literal formulas, `r`-step and per-step counts, the merging
identity `2r -> r+1` layers, zero-coupling and negative-`t` behaviour, every rejection path (`n < 2`,
`r < 1`, bad order, bad boundary, NaN/inf parameters, non-integer `n`), scheduling on disjoint and shared
qubits, two-qubit depth, report/JSON/CSV structure, and two regressions: `trotter.trotter_unitary` still
matches `expm`, and a committed `results.csv` row is recomputed to `1e-12`.

Regression on the existing pipeline: `python trotter.py` re-run into a temporary directory reproduced
`results.csv`, `results/floor_study.csv` and `results/trotter_number.csv` **byte for byte**.

### 16.9 A real report

`python circuit_cost.py` prints and exports entries of this form (`n = 6`, `r = 50`, order 2 - computed, not
hardcoded):

```json
{
  "system":    {"model": "TFIM", "num_qubits": 6, "interaction_strength": 1.0,
                "transverse_field": 1.0, "simulation_time": 1.0,
                "boundary_condition": "open"},
  "trotter":   {"formula_order": 2, "steps": 50, "step_size": 0.02,
                "half_step_merging": true},
  "resources": {"cnot_count": 510, "rx_count": 300, "rz_count": 255,
                "total_gates": 1065, "circuit_depth": 365, "two_qubit_depth": 361,
                "cnot_per_step": 10.2, "rx_per_step": 6.0, "rz_per_step": 5.1,
                "a_layers": 51, "x_layers": 50},
  "assumptions": {"zero_angles_kept": false,
                 "scheduling": "ASAP list scheduling, conservative, logical depth only",
                 "hamiltonian_sign": "H = -J sum ZZ - h sum X (as in trotter.py)",
                 "gate_set": "CNOT, RX, RZ"}
}
```

`cnot_per_step = 10.2 = 510/50` is not an integer because the merged circuit has `r + 1 = 51` ZZ layers
for `r = 50` steps - the two end half-layers are charged to the run as a whole.

### 16.10 What this is not, and what Phase 2 would do

* Counts are **logical**. No connectivity, native gate set, error correction, magic-state cost, hardware
  timing or fidelity. `RZ` here is a single-parameter rotation; on hardware it is not one physical gate.
* Depth is a **conservative logical estimate** from one scheduler, not a compiled, optimised depth.
* There is **no noise model**, so no statement about which `r` is optimal on a device - the interesting
  crossover needs depolarising noise (Phase 2/3), and this module deliberately has no hook for it.
* Phase 2 is analysis, not new machinery: the committed `circuit_resources.csv` already joins error with
  CNOT count, total gates and depth, so "which formula gives the best accuracy per CNOT", "how does error
  scale with depth" and "what circuit size reaches `1e-3`" are group-by queries over that table.

---

*Project: Trotterisation error vs step count for spin chains. Pure `numpy` / `scipy` / `matplotlib`; one
command; deterministic; every figure and number in this README is produced by `python trotter.py`.*

## 17. Accuracy versus circuit resources (Phase 2)

§16 established what a circuit costs. This section asks the research question that follows from it:

> **How efficiently do the two product formulas buy accuracy when the budget is measured in quantum
> circuit resources rather than in Trotter steps?**

No new physics is simulated here: Phase 2 reads the errors Phase 1 measured and joins them to the gate
counts Phase 1 computed. `results/circuit_resources.csv` is opened read-only (a test hashes it before and
after a full run), and every output lands in `results/phase2/`.

### 17.1 The data, and the one gap that had to be measured

Audit of Phase 1 (§16.7): 132 rows, `n = 4, 5, 6`, both orders, `r = 1...200`, error strictly decreasing in
`r` for all six configurations, every row carrying consistent `CNOT + RX + RZ = total` resources, no empty
fields, no failed experiments.

That data cannot answer the brief, because at `r = 200` first order is still at `6.9e-3 ... 1.1e-2`:
the targets `1e-3` and `1e-4` are **unreachable inside the Phase 1 range**, and the equal-budget comparisons
would all sit at coarse, pre-asymptotic step counts. Rather than extrapolate, Phase 2 adds a
**supplementary measurement**: the unchanged `trotter.sweep` plus `circuit_cost.estimate_resources` at 14
extra step counts `r = 240 ... 30000` for the same three chains, written to
`results/phase2/supplementary_sweep.csv` (84 rows, clearly marked `source = supplementary`).

It is a measurement, not an interpolation: those points are plotted as hollow markers in every figure, and
`--no-supplement` reproduces the analysis on Phase 1 data alone. With it, the usable dataset is 216 rows
over 36 step counts. **No point reaches the `1e-10` numerical floor** (smallest measured error `1.26e-9`),
so every error discussed below is truncation error. Where a selection sits at the very edge of the measured
range, the minimum-resource table says `at largest measured r`.

### 17.2 Architecture

| module | responsibility |
|---|---|
| `analysis/resource_analysis.py` | CSV loading with per-row validation (missing column, non-numeric, non-integer, `qubits != n`, `CNOT+RX+RZ != total`), audit, log-safe helper |
| `analysis/comparison.py` | equal-step pairing, equal-budget selection, log-log interpolation, efficiency helpers |
| `analysis/scaling.py` | closed-form gate counts (imported from `circuit_cost`), verification, empirical exponents |
| `analysis/accuracy_targets.py` | minimum-resource search for `1e-1, 1e-2, 1e-3, 1e-4` |
| `analysis/plotting.py` | the ten figures |
| `analysis/report.py` | `phase2_summary.json` + `phase2_report.md`, generated from the numbers |
| `analysis/run_phase2.py` | driver, including the supplementary measurement |

Bad rows are never repaired: they are dropped and recorded with their line number and reason. A malformed
CSV is refused with a `SchemaError` naming the missing columns.

### 17.3 Findings (all from the committed run)

**1. Error falls as a power of the CNOT count, and the exponents differ by exactly the expected factor.**
Fitting `log(error)` against `log(CNOT)` gives slopes `-1.01` (first order) and `-2.07` (second order),
averaged over `n = 4, 5, 6`. One extra decade of accuracy therefore costs about `9.9x` more CNOTs at first
order but only `3.0x` at second order.

**2. At equal step count, second order is far more accurate for a nearly free fixed overhead.** At `r = 200`,
`n = 4`: error drops from `6.9e-3` to `2.8e-5` (**243x**) while CNOTs go from `1200` to `1206` (`+0.5%`) and
total gates from `2600` to `2609` (`+0.35%`). The overhead is `2(n-1)` CNOTs and `3(n-1)` gates - a
constant, not a per-step cost - so it vanishes as a fraction: `+0.003%` CNOTs at `r = 30000`. Across all
configurations the improvement factor spans `1.03x` (at `r = 1`, where the formulas have not separated yet)
to `3.6e4` (at `r = 30000`), median `113x`.

**3. At equal CNOT or equal total-gate budgets, second order wins almost everywhere.** Comparing at 105
matched budgets per metric, second order gives the lower error in **102 of 105**, by `32%` to more than
`99.99%` (median `99.1%`). The three exceptions are all at `r = 1`: there the `2(n-1)` fixed overhead spends
gates that the first-order circuit converts into an extra step instead, so order 1 is briefly better. This is
the one place where the ranking reverses, and it is reported rather than smoothed away.

**4. For a fixed achieved accuracy, second order needs 70-99% fewer resources.** From the minimum-resource
table (cheapest configuration by CNOT count):

| n | target | 1st order: r / CNOT / gates | 2nd order: r / CNOT / gates | gate saving |
|---|---|---|---|---|
| 4 | `1e-3` | 1537 / 9222 / 19981 | 37 / 228 / 490 | 97.5% |
| 5 | `1e-3` | 2229 / 17832 / 37893 | 45 / 368 / 777 | 97.9% |
| 6 | `1e-3` | 3231 / 32310 / 67851 | 45 / 460 / 960 | 98.6% |
| 6 | `1e-4` | 30000 / 300000 / 630000 | 162 / 1630 / 3417 | 99.5% |

**5. Resources are exactly linear in bonds and in steps, and depth is almost independent of `n`.** All 216
rows match the closed forms `2r(n-1)`, `r(4n-3)`, `2(r+1)(n-1)`, `3(r+1)(n-1)+rn` with maximum absolute
difference `0`. CNOTs scale as `n^1.00` at fixed `r` and as `r^1.00` at fixed `n`. Logical depth at fixed
`r` changes by only `+3` layers per extra qubit (`0.43%` from `n = 4` to `n = 6`), while the same depth
range covers `r = 45` to `r = 200` - so **on this conservative scheduler, depth is set by the step count,
not by the system size**. That is a property of the scheduler (bonds are serialised, no even/odd bond
colouring), stated as such, not a hardware result.

**6. Depth-based and gate-based comparisons agree on the ranking but not on the numbers.** Because depth is
nearly `n`-independent, error-versus-depth curves for `n = 4, 5, 6` almost coincide (they differ by the
prefactor only), whereas error-versus-CNOT curves are clearly ordered by `n`. Both views put second order
below first order at any fixed resource value, so the qualitative conclusion is robust to the choice of cost
axis; the quantitative gap is not the same on the two axes.

**7. Numerical reliability.** 216 rows, none at the floor, all six curves strictly decreasing, no duplicate
`(n, order, r)` rows, no non-positive errors. Raw values are preserved: the CSV is read-only and log axes
clip at `1e-15` in a *copy* of the data (`safe_log`), tested by comparing the input array before and after.

### 17.4 Outputs

| file | contents |
|---|---|
| `error_vs_cnot.png` | error vs CNOT count, both formulas, three chain lengths, fitted slopes |
| `error_vs_total_gates.png` | same against `CNOT + RX + RZ` |
| `error_vs_circuit_depth.png` | error vs logical depth (conservative scheduler) |
| `error_vs_two_qubit_depth.png` | error vs two-qubit depth |
| `order_comparison_equal_steps.png` | left: error vs `r`, both formulas; right: the `2(n-1)` / `3(n-1)` overhead vs `r` |
| `order_comparison_equal_cnot.png` | one panel per `n`, equal-CNOT budgets, gap factors annotated |
| `order_comparison_equal_gates.png` | the same for total-gate budgets |
| `cnot_vs_qubits.png`, `total_gates_vs_qubits.png`, `depth_vs_qubits.png` | resource scaling, dotted analytic lines |
| `minimum_resources_for_accuracy.csv` | 48 rows: cheapest-by-CNOT and cheapest-by-gates per `(n, order, target)` |
| `resource_efficiency.csv` | 216 rows: error per CNOT, error per gate, equal-step error reduction |
| `phase2_summary.json` | the whole analysis, machine-readable (strict JSON: non-finite values become `null`) |
| `phase2_report.md` | the generated report, including the full minimum-resource table |

Every figure is produced from measured data; nothing is sketched or hardcoded.

### 17.5 Verification

* `python -m pytest tests/ -q` -> **78 tests pass**: the 48 Phase 1 tests plus 30 Phase 2 tests covering
  loading/validation (missing column, empty error, non-integer `n`, `qubits != n`, `CNOT+RX+RZ != total`),
  equal-step pairing, budget selection against a known power law, refusal to extrapolate, threshold search
  (including an unachievable target, a floor-level error, and a non-monotonic curve), analytic scaling
  identities, `safe_log` not mutating its input, and a regression that re-runs the whole driver and checks
  the Phase 1 files are byte-identical afterwards.
* Analytic gate counts re-verified against all 216 rows: max difference `0`.
* `python trotter.py` outputs are unchanged; `results/circuit_resources.csv` has the same 132 rows as Phase 1.

### 17.6 What Phase 2 does not claim

* Not quantum advantage. "Fewer gates for the same error" is a resource-efficiency statement about two
  classical descriptions of one ideal simulator.
* No noise model. Without depolarising noise there is no optimal `r` and no statement about what a device
  could do; the ranking here is a statement about ideal circuits only.
* No hardware depth. The scheduler is logical and conservative; a hardware-aware schedule (bond colouring,
  connectivity, native gates) would change absolute depth and could change the depth-based ordering.
* Only `n = 4, 5, 6` and `t = 1`, `J = h = 1`. Nothing is extrapolated beyond that, and the one selection
  sitting at `r = 30000` is marked as such.

---

*Phase 1 (circuit costs): every gate count, depth and figure in §16 is produced by `python circuit_cost.py`
and checked by `python -m pytest tests/ -q`.*

*Phase 2 (resource trade-offs): every number and figure in §17 is produced by
`python -m analysis.run_phase2`, with the written report at `results/phase2/phase2_report.md`.*
