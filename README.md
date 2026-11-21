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

**Runtime: about 30 seconds** on a normal laptop (up to a minute on a slow machine). The `n`-scaling
study and the round-off study dominate. Nothing is parallelised; results are deterministic, so two runs
produce identical CSVs.

Expected console output ends with a table like the one in [§10.1](#101-the-slope-table).

---

## 6. Repository layout

```
quantum_computing/
├── trotter.py                  # the entire project: model, methods, experiments, plots (794 lines)
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
    ├── floor_study.csv         # 108 rows: error vs r up to 1e8
    └── trotter_number.csv      # 12 rows: r(eps) for n = 2...7, both orders
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

**Plot G - `plot_G_trotter_number.png`.** Left: smallest `r` with operator error `<= 1e-3` vs `n`, log-log.
First order needs `r = 704, 1099, 1381, 1746, 2254, 2554` (`r ~ n^1.03`); second order needs
`r = 23.6, 29.2, 33.7, 39.0, 43.1, 46.9` (`r ~ n^0.55`). Hollow markers are extrapolated past `r = 200`
(all first-order points, since `1e-3` needs `r > 10^3`); second-order points are interpolated inside the
measured range. Right: the fitted exponent per `n`, flat at `-1.02` and `-2.02`.

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
* **Gate-level cost model.** The cost is counted in matrix powers `r`, not in CNOTs. A hardware-facing
  study would convert `r` into gate counts.
* **More models** (XY, long-range, dimerised) - the machinery is model-agnostic but each needs its own `A/B`
  split and a meaningful initial state.
* **Advanced bounds** (recursive commutator bounds, random product formulas, commuting-term refinements).
  Only the standard first-order commutator bound is compared.

Cheap extensions, in order of payoff: add gate counts; add random-state averaging; add a depolarising-noise
run; scale `t` with `n`; compare one number against a published figure.

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

The code is organised so each of these is one small function; adding a third model means writing one
`*_terms` builder and a two-line entry in `main`.

---

*Project: Trotterisation error vs step count for spin chains. Pure `numpy` / `scipy` / `matplotlib`; one
command; deterministic; every figure and number in this README is produced by `python trotter.py`.*
