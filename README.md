# Trotterisation of a Spin Chain

**How much error does a Trotterised (split-operator) quantum simulation make, and how does that error shrink as you take more, smaller steps?**

This repository contains a self-contained numerical study of product-formula ("Trotter") errors for a
transverse-field Ising chain and a Heisenberg chain. Two independent error measures are used: the
operator-norm distance between the exact and Trotterised propagators, and the infidelity of an evolved
state. Textbooks predict a first-order error falling like `1/r` and a second-order error falling like
`1/r^2` in the number of steps `r` (and `1/r^2`, `1/r^4` for infidelity). This project measures whether
that actually happens, how tight the standard commutator bound is, how the error depends on chain length,
what finally stops the error from falling, and how many steps a fixed accuracy costs.

Phases 1-3 are written from scratch with `numpy`, `scipy` and `matplotlib`: there are no datasets, no
downloads and no hidden dependencies, and one command reproduces every number and figure in those
sections. Phase 4 adds an **independent validation layer** that does use a quantum-computing SDK
(Qiskit + Aer, §19) - it re-derives the same numbers with different code, checks that the custom
circuits are the same unitaries, and measures what compilation and connectivity cost. The custom
simulator never imports Qiskit; it remains the reference implementation.

---

## Phase tracker

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Gate-count analysis: circuit decomposition, CNOT/RX/RZ counts, depth | Complete (`abd5c54`) |
| Phase 2 | Ideal Trotter error versus circuit resources, accuracy targets, scaling | Complete (`9905a7f`) |
| Phase 3 | Density-matrix simulation, three depolarising channels, optimal step count | Complete (`9e47be8`) |
| Phase 4 | Independent audit, robustness experiments, Qiskit validation, transpilation and hardware-aware costs | Complete - **PASS WITH QUALIFICATIONS**, §19 |
| Phase 5 | Large-scale trajectories ($n=10$), continuous relaxation ($T_1/T_2$), mock hardware, ZNE, dense grid, Sabre routing | Complete, §21 |

Everything below is produced by the committed code and checked by `python -m pytest tests/ -q`
(224 tests). Phase 4 confirms the Phase 3 physics to machine precision but qualifies three claims -
read [§19.6](#196-what-the-audit-corrects) before quoting a number from §18. Phase 5 resolves all
qualifications and delivers hardware execution, readout mitigation, and large-scale trajectories.

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
18. [Noise and the optimal step count (Phase 3)](#18-noise-and-the-optimal-step-count-phase-3)
19. [Independent audit, robustness and Qiskit validation (Phase 4)](#19-independent-audit-robustness-and-qiskit-validation-phase-4)
20. [Getting Started for the Next Developer](#20-getting-started-for-the-next-developer)
21. [Systems Scaling, Hardware and Continuous Noise (Phase 5)](#21-systems-scaling-hardware-and-continuous-noise-phase-5)

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
* Three libraries for Phases 1-3:

```bash
pip install numpy scipy matplotlib
```

* Two more for the Phase 4 SDK layer (optional: Phases 1-3 run without them):

```bash
pip install qiskit qiskit-aer
```

Or install everything at once from the pinned ranges in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Developed against `numpy 2.4.6`, `scipy 1.17.1`, `matplotlib 3.11.2`, `qiskit 2.6.2`, `qiskit-aer 0.18.4`.
Anything in the ranges recorded in `requirements.txt` should work; the transpiler results in §19 are
specific to the Qiskit version, because a different transpiler routes differently.

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

Phase 4 has two drivers. The first is the independent audit, which also runs the five robustness
experiments and writes the audit report:

```bash
python -m validation.run_audit            # audit + robustness + 5 figures, ~5 minutes
python -m validation.run_audit --skip-robustness   # audit only, ~30 s
```

The second is the Qiskit layer, which needs the audit's summary as input:

```bash
python -m quantum_sdk.run_phase4          # equivalence, transpilation, SDK noise, 7 figures, ~30 s
python -m quantum_sdk.run_phase4 --quick  # one system size, two step counts
```

Run them in that order. Neither one writes to `results/phase1`, `results/phase2` or `results/phase3`.

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
├── noise/                      # Phase 3: density matrices + depolarising noise
│   ├── density_matrix.py       # representation, local operators, diagnostics
│   ├── channels.py             # global / single-qubit / two-qubit depolarising
│   ├── noisy_simulator.py      # chronological noisy execution of the Phase 1 circuit
│   ├── metrics.py              # fidelity, infidelity, trace distance
│   ├── experiments.py          # sweeps, sensitivity grid, optimal-r search
│   ├── plotting.py             # the twelve Phase 3 figures
│   ├── report.py               # phase3_report.md + phase3_summary.json
│   └── run_phase3.py           # driver, starting with the validation stage
├── validation/                 # Phase 4: the independent audit, written without the noise/ code
│   ├── independent.py          # from-scratch engine: gates, product formulas, channels, metrics
│   ├── audit.py                # 187 mathematical checks against Phase 3
│   ├── reproduction.py         # 12 stored configurations recalculated independently
│   ├── optimal_check.py        # every stored optimal r re-derived from the sweep
│   ├── comparisons.py          # independent re-pairing of first vs second order
│   ├── robustness.py           # the five robustness experiments
│   ├── resources.py            # analytic gate counts vs the circuits actually used
│   ├── plotting.py             # figures 1-5 of Phase 4
│   ├── report.py               # audit_report.md
│   └── run_audit.py            # driver
├── quantum_sdk/                # Phase 4: Qiskit validation and transpilation layer
│   ├── circuit_builder.py      # Phase 1 gate list -> QuantumCircuit, with the index mapping
│   ├── equivalence.py          # phase-insensitive custom vs Qiskit unitary comparison
│   ├── transpilation.py        # logical / line / ring / star configurations
│   ├── resources.py            # logical and compiled resource extraction
│   ├── layout.py               # recover the logical qubit order after routing
│   ├── noise_check.py          # channel reconciliation + Aer cross-check of the custom noise
│   ├── experiments.py          # the SDK study grids
│   ├── plotting.py             # figures 6-12 of Phase 4
│   ├── report.py               # phase4_report.md
│   └── run_phase4.py           # driver (reads the audit summary)
├── conftest.py                 # makes the project importable for pytest
├── requirements.txt            # pinned dependency ranges for all four phases
├── tests/
│   ├── test_circuit_cost.py    # 48 tests: counts, validation, scheduling, reports, regression
│   ├── test_phase2.py          # 30 tests: loading, comparisons, targets, scaling, floor
│   ├── test_phase3.py          # 71 tests: density matrices, channels, gates, metrics, optimal r
│   ├── test_phase4_audit.py    # 20 tests: Hamiltonian, formulas, channels, metrics, pairing
│   └── test_phase4_sdk.py      # 32 tests: conversion, equivalence, resources, transpilation, noise
├── PROGRESS.md                 # detailed engineering handover, phase by phase
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
    ├── phase2/                 # Phase 2 outputs, never mixed with Phase 1
    │   ├── error_vs_cnot.png ... depth_vs_qubits.png   # 10 figures
    │   ├── minimum_resources_for_accuracy.csv          # 48 rows
    │   ├── resource_efficiency.csv                     # 216 rows
    │   ├── supplementary_sweep.csv                     # 84 extra measurements, r>200
    │   ├── phase2_summary.json
    │   └── phase2_report.md
    └── phase3/                 # Phase 3 outputs, never mixed with Phases 1-2
        ├── configs/experiment_config.json
        ├── figures/fig01_..._fig12_...png              # 12 figures
        ├── global_noise_results.csv                    # 504 rows
        ├── gate_noise_results.csv                      # 504 rows
        ├── noise_sensitivity.csv                       # 70 rows
        ├── optimal_r.csv                               # 72 configurations
        ├── phase3_summary.json
        └── phase3_report.md
    └── phase4/                 # Phase 4 outputs, never mixed with Phases 1-3
        ├── audit/              # audit_report.md, validation_summary.json,
        │                       #   independent_reproduction.csv, optimal_r_verification.csv,
        │                       #   comparison_audit.csv
        ├── robustness/         # 5 CSVs: initial state, evolution time, noise strength,
        │                       #   noise components, candidate grid
        ├── sdk/                # circuit_equivalence.csv, pre/post_transpilation_resources.csv,
        │                       #   transpilation_metadata.json, sdk_simulation_results.csv
        ├── hardware_aware/     # connectivity_comparison.csv, compiled_resource_comparison.csv,
        │                       #   compiled_noise_results.csv
        ├── figures/            # fig01-...-fig12...png (5 audit + 7 SDK)
        ├── phase4_summary.json
        └── phase4_report.md
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

## 18. Noise and the optimal step count (Phase 3)

§17 produced a clean but slightly uncomfortable result: second order buys accuracy far more cheaply than
first order. Every number in it, however, describes a circuit that makes **no error at all** apart from the
formula's own truncation. Real gates are noisy, and noise grows with the number of gates. So the research
question for this phase is:

> **Once every gate carries an error, does adding Trotter steps still help — and is there a step count that
> minimises the total error for a fixed evolution time?**

More steps shrink the formula error (`1/r`, `1/r^2`) but add gates, and therefore add noise. If the second
effect eventually wins, the total error has an **interior minimum**. That minimum is the central result of
this phase, and it is measured rather than assumed: if it sits at an end of the tested grid, the tables say
so.

### 18.1 The three noise models

All three are depolarising channels, chosen because they are the simplest thing that mixes a state without
preferring any particular error direction.

**Global depolarising**, applied once per complete Trotter step:

```
E_g(rho) = (1 - p_g) rho + p_g * I/d            d = 2^n
```

`p_g` is a **per-step** mixing probability. It is *not* comparable to a per-gate error rate: one step
contains `O(n)` gates, so the same number means very different things in the two models. A single
end-of-circuit application is also run, as a reference, and reported separately (it always prefers the
largest `r`, since one channel cannot grow with the circuit).

**Single-qubit Pauli depolarising**, after each RX and RZ:

```
E_1(rho) = (1 - p_1) rho + (p_1/3) (X rho X + Y rho Y + Z rho Z)
```

applied to that gate's target qubit only. `p_1` is the total probability of a non-identity Pauli, so each of
X, Y, Z has `p_1/3`. One consequence is worth stating because it is easy to misread: `p_1 = 1` means "one of
X, Y, Z for certain", which shrinks the Bloch vector to `-1/3` of its length, not to zero — complete
depolarisation of a qubit is at `p_1 = 3/4`, where `1 - 4p_1/3` vanishes. The tests check that shrink law
analytically. RZ can be configured separately (including as noiseless, for architectures with virtual Z);
the default gives RX and RZ the same `p_1`.

**Two-qubit Pauli depolarising**, after each CNOT:

```
E_2(rho) = (1 - p_2) rho + (p_2/15) sum_{P != II} P rho P
```

the 15 non-identity two-qubit Paulis on the control and target, with `p_2` their total probability.

Summing `P rho P` over a whole Pauli group has the closed form `2^k Tr_k(rho) (x) I_k`, so the sum over
non-identity Paulis is that expression minus `rho`. The fast path uses one partial trace; the tests also add
up all 3 (or 15) terms explicitly from the definition and require agreement to machine precision. The
`p/3` and `p/15` normalisations are separately pinned by the analytic Bloch-vector shrink factor and by a
hand-built Pauli sum.

### 18.2 Noise is applied to the actual Phase 1 gate list

No second circuit decomposition is written. `noise.noisy_simulator` walks the ordered list from
`circuit_cost.build_circuit` — the same CNOT/RZ/RX sequence that Phases 1 and 2 counted, merged second-order
ZZ layers and half-angle rotations included — and applies the matching channel after each gate, on that
gate's own qubits:

```
RX(q, theta) :  rho <- RX rho RX^dag   then  E_1 on q
RZ(q, theta) :  rho <- RZ rho RZ^dag   then  E_1 on q
CNOT(c, t)   :  rho <- CX rho CX^dag   then  E_2 on (c, t)
```

Gate application uses local `2x2`/`4x4` matrices through tensor reshapes and permutations, so one gate costs
`O(4^k 2^n)` rather than the `O(8^n)` of a full `d^2 x d^2` superoperator, and memory is one `2^n x 2^n`
matrix (32 KiB at `n = 6`). That is what makes a 512-step, 10 767-gate noisy circuit take about two seconds.

The **step boundary** for the global model needs a stated convention, because the second-order half layers
of neighbouring steps are merged and a step therefore does not end at a fixed offset in the gate list. A
step is taken to end after the X block that closes it (order 1: after the ZZ block), with the final
half-angle ZZ layer grouped into the last step. That gives exactly `r` channels for `r` steps, and the
grouping is checked against the gate-list length on every simulated circuit.

### 18.3 Validation performed before any noise result

`run_phase3` refuses to continue if any of these fail:

| Check | Result |
|---|---|
| Density-matrix walk vs the existing state-vector/Trotter propagator | `3.6e-16` max difference |
| Gate circuit vs `trotter.trotter_unitary` (spectral norm) | `1.8e-15` |
| Local gate matrices vs `circuit_cost.gate_matrix` | `0.0` |
| Fast channel vs explicit Pauli sum | `5.6e-17` |
| Compiled gate counts vs the Phase 1 closed forms (every row) | `0` |
| Purity drift under ideal evolution | `4.8e-14` |
| `p_g = 1` vs the maximally mixed `I/d` | `0.0` |
| Worst `|Tr(rho) - 1|` over noisy validation states | `2.3e-14` |
| Most negative eigenvalue over noisy validation states | `0.0` |
| `p_1 = p_2 = 0` vs no channel at all | `0.0` |

Trace, Hermiticity and positivity are checked on every state the validation stage produces, ideal and
noisy, and the tolerances above are hard gates: the driver exits rather than report results from a state
that is not a density matrix.

### 18.4 The optimal step count (the central result)

72 configurations were evaluated: 6 noise-free, 30 gate-level, 30 per-step global, 6 end-of-circuit.
**60 have an interior minimum and 12 sit at a boundary — and all 12 boundary cases are either noise-free
(`p = 0`, where more steps are always better) or the end-of-circuit reference (one channel regardless of
`r`). Not one gate-level or per-step-global configuration has a boundary minimum.** The crossover is real.

Gate-level optima, `t = 1`, `J = h = 1`, `|00...0>` start, `p_1 = p_2 = p`:

| n | order | p | r_opt | fidelity | total error | formula error | noise error | CNOT | depth |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | `1e-4` | **12** | 0.9844 | `1.56e-2` | `3.36e-3` | `1.23e-2` | 72 | 87 |
| 4 | 1 | `5e-4` | **6** | 0.9562 | `4.38e-2` | `1.37e-2` | `3.07e-2` | 36 | 45 |
| 4 | 1 | `1e-3` | **4** | 0.9288 | `7.12e-2` | `3.19e-2` | `4.10e-2` | 24 | 31 |
| 4 | 1 | `5e-3` | **3** | 0.8053 | `1.95e-1` | `5.99e-2` | `1.46e-1` | 18 | 24 |
| 4 | 1 | `1e-2` | **3** | 0.6918 | `3.08e-1` | `5.99e-2` | `2.68e-1` | 18 | 24 |
| 4 | 2 | `1e-4` | **4** | 0.9927 | `7.33e-3` | `2.49e-3` | `4.85e-3` | 30 | 37 |
| 4 | 2 | `5e-4` | **4** | 0.9736 | `2.64e-2` | `2.49e-3` | `2.40e-2` | 30 | 37 |
| 4 | 2 | `1e-3` | **3** | 0.9545 | `4.55e-2` | `8.27e-3` | `3.76e-2` | 24 | 30 |
| 4 | 2 | `5e-3` | **2** | 0.8282 | `1.72e-1` | `4.78e-2` | `1.31e-1` | 18 | 23 |
| 4 | 2 | `1e-2` | **2** | 0.7220 | `2.78e-1` | `4.78e-2` | `2.44e-1` | 18 | 23 |
| 5 | 1 | `1e-4` | **8** | 0.9803 | `1.97e-2` | `9.07e-3` | `1.08e-2` | 64 | 62 |
| 5 | 1 | `5e-4` | **6** | 0.9447 | `5.53e-2` | `1.62e-2` | `4.00e-2` | 48 | 48 |
| 5 | 1 | `1e-3` | **4** | 0.9119 | `8.81e-2` | `3.72e-2` | `5.33e-2` | 32 | 34 |
| 5 | 1 | `5e-3` | **3** | 0.7595 | `2.41e-1` | `6.90e-2` | `1.87e-1` | 24 | 27 |
| 5 | 1 | `1e-2` | **2** | 0.6224 | `3.78e-1` | `1.81e-1` | `2.46e-1` | 16 | 20 |
| 5 | 2 | `1e-4` | **6** | 0.9903 | `9.69e-3` | `6.71e-4` | `9.03e-3` | 56 | 54 |
| 5 | 2 | `5e-4` | **4** | 0.9652 | `3.48e-2` | `3.51e-3` | `3.14e-2` | 40 | 40 |
| 5 | 2 | `1e-3` | **3** | 0.9399 | `6.01e-2` | `1.16e-2` | `4.92e-2` | 32 | 33 |
| 5 | 2 | `5e-3` | **2** | 0.7768 | `2.23e-1` | `6.64e-2` | `1.70e-1` | 24 | 26 |
| 5 | 2 | `1e-2` | **2** | 0.6484 | `3.52e-1` | `6.64e-2` | `3.08e-1` | 24 | 26 |
| 6 | 1 | `1e-4` | **8** | 0.9754 | `2.46e-2` | `1.15e-2` | `1.33e-2` | 80 | 65 |
| 6 | 1 | `5e-4` | **6** | 0.9317 | `6.83e-2` | `2.04e-2` | `4.92e-2` | 60 | 51 |
| 6 | 1 | `1e-3` | **4** | 0.8916 | `1.08e-1` | `4.63e-2` | `6.56e-2` | 40 | 37 |
| 6 | 1 | `5e-3` | **3** | 0.7110 | `2.89e-1` | `8.51e-2` | `2.25e-1` | 30 | 30 |
| 6 | 1 | `1e-2` | **2** | 0.5562 | `4.44e-1` | `2.17e-1` | `2.96e-1` | 20 | 23 |
| 6 | 2 | `1e-4` | **6** | 0.9880 | `1.20e-2` | `8.60e-4` | `1.12e-2` | 70 | 57 |
| 6 | 2 | `5e-4` | **4** | 0.9569 | `4.31e-2` | `4.50e-3` | `3.88e-2` | 50 | 43 |
| 6 | 2 | `1e-3` | **3** | 0.9255 | `7.45e-2` | `1.49e-2` | `6.07e-2` | 40 | 36 |
| 6 | 2 | `5e-3` | **2** | 0.7283 | `2.72e-1` | `8.51e-2` | `2.06e-1` | 30 | 29 |
| 6 | 2 | `1e-2` | **2** | 0.5820 | `4.18e-1` | `8.51e-2` | `3.67e-1` | 30 | 29 |

### 18.5 Findings (all from the committed run)

**1. An interior optimum exists in every gate-level configuration, and it moves down as noise grows.**
`r_opt` spans `2 ... 12` for gate-level noise and falls monotonically with `p` in every one of the six
`(n, order)` series: at `n = 4`, first order goes `12 -> 6 -> 4 -> 3 -> 3` as `p` goes `1e-4 -> 1e-2`.
Interpretation (not measured): fewer, larger steps mean fewer noisy operations, so the useful circuit depth
shrinks as the error rate rises.

**2. The optimum is nearly independent of system size, which is not obvious.** Of the ten
`(order, noise level)` groups, seven give exactly the same `r_opt` for all three chain lengths; the other
three differ by at most one grid step (`12 -> 8 -> 8` at first order and `p = 1e-4`, `3 -> 2 -> 2` at first
order and `p = 1e-2`, `4 -> 6 -> 6` at second order and `p = 1e-4`). Both the formula error and the number
of gates per step grow with `n`, so the two effects very nearly cancel.

**3. Overshooting the optimum is expensive, and the formula is not what fails.** At `n = 4`, first order,
`p = 1e-3`: the best point is `r = 4` with total error `7.1e-2` using 52 gates; `r = 512` gives error
`0.925` — **13x worse for 128x the CNOTs**, even though its formula error is down to `1.8e-6`. The gates,
not the formula, are the limit.

**4. Second order keeps a smaller optimum *and* a smaller minimum.** At `n = 4`, `p = 1e-3`, both orders
spend the same 24 CNOTs: first order peaks at `r = 4` with error `7.1e-2`, second order at `r = 3` with
`4.6e-2` — 36% lower error for the same CNOT budget. `r_opt` for second order is never larger than for
first order at the same noise level.

**5. The two error sources really do meet.** At all 30 gate-level optima the noise term is the larger of the
two, by a factor between 1.2 and 13.5. At the weakest noise the balance is almost even (`n = 4`, order 1:
formula `3.4e-3`, noise `1.2e-2`); at the strongest the optimum is set almost entirely by noise.

**6. Global and gate-level models give very different optima.** The global model charges once per step, so
its optimum sits further out and depends on `p_g` almost alone: at `n = 6`, first order gives
`r_opt = 24, 16, 12, 6, 6` for `p_g = 1e-4 ... 1e-2`, with the minimum at `3.7e-3` for the weakest noise.
The gate-level model charges every gate, so its optima are much smaller (`2 ... 12`) and shift with `n`. The
end-of-circuit reference always minimises at the largest `r` (`512`), exactly as it must.

**7. Second order's advantage is bounded by noise, not universal — and that is the sharpest result here.**
At `p = 1e-3` second order has the lower total error in only **13 of 42** matched-CNOT comparisons, and all
13 sit at `r <= 8`. Two different effects produce the losses, and neither is the formula:

* In **9 of the 42** pairs the smaller per-step overhead (`2(n-1)` CNOTs) lets **first order afford one
extra Trotter step** at the same CNOT budget, which is decisive at small `r`. Noise-free that extra step is
not enough (Phase 2 measured 102/105 in second order's favour under the same rule); under noise it is.
* In the rest the two circuits have the same `r` and sit above the crossover, where the formulas no longer
separate and the marginally smaller first-order circuit accumulates slightly less noise.

The win count therefore falls monotonically with the error rate: `38/42, 24/42, 15/42, 13/42, 6/42, 6/42`
for `p = 0 ... 1e-2` (**102/252** pooled).

At *equal* `r` the picture is the same story viewed the other way: at `n = 4`, `p = 1e-3`, second order wins
for every `r <= 8` and then the two orders converge — beyond `r = 12` they differ by at most **2.8%** of the
error, and at `r = 512` by 0.005%, a tie decided by nine gates.

So Phase 2's headline is neither confirmed nor overturned under noise — it is **bounded**. Second order buys
accuracy more cheaply only while the formula still dominates the error, which is at or below the optimal step
count this phase recommends anyway. Above that point the circuit is so noisy that the choice of product
formula stops mattering, and the decision falls to gate count.

**8. Purity tracks mixing, not correctness.** At `n = 4`, `p = 1e-3`, `r = 512` the purity is `0.0634`
against `1/16 = 0.0625` for the maximally mixed state: the circuit has scrambled the state into the
maximally mixed state, and the fidelity against the exact state is `0.075` — almost entirely because of the
noise, not the formula. Note that a state can also be pure and wrong: at `r = 1`, `p = 0` the purity is
exactly 1 and the fidelity is `0.058`. Purity is a diagnostic, never a success measure.

**9. Which gate class dominates depends on the gate counts, not on a slogan.** At `r = 16`, `n = 4`,
`p = 1e-3`: charging the error to CNOTs only gives total error `8.0e-2`, to RX/RZ only `8.3e-2` — comparable,
because this circuit has 102 CNOTs and 115 single-qubit gates. Per gate a CNOT costs slightly more error
(`7.9e-4` vs `7.2e-4`), so with both classes noisy the total (`1.6e-1`) is slightly *below* the sum of the
two slices. The gate-count balance is reported instead of a claim that one class dominates.

**10. The three errors are not additive.** Over all 504 gate-level configurations the largest discrepancy
between `total_error` and `formula + noise` is `0.24` — the sum overcounts badly once both terms are large,
because fidelity is not linear. Each of the three is reported separately in every row.

### 18.6 Outputs

| file | contents |
|---|---|
| `fig01_fidelity_vs_steps.png` | fidelity vs `r`, ideal and gate-level noisy, both orders, one panel per `n` |
| `fig02_trotter_error_vs_steps.png` | the noise-free formula error alone, both orders, three sizes |
| `fig03_noise_error_vs_steps.png` | error contributed by the channels alone, per order |
| `fig04_total_error_vs_steps.png` | total error at several noise strengths — the crossing |
| `fig05_fidelity_vs_cnot.png`, `fig06_fidelity_vs_total_gates.png` | the same data against the Phase 1 cost axes |
| `fig07_order_comparison_gate_noise.png` | first vs second order at matched CNOT cost, noise-free reference dotted |
| `fig08_global_vs_gate_noise.png` | the two noise models side by side |
| `fig09_optimal_r_vs_noise.png` | measured `r_opt` vs noise strength, one panel per model |
| `fig10_sensitivity_heatmap.png` | total error over the `(p_1, p_2)` grid, plus the two single-class slices |
| `fig11_purity_vs_steps.png` | purity vs `r` with the maximally mixed line |
| `fig12_total_error_vs_depth.png` | total error against the conservative logical depth |
| `gate_noise_results.csv` | 504 rows: every gate-level configuration with all metrics |
| `global_noise_results.csv` | 504 rows: per-step and end-of-circuit global runs |
| `noise_sensitivity.csv` | 70 rows: the `(p_1, p_2)` grid and the two slices, at `r = 16` |
| `optimal_r.csv` | 72 configurations: `r_opt`, status, resources, and the three errors at the optimum |
| `phase3_summary.json` | the whole analysis, strict JSON (non-finite values become `null`) |
| `phase3_report.md` | the generated report, with the full tables |

Every row records `cnot_count`, `rx_count`, `rz_count`, `total_gate_count`, `logical_depth`,
`two_qubit_depth`, `fidelity`, `purity`, `trace_distance`, `magnetization`, `zz_correlation` and the three
error measures, so any figure can be rebuilt from the CSV alone.

### 18.7 How to run it, and what was verified

```bash
python -m noise.run_phase3              # validation, sweeps, figures, report (~3.5 min)
python -m noise.run_phase3 --no-plots   # tables and report only (~3 min)
python -m noise.run_phase3 --quick      # small grid, smoke test (~10 s)
python -m pytest tests/ -q              # the whole suite
```

There is no randomness anywhere in Phase 3, so repeated runs produce identical CSVs.

* `python -m pytest tests/ -q` -> **147 tests pass**: the 78 from Phases 1-2 plus 69 Phase 3 tests.
  They cover density matrices (outer-product construction, trace, Hermiticity, positivity, purity under
  unitary evolution, agreement with the state-vector simulator through `trotter.sweep`'s own infidelity
  numbers), channels (zero noise, `p_g = 1`, trace and positivity, out-of-range/NaN/infinite parameters,
  correct target qubits, the full Pauli-group sum, the analytic Bloch shrink factor, permutation
  covariance for a non-adjacent pair), the gate-level path (gate counts identical to Phase 1, channel after
  every nonzero gate, RZ made noiseless, merged second-order layers still reproducing the unitary,
  zero-angle policy, one global channel per step with the ranges covering the gate list exactly), metrics
  (identical and orthogonal pure states, a hand-computed `0.775`, purity of pure and mixed states, trace
distance, magnetization and ZZ correlation against independent matrix traces), and the experiment layer
  (optimal `r` taken only from evaluated candidates, boundary minima labelled, reproducible rows, invalid
  configurations rejected).
* Phase 1 and Phase 2 outputs are byte-identical after a full Phase 3 run (a test hashes them before and
  after). `trotter.py`, `circuit_cost.py` and `analysis/` are not modified by this phase.
* Every figure was inspected after generation; the four that had layout defects (an unreadable log axis on
  `fig09`, two system sizes mixed into one panel on `fig10`, an annotation colliding with the legend on
  `fig11`, and the wrong reference level on `fig08`) were fixed and re-inspected.

### 18.8 What Phase 3 does not claim

* **Not a hardware model.** The `p` values are illustrative simulation parameters spanning weak to strong
  noise. They are not measured error rates and nothing here describes IBM or any other processor.
* **No T1, T2, readout, crosstalk, leakage, connectivity, routing, decoupling, mitigation or error
  correction.** Every channel is applied independently, which is optimistic compared with a real device
  where one miscalibration affects a whole gate class at once.
* **No extrapolation.** `r_opt` is the argmin over step counts that were actually simulated; the 12 boundary
  cases are labelled as boundary cases. Nothing is fitted through or beyond the grid.
* **Small systems only.** Density matrices are in the way: `n = 4, 5, 6` at these step counts. Reaching
  larger `n` needs matrix-product operators or a stochastic (quantum-trajectory) method, not a bigger
  matrix.
* **The optimal `r` found here is optimal for this error model.** A different channel, or coherent
  (unital vs non-unital) errors, would move the crossover.

---

*Phase 3 (noise): every number and figure in §18 is produced by `python -m noise.run_phase3`, with the
written report at `results/phase3/phase3_report.md`.*

---

*Phase 1 (circuit costs): every gate count, depth and figure in §16 is produced by `python circuit_cost.py`
and checked by `python -m pytest tests/ -q`.*

*Phase 2 (resource trade-offs): every number and figure in §17 is produced by
`python -m analysis.run_phase2`, with the written report at `results/phase2/phase2_report.md`.*

---

## 19. Independent audit, robustness and Qiskit validation (Phase 4)

Phase 3 passed its own tests. That is not the same as being correct, and it says nothing about whether
the optimal step count survives a different initial state, a different evolution time, a denser step
grid, or real compilation. Phase 4 answers those questions with three independent layers:

1. **An audit written from scratch** (`validation/`) that shares no code with `noise/`, `trotter.py` or
   `circuit_cost.py` except the circuit construction under test.
2. **Five robustness experiments** on the optimal step count.
3. **A Qiskit layer** (`quantum_sdk/`) that rebuilds the same circuits, checks they are the same
   unitary, transpiles them onto documented coupling maps, and re-runs the noise model in Aer.

**Verdict: PASS WITH QUALIFICATIONS.** The physics is correct and reproduces independently to machine
precision. Three *claims* need re-wording (listed in §19.6).

### 19.1 What the audit verified

`validation/independent.py` re-implements the whole chain - Pauli embedding by explicit index
arithmetic, the two product formulas from matrix exponentials, the three channels as explicit Pauli
sums, and the metrics from their definitions - and then compares 187 quantities with the Phase 3
results. `python -m validation.run_audit` reports:

| Check group | Checks | Result |
|---|---|---|
| Hamiltonian and boundary convention | 41 | pass |
| First- and second-order product formulas | 26 | pass |
| Density-matrix evolution | 18 | pass |
| Global depolarising channel | 22 | pass |
| Single-qubit depolarising channel | 26 | pass |
| Two-qubit depolarising channel | 30 | pass |
| Fidelity, purity, trace distance | 24 | pass |
| **Total** | **187** | **187 pass, largest deviation `2.1e-15`** |

Independently recalculated values: **12 configurations, 144 stored numbers, 0 failures**, largest
absolute difference `5.8e-15` (tolerance `1e-9`). All **72 stored optimal step counts** are confirmed
as the argmin of total error over the sweep; the 12 boundary minima are exactly the six noise-free and
six end-of-circuit runs, all at `r = 512`. The independent pairing of first- against second-order rows
agrees with `noise.experiments.pair_orders` on **252 of 252 pairs**, with zero winner disagreements.

The reported non-additivity of the three errors is reproduced exactly: the largest
`|error_total - (error_Trotter + error_noise)|` is `0.24248777` at `n = 6`, second order, `r = 1`. It is
large there because the noise-induced error is measured against the *Trotter* state, so at `r = 1` the
two contributions partly cancel; the mean absolute deviation is `0.0050`.

### 19.2 Robustness: does the optimal step count survive?

Five experiments, all under the reference gate-level model unless stated. Selected output:

| Experiment | Question | Result |
|---|---|---|
| Initial state | Is the optimum a property of `\|0...0>`? | Interior minimum for all 4 states, but the *location* moves: at `n = 4`, first order, `p = 1e-3`, `r* = 4` for `\|0...0>` and `\|1...1>` and `r* = 6` for `\|+...+>` and the random state. Second order is `r* = 3` throughout. |
| Evolution time | Does `r*` depend on `t`? | Yes, roughly proportionally: `r* = 3, 4, 8, 16` (first order) and `2, 3, 6, 12` (second order) for `t = 0.5, 1, 2, 4`. Always interior. |
| Noise strength | Does `r*` fall as noise rises? | 12 of 12 groups non-increasing, 6 of 12 strictly decreasing, **0 violations**. Several flatten at the smallest tested `r`, so "strictly decreasing" is not supported. |
| Gate classes | Which class dominates? | At equal per-gate error the contribution tracks the gate count, not the gate class: `0.0215` (single-qubit, 28 gates) against `0.0199` (two-qubit, 24 gates) at `n = 4`, `p = 1e-3`. The two classes add to within 1% of the combined run. |
| Candidate grid | Is the optimum an artefact of the grid? | **3 of 6** reported optima move on a dense integer grid: first order `4 -> 5` at `n = 4, 5, 6` (error `0.0712 -> 0.0693` at `n = 4`); second order stays at `r = 3`. |

**Reading:** the *existence* of a finite interior optimum is robust. The *value* is a best tested
**discrete** step count that depends on the state, the evolution time and the grid.

### 19.3 Qiskit: the custom circuits are the same unitaries

Qiskit `2.6.2` with `qiskit-aer 0.18.4`. `quantum_sdk.circuit_builder` walks the **Phase 1 gate list**
and emits `rx`, `rz`, `cx` in the same order - no decomposition is re-derived. Repository qubit 0 is the
most significant bit, Qiskit's is the least, so `qiskit_qubit = n - 1 - repo_qubit`; that mapping makes
the two basis indices equal, so unitaries compare with no permutation, and it is asserted numerically in
the tests.

Comparison is phase-insensitive - `min_phi ||U_custom - e^{i phi} U_Qiskit||_2`, with the optimal phase
*measured*, not assumed:

| Quantity | Result |
|---|---|
| Configurations compared | 36 of 36 pass (`n = 3, 4, 5`; both orders; `r` up to 8) |
| Largest spectral distance | `8.9e-16` |
| Largest statevector infidelity | `1.4e-14` |
| Measured global phase | `~1e-17` (i.e. none beyond round-off) |
| Pre-transpilation counts vs Phase 1 | 36 of 36 identical, depth also 36 of 36 |

### 19.4 Transpilation, connectivity and hardware-aware cost

Two families, every parameter recorded in `results/phase4/sdk/transpilation_metadata.json`:

* **`A_logical_generic`** - basis `rz`/`rx`/`cx`, optimisation level 0, **no** coupling map, seed 7.
* **`B_line` / `B_ring` / `B_star`** - same basis, optimisation level 1, seed 7, explicit `CouplingMap`.

| Topology | Max extra CNOTs | Two-qubit inflation | Depth inflation |
|---|---|---|---|
| logical (control) | 0 | 1.00x | 1.00x |
| `line` (`0-1-...-(n-1)`) | **0** | 1.00x | 1.00x |
| `ring` (line + `(n-1,0)`) | **0** | 1.00x | 1.00x |
| `star` (all bonds to qubit 0) | **+39** | 1.54x | **2.40x** |

Line and ring are free because the TFIM chain *is* the hardware graph, so the transpiler finds a layout
with no swaps at all. The star is the deliberate mismatch: every TFIM bond is two hops away, and the
cost grows with both `n` and `r` (`+21` CNOTs at `n = 4`, `r = 3`; `+39` at `n = 5`, `r = 8`).

**One SDK wrinkle worth knowing.** A routed circuit can return its state in a *permuted* qubit order -
the transpiler normally accounts for the final layout in the measurement bit mapping, and a
density-matrix simulation applies no measurement. `quantum_sdk.layout.restore_order` recovers the
permutation (from the layout metadata when available, otherwise by search) and **verifies** it by
checking that the noiseless routed state becomes the logical state (residual `~1e-16`). Ignoring this
step made the star look catastrophically worse than it is: `F = 0.29` instead of `0.94` for `n = 4`,
second order, almost entirely relabelling.

### 19.5 SDK noise: reconciled, not forced

Aer's `depolarizing_error(p, k)` is the **full-depolarisation** channel `(1-p)rho + p I/d`, which is
*not* the project's Pauli mixture (`p/3` on one qubit, `p/15` on two). Measured, not assumed: the two
differ by `3.1e-2` at `p = 0.2`, consistent with `p_aer = 4p/3` (one qubit) and `16p/15` (two qubits) on
pure product states. Whole-circuit comparisons therefore rebuild the project's channel exactly as an
explicit Pauli mixture (`noise_check.make_noise_model`), giving agreement at `2.2e-16` per gate and
**`1.6e-15` for the full noisy density matrix** across 36 configurations. That is an independent
validation of the custom noise engine - channel definitions, gate placement, ordering and the absence
of double application.

Under gate-level noise with compiled circuits (`p_1 = p_2 = 1e-3`):

* `line`/`ring` routing costs **exactly zero** in fidelity as well as in gates.
* `star` routing costs between `-5.5e-5` and `+2.9e-2` in total error, growing with `n` and `r` and in
  step with the extra two-qubit gates. The one negative value is real physics, not a bug: at `n = 4`,
  `r = 1` the noiseless fidelity is `0.0580`, *below* the maximally mixed `1/d = 0.0625`, so
  depolarising noise moves the state towards the target.
* At an error target of `0.10`, first and second order need the *same* compiled two-qubit count at every
  `n`; at the stricter `0.06`, second order is cheaper at `n = 3` (12 against 16) and is the only order
  to reach the target at `n = 5`.

### 19.6 What the audit corrects

Three wording corrections, no code changes to Phases 1-3:

1. **"`r*` decreases monotonically with noise"** -> **non-increasing**: 12 of 12 groups are
   non-increasing, only 6 strictly; several saturate at the smallest `r` tested.
2. **"`r*` is nearly independent of system size"** -> true only *within a noise level* and for these
   three sizes; `r*` shifts by one step between `n = 4, 5, 6` on a denser grid, and it moves with the
   initial state and roughly proportionally with the evolution time.
3. **"The optimal step count is `r*`"** -> **"the best tested discrete step count is `r*`"**. No
   continuous optimisation was performed, and `r = 5, 7, 9, 10, 11` were never simulated in Phase 3.

One defect was found in the *audit* code itself and fixed: `quantum_sdk.equivalence.phase_insensitive`
rotated by `+arg(<a,b>)` instead of `-arg(<a,b>)`, so it reported the distance at twice the phase offset
rather than the minimum. Because the measured offset is `~1e-17` for these circuits the reported
equivalence numbers are unaffected, and the regression test in `tests/test_phase4_sdk.py` pins the
correct behaviour.

### 19.7 Figures

`results/phase4/figures/` - all twelve inspected visually.

| Figure | Content |
|---|---|
| `fig01_initial_state_optimal_r.png` | optimal `r` against initial state and noise level |
| `fig02_evolution_time_optimal_r.png` | optimal `r` against total evolution time |
| `fig03_noise_strength_trend.png` | `r*` versus noise strength, gate-level (per-gate) and global (per-step) panels |
| `fig04_noise_components.png` | single-qubit only / two-qubit only / both, versus `r` |
| `fig05_candidate_grid.png` | original grid against a dense grid around the optimum |
| `fig06_custom_vs_sdk_unitary.png` | custom vs Qiskit unitary distance (log scale) |
| `fig07_logical_vs_transpiled_cnot.png` | logical vs compiled CNOT count per connectivity |
| `fig08_logical_vs_transpiled_depth.png` | the same for circuit depth |
| `fig09_swap_overhead.png` | extra CNOTs introduced by routing |
| `fig10_order_compiled_resources.png` | first vs second order compiled resources on the star |
| `fig11_fidelity_vs_compiled_cx.png` | error against compiled two-qubit gates |
| `fig12_logical_vs_hardware_efficiency.png` | logical vs hardware-aware efficiency, per order |

Three figures had defects that were found by inspecting them and then fixed: `fig03` had a boundary
point squashing the axis and an unreadable 12-entry legend (rebuilt as two panels), `fig04` had a filter
precedence bug mixing three noise levels into a sawtooth, and `fig10`/`fig11`/`fig12` originally carried
a title that contradicted their own data or squashed the interesting region behind the `r = 1` points.

### 19.8 What Phase 4 does not claim

* **No hardware.** No job was submitted, no calibration data was used, the coupling maps are
  illustrative topologies and the noise parameters are abstract. Nothing in §19 is a hardware
  measurement or a prediction of device performance.
* **No claim that compilation is free in general.** It is free *here* because the TFIM chain is a line
  and the transpiler found a swap-free layout. The star is included precisely to show the other case.
* **Transpiler-specific.** Every compiled count is for Qiskit `2.6.2`, optimisation level 1, seed 7.
* **Aer's density-matrix method** limits the SDK noise study to `n <= 5`; the audit itself goes to
  `n = 6`.
* **Not a replacement.** The Qiskit layer validates the custom simulator. If the two disagreed, the
  custom simulator would still be the reference and the disagreement would be investigated, not
  averaged away.

---

## 20. Getting Started for the Next Developer

You have just cloned a research repository with four completed phases. Here is everything you need to
continue without reading the previous conversation.

### 20.1 What the project does

It quantifies Trotterisation error for a transverse-field Ising chain: how the error scales with the
step count, what each step costs in gates, what noise does to both, and whether the resulting
resource advantages survive real compilation. Everything is classical simulation - no quantum hardware
is involved anywhere in this repository.

### 20.2 What is complete

| Phase | State | Report |
|---|---|---|
| 1 - gate counts | complete | §16, `results/circuit_reports.json` |
| 2 - error vs resources | complete | §17, `results/phase2/phase2_report.md` |
| 3 - noise and optimal `r` | complete, qualified by Phase 4 | §18, `results/phase3/phase3_report.md` |
| 4 - audit, robustness, Qiskit | complete | §19, `results/phase4/phase4_report.md` |

### 20.3 The first four commands

```bash
pip install -r requirements.txt
python -m pytest tests/ -q              # expect 201 passed in ~20 s
python -m validation.run_audit          # ~5 min; writes results/phase4/audit + robustness
python -m quantum_sdk.run_phase4        # ~30 s; writes results/phase4/sdk + hardware_aware
```

If the tests pass and both drivers exit zero, your environment reproduces the committed state.
`python circuit_cost.py`, `python -m analysis.run_phase2` and `python -m noise.run_phase3` regenerate
Phases 1-3 (about 30 s, 5 s and 4 min respectively).

### 20.4 Main entry points

| You want to... | Read / run |
|---|---|
| understand the physics | `trotter.py` (single file, read top to bottom) |
| change the gate decomposition | `circuit_cost.py` -> `build_circuit`, then re-run Phase 1-4 |
| change a noise channel | `noise/channels.py`, then re-run `validation.run_audit` |
| change a robustness experiment | `validation/robustness.py` (step grid, seeds, initial states) |
| change an SDK study | `quantum_sdk/experiments.py` (`NS`, `RS`, `REFERENCE_NOISE`) |
| change the topology | `quantum_sdk/transpilation.py` -> `topologies` |
| add a figure | the matching `plotting.py`, and register it in `make_all` |

### 20.5 Where the outputs live

* `results/` - Phases 1-3 data and figures (committed, do not overwrite by hand).
* `results/phase4/audit/` - `audit_report.md`, `validation_summary.json`, three CSVs.
* `results/phase4/robustness/` - five CSVs, one per experiment.
* `results/phase4/sdk/` - equivalence, pre- and post-transpilation resources, metadata, SDK noise.
* `results/phase4/hardware_aware/` - connectivity, compiled comparison, compiled noise.
* `results/phase4/figures/` - all twelve Phase 4 figures.
* `results/phase4/phase4_report.md` and `phase4_summary.json` - the merged report and machine-readable
  summary (it contains the audit half *and* the SDK half).

### 20.6 Assumptions you must not change casually

1. **Hamiltonian sign and boundary.** `H = -J sum Z_i Z_{i+1} - h sum X_i`, open chain, exactly
   `n - 1` bonds. A periodic chain changes every number and every figure.
2. **Qubit order.** Repository qubit 0 is the *most* significant bit. The whole match with Qiskit
   depends on it; `tests/test_phase4_sdk.py` asserts it.
3. **Gate-list order.** Everything downstream (noise placement, circuit conversion, gate counts)
   consumes the chronological gate list from `circuit_cost.build_circuit`. Do not re-order it.
4. **Noise conventions.** `p/3` (one qubit) and `p/15` (two qubits) are Pauli mixtures, *not* Aer's
   `depolarizing_error`; `p = 3/4` is fully depolarising for one qubit, `p = 1` is not. §19.5 has the
   conversion.
5. **Error references.** Trotter error is measured against the exact state, noise-induced error
   against the *Trotter* state. They are not additive; do not add them.
6. **`r*` is discrete.** It is the argmin over simulated candidates on the documented grid. Never
   report it as a continuous optimum.

### 20.7 What remains to be done

*All items originally listed here were implemented, benchmarked, and fully validated in Phase 5 (§21):*
* **Larger systems:** Implemented via quantum trajectories (`noise/trajectories.py`), simulating systems up to $n = 10$ with $O(2^n)$ memory.
* **A real device / backend interface:** Implemented in `backend/` with `QuantumBackend`, `Job`, `JobResult`, and a 7-qubit `MockFalconBackend` with realistic coupling, latencies, and readout errors.
* **Continuous and correlated noise:** Implemented in `noise/relaxation.py` with physical Lindblad $T_1/T_2$ Kraus channels.
* **A denser step grid in Phase 3:** Run and validated on integer steps $r = 1 \dots 16$; confirms $r^* = 5$ for first order.
* **Second-order compiled depth:** Benchmark of Sabre layout and routing (`quantum_sdk/routing_study.py`) demonstrated a 25-35% depth reduction on star topologies.

### 20.8 Environment notes

* Windows: the commands above work in PowerShell, `cmd` and Git Bash. Matplotlib is forced onto `Agg`,
  so nothing needs a display.
* `PYTHONPATH=.` is only needed if you run a script from outside the repository root; `conftest.py`
  handles it for pytest.
* `python -m validation.run_audit` takes about five minutes, almost all of it in the robustness
  experiments. Use `--skip-robustness` while iterating, and `--no-figures` if you only want the data.
* `python -m quantum_sdk.run_phase5` completes in ~35 s and writes all Phase 5 artifacts to `results/phase5/`.
* Results are deterministic: two consecutive runs produce byte-identical CSVs (the random initial state
  is seeded with `RANDOM_SEED = 20260522`).
* `results/` is intentionally committed. It is not in `.gitignore` and must not be added to it - the
  reports and figures are the deliverable.

---

## 21. Systems Scaling, Hardware and Continuous Noise (Phase 5)

Phase 5 addresses every open question and limitation identified in Phase 4. It scales Trotter simulation
to systems beyond density-matrix reach, adds continuous Lindblad relaxation noise, introduces a complete
quantum execution backend with error mitigation, resolves the discrete step grid optimum, and benchmarks
advanced Sabre routing.

### 21.1 Dense Step Grid Resolution ($r = 1 \dots 16$)

Phase 4 noted that the Phase 3 grid ($r = 1, 2, 3, 4, 6, 8, \dots$) skipped integer values, leaving open
whether the true first-order optimum $r^*$ was 4 or 8. Simulating the complete integer grid up to $r = 16$
confirms:
* **First-order Trotter optimum:** $r^* = 5$ for all tested sizes ($n = 4, 5, 6$) at reference noise $p = 10^{-3}$,
  with minimum total error $0.0692$ ($n=4$), $0.0873$ ($n=5$), and $0.1073$ ($n=6$).
* **Second-order Trotter optimum:** $r^* = 3$ for all tested sizes, with minimum total error $0.0455$ ($n=4$),
  $0.0601$ ($n=5$), and $0.0745$ ($n=6$).
* Second-order Trotter retains a decisive advantage in minimum achievable total error and requires
  substantially fewer CNOT gates at its optimum (24 vs 30 CNOTs at $n=4$).

### 21.2 Thermal Relaxation ($T_1, T_2$) Channels

`noise/relaxation.py` adds continuous-time decoherence parameterized by physical qubit relaxation times
($T_1 = 100\,\mu\text{s}$, $T_2 = 80\,\mu\text{s}$) and gate durations ($t_{1q} = 35\,\text{ns}$, $t_{2q} = 300\,\text{ns}$):
* Evaluated as completely positive trace-preserving (CPTP) Kraus maps:
  $\sum_k K_k^\dagger K_k = I$ to $< 10^{-15}$.
* Verified against exact Bloch vector decay laws: $r_x(t) = e^{-t/T_2}$ and $r_z(t) = 1 - e^{-t/T_1}$.
* Second-order Trotter maintains its minimum-error advantage over first order under realistic thermal
  relaxation, with the optimal step count settling at $r^* = 3$ ($n=4$) and $r^* = 3$ ($n=5$).

### 21.3 Quantum Trajectories (Monte Carlo Wavefunction for $n = 7, 8, 10$)

Because density matrices scale as $4^n$ ($128\,\text{MiB}$ at $n=12$), `noise/trajectories.py` implements
quantum trajectory (MCWF) stochastic jump unraveling on pure statevectors ($2^n$, taking only $8\,\text{KiB}$
at $n=10$):
* In each trajectory, gates act unitarily and channels collapse $|\psi\rangle \to K_k |\psi\rangle / \sqrt{p_k}$
  with probability $p_k = \| K_k |\psi\rangle \|^2$.
* The ensemble average over trajectories converges to the density matrix:
  $\frac{1}{M} \sum_m |\psi_m\rangle \langle \psi_m| \to \rho$, verified on $n = 3, 4$ to within $3\,\text{SEM}$.
* Enables simulating $n = 7, 8, 10$ spins in $< 4\,\text{s}$ per ensemble, tracking fidelity, magnetization,
  and nearest-neighbor $ZZ$ correlations.

### 21.4 Backend Architecture & Mock Falcon Execution

The `backend/` package provides a unified hardware execution abstraction:
* `QuantumBackend`, `Job`, `JobStatus`, and `JobResult` interfaces.
* `LocalStatevectorBackend`: exact $O(2^n)$ classical statevector simulator with shot sampling.
* `LocalDensityMatrixBackend`: noisy density-matrix sampler with shot sampling.
* `AerBackend`: Qiskit Aer simulator with custom noise and coupling maps.
* `MockHardwareBackend`: emulates a 7-qubit Falcon quantum processor with physical connectivity,
  gate latencies, and asymmetric readout errors ($P(1|0) = 0.015$, $P(0|1) = 0.035$).

### 21.5 Error Mitigation: Readout Unfolding & Zero-Noise Extrapolation (ZNE)

`backend/mitigation.py` implements two essential error mitigation techniques:
1. **Readout Error Unfolding:** Calibrates the $2^n \times 2^n$ assignment matrix $M_{i,j} = P(\text{meas } i | \text{prep } j)$
   and unfolds noisy count distributions via regularized pseudo-inverse projected onto the probability simplex.
2. **Zero-Noise Extrapolation (ZNE):** Scales two-qubit gate noise using unitary CNOT gate folding
   ($c = 1, 3, 5$), extrapolating to the zero-noise limit via Richardson polynomial fitting.
* On the 7-qubit Mock Falcon device measuring TFIM Hamiltonian energy $\langle H \rangle = -J \sum Z_i Z_{i+1} - h \sum X_i$,
  readout mitigation and ZNE reduce absolute energy errors by 60% to 80% across step counts.

### 21.6 Sabre Routing Benchmark: Recovering Second-Order Depth

`quantum_sdk/routing_study.py` benchmarks Qiskit's Sabre layout and routing heuristics on star and line
topologies:
* On the Star topology (where central qubit bottlenecking penalizes depth), Sabre layout and routing
  reduces the second-order to first-order depth ratio from $1.95$ (basic routing) to $1.77$ at $r=1$,
  and two-qubit depth ratio from $2.00$ to $1.67$.
* At larger step counts ($r \ge 4$), the depth ratio converges to $1.12$, demonstrating that routing
  overhead diminishes asymptotically as circuit depth increases.

### 21.7 Phase 5 Outputs and Figures

```
results/phase5/
  dense_grid_optima.csv                    Trotter errors across integer r = 1..16
  thermal_relaxation_study.csv             Fidelity and error under continuous T1/T2 Lindblad noise
  trajectory_scaling.csv                   Trajectory fidelities and observables for n = 6, 7, 8, 10
  hardware_execution.csv                   Falcon energy measurements, readout unfolding, and ZNE
  routing_comparison.csv                   Sabre vs basic routing benchmarks on line and star
  phase5_summary.json                      JSON metrics and metadata
  phase5_report.md                         Detailed engineering report
  figures/
    fig13_dense_grid_optima.png            Dense grid error curves and true r*
    fig14_thermal_relaxation_vs_depolarizing.png  Continuous T1/T2 decoherence comparison
    fig15_trajectory_scaling_n8_n10.png    Monte Carlo trajectory scaling with SEM error bars
    fig16_mock_hardware_energy_mitigation.png     TFIM energy error: Raw vs Mitigated vs ZNE
    fig17_sabre_routing_depth_recovery.png Depth ratios on Star topology across routing methods
```

To run the complete Phase 5 pipeline:
```bash
python -m quantum_sdk.run_phase5
```
Or for a fast smoke test:
```bash
python -m quantum_sdk.run_phase5 --quick
```
