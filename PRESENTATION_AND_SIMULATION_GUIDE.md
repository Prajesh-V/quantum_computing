# Quantum Simulation Guide & 2-Minute Presentation Script

This document contains:
1. **Part 1:** Step-by-Step Guide on How to Run Every Type of Simulation
2. **Part 2:** Word-for-Word 2-Minute Oral Presentation Script (with timing and slide cues)
3. **Part 3:** Quick Q&A Defense Sheet for Follow-up Questions

---

# Part 1: How to Run Simulations

This repository provides multiple ways to simulate depending on your situation:

```
                               HOW TO SIMULATE
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
   [1] QUICK CLI                [2] WIZARD (UI)              [3] BATCH RUNNER
  python simulate.py            interactive.bat                  run.bat
  (Instant custom numbers)      (Menu prompts for values)     (Executes whole project)
```

---

### Method A: Interactive Wizard (Best for Live Professor Demos)

Whenever your professor gives you values in real time, run:
```cmd
interactive.bat
```
*(Or in terminal: `python simulate.py`)*

1. The menu appears:
   ```
   Select a Calculation Task:
     [1] Single Simulation (Enter n, r, t, order, noise -> get all metrics & depth)
     [2] Compare 1st vs 2nd Order (Head-to-head accuracy and CNOT budget)
     [3] Find Optimal Step Count r* (Scan step counts under gate noise)
     [4] Target Accuracy Solver (Find min steps/gates for a target error eps)
     [5] Mock Hardware Execution & Error Mitigation (Falcon 7Q + ZNE)
     [6] Large-Scale Quantum Trajectories (n = 6 to 10 qubits with statevectors)
   ```
2. Type the task number (e.g., `1`).
3. Enter the parameters requested by your professor (or press **Enter** to accept the default in brackets `[...]`).
4. The exact results, fidelities, and circuit resources print instantly in a clean ASCII table.

---

### Method B: Single-Line Command Simulation (Fastest)

Run [simulate.py](file:///c:/dev/quantumcomputing/simulate.py) directly with flags:

#### 1. Custom Single Run (Fidelity, Errors, Gate Counts, Depth)
```cmd
python simulate.py -n 4 -r 5 -o 2 -t 1.0 -p 0.001
```
* **`-n`**: Number of qubits (e.g. `4`)
* **`-r`**: Trotter steps (e.g. `5`)
* **`-o`**: Order (`1` for Lie-Trotter, `2` for Strang-Suzuki)
* **`-t`**: Total evolution time (e.g. `1.0`)
* **`-p`**: Depolarising noise rate (e.g. `0.001` = $0.1\%$, `0` = noiseless)

#### 2. Head-to-Head Comparison (Order 1 vs Order 2)
```cmd
python simulate.py --compare -n 4 -r 8 -t 1.0 -p 0.001
```
*Displays a side-by-side comparison table showing error reduction and the fixed $+2(n-1)$ CNOT overhead.*

#### 3. Find the Optimal Step Count ($r^*$) for a Given Noise Rate
```cmd
python simulate.py --optimal -n 4 -p 0.001
```
*Scans $r \in [1 \dots 24]$ and highlights the U-curve sweet spot where truncation error crosses gate noise.*

#### 4. Target Accuracy Solver (Find Required Steps & Gates)
```cmd
python simulate.py --target-error 0.005 -n 4
```
*Solves for the minimum $r$ and CNOT budget needed to achieve total error $\le 0.005$.*

#### 5. Real Hardware Execution with Readout Unfolding & ZNE
```cmd
python simulate.py --hardware falcon -n 4 -r 2 --shots 2048
```
*Simulates the IBM Falcon 7-qubit processor, returning raw energy, readout-mitigated energy, and Zero-Noise Extrapolated energy.*

#### 6. Large-Scale Systems ($n = 7, 8, 10$ Qubits)
```cmd
python simulate.py -n 8 -r 2 -o 2 -p 0.001
```
*Automatically switches to Monte Carlo Wavefunction (MCWF) statevector trajectories to bypass the $4^n$ memory bottleneck.*

---

### Method C: Full Project Pipeline Runner

To verify the entire project from Unit Tests up to Phase 5:
```cmd
run.bat
```
* Runs all 232 pytest unit tests.
* Runs Phase 1 (Analytical Resource Estimation).
* Runs Phase 2 (Gate Budget Accuracy Laws).
* Runs Phase 3 (Density Matrix Noise Sweeps).
* Runs Phase 4 (Independent Audit & SDK Transpilation).
* Runs Phase 5 (Dense Grid, Thermal Relaxation, Trajectories, Falcon ZNE, Sabre Routing).
* **Execution time:** ~25–30 seconds.

---

# Part 2: 2-Minute Presentation Script

* **Target Duration:** Exactly 2 minutes (approx. 270 words spoken at a clear, confident pace of 135 words/minute).
* **Goal:** Hook the professor, explain the core physical dilemma, showcase your empirical breakthroughs, and demonstrate the interactive tool.

---

### [0:00 - 0:25] The Problem & The Core Dilemma
> *"Good morning, Professor. Today I am presenting our work on **digital Trotterised quantum simulation of spin chains under physical hardware noise**.*
>
> *When simulating quantum materials like the Transverse-Field Ising Model on a digital quantum computer, we face a fundamental trade-off. Increasing the Trotter step count $r$ reduces algorithmic truncation error. However, on NISQ hardware, more steps mean more CNOT gates and deeper circuits, causing physical decoherence to rapidly destroy quantum fidelity. This creates a non-monotonic, U-shaped error curve with an interior optimum step count, $r^*$."*

### [0:25 - 0:55] Theory & Analytical Resource Invariants
> *"We started by proving exact, closed-form resource invariants. Using half-step merging, we proved that second-order Strang-Suzuki formulas require **$2(n-1)r + 2(n-1)$ CNOTs**, incurring a fixed overhead of exactly $2(n-1)$ CNOTs—just 6 CNOTs for 4 qubits—that never grows with $r$."*

### [0:55 - 1:30] Key Scientific Findings
> *"Next, by replacing coarse power-of-two grids with dense integer sweeps, we resolved a key artefact in the literature: the true optimum at $p = 10^{-3}$ is **$r^* = 5$ for first order**, but **$r^* = 3$ for second order**. Second order achieves a **34.3% lower total error** because its faster $O(1/r^2)$ convergence allows it to reach minimal Trotter error before gate noise accumulates.*
>
> *We extended this to continuous Lindblad thermal relaxation ($T_1 = 50\,\mu\text{s}, T_2 = 70\,\mu\text{s}$) and demonstrated that the second-order advantage remains robust.*
>
> *To overcome the $O(4^n)$ density matrix memory limit, we developed a Monte Carlo Wavefunction trajectory simulator, scaling calculations up to **10 qubits** in just two seconds."*

### [1:30 - 1:55] Hardware Mitigation & Compiler Routing
> *"Finally, we evaluated execution on an IBM Falcon 7-qubit architecture. Combining simplex-projected readout error unfolding with polynomial Zero-Noise Extrapolation via CNOT folding, we recovered **25% to 38%** of the ground-state energy error. Furthermore, applying SABRE heuristic routing on constrained star topologies suppressed two-qubit depth inflation from $2.0\times$ down to $1.36\times$."*

### [1:55 - 2:00] Closing & Live Demo Offer
> *"We have built an interactive simulation tool and an automated test suite of 232 passing tests. I would be glad to run any custom parameters you'd like to test. Thank you!"*

---

# Part 3: Quick Q&A Defense Sheet (For Professor's Questions)

| Question | 10-Second Crisp Answer |
|---|---|
| **"Why does total error go up after $r^*$?"** | *"Because every extra step adds $2(n-1)$ CNOT gates. Past $r^*$, the hardware noise accumulated from additional gates is greater than the minuscule algorithmic accuracy gained."* |
| **"Why is the second-order CNOT overhead fixed at $+2(n-1)$?"** | *"Because second order is symmetric: $e^{-i A dt/2} e^{-i B dt} e^{-i A dt/2}$. In consecutive steps, adjacent $A$ half-steps fuse into single full-angle layers, leaving only one extra layer split across the two ends."* |
| **"Does 2nd Order always beat 1st Order?"** | *"At coarse steps ($r=1$) or extreme noise, 1st order can win due to 2nd order's fixed 6-CNOT overhead. But at their optimal points $r^*$, 2nd order achieves ~35% lower error."* |
| **"How does Zero-Noise Extrapolation (ZNE) work?"** | *"Since $\text{CNOT}^2 = I$, we intentionally scale noise by inserting 3 and 5 CNOTs. We measure energy at scale factors 1, 3, and 5, fit a Richardson polynomial curve, and extrapolate back to scale factor 0 (zero noise)."* |
| **"Why use trajectories instead of density matrices for $n=10$?"** | *"A density matrix scales as $4^n$, requiring $16\,\text{MB}$ at $n=10$ and gigabytes for $n=12$. Quantum trajectories track pure $2^n$ statevectors ($8\,\text{KB}$ at $n=10$) with stochastic Kraus quantum jumps, reproducing the master equation within $3\,\text{SEM}$."* |
| **"What Hamiltonian did you simulate?"** | *"The 1D Transverse-Field Ising Model (TFIM): $H = -J \sum Z_i Z_{i+1} - h \sum X_i$ with open boundaries, which models quantum ferromagnetism and phase transitions."* |
