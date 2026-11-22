#!/usr/bin/env python3
"""Interactive Trotter Simulation & Evaluation CLI.

Designed for interactive evaluation and professor queries:
Input any parameters (n, r, t, order, noise, target accuracy, hardware)
and immediately get exact results, gate resources, noise fidelities,
optimal step counts, and hardware error mitigation metrics.

Usage:
    python simulate.py                      # Interactive menu wizard
    python simulate.py -n 4 -r 8 -o 2       # Direct command-line run
    python simulate.py --compare -n 4 -t 1  # Compare 1st vs 2nd order
    python simulate.py --optimal -n 4 -p 1e-3 # Find optimal r*
    python simulate.py --target-error 1e-3  # Find steps/gates for target error
    python simulate.py --hardware falcon    # Run mock hardware execution
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import circuit_cost
import trotter
from backend.mitigation import calibrate_readout_matrix, fold_circuit_cnot, mitigate_readout_counts, richardson_extrapolate
from backend.mock_device import create_mock_falcon
from backend.observables import exact_tfim_energy, expectation_z_from_counts, expectation_zz_from_counts, measure_tfim_energy
from noise.density_matrix import purity
from noise.metrics import fidelity_pure, trace_distance
from noise.noisy_simulator import NoiseModel, exact_state, ideal_trotter_state, simulate
from noise.trajectories import simulate_trajectories


def _box(title: str, lines: List[str], width: int = 76) -> str:
    """Format text inside a clean ASCII box."""
    out = []
    out.append("+" + "=" * (width - 2) + "+")
    out.append(f"| {title.center(width - 4)} |")
    out.append("+" + "-" * (width - 2) + "+")
    for line in lines:
        if len(line) > width - 4:
            line = line[: width - 7] + "..."
        out.append(f"| {line.ljust(width - 4)} |")
    out.append("+" + "=" * (width - 2) + "+")
    return "\n".join(out)


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    """Render a simple markdown-style table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(str(val)))
    
    header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)) for row in rows]
    return f"{header_str}\n{sep_str}\n" + "\n".join(row_strs)


def prompt_int(prompt: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Prompt user for an integer with default and validation."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if min_val is not None and val < min_val:
                print(f"  [!] Must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  [!] Please enter a valid integer.")


def prompt_float(prompt: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Prompt user for a float with default and validation."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  [!] Must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  [!] Please enter a valid real number.")


def prompt_choice(prompt: str, choices: List[str], default: str) -> str:
    """Prompt user for one of several choices."""
    choices_lower = [c.lower() for c in choices]
    while True:
        raw = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip().lower()
        if not raw:
            return default.lower()
        if raw in choices_lower:
            return raw
        print(f"  [!] Invalid choice. Select from {choices}.")


def run_single_simulation(
    n: int,
    r: int,
    order: int,
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    noise_prob: float = 0.0,
    p1: Optional[float] = None,
    p2: Optional[float] = None,
    boundary: str = "open",
    use_trajectories_if_large: bool = True,
    num_trajectories: int = 50,
) -> Dict[str, Any]:
    """Execute a single Trotter simulation and compute full metrics."""
    p_1 = noise_prob if p1 is None else p1
    p_2 = noise_prob if p2 is None else p2

    # Gate & circuit resource estimation
    resources = circuit_cost.estimate_resources(
        n=n, r=r, t=t, J=J, h=h, order=order, boundary=boundary, merge_half_steps=True
    )
    
    # Exact Matrix & Unitary Error (for n <= 6)
    op_norm_error = None
    exact_infidelity = None
    if n <= 8:
        A, B, H = trotter.tfim_terms(n, J, h)
        U_exact = trotter.expm(-1j * H * t)
        U_trot = trotter.trotter_unitary(A, B, t, r, order)
        op_norm_error = trotter.operator_error(U_exact, U_trot)
        
        psi0 = np.zeros(2**n, dtype=complex)
        psi0[0] = 1.0
        psi_exact = U_exact @ psi0
        psi_trot = U_trot @ psi0
        exact_infidelity = trotter.infidelity(psi_exact, psi_trot)

    # Simulation with Noise
    if p_1 > 0.0 or p_2 > 0.0:
        model = NoiseModel(kind="gate", single_qubit_error=p_1, two_qubit_error=p_2)
    else:
        model = NoiseModel(kind="none")

    if n <= 6:
        sim_res = simulate(n=n, r=r, order=order, t=t, J=J, h=h, model=model, merge_half_steps=True)
        rho_noisy = sim_res["rho"]
        exact_psi = exact_state(n, t, J, h)
        ideal_psi = ideal_trotter_state(n, r, order, t, J, h)
        
        noisy_fidelity = float(np.real(fidelity_pure(rho_noisy, exact_psi)))
        ideal_fidelity = float(np.clip(abs(np.vdot(ideal_psi, exact_psi)) ** 2, 0.0, 1.0))
        total_error = 1.0 - noisy_fidelity
        trotter_error = 1.0 - ideal_fidelity
        noise_induced_error = float(np.real(1.0 - fidelity_pure(rho_noisy, ideal_psi)))
        pur = float(np.real(purity(rho_noisy)))
        tr_dist = float(np.real(trace_distance(rho_noisy, np.outer(exact_psi, exact_psi.conj()))))
        method = "density_matrix"
        std_err = 0.0
    else:
        # Quantum Trajectories for n >= 7
        traj_res = simulate_trajectories(
            n=n, r=r, order=order, t=t, J=J, h=h, model=model,
            num_trajectories=num_trajectories, seed=42, merge_half_steps=True
        )
        noisy_fidelity = traj_res["fidelity"]
        total_error = traj_res["total_error"]
        trotter_error = 1.0 - traj_res["trotter_fidelity"]
        noise_induced_error = traj_res["noise_induced_error"]
        pur = None
        tr_dist = None
        method = f"mcwf_trajectories (K={num_trajectories})"
        std_err = traj_res["sem_fidelity"]

    return {
        "num_qubits": n,
        "trotter_steps": r,
        "formula_order": order,
        "evolution_time": t,
        "J": J,
        "h": h,
        "p1": p_1,
        "p2": p_2,
        "simulation_method": method,
        "cnot_count": resources.num_cnot,
        "rx_count": resources.num_rx,
        "rz_count": resources.num_rz,
        "total_gates": resources.total_gates,
        "circuit_depth": resources.circuit_depth,
        "two_qubit_depth": resources.two_qubit_depth,
        "operator_norm_error": op_norm_error,
        "state_infidelity": exact_infidelity,
        "fidelity": noisy_fidelity,
        "fidelity_std_err": std_err,
        "total_error": total_error,
        "trotter_error": trotter_error,
        "noise_induced_error": noise_induced_error,
        "purity": pur,
        "trace_distance": tr_dist,
    }


def display_simulation_result(res: Dict[str, Any]) -> None:
    """Print the result of a single simulation in a structured table."""
    lines = [
        f"Model: TFIM Chain (n={res['num_qubits']} spins, J={res['J']}, h={res['h']}, t={res['evolution_time']})",
        f"Trotter: Order {res['formula_order']}, Steps r = {res['trotter_steps']} (dt = {res['evolution_time']/res['trotter_steps']:.4f})",
        f"Noise: p1 = {res['p1']:g}, p2 = {res['p2']:g} [{res['simulation_method']}]",
        "-" * 72,
        f"  * Total Achieved Fidelity (F):     {res['fidelity']:.6f}" + (f" +/- {res['fidelity_std_err']:.4f}" if res['fidelity_std_err'] > 0 else ""),
        f"  * Measured Total Error (1 - F):   {res['total_error']:.6e}",
        f"  * Trotter Truncation Error:       {res['trotter_error']:.6e}",
        f"  * Noise-Induced Decoherence:      {res['noise_induced_error']:.6e}",
    ]
    if res["operator_norm_error"] is not None:
        lines.append(f"  * Operator Norm Error ||U - U_t||: {res['operator_norm_error']:.6e}")
    if res["purity"] is not None:
        lines.append(f"  * Density Matrix Purity Tr(rho^2): {res['purity']:.6f}")
    if res["trace_distance"] is not None:
        lines.append(f"  * Trace Distance D(rho, exact):   {res['trace_distance']:.6e}")

    lines.extend([
        "-" * 72,
        "Circuit Resources:",
        f"  * CNOT Count:      {res['cnot_count']:>6}    * Total Gates:     {res['total_gates']:>6}",
        f"  * Single-Qubit RX: {res['rx_count']:>6}    * Circuit Depth:   {res['circuit_depth']:>6}",
        f"  * Single-Qubit RZ: {res['rz_count']:>6}    * Two-Qubit Depth: {res['two_qubit_depth']:>6}",
    ])
    print()
    print(_box("TROTTER SIMULATION RESULTS", lines))


def compare_orders(n: int, t: float, r: int, p: float = 0.0) -> None:
    """Run head-to-head comparison between Order 1 and Order 2."""
    print(f"\n[*] Evaluating Order 1 vs Order 2 at n={n}, t={t}, r={r}, p={p}...")
    res1 = run_single_simulation(n=n, r=r, order=1, t=t, noise_prob=p)
    res2 = run_single_simulation(n=n, r=r, order=2, t=t, noise_prob=p)

    headers = ["Metric", "1st Order (Lie-Trotter)", "2nd Order (Strang-Suzuki)", "Difference / Ratio"]
    rows = [
        ["Total Error (1 - F)", f"{res1['total_error']:.4e}", f"{res2['total_error']:.4e}",
         f"Order 2 is {res1['total_error']/max(res2['total_error'], 1e-15):.2f}x more accurate" if res1['total_error'] > res2['total_error'] else "Order 1 wins"],
        ["Trotter Truncation", f"{res1['trotter_error']:.4e}", f"{res2['trotter_error']:.4e}",
         f"{res1['trotter_error']/max(res2['trotter_error'], 1e-15):.2f}x reduction"],
        ["Noise-Induced Error", f"{res1['noise_induced_error']:.4e}", f"{res2['noise_induced_error']:.4e}",
         f"+{abs(res2['noise_induced_error'] - res1['noise_induced_error']):.4e} extra noise"],
        ["CNOT Count", f"{res1['cnot_count']}", f"{res2['cnot_count']}", f"+{res2['cnot_count'] - res1['cnot_count']} CNOTs (fixed)"],
        ["Total Gates", f"{res1['total_gates']}", f"{res2['total_gates']}", f"+{res2['total_gates'] - res1['total_gates']} gates"],
        ["Circuit Depth", f"{res1['circuit_depth']}", f"{res2['circuit_depth']}", f"+{res2['circuit_depth'] - res1['circuit_depth']} depth"],
        ["Two-Qubit Depth", f"{res1['two_qubit_depth']}", f"{res2['two_qubit_depth']}", f"+{res2['two_qubit_depth'] - res1['two_qubit_depth']} 2Q depth"],
    ]
    print()
    print(_table(headers, rows))
    
    winner = "2nd Order" if res2["total_error"] < res1["total_error"] else "1st Order"
    print(f"\n=> Conclusion: {winner} achieves the lower error at step count r = {r}.")
    if p > 0:
        print("   Note: At high noise or tight CNOT budgets, 2nd order's fixed CNOT overhead can outweigh its accuracy advantage.")


def find_optimal_step_count(n: int, p: float, t: float = 1.0, max_r: int = 24) -> None:
    """Find the optimal step count r* that minimizes total error."""
    print(f"\n[*] Scanning step counts r = 1..{max_r} at n={n}, t={t}, p={p}...")
    
    orders = (1, 2)
    optima = {}
    
    for order in orders:
        best_r = 1
        min_err = float("inf")
        history = []
        for r in range(1, max_r + 1):
            res = run_single_simulation(n=n, r=r, order=order, t=t, noise_prob=p)
            history.append((r, res["total_error"], res["trotter_error"], res["noise_induced_error"], res["cnot_count"]))
            if res["total_error"] < min_err:
                min_err = res["total_error"]
                best_r = r
        optima[order] = {"r_star": best_r, "min_error": min_err, "history": history}

    headers = ["r", "O1 Total Err", "O1 CNOT", "O2 Total Err", "O2 CNOT", "Best Order"]
    rows = []
    for r in range(1, min(13, max_r + 1)):
        o1_row = optima[1]["history"][r - 1]
        o2_row = optima[2]["history"][r - 1]
        best = "O2" if o2_row[1] < o1_row[1] else "O1"
        rows.append([
            str(r) + (" *" if r == optima[1]["r_star"] or r == optima[2]["r_star"] else ""),
            f"{o1_row[1]:.4e}", str(o1_row[4]),
            f"{o2_row[1]:.4e}", str(o2_row[4]),
            best
        ])
    
    print()
    print(_table(headers, rows))
    print()
    print("=" * 60)
    print(f"Optimal Step Count Findings for n={n}, noise p={p}:")
    print(f"  * 1st Order optimal: r* = {optima[1]['r_star']}  (minimum error: {optima[1]['min_error']:.4e})")
    print(f"  * 2nd Order optimal: r* = {optima[2]['r_star']}  (minimum error: {optima[2]['min_error']:.4e})")
    overall_best = 2 if optima[2]["min_error"] < optima[1]["min_error"] else 1
    gain = (optima[1]["min_error"] - optima[2]["min_error"]) / optima[1]["min_error"] * 100
    print(f"  * Global optimum is Order {overall_best} with r* = {optima[overall_best]['r_star']}")
    if gain > 0:
        print(f"    (Second order achieves {gain:.1f}% lower error than first order)")
    print("=" * 60)


def solve_target_accuracy(n: int, target_eps: float, t: float = 1.0) -> None:
    """Find the minimum r and CNOT count required to achieve error <= target_eps."""
    print(f"\n[*] Solving for target error eps <= {target_eps} at n={n}, t={t}...")
    
    results = {}
    for order in (1, 2):
        found = False
        # Test geometrically up to 2048
        candidates = list(range(1, 33)) + [40, 50, 64, 80, 100, 128, 160, 200, 256, 512, 1024]
        for r in candidates:
            res = run_single_simulation(n=n, r=r, order=order, t=t, noise_prob=0.0)
            if res["trotter_error"] <= target_eps:
                results[order] = (r, res["trotter_error"], res["cnot_count"], res["total_gates"])
                found = True
                break
        if not found:
            results[order] = (">1024", float("nan"), ">10000", ">20000")

    headers = ["Formula Order", "Required Steps (r)", "Achieved Error", "CNOT Budget", "Total Gates"]
    rows = [
        ["1st Order (Lie-Trotter)", str(results[1][0]), f"{results[1][1]:.2e}", str(results[1][2]), str(results[1][3])],
        ["2nd Order (Strang)", str(results[2][0]), f"{results[2][1]:.2e}", str(results[2][2]), str(results[2][3])],
    ]
    print()
    print(_table(headers, rows))
    if isinstance(results[1][2], int) and isinstance(results[2][2], int):
        savings = (results[1][2] - results[2][2]) / results[1][2] * 100
        print(f"\n=> 2nd Order saves {savings:.1f}% CNOT gates compared to 1st order for target error {target_eps}.")


def run_hardware_mitigation_demo(n: int = 4, r: int = 2, shots: int = 2048) -> None:
    """Run circuit execution on Mock Falcon 7Q with Readout Unfolding & ZNE."""
    print(f"\n[*] Initializing Mock Falcon 7-qubit device (coupling, latencies, readout errors)...")
    falcon = create_mock_falcon()
    
    print(f"[*] Calibrating {n}-qubit readout assignment matrix ({shots} shots)...")
    calib = calibrate_readout_matrix(falcon, num_qubits=n, shots=shots, seed=42)
    
    print(f"[*] Executing TFIM Hamiltonian measurement (r={r}, shots={shots})...")
    # Raw execution
    raw = measure_tfim_energy(falcon, n, r, order=1, shots=shots, seed=42, mitigate_readout=False)
    # Readout mitigated
    mit = measure_tfim_energy(falcon, n, r, order=1, shots=shots, seed=42, mitigate_readout=True, assignment_matrix=calib)
    
    # Zero-Noise Extrapolation
    scales = [1, 3, 5]
    zne_vals = []
    base_qc = circuit_cost.build_circuit(n, r, 1.0, order=1, merge_half_steps=True)
    for c in scales:
        folded = fold_circuit_cnot(base_qc, scale_factor=c)
        job = falcon.submit(folded, shots=shots, seed=42 + c)
        unfolded_counts = mitigate_readout_counts(job.result().counts, calib, shots)
        
        # Measure in X basis
        from backend.observables import add_x_basis_rotation
        folded_x = add_x_basis_rotation(folded, n)
        job_x = falcon.submit(folded_x, shots=shots, seed=42 + c + 10)
        unfolded_x = mitigate_readout_counts(job_x.result().counts, calib, shots)
        
        zz = sum(expectation_zz_from_counts(unfolded_counts, q, q + 1) for q in range(n - 1))
        x = sum(expectation_z_from_counts(unfolded_x, q) for q in range(n))
        zne_vals.append(float(-1.0 * zz - 1.0 * x))
        
    zne_energy = richardson_extrapolate(scales, zne_vals, order=1)
    exact_e = exact_tfim_energy(n, 1.0, 1.0, 1.0)
    
    raw_err = abs(raw["measured_energy"] - exact_e)
    mit_err = abs(mit["measured_energy"] - exact_e)
    zne_err = abs(zne_energy - exact_e)

    headers = ["Execution Stage", "Energy <H>", "Exact Ground/State Energy", "Absolute Error", "Error Reduction"]
    rows = [
        ["Raw Hardware (with noise)", f"{raw['measured_energy']:.4f}", f"{exact_e:.4f}", f"{raw_err:.4f}", "Baseline (0%)"],
        ["+ Readout Unfolding Mitigated", f"{mit['measured_energy']:.4f}", f"{exact_e:.4f}", f"{mit_err:.4f}", f"{(raw_err - mit_err)/raw_err*100:+.1f}%"],
        ["+ Zero-Noise Extrapolated (ZNE)", f"{zne_energy:.4f}", f"{exact_e:.4f}", f"{zne_err:.4f}", f"{(raw_err - zne_err)/raw_err*100:+.1f}%"],
    ]
    print()
    print(_table(headers, rows))


def interactive_menu() -> None:
    """Main interactive terminal loop."""
    print("=" * 76)
    print("       TROTTER QUANTUM SIMULATION - INTERACTIVE EVALUATION TOOL")
    print("=" * 76)
    print("Use this tool to input values given by your professor or evaluator")
    print("and instantly calculate circuit resources, Trotter errors, and fidelities.\n")

    while True:
        print("\nSelect a Calculation Task:")
        print("  [1] Single Simulation (Enter n, r, t, order, noise -> get all metrics & depth)")
        print("  [2] Compare 1st vs 2nd Order (Head-to-head accuracy and CNOT budget)")
        print("  [3] Find Optimal Step Count r* (Scan step counts under gate noise)")
        print("  [4] Target Accuracy Solver (Find min steps/gates for a target error eps)")
        print("  [5] Mock Hardware Execution & Error Mitigation (Falcon 7Q + Readout Unfolding + ZNE)")
        print("  [6] Large-Scale Quantum Trajectories (n = 6 to 10 qubits with statevectors)")
        print("  [7] Exit")
        
        choice = input("\nEnter choice [1-7]: ").strip()
        if choice in ("7", "exit", "quit", "q"):
            print("\nExiting. Good luck with your evaluation!")
            break

        if choice == "1":
            print("\n--- [1] Single Simulation Parameters ---")
            n = prompt_int("Number of qubits (n)", default=4, min_val=2, max_val=12)
            r = prompt_int("Trotter steps (r)", default=5, min_val=1)
            order = prompt_int("Formula order (1 or 2)", default=2, min_val=1, max_val=2)
            t = prompt_float("Total evolution time (t)", default=1.0, min_val=0.01)
            p = prompt_float("Depolarizing gate noise p (0 for noiseless)", default=0.001, min_val=0.0, max_val=0.5)
            J = prompt_float("Interaction strength J", default=1.0)
            h = prompt_float("Transverse field h", default=1.0)
            
            print("\nCalculating...")
            res = run_single_simulation(n=n, r=r, order=order, t=t, J=J, h=h, noise_prob=p)
            display_simulation_result(res)

        elif choice == "2":
            print("\n--- [2] Compare 1st vs 2nd Order ---")
            n = prompt_int("Number of qubits (n)", default=4, min_val=2, max_val=8)
            t = prompt_float("Total evolution time (t)", default=1.0, min_val=0.01)
            r = prompt_int("Trotter steps (r)", default=6, min_val=1)
            p = prompt_float("Gate noise probability p (0 for noiseless)", default=0.001, min_val=0.0, max_val=0.2)
            compare_orders(n=n, t=t, r=r, p=p)

        elif choice == "3":
            print("\n--- [3] Find Optimal Step Count r* ---")
            n = prompt_int("Number of qubits (n)", default=4, min_val=2, max_val=6)
            p = prompt_float("Noise probability p", default=0.001, min_val=1e-5, max_val=0.1)
            t = prompt_float("Total evolution time (t)", default=1.0, min_val=0.1)
            max_r = prompt_int("Maximum step count to scan", default=16, min_val=4, max_val=32)
            find_optimal_step_count(n=n, p=p, t=t, max_r=max_r)

        elif choice == "4":
            print("\n--- [4] Target Accuracy Solver ---")
            n = prompt_int("Number of qubits (n)", default=4, min_val=2, max_val=8)
            eps = prompt_float("Target error tolerance eps", default=0.001, min_val=1e-8, max_val=0.5)
            t = prompt_float("Total evolution time (t)", default=1.0, min_val=0.1)
            solve_target_accuracy(n=n, target_eps=eps, t=t)

        elif choice == "5":
            print("\n--- [5] Mock Hardware Execution with ZNE & Readout Mitigation ---")
            n = prompt_int("Number of qubits (n)", default=4, min_val=2, max_val=7)
            r = prompt_int("Trotter steps (r)", default=2, min_val=1, max_val=8)
            shots = prompt_int("Number of measurement shots", default=2048, min_val=500, max_val=10000)
            run_hardware_mitigation_demo(n=n, r=r, shots=shots)

        elif choice == "6":
            print("\n--- [6] Large-Scale Quantum Trajectories (n >= 6) ---")
            n = prompt_int("Number of qubits (n)", default=8, min_val=6, max_val=10)
            r = prompt_int("Trotter steps (r)", default=4, min_val=1, max_val=16)
            order = prompt_int("Formula order (1 or 2)", default=2, min_val=1, max_val=2)
            p = prompt_float("Gate noise probability p", default=0.001, min_val=0.0, max_val=0.1)
            trajs = prompt_int("Number of stochastic trajectories", default=40, min_val=10, max_val=200)
            
            print(f"\nSimulating {trajs} Monte Carlo trajectories for n={n} qubits...")
            res = run_single_simulation(n=n, r=r, order=order, noise_prob=p, num_trajectories=trajs)
            display_simulation_result(res)

        else:
            print("  [!] Please select a number from 1 to 7.")

        input("\nPress Enter to continue...")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive Trotter simulation and evaluation calculator for spin chains."
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive menu wizard")
    parser.add_argument("-n", "--qubits", type=int, default=None, help="Number of qubits/spins")
    parser.add_argument("-r", "--steps", type=int, default=None, help="Trotter step count")
    parser.add_argument("-o", "--order", type=int, default=2, choices=[1, 2], help="Formula order (1 or 2)")
    parser.add_argument("-t", "--time", type=float, default=1.0, help="Total evolution time")
    parser.add_argument("-J", type=float, default=1.0, help="ZZ interaction coupling")
    parser.add_argument("-hf", "--field", type=float, default=1.0, help="Transverse magnetic field h")
    parser.add_argument("-p", "--noise", type=float, default=0.0, help="Depolarizing noise probability")
    parser.add_argument("--compare", action="store_true", help="Run head-to-head comparison of 1st vs 2nd order")
    parser.add_argument("--optimal", action="store_true", help="Find optimal step count r* for given noise")
    parser.add_argument("--target-error", type=float, default=None, help="Solve for min r and CNOTs for target error")
    parser.add_argument("--hardware", choices=["falcon"], default=None, help="Run mock hardware execution")
    parser.add_argument("--shots", type=int, default=2048, help="Shots for hardware simulation")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args(argv)

    # If no flags passed, launch interactive mode
    if len(sys.argv) == 1 or args.interactive:
        try:
            interactive_menu()
            return 0
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return 0

    # Command-line execution modes
    if args.hardware:
        n = args.qubits or 4
        r = args.steps or 2
        run_hardware_mitigation_demo(n=n, r=r, shots=args.shots)
        return 0

    if args.target_error is not None:
        n = args.qubits or 4
        solve_target_accuracy(n=n, target_eps=args.target_error, t=args.time)
        return 0

    if args.optimal:
        n = args.qubits or 4
        p = args.noise if args.noise > 0 else 0.001
        find_optimal_step_count(n=n, p=p, t=args.time)
        return 0

    if args.compare:
        n = args.qubits or 4
        r = args.steps or 6
        compare_orders(n=n, t=args.time, r=r, p=args.noise)
        return 0

    # Default single simulation via CLI
    n = args.qubits or 4
    r = args.steps or 4
    res = run_single_simulation(
        n=n, r=r, order=args.order, t=args.time, J=args.J, h=args.field, noise_prob=args.noise
    )

    if args.json:
        # Convert non-serializable items
        clean_res = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}
        print(json.dumps(clean_res, indent=2))
    else:
        display_simulation_result(res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
