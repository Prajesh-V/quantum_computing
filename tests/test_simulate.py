"""Unit tests for the interactive simulation and evaluation tool (simulate.py)."""

import json
import pytest
from simulate import (
    run_single_simulation,
    solve_target_accuracy,
    find_optimal_step_count,
    compare_orders,
    main,
)


def test_simulate_single_noiseless():
    """Verify single noiseless Trotter calculation."""
    res = run_single_simulation(n=4, r=4, order=2, t=1.0, noise_prob=0.0)
    assert res["num_qubits"] == 4
    assert res["trotter_steps"] == 4
    assert res["formula_order"] == 2
    assert res["cnot_count"] == 30
    assert res["fidelity"] > 0.99
    assert res["total_error"] < 0.01
    assert res["operator_norm_error"] is not None


def test_simulate_single_noisy():
    """Verify single noisy simulation under depolarizing noise."""
    res = run_single_simulation(n=4, r=4, order=1, t=1.0, noise_prob=0.001)
    assert res["num_qubits"] == 4
    assert res["trotter_steps"] == 4
    assert res["formula_order"] == 1
    assert res["cnot_count"] == 24
    assert res["noise_induced_error"] > 0.0
    assert res["total_error"] > 0.0


def test_simulate_trajectories_scaling():
    """Verify trajectory simulation path for n >= 7."""
    res = run_single_simulation(n=7, r=2, order=2, t=1.0, noise_prob=0.001, num_trajectories=10)
    assert res["num_qubits"] == 7
    assert "trajectories" in res["simulation_method"]
    assert 0.0 <= res["fidelity"] <= 1.0


def test_simulate_cli_json(capsys):
    """Verify CLI json output mode."""
    exit_code = main(["-n", "4", "-r", "2", "-o", "1", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["num_qubits"] == 4
    assert data["cnot_count"] == 12
    assert "fidelity" in data


def test_simulate_cli_compare(capsys):
    """Verify CLI compare mode."""
    exit_code = main(["--compare", "-n", "4", "-r", "4", "-t", "1.0"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "1st Order" in captured.out
    assert "2nd Order" in captured.out


def test_simulate_cli_optimal(capsys):
    """Verify CLI optimal-r scan mode."""
    exit_code = main(["--optimal", "-n", "4", "-p", "0.001"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Optimal Step Count Findings" in captured.out
    assert "1st Order optimal" in captured.out


def test_simulate_cli_target_error(capsys):
    """Verify CLI target error solver mode."""
    exit_code = main(["--target-error", "0.05", "-n", "4"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Required Steps" in captured.out
    assert "CNOT Budget" in captured.out
