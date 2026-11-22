"""Tests for the advanced routing study (Sabre vs Basic routing)."""

import pytest

from quantum_sdk.routing_study import (
    ROUTING_METHODS,
    compare_routing_strategies,
    evaluate_second_order_recovery,
    run_routing_benchmark,
)


def test_routing_benchmark_line_topology():
    res = run_routing_benchmark(
        n=4,
        r=2,
        order=1,
        topology_name="line",
        method_config=ROUTING_METHODS[0],
    )
    assert res["system_size"] == 4
    assert res["trotter_steps"] == 2
    assert res["cnot_count"] > 0
    assert res["total_depth"] > 0


def test_routing_benchmark_star_topology_sabre():
    res_basic = run_routing_benchmark(
        n=4,
        r=2,
        order=2,
        topology_name="star",
        method_config=ROUTING_METHODS[0],  # basic_opt1
    )
    res_sabre = run_routing_benchmark(
        n=4,
        r=2,
        order=2,
        topology_name="star",
        method_config=ROUTING_METHODS[2],  # sabre_opt3
    )
    assert res_basic["cnot_count"] > 0
    assert res_sabre["cnot_count"] > 0
    # Sabre with opt level 3 should achieve <= depth than basic opt 1
    assert res_sabre["total_depth"] <= res_basic["total_depth"]


def test_evaluate_second_order_recovery():
    bench_rows = compare_routing_strategies(
        ns=(4,),
        rs=(2,),
        topologies_to_test=("line", "star"),
        methods=ROUTING_METHODS[:2],
    )
    eval_rows = evaluate_second_order_recovery(bench_rows)
    assert len(eval_rows) > 0
    for r in eval_rows:
        assert "depth_ratio_o2_o1" in r
        assert "cnot_ratio_o2_o1" in r
        assert isinstance(r["recovered_depth_advantage"], (bool, int))


def test_invalid_topology_raises():
    with pytest.raises(ValueError, match="Unknown topology"):
        run_routing_benchmark(
            n=4,
            r=1,
            order=1,
            topology_name="nonexistent_topology",
            method_config=ROUTING_METHODS[0],
        )
