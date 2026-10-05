"""Tests for capital rationing and knapsack budgeting optimization."""

import pytest
import numpy as np
from stochastic_econ.optimization import (
    CapitalRationingOptimizer,
    ProjectCandidate,
)
from stochastic_econ.reporting import SensitivityAnalyzer


def test_capital_rationing_knapsack_optimality():
    """Verify 0-1 Knapsack budget allocation beats or matches greedy PI heuristic."""
    candidates = [
        ProjectCandidate(name="Project A", capex=50_000, npv=40_000),  # PI = 1.80
        ProjectCandidate(name="Project B", capex=40_000, npv=35_000),  # PI = 1.875
        ProjectCandidate(name="Project C", capex=30_000, npv=24_000),  # PI = 1.80
        ProjectCandidate(name="Project D", capex=20_000, npv=15_000),  # PI = 1.75
    ]
    budget = 70_000.0

    optimizer = CapitalRationingOptimizer(candidates, total_budget=budget)
    exact_res = optimizer.solve_exact_knapsack()
    pi_res = optimizer.solve_profitability_index_heuristic()

    # Exact knapsack must never violate budget
    assert exact_res.total_capex_spent <= budget
    assert pi_res.total_capex_spent <= budget

    # Exact knapsack must achieve at least as much NPV as greedy heuristic
    assert exact_res.total_npv_achieved >= pi_res.total_npv_achieved
    assert exact_res.total_npv_achieved > 0


def test_sensitivity_analyzer():
    """Verify tornado analysis calculation and breakeven point solving."""
    analyzer = SensitivityAnalyzer(
        horizon_years=5,
        base_capex=100_000.0,
        base_revenue=50_000.0,
        base_opex=15_000.0,
        base_marr=0.10,
        base_tax_rate=0.25,
    )
    tornado_df = analyzer.compute_tornado(perturbation_pct=0.20)

    assert len(tornado_df) == 5
    # Parameters must be sorted by swing range descending
    swings = tornado_df["SwingRange"].to_numpy()
    assert np.all(np.diff(swings) <= 0.0)

    # Breakeven revenue should make NPV close to zero
    be_rev = analyzer.compute_breakeven_revenue()
    assert be_rev > 0.0
    from stochastic_econ.cashflow import CashFlowEngine
    eng = CashFlowEngine(
        horizon_years=5,
        initial_capex=100_000.0,
        revenues=be_rev,
        operating_costs=15_000.0,
        tax_rate=0.25,
    )
    assert np.isclose(eng.evaluate(marr=0.10).npv, 0.0, atol=1e-1)
