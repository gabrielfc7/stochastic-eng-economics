"""Comprehensive simulation and capital budgeting valuation benchmark script."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from stochastic_econ.cashflow import CashFlowEngine, MACRSDepreciation
from stochastic_econ.simulation import (
    BetaPERTParam,
    ConstantParam,
    MonteCarloSimulator,
    NormalParam,
    UniformParam,
)
from stochastic_econ.optimization import (
    AssetReplacementOptimizer,
    CapitalRationingOptimizer,
    ProjectCandidate,
)
from stochastic_econ.reporting import SensitivityAnalyzer


def main():
    print("=" * 75)
    print("STOCHASTIC ENGINEERING ECONOMICS & CAPITAL BUDGETING BENCHMARK")
    print("=" * 75)

    # 1. Deterministic Baseline Valuation
    print("\n[1] Deterministic Project Baseline (7-Year Chemical Plant Expansion)...")
    base_engine = CashFlowEngine(
        horizon_years=7,
        initial_capex=250_000.0,
        revenues=110_000.0,
        operating_costs=30_000.0,
        tax_rate=0.25,
        working_capital=20_000.0,
        salvage_value=35_000.0,
        depreciation_schedule=MACRSDepreciation(recovery_period=7),
    )
    det_val = base_engine.evaluate(marr=0.12)
    print(f"  Baseline NPV                 : ${det_val.npv:,.2f}")
    print(f"  Internal Rate of Return (IRR): {det_val.irr * 100:.2f}%" if det_val.irr else "  IRR: None")
    print(f"  Profitability Index (PI)     : {det_val.profitability_index:.3f}")
    print(f"  Discounted Payback Period    : {det_val.discounted_payback_years:.2f} years")

    # 2. Monte Carlo Risk & Value-at-Risk
    print("\n[2] Running Vectorized Monte Carlo Simulation (20,000 trials)...")
    sim = MonteCarloSimulator(
        horizon_years=7,
        capex_dist=NormalParam(250_000.0, 20_000.0, clip_min=180_000.0),
        revenue_dist=BetaPERTParam(low=75_000.0, mode=110_000.0, high=155_000.0),
        opex_dist=NormalParam(30_000.0, 4_000.0, clip_min=18_000.0),
        tax_rate=0.25,
        marr_dist=UniformParam(0.10, 0.14),
        salvage_dist=NormalParam(35_000.0, 8_000.0, clip_min=10_000.0),
        working_capital_ratio=0.08,
        macrs_period=7,
    )
    sim_res = sim.run(n_trials=20_000, seed=42)
    print(sim_res.summary())

    # 3. Deterministic Tornado Sensitivity Analysis
    print("\n[3] Running One-at-a-Time (OAT) Tornado Sensitivity Swings (+/- 20%)...")
    analyzer = SensitivityAnalyzer(
        horizon_years=7,
        base_capex=250_000.0,
        base_revenue=110_000.0,
        base_opex=30_000.0,
        base_marr=0.12,
        base_tax_rate=0.25,
        base_salvage=35_000.0,
        macrs_period=7,
    )
    tornado_df = analyzer.compute_tornado(perturbation_pct=0.20)
    print("\nTornado Sensitivity Ranking by NPV Swing:")
    print(tornado_df[["Parameter", "BaseValue", "NPV_at_Low", "NPV_at_High", "SwingRange"]].to_string(
        index=False,
        float_format=lambda x: f"${x:,.2f}",
    ))

    be_revenue = analyzer.compute_breakeven_revenue()
    print(f"\n  Breakeven Annual Revenue (NPV = $0): ${be_revenue:,.2f}")

    # 4. Economic Service Life (ESL) Optimization
    print("\n[4] Asset Replacement: Economic Service Life (ESL) Evaluation...")
    repl_opt = AssetReplacementOptimizer(
        initial_cost=120_000.0,
        salvage_schedule=[85_000, 60_000, 42_000, 28_000, 18_000, 10_000],
        operating_costs=[14_000, 18_000, 24_000, 32_000, 42_000, 55_000],
        interest_rate=0.12,
    )
    esl_res = repl_opt.compute_esl()
    print(f"  Optimal Service Life (ESL): {esl_res.optimal_life_years} years")
    print(f"  Minimum EUAC              : ${esl_res.minimum_euac:,.2f}/year")

    # 5. Capital Budgeting & Rationing (0-1 Knapsack vs PI)
    print("\n[5] Capital Rationing: Optimal Portfolio Allocation under Budget Constraint...")
    projects = [
        ProjectCandidate("Project Alpha", capex=80_000, npv=65_000),
        ProjectCandidate("Project Beta", capex=70_000, npv=55_000),
        ProjectCandidate("Project Gamma", capex=50_000, npv=42_000),
        ProjectCandidate("Project Delta", capex=40_000, npv=31_000),
        ProjectCandidate("Project Epsilon", capex=30_000, npv=22_000),
    ]
    budget_limit = 140_000.0
    budget_opt = CapitalRationingOptimizer(projects, total_budget=budget_limit)
    exact_alloc = budget_opt.solve_exact_knapsack()
    pi_alloc = budget_opt.solve_profitability_index_heuristic()

    print(f"  Total Budget Available: ${budget_limit:,.2f}")
    print(f"  Exact 0-1 Knapsack NPV: ${exact_alloc.total_npv_achieved:,.2f} (Selected: {exact_alloc.selected_projects})")
    print(f"  Greedy PI Heuristic NPV: ${pi_alloc.total_npv_achieved:,.2f} (Selected: {pi_alloc.selected_projects})")
    gap = exact_alloc.total_npv_achieved - pi_alloc.total_npv_achieved
    print(f"  Knapsack Optimality Advantage: +${gap:,.2f}")

    print("\n" + "=" * 75)
    print("ALL VALUATIONS & OPTIMIZATIONS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
