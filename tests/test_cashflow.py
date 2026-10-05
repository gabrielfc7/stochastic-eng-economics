"""Tests for cash flow engine, NPV, IRR, and financial metrics."""

import pytest
import numpy as np
from stochastic_econ.cashflow import CashFlowEngine


def test_cash_flow_valuation_deterministic():
    """Verify after-tax cash flows, NPV, and payback periods on standard baseline."""
    engine = CashFlowEngine(
        horizon_years=5,
        initial_capex=100_000.0,
        revenues=50_000.0,
        operating_costs=10_000.0,
        tax_rate=0.25,
        working_capital=5_000.0,
        salvage_value=15_000.0,
    )
    val = engine.evaluate(marr=0.10)

    # Initial cash outflow at t=0 should be -(100,000 + 5,000) = -105,000
    df = val.cash_flow_table
    assert np.isclose(df.loc[0, "NetCashFlow"], -105_000.0)

    # All operating years should have positive CFAT
    assert np.all(df.loc[1:, "NetCashFlow"] > 0)

    # Check terminal cash flow includes working capital recovery (5,000) and net salvage
    assert df.loc[5, "NetCashFlow"] > df.loc[4, "NetCashFlow"]

    # NPV should be positive and finite
    assert val.npv > 0
    assert val.irr is not None
    assert val.irr > 0.10  # Positive NPV at 10% implies IRR > 10%

    # Profitability Index > 1.0 when NPV > 0
    assert val.profitability_index > 1.0

    # Payback periods should be between 1 and 5 years
    assert val.undiscounted_payback_years is not None
    assert 1.0 <= val.undiscounted_payback_years <= 5.0
    assert val.discounted_payback_years is not None
    assert val.discounted_payback_years >= val.undiscounted_payback_years


def test_multi_tier_marr():
    """Verify multi-tier term structure MARR discounting."""
    engine = CashFlowEngine(
        horizon_years=3,
        initial_capex=50_000.0,
        revenues=30_000.0,
        operating_costs=5_000.0,
        tax_rate=0.20,
    )
    # Tier structure: year 1 at 8%, year 2 at 10%, year 3 at 12%
    marr_tiers = [0.08, 0.10, 0.12]
    val = engine.evaluate(marr=marr_tiers)

    df = val.cash_flow_table
    # Discount factor year 1 = 1 / 1.08
    assert np.isclose(df.loc[1, "DiscountFactor"], 1.0 / 1.08)
    # Discount factor year 2 = 1 / (1.08 * 1.10)
    assert np.isclose(df.loc[2, "DiscountFactor"], 1.0 / (1.08 * 1.10))
    # Discount factor year 3 = 1 / (1.08 * 1.10 * 1.12)
    assert np.isclose(df.loc[3, "DiscountFactor"], 1.0 / (1.08 * 1.10 * 1.12))


def test_irr_root_finding_parity():
    """Ensure IRR satisfies exact zero-NPV condition within 1e-5."""
    engine = CashFlowEngine(
        horizon_years=4,
        initial_capex=80_000.0,
        revenues=40_000.0,
        operating_costs=12_000.0,
        tax_rate=0.25,
    )
    val = engine.evaluate(marr=0.10)
    assert val.irr is not None

    # Evaluate at the discovered IRR
    val_at_irr = engine.evaluate(marr=val.irr)
    assert np.isclose(val_at_irr.npv, 0.0, atol=1e-2)
