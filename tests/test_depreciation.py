"""Tests for depreciation schedules and tax recapture mechanics."""

import pytest
import numpy as np
from stochastic_econ.cashflow import (
    MACRS_RATES,
    MACRSDepreciation,
    StraightLineDepreciation,
    compute_salvage_tax_impact,
)


def test_macrs_tables_sum_to_one():
    """Verify official IRS MACRS recovery rate tables sum to 100%."""
    for period, rates in MACRS_RATES.items():
        total_rate = sum(rates)
        assert np.isclose(total_rate, 1.0, atol=1e-3), f"MACRS {period}-year rates do not sum to 1.0 (got {total_rate})"


def test_macrs_schedule_generation():
    """Verify MACRS schedule generates accurate depreciation and monotonic book values."""
    macrs5 = MACRSDepreciation(recovery_period=5)
    cost = 100_000.0
    sched = macrs5.schedule(cost)

    assert len(sched) == 6  # 5-year recovery has 6 tax years under half-year convention
    assert np.isclose(sched["Depreciation"].sum(), cost, atol=1e-2)
    assert np.isclose(sched.iloc[-1]["BookValue"], 0.0, atol=1e-2)
    # Book value must be strictly non-increasing
    bvs = sched["BookValue"].to_numpy()
    assert np.all(np.diff(bvs) <= 0.0)


def test_straight_line_depreciation():
    sl = StraightLineDepreciation(life_years=5)
    sched = sl.schedule(cost_basis=100_000.0, salvage_value=10_000.0)

    assert len(sched) == 5
    assert np.allclose(sched["Depreciation"], 18_000.0)
    assert np.isclose(sched.iloc[-1]["BookValue"], 10_000.0)


def test_salvage_tax_impact_gain_and_recapture():
    """Test asset disposal above book value and above original cost."""
    # Case 1: Sold above BV but below original cost -> pure depreciation recapture
    tax, net_cf = compute_salvage_tax_impact(
        salvage_realized=40_000.0,
        book_value=20_000.0,
        original_cost=100_000.0,
        tax_rate=0.25,
    )
    # Gain is 20,000; tax = 20,000 * 0.25 = 5,000; net cash flow = 35,000
    assert np.isclose(tax, 5_000.0)
    assert np.isclose(net_cf, 35_000.0)

    # Case 2: Sold above original cost -> recapture + capital gain
    tax_cap, net_cf_cap = compute_salvage_tax_impact(
        salvage_realized=120_000.0,
        book_value=20_000.0,
        original_cost=100_000.0,
        tax_rate=0.25,
        capital_gains_rate=0.20,
    )
    # Recapture = 80,000 @ 25% = 20,000; Cap gain = 20,000 @ 20% = 4,000; Total tax = 24,000
    assert np.isclose(tax_cap, 24_000.0)
    assert np.isclose(net_cf_cap, 96_000.0)

    # Case 3: Sold below book value -> tax deduction on loss
    tax_loss, net_cf_loss = compute_salvage_tax_impact(
        salvage_realized=10_000.0,
        book_value=20_000.0,
        original_cost=100_000.0,
        tax_rate=0.25,
    )
    # Loss = 10,000 -> tax savings = -2,500; net cash flow = 12,500
    assert np.isclose(tax_loss, -2_500.0)
    assert np.isclose(net_cf_loss, 12_500.0)
