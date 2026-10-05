"""Tests for Economic Service Life (ESL) asset replacement optimization."""

import pytest
import numpy as np
from stochastic_econ.optimization import AssetReplacementOptimizer


def test_economic_service_life_minimization():
    """Verify that ESL analysis identifies the minimum EUAC year."""
    # Machine costs $50,000; Salvage values drop rapidly; OM costs escalate
    salvage = [35_000, 25_000, 18_000, 12_000, 8_000]
    om = [5_000, 8_000, 12_000, 17_000, 24_000]

    optimizer = AssetReplacementOptimizer(
        initial_cost=50_000.0,
        salvage_schedule=salvage,
        operating_costs=om,
        interest_rate=0.10,
    )
    result = optimizer.compute_esl()

    df = result.annual_cost_table
    assert len(df) == 5
    assert result.optimal_life_years in [1, 2, 3, 4, 5]

    # Verify that the minimum EUAC matches the table's minimum value
    assert np.isclose(result.minimum_euac, df["EUAC"].min())


def test_challenger_defender_decision():
    """Verify Challenger vs Defender replacement decision rule."""
    decision_replace = AssetReplacementOptimizer.compare_challenger_defender(
        defender_current_market_value=15_000.0,
        defender_salvage_next_year=8_000.0,
        defender_om_next_year=12_000.0,
        challenger_min_euac=16_000.0,
        interest_rate=0.10,
    )
    # Loss in value = 7,000; Interest = 1,500; OM = 12,000 -> Marginal Cost = 20,500 > 16,000
    assert "Replace immediately" in decision_replace.recommendation

    decision_retain = AssetReplacementOptimizer.compare_challenger_defender(
        defender_current_market_value=15_000.0,
        defender_salvage_next_year=12_000.0,
        defender_om_next_year=3_000.0,
        challenger_min_euac=16_000.0,
        interest_rate=0.10,
    )
    # Loss in value = 3,000; Interest = 1,500; OM = 3,000 -> Marginal Cost = 7,500 <= 16,000
    assert "Retain Defender" in decision_retain.recommendation
