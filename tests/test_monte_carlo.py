"""Tests for stochastic distributions and Monte Carlo simulation engine."""

import pytest
import numpy as np
from stochastic_econ.simulation import (
    BetaPERTParam,
    ConstantParam,
    LogNormalParam,
    MonteCarloSimulator,
    NormalParam,
    TriangularParam,
    UniformParam,
)


def test_distribution_samplers(rng):
    """Verify statistical moments and bounds of custom parameter distributions."""
    n = 10_000

    # Uniform
    u = UniformParam(10.0, 20.0).sample(n, rng)
    assert np.all(u >= 10.0) and np.all(u <= 20.0)
    assert np.isclose(np.mean(u), 15.0, atol=0.2)

    # Triangular
    tri = TriangularParam(10.0, 15.0, 20.0).sample(n, rng)
    assert np.all(tri >= 10.0) and np.all(tri <= 20.0)
    assert np.isclose(np.mean(tri), 15.0, atol=0.2)

    # Beta-PERT: (a + 4m + b) / 6 = (10 + 4*20 + 40) / 6 = 130 / 6 = 21.667
    pert = BetaPERTParam(10.0, 20.0, 40.0, gamma=4.0).sample(n, rng)
    assert np.all(pert >= 10.0) and np.all(pert <= 40.0)
    expected_mean = (10.0 + 4.0 * 20.0 + 40.0) / 6.0
    assert np.isclose(np.mean(pert), expected_mean, atol=0.3)


def test_monte_carlo_simulation_reproducibility():
    """Verify Monte Carlo simulation produces deterministic outputs when seeded."""
    sim = MonteCarloSimulator(
        horizon_years=5,
        capex_dist=NormalParam(100_000.0, 5_000.0),
        revenue_dist=BetaPERTParam(40_000.0, 50_000.0, 70_000.0),
        opex_dist=NormalParam(10_000.0, 1_000.0),
        tax_rate=0.25,
        marr_dist=ConstantParam(0.10),
    )

    res1 = sim.run(n_trials=5_000, seed=123)
    res2 = sim.run(n_trials=5_000, seed=123)

    assert np.isclose(res1.mean_npv, res2.mean_npv)
    assert np.isclose(res1.std_npv, res2.std_npv)
    assert np.isclose(res1.var_95, res2.var_95)
    assert np.isclose(res1.cvar_95, res2.cvar_95)
    assert np.allclose(res1.npv_samples, res2.npv_samples)


def test_risk_metrics_properties():
    """Verify mathematical properties of VaR, CVaR, and probability of loss."""
    sim = MonteCarloSimulator(
        horizon_years=5,
        capex_dist=NormalParam(120_000.0, 10_000.0),
        revenue_dist=NormalParam(35_000.0, 8_000.0),
        opex_dist=NormalParam(15_000.0, 2_000.0),
        tax_rate=0.25,
        marr_dist=ConstantParam(0.12),
    )
    res = sim.run(n_trials=10_000, seed=42)

    # 1. Probability of loss is bounded in [0, 1]
    assert 0.0 <= res.prob_loss <= 1.0

    # 2. CVaR(95%) >= VaR(95%)
    assert res.cvar_95 >= res.var_95

    # 3. VaR(99%) >= VaR(95%) and CVaR(99%) >= CVaR(95%)
    assert res.var_99 >= res.var_95
    assert res.cvar_99 >= res.cvar_95

    # 4. Percentiles monotonically increase
    perc_vals = [
        res.percentiles["1%"],
        res.percentiles["5%"],
        res.percentiles["10%"],
        res.percentiles["25%"],
        res.percentiles["50%"],
        res.percentiles["75%"],
        res.percentiles["90%"],
        res.percentiles["95%"],
        res.percentiles["99%"],
    ]
    assert np.all(np.diff(perc_vals) >= 0.0)
