"""Vectorized Monte Carlo capital budgeting simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd

from stochastic_econ.cashflow.depreciation import MACRSDepreciation, compute_salvage_tax_impact
from stochastic_econ.simulation.distributions import DistributionParam, ConstantParam


@dataclass
class SimulationResults:
    """Consolidated statistical and risk metrics from Monte Carlo project valuation."""

    n_trials: int
    npv_samples: np.ndarray
    mean_npv: float
    std_npv: float
    median_npv: float
    prob_loss: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    percentiles: dict[str, float]

    def summary(self) -> str:
        """Formatted publication-grade summary string."""
        return (
            f"=== Monte Carlo Capital Budgeting Valuation ({self.n_trials:,} trials) ===\n"
            f"Mean Expected NPV : ${self.mean_npv:,.2f}\n"
            f"Standard Deviation: ${self.std_npv:,.2f}\n"
            f"Median NPV        : ${self.median_npv:,.2f}\n"
            f"Probability of Loss: {self.prob_loss * 100:.2f}%\n"
            f"VaR (95% Conf)    : ${self.var_95:,.2f}\n"
            f"CVaR / Exp Shortfall: ${self.cvar_95:,.2f}\n"
            f"VaR (99% Conf)    : ${self.var_99:,.2f}\n"
            f"CVaR (99% Conf)   : ${self.cvar_99:,.2f}\n"
            f"5th Percentile    : ${self.percentiles['5%']:,.2f}\n"
            f"95th Percentile   : ${self.percentiles['95%']:,.2f}"
        )


class MonteCarloSimulator:
    """Vectorized multi-dimensional Monte Carlo simulation for stochastic capital budgeting.

    Parameters
    ----------
    horizon_years : int
        Project operating lifespan in years.
    capex_dist : DistributionParam
        Initial capital investment distribution at t=0.
    revenue_dist : DistributionParam | Sequence[DistributionParam]
        Annual revenue distribution (single distribution or list of T distributions).
    opex_dist : DistributionParam | Sequence[DistributionParam]
        Annual operating expenses distribution.
    tax_rate : float, default=0.25
        Corporate income tax rate.
    marr_dist : DistributionParam, default=ConstantParam(0.10)
        Discount rate / MARR distribution.
    salvage_dist : DistributionParam, default=ConstantParam(0.0)
        Terminal salvage value distribution at t=T.
    working_capital_ratio : float, default=0.0
        Working capital requirement as fraction of initial capex (recovered at t=T).
    macrs_period : int, default=5
        IRS MACRS recovery period.
    """

    def __init__(
        self,
        horizon_years: int,
        capex_dist: DistributionParam,
        revenue_dist: DistributionParam | Sequence[DistributionParam],
        opex_dist: DistributionParam | Sequence[DistributionParam],
        tax_rate: float = 0.25,
        marr_dist: DistributionParam | None = None,
        salvage_dist: DistributionParam | None = None,
        working_capital_ratio: float = 0.0,
        macrs_period: int = 5,
    ):
        self.horizon_years = horizon_years
        self.capex_dist = capex_dist
        self.revenue_dist = revenue_dist
        self.opex_dist = opex_dist
        self.tax_rate = tax_rate
        self.marr_dist = marr_dist if marr_dist is not None else ConstantParam(0.10)
        self.salvage_dist = salvage_dist if salvage_dist is not None else ConstantParam(0.0)
        self.working_capital_ratio = working_capital_ratio
        self.macrs = MACRSDepreciation(recovery_period=macrs_period)

    def run(self, n_trials: int = 10000, seed: int = 42) -> SimulationResults:
        """Execute vectorized Monte Carlo trials across all uncertain variables."""
        rng = np.random.default_rng(seed)

        # 1. Sample capex and working capital (shape: (n_trials,))
        capex = np.maximum(0.0, self.capex_dist.sample(n_trials, rng))
        working_cap = capex * self.working_capital_ratio

        # 2. Sample revenues: shape (n_trials, horizon_years)
        if isinstance(self.revenue_dist, Sequence):
            if len(self.revenue_dist) != self.horizon_years:
                raise ValueError("Length of revenue_dist list must equal horizon_years.")
            rev_cols = [np.maximum(0.0, dist.sample(n_trials, rng)) for dist in self.revenue_dist]
            revenues = np.column_stack(rev_cols)
        else:
            rev_draws = [np.maximum(0.0, self.revenue_dist.sample(n_trials, rng)) for _ in range(self.horizon_years)]
            revenues = np.column_stack(rev_draws)

        # 3. Sample OPEX: shape (n_trials, horizon_years)
        if isinstance(self.opex_dist, Sequence):
            if len(self.opex_dist) != self.horizon_years:
                raise ValueError("Length of opex_dist list must equal horizon_years.")
            opex_cols = [np.maximum(0.0, dist.sample(n_trials, rng)) for dist in self.opex_dist]
            opex = np.column_stack(opex_cols)
        else:
            opex_draws = [np.maximum(0.0, self.opex_dist.sample(n_trials, rng)) for _ in range(self.horizon_years)]
            opex = np.column_stack(opex_draws)

        # 4. Sample salvage and MARR
        salvage = np.maximum(0.0, self.salvage_dist.sample(n_trials, rng))
        marr = np.maximum(0.001, self.marr_dist.sample(n_trials, rng))

        # 5. Vectorized Depreciation Calculation
        # MACRS rates array of length horizon_years
        rates = self.macrs.rates
        n_rates = len(rates)
        padded_rates = np.zeros(self.horizon_years, dtype=np.float64)
        padded_rates[: min(n_rates, self.horizon_years)] = rates[: min(n_rates, self.horizon_years)]

        # Depreciation: outer product (n_trials, horizon_years)
        depreciation = np.outer(capex, padded_rates)

        # 6. Cash flow components
        operating_margin = revenues - opex
        ebt = operating_margin - depreciation
        taxes = ebt * self.tax_rate
        cfat_operating = operating_margin - taxes  # (n_trials, horizon_years)

        # 7. Terminal Cash Flow adjustments at t=T
        accum_dep = np.sum(depreciation, axis=1)
        terminal_bv = np.maximum(0.0, capex - accum_dep)

        # Tax on salvage
        tax_on_salvage = np.zeros(n_trials, dtype=np.float64)
        for i in range(n_trials):
            t_impact, _ = compute_salvage_tax_impact(
                salvage_realized=salvage[i],
                book_value=terminal_bv[i],
                original_cost=capex[i],
                tax_rate=self.tax_rate,
            )
            tax_on_salvage[i] = t_impact

        net_terminal_salvage = salvage - tax_on_salvage + working_cap
        cfat_operating[:, -1] += net_terminal_salvage

        # 8. Vectorized Discounting
        # Discount matrix of shape (n_trials, horizon_years): factor for year t is 1 / (1 + marr)^t
        t_years = np.arange(1, self.horizon_years + 1)
        # Power broadcast: (n_trials, 1) ** (1, horizon_years)
        discount_denom = np.power(1.0 + marr[:, np.newaxis], t_years[np.newaxis, :])
        pv_operating = np.sum(cfat_operating / discount_denom, axis=1)

        # Total NPV = - (Capex + Working Capital) + PV(operating CF)
        npv_samples = -(capex + working_cap) + pv_operating

        # 9. Statistical & Risk Metrics
        mean_npv = float(np.mean(npv_samples))
        std_npv = float(np.std(npv_samples, ddof=1))
        median_npv = float(np.median(npv_samples))
        prob_loss = float(np.mean(npv_samples < 0.0))

        # VaR and CVaR calculations
        # VaR(95%) is the loss threshold at 95% confidence = -5th percentile of NPV
        p05 = float(np.percentile(npv_samples, 5.0))
        var_95 = max(0.0, -p05)
        # CVaR: expected loss conditional on exceeding VaR
        tail_95 = npv_samples[npv_samples <= p05]
        cvar_95 = float(np.mean(-tail_95)) if len(tail_95) > 0 else var_95

        p01 = float(np.percentile(npv_samples, 1.0))
        var_99 = max(0.0, -p01)
        tail_99 = npv_samples[npv_samples <= p01]
        cvar_99 = float(np.mean(-tail_99)) if len(tail_99) > 0 else var_99

        percentiles = {
            "1%": p01,
            "5%": p05,
            "10%": float(np.percentile(npv_samples, 10.0)),
            "25%": float(np.percentile(npv_samples, 25.0)),
            "50%": median_npv,
            "75%": float(np.percentile(npv_samples, 75.0)),
            "90%": float(np.percentile(npv_samples, 90.0)),
            "95%": float(np.percentile(npv_samples, 95.0)),
            "99%": float(np.percentile(npv_samples, 99.0)),
        }

        return SimulationResults(
            n_trials=n_trials,
            npv_samples=npv_samples,
            mean_npv=mean_npv,
            std_npv=std_npv,
            median_npv=median_npv,
            prob_loss=prob_loss,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            percentiles=percentiles,
        )
