"""Depreciation models including IRS MACRS, Straight-Line, and Declining Balance schedules."""

from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd


# Official IRS MACRS Half-Year Convention percentage tables (Pub 946)
MACRS_RATES: dict[int, list[float]] = {
    3: [0.3333, 0.4445, 0.1481, 0.0741],
    5: [0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576],
    7: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446],
    10: [0.1000, 0.1800, 0.1440, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655, 0.0328],
    15: [
        0.0500, 0.0950, 0.0855, 0.0770, 0.0693, 0.0623, 0.0590, 0.0590,
        0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0295,
    ],
    20: [
        0.03750, 0.07219, 0.06677, 0.06177, 0.05713, 0.05285, 0.04888, 0.04522,
        0.04462, 0.04461, 0.04462, 0.04461, 0.04462, 0.04461, 0.04462, 0.04461,
        0.04462, 0.04461, 0.04462, 0.04461, 0.02231,
    ],
}


class MACRSDepreciation:
    """IRS MACRS (Modified Accelerated Cost Recovery System) with Half-Year Convention.

    Parameters
    ----------
    recovery_period : int
        MACRS recovery period in years (3, 5, 7, 10, 15, or 20).
    """

    def __init__(self, recovery_period: int = 5):
        if recovery_period not in MACRS_RATES:
            valid_periods = list(MACRS_RATES.keys())
            raise ValueError(f"Invalid recovery period {recovery_period}. Must be one of {valid_periods}.")
        self.recovery_period = recovery_period
        self.rates = np.array(MACRS_RATES[recovery_period], dtype=np.float64)

    def schedule(self, cost_basis: float) -> pd.DataFrame:
        """Generate complete annual depreciation schedule and book values.

        Parameters
        ----------
        cost_basis : float
            Total initial unadjusted basis of the capital asset.

        Returns
        -------
        pd.DataFrame
            Table with columns ['Year', 'Rate', 'Depreciation', 'BookValue'].
        """
        if cost_basis < 0:
            raise ValueError("Cost basis must be non-negative.")

        depreciation = cost_basis * self.rates
        book_value = cost_basis - np.cumsum(depreciation)
        # Numerical hygiene: ensure final BV doesn't drift negative due to rounding
        book_value = np.maximum(0.0, book_value)

        years = np.arange(1, len(self.rates) + 1)
        return pd.DataFrame({
            "Year": years,
            "Rate": self.rates,
            "Depreciation": depreciation,
            "BookValue": book_value,
        })

    def get_annual_depreciation(self, cost_basis: float, horizon_years: int) -> np.ndarray:
        """Return depreciation array truncated or zero-padded to match project horizon."""
        full_dep = cost_basis * self.rates
        n_rates = len(full_dep)
        if horizon_years <= n_rates:
            return full_dep[:horizon_years]
        else:
            padding = np.zeros(horizon_years - n_rates, dtype=np.float64)
            return np.concatenate([full_dep, padding])


class StraightLineDepreciation:
    """Straight-line depreciation with uniform annual deduction."""

    def __init__(self, life_years: int):
        if life_years <= 0:
            raise ValueError("Life years must be strictly positive.")
        self.life_years = life_years

    def schedule(self, cost_basis: float, salvage_value: float = 0.0) -> pd.DataFrame:
        annual_dep = (cost_basis - salvage_value) / self.life_years
        dep_series = np.full(self.life_years, annual_dep, dtype=np.float64)
        bv_series = cost_basis - np.cumsum(dep_series)

        return pd.DataFrame({
            "Year": np.arange(1, self.life_years + 1),
            "Depreciation": dep_series,
            "BookValue": np.maximum(salvage_value, bv_series),
        })


def compute_salvage_tax_impact(
    salvage_realized: float,
    book_value: float,
    original_cost: float,
    tax_rate: float,
    capital_gains_rate: float | None = None,
) -> tuple[float, float]:
    """Compute depreciation recapture and net after-tax salvage cash flow.

    Parameters
    ----------
    salvage_realized : float
        Actual selling/salvage price obtained at retirement.
    book_value : float
        Current tax book value at the time of disposal.
    original_cost : float
        Original unadjusted purchase cost basis.
    tax_rate : float
        Ordinary corporate income tax rate.
    capital_gains_rate : float, optional
        Capital gains tax rate (defaults to ordinary tax rate if None).

    Returns
    -------
    tuple[float, float]
        (tax_liability, after_tax_salvage_cash_flow)
    """
    if capital_gains_rate is None:
        capital_gains_rate = tax_rate

    if salvage_realized > book_value:
        recapture_gain = min(salvage_realized - book_value, original_cost - book_value)
        capital_gain = max(0.0, salvage_realized - original_cost)
        total_tax = recapture_gain * tax_rate + capital_gain * capital_gains_rate
    elif salvage_realized < book_value:
        # Loss on disposal provides a tax deduction
        loss = book_value - salvage_realized
        total_tax = -loss * tax_rate
    else:
        total_tax = 0.0

    net_salvage_cf = salvage_realized - total_tax
    return total_tax, net_salvage_cf
