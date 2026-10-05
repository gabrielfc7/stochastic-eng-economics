"""Multi-horizon after-tax cash flow modeling and project valuation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from stochastic_econ.cashflow.depreciation import MACRSDepreciation, compute_salvage_tax_impact


@dataclass
class ProjectValuation:
    """Consolidated project valuation financial metrics."""

    npv: float
    irr: float | None
    mirr: float | None
    profitability_index: float
    undiscounted_payback_years: float | None
    discounted_payback_years: float | None
    equivalent_annual_worth: float
    cash_flow_table: pd.DataFrame


class CashFlowEngine:
    """Vectorized multi-horizon after-tax cash flow engine with multi-tier MARR.

    Parameters
    ----------
    horizon_years : int
        Project operating lifespan in years.
    initial_capex : float
        Initial capital expenditure at time t=0.
    revenues : Sequence[float] | float
        Annual gross revenue (scalar or array of length horizon_years).
    operating_costs : Sequence[float] | float
        Annual operating expenses/OPEX (scalar or array of length horizon_years).
    tax_rate : float
        Corporate income tax rate (e.g. 0.25 for 25%).
    working_capital : float, default=0.0
        Initial working capital commitment at t=0, recovered at t=T.
    salvage_value : float, default=0.0
        Estimated pre-tax market salvage value recovered at t=T.
    depreciation_schedule : np.ndarray | Sequence[float] | MACRSDepreciation | None
        Depreciation model or annual schedule. Defaults to 5-year MACRS.
    """

    def __init__(
        self,
        horizon_years: int,
        initial_capex: float,
        revenues: Sequence[float] | float,
        operating_costs: Sequence[float] | float,
        tax_rate: float = 0.25,
        working_capital: float = 0.0,
        salvage_value: float = 0.0,
        depreciation_schedule: np.ndarray | Sequence[float] | MACRSDepreciation | None = None,
    ):
        if horizon_years <= 0:
            raise ValueError("Horizon must be at least 1 year.")
        self.horizon_years = horizon_years
        self.initial_capex = float(initial_capex)
        self.tax_rate = float(tax_rate)
        self.working_capital = float(working_capital)
        self.salvage_value = float(salvage_value)

        # Standardize revenue and cost arrays
        if np.isscalar(revenues):
            self.revenues = np.full(horizon_years, float(revenues), dtype=np.float64)
        else:
            self.revenues = np.asarray(revenues, dtype=np.float64)
            if len(self.revenues) != horizon_years:
                raise ValueError(f"Revenues length {len(self.revenues)} does not match horizon {horizon_years}.")

        if np.isscalar(operating_costs):
            self.operating_costs = np.full(horizon_years, float(operating_costs), dtype=np.float64)
        else:
            self.operating_costs = np.asarray(operating_costs, dtype=np.float64)
            if len(self.operating_costs) != horizon_years:
                raise ValueError(f"Operating costs length {len(self.operating_costs)} does not match horizon {horizon_years}.")

        # Determine depreciation schedule
        if depreciation_schedule is None:
            dep_engine = MACRSDepreciation(recovery_period=min(5, horizon_years if horizon_years in (3, 5, 7, 10) else 5))
            self.depreciation = dep_engine.get_annual_depreciation(self.initial_capex, horizon_years)
        elif isinstance(depreciation_schedule, MACRSDepreciation):
            self.depreciation = depreciation_schedule.get_annual_depreciation(self.initial_capex, horizon_years)
        else:
            self.depreciation = np.asarray(depreciation_schedule, dtype=np.float64)
            if len(self.depreciation) != horizon_years:
                raise ValueError(f"Depreciation length {len(self.depreciation)} does not match horizon {horizon_years}.")

    def generate_cash_flows(self) -> pd.DataFrame:
        """Construct full after-tax cash flow statement from t=0 to t=T."""
        years = np.arange(0, self.horizon_years + 1)
        gross_revenues = np.zeros(self.horizon_years + 1, dtype=np.float64)
        opex = np.zeros(self.horizon_years + 1, dtype=np.float64)
        depreciation = np.zeros(self.horizon_years + 1, dtype=np.float64)
        taxable_income = np.zeros(self.horizon_years + 1, dtype=np.float64)
        taxes = np.zeros(self.horizon_years + 1, dtype=np.float64)
        net_cfat = np.zeros(self.horizon_years + 1, dtype=np.float64)

        # Year 0 initial outflows
        net_cfat[0] = -(self.initial_capex + self.working_capital)

        # Operating years 1 to T
        gross_revenues[1:] = self.revenues
        opex[1:] = self.operating_costs
        depreciation[1:] = self.depreciation

        operating_margin = gross_revenues[1:] - opex[1:]
        ebt = operating_margin - depreciation[1:]
        taxable_income[1:] = ebt

        # Tax calculation
        tax_payments = ebt * self.tax_rate
        taxes[1:] = tax_payments

        # After-tax operating cash flow: (Rev - Opex) * (1 - tax) + Dep * tax
        cfat_operating = operating_margin - tax_payments
        net_cfat[1:] = cfat_operating

        # Terminal year salvage and working capital release at t=T
        accum_dep = np.sum(self.depreciation)
        terminal_bv = max(0.0, self.initial_capex - accum_dep)
        _, net_salvage = compute_salvage_tax_impact(
            salvage_realized=self.salvage_value,
            book_value=terminal_bv,
            original_cost=self.initial_capex,
            tax_rate=self.tax_rate,
        )
        net_cfat[-1] += (net_salvage + self.working_capital)

        return pd.DataFrame({
            "Year": years,
            "Revenues": gross_revenues,
            "OPEX": opex,
            "Depreciation": depreciation,
            "TaxableIncome": taxable_income,
            "Taxes": taxes,
            "NetCashFlow": net_cfat,
        })

    def evaluate(self, marr: float | Sequence[float] = 0.10) -> ProjectValuation:
        """Evaluate complete financial metrics under given MARR / discount rate.

        Parameters
        ----------
        marr : float or Sequence[float]
            Minimum Attractive Rate of Return. If a float, constant rate applied across all years.
            If a sequence of length horizon_years, represents year-specific multi-tier MARR.
        """
        df_cf = self.generate_cash_flows()
        cfs = df_cf["NetCashFlow"].to_numpy()

        # Build discount factors
        discount_factors = np.ones(self.horizon_years + 1, dtype=np.float64)
        if np.isscalar(marr):
            r = float(marr)
            discount_factors = 1.0 / np.power(1.0 + r, np.arange(self.horizon_years + 1))
            flat_marr = r
        else:
            marr_arr = np.asarray(marr, dtype=np.float64)
            if len(marr_arr) != self.horizon_years:
                raise ValueError("Multi-tier MARR sequence length must equal horizon_years.")
            cum_factors = np.cumprod(1.0 + marr_arr)
            discount_factors[1:] = 1.0 / cum_factors
            flat_marr = float(np.mean(marr_arr))

        discounted_cfs = cfs * discount_factors
        df_cf["DiscountFactor"] = discount_factors
        df_cf["DiscountedCashFlow"] = discounted_cfs
        df_cf["CumulativeCashFlow"] = np.cumsum(cfs)
        df_cf["CumulativeDiscountedCF"] = np.cumsum(discounted_cfs)

        npv = float(np.sum(discounted_cfs))

        # Internal Rate of Return (IRR)
        irr = self._compute_irr(cfs)

        # Modified Internal Rate of Return (MIRR)
        mirr = self._compute_mirr(cfs, financing_rate=flat_marr, reinvestment_rate=flat_marr)

        # Profitability Index / Benefit-Cost Ratio
        pv_future_inflows = float(np.sum(discounted_cfs[1:]))
        initial_investment = abs(float(cfs[0]))
        pi = (pv_future_inflows / initial_investment) if initial_investment > 0 else np.nan

        # Payback periods
        undisc_payback = self._compute_payback(df_cf["CumulativeCashFlow"].to_numpy())
        disc_payback = self._compute_payback(df_cf["CumulativeDiscountedCF"].to_numpy())

        # Equivalent Annual Worth (EAW / AW)
        # AW = NPV * (A/P, i, N) = NPV * [i(1+i)^N / ((1+i)^N - 1)]
        if flat_marr > 0:
            crf = (flat_marr * ((1.0 + flat_marr) ** self.horizon_years)) / (((1.0 + flat_marr) ** self.horizon_years) - 1.0)
            eaw = npv * crf
        else:
            eaw = npv / self.horizon_years

        return ProjectValuation(
            npv=npv,
            irr=irr,
            mirr=mirr,
            profitability_index=pi,
            undiscounted_payback_years=undisc_payback,
            discounted_payback_years=disc_payback,
            equivalent_annual_worth=eaw,
            cash_flow_table=df_cf,
        )

    @staticmethod
    def _compute_irr(cfs: np.ndarray) -> float | None:
        """Find exact IRR using root-finding across viable brackets."""
        # Check sign changes
        signs = np.sign(cfs[cfs != 0])
        if len(signs) < 2 or not (np.any(signs > 0) and np.any(signs < 0)):
            return None

        def npv_at_rate(r: float) -> float:
            powers = np.arange(len(cfs))
            return float(np.sum(cfs / np.power(1.0 + r, powers)))

        # Bracket search from -0.5 (-50%) to 5.0 (500%)
        test_rates = np.linspace(-0.5, 3.0, 35)
        vals = [npv_at_rate(r) for r in test_rates]
        for i in range(len(test_rates) - 1):
            if vals[i] * vals[i + 1] <= 0.0:
                try:
                    return float(brentq(npv_at_rate, test_rates[i], test_rates[i + 1], xtol=1e-7))
                except (RuntimeError, ValueError):
                    pass
        return None

    @staticmethod
    def _compute_mirr(cfs: np.ndarray, financing_rate: float, reinvestment_rate: float) -> float | None:
        """Calculate Modified Internal Rate of Return (MIRR)."""
        n = len(cfs) - 1
        negative_cfs = np.where(cfs < 0, cfs, 0.0)
        positive_cfs = np.where(cfs > 0, cfs, 0.0)

        # Present value of negative cash flows discounted at financing rate
        pv_outflows = np.sum(negative_cfs / np.power(1.0 + financing_rate, np.arange(len(cfs))))
        # Future value of positive cash flows compounded at reinvestment rate
        fv_inflows = np.sum(positive_cfs * np.power(1.0 + reinvestment_rate, n - np.arange(len(cfs))))

        if pv_outflows >= 0 or fv_inflows <= 0:
            return None

        mirr_val = np.power(-fv_inflows / pv_outflows, 1.0 / n) - 1.0
        return float(mirr_val)

    @staticmethod
    def _compute_payback(cumulative_cf: np.ndarray) -> float | None:
        """Linear interpolation of payback horizon."""
        pos_indices = np.where(cumulative_cf >= 0)[0]
        if len(pos_indices) == 0:
            return None  # Never breaks even

        first_pos = pos_indices[0]
        if first_pos == 0:
            return 0.0

        prev_val = cumulative_cf[first_pos - 1]
        curr_val = cumulative_cf[first_pos]
        # Linear interpolation between t-1 and t
        fraction = (-prev_val) / (curr_val - prev_val)
        return float((first_pos - 1) + fraction)
