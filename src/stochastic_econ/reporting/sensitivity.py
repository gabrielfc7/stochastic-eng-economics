"""Deterministic sensitivity analysis, tornado rankings, and breakeven evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from stochastic_econ.cashflow.cashflow_model import CashFlowEngine


@dataclass
class TornadoEntry:
    """Tornado analysis swing for a single economic variable."""

    parameter: str
    baseline_value: float
    low_value: float
    high_value: float
    npv_low: float
    npv_high: float
    npv_swing: float


class SensitivityAnalyzer:
    """Performs One-at-a-Time (OAT) sensitivity swings and breakeven point solving."""

    def __init__(
        self,
        horizon_years: int,
        base_capex: float,
        base_revenue: float,
        base_opex: float,
        base_marr: float = 0.10,
        base_tax_rate: float = 0.25,
        base_salvage: float = 0.0,
        macrs_period: int = 5,
    ):
        self.horizon_years = horizon_years
        self.base_capex = base_capex
        self.base_revenue = base_revenue
        self.base_opex = base_opex
        self.base_marr = base_marr
        self.base_tax_rate = base_tax_rate
        self.base_salvage = base_salvage
        self.macrs_period = macrs_period

        # Evaluate baseline NPV
        engine = CashFlowEngine(
            horizon_years=self.horizon_years,
            initial_capex=self.base_capex,
            revenues=self.base_revenue,
            operating_costs=self.base_opex,
            tax_rate=self.base_tax_rate,
            salvage_value=self.base_salvage,
        )
        self.baseline_npv = engine.evaluate(marr=self.base_marr).npv

    def compute_tornado(self, perturbation_pct: float = 0.20) -> pd.DataFrame:
        """Compute NPV swings across each parameter perturbed by +/- perturbation_pct."""
        params = {
            "Annual Revenue": (self.base_revenue, "revenue"),
            "Capital Expenditure (Capex)": (self.base_capex, "capex"),
            "Annual OPEX": (self.base_opex, "opex"),
            "Discount Rate (MARR)": (self.base_marr, "marr"),
            "Corporate Tax Rate": (self.base_tax_rate, "tax"),
        }

        records = []

        for name, (base_val, ptype) in params.items():
            val_low = base_val * (1.0 - perturbation_pct)
            val_high = base_val * (1.0 + perturbation_pct)

            npv_low = self._evaluate_custom(ptype, val_low)
            npv_high = self._evaluate_custom(ptype, val_high)
            swing = abs(npv_high - npv_low)

            records.append({
                "Parameter": name,
                "BaseValue": base_val,
                "LowValue": val_low,
                "HighValue": val_high,
                "NPV_at_Low": npv_low,
                "NPV_at_High": npv_high,
                "SwingRange": swing,
            })

        df = pd.DataFrame(records)
        return df.sort_values(by="SwingRange", ascending=False).reset_index(drop=True)

    def compute_breakeven_revenue(self) -> float:
        """Find annual revenue level yielding exactly zero NPV."""
        def obj(rev: float) -> float:
            eng = CashFlowEngine(
                horizon_years=self.horizon_years,
                initial_capex=self.base_capex,
                revenues=rev,
                operating_costs=self.base_opex,
                tax_rate=self.base_tax_rate,
                salvage_value=self.base_salvage,
            )
            return eng.evaluate(marr=self.base_marr).npv

        return float(brentq(obj, 0.0, self.base_revenue * 10.0))

    def compute_breakeven_capex(self) -> float:
        """Find maximum initial capex yielding non-negative NPV."""
        def obj(cap: float) -> float:
            eng = CashFlowEngine(
                horizon_years=self.horizon_years,
                initial_capex=cap,
                revenues=self.base_revenue,
                operating_costs=self.base_opex,
                tax_rate=self.base_tax_rate,
                salvage_value=self.base_salvage,
            )
            return eng.evaluate(marr=self.base_marr).npv

        return float(brentq(obj, 0.0, self.base_capex * 10.0))

    def _evaluate_custom(self, param_type: str, test_value: float) -> float:
        capex = test_value if param_type == "capex" else self.base_capex
        rev = test_value if param_type == "revenue" else self.base_revenue
        opex = test_value if param_type == "opex" else self.base_opex
        tax = test_value if param_type == "tax" else self.base_tax_rate
        marr = test_value if param_type == "marr" else self.base_marr

        eng = CashFlowEngine(
            horizon_years=self.horizon_years,
            initial_capex=capex,
            revenues=rev,
            operating_costs=opex,
            tax_rate=tax,
            salvage_value=self.base_salvage,
        )
        return eng.evaluate(marr=marr).npv
