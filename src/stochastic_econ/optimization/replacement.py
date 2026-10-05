"""Asset replacement optimization and Economic Service Life (ESL) analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class ESLAnalysisResult:
    """Result container for Economic Service Life evaluation."""

    optimal_life_years: int
    minimum_euac: float
    annual_cost_table: pd.DataFrame


@dataclass
class ReplacementDecision:
    """Challenger vs. Defender replacement decision verdict."""

    recommendation: str  # 'Replace immediately with Challenger' or 'Retain Defender for at least 1 more year'
    defender_marginal_cost_next_year: float
    challenger_minimum_euac: float
    explanation: str


class AssetReplacementOptimizer:
    """Determines Economic Service Life (ESL) via Equivalent Uniform Annual Cost (EUAC).

    Parameters
    ----------
    initial_cost : float
        Purchase price of new asset, or current net realizable market value of existing asset.
    salvage_schedule : Sequence[float]
        Market salvage value at the end of each year [S_1, S_2, ..., S_N].
    operating_costs : Sequence[float]
        Operating and maintenance costs during each year [OM_1, OM_2, ..., OM_N].
    interest_rate : float, default=0.10
        Discount rate / MARR.
    """

    def __init__(
        self,
        initial_cost: float,
        salvage_schedule: Sequence[float],
        operating_costs: Sequence[float],
        interest_rate: float = 0.10,
    ):
        if len(salvage_schedule) != len(operating_costs):
            raise ValueError("Salvage schedule and operating costs must have identical length.")
        self.initial_cost = float(initial_cost)
        self.salvage = np.asarray(salvage_schedule, dtype=np.float64)
        self.om = np.asarray(operating_costs, dtype=np.float64)
        self.i = float(interest_rate)
        self.n_years = len(self.salvage)

    def _crf(self, k: int) -> float:
        """Capital Recovery Factor (A/P, i, k)."""
        if self.i == 0:
            return 1.0 / k
        factor = (1.0 + self.i) ** k
        return (self.i * factor) / (factor - 1.0)

    def compute_esl(self) -> ESLAnalysisResult:
        """Compute EUAC for all candidate service lives k = 1..N and identify ESL."""
        records = []
        pw_om_running = 0.0

        for k in range(1, self.n_years + 1):
            idx = k - 1
            sk = self.salvage[idx]
            om_k = self.om[idx]

            # Present worth of OM costs up to year k: sum_{t=1}^k OM_t / (1+i)^t
            pw_om_running += om_k / ((1.0 + self.i) ** k)

            crf_k = self._crf(k)

            # Capital Recovery: CR(k) = (P - S_k)(A/P, i, k) + i * S_k
            cr = (self.initial_cost - sk) * crf_k + (self.i * sk)

            # Equivalent Uniform Annual Operating Cost: AOC(k) = PW_OM * (A/P, i, k)
            aoc = pw_om_running * crf_k

            euac = cr + aoc

            records.append({
                "Year": k,
                "SalvageValue": sk,
                "OMCost": om_k,
                "CapitalRecovery": cr,
                "AnnualOperatingCost": aoc,
                "EUAC": euac,
            })

        df = pd.DataFrame(records)
        optimal_idx = int(df["EUAC"].idxmin())
        optimal_life = int(df.loc[optimal_idx, "Year"])
        min_euac = float(df.loc[optimal_idx, "EUAC"])

        return ESLAnalysisResult(
            optimal_life_years=optimal_life,
            minimum_euac=min_euac,
            annual_cost_table=df,
        )

    @staticmethod
    def compare_challenger_defender(
        defender_current_market_value: float,
        defender_salvage_next_year: float,
        defender_om_next_year: float,
        challenger_min_euac: float,
        interest_rate: float,
    ) -> ReplacementDecision:
        """Evaluate marginal cost of keeping Defender for 1 more year vs Challenger EUAC.

        Marginal Cost of Defender for Year 1:
        MC_1 = P_0 - S_1 + i * P_0 + OM_1 = (P_0 - S_1) + i * P_0 + OM_1
        """
        loss_in_market_value = defender_current_market_value - defender_salvage_next_year
        capital_cost = interest_rate * defender_current_market_value
        marginal_cost = loss_in_market_value + capital_cost + defender_om_next_year

        if marginal_cost > challenger_min_euac:
            decision = "Replace immediately with Challenger"
            explanation = (
                f"Defender marginal cost for next year (${marginal_cost:,.2f}) exceeds "
                f"Challenger minimum EUAC (${challenger_min_euac:,.2f}). Immediate replacement is optimal."
            )
        else:
            decision = "Retain Defender for at least 1 more year"
            explanation = (
                f"Defender marginal cost for next year (${marginal_cost:,.2f}) is lower than or equal to "
                f"Challenger minimum EUAC (${challenger_min_euac:,.2f}). Retain Defender."
            )

        return ReplacementDecision(
            recommendation=decision,
            defender_marginal_cost_next_year=marginal_cost,
            challenger_minimum_euac=challenger_min_euac,
            explanation=explanation,
        )
