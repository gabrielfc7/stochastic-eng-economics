"""Capital rationing optimization under discrete project investment constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class ProjectCandidate:
    """Individual discrete capital investment candidate."""

    name: str
    capex: float
    npv: float
    pv_inflows: float | None = None

    def __post_init__(self):
        if self.pv_inflows is None:
            self.pv_inflows = self.capex + self.npv

    @property
    def profitability_index(self) -> float:
        return self.pv_inflows / self.capex if self.capex > 0 else 0.0


@dataclass
class CapitalAllocationResult:
    """Outcome of capital rationing optimization."""

    method: str
    selected_projects: list[str]
    total_capex_spent: float
    remaining_budget: float
    total_npv_achieved: float
    allocation_table: pd.DataFrame


class CapitalRationingOptimizer:
    """Solves capital budgeting under fixed expenditure limits via exact integer optimization and PI ranking."""

    def __init__(self, candidates: Sequence[ProjectCandidate], total_budget: float):
        if total_budget < 0:
            raise ValueError("Budget must be non-negative.")
        self.candidates = list(candidates)
        self.total_budget = float(total_budget)

    def solve_exact_knapsack(self) -> CapitalAllocationResult:
        """Find optimal 0-1 investment portfolio maximizing aggregate NPV within budget.

        Uses branch-and-bound / combinatorial state search.
        """
        n = len(self.candidates)
        best_npv = -np.inf
        best_subset = []
        best_cost = 0.0

        # Exact power-set search for moderate N (up to 24), branch-and-bound pruned
        # Sort candidates descending by efficiency to prune subtrees faster
        sorted_indices = sorted(range(n), key=lambda i: self.candidates[i].profitability_index, reverse=True)

        def search(idx: int, current_cost: float, current_npv: float, chosen: list[int]):
            nonlocal best_npv, best_subset, best_cost
            if current_cost > self.total_budget:
                return

            if current_npv > best_npv:
                best_npv = current_npv
                best_subset = list(chosen)
                best_cost = current_cost

            if idx >= n:
                return

            # Upper bound heuristic on remaining items
            rem_budget = self.total_budget - current_cost
            max_possible_npv = current_npv
            for rem_i in range(idx, n):
                c_idx = sorted_indices[rem_i]
                c = self.candidates[c_idx]
                if c.capex <= rem_budget:
                    max_possible_npv += max(0.0, c.npv)
                else:
                    if c.capex > 0:
                        max_possible_npv += max(0.0, c.npv) * (rem_budget / c.capex)
                    break

            if max_possible_npv <= best_npv:
                return

            candidate_idx = sorted_indices[idx]
            cand = self.candidates[candidate_idx]

            # Branch 1: Include candidate
            if current_cost + cand.capex <= self.total_budget:
                chosen.append(candidate_idx)
                search(idx + 1, current_cost + cand.capex, current_npv + cand.npv, chosen)
                chosen.pop()

            # Branch 2: Exclude candidate
            search(idx + 1, current_cost, current_npv, chosen)

        search(0, 0.0, 0.0, [])

        selected_names = [self.candidates[i].name for i in best_subset]
        return self._format_result("Exact 0-1 Knapsack (Optimal)", best_subset, best_cost, best_npv)

    def solve_profitability_index_heuristic(self) -> CapitalAllocationResult:
        """Greedy heuristic: rank by Profitability Index (PI) descending until budget exhausted."""
        sorted_cands = sorted(enumerate(self.candidates), key=lambda x: x[1].profitability_index, reverse=True)

        selected_indices = []
        spent = 0.0
        total_npv = 0.0

        for idx, cand in sorted_cands:
            if spent + cand.capex <= self.total_budget and cand.npv > 0:
                selected_indices.append(idx)
                spent += cand.capex
                total_npv += cand.npv

        return self._format_result("Profitability Index Heuristic (Greedy)", selected_indices, spent, total_npv)

    def _format_result(
        self,
        method: str,
        selected_indices: list[int],
        spent: float,
        total_npv: float,
    ) -> CapitalAllocationResult:
        records = []
        selected_set = set(selected_indices)

        for i, c in enumerate(self.candidates):
            records.append({
                "Project": c.name,
                "Capex": c.capex,
                "NPV": c.npv,
                "ProfitabilityIndex": c.profitability_index,
                "Selected": i in selected_set,
            })

        df = pd.DataFrame(records)
        selected_names = [self.candidates[i].name for i in selected_indices]

        return CapitalAllocationResult(
            method=method,
            selected_projects=selected_names,
            total_capex_spent=spent,
            remaining_budget=self.total_budget - spent,
            total_npv_achieved=total_npv,
            allocation_table=df,
        )
