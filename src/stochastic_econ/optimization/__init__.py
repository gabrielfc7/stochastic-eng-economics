"""Optimization module for asset replacement and capital budgeting."""

from stochastic_econ.optimization.capital_rationing import (
    CapitalAllocationResult,
    CapitalRationingOptimizer,
    ProjectCandidate,
)
from stochastic_econ.optimization.replacement import (
    AssetReplacementOptimizer,
    ESLAnalysisResult,
    ReplacementDecision,
)

__all__ = [
    "AssetReplacementOptimizer",
    "ESLAnalysisResult",
    "ReplacementDecision",
    "ProjectCandidate",
    "CapitalRationingOptimizer",
    "CapitalAllocationResult",
]
