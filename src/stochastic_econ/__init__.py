"""Stochastic Engineering Economics, Dynamic Project Valuation, and Capital Budgeting Engine."""

from stochastic_econ.cashflow import (
    CashFlowEngine,
    MACRS_RATES,
    MACRSDepreciation,
    ProjectValuation,
    StraightLineDepreciation,
    compute_salvage_tax_impact,
)
from stochastic_econ.optimization import (
    AssetReplacementOptimizer,
    CapitalAllocationResult,
    CapitalRationingOptimizer,
    ESLAnalysisResult,
    ProjectCandidate,
    ReplacementDecision,
)
from stochastic_econ.reporting import SensitivityAnalyzer, TornadoEntry
from stochastic_econ.simulation import (
    BetaPERTParam,
    ConstantParam,
    DistributionParam,
    LogNormalParam,
    MonteCarloSimulator,
    NormalParam,
    SimulationResults,
    TriangularParam,
    UniformParam,
)

__version__ = "0.1.0"

__all__ = [
    # Cash flow & depreciation
    "CashFlowEngine",
    "ProjectValuation",
    "MACRSDepreciation",
    "StraightLineDepreciation",
    "compute_salvage_tax_impact",
    "MACRS_RATES",
    # Simulation & risk
    "DistributionParam",
    "ConstantParam",
    "UniformParam",
    "NormalParam",
    "LogNormalParam",
    "TriangularParam",
    "BetaPERTParam",
    "MonteCarloSimulator",
    "SimulationResults",
    # Optimization
    "AssetReplacementOptimizer",
    "ESLAnalysisResult",
    "ReplacementDecision",
    "ProjectCandidate",
    "CapitalRationingOptimizer",
    "CapitalAllocationResult",
    # Reporting & sensitivity
    "SensitivityAnalyzer",
    "TornadoEntry",
]
