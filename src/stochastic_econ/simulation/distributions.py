"""Stochastic parameter distributions and vector samplers for engineering simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from scipy import stats


class DistributionParam(ABC):
    """Abstract base class for stochastic simulation parameters."""

    @abstractmethod
    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw vector of random samples."""
        pass


class ConstantParam(DistributionParam):
    """Deterministic constant parameter."""

    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(size, self.value, dtype=np.float64)


class UniformParam(DistributionParam):
    """Uniform continuous distribution between low and high."""

    def __init__(self, low: float, high: float):
        if low > high:
            raise ValueError(f"low ({low}) cannot exceed high ({high}).")
        self.low = float(low)
        self.high = float(high)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=size)


class NormalParam(DistributionParam):
    """Gaussian distribution with optional clipping bounds."""

    def __init__(
        self,
        mean: float,
        std: float,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ):
        if std < 0:
            raise ValueError("Standard deviation must be non-negative.")
        self.mean = float(mean)
        self.std = float(std)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        draws = rng.normal(self.mean, self.std, size=size)
        if self.clip_min is not None or self.clip_max is not None:
            draws = np.clip(
                draws,
                a_min=self.clip_min if self.clip_min is not None else -np.inf,
                a_max=self.clip_max if self.clip_max is not None else np.inf,
            )
        return draws


class LogNormalParam(DistributionParam):
    """Log-Normal distribution parameterized by underlying normal mean and standard deviation."""

    def __init__(self, mean_underlying: float, std_underlying: float):
        if std_underlying < 0:
            raise ValueError("Standard deviation must be non-negative.")
        self.mean_underlying = float(mean_underlying)
        self.std_underlying = float(std_underlying)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mean_underlying, self.std_underlying, size=size)


class TriangularParam(DistributionParam):
    """Triangular distribution with minimum, mode, and maximum."""

    def __init__(self, low: float, mode: float, high: float):
        if not (low <= mode <= high):
            raise ValueError(f"Must satisfy low ({low}) <= mode ({mode}) <= high ({high}).")
        self.low = float(low)
        self.mode = float(mode)
        self.high = float(high)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, size=size)


class BetaPERTParam(DistributionParam):
    """Beta-PERT distribution commonly used in project management and risk engineering.

    Parameters
    ----------
    low : float
        Pessimistic / minimum bound ($a$).
    mode : float
        Most likely / modal estimate ($m$).
    high : float
        Optimistic / maximum bound ($b$).
    gamma : float, default=4.0
        Shape confidence weighting (standard PERT uses 4.0).
    """

    def __init__(self, low: float, mode: float, high: float, gamma: float = 4.0):
        if not (low <= mode <= high):
            raise ValueError(f"Must satisfy low ({low}) <= mode ({mode}) <= high ({high}).")
        if low == high:
            raise ValueError("Low and high cannot be equal for Beta-PERT.")
        self.low = float(low)
        self.mode = float(mode)
        self.high = float(high)
        self.gamma = float(gamma)

        # Standard PERT moments
        mu = (self.low + self.gamma * self.mode + self.high) / (self.gamma + 2.0)
        # Prevent division by zero if mode == mean
        if abs(mu - self.mode) < 1e-12:
            self.alpha1 = 1.0 + self.gamma / 2.0
            self.alpha2 = 1.0 + self.gamma / 2.0
        else:
            self.alpha1 = ((mu - self.low) / (self.high - self.low)) * (
                ((self.high - mu) * (mu - self.low) / (((self.high - self.low) / (self.gamma + 2.0)) ** 2)) - 1.0
            )
            self.alpha2 = self.alpha1 * (self.high - mu) / (mu - self.low)

        # Numerical stabilization
        self.alpha1 = max(1e-4, self.alpha1)
        self.alpha2 = max(1e-4, self.alpha2)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        beta_draws = stats.beta.rvs(self.alpha1, self.alpha2, size=size, random_state=rng)
        return self.low + beta_draws * (self.high - self.low)
