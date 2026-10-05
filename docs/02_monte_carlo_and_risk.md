# Monte Carlo Simulation, Risk Metrics & Beta-PERT Distributions

Static engineering estimates frequently suffer from the "flaw of averages": $\mathbb{E}[f(X)] \ne f(\mathbb{E}[X])$ for non-linear valuation functions.
Vectorized Monte Carlo simulation computes empirical distributions for project performance metrics under multi-dimensional uncertainty.

---

## 1. Beta-PERT Distribution Formulation

The Beta-PERT (Program Evaluation and Review Technique) distribution models expert estimates with minimum $a$, mode $m$, and maximum $b$:

### Mean and Variance:
$$
\mu = \frac{a + \gamma m + b}{\gamma + 2}, \quad \sigma^2 = \left(\frac{b - a}{\gamma + 2}\right)^2
$$
where $\gamma = 4.0$ is the standard mode weighting parameter.

### Shape Parameters:
Given normalized variable $X' = \frac{X - a}{b - a} \sim \text{Beta}(\alpha_1, \alpha_2)$:
$$
\alpha_1 = \left(\frac{\mu - a}{b - a}\right) \left[ \frac{(b - \mu)(\mu - a)}{\sigma^2} - 1 \right]
$$
$$
\alpha_2 = \alpha_1 \left(\frac{b - \mu}{\mu - a}\right)
$$

---

## 2. Risk Metrics: Value-at-Risk (VaR) & Expected Shortfall (CVaR)

Let $L = -NPV$ denote the project financial loss variable, and $F_L(l) = P(L \le l)$ be its cumulative distribution function.

### Value-at-Risk ($\text{VaR}_\alpha$):
At confidence level $\alpha \in (0, 1)$ (typically $95\%$ or $99\%$):
$$
\text{VaR}_\alpha = \inf \{ l \in \mathbb{R} : F_L(l) \ge \alpha \} = -q_{1-\alpha}(NPV)
$$
where $q_p(NPV)$ is the $p$-th quantile of the NPV distribution.

### Conditional Value-at-Risk / Expected Shortfall ($\text{CVaR}_\alpha$):
Unlike VaR, which only identifies the boundary threshold and is not subadditive (not a coherent risk measure), CVaR measures the expected loss conditional on the loss exceeding VaR:
$$
\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha] = \frac{1}{1 - \alpha} \int_\alpha^1 \text{VaR}_u \, du
$$
In discrete Monte Carlo sample terms with sorted losses $L_{(1)} \le \dots \le L_{(N)}$:
$$
\widehat{\text{CVaR}}_\alpha = \frac{1}{\lfloor N(1-\alpha) \rfloor} \sum_{k = N - \lfloor N(1-\alpha) \rfloor + 1}^N L_{(k)}
$$

### Probability of Loss:
$$
P_{\text{loss}} = P(NPV < 0) = \frac{1}{N}\sum_{i=1}^N \mathbf{1}_{\{NPV^{(i)} < 0\}}
$$
