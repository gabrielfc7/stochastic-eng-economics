# Asset Replacement Optimization & Capital Rationing

Industrial systems require dynamic asset renewal models (Economic Service Life) and optimal capital budgeting under capital constraints.

---

## 1. Economic Service Life (ESL) Optimization

The Economic Service Life is the ownership period $k^*$ that minimizes the **Equivalent Uniform Annual Cost (EUAC)**.

### EUAC Decomposition for Service Life $k$:
$$
\text{EUAC}(k) = \text{CR}(k) + \text{AOC}(k)
$$

1. **Capital Recovery Cost $\text{CR}(k)$:**
   $$\text{CR}(k) = (P - S_k)(A/P, i, k) + i \cdot S_k$$
   where $P$ is initial asset acquisition cost (or defender market value), $S_k$ is salvage value at year $k$, and $(A/P, i, k) = \frac{i(1+i)^k}{(1+i)^k - 1}$ is the capital recovery factor.

2. **Annual Operating Cost Equivalent $\text{AOC}(k)$:**
   $$\text{AOC}(k) = \left[ \sum_{t=1}^k \frac{C_t}{(1 + i)^t} \right] (A/P, i, k)$$
   where $C_t$ represents annual operating and maintenance expenses in year $t$.

### Optimization Objective:
$$
k^* = \arg\min_{k \in \{1, \dots, N\}} \text{EUAC}(k)
$$

---

## 2. Challenger vs. Defender Marginal Decision Rule

When deciding whether to replace an existing operating asset ("Defender") with a modern alternative ("Challenger"):

Let $MC_{\text{def}}(1)$ be the marginal cost of keeping the Defender for exactly one additional year:
$$
MC_{\text{def}}(1) = (P_0 - S_1) + i \cdot P_0 + C_1
$$
where:
- $(P_0 - S_1)$: Loss in net market resale value over the year.
- $i \cdot P_0$: Foregone capital opportunity cost on current market value.
- $C_1$: Operating and maintenance cost for the year.

### Decision Verdict:
- If $MC_{\text{def}}(1) > \min \text{EUAC}_{\text{chal}}$: **Replace immediately with Challenger**.
- If $MC_{\text{def}}(1) \le \min \text{EUAC}_{\text{chal}}$: **Retain Defender for at least one more year**.

---

## 3. Capital Rationing via 0-1 Knapsack Optimization

When a firm has a fixed capital expenditure limit $B$ and $M$ independent positive-NPV projects:

### Mathematical Program:
$$
\max_{\mathbf{x}} \sum_{j=1}^M x_j \cdot NPV_j
$$
subject to:
$$
\sum_{j=1}^M x_j \cdot \text{Capex}_j \le B, \quad x_j \in \{0, 1\}
$$

### Optimality Gap of Profitability Index (PI) Heuristic:
Ranking projects purely by $PI_j = \frac{PV_j}{\text{Capex}_j}$ is optimal only when projects are infinitely divisible.
With discrete, lumpy capital outlays, the greedy PI heuristic leaves unutilized budget slacks that often underperform the exact 0-1 integer solution.
