# Multi-Horizon After-Tax Cash Flow & Depreciation Mechanics

Engineering economic valuation models cash flows after taxes ($CFAT$), accounting for capital outlays, operational performance, working capital dynamics, and depreciation tax shields.

---

## 1. After-Tax Cash Flow ($CFAT$) Formulation

For project year $t \in \{1, \dots, T\}$:
$$
CFAT_t = (R_t - C_t)(1 - \tau) + \tau \cdot D_t
$$
where:
- $R_t$: Gross operating revenues in year $t$.
- $C_t$: Operating expenses (OPEX) in year $t$.
- $\tau$: Marginal corporate income tax rate.
- $D_t$: Allowable tax depreciation deduction in year $t$.
- $\tau \cdot D_t$: **Depreciation tax shield** (non-cash deduction shielding income from taxation).

At project inception ($t = 0$):
$$
CFAT_0 = -(I_0 + \Delta NWC)
$$
where $I_0$ is total initial capital outlay (Capex) and $\Delta NWC$ is net working capital commitment.

At terminal disposition ($t = T$):
$$
CFAT_T = (R_T - C_T)(1 - \tau) + \tau \cdot D_T + \Delta NWC + S_T^{\text{after-tax}}
$$
where working capital is fully recaptured, and $S_T^{\text{after-tax}}$ is the net after-tax salvage recovery.

---

## 2. IRS MACRS Half-Year Convention

Under the Modified Accelerated Cost Recovery System (MACRS, 26 U.S.C. § 168), tangible property is depreciated across standard recovery periods ($N \in \{3, 5, 7, 10, 15, 20\}$) using the half-year convention:

| Year $t$ | 3-Year | 5-Year | 7-Year | 10-Year |
|---|---|---|---|---|
| 1 | 33.33% | 20.00% | 14.29% | 10.00% |
| 2 | 44.45% | 32.00% | 24.49% | 18.00% |
| 3 | 14.81% | 19.20% | 17.49% | 14.40% |
| 4 | 7.41% | 11.52% | 12.49% | 11.52% |
| 5 | — | 11.52% | 8.93% | 9.22% |
| 6 | — | 5.76% | 8.92% | 7.37% |
| 7 | — | — | 8.93% | 6.55% |
| 8 | — | — | 4.46% | 6.55% |
| 9 | — | — | — | 6.56% |
| 10 | — | — | — | 6.55% |
| 11 | — | — | — | 3.28% |

Because of the half-year convention, an $N$-year property spans $N+1$ tax recovery years.
The unadjusted basis is not reduced by expected salvage value ($S$).

---

## 3. Salvage Recapture and Tax Impact

When an asset is liquidated at time $T$ for salvage price $S_T$, the tax liability depends on the tax book value $BV_T = I_0 - \sum_{t=1}^T D_t$:

1. **Depreciation Recapture ($BV_T < S_T \le I_0$):**
   $$\text{Tax Liability} = (S_T - BV_T) \cdot \tau$$
2. **Capital Gain ($S_T > I_0$):**
   $$\text{Tax Liability} = (I_0 - BV_T) \cdot \tau + (S_T - I_0) \cdot \tau_{\text{cg}}$$
3. **Loss on Disposal ($S_T < BV_T$):**
   $$\text{Tax Benefit} = -(BV_T - S_T) \cdot \tau$$

Net after-tax cash inflow:
$$
S_T^{\text{after-tax}} = S_T - \text{Tax Liability}
$$

---

## 4. Multi-Tier MARR Discounting

When capital cost varies across horizons due to debt amortization, macroeconomic liquidity, or risk profiles, a term structure of MARR $\{r_1, r_2, \dots, r_T\}$ is applied:

$$
NPV = CFAT_0 + \sum_{t=1}^T \frac{CFAT_t}{\prod_{s=1}^t (1 + r_s)}
$$
This generalizes the static discount formula $NPV = \sum_{t=0}^T CFAT_t (1 + r)^{-t}$.
