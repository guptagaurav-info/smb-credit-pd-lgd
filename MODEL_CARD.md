# Model Card: SMB PD (Monotone LightGBM) + EL Policy

**Intended use.** Rank-order and price US small-business term loans at origination using application and program fields. Portfolio-level expected-loss estimation. This is a demonstration model, not for actual credit decisions.

**Training data.**

- SBA 7(a)/504 loans approved FY1990–2003, n = 480K, default rate 7.7%.
- Validation: FY2004–05, n = 145K.
- Out-of-time test: FY2006–07, n = 147K.

**Target.** `MIS_Status = CHGOFF` (charged off) vs paid in full. The dataset only contains resolved loans.

**Features.** 17 origination fields:

- loan size
- guarantee %
- employees
- new/existing business
- franchise flag
- revolver / LowDoc flags
- in-/out-of-state lender
- jobs created/retained
- approval-to-disbursement time
- rural flag
- NAICS sector
- three point-in-time relative network charge-off rates (lender, industry, state)

**Excluded features.**

- `Term`, which looks like post-default leakage.
- All outcome fields: `ChgOffDate`, `ChgOffPrinGr`, `BalanceGross`.
- Borrower name and address.

**Constraints.** PD is monotone increasing in the three network charge-off rates and monotone decreasing in employee count.

**Performance (OOT).** AUC 0.664 / KS 0.24, vs 0.616 for the WoE scorecard. Calibration ECE is 0.128 with valid-period isotonic calibration and 0.077 with the stressed intercept overlay.

**Limitations.**

1. Survivorship / selection bias from resolved-only loans.
2. No cash-flow, bureau or collateral data, so discrimination is capped.
3. Program mix shifted over time (SBA Express growth, guarantee % falling from 85% to 59%). This shows up as high score PSI (1.64, test vs train), which on its own would trigger a recalibration or retrain review.
4. LGD is set at the portfolio mean; there is no predictive LGD model.

**Fairness.** No protected attributes are used. Geography enters only through the rural flag and the state-relative charge-off rate. Both should be reviewed for proxy effects before any real use.

**Monitoring plan.**

| What | Frequency | Action trigger |
|---|---|---|
| Score PSI | monthly | > 0.25 |
| Feature CSI | monthly | > 0.25 |
| AUC by vintage | as outcomes mature | > 5 pts drop vs validation |
| Calibration by decile | as outcomes mature | — |
| Reason-code distribution | monthly | — |
| Champion / challenger | at each retrain | — |
