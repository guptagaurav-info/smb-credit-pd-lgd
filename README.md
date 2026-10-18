# SMB Credit: PD, LGD and Expected-Loss Pricing on 772K Real SBA Loans

An end-to-end small-business credit model built the way a lender's risk team would build it. It runs as one pipeline:

- raw government data → warehouse SQL → point-in-time network features
- PD models (scorecard vs monotone GBM) → LGD → expected loss and risk-based pricing → approval policy
- monitoring and reason codes

```bash
pip install -r requirements.txt
make all          # download (~180 MB, sha256-verified) -> transform -> train -> report  (~3 min on a laptop)
make test         # unit tests on the SQL transforms, point-in-time logic, metrics and policy
```

## Why this dataset

It is the **SBA 7(a) / 504 national loan file**: 899K US small-business loans approved from 1962 to 2014, released under FOIA. It has **real outcomes**:

- paid-in-full vs charged-off
- charged-off principal, which gives both PD *and* LGD labels
- lender, industry (NAICS), geography, guarantee and program fields

Source: U.S. Small Business Administration. Mirror used: [`MaddRaf/SBAnational`](https://huggingface.co/datasets/MaddRaf/SBAnational). Reference: Li, Mickel & Taylor (2018), *Journal of Statistics Education* 26(1).

## Pipeline

| Stage | Where | What it does |
|---|---|---|
| Ingest | `src/smbcredit/ingest.py` | Download, verify the sha256, and land raw data **all-VARCHAR** as parquet, so every cast happens in auditable SQL |
| Clean | `sql/01_clean.sql` | Typed loan table: currency strings, **two-digit-year century fix**, coded fields, and a data-quality report (`reports/data_quality.csv`) |
| Network features | `sql/02_features.sql` | **Point-in-time** lender / industry / state charge-off rates. Shrunk toward the network rate and expressed *relative* to it. They use only outcomes observable before the loan's fiscal year (unit-tested) |
| PD | `src/smbcredit/pd_model.py` | (1) WoE logistic scorecard. (2) LightGBM with **economically justified monotone constraints**, early-stopped and isotonic-calibrated on validation vintages |
| LGD | `src/smbcredit/lgd_model.py` | Severity on defaulted loans: portfolio mean vs segment lookup vs fractional logit vs GBM. Champion picked on validation, with a **materiality rule** |
| Policy | `src/smbcredit/policy.py` | EL = PD × LGD × EAD, risk-based rate grid, profit-curve cutoff, **swap-set** analysis |
| Monitoring | `src/smbcredit/metrics.py` | AUC by vintage, score PSI, feature CSI, calibration (ECE) |
| Portfolio SQL | `sql/03_vintage_curves.sql` | Cumulative charge-off vintage curves |

**Validation design: out-of-time by approval vintage.** Train on FY1990–2003, validate on FY2004–05, and test on **FY2006–07**, the cohort that went straight into the financial crisis. That makes the test a ranking test under stress, not a random split.

## Results (out-of-time test, FY2006–07, n = 147,405, default rate 38.8%)

| Model | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| WoE logistic scorecard | 0.616 | 0.23 | 0.19 | 0.278 |
| **Monotone LightGBM (champion)** | **0.664** | **0.33** | **0.24** | **0.237** |

On the same test set, the GBM:

- beats the scorecard in **every vintage**, FY1990–2007 (`reports/auc_by_vintage.csv`)
- adds about **+$21M (+3.5%)** of illustrative lifetime profit at the optimal cutoff (`reports/figures/profit_curve.png`)
- at a matched 42.5% approval rate, **swaps in** 14,687 loans with a 31.4% bad rate for 14,687 loans with a **45.0%** bad rate (`reports/swap_set.csv`)

### Three findings I'd raise with a credit committee

**1. The most predictive field was leakage, and it would have produced a 0.96 AUC.**

- Adding `Term` gives a test AUC of **0.964**.
- On inspection, loans with *non-standard* recorded terms (42, 46 months …) default at **51%**, while standard terms (60/84/120/240) default at **1.8%**.
- That pattern is consistent with the term being rewritten after default or modification.
- `Term` is therefore excluded and kept only as a diagnostic (`reports/figures/leakage_term.png`).
- The honest number is 0.664.

**2. Calibration breaks through the cycle, so the ranking and the level are handled separately.**

- Isotonic calibration on FY2004–05 under-predicts the crisis cohort: mean PD 26% vs 39% observed.
- The rank order holds up, so the level is adjusted with an **intercept shift to a stressed central tendency** (1.5× the validation default rate).
- That shift is set by a scenario, not taken from test outcomes. It cuts ECE from 0.128 to 0.077 without touching the ranking.
- This is the same separation as point-in-time PD vs through-the-cycle PD in a regulatory setting.

**3. LGD is not predictable from origination fields, so the model doesn't ship one.**

- On defaulted loans, no model beats the **portfolio-mean LGD** on validation by the 2% materiality bar.
- Out of time, every model's rank correlation turns negative: the crisis changed severity drivers.
- The pipeline therefore uses portfolio-mean gross LGD × (1 − SBA guarantee) for lender loss.
- Real LGD lift needs collateral, lien position and cash-flow data at default.

### Other design notes

- **Network features are point-in-time and relative.** Absolute cumulative charge-off rates drift upward over time and inflate PSI. Expressing each entity's rate relative to the network rate makes them more stationary. The CSI table still flags drift in `rel_naics_chgoff`, as it should: NAICS coding started around 1997.
- **Data-capture regimes masquerade as signal.** For example, `UrbanRural = 0` ("undefined") covers 99% of pre-2000 loans and 0% after, so it is a vintage marker, not a borrower trait. It is recoded to NULL.
- **Selection bias.** The FOIA file holds only *resolved* loans, so recent vintages over-represent defaults. The OOT test is therefore read mainly as a ranking test; levels are set via the stress overlay.
- **Reason codes.** Exact TreeSHAP contributions from LightGBM give each applicant's top-3 adverse-action-style reasons, value-aware (e.g. "Out-of-state lender", "Low SBA guarantee share"). See `reports/sample_reason_codes.csv`.

## Outputs

- `reports/metrics.json`: every number above
- `reports/figures/`:
  - `roc_test.png`
  - `calibration_test.png`
  - `auc_by_vintage.png`
  - `shap_importance.png`
  - `profit_curve.png`
  - `lgd_distribution.png`
  - `vintage_curves.png`
  - `leakage_term.png`
- `reports/pricing_grid_by_pd_decile.csv`: PD, LGD, EL % and indicated rate by risk decile
- `MODEL_CARD.md`: intended use, limitations, monitoring plan

## How this maps to cash-flow underwriting

This repo uses **origination-only** fields, which is why the AUC is about 0.66. That is the gap cash-flow data closes. The production version of this pipeline would replace or augment the feature layer with **bank-transaction features**:

- true revenue
- NSF and negative-balance days
- MCA / debt-service burden
- revenue volatility

It would also add **network-level signals**, such as point-in-time lender and industry performance and inquiry or stacking velocity. The validation, calibration, policy and monitoring layers stay the same. See the companion repo, **cashflow-underwriting-fraud**.

---

*Built as a portfolio project. Pricing parameters are illustrative, not any lender's actual economics.*
