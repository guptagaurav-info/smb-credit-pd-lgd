# How this project works

Three diagrams:
1. **Pipeline**: every step, its source file, and the logic inside it.
2. **Model and decision logic**: how a loan goes from features to a PD, an LGD, expected loss, a price and an approve/decline.
3. **Production design**: how the same code would run as a live service. This is a proposed design and is not implemented in this repo.

Colour key: **blue** = external data source · **lavender** = code step (file named in the box) · **yellow** = model · **red** = check or gate · **grey** = output

## 1. Pipeline (what `make all` runs)

```mermaid
flowchart TD
    SRC["SOURCE: SBA 7(a) & 504 loan file (US SBA FOIA, public domain)<br/>899,164 loans, FY1962–2014<br/>Mirror: huggingface.co/datasets/MaddRaf/SBAnational"]

    subgraph S0["Stage 0 · Ingest — src/smbcredit/ingest.py"]
      I1["Download if missing"] --> I2{"sha256 matches<br/>configs/config.yaml?"}
      I2 -- "no" --> IX["Stop: source changed"]
      I2 -- "yes" --> I3["Land raw as ALL-TEXT parquet<br/>(every cast happens later, in visible SQL)"]
    end

    subgraph S1["Stage 1 · Clean — sql/01_clean.sql (DuckDB)"]
      C1["Parse '$60,000.00' strings → numbers"]
      C2["Fix 2-digit years: any date after 2014 → minus 100 years"]
      C3["Recode flags: NewExist, RevLineCr, LowDoc, UrbanRural<br/>('undefined' urban/rural = pre-2000 capture regime → NULL)"]
      C4["Label: CHGOFF = 1, P I F = 0<br/>Outcome fields kept as labels only"]
      C5["reports/data_quality.csv<br/>(count for every rule that drops or nulls a row)"]
      C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph S2["Stage 2 · Point-in-time features — sql/02_features.sql"]
      F1["For each lender / industry / state and fiscal year FY:<br/>loans approved BEFORE FY and charge-offs dated BEFORE FY"]
      F2["Shrink toward network rate (k = 50)<br/>÷ network rate → relative, stationary features"]
      F3["Origination features: size, SBA guarantee %, employees,<br/>new business, franchise, revolver, LowDoc, out-of-state lender,<br/>jobs, days to disburse, rural, NAICS sector"]
      F4["Labels: is_default · LGD gross = charged-off ÷ disbursed<br/>· LGD lender = gross × (1 − SBA guarantee)"]
      F1 --> F2 --> F3 --> F4
    end

    subgraph S3["Stage 3 · Split by approval year (out of time)"]
      SP["Train FY1990–2003 (480K) · Validate FY2004–05 (145K)<br/>· Test FY2006–07 (147K, went into the 2008 crisis)"]
    end

    subgraph S4["Stage 4 · Models — src/smbcredit/pipeline.py"]
      L1{"Leakage check: add Term → AUC 0.964<br/>non-standard terms default 51% vs 1.8%"}
      L1 -- "Term excluded" --> M1["WoE logistic scorecard (baseline)"]
      L1 -- "Term excluded" --> M2["Monotone LightGBM (champion)<br/>early stop + isotonic calibration on validation"]
      M3["LGD: portfolio mean · segment lookup ·<br/>fractional logit · GBM (defaulted loans only)"]
    end

    subgraph S5["Stage 5 · Policy & monitoring"]
      P1["Expected loss = PD × LGD × EAD"]
      P2["Risk-based rate = cost of funds + opex<br/>+ capital charge + margin + annual EL"]
      P3["Profit curve → approval cutoff<br/>Swap-set vs scorecard at same approval rate"]
      P4["Score PSI, feature CSI, AUC by vintage,<br/>vintage curves (sql/03_vintage_curves.sql)"]
    end

    OUT["reports/metrics.json · reports/figures/*.png<br/>pricing_grid_by_pd_decile.csv · swap_set.csv<br/>sample_reason_codes.csv · MODEL_CARD.md"]

    SRC --> S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> OUT
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class SRC src
    class I2,IX,L1 gate
    class M1,M2,M3 model
    class C5,OUT out
```

## 2. Model and decision logic for one loan

```mermaid
flowchart LR
    A["Loan application<br/>(origination fields only)"] --> B["Point-in-time network features<br/>(lender, industry, state history)"]
    B --> C["Monotone LightGBM<br/>→ raw score"]
    C --> D["Isotonic calibration<br/>(fitted on FY2004–05)"]
    D --> E{"Stress view needed?"}
    E -- "yes" --> E2["Intercept shift to stressed<br/>default rate (1.5× validation)<br/>ranking unchanged"]
    E -- "no" --> F
    E2 --> F["PD"]
    C --> R["Exact TreeSHAP contributions<br/>→ top-3 reason codes"]
    G["LGD champion = portfolio mean<br/>(no model beat it by 2% on validation)"] --> H["LGD lender = LGD × (1 − SBA guarantee)"]
    F --> I["Expected loss = PD × LGD × EAD"]
    H --> I
    I --> J["Risk-based rate"]
    F --> K{"PD below<br/>profit-curve cutoff?"}
    K -- "yes" --> L["Approve at risk-based rate<br/>+ reason codes"]
    K -- "no" --> M["Decline + top-3 reasons<br/>(adverse-action style)"]
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class C,G model
    class E,K gate
    class L,M out
```

Monotone constraints: PD rises with lender, industry and state relative charge-off rates and falls with employee count. Every other feature is unconstrained.

## 3. Production design (proposed; not implemented in this repo)

```mermaid
flowchart TD
    subgraph OFF["Offline: monthly or quarterly retrain"]
      W1["Warehouse tables<br/>(Snowflake/Postgres; same SQL as sql/*.sql)"] --> W2["Point-in-time feature snapshot<br/>as of each application date"]
      W2 --> W3["Train challenger<br/>(same code as pipeline.py)"]
      W3 --> W4{"Validation gate:<br/>OOT AUC ≥ champion − tolerance,<br/>calibration ECE, PSI, fairness proxy check"}
      W4 -- "pass" --> W5["Model registry:<br/>version + config + data hash + metrics"]
      W4 -- "fail" --> W6["Keep champion; log why"]
    end

    subgraph ON["Online: per application"]
      A1["Application"] --> A2["Feature service<br/>(same definitions as offline)"]
      A2 --> A3["Scoring API<br/>(PD, LGD, EL, rate, reason codes)"]
      A3 --> A4["Decision engine<br/>(cutoffs and pricing grid from config)"]
      A4 --> A5["Decision + reasons to the lender"]
    end

    W5 -- "shadow mode first,<br/>then champion/challenger split" --> A3

    subgraph MON["Monitoring & feedback"]
      M1["Log inputs, scores, decisions"] --> M2["Daily: score PSI, feature CSI, latency"]
      M2 --> M3["As outcomes mature: AUC by vintage,<br/>calibration by band, loss vs EL"]
      M3 --> M4{"Threshold breached?<br/>(PSI > 0.25, AUC −5 pts, calibration drift)"}
      M4 -- "yes" --> M5["Recalibrate or retrain"]
      M5 --> W3
    end

    A3 --> M1
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class W4,M4 gate
    class W3 model
    class A5 out
```

## How to run

| Step | Command | Time |
|---|---|---|
| Install | `pip install -r requirements.txt` | ~1 min |
| Everything | `make all` (or `PYTHONPATH=src python -m smbcredit.pipeline`) | ~3 min |
| Tests | `make test` | seconds |
| Change splits / pricing | edit `configs/config.yaml`, rerun | — |
