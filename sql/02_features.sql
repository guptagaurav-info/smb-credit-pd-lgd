-- Stage 2: point-in-time "network" features.
-- For a loan approved in fiscal year FY we only use outcomes that were OBSERVABLE before
-- FY began: loans approved in earlier FYs (denominator) and charge-offs whose charge-off
-- date fell in an earlier FY (numerator). This is the same discipline Ocrolus would need
-- when using network-wide performance data as a feature: no peeking at the future.

CREATE OR REPLACE MACRO fiscal_year(d) AS CASE WHEN MONTH(d) >= 10 THEN YEAR(d) + 1 ELSE YEAR(d) END;

CREATE OR REPLACE TABLE fy_grid AS SELECT range AS fy FROM range(1985, 2016);

-- Generic builder: counts by entity x FY, cumulative strictly before FY.
CREATE OR REPLACE TABLE pit_global AS
WITH a AS (SELECT approval_fy fy, COUNT(*) n FROM loans GROUP BY 1),
     c AS (SELECT fiscal_year(chgoff_date) fy, COUNT(*) co FROM loans WHERE is_default = 1 AND chgoff_date IS NOT NULL GROUP BY 1),
     g AS (SELECT g.fy, COALESCE(a.n, 0) n, COALESCE(c.co, 0) co FROM fy_grid g LEFT JOIN a USING (fy) LEFT JOIN c USING (fy))
SELECT fy,
       SUM(n)  OVER (ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior,
       SUM(co) OVER (ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS co_prior
FROM g;

CREATE OR REPLACE MACRO pit_rate(co, n, g, k) AS (COALESCE(co, 0) + k * g) / (COALESCE(n, 0) + k);

CREATE OR REPLACE TABLE pit_bank AS
WITH a AS (SELECT bank, approval_fy fy, COUNT(*) n FROM loans GROUP BY 1, 2),
     c AS (SELECT bank, fiscal_year(chgoff_date) fy, COUNT(*) co FROM loans WHERE is_default = 1 AND chgoff_date IS NOT NULL GROUP BY 1, 2),
     k AS (SELECT DISTINCT bank FROM loans),
     g AS (SELECT k.bank, f.fy, COALESCE(a.n, 0) n, COALESCE(c.co, 0) co
           FROM k CROSS JOIN fy_grid f LEFT JOIN a ON a.bank = k.bank AND a.fy = f.fy
           LEFT JOIN c ON c.bank = k.bank AND c.fy = f.fy)
SELECT bank, fy,
       SUM(n)  OVER (PARTITION BY bank ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior,
       SUM(co) OVER (PARTITION BY bank ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS co_prior
FROM g;

CREATE OR REPLACE TABLE pit_naics AS
WITH a AS (SELECT naics2, approval_fy fy, COUNT(*) n FROM loans GROUP BY 1, 2),
     c AS (SELECT naics2, fiscal_year(chgoff_date) fy, COUNT(*) co FROM loans WHERE is_default = 1 AND chgoff_date IS NOT NULL GROUP BY 1, 2),
     k AS (SELECT DISTINCT naics2 FROM loans),
     g AS (SELECT k.naics2, f.fy, COALESCE(a.n, 0) n, COALESCE(c.co, 0) co
           FROM k CROSS JOIN fy_grid f LEFT JOIN a ON a.naics2 IS NOT DISTINCT FROM k.naics2 AND a.fy = f.fy
           LEFT JOIN c ON c.naics2 IS NOT DISTINCT FROM k.naics2 AND c.fy = f.fy)
SELECT naics2, fy,
       SUM(n)  OVER (PARTITION BY naics2 ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior,
       SUM(co) OVER (PARTITION BY naics2 ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS co_prior
FROM g;

CREATE OR REPLACE TABLE pit_state AS
WITH a AS (SELECT state, approval_fy fy, COUNT(*) n FROM loans GROUP BY 1, 2),
     c AS (SELECT state, fiscal_year(chgoff_date) fy, COUNT(*) co FROM loans WHERE is_default = 1 AND chgoff_date IS NOT NULL GROUP BY 1, 2),
     k AS (SELECT DISTINCT state FROM loans),
     g AS (SELECT k.state, f.fy, COALESCE(a.n, 0) n, COALESCE(c.co, 0) co
           FROM k CROSS JOIN fy_grid f LEFT JOIN a ON a.state IS NOT DISTINCT FROM k.state AND a.fy = f.fy
           LEFT JOIN c ON c.state IS NOT DISTINCT FROM k.state AND c.fy = f.fy)
SELECT state, fy,
       SUM(n)  OVER (PARTITION BY state ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior,
       SUM(co) OVER (PARTITION BY state ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS co_prior
FROM g;

-- Model-ready table: one row per resolved loan in the modelling window.
CREATE OR REPLACE TABLE model_base AS
SELECT
    l.loan_id, l.approval_fy, l.bank, l.state, l.naics2, l.borrower_name,
    -- origination features
    l.term_months,
    LN(1 + l.approved_amt)                                     AS log_approved_amt,
    l.sba_guaranteed_amt / NULLIF(l.approved_amt, 0)           AS sba_guarantee_pct,
    LN(1 + l.n_employees)                                      AS log_employees,
    l.is_new_business, l.is_franchise, l.is_revolver, l.is_lowdoc,
    CASE WHEN l.term_months >= 240 THEN 1 ELSE 0 END           AS is_real_estate_backed,
    CASE WHEN l.bank_state <> l.state THEN 1 ELSE 0 END        AS is_out_of_state_lender,
    l.jobs_created, l.jobs_retained,
    -- 'undefined' urban/rural is a pre-2000 data-capture regime, not a borrower trait -> NULL
    CASE l.urban_rural WHEN 'rural' THEN 1 WHEN 'urban' THEN 0 END AS is_rural,
    DATEDIFF('day', l.approval_date, l.disbursement_date)      AS days_to_disburse,
    -- point-in-time network features: entity charge-off rate (shrunk toward the network rate)
    -- expressed RELATIVE to the network-wide rate, so they stay stationary as cumulative loss
    -- experience builds up over time (absolute cumulative rates drift and inflate PSI).
    pit_rate(b.co_prior, b.n_prior, g.co_prior / NULLIF(g.n_prior, 0), getvariable('k'))
        / NULLIF(g.co_prior / NULLIF(g.n_prior, 0), 0)          AS rel_bank_chgoff,
    LN(1 + COALESCE(b.n_prior, 0))                              AS log_bank_prior_loans,
    pit_rate(n.co_prior, n.n_prior, g.co_prior / NULLIF(g.n_prior, 0), getvariable('k'))
        / NULLIF(g.co_prior / NULLIF(g.n_prior, 0), 0)          AS rel_naics_chgoff,
    pit_rate(s.co_prior, s.n_prior, g.co_prior / NULLIF(g.n_prior, 0), getvariable('k'))
        / NULLIF(g.co_prior / NULLIF(g.n_prior, 0), 0)          AS rel_state_chgoff,
    -- labels and exposure
    l.is_default,
    l.disbursed_amt                                            AS ead,
    LEAST(GREATEST(COALESCE(l.chgoff_principal, 0) / NULLIF(l.disbursed_amt, 0), 0), 1) AS lgd_gross,
    LEAST(GREATEST(COALESCE(l.chgoff_principal, 0) * (1 - l.sba_guaranteed_amt / NULLIF(l.approved_amt, 0))
          / NULLIF(l.disbursed_amt, 0), 0), 1)                AS lgd_lender
FROM loans l
JOIN pit_global g ON g.fy = l.approval_fy
LEFT JOIN pit_bank  b ON b.bank = l.bank AND b.fy = l.approval_fy
LEFT JOIN pit_naics n ON n.naics2 IS NOT DISTINCT FROM l.naics2 AND n.fy = l.approval_fy
LEFT JOIN pit_state s ON s.state IS NOT DISTINCT FROM l.state AND s.fy = l.approval_fy
WHERE l.is_default IS NOT NULL
  AND l.approval_fy BETWEEN getvariable('fy_min') AND getvariable('fy_max')
  AND l.disbursed_amt > 0;
