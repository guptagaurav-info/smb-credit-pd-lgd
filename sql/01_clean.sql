-- Stage 1: type the raw FOIA extract into a clean loan-level table.
-- Raw is all-VARCHAR on purpose: we own every cast and can audit failures.
CREATE OR REPLACE MACRO money(x) AS TRY_CAST(REPLACE(REPLACE(TRIM(x), '$', ''), ',', '') AS DOUBLE);
-- Two-digit years: strptime maps 00-68 -> 20xx. Any date after the 2014 data cut is a 19xx date.
CREATE OR REPLACE MACRO sba_date(x) AS
  CASE WHEN YEAR(TRY_STRPTIME(x, '%d-%b-%y')) > 2014
       THEN CAST(TRY_STRPTIME(x, '%d-%b-%y') - INTERVAL 100 YEAR AS DATE)
       ELSE CAST(TRY_STRPTIME(x, '%d-%b-%y') AS DATE) END;

CREATE OR REPLACE TABLE loans AS
SELECT
    LoanNr_ChkDgt                                             AS loan_id,
    UPPER(TRIM(Name))                                         AS borrower_name,
    UPPER(TRIM(City))                                         AS city,
    State                                                     AS state,
    UPPER(TRIM(Bank))                                         AS bank,
    BankState                                                 AS bank_state,
    NULLIF(LEFT(NAICS, 2), '0')                               AS naics2,
    sba_date(ApprovalDate)                                    AS approval_date,
    TRY_CAST(REGEXP_REPLACE(ApprovalFY, '[^0-9]', '', 'g') AS INT) AS approval_fy,
    TRY_CAST(Term AS INT)                                     AS term_months,
    TRY_CAST(NoEmp AS INT)                                    AS n_employees,
    CASE NewExist WHEN '2' THEN 1 WHEN '1' THEN 0 END         AS is_new_business,
    TRY_CAST(CreateJob AS INT)                                AS jobs_created,
    TRY_CAST(RetainedJob AS INT)                              AS jobs_retained,
    CASE WHEN FranchiseCode IN ('0', '1') THEN 0 ELSE 1 END   AS is_franchise,
    CASE UrbanRural WHEN '1' THEN 'urban' WHEN '2' THEN 'rural' ELSE 'undefined' END AS urban_rural,
    CASE RevLineCr WHEN 'Y' THEN 1 WHEN 'N' THEN 0 END        AS is_revolver,   -- other codes -> NULL
    CASE LowDoc    WHEN 'Y' THEN 1 WHEN 'N' THEN 0 END        AS is_lowdoc,
    sba_date(DisbursementDate)                                AS disbursement_date,
    money(DisbursementGross)                                  AS disbursed_amt,
    money(GrAppv)                                             AS approved_amt,
    money(SBA_Appv)                                           AS sba_guaranteed_amt,
    -- outcome fields (labels only; never features)
    MIS_Status                                                AS status,
    CASE MIS_Status WHEN 'CHGOFF' THEN 1 WHEN 'P I F' THEN 0 END AS is_default,
    sba_date(ChgOffDate)                                      AS chgoff_date,
    money(ChgOffPrinGr)                                       AS chgoff_principal
FROM read_parquet(getvariable('raw_parquet'));

-- Data-quality log: every rule that drops or nulls a row is counted.
CREATE OR REPLACE TABLE dq_report AS
SELECT 'rows_total' AS check_name, COUNT(*) AS n FROM loans UNION ALL
SELECT 'status_missing', COUNT(*) FROM loans WHERE is_default IS NULL UNION ALL
SELECT 'approval_date_unparsed', COUNT(*) FROM loans WHERE approval_date IS NULL UNION ALL
SELECT 'naics_missing', COUNT(*) FROM loans WHERE naics2 IS NULL UNION ALL
SELECT 'revolver_code_invalid', COUNT(*) FROM loans WHERE is_revolver IS NULL UNION ALL
SELECT 'lowdoc_code_invalid', COUNT(*) FROM loans WHERE is_lowdoc IS NULL UNION ALL
SELECT 'chgoff_principal_gt_disbursed', COUNT(*) FROM loans WHERE chgoff_principal > disbursed_amt UNION ALL
SELECT 'default_without_chgoff_amt', COUNT(*) FROM loans WHERE is_default = 1 AND COALESCE(chgoff_principal, 0) = 0;
