"""SQL transform tests on a hand-built raw extract (no download needed)."""
import duckdb
import pandas as pd
import pytest

from smbcredit.transform import run_sql

COLS = ["LoanNr_ChkDgt", "Name", "City", "State", "Zip", "Bank", "BankState", "NAICS", "ApprovalDate",
        "ApprovalFY", "Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob", "FranchiseCode", "UrbanRural",
        "RevLineCr", "LowDoc", "ChgOffDate", "DisbursementDate", "DisbursementGross", "BalanceGross",
        "MIS_Status", "ChgOffPrinGr", "GrAppv", "SBA_Appv"]


def row(i, bank, appr, fy, status, chg=None, chg_amt="$0.00 ", naics="722410", ur="1"):
    return dict(zip(COLS, [str(i), f"BIZ {i}", "NYC", "NY", "10001", bank, "NY", naics, appr, fy, "84", "5", "2",
                           "0", "3", "1", ur, "N", "N", chg, appr, "$100,000.00 ", "$0.00 ", status, chg_amt,
                           "$100,000.00 ", "$75,000.00 "]))


@pytest.fixture
def con(tmp_path):
    rows = [
        row(1, "BANK A", "28-Feb-80", "1980", "P I F"),                         # 2-digit 19xx year
        row(2, "BANK A", "15-Nov-99", "1976A", "CHGOFF", "10-Jan-01", "$60,000.00 "),
        row(3, "BANK A", "05-Mar-01", "2001", "P I F"),                         # FY2001: must NOT see #2's charge-off
        row(4, "BANK A", "05-Mar-03", "2003", "P I F"),                         # FY2003: must see it
        row(5, "BANK B", "05-Mar-03", "2003", None),                            # unresolved -> excluded
    ]
    raw = tmp_path / "raw.parquet"
    pd.DataFrame(rows).to_parquet(raw)
    c = duckdb.connect()
    run_sql(c, raw, k=1.0, fy_min=1970, fy_max=2014, verbose=False)
    return c


def test_types_and_dates(con):
    d = con.sql("SELECT * FROM loans ORDER BY loan_id").df()
    assert d.loc[0, "approval_date"].year == 1980            # not 2080
    assert d.loc[1, "approval_fy"] == 1976                    # '1976A' cleaned
    assert d.loc[1, "chgoff_principal"] == 60000.0            # '$60,000.00 ' parsed
    assert d.loc[1, "is_default"] == 1 and d.loc[0, "is_default"] == 0
    assert pd.isna(d.loc[4, "is_default"])


def test_model_base_excludes_unresolved(con):
    ids = set(con.sql("SELECT loan_id FROM model_base").df().loan_id)
    assert "5" not in ids


def test_point_in_time_no_lookahead(con):
    """Loan 2 charges off in FY2001 (Jan-2001). Bank A's rate for an FY2001 loan must ignore it;
    an FY2003 loan must include it."""
    b = con.sql("SELECT fy, n_prior, co_prior FROM pit_bank WHERE bank='BANK A' AND fy IN (2001, 2003) ORDER BY fy").df()
    assert b.loc[0, "co_prior"] == 0
    assert b.loc[1, "co_prior"] == 1


def test_lgd_bounds(con):
    d = con.sql("SELECT lgd_gross, lgd_lender FROM model_base").df()
    assert d.lgd_gross.between(0, 1).all() and (d.lgd_lender <= d.lgd_gross + 1e-12).all()
