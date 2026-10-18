"""Stage 1-2: SQL transforms in a local DuckDB warehouse (stand-in for Snowflake/Postgres)."""
import duckdb
import pandas as pd

from .config import load_config, p


def connect(cfg: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(p(cfg["data"]["warehouse"])))
    con.execute("SET enable_progress_bar = false")
    return con


def run_sql(con, raw_parquet, k: float, fy_min: int, fy_max: int, verbose: bool = True) -> None:
    con.execute(f"SET VARIABLE raw_parquet = '{raw_parquet}'")
    con.execute(f"SET VARIABLE k = {float(k)}")
    con.execute(f"SET VARIABLE fy_min = {int(fy_min)}")
    con.execute(f"SET VARIABLE fy_max = {int(fy_max)}")
    for f in ("01_clean.sql", "02_features.sql"):
        con.execute(p(f"sql/{f}").read_text())
        if verbose:
            print(f"[transform] ran sql/{f}")


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    con = connect(cfg)
    run_sql(con, p(cfg["data"]["raw_parquet"]), cfg["pit_shrinkage_k"],
            cfg["split"]["train"][0], cfg["split"]["test"][1])
    dq = con.sql("SELECT * FROM dq_report").df()
    print(dq.to_string(index=False))
    n = con.sql("SELECT COUNT(*) FROM model_base").fetchone()[0]
    print(f"[transform] model_base: {n:,} loans")
    dq.to_csv(p("reports/data_quality.csv"), index=False)
    con.close()
    return dq


def load_model_base(cfg: dict) -> pd.DataFrame:
    con = connect(cfg)
    df = con.sql("SELECT * FROM model_base").df()
    con.close()
    return df


if __name__ == "__main__":
    run()
