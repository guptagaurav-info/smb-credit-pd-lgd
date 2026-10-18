"""Stage 0: download the raw SBA FOIA extract, verify integrity, land it as parquet.

Raw is kept all-VARCHAR so that every type cast happens in SQL where it is visible
and counted in the data-quality report (sql/01_clean.sql).
"""
import hashlib
import urllib.request

import duckdb

from .config import load_config, p


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    d = cfg["data"]
    raw = p(d["raw_csv"])
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print(f"[ingest] downloading {d['url']}")
        urllib.request.urlretrieve(d["url"], raw)
    digest = sha256(raw)
    if digest != d["sha256"]:
        raise ValueError(f"[ingest] checksum mismatch: {digest} != {d['sha256']} (source changed?)")
    print(f"[ingest] sha256 verified {digest[:12]}…")
    duckdb.execute("SET enable_progress_bar = false")
    out = p(d["raw_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(
        f"COPY (SELECT * FROM read_csv('{raw}', all_varchar=true, header=true)) "
        f"TO '{out}' (FORMAT parquet)"
    )
    n = duckdb.sql(f"SELECT COUNT(*) FROM '{out}'").fetchone()[0]
    print(f"[ingest] landed {n:,} rows -> {out.name}")


if __name__ == "__main__":
    run()
