"""Turn PD x LGD into decisions: expected loss, risk-based pricing, approval cutoffs, swap sets."""
import numpy as np
import pandas as pd


AVG_LIFE_YEARS = 3.0   # term is unusable (see features.LEAKY); assume a portfolio-average life


def avg_life_years(n) -> np.ndarray:
    return np.full(len(n) if hasattr(n, "__len__") else 1, AVG_LIFE_YEARS)


def expected_loss(pd_, lgd, ead) -> np.ndarray:
    return np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead)


def risk_based_rate(pd_, lgd, pricing: dict) -> np.ndarray:
    """All-in annual rate = cost of funds + opex + capital charge + target margin + annualised EL."""
    el_annual = np.asarray(pd_) * np.asarray(lgd) / AVG_LIFE_YEARS
    base = pricing["cost_of_funds"] + pricing["opex"] + pricing["capital_charge"] + pricing["target_margin"]
    return base + el_annual


def realised_profit(df: pd.DataFrame, approve: np.ndarray, net_spread: float) -> float:
    """Illustrative lifetime P&L at a flat price: good loans earn net spread over their
    average life; defaulted loans lose the lender's (post-SBA-guarantee) charged-off principal."""
    d = df[approve]
    life = AVG_LIFE_YEARS
    good = (1 - d["is_default"]) * net_spread * d["ead"] * life
    bad = d["is_default"] * d["lgd_lender"] * d["ead"]
    return float((good - bad).sum())


def profit_curve(df: pd.DataFrame, risk: np.ndarray, net_spread: float, grid=None) -> pd.DataFrame:
    grid = np.linspace(0.05, 1.0, 39) if grid is None else grid
    order = np.argsort(risk)
    rows = []
    for ar in grid:
        approve = np.zeros(len(df), bool)
        approve[order[: int(round(ar * len(df)))]] = True
        rows.append({"approval_rate": ar,
                     "bad_rate": float(df.loc[approve, "is_default"].mean()),
                     "profit_musd": realised_profit(df, approve, net_spread) / 1e6})
    return pd.DataFrame(rows)


def swap_set(df: pd.DataFrame, risk_old: np.ndarray, risk_new: np.ndarray, approval_rate: float) -> pd.DataFrame:
    """Who changes decision when moving from the old model to the new one at the same approval rate."""
    n = int(round(approval_rate * len(df)))
    a_old = np.zeros(len(df), bool); a_old[np.argsort(risk_old)[:n]] = True
    a_new = np.zeros(len(df), bool); a_new[np.argsort(risk_new)[:n]] = True
    groups = {"approved_by_both": a_old & a_new, "swap_in (new approves)": ~a_old & a_new,
              "swap_out (new declines)": a_old & ~a_new, "declined_by_both": ~a_old & ~a_new}
    return pd.DataFrame([{"segment": k, "n": int(m.sum()), "bad_rate": float(df.loc[m, "is_default"].mean())}
                         for k, m in groups.items()])
