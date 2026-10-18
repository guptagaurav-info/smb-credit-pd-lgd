"""Credit-model metrics: discrimination, calibration, stability."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return t.reset_index()


def ece(y, p, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float((t["n"] * (t["predicted"] - t["observed"]).abs()).sum() / t["n"].sum())


def summary(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "default_rate": float(np.mean(y)),
        "mean_pd": float(np.mean(p)),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index using expected-sample quantile bins."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    vals = np.unique(expected)
    if len(vals) <= bins:   # discrete / binary: one bin per observed value
        e = np.array([(expected == v).mean() for v in vals])
        a = np.array([(actual == v).mean() for v in vals])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
