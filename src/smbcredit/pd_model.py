"""PD models: a WoE logistic scorecard (champion-style baseline) and a monotone LightGBM challenger."""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .features import CATEGORICAL, FEATURES, MONOTONE, NUMERIC, reason_text


def split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    s = cfg["split"]
    fy = df["approval_fy"]
    return {k: df[fy.between(*s[k])].reset_index(drop=True) for k in ("train", "valid", "test")}


def to_model_frame(df: pd.DataFrame, categories: dict | None = None, features=None) -> tuple[pd.DataFrame, dict]:
    X = df[features or FEATURES].copy()
    if categories is None:
        categories = {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X, categories


# --------------------------------------------------------------------------- scorecard
@dataclass
class WoEScorecard:
    """Classic bureau-style scorecard: quantile-bin -> Weight of Evidence -> logistic regression."""
    n_bins: int = 10
    smoothing: float = 0.5
    bins_: dict = field(default_factory=dict)
    woe_: dict = field(default_factory=dict)
    iv_: dict = field(default_factory=dict)

    def _bin(self, col: str, x: pd.Series) -> pd.Series:
        if col in CATEGORICAL:
            return x.astype("object").fillna("__NA__").astype(str)
        if self.bins_[col] is None:
            return x.astype("float").fillna(-999).astype(str)
        b = pd.cut(x, self.bins_[col], include_lowest=True, labels=False, duplicates="drop")
        return b.astype("float").fillna(-1).astype(int).astype(str)

    def fit(self, df: pd.DataFrame, y: pd.Series):
        for col in FEATURES:
            if col in NUMERIC and df[col].nunique() <= self.n_bins:
                self.bins_[col] = None            # low-cardinality / binary: one bin per value
            elif col in NUMERIC:
                edges = np.unique(np.nanquantile(df[col].astype(float), np.linspace(0, 1, self.n_bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                self.bins_[col] = edges
            b = self._bin(col, df[col])
            t = pd.crosstab(b, y) + self.smoothing
            dist_bad, dist_good = t[1] / t[1].sum(), t[0] / t[0].sum()
            woe = np.log(dist_good / dist_bad)
            self.woe_[col] = woe.to_dict()
            self.iv_[col] = float(((dist_good - dist_bad) * woe).sum())
        self.lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(self.transform(df), y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self._bin(c, df[c]).map(self.woe_[c]).fillna(0.0) for c in FEATURES})

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.transform(df))[:, 1]

    @staticmethod
    def to_points(pd_: np.ndarray, base: int = 600, base_odds: float = 30, pdo: int = 20) -> np.ndarray:
        """Map PD to a score: `base` points at `base_odds`:1 good/bad, +`pdo` doubles the odds."""
        odds = (1 - pd_) / np.clip(pd_, 1e-9, None)
        return base + pdo / np.log(2) * np.log(odds / base_odds)


# --------------------------------------------------------------------------- GBM
@dataclass
class MonotoneGBM:
    params: dict
    model_: lgb.Booster | None = None
    categories_: dict | None = None
    calibrator_: IsotonicRegression | None = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame):
        Xtr, self.categories_ = to_model_frame(train)
        Xva, _ = to_model_frame(valid, self.categories_)
        self.medians_ = train[NUMERIC].median().to_dict()
        p = dict(self.params)
        n_est, es = p.pop("n_estimators"), p.pop("early_stopping_rounds")
        p.update(objective="binary", metric="auc", verbose=-1, seed=42, num_threads=2, deterministic=True, force_row_wise=True,
                 monotone_constraints=[MONOTONE.get(c, 0) for c in FEATURES],
                 monotone_constraints_method="advanced")
        dtr = lgb.Dataset(Xtr, train["is_default"], categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(Xva, valid["is_default"], reference=dtr)
        self.model_ = lgb.train(p, dtr, n_est, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
        # Calibrate on the validation vintages (held out from tree fitting decisions except early stop).
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
        self.calibrator_.fit(self.predict_raw(valid), valid["is_default"])
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = to_model_frame(df, self.categories_)
        return self.model_.predict(X, num_iteration=self.model_.best_iteration)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator_.predict(self.predict_raw(df))

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact TreeSHAP contributions (log-odds) from LightGBM; last column is the bias."""
        X, _ = to_model_frame(df, self.categories_)
        c = self.model_.predict(X, num_iteration=self.model_.best_iteration, pred_contrib=True)
        return pd.DataFrame(c, columns=FEATURES + ["_bias"])

    def reason_codes(self, df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Top-k features pushing each applicant's risk UP (adverse-action style)."""
        c = self.contributions(df).drop(columns="_bias")
        vals = c.to_numpy()
        order = np.argsort(-vals, axis=1)[:, :k]
        rows = []
        for i in range(len(df)):
            r = {}
            for j in range(k):
                f = FEATURES[order[i, j]]
                r[f"reason_{j + 1}"] = (reason_text(f, df.iloc[i][f], self.medians_.get(f))
                                        if vals[i, order[i, j]] > 0 else "")
            rows.append(r)
        return pd.DataFrame(rows)


def recalibrate_intercept(p: np.ndarray, target_rate: float) -> np.ndarray:
    """Shift log-odds so the portfolio mean PD matches a target central tendency
    (e.g. a through-the-cycle or stressed default rate) while preserving rank order."""
    from scipy.optimize import brentq
    logit = np.log(p / (1 - p))
    f = lambda d: np.mean(1 / (1 + np.exp(-(logit + d)))) - target_rate
    d = brentq(f, -10, 10)
    return 1 / (1 + np.exp(-(logit + d)))
