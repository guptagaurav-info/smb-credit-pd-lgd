"""LGD: severity given default, modelled on charged-off loans only (two-stage EL = PD x LGD x EAD)."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import CATEGORICAL, FEATURES
from .pd_model import to_model_frame

LGD_NUMERIC = ["log_approved_amt", "log_employees", "is_new_business", "sba_guarantee_pct",
               "is_revolver", "is_lowdoc", "rel_naics_chgoff", "rel_bank_chgoff"]


class FractionalLogitLGD:
    """Papke-Wooldridge fractional logit: GLM(binomial, logit) on a [0,1] target."""

    def fit(self, df, y):
        self.m_ = sm.GLM(y, self._X(df), family=sm.families.Binomial()).fit()
        return self

    def _X(self, df):
        return sm.add_constant(df[LGD_NUMERIC].astype(float).fillna(df[LGD_NUMERIC].median()), has_constant="add")

    def predict(self, df):
        return np.asarray(self.m_.predict(self._X(df)))


class GBMLGD:
    """LightGBM with cross-entropy objective, which accepts continuous labels in [0, 1]."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train, valid):
        Xtr, self.cats_ = to_model_frame(train)
        Xva, _ = to_model_frame(valid, self.cats_)
        p = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=31, min_child_samples=300,
                 feature_fraction=0.8, lambda_l2=5.0, verbose=-1, seed=self.seed,
                 deterministic=True, force_row_wise=True)
        dtr = lgb.Dataset(Xtr, train["lgd_gross"], categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(Xva, valid["lgd_gross"], reference=dtr)
        self.m_ = lgb.train(p, dtr, 2000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        return self

    def predict(self, df):
        X, _ = to_model_frame(df, self.cats_)
        return self.m_.predict(X, num_iteration=self.m_.best_iteration)


def lgd_metrics(y, p) -> dict:
    from scipy.stats import spearmanr
    y, p = np.asarray(y), np.asarray(p)
    return {"n": int(len(y)), "mean_actual": float(y.mean()), "mean_pred": float(p.mean()),
            "mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
            "spearman": float(spearmanr(y, p).statistic)}


class SegmentLGD:
    """Benchmark a credit committee understands: shrunk mean LGD by segment."""

    def __init__(self, keys=("naics2", "is_revolver", "guar_bucket"), k: float = 100.0):
        self.keys, self.k = list(keys), k

    @staticmethod
    def _prep(df):
        return df.assign(guar_bucket=pd.cut(df["sba_guarantee_pct"], [-1, .55, .76, 2], labels=["<=55%", "55-76%", ">76%"]).astype(str),
                         naics2=df["naics2"].fillna("NA"), is_revolver=df["is_revolver"].fillna(-1))

    def fit(self, df, y):
        d = self._prep(df).assign(y=np.asarray(y))
        self.mu_ = float(d.y.mean())
        g = d.groupby(self.keys).y.agg(["sum", "count"])
        self.table_ = ((g["sum"] + self.k * self.mu_) / (g["count"] + self.k)).rename("lgd").reset_index()
        return self

    def predict(self, df):
        d = self._prep(df)[self.keys].merge(self.table_, on=self.keys, how="left")
        return d["lgd"].fillna(self.mu_).to_numpy()
