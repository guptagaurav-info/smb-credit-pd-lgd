"""End-to-end run: ingest -> SQL transforms -> PD (scorecard vs GBM) -> LGD -> EL/pricing/policy -> monitoring -> report."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from . import ingest, transform
from .config import load_config, p
from .features import FEATURES, LEAKY, NUMERIC
from .lgd_model import FractionalLogitLGD, GBMLGD, SegmentLGD, lgd_metrics
from .metrics import calibration_table, psi, summary
from .pd_model import MonotoneGBM, WoEScorecard, recalibrate_intercept, split
from .policy import expected_loss, profit_curve, risk_based_rate, swap_set

FIG = p("reports/figures")


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


def plot_auc_by_vintage(by_fy: pd.DataFrame, splits: dict, path, dpi: int = 130) -> None:
    """AUC by approval fiscal year (integer year ticks) with default-rate bars and shaded holdout periods."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax2 = ax.twinx()
    ax2.bar(by_fy.fy, by_fy.default_rate * 100, width=.7, alpha=.25, color="grey")
    ax2.set_ylabel("Default rate (%)")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    shade = [("valid", "orange", "Validation FY{0}–{1}"), ("test", "red", "Test FY{0}–{1} (crisis)")]
    for k, c, _ in shade:
        ax.axvspan(splits[k][0] - .5, splits[k][1] + .5, color=c, alpha=.08)
    l1, = ax.plot(by_fy.fy, by_fy.auc_gbm, "o-", color="tab:orange", label="Monotone GBM (champion)")
    l2, = ax.plot(by_fy.fy, by_fy.auc_scorecard, "o-", color="tab:blue", label="WoE scorecard")
    ax.xaxis.set_major_locator(MultipleLocator(2)); ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(by_fy.fy.min() - .7, by_fy.fy.max() + .7)
    ax.set(xlabel="Approval fiscal year", ylabel="AUC", title="Ranking power by approval year")
    handles = [l1, l2, Patch(color="grey", alpha=.25, label="Default rate (right axis)")]
    handles += [Patch(color=c, alpha=.15, label=lab.format(*splits[k])) for k, c, lab in shade]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def leakage_check(cfg, tr, va, te) -> dict:
    """Train the same GBM with the leaky `term_months` added, and tabulate default rate by exact term."""
    import lightgbm as lgb
    from .features import CATEGORICAL
    from .pd_model import to_model_frame
    feats = FEATURES + LEAKY
    Xtr, cats = to_model_frame(tr, features=feats)
    Xte, _ = to_model_frame(te, cats, features=feats)
    Xva, _ = to_model_frame(va, cats, features=feats)
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, seed=1, deterministic=True, force_row_wise=True),
                  lgb.Dataset(Xtr, tr.is_default, categorical_feature=CATEGORICAL), 2000,
                  valid_sets=[lgb.Dataset(Xva, va.is_default)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    auc_naive = roc_auc_score(te.is_default, m.predict(Xte, num_iteration=m.best_iteration))
    standard = {12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300}
    allv = pd.concat([tr, va, te])
    allv = allv.assign(term_type=np.where(allv.term_months.isin(standard), "standard term", "non-standard term"))
    tab = allv.groupby("term_type").agg(n=("is_default", "size"), default_rate=("is_default", "mean"))
    top = (allv.groupby("term_months").agg(n=("is_default", "size"), default_rate=("is_default", "mean"))
           .query("n >= 2000").sort_values("default_rate"))
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue" if t in standard else "tab:red" for t in top.index]
    ax.bar(top.index.astype(str), top.default_rate, color=colors)
    ax.set(xlabel="Recorded term (months), n>=2,000; red = non-standard",
           ylabel="Default rate", title="Leakage check: default rate by recorded term")
    ax.tick_params(axis="x", rotation=90); _save(fig, "leakage_term.png")
    return {"test_auc_with_term": float(auc_naive),
            "default_rate_by_term_type": tab.reset_index().to_dict("records"),
            "summary": f"naive test AUC with term = {auc_naive:.3f}"}


def main():
    t0 = time.time()
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    ingest.run(cfg)
    transform.run(cfg)
    df = transform.load_model_base(cfg)
    s = split(df, cfg)
    tr, va, te = s["train"], s["valid"], s["test"]
    out: dict = {"split": {k: {"n": len(v), "default_rate": float(v.is_default.mean()),
                               "fy": cfg["split"][k]} for k, v in s.items()}}
    print("[split]", json.dumps(out["split"]))

    # ---------------- Leakage diagnostic: what a naive build would have reported
    out["leakage_diagnostic"] = leakage_check(cfg, tr, va, te)
    print("[leakage]", out["leakage_diagnostic"]["summary"])

    # ---------------- PD
    sc = WoEScorecard().fit(tr, tr["is_default"])
    gbm = MonotoneGBM(cfg["gbm"]).fit(tr, va)
    preds = {}
    for name, m in [("scorecard", sc), ("gbm", gbm)]:
        preds[name] = {k: m.predict_proba(v) for k, v in s.items()}
        out[f"pd_{name}"] = {k: summary(s[k]["is_default"], preds[name][k]) for k in s}
        print(f"[pd] {name}: " + ", ".join(f"{k} AUC={out[f'pd_{name}'][k]['auc']:.3f}" for k in s))
    out["gbm_best_iteration"] = int(gbm.model_.best_iteration)
    out["scorecard_iv"] = dict(sorted(sc.iv_.items(), key=lambda kv: -kv[1]))

    # Stressed recalibration: same ranking, central tendency moved to a stressed scenario.
    # (Target here = the observed validation default rate x 1.5 — a stress overlay, NOT test outcomes.)
    stress_target = 1.5 * float(va.is_default.mean())
    p_te_stress = recalibrate_intercept(preds["gbm"]["test"], stress_target)
    out["pd_gbm_test_stress_overlay"] = {"target_mean_pd": stress_target, **summary(te.is_default, p_te_stress)}

    # AUC by vintage — rank-ordering stability through the cycle
    allv = pd.concat([tr, va, te], ignore_index=True)
    allp = {n: np.concatenate([preds[n][k] for k in ("train", "valid", "test")]) for n in preds}
    by_fy = []
    for fy, g in allv.groupby("approval_fy"):
        idx = g.index.to_numpy()
        by_fy.append({"fy": int(fy), "n": len(g), "default_rate": float(g.is_default.mean()),
                      **{f"auc_{n}": float(roc_auc_score(g.is_default, allp[n][idx])) for n in allp}})
    by_fy = pd.DataFrame(by_fy); by_fy.to_csv(p("reports/auc_by_vintage.csv"), index=False)

    # Explainability: global mean |SHAP| and per-loan reason codes on the test set
    contrib = gbm.contributions(te.sample(50_000, random_state=0) if len(te) > 50_000 else te)
    imp = contrib.drop(columns="_bias").abs().mean().sort_values(ascending=False)
    out["gbm_mean_abs_shap"] = imp.round(4).to_dict()
    sample = te.sample(25, random_state=7).reset_index(drop=True)
    rc = gbm.reason_codes(sample)
    sample_out = pd.concat([sample[["loan_id", "approval_fy", "naics2", "state", "bank"]],
                            pd.Series(gbm.predict_proba(sample), name="pd"), rc,
                            sample[["is_default"]]], axis=1)
    sample_out.to_csv(p("reports/sample_reason_codes.csv"), index=False)

    # ---------------- LGD (defaults only)
    dtr, dva, dte = (d[d.is_default == 1] for d in (tr, va, te))
    class _Mean:  # naive benchmark every model must beat
        def fit(self, d, y): self.m = float(np.mean(y)); return self
        def predict(self, d): return np.full(len(d), self.m)
    lgd_models = {"portfolio_mean": _Mean().fit(dtr, dtr["lgd_gross"]),
                  "segment_lookup": SegmentLGD().fit(dtr, dtr["lgd_gross"]),
                  "fractional_logit": FractionalLogitLGD().fit(dtr, dtr["lgd_gross"]),
                  "gbm": GBMLGD().fit(dtr, dva)}
    out["lgd"] = {n: {"valid": lgd_metrics(dva.lgd_gross, m.predict(dva)),
                      "test": lgd_metrics(dte.lgd_gross, m.predict(dte))} for n, m in lgd_models.items()}
    # Champion chosen on VALIDATION only (never on test), and a model must beat the naive
    # portfolio mean by a material margin (2% MAE) to justify its complexity.
    base_mae = out["lgd"]["portfolio_mean"]["valid"]["mae"]
    eligible = {n: r["valid"]["mae"] for n, r in out["lgd"].items() if r["valid"]["mae"] <= 0.98 * base_mae}
    lgd_champ = min(eligible, key=eligible.get) if eligible else "portfolio_mean"
    gl = lgd_models[lgd_champ]
    out["lgd_champion"] = lgd_champ
    for n, r in out["lgd"].items():
        print(f"[lgd] {n:17s} valid MAE={r['valid']['mae']:.3f}  test MAE={r['test']['mae']:.3f}  test rho={r['test']['spearman']:.3f}")
    print(f"[lgd] champion (by valid MAE): {lgd_champ}")

    # ---------------- EL, pricing, policy on the test vintages
    te = te.copy()
    te["pd"] = preds["gbm"]["test"]
    te["lgd_gross_hat"] = gl.predict(te)
    te["lgd_lender_hat"] = te["lgd_gross_hat"] * (1 - te["sba_guarantee_pct"].clip(0, 1))
    te["el_usd"] = expected_loss(te.pd, te.lgd_lender_hat, te.ead)
    te["risk_rate"] = risk_based_rate(te.pd, te.lgd_lender_hat, cfg["pricing"])
    te["pd_decile"] = pd.qcut(te.pd.rank(method="first"), 10, labels=range(1, 11))
    grid = te.groupby("pd_decile", observed=True).agg(
        n=("pd", "size"), mean_pd=("pd", "mean"), observed_dr=("is_default", "mean"),
        mean_lgd_lender_hat=("lgd_lender_hat", "mean"),
        el_pct_of_ead=("el_usd", lambda x: x.sum() / te.loc[x.index, "ead"].sum()),
        indicated_rate=("risk_rate", "median")).reset_index()
    grid.to_csv(p("reports/pricing_grid_by_pd_decile.csv"), index=False)

    pr = cfg["pricing"]
    net_spread = pr["target_margin"] + pr["capital_charge"]   # flat-price book: spread over CoF + opex
    curves = {n: profit_curve(te, preds[n]["test"], net_spread) for n in preds}
    best = {n: c.loc[c.profit_musd.idxmax()].to_dict() for n, c in curves.items()}
    out["policy_best_cutoff_test"] = best
    ar = best["gbm"]["approval_rate"]
    sw = swap_set(te, preds["scorecard"]["test"], preds["gbm"]["test"], ar)
    sw.to_csv(p("reports/swap_set.csv"), index=False)
    out["swap_set_at_gbm_optimal_approval"] = {"approval_rate": ar, "table": sw.to_dict("records")}

    # ---------------- Monitoring: score PSI and feature CSI
    out["psi_score"] = {"valid_vs_train": psi(preds["gbm"]["train"], preds["gbm"]["valid"]),
                        "test_vs_train": psi(preds["gbm"]["train"], preds["gbm"]["test"])}
    out["csi_test_vs_train"] = {c: round(psi(tr[c], te[c]), 4) for c in NUMERIC}

    # ---------------- Figures
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for n in preds:
        f_, t_, _ = roc_curve(te.is_default, preds[n]["test"])
        ax.plot(f_, t_, label=f"{n} (AUC {out[f'pd_{n}']['test']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set(xlabel="FPR", ylabel="TPR", title="ROC — OOT test FY2006-07")
    ax.legend(); _save(fig, "roc_test.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lab, pp in [("isotonic (valid-calibrated)", preds["gbm"]["test"]), ("stress overlay", p_te_stress)]:
        ct = calibration_table(te.is_default, pp)
        ax.plot(ct.predicted, ct.observed, "o-", label=lab)
    m = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([0, m], [0, m], "k:", lw=.8)
    ax.set(xlabel="Predicted PD (decile mean)", ylabel="Observed default rate", title="Calibration — OOT test")
    ax.legend(); _save(fig, "calibration_test.png")

    plot_auc_by_vintage(by_fy, cfg["split"], FIG / "auc_by_vintage.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    imp.head(15)[::-1].plot.barh(ax=ax); ax.set(title="GBM global importance (mean |SHAP|, log-odds)")
    _save(fig, "shap_importance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for n, c in curves.items():
        ax.plot(c.approval_rate, c.profit_musd, label=n)
    ax.set(xlabel="Approval rate", ylabel="Illustrative lifetime profit ($M)", title="Profit curve — OOT test")
    ax.legend(); _save(fig, "profit_curve.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dte.lgd_gross, bins=50, alpha=.6, density=True, label="actual")
    ax.hist(gl.predict(dte), bins=50, alpha=.6, density=True, label=f"{lgd_champ} predicted")
    ax.set(xlabel="Gross LGD (charged-off principal / disbursed)", title="LGD on defaulted test loans")
    ax.legend(); _save(fig, "lgd_distribution.png")

    # Vintage curves straight from the warehouse (SQL analytics)
    con = transform.connect(cfg)
    vc = con.sql(p("sql/03_vintage_curves.sql").read_text()).df(); con.close()
    vc.to_csv(p("reports/vintage_curves.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cmap = plt.get_cmap("viridis")
    fys = sorted(vc.approval_fy.unique())
    for i, fy in enumerate(fys):
        g = vc[vc.approval_fy == fy]
        ax.plot(g.mob, g.cum_chgoff_rate, color=cmap(i / max(1, len(fys) - 1)), label=str(fy) if fy % 2 else None)
    ax.set(xlabel="Months on book", ylabel="Cumulative charge-off rate", title="Vintage curves (resolved loans)")
    ax.legend(fontsize=7, ncol=2); _save(fig, "vintage_curves.png")

    out["runtime_sec"] = round(time.time() - t0, 1)
    with open(p("reports/metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[done] {out['runtime_sec']}s -> reports/metrics.json")


if __name__ == "__main__":
    main()
