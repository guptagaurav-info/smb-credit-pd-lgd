import numpy as np
import pandas as pd

from smbcredit.metrics import ks, psi
from smbcredit.pd_model import recalibrate_intercept
from smbcredit.policy import swap_set


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-9


def test_psi_binary_detects_shift():
    a = np.r_[np.zeros(900), np.ones(100)]
    b = np.r_[np.zeros(500), np.ones(500)]
    assert psi(a, b) > 0.25


def test_ks_perfect():
    y = np.r_[np.zeros(50), np.ones(50)]
    assert ks(y, y) == 1.0


def test_recalibration_hits_target_and_keeps_rank():
    p = np.random.default_rng(1).uniform(0.01, 0.4, 1000)
    q = recalibrate_intercept(p, 0.30)
    assert abs(q.mean() - 0.30) < 1e-6
    assert (np.argsort(p) == np.argsort(q)).all()


def test_swap_set_partitions_population():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"is_default": rng.integers(0, 2, 1000)})
    t = swap_set(df, rng.random(1000), rng.random(1000), 0.6)
    assert t.n.sum() == 1000
    assert t.set_index("segment").loc["swap_in (new approves)", "n"] == t.set_index("segment").loc["swap_out (new declines)", "n"]
