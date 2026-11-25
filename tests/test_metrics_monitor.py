import numpy as np
import pandas as pd

from caresignal import metrics as M
from caresignal.monitor import drift_report, psi_categorical, psi_numeric


def test_ece_is_small_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.4, 50_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert M.ece(y, p) < 0.01
    assert M.ece(y, np.clip(p * 2, 0, 1)) > 0.1  # over-confident scores are caught


def test_capture_at_top_fraction():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    out = M.capture_at(y, p, 0.2)
    assert out == {"recall": 1.0, "precision": 1.0, "lift": 5.0}


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = y * 0.3 + rng.uniform(size=2000)
    lo, hi = M.bootstrap_auc(y, p, n=200)
    from sklearn.metrics import roc_auc_score
    assert lo < roc_auc_score(y, p) < hi


def test_subgroup_table_hides_auroc_for_small_groups():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.integers(0, 2, 1000), "p": rng.uniform(size=1000),
                       "g": ["big"] * 900 + ["small"] * 100})
    t = M.subgroup_table(df, "y", "p", "g", 0.5)
    assert not np.isnan(t.set_index("group").loc["big", "auroc"])
    assert np.isnan(t.set_index("group").loc["small", "auroc"])


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(3)
    a = rng.normal(size=20_000)
    assert psi_numeric(a, rng.normal(size=20_000)) < 0.01
    assert psi_numeric(a, rng.normal(1.0, 1, size=20_000)) > 0.25
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) < 1e-6


def test_drift_report_flags_status():
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"x": rng.normal(size=5000), "c": ["u"] * 5000})
    cur = pd.DataFrame({"x": rng.normal(2, 1, size=5000), "c": ["u"] * 4000 + ["v"] * 1000})
    rep = drift_report(ref, cur, ["x"], ["c"]).set_index("feature")
    assert rep.loc["x", "status"] == "investigate"
    assert rep.loc["c", "status"] in {"watch", "investigate"}
