"""Evaluation helpers: discrimination, calibration, operating-point and subgroup metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-frequency bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(err)


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def capture_at(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    """Share of all readmissions found if outreach covers the top `frac` of patients by risk."""
    k = max(1, int(round(frac * len(p))))
    top = np.argsort(-p)[:k]
    recall = y[top].sum() / max(1, y.sum())
    return {"recall": float(recall), "precision": float(y[top].mean()), "lift": float(y[top].mean() / y.mean())}


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "top10": capture_at(y, p, 0.10),
        "top20": capture_at(y, p, 0.20),
    }


def subgroup_table(df: pd.DataFrame, y: str, p: str, group: str, threshold: float, min_n: int = 300,
                   min_events: int = 30) -> pd.DataFrame:
    """Per-group metrics. AUROC is only reported where the group has >= min_n rows and
    >= min_events positives; below that the estimate is too noisy to compare."""
    rows = []
    for g, d in df.groupby(group, observed=True):
        yy, pp = d[y].to_numpy(), d[p].to_numpy()
        flagged = pp >= threshold
        ok = len(d) >= min_n and min_events <= yy.sum() < len(yy)
        lo = hi = np.nan
        if ok:
            lo, hi = bootstrap_auc(yy, pp, n=300, seed=1)
        rows.append({
            "attribute": group, "group": g, "n": len(d), "events": int(yy.sum()), "prevalence": yy.mean(),
            "auroc": roc_auc_score(yy, pp) if ok else np.nan, "auroc_lo": lo, "auroc_hi": hi,
            "flag_rate": flagged.mean(),
            "tpr": flagged[yy == 1].mean() if yy.sum() else np.nan,
            "mean_pred": pp.mean(),
        })
    return pd.DataFrame(rows)
