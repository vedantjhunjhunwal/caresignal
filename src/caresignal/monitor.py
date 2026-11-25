"""Feature and score drift monitoring with the Population Stability Index (PSI).
Rule of thumb: < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate / consider retraining."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: fall back to value counts
        return psi_categorical(ref.astype(str), cur.astype(str))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    c = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur) -> float:
    r = pd.Series(ref).astype(str).value_counts(normalize=True)
    c = pd.Series(cur).astype(str).value_counts(normalize=True)
    cats = r.index.union(c.index)
    r, c = r.reindex(cats, fill_value=0) + EPS, c.reindex(cats, fill_value=0) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def status(psi: float) -> str:
    return "stable" if psi < 0.10 else "watch" if psi < 0.25 else "investigate"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "type": "numeric", "psi": psi_numeric(ref[c], cur[c])} for c in numeric]
    rows += [{"feature": c, "type": "categorical", "psi": psi_categorical(ref[c], cur[c])} for c in categorical]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out.psi.map(status)
    return out
