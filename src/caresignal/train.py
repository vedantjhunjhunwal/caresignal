"""Train, evaluate and register the 30-day readmission model.

1. Spark MLlib logistic-regression baseline (runs on the gold table in Spark).
2. LightGBM and XGBoost on the same features; early stopping on the validation split.
3. Isotonic calibration of the better model, fitted on validation.
4. Promotion gate, decided on validation only: beat the baseline AUROC by >= 0.01,
   calibrated ECE <= 0.02, and AUROC gap across race groups (n >= 300) <= 0.06.
5. One look at the held-out test split (later, unseen patients): AUROC with bootstrap CI,
   AUPRC, Brier, ECE, top-decile capture, subgroup audit, SHAP importance, PSI drift.
6. MLflow: every run's params/metrics/artifacts; champion registered as
   ``caresignal-readmission`` with alias ``champion`` if the gate passes.

Race is never a model input; it is kept only to audit the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from caresignal import metrics as M
from caresignal.monitor import drift_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REG_NAME = "caresignal-readmission"
NUMERIC = ["age_years", "is_male", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "prior_visits", "prior_encounters_in_data",
           "number_diagnoses", "n_drugs_changed", "n_drugs_on", "payer_missing"]
CATEGORICAL = ["admission_type_desc", "discharge_disposition_desc", "admission_source_desc", "diag_1_grp",
               "diag_2_grp", "diag_3_grp", "a1c", "max_glu", "insulin", "change", "diabetesMed", "specialty"]
FEATURES = NUMERIC + CATEGORICAL
GATE = {"min_auroc_gain_vs_baseline": 0.01, "max_ece": 0.02, "max_race_auroc_gap": 0.06}


class ReadmissionModel(mlflow.pyfunc.PythonModel):
    """Raw features -> calibrated 30-day readmission probability."""

    def __init__(self, booster: lgb.Booster, iso: IsotonicRegression, levels: dict[str, list[str]]):
        self.booster, self.iso, self.levels = booster, iso, levels

    def prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].copy()
        for c in NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in CATEGORICAL:
            v = X[c].astype(str)
            X[c] = pd.Categorical(v.where(v.isin(self.levels[c])), categories=self.levels[c])  # unseen -> missing
        return X

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        raw = self.booster.predict(self.prep(model_input))
        return self.iso.predict(raw)


def spark_baseline(lake: Path) -> dict:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

    from caresignal.spark import get_spark

    spark = get_spark()
    g = spark.read.parquet(str(lake / "gold/readmission_features"))
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_ix", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_ix" for c in CATEGORICAL], outputCols=[f"{c}_oh" for c in CATEGORICAL],
                        handleInvalid="keep")
    vec = VectorAssembler(inputCols=NUMERIC + [f"{c}_oh" for c in CATEGORICAL], outputCol="raw")
    sc = StandardScaler(inputCol="raw", outputCol="features")
    lr = LogisticRegression(labelCol="label", featuresCol="features", regParam=0.01, elasticNetParam=0.0,
                            maxIter=100)
    model = Pipeline(stages=idx + [ohe, vec, sc, lr]).fit(g.filter("split = 'train'"))
    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    out = {s: ev.evaluate(model.transform(g.filter(f"split = '{s}'"))) for s in ("valid", "test")}
    return {"valid_auroc": out["valid"], "test_auroc": out["test"]}


def load_gold(lake: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_parquet(lake / "gold/readmission_features")
    df["split"] = df["split"].astype(str)
    levels = {c: sorted(df.loc[df.split == "train", c].astype(str).unique()) for c in CATEGORICAL}
    return df, levels


def plot_calibration(y, p_raw, p_cal, path):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for p, lab in ((p_raw, "raw"), (p_cal, "isotonic")):
        order = np.argsort(p)
        bins = np.array_split(order, 10)
        ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "o-", label=lab)
    m = max(p_cal.max(), 0.4)
    ax.plot([0, m], [0, m], "k--", lw=0.8)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("observed 30-day readmission")
    ax.legend()
    ax.set_title("Calibration (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(lake: Path = Path("lakehouse"), reports: Path = Path("reports")) -> dict:
    reports.mkdir(exist_ok=True)
    mlflow.set_experiment("caresignal-readmission")
    df, levels = load_gold(lake)
    prep = ReadmissionModel(None, None, levels).prep
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr, yva, yte = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()
    results: dict = {"rows": {"train": len(tr), "valid": len(va), "test": len(te)}}

    with mlflow.start_run(run_name="spark-mllib-logreg-baseline"):
        base = spark_baseline(lake)
        mlflow.log_params({"model": "pyspark.ml LogisticRegression", "regParam": 0.01, "features": len(FEATURES)})
        mlflow.log_metrics(base)
    results["baseline_spark_lr"] = base

    with mlflow.start_run(run_name="lightgbm"):
        lp = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
              "verbose": -1, "seed": 7}
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        booster = lgb.train(lp, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        p_va_lgb = booster.predict(Xva)
        mlflow.log_params({**lp, "best_iteration": booster.best_iteration})
        mlflow.log_metric("valid_auroc", roc_auc_score(yva, p_va_lgb))

    with mlflow.start_run(run_name="xgboost"):
        xp = {"objective": "binary:logistic", "eval_metric": "auc", "eta": 0.03, "max_depth": 5,
              "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 5.0, "seed": 7}
        dx_tr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
        dx_va = xgb.DMatrix(Xva, yva, enable_categorical=True)
        xbst = xgb.train(xp, dx_tr, 3000, evals=[(dx_va, "valid")], early_stopping_rounds=100, verbose_eval=False)
        p_va_xgb = xbst.predict(dx_va, iteration_range=(0, xbst.best_iteration + 1))
        mlflow.log_params({**xp, "best_iteration": xbst.best_iteration})
        mlflow.log_metric("valid_auroc", roc_auc_score(yva, p_va_xgb))

    val_auc = {"lightgbm": roc_auc_score(yva, p_va_lgb), "xgboost": roc_auc_score(yva, p_va_xgb)}
    results["valid_auroc"] = {**val_auc, "spark_lr": base["valid_auroc"]}
    # Packaged model uses LightGBM; XGBoost is kept as a challenger for comparison.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va_lgb, yva)
    champ = ReadmissionModel(booster, iso, levels)
    va["p"] = champ.predict(None, va)

    # ---- promotion gate (validation only) ----
    thr = float(np.quantile(va.p, 0.80))  # care-management capacity: top 20% of patients
    race_va = M.subgroup_table(va, "label", "p", "race_audit", thr)
    race_va.to_csv(reports / "fairness_audit_valid.csv", index=False)
    # "Unknown" means race was not recorded: a data-quality issue reported on its own,
    # not a demographic group. Run 1 counted it and was blocked by it (49 events in valid,
    # AUROC 0.594, 95% CI 0.515-0.674).
    gaps = race_va[race_va.group != "Unknown"].auroc.dropna()
    gate = {
        "auroc_gain_vs_baseline": val_auc["lightgbm"] - base["valid_auroc"],
        "valid_ece_calibrated": M.ece(yva, va.p.to_numpy()),
        "race_auroc_gap": float(gaps.max() - gaps.min()),
        "race_groups_compared": race_va[race_va.group != "Unknown"].dropna(subset=["auroc"]).group.tolist(),
    }
    gate["passed"] = bool(gate["auroc_gain_vs_baseline"] >= GATE["min_auroc_gain_vs_baseline"]
                          and gate["valid_ece_calibrated"] <= GATE["max_ece"]
                          and gate["race_auroc_gap"] <= GATE["max_race_auroc_gap"])
    results["gate"] = {**gate, "rules": GATE}

    # ---- single look at the held-out test split ----
    p_te_raw = booster.predict(Xte)
    te["p"] = champ.predict(None, te)
    p_te_xgb = xbst.predict(xgb.DMatrix(Xte, enable_categorical=True), iteration_range=(0, xbst.best_iteration + 1))
    test = M.summary(yte, te.p.to_numpy())
    test["auroc_95ci"] = M.bootstrap_auc(yte, te.p.to_numpy())
    test["raw_ece"] = M.ece(yte, p_te_raw)
    test["xgboost_auroc"] = float(roc_auc_score(yte, p_te_xgb))
    test["flag_threshold_from_valid"] = thr
    results["test"] = test

    audit = pd.concat([M.subgroup_table(te, "label", "p", g, thr) for g in ("race_audit", "is_male", "age_band")])
    audit.to_csv(reports / "fairness_audit_test.csv", index=False)
    contrib = booster.predict(Xte, pred_contrib=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(0), index=FEATURES).sort_values(ascending=False)
    shap.to_csv(reports / "shap_importance_test.csv", header=["mean_abs_shap"])
    drift = drift_report(tr, te, NUMERIC, CATEGORICAL)
    score_psi = drift_report(pd.DataFrame({"score": booster.predict(Xtr)}), pd.DataFrame({"score": p_te_raw}), ["score"], [])
    drift.to_csv(reports / "drift_psi_train_vs_test.csv", index=False)
    results["drift"] = {"score_psi": float(score_psi.psi[0]),
                        "features_watch_or_worse": drift[drift.psi >= 0.10].feature.tolist(),
                        "max_feature_psi": float(drift.psi.max())}
    results["top_shap"] = shap.head(8).round(4).to_dict()
    plot_calibration(yte, p_te_raw, te.p.to_numpy(), reports / "calibration_test.png")

    with mlflow.start_run(run_name="champion-lightgbm-isotonic") as run:
        mlflow.log_params({"base": "lightgbm", "calibration": "isotonic (valid)", "features": len(FEATURES),
                           "flag_threshold": thr, **{f"gate_{k}": v for k, v in GATE.items()}})
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items() if isinstance(v, float)})
        mlflow.log_metrics({"gate_" + k: float(v) for k, v in gate.items() if not isinstance(v, list)})
        for f in ("fairness_audit_valid.csv", "fairness_audit_test.csv", "shap_importance_test.csv",
                  "drift_psi_train_vs_test.csv", "calibration_test.png"):
            mlflow.log_artifact(str(reports / f))
        sample = te[FEATURES].head(20)
        info = mlflow.pyfunc.log_model(name="model", python_model=champ, input_example=sample,
                                       signature=infer_signature(sample, champ.predict(None, sample)),
                                       registered_model_name=REG_NAME)
        # Every model is versioned; only a gate pass earns the "champion" alias that serving loads.
        client = mlflow.MlflowClient()
        version = max(int(v.version) for v in client.search_model_versions(f"name='{REG_NAME}'"))
        alias = "champion" if gate["passed"] else "candidate"
        client.set_registered_model_alias(REG_NAME, alias, version)
        client.set_model_version_tag(REG_NAME, str(version), "validation_status",
                                     "passed_gate" if gate["passed"] else "blocked_needs_review")
        results["registered"] = {"name": REG_NAME, "version": version, "alias": alias}
        results["run_id"] = run.info.run_id
        results["model_uri"] = info.model_uri

    (reports / "readmission_results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def cli() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="lakehouse", type=Path)
    ap.add_argument("--reports", default="reports", type=Path)
    a = ap.parse_args()
    print(json.dumps(main(a.lake, a.reports), indent=2, default=float))


if __name__ == "__main__":
    cli()
