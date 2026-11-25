"""Patient utilisation segments with Spark MLlib K-means.

Clusters training encounters on standardised utilisation features (prior visits, length of stay,
medications, labs, diagnoses), picks k by silhouette on a held-out sample, then reports each
segment's size and 30-day readmission rate on the test split. Segments are descriptive (for
care-management targeting), not a model input. Logged to MLflow.

    python -m caresignal.segment
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F

from caresignal.spark import get_spark

FEATS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
         "number_emergency", "number_inpatient", "number_diagnoses", "n_drugs_on"]


def main(lake: Path = Path("lakehouse"), reports: Path = Path("reports")) -> dict:
    spark = get_spark()
    g = spark.read.parquet(str(lake / "gold/readmission_features"))
    # log1p tames the heavy right tail of visit counts before scaling
    for c in FEATS:
        g = g.withColumn(f"{c}_l", F.log1p(F.col(c).cast("double")))
    tr = g.filter("split = 'train'").cache()
    fit_df, sel_df = tr.randomSplit([0.8, 0.2], seed=7)
    vec = VectorAssembler(inputCols=[f"{c}_l" for c in FEATS], outputCol="raw")
    sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    ev = ClusteringEvaluator(featuresCol="features", metricName="silhouette")

    mlflow.set_experiment("caresignal-segmentation")
    scores, models = {}, {}
    for k in range(3, 8):
        with mlflow.start_run(run_name=f"kmeans-k{k}"):
            m = Pipeline(stages=[vec, sc, KMeans(k=k, seed=7, featuresCol="features", maxIter=50)]).fit(fit_df)
            s = ev.evaluate(m.transform(sel_df))
            mlflow.log_params({"k": k, "features": ",".join(FEATS)})
            mlflow.log_metric("silhouette_holdout", s)
            scores[k], models[k] = s, m
    best = max(scores, key=scores.get)
    te = models[best].transform(g.filter("split = 'test'"))
    prof = (te.groupBy("prediction").agg(
        F.count("*").alias("n"), F.round(F.avg("label"), 4).alias("readmit_30_rate"),
        *[F.round(F.avg(c), 2).alias(c) for c in ("number_inpatient", "number_emergency", "time_in_hospital",
                                                  "num_medications", "number_diagnoses")])
        .orderBy(F.desc("readmit_30_rate")).toPandas())
    base = te.agg(F.avg("label")).first()[0]
    prof["lift"] = (prof.readmit_30_rate / base).round(2)
    reports.mkdir(exist_ok=True)
    prof.to_csv(reports / "segments_test.csv", index=False)
    out = {"silhouette_by_k": {k: round(v, 4) for k, v in scores.items()}, "chosen_k": best,
           "test_readmit_rate": round(base, 4),
           "highest_risk_segment": prof.iloc[0][["n", "readmit_30_rate", "lift", "number_inpatient"]].to_dict(),
           "lowest_risk_segment": prof.iloc[-1][["n", "readmit_30_rate", "lift", "number_inpatient"]].to_dict()}
    with mlflow.start_run(run_name=f"kmeans-chosen-k{best}"):
        mlflow.log_params({"k": best})
        mlflow.log_metrics({"silhouette_holdout": scores[best], "top_segment_lift": float(prof.lift.iloc[0])})
        mlflow.log_artifact(str(reports / "segments_test.csv"))
    (reports / "segmentation.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
