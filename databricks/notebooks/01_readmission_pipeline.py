# Databricks notebook source
# MAGIC %md
# MAGIC # CareSignal on Databricks (Free Edition, serverless)
# MAGIC Runs the same package as the local pipeline: bronze/silver/gold on a Unity Catalog volume,
# MAGIC Spark MLlib baseline + LightGBM, MLflow tracking and Unity Catalog model registry, K-means segments.
# MAGIC Steps to set this up are in `docs/DATABRICKS.md`.

# COMMAND ----------
# MAGIC %pip install -e ../.. --quiet

# COMMAND ----------
dbutils.library.restartPython()  # noqa: F821  (Databricks global)

# COMMAND ----------
import os
from pathlib import Path

user = spark.sql("select current_user()").first()[0]  # noqa: F821
VOL = "/Volumes/workspace/default/caresignal"
os.environ["CARESIGNAL_EXPERIMENT_DIR"] = f"/Users/{user}/"
os.environ["CARESIGNAL_MODEL_NAME"] = "workspace.default.caresignal_readmission"  # Unity Catalog name
os.environ.setdefault("CARESIGNAL_SALT", "set-your-own-secret")  # de-identification salt
import mlflow

mlflow.set_registry_uri("databricks-uc")
raw, lake, reports = Path(f"{VOL}/raw"), Path(f"{VOL}/lake"), Path(f"{VOL}/reports")

# COMMAND ----------
import time

from caresignal import pipeline

t = time.time()
dq = pipeline.run(raw, lake)
print(f"pipeline {time.time() - t:.0f}s", dq)

# COMMAND ----------
from caresignal import train

t = time.time()
res = train.main(lake, reports)
print(f"train {time.time() - t:.0f}s", res["test"]["auroc"], res["gate"]["passed"], res.get("registered"))

# COMMAND ----------
from caresignal import segment

print(segment.main(lake, reports))

# COMMAND ----------
# MAGIC %md
# MAGIC Save `reports/` (Catalog → Volumes → caresignal/reports) and a screenshot of the job run and the
# MAGIC registered model page; those are the evidence for the CV line.
