# Databricks notebook source
# MAGIC %md
# MAGIC # CareSignal: readmission pipeline on Databricks
# MAGIC Runs the same package as the local pipeline. On a cluster, `get_spark()` reuses the
# MAGIC notebook's SparkSession and MLflow logs to the workspace tracking server.

# COMMAND ----------
# MAGIC %pip install /Workspace/Shared/caresignal/dist/caresignal-1.0.0-py3-none-any.whl

# COMMAND ----------
from pathlib import Path

dbutils.widgets.text("raw", "/Volumes/main/caresignal/raw")  # noqa: F821  (Databricks global)
dbutils.widgets.text("lake", "/Volumes/main/caresignal/lake")  # noqa: F821
raw, lake = Path(dbutils.widgets.get("raw")), Path(dbutils.widgets.get("lake"))  # noqa: F821

# COMMAND ----------
from caresignal import pipeline

report = pipeline.run(raw, lake, salt=dbutils.secrets.get("caresignal", "deid_salt"))  # noqa: F821
display(report)  # noqa: F821

# COMMAND ----------
from caresignal import train

results = train.main(lake, Path("/tmp/caresignal_reports"))
results["test"]["auroc"], results["gate"]
