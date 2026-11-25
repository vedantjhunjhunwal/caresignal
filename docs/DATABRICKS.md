# Running CareSignal on Databricks Free Edition

Free Edition (databricks.com/learn/free-edition) gives a serverless workspace with Unity Catalog,
MLflow and Git folders at no cost. About 30 minutes end to end.

1. **Sign up** for Free Edition and open the workspace.
2. **Git folder:** Workspace → your home → Create → Git folder →
   `https://github.com/vedantjhunjhunwal/caresignal`.
3. **Volume for the data:** Catalog → `workspace` → `default` → Create → Volume, name `caresignal`.
   Inside it create a folder `raw` and upload `diabetic_data.csv` and `IDs_mapping.csv`
   (from `scripts/get_data.sh`, or the UCI page).
4. **Run** `databricks/notebooks/01_readmission_pipeline.py` on serverless compute, top to bottom.
   - The pipeline writes bronze/silver/gold Parquet under `/Volumes/workspace/default/caresignal/lake`.
   - MLflow runs appear under `/Users/<you>/caresignal-readmission`, and the champion model is registered
     in Unity Catalog as `workspace.default.caresignal_readmission` with alias `champion`.
5. **Keep evidence:** the reports folder in the volume, a screenshot of the notebook/job run, and the
   model page in Catalog Explorer.
6. **Optional, as a job:** Workflows → Create job → task type Notebook → this notebook → serverless.
   `databricks.yml` holds the same job as an Asset Bundle for a classic (non-Free) workspace.

If the Spark MLlib baseline fails on serverless (Spark Connect ML support depends on the runtime
version), note the error; the rest of the pipeline and LightGBM run as plain Python on the driver.
