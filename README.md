# CareSignal

Healthcare data science on public data, end to end:

1. **Readmission risk on PySpark + MLflow.** A bronze/silver/gold lakehouse pipeline on 101,766 real
   inpatient encounters from 130 US hospitals, a LightGBM model of 30-day readmission that beats a Spark
   MLlib baseline, isotonic calibration, a subgroup fairness audit, PSI drift monitoring, SHAP, an MLflow
   model-registry promotion gate, and a FastAPI scoring service.
2. **Clinical text transformer fine-tuning.** RoBERTa-base fine-tuned with Hugging Face Transformers and
   PyTorch to label sentences of randomized-trial abstracts (PubMed 20k RCT), against TF-IDF baselines,
   tracked and registered in MLflow.

Everything below was produced by the code in this repo; numbers come from `reports/`.

## Results

### 30-day readmission (held-out later patients, evaluated once)

| model | valid AUROC | test AUROC |
|---|---|---|
| Spark MLlib logistic regression (baseline) | 0.643 | 0.630 |
| XGBoost (challenger) | 0.666 | 0.646 |
| **LightGBM + isotonic calibration (registered `champion`)** | **0.666** | **0.646** (95% CI 0.627–0.663) |

AUPRC 0.150 at 8.8% prevalence · Brier 0.078 · calibration error (ECE) 0.014 → 0.011 after isotonic ·
the top 10% of patients by risk contain 21.7% of all readmissions (lift 2.2×).
Readmission from administrative data is hard: published models on this dataset report AUROC in the mid-0.60s.

Promotion gate (validation only), its first-run block and the subgroup audit are in **[MODEL_CARD.md](MODEL_CARD.md)**.

### Excel monitoring dashboard

`reports/caresignal_dashboard.xlsx` (built by `scripts/build_dashboard.py` from the registered champion model):
KPI summary, an outreach-capacity what-if (edit the yellow cell), calibration by risk decile, fairness by group,
PSI drift and K-means segments. 109 Excel formulas on raw counts, 5 native charts.

### Patient segments (Spark MLlib K-means)

K-means on 9 log-scaled, standardised utilisation features; k chosen by silhouette on a held-out 20% of
training (k=4, silhouette 0.257). On the test split the high-emergency-use segment (1,081 encounters,
0.91 ER visits/yr) is readmitted at 11.6% vs 7.2% for the lowest segment (`reports/segments_test.csv`).
Segments describe patients for outreach planning; they are not a model input.

### Clinical NLP: RCT abstract sentence labels (5 classes, 5,000 test sentences)

| model | train sentences | macro-F1 | accuracy | Background | Objective | Methods | Results | Conclusions |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 8,000 | 0.688 | 0.764 | 0.55 | 0.53 | 0.85 | 0.84 | 0.67 |
| TF-IDF + logistic regression | 180,040 | 0.759 | 0.825 | 0.64 | 0.60 | 0.91 | 0.89 | 0.75 |
| **RoBERTa-base fine-tuned** | 8,000 | 0.789 | 0.846 | 0.70 | 0.62 | 0.92 | 0.90 | 0.80 |

Fine-tuned on 8,000 sentences for 2 epochs (best epoch picked on 1,000 dev sentences: dev macro-F1 0.786), CPU, 66 minutes. Same 8k data: +0.102 macro-F1 over TF-IDF; it also beats TF-IDF trained on 22× more sentences. Sentences are labelled without their neighbours, so scores sit below published sequence models that use the whole abstract. Registered in MLflow as `caresignal-rct-sentence-roberta`.

## Pipeline

```
data/raw/diabetic_data.csv
  └─ bronze   raw strings + source file, load time, row hash                      101,766 rows
     └─ silver '?'→null, typed, 14 data-quality rules → quarantine table (rule names) 3 quarantined
               cohort: drop discharges to death/hospice                             2,423 excluded
               patient/encounter IDs → salted SHA-256 keys (raw IDs dropped)
        └─ gold one row per encounter, 27 features + label, ICD-9 groups,          99,340 rows
                prior-encounter count via window (no look-ahead),
                patient-disjoint split ordered by first encounter (0 patients in 2 splits)
```

- `src/caresignal/pipeline.py`: PySpark bronze/silver/gold, data-quality rules, de-identification
- `src/caresignal/train.py`: Spark MLlib baseline, LightGBM/XGBoost, calibration, gate, MLflow registry
- `src/caresignal/metrics.py`, `monitor.py`: calibration, bootstrap CIs, subgroup audit, PSI drift
- `src/caresignal/segment.py`: Spark MLlib K-means patient segmentation
- `src/caresignal/serve.py`: FastAPI service loading `models:/caresignal-readmission@champion`
- `src/caresignal/nlp/`: RoBERTa weight conversion, fine-tuning, TF-IDF baselines
- `databricks.yml`, `databricks/notebooks/`: the same entry points packaged as a Databricks Asset Bundle job
- `tests/`: 37 pytest tests (pipeline on a fixture with planted bad rows, Spark vs Python ICD-9 parity,
  no raw IDs in gold, patient-disjoint splits, metrics, drift, model, API schema, weight conversion)

## Run it

```bash
pip install -e ".[dev]"
./scripts/get_data.sh                         # datasets + RoBERTa weights
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db CARESIGNAL_SALT=<secret>
python -m caresignal.pipeline                 # bronze/silver/gold → lakehouse/
python -m caresignal.train                    # models, gate, registry, reports/
python -m caresignal.segment                  # K-means patient segments
python -m caresignal.nlp.baseline
python -m caresignal.nlp.finetune --model models/roberta-base --train-n 8000 --epochs 2
uvicorn caresignal.serve:app                  # POST /score
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q
```

`finetune.py` uses CUDA with fp16 autocast when a GPU is present; the reported run used CPU (2 cores).

- **GPU + LLM serving:** `notebooks/colab_gpu_finetune_vllm.ipynb` reruns fine-tuning on a Colab T4 and
  benchmarks vLLM against Hugging Face `generate` (`scripts/vllm_bench.py`).
- **Databricks:** `docs/DATABRICKS.md` runs the pipeline, training, Unity Catalog registry and segments on
  Databricks Free Edition (serverless).

## Notes on choices

- **Patient-disjoint, time-ordered split.** All of a patient's encounters sit in one split, assigned by
  the order of their first encounter, so the test set is later patients the model never saw.
- **Race is audited, not used.** It stays in gold as `race_audit` and is excluded from `FEATURES`.
- **Gate history.** Run 1 counted "Unknown" race as a group and was blocked (AUROC gap 0.083). "Unknown" is
  missing data, not a demographic group, so the gate compares recorded groups with >= 30 readmissions and
  reports Unknown separately. Run 1's output is kept in `reports/run1_*`.
- **Where the RoBERTa weights come from.** The build machine could not reach the Hugging Face Hub, so
  `convert_roberta.py` rebuilds a Hugging Face RoBERTa-base checkpoint from the encoder inside spaCy's
  `en_core_web_trf` 3.8.0 release (loaded with `torch.load(weights_only=True)`). spaCy trained that encoder
  further on English tagging/parsing/NER, so it is RoBERTa-base plus that training. The fused attention
  layout was confirmed with a frozen linear probe (0.687 vs 0.400 for the alternative layout).
- **Databricks.** `databricks.yml` defines the job; it has not been deployed to a workspace from this repo.

## Data
- Diabetes 130-US Hospitals 1999–2008, Strack et al., *BioMed Research International* (2014), UCI ML
  Repository, doi:10.24432/C5230J (CC BY 4.0). De-identified by its publishers; this repo hashes IDs again.
- PubMed 200k RCT (20k subset), Dernoncourt & Lee, IJCNLP 2017.

Not for clinical use.
