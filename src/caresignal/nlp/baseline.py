"""TF-IDF + logistic-regression baselines for the RCT sentence task, scored on the same
seeded 5,000-sentence test sample as the transformer.

Two baselines: one trained on the same 8,000 sentences as the transformer (fair comparison)
and one trained on all ~180k training sentences (data-rich reference).
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from caresignal.nlp.data import LABELS, load_split, sample


def run(data="data/raw", train_n=8000, test_n=5000, seed=13) -> dict:
    full_x, full_y = load_split(Path(data) / "pubmed20k_train.txt")
    te_x, te_y = sample(*load_split(Path(data) / "pubmed20k_test.txt"), test_n, seed)
    out = {}
    mlflow.set_experiment("caresignal-clinical-nlp")
    for name, (x, y) in {"tfidf_lr_8k": sample(full_x, full_y, train_n, seed), "tfidf_lr_full": (full_x, full_y)}.items():
        with mlflow.start_run(run_name=name):
            pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                 LogisticRegression(max_iter=2000, C=4.0))
            pipe.fit(x, y)
            rep = classification_report(te_y, pipe.predict(te_x), target_names=LABELS, output_dict=True)
            mlflow.log_params({"n_train": len(x), "n_test": len(te_x), "features": "tfidf 1-2gram"})
            mlflow.log_metrics({"test_macro_f1": rep["macro avg"]["f1-score"], "test_accuracy": rep["accuracy"]})
            out[name] = {"n_train": len(x), "macro_f1": rep["macro avg"]["f1-score"], "accuracy": rep["accuracy"],
                         "per_class_f1": {k: rep[k]["f1-score"] for k in LABELS}}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/nlp_baselines.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
