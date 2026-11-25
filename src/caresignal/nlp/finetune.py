"""Fine-tune RoBERTa-base to label sentences of randomized-controlled-trial abstracts
(BACKGROUND / OBJECTIVE / METHODS / RESULTS / CONCLUSIONS) on PubMed 20k RCT.

Runs on CPU or GPU: on CUDA it switches on fp16 autocast and a larger batch.
Everything (params, per-epoch dev metrics, test report, confusion matrix) goes to MLflow,
and the fine-tuned model is logged to the MLflow Model Registry.

    python -m caresignal.nlp.finetune --model models/roberta-base --train-n 8000 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from caresignal.nlp.data import LABELS, load_split, sample

REG_NAME = "caresignal-rct-sentence-roberta"


def batches(texts, labels, tok, bs, max_len, shuffle, seed):
    idx = list(range(len(texts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield enc, torch.tensor([labels[k] for k in j])


@torch.no_grad()
def predict(model, tok, texts, device, bs=64, max_len=64, amp=False):
    model.eval()
    out = []
    for enc, _ in batches(texts, [0] * len(texts), tok, bs, max_len, False, 0):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(model(**enc).logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(out)


def register(model, tok) -> None:
    """Log the fine-tuned model to the active run and the Model Registry.
    Requirements are pinned explicitly so MLflow does not probe for optional packages."""
    import tokenizers
    import transformers

    reqs = [f"torch=={torch.__version__.split('+')[0]}", f"transformers=={transformers.__version__}",
            f"tokenizers=={tokenizers.__version__}"]
    mlflow.transformers.log_model(transformers_model={"model": model, "tokenizer": tok}, task="text-classification",
                                  name="model", registered_model_name=REG_NAME, pip_requirements=reqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--dev-n", type=int, default=1000)
    ap.add_argument("--test-n", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="models/rct-roberta")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    if device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))

    tr_x, tr_y = sample(*load_split(Path(a.data) / "pubmed20k_train.txt"), a.train_n, a.seed)
    dv_x, dv_y = sample(*load_split(Path(a.data) / "pubmed20k_dev.txt"), a.dev_n, a.seed)
    te_x, te_y = sample(*load_split(Path(a.data) / "pubmed20k_test.txt"), a.test_n, a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={lab: i for i, lab in enumerate(LABELS)}
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr_x) + a.bs - 1) // a.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    mlflow.set_experiment("caresignal-clinical-nlp")
    with mlflow.start_run(run_name="roberta-base-finetune") as run:
        mlflow.log_params({**vars(a), "device": device, "fp16": amp, "base_model": "roberta-base (spaCy en_core_web_trf 3.8.0)",
                           "n_train": len(tr_x), "n_dev": len(dv_x), "n_test": len(te_x)})
        best, step, t0 = -1.0, 0, time.time()
        for ep in range(a.epochs):
            model.train()
            for enc, y in batches(tr_x, tr_y, tok, a.bs, a.max_len, True, a.seed + ep):
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = model(**enc, labels=y.to(device)).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                if step % 50 == 0:
                    mlflow.log_metric("train_loss", float(loss), step=step)
                    print(f"step {step}/{steps} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
            dev_f1 = f1_score(dv_y, predict(model, tok, dv_x, device, max_len=a.max_len, amp=amp), average="macro")
            mlflow.log_metric("dev_macro_f1", dev_f1, step=ep + 1)
            print(f"epoch {ep + 1} dev macro-F1 {dev_f1:.4f}", flush=True)
            if dev_f1 > best:  # keep the best epoch on dev, never on test
                best = dev_f1
                model.save_pretrained(a.out)
                tok.save_pretrained(a.out)

        model = AutoModelForSequenceClassification.from_pretrained(a.out).to(device)
        pred = predict(model, tok, te_x, device, max_len=a.max_len, amp=amp)
        rep = classification_report(te_y, pred, target_names=LABELS, output_dict=True, digits=4)
        cm = confusion_matrix(te_y, pred).tolist()
        mlflow.log_metrics({"test_macro_f1": rep["macro avg"]["f1-score"], "test_accuracy": rep["accuracy"],
                            "train_minutes": (time.time() - t0) / 60})
        res = {"model": "roberta-base fine-tuned", "best_dev_macro_f1": best, "test": rep, "confusion": cm,
               "labels": LABELS, "minutes": round((time.time() - t0) / 60, 1), "device": device}
        Path("reports").mkdir(exist_ok=True)
        Path("reports/nlp_roberta.json").write_text(json.dumps(res, indent=2))
        mlflow.log_artifact("reports/nlp_roberta.json")
        register(model, tok)
        print(json.dumps({"run_id": run.info.run_id, "test_macro_f1": rep["macro avg"]["f1-score"],
                          "test_acc": rep["accuracy"], "best_dev": best}, indent=1))


if __name__ == "__main__":
    main()
