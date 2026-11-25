"""PubMed 20k RCT loader (Dernoncourt & Lee, 2017): one labelled sentence per line,
abstracts separated by '###<PMID>' headers."""
from __future__ import annotations

import random
from pathlib import Path

LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
URL = "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/{split}.txt"


def load_split(path: str | Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("###"):
            continue
        lab, text = line.split("\t", 1)
        texts.append(text)
        labels.append(LABELS.index(lab))
    return texts, labels


def sample(texts: list[str], labels: list[int], n: int | None, seed: int) -> tuple[list[str], list[int]]:
    """Seeded random subset (n=None or n>=len keeps everything)."""
    if n is None or n >= len(texts):
        return texts, labels
    idx = random.Random(seed).sample(range(len(texts)), n)
    return [texts[i] for i in idx], [labels[i] for i in idx]
