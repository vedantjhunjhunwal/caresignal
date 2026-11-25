#!/usr/bin/env bash
# Download the two public datasets and the RoBERTa source wheel.
set -euo pipefail
mkdir -p data/raw models

# Diabetes 130-US Hospitals 1999-2008 (Strack et al., 2014; UCI ML Repository, CC BY 4.0).
# The UCI page is the canonical source; this run used a GitHub mirror of it
# (101,766 rows, 50 columns; md5 502f0cc004e4cf71536ec2e61596e42f).
curl -fsSL -o data/raw/diabetic_data.csv https://raw.githubusercontent.com/lugooe/Diabetes/main/diabetic_data.csv
curl -fsSL -o data/raw/IDs_mapping.csv https://raw.githubusercontent.com/lugooe/Diabetes/main/IDs_mapping.csv

# PubMed 20k RCT (Dernoncourt & Lee, 2017).
for s in train dev test; do
  curl -fsSL -o "data/raw/pubmed20k_${s}.txt" \
    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/${s}.txt"
done

# RoBERTa-base weights inside spaCy's en_core_web_trf 3.8.0 release (GitHub release asset).
curl -fsSL -o models/en_core_web_trf-3.8.0-py3-none-any.whl \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
python -m caresignal.nlp.convert_roberta --wheel models/en_core_web_trf-3.8.0-py3-none-any.whl --out models/roberta-base
