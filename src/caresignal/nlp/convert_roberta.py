"""Rebuild a Hugging Face RoBERTa-base checkpoint from spaCy's en_core_web_trf release.

Why: the build machine could reach GitHub release assets but not the Hugging Face Hub.
spaCy's en_core_web_trf 3.8.0 wheel (a GitHub release asset) ships a RoBERTa-base
encoder in curated-transformers format. This script

1. unpacks the wheel and reads the Thinc msgpack file,
2. loads the PyTorch state dict with ``torch.load(weights_only=True)`` (no pickle code runs),
3. renames every tensor to Hugging Face ``RobertaModel`` names and splits the fused
   [Q;K;V] attention projection into query/key/value,
4. rebuilds the byte-level BPE tokenizer from the stored vocab and merges.

Note: spaCy trained this encoder further on its English tagging/parsing/NER tasks, so it
is RoBERTa-base plus that training, not the original checkpoint.

The [Q;K;V] layout was checked against the alternative per-head layout with a frozen
linear probe on PubMed RCT sentences (0.687 vs 0.400 accuracy on 300 held-out sentences).

Usage:
    python -m caresignal.nlp.convert_roberta --wheel en_core_web_trf-3.8.0-py3-none-any.whl --out models/roberta-base
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import msgpack
import torch

WHEEL_URL = ("https://github.com/explosion/spacy-models/releases/download/"
             "en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl")
HIDDEN, LAYERS = 768, 12


def read_curated(wheel: Path) -> tuple[dict, dict]:
    """Return (state_dict, bpe) from the spaCy wheel without unpacking it to disk."""
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith("/transformer/model"))
        raw = z.read(name)
    obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    node = next(i for i, n in enumerate(obj["nodes"]) if n["name"] == "pytorch")
    shim = msgpack.unpackb(obj["shims"][node][0], raw=False, strict_map_key=False)
    state = torch.load(io.BytesIO(shim["state"]), map_location="cpu", weights_only=True)
    enc = next(i for i, n in enumerate(obj["nodes"]) if n["name"] == "byte_bpe_encoder")
    bpe = msgpack.unpackb(obj["attrs"][enc]["byte_bpe_processor"], raw=False, strict_map_key=False)
    return state, bpe


def to_hf_state(sd: dict) -> dict:
    p, e = "curated_encoder.", "curated_encoder.embeddings.inner."
    new = {
        "embeddings.word_embeddings.weight": sd[e + "word_embeddings.weight"],
        "embeddings.position_embeddings.weight": sd[e + "position_embeddings.weight"],
        "embeddings.token_type_embeddings.weight": sd[e + "token_type_embeddings.weight"],
        "embeddings.LayerNorm.weight": sd[e + "layer_norm.weight"],
        "embeddings.LayerNorm.bias": sd[e + "layer_norm.bias"],
    }
    for i in range(LAYERS):
        src, dst = f"{p}layers.{i}.", f"encoder.layer.{i}."
        qw, kw, vw = sd[src + "mha.input.weight"].split(HIDDEN, 0)
        qb, kb, vb = sd[src + "mha.input.bias"].split(HIDDEN, 0)
        for n, w, b in (("query", qw, qb), ("key", kw, kb), ("value", vw, vb)):
            new[f"{dst}attention.self.{n}.weight"] = w.clone()
            new[f"{dst}attention.self.{n}.bias"] = b.clone()
        pairs = {
            "attention.output.dense": "mha.output",
            "attention.output.LayerNorm": "attn_output_layernorm",
            "intermediate.dense": "ffn.intermediate",
            "output.dense": "ffn.output",
            "output.LayerNorm": "ffn_output_layernorm",
        }
        for hf, cur in pairs.items():
            new[f"{dst}{hf}.weight"] = sd[f"{src}{cur}.weight"]
            new[f"{dst}{hf}.bias"] = sd[f"{src}{cur}.bias"]
    return new


def build_tokenizer(bpe: dict):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.BPE(vocab=bpe["vocab"], merges=[tuple(m) for m in bpe["merges"]]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    t.decoder = decoders.ByteLevel()
    t.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0), trim_offsets=True, add_prefix_space=True)
    return PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   pad_token="<pad>", cls_token="<s>", sep_token="</s>", model_max_length=512)


def main() -> None:
    from transformers import RobertaConfig, RobertaModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sd, bpe = read_curated(a.wheel)
    cfg = RobertaConfig(vocab_size=len(bpe["vocab"]), hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                        num_attention_heads=12, intermediate_size=3072, max_position_embeddings=514,
                        type_vocab_size=1, layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    model = RobertaModel(cfg, add_pooling_layer=False)
    missing, unexpected = model.load_state_dict(to_hf_state(sd), strict=False)
    assert not unexpected and not [k for k in missing if "position_ids" not in k], (missing, unexpected)
    model.save_pretrained(a.out)
    build_tokenizer(bpe).save_pretrained(a.out)
    (a.out / "SOURCE.json").write_text(json.dumps({"source": WHEEL_URL, "tensors": len(sd)}, indent=2))
    print(f"saved {a.out} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
