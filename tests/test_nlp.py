import torch

from caresignal.nlp.convert_roberta import HIDDEN, LAYERS, to_hf_state
from caresignal.nlp.data import LABELS, load_split, sample


def test_load_split_parses_abstract_format(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("###123\nOBJECTIVE\tTo test.\nMETHODS\tWe randomized 40 adults .\n\n###456\nRESULTS\tIt worked .\n")
    texts, labels = load_split(f)
    assert texts == ["To test.", "We randomized 40 adults .", "It worked ."]
    assert [LABELS[i] for i in labels] == ["OBJECTIVE", "METHODS", "RESULTS"]


def test_sample_is_seeded_and_bounded():
    t, y = [str(i) for i in range(100)], list(range(100))
    assert sample(t, y, 10, 1) == sample(t, y, 10, 1)
    assert sample(t, y, 1000, 1) == (t, y)


def test_qkv_split_maps_fused_projection():
    sd = {}
    e = "curated_encoder.embeddings.inner."
    for n, shp in [("word_embeddings.weight", (10, HIDDEN)), ("position_embeddings.weight", (12, HIDDEN)),
                   ("token_type_embeddings.weight", (1, HIDDEN)), ("layer_norm.weight", (HIDDEN,)),
                   ("layer_norm.bias", (HIDDEN,))]:
        sd[e + n] = torch.zeros(shp)
    for i in range(LAYERS):
        p = f"curated_encoder.layers.{i}."
        sd[p + "mha.input.weight"] = torch.cat([torch.full((HIDDEN, HIDDEN), float(k)) for k in (1, 2, 3)])
        sd[p + "mha.input.bias"] = torch.cat([torch.full((HIDDEN,), float(k)) for k in (1, 2, 3)])
        for n, shp in [("mha.output", (HIDDEN, HIDDEN)), ("attn_output_layernorm", (HIDDEN,)),
                       ("ffn.intermediate", (3072, HIDDEN)), ("ffn.output", (HIDDEN, 3072)),
                       ("ffn_output_layernorm", (HIDDEN,))]:
            sd[f"{p}{n}.weight"] = torch.zeros(shp)
            sd[f"{p}{n}.bias"] = torch.zeros(shp[0])
    hf = to_hf_state(sd)
    for name, val in (("query", 1.0), ("key", 2.0), ("value", 3.0)):
        assert (hf[f"encoder.layer.0.attention.self.{name}.weight"] == val).all()
        assert hf[f"encoder.layer.11.attention.self.{name}.bias"].shape == (HIDDEN,)
    assert len(hf) == 5 + LAYERS * 16

