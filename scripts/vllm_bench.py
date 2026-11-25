"""Benchmark LLM serving: vLLM vs plain Hugging Face `generate`, same model, same prompts, on GPU.

Task: summarise the objective of randomized-trial abstracts from PubMed 20k RCT (test split).
Both engines use greedy decoding and the same max_new_tokens, so the comparison is throughput
(generated tokens per second) and per-request latency, not answer quality.

Results go to reports/vllm_bench.json and to MLflow.

    python scripts/vllm_bench.py --model Qwen/Qwen2.5-1.5B-Instruct --n 200
Needs a CUDA GPU (e.g. a Colab T4) and `pip install vllm`.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path


def abstracts(path: Path, n: int, seed: int = 13) -> list[str]:
    docs, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("###"):
            if cur:
                docs.append(" ".join(cur))
            cur = []
        elif "\t" in line:
            cur.append(line.split("\t", 1)[1])
    if cur:
        docs.append(" ".join(cur))
    return random.Random(seed).sample(docs, n)


def prompts(tok, docs: list[str]) -> list[str]:
    msgs = [[{"role": "user", "content": "In one sentence, state the objective of this clinical trial.\n\n" + d[:2500]}]
            for d in docs]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]


def run_hf(model_id: str, ps: list[str], max_new: int, bs: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    lat, gen_tokens = [], 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(ps), bs):
        enc = tok(ps[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        s = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        new = out[:, enc.input_ids.shape[1]:]
        gen_tokens += int((new != tok.eos_token_id).sum())
        lat += [time.perf_counter() - s] * len(new)  # batch latency, shared by its requests
    wall = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return {"engine": "hf-generate", "batch_size": bs, "wall_s": wall, "gen_tokens": gen_tokens,
            "tokens_per_s": gen_tokens / wall, "p50_latency_s": statistics.median(lat)}


def run_vllm(model_id: str, ps: list[str], max_new: int) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, dtype="float16", gpu_memory_utilization=0.85, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp)
    wall = time.perf_counter() - t0
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    lat = [o.metrics.finished_time - o.metrics.arrival_time for o in outs if getattr(o, "metrics", None)]
    return {"engine": "vllm", "wall_s": wall, "gen_tokens": gen_tokens, "tokens_per_s": gen_tokens / wall,
            "p50_latency_s": statistics.median(lat) if lat else None,
            "sample_output": outs[0].outputs[0].text.strip()}


def main() -> None:
    import mlflow
    import torch
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/raw/pubmed20k_test.txt", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--hf-batch", type=int, default=16)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    tok = AutoTokenizer.from_pretrained(a.model)
    ps = prompts(tok, abstracts(a.data, a.n))
    hf = run_hf(a.model, ps, a.max_new, a.hf_batch)
    vl = run_vllm(a.model, ps, a.max_new)
    res = {"model": a.model, "gpu": torch.cuda.get_device_name(0), "n_prompts": len(ps), "max_new_tokens": a.max_new,
           "hf": hf, "vllm": vl, "throughput_speedup": vl["tokens_per_s"] / hf["tokens_per_s"]}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/vllm_bench.json").write_text(json.dumps(res, indent=2))
    mlflow.set_experiment("caresignal-llm-serving")
    with mlflow.start_run(run_name="vllm-vs-hf"):
        mlflow.log_params({"model": a.model, "gpu": res["gpu"], "n": len(ps), "max_new": a.max_new})
        mlflow.log_metrics({"hf_tokens_per_s": hf["tokens_per_s"], "vllm_tokens_per_s": vl["tokens_per_s"],
                            "speedup": res["throughput_speedup"]})
        mlflow.log_artifact("reports/vllm_bench.json")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
