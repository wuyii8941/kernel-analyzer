"""Capture the first launch of every Triton kernel in one compiled training step (forward + backward) of a model.

    python capture_model.py MODEL_DIR OUT_DIR --config bf16|fp32tf32 [--layers 2] [--tokens 128] [--cap-mb 512]

bf16: weights in bf16, no autocast (pure bf16 step); fp32tf32: fp32 weights, TF32 allowed, Inductor generates the
matrix multiplications as Triton templates.  Input: the first --tokens tokens of the wikitext-103 validation text.
Launches whose tensor copies exceed --cap-mb are not saved (listed in manifest.json as skipped).
"""
import argparse, json, time, hashlib
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out")
ap.add_argument("--config", choices=("bf16", "fp32tf32"), required=True)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--cap-mb", type=float, default=512)
a = ap.parse_args()

import torch._inductor.config as ic
ic.use_static_cuda_launcher = False
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

TritonLaunchRecorder.install_hook()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cfg = AutoConfig.from_pretrained(a.model_dir)
text_cfg = getattr(cfg, "text_config", None) or cfg
KEEP_ALL_LAYERS = ("gemma4",)      # deviation D5: per-layer embedding tables are sized by the layer count
if cfg.model_type not in KEEP_ALL_LAYERS:
    for key in ("num_hidden_layers", "n_layer", "num_layers", "num_decoder_layers", "n_layers"):
        if isinstance(getattr(text_cfg, key, None), int):
            setattr(text_cfg, key, min(getattr(text_cfg, key), a.layers))
dtype = torch.bfloat16 if a.config == "bf16" else torch.float32
if a.config == "fp32tf32":
    torch.backends.cuda.matmul.allow_tf32 = True
    ic.max_autotune_gemm_backends = "TRITON"
kind = "causal"
CLASSES = [(AutoModelForCausalLM, "causal"), (AutoModelForMaskedLM, "masked"), (AutoModelForSeq2SeqLM, "seq2seq")]
try:   # multimodal checkpoints (Mistral3, Gemma4): the image-text model run on text only
    from transformers import AutoModelForImageTextToText
    CLASSES.append((AutoModelForImageTextToText, "image_text_on_text"))
except ImportError:
    pass
for cls, k in CLASSES:
    try:
        model = cls.from_pretrained(a.model_dir, config=cfg, dtype=dtype)
        kind = k
        break
    except (ValueError, KeyError):
        continue
else:
    raise SystemExit(f"no CausalLM / MaskedLM / Seq2SeqLM class for model type {cfg.model_type}")
model = model.cuda().train()
tok = AutoTokenizer.from_pretrained(a.model_dir)
text = Path(__file__).with_name("data").joinpath("wikitext103_validation.txt").read_text()
ids = tok(text[:20000], return_tensors="pt").input_ids[:, :a.tokens].cuda()


def step():
    if kind == "seq2seq":
        loss = model(input_ids=ids, labels=ids).loss
    else:
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()


compiled = torch.compile(model.forward, mode="max-autotune-no-cudagraphs" if a.config == "fp32tf32" else "default")
model.forward = compiled
t0 = time.time(); step(); model.zero_grad(set_to_none=True); t_compile = time.time() - t0
seen = set()
def select(name, index):
    if name in seen:
        return False
    seen.add(name)
    return True
with TritonLaunchRecorder(select=select) as R:
    step()
    torch.cuda.synchronize()
saved, skipped = [], []
for i, L in enumerate(R.launches):
    size = sum(t.numel() * t.element_size() for arg in L.args if arg.kind == "tensor"
               for t in (arg.before, arg.after) if t is not None) / 2**20
    name = f"{i:03d}_{L.kernel_name}"[:120]
    if size > a.cap_mb:
        skipped.append({"kernel": L.kernel_name, "mb": round(size, 1)})
        continue
    save_launch(L, out / name)
    saved.append({"dir": name, "kernel": L.kernel_name, "mb": round(size, 1)})
manifest = {"model_dir": a.model_dir, "config": a.config,
            "layers": "all" if cfg.model_type in KEEP_ALL_LAYERS else a.layers, "tokens": int(ids.shape[1]),
            "kind": kind, "launches_in_step": R.launch_count, "distinct_kernels": len(R.launches),
            "saved": saved, "skipped": skipped, "compile_s": round(t_compile, 1),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "triton": __import__("triton").__version__, "ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({k: v for k, v in manifest.items() if k not in ("saved", "skipped")}), len(saved), "saved", len(skipped), "skipped")
