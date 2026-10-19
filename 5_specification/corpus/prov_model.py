"""Exact provenance of the Inductor kernels of one three_way model-config (PLAN_v3 V3, binding).

Recompiles the same training step as 1_experiments/three_way/capture_model.py (same model loading, layer cut,
tokens, config and compile mode) with a hook on TritonScheduling.define_kernel that records, for every generated
Triton kernel: its name, the kernel file path Inductor derives from the source hash (the path that the captured
TTIR names in its #loc line), and for each kernel parameter the buffer it receives and that buffer's origin FX nodes
(post-grad names, the names used in the graph fragments). With --capture-dir, also records the first launch of
each kernel in a second training step in the same process, after resetting gradients. All output paths must be new.

    python prov_model.py MODEL_DIR OUT.jsonl --config bf16|fp32tf32
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out")
ap.add_argument("--config", choices=("bf16", "fp32tf32"), required=True)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--capture-dir", type=Path, help="new directory for actual device captures from this compilation")
ap.add_argument("--cap-mb", type=float, default=512, help="maximum saved operand-copy size per launch")
a = ap.parse_args()
if (Path(a.out).exists() or Path(a.out).with_suffix(".graphs").exists()
        or (a.capture_dir is not None and a.capture_dir.exists())):
    ap.error("output paths must be new; existing records are never overwritten")
if a.layers <= 0 or a.tokens <= 0 or not math.isfinite(a.cap_mb) or a.cap_mb <= 0:
    ap.error("layers, tokens and cap-mb must be positive and finite")
if a.capture_dir is not None:
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch
    TritonLaunchRecorder.install_hook()
    a.capture_dir.mkdir(parents=True, exist_ok=False)

import torch._inductor.config as ic
from torch._inductor.codegen import triton as tri
from torch._inductor.codecache import code_hash, get_path
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from graph_provenance import snapshot_graph

ic.use_static_cuda_launcher = False
ic.fx_graph_cache = False                    # force code generation so that the hook runs; autotune caches are kept
records = []
record_owners = []
_orig = tri.TritonScheduling.define_kernel
_orig_compile = GraphLowering._compile_to_module


def _names(nodes):
    return sorted(getattr(n, "name", str(n)) for n in nodes)


def _origins(buf_name):
    """origins: every FX node fused into the buffer; origin_node: the one node whose value the buffer holds, as set by
    GraphLowering.run_node (assign_origin_node), e.g. the getitem behind a MultiOutput; None when Inductor set none."""
    g = V.graph
    if buf_name in g.graph_inputs:
        return {"kind": "graph_input", "origins": [buf_name], "origin_node": buf_name}
    b = g.name_to_buffer.get(buf_name)
    if b is None:
        return {"kind": "unknown", "origins": [], "origin_node": None}
    node = getattr(b, "origin_node", None)
    out = {"kind": type(b).__name__, "origin_node": getattr(node, "name", None)}
    try:
        out["origins"] = _names(b.get_origins())
    except Exception as e:  # noqa: BLE001
        out.update(origins=[], error=str(e)[:80])
    return out


def define_kernel(self, src_code, node_schedule, kernel):
    name = _orig(self, src_code, node_schedule, kernel)
    try:
        args = kernel.args
        params = {}
        for buf, arg in args.input_buffers.items():
            params[arg] = {"role": "input", "buffer": buf, **_origins(buf)}
        for buf, arg in args.output_buffers.items():
            if isinstance(arg, str) and not arg.startswith("REMOVED"):
                params[arg] = {"role": "output", "buffer": buf, **_origins(buf)}
        for buf, ib in args.inplace_buffers.items():
            if hasattr(ib, "inner_name") and not str(ib.inner_name).startswith("REMOVED"):
                others = list(getattr(ib, "other_names", []))
                # an in-place buffer reuses the storage of the buffers in other_names: the pre-state is their value
                params[ib.inner_name] = {"role": "inplace", "buffer": buf, **_origins(buf), "other_names": others,
                                         "other_origin_nodes": {o: _origins(o)["origin_node"] for o in others}}
        final_src = src_code.replace(str(tri.Placeholder.DESCRIPTIVE_NAME), name).replace(
            str(tri.Placeholder.KERNEL_NAME), name if ic.triton.unique_kernel_names else "triton_").replace("#pragma CMT", "#")
        _, _, path = get_path(code_hash(final_src.strip()), "py")
        records.append({"kernel": name, "kernel_path": path, "params": params,
                        "nodes": [n.get_name() for n in node_schedule if hasattr(n, "get_name")]})
    except Exception as e:  # noqa: BLE001 -- provenance is best effort; a failure is recorded, never hidden
        records.append({"kernel": name, "error": f"{type(e).__name__}: {e}"[:200]})
    record_owners.append((V.graph, records[-1]))
    return name


def compile_to_module(graph, *args, **kwargs):
    # Save before code generation mutates or frees compiler state. The full graph
    # retains tuple selections and dependencies omitted from per-kernel fragments.
    try:
        task_graph = snapshot_graph(graph.module)
    except Exception as exc:
        task_graph = {"error": f"{type(exc).__name__}: {exc}", "semantic_verdict": "not_established"}
    module = _orig_compile(graph, *args, **kwargs)
    # The same GraphLowering object owns kernel generation and the returned
    # module. Do not infer ownership from kernel names or repeated graph text.
    owned = [record for owner, record in record_owners if owner is graph and "output_code" not in record]
    if owned:
        graph_dir = Path(a.out).with_suffix(".graphs")
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / ("graph_%03d.json" % len(list(graph_dir.glob("graph_*.json"))))
        task_graph["output_code"] = str(Path(module.__file__).resolve())
        with graph_path.open("x") as f:
            json.dump(task_graph, f, indent=1)
    for owner, record in record_owners:
        if owner is graph and "output_code" not in record:
            record["output_code"] = str(Path(module.__file__).resolve())
            record["task_graph"] = str(graph_path.resolve())
    return module


tri.TritonScheduling.define_kernel = define_kernel
GraphLowering._compile_to_module = compile_to_module

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

cfg = AutoConfig.from_pretrained(a.model_dir)
text_cfg = getattr(cfg, "text_config", None) or cfg
KEEP_ALL_LAYERS = ("gemma4",)
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
try:
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
    raise SystemExit(f"no model class for {cfg.model_type}")
model = model.cuda().train()
tok = AutoTokenizer.from_pretrained(a.model_dir)
text = (Path(__file__).resolve().parents[2] / "1_experiments/three_way/data/wikitext103_validation.txt")
if not text.exists():
    text = Path("/data1/tzh/kernel-analyzer/.cache/three_way/data/wikitext103_validation.txt")
ids = tok(text.read_text()[:20000], return_tensors="pt").input_ids[:, :a.tokens].cuda()
model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs" if a.config == "fp32tf32" else "default")
loss = model(input_ids=ids, labels=ids).loss
loss.backward()
torch.cuda.synchronize()
if a.capture_dir is not None:
    model.zero_grad(set_to_none=True)
    seen = set()

    def first_launch(name, index):
        if name in seen:
            return False
        seen.add(name)
        return True

    with TritonLaunchRecorder(select=first_launch) as recorder:
        model(input_ids=ids, labels=ids).loss.backward()
        torch.cuda.synchronize()
    saved, skipped = [], []
    for i, launch in enumerate(recorder.launches):
        size = sum(t.numel() * t.element_size() for arg in [*launch.args, *launch.implicit]
                   for t in (arg.before, arg.after) if t is not None) / 2**20
        if size > a.cap_mb:
            skipped.append({"kernel": launch.kernel_name, "mb": size, "reason": "capture size limit"})
            continue
        name = f"{i:03d}_{launch.kernel_name}"[:120]
        save_launch(launch, a.capture_dir / name)
        saved.append({"dir": name, "kernel": launch.kernel_name, "launch_index": launch.index,
                      "executed": launch.executed, "implicit_recorded": launch.implicit_recorded,
                      "mb": size})
    manifest = {"model_dir": a.model_dir, "config": a.config, "kind": kind,
                "layers": "all" if cfg.model_type in KEEP_ALL_LAYERS else a.layers,
                "tokens": int(ids.shape[1]), "input_ids": ids.cpu().tolist(),
                "provenance_file": str(Path(a.out).resolve()),
                "launches_in_step": recorder.launch_count, "saved": saved, "skipped": skipped,
                "capture_single_recorder_run": True, "capture_storages_retained": recorder.keep_storages,
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "triton": __import__("triton").__version__,
                "binding_status": "not_established", "semantic_verdict": "not_established"}
    with (a.capture_dir / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=1)
    print(f"actual captures {len(saved)}, skipped {len(skipped)}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "x") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"kernels {len(records)}", flush=True)
