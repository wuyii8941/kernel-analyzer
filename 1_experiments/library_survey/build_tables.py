"""Check the per-library inventories (inventory/<lib>.json) and write families.md: per library, every kernel family
with its category and either the torch reference program or the decisions still open.

    python build_tables.py            # checks, writes families.md, prints the summary counts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATEGORIES = {"torch_op": "torch 运算", "torch_composition": "torch 组合", "needs_definition": "需要定义",
              "random": "随机"}
DIRECTIONS = {"forward": "前向", "backward": "反向", "both": "前向+反向", "n/a": "—"}

# spec classes of the families still to be defined (needs_definition) or random: keyword rules on the family and kernel
# names (then on the open decisions), first match wins, then the reviewed overrides below
SPEC_CLASSES = {
    "A": "线性注意力与状态递推", "B": "分页 / 变长 KV 注意力", "C": "稀疏注意力、索引器与 KV 压缩",
    "E": "KV cache 写入、读出与 KV 量化", "F": "量化格式与量化 / 反量化", "G": "量化与低比特 GEMM",
    "H": "MoE：路由、排序对齐、分组 GEMM、EP 分发", "K": "块稀疏矩阵乘", "I": "采样、投机解码、哈希与水印",
    "J": "布局、元数据与调度", "M": "RoPE 与位置编码变体", "L": "LoRA", "O": "其他", "R": "随机",
}
RULES = [
    ("A", r"\bgla\b|delta|rwkv|hgrn|\bgsa\b|\babc\b|based|retention|retnet|linear[_ ]attn|linear attention|fused_recurrent|"
          r"recurren|\bkda\b|\bgdn\b|gated[_ ]delta|mamba|\bssm\b|\bssd\b|selective|causal[_ ]conv|conv1d|lightning|decay|"
          r"\bttt\b|mesa|comba|\boja\b|dplr|iplr|\bpath\b|log-linear|state pool|chunk_h|wy_|solve_tril|deltaformer|cyfa|\batk\b|"
          r"stick"),
    ("C", r"\bnsa\b|sparse[_ ]attn|sparse attention|indexer|\bdsa\b|\bqsa\b|minimax|v3\.2|dsv4|ds v4|flashmla|sparse[- ]index|"
          r"topk[_ ]indices"),
    ("B", r"\bmla\b|latent|paged|decode|extend|prefill|kv_indptr|req_to_token|block[_ ]table|varlen|split[-_ ]?kv|"
          r"flash[_ ]?decod|attention|attn"),
    ("E", r"kv[_ ]?cache|copy_kv|kv copy|store[_ ]kv|fill|flatten"),
    ("H", r"\bmoe\b|expert|rout|gating|align|\bep\b|grouped[_ ]gemm|topk_softmax|group_topk"),
    ("K", r"\bsdd\b|\bdsd\b|\bdds\b|block-sparse|block sparse"),
    ("L", r"lora|sgmm|sgemm"),
    ("I", r"sampl|penalt|rejection|verify|speculat|eagle|draft|top_p|top-p|multinomial|gumbel|watermark|logit"),
    ("G", r"gemm|matmul|\bmm\b|gemv|dot_scaled|w8a8|w4a16|gptq|awq|marlin|dequant.*mat|bitplane|a16w"),
    ("F", r"quant|fp8|int8|int4|mxfp|nvfp4|e8m0|ue8m0|fp4"),
    ("J", r"swizzle|layout|transpose|pack"),
    ("M", r"rope|rotary"),
]
OVERRIDES = {   # (library name prefix, family name prefix) -> class, after reading the families
    ("LMDeploy", "fill_paged_kv_cache"): "E", ("LMDeploy", "flatten_paged_kv_cache"): "E",
    ("LMDeploy", "bitonic_topk_varlen"): "C", ("LMDeploy", "moe_route_sort_metadata"): "H",
    ("LMDeploy", "ep_token_scatter_fp8"): "H", ("LMDeploy", "dsv4_swiglu_quant_fp8"): "F",
    ("SageAttention", "per-block INT8 quantisation of Q and K (varlen)"): "F",
    ("SageAttention", "per-thread INT8 quantisation"): "F",
    ("sglang", "dsa_kpool_compress_fp8_write"): "C", ("sglang", "qsa_"): "C", ("sglang", "qwen38_qsa_packed_decode"): "C",
    ("sglang", "dsa_fp8_kv_cache_pack_quant_dequant"): "C", ("sglang", "dsa_sparse_mla_attention_fwd"): "C",
    ("sglang", "dsa_kpool_fused_metadata"): "C", ("sglang", "dsa_kpool_bookkeeping"): "C",
    ("sglang", "sla_block_sparse_attention_fwd"): "C", ("sglang", "vsa_h3_block_sparse_attention_fwd"): "C",
    ("sglang", "minicpm_sparse_compress_k"): "C", ("sglang", "seg_la_segmented_linear_attention"): "A",
    ("sglang", "kda_chunked_prefill"): "A", ("sglang", "lfm2_short_conv"): "A", ("sglang", "mxfp4_upscale_decode"): "F",
    ("sglang", "attn_output_sigmoid_gate"): "F", ("sglang", "mxfp8_paged_kv_store"): "E",
    ("sglang", "engram_ngram_hash"): "I", ("sglang", "spec.draft_decode_kv_indices"): "J",
    ("sglang", "decode_num_kv_splits_heuristic"): "J", ("sglang", "vattn_asm_mtp_verify_segment_plan"): "J",
    ("sglang", "dsv4_wo_a_mxfp8_epilogue"): "F", ("sglang", "aiter_moe_sorting_small_mxfp8"): "H",
    ("sglang", "mhc_"): "O", ("sglang", "waterfill_shared_expert_expand"): "H", ("sglang", "dcp_kv_index_builders"): "J",
    ("sglang", "dsa_cp_interleave_q_split"): "J", ("sglang", "spec.selector_walk"): "I",
    ("sglang", "spec.lilicorr_topk_lse"): "I", ("sglang", "qwen4_ple_ngram_hash"): "I", ("sglang", "gpu_tensor_hash"): "I",
    ("sglang", "fused_moe_triton_gemm"): "H", ("sglang", "fused_moe_wna16_gemm"): "H",
    ("sglang", "grouped_fp8_quant_moe_layouts"): "H", ("sglang", "spec.align_evict_mask_to_page_size"): "J",
    ("sglang", "swa_mla_fallback_gather_and_mask"): "B",
    ("lightllm", "dsa_fp8_paged_mqa_logits"): "C", ("lightllm", "flashinfer_mla_decode_plan"): "J",
    ("lightllm", "mhc_pre_norm"): "O", ("lightllm", "glm5_"): "C", ("lightllm", "fp8_activation_group_quant"): "F",
    ("LMDeploy", "dsa_fp8_index_logits"): "C",
    ("kernl", "experimental llama: FP8"): "F", ("kernl", "experimental llama: fused rmsnorm"): "M",
    ("flash-linear-attention", "[Ascend] GDN2"): "A", ("flash-linear-attention", "Common chunk state passing"): "A",
    ("OpenAI", "reduce backward"): "H", ("OpenAI", "masked/scaled reduction"): "H",
    ("OpenAI", "Hopper MXFP4 value swizzle"): "J", ("OpenAI", "Blackwell MXFP4 value repack"): "J",
    ("OpenAI", "Blackwell MX4 shuffled"): "J", ("OpenAI", "FPSan"): "O", ("OpenAI", "SwiGLU"): "O",
    ("OpenAI", "reduce row scheduler"): "J", ("OpenAI", "bitmatrix metadata"): "J", ("OpenAI", "ragged tensor metadata"): "J",
    ("applied-ai", "gptq_w4a16_fused_moe_gemm"): "H", ("stk", "block_sparse_matmul"): "K",
}


def spec_class(lib: dict, f: dict) -> str:
    import re
    if f["category"] == "random":
        return "R"
    for (lp, fp), c in OVERRIDES.items():
        if lib["library"].startswith(lp) and f["family"].startswith(fp):
            return c
    names = (f["family"] + " " + " ".join(f["kernels"])).lower()
    for text in (names, names + " " + " ".join(f.get("open_decisions") or []).lower()):
        for c, pat in RULES:
            if re.search(pat, text):
                return c
    return "O"


def load() -> list:
    libs, errors = [], []
    for path in sorted((HERE / "inventory").glob("*.json")):
        lib = json.loads(path.read_text())
        for key in ("library", "repo", "commit", "license", "jit_functions", "entry_kernels", "families"):
            if key not in lib:
                errors.append(f"{path.name}: missing {key}")
        for i, f in enumerate(lib.get("families", [])):
            where = f"{path.name} family {i} ({f.get('family')})"
            if f.get("category") not in CATEGORIES:
                errors.append(f"{where}: category {f.get('category')!r}")
            elif f["category"] in ("torch_op", "torch_composition") and not f.get("reference"):
                errors.append(f"{where}: {f['category']} without a reference")
            elif f["category"] == "needs_definition" and not f.get("open_decisions"):
                errors.append(f"{where}: needs_definition without open decisions")
            if f.get("sm86") not in ("yes", "no", "unknown"):
                errors.append(f"{where}: sm86 {f.get('sm86')!r}")
        libs.append(lib)
    if errors:
        sys.exit("\n".join(errors))
    return libs


def cell(text) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").replace("](", "]\\(")   # not a markdown link


def summary(libs: list) -> list:
    rows = []
    for lib in libs:
        fam = lib["families"]
        count = {c: sum(f["category"] == c for f in fam) for c in CATEGORIES}
        kernels = {c: sum(len(f["kernels"]) for f in fam if f["category"] == c) for c in CATEGORIES}
        rows.append({"library": lib["library"], "entry_kernels": lib["entry_kernels"], "families": len(fam),
                     "count": count, "kernels": kernels, "sm86_no": sum(f["sm86"] == "no" for f in fam)})
    return rows


def write(libs: list) -> str:
    out = ["# 各库的 kernel 分族明细", "",
           "由 `build_tables.py` 从 `inventory/*.json` 生成，不要手改。类别：torch 运算 / torch 组合（参照程序已定义）、"
           "需要定义、随机。反向 kernel 的参照程序是前向参照程序的梯度。", ""]
    out += ["| 库 | 入口 kernel | 分族 | torch 运算 | torch 组合 | 需要定义 | 随机 | sm_86 不能跑的分族 |",
            "|---|---|---|---|---|---|---|---|"]
    for r in summary(libs):
        c = r["count"]
        out.append(f"| {r['library']} | {r['entry_kernels']} | {r['families']} | {c['torch_op']} | "
                   f"{c['torch_composition']} | {c['needs_definition']} | {c['random']} | {r['sm86_no']} |")
    for lib in libs:
        out += ["", f"## {lib['library']}", "",
                f"{lib['repo']}，提交 `{lib['commit'][:12]}`，许可 {lib['license']}；`@triton.jit` 函数 "
                f"{lib['jit_functions']} 个，入口 kernel {lib['entry_kernels']} 个。", "",
                "| 分族 | 方向 | 类别 | 规格类别 | 参照程序，或待定事项 | sm_86 | 说明 |", "|---|---|---|---|---|---|---|"]
        order = list(CATEGORIES)
        for f in sorted(lib["families"], key=lambda f: (order.index(f["category"]), f["family"])):
            what = f["reference"] if f["category"] in ("torch_op", "torch_composition") else \
                "；".join(f.get("open_decisions") or []) or f.get("reference") or ""
            sc = spec_class(lib, f) if f["category"] in ("needs_definition", "random") else ""
            out.append(f"| {cell(f['family'])} | {DIRECTIONS.get(f['direction'], f['direction'])} | "
                       f"{CATEGORIES[f['category']]} | {sc} | {cell(what)} | {f['sm86']} | {cell(f.get('notes'))} |")
    return "\n".join(out) + "\n"


def classes(libs: list) -> dict:
    """spec class -> its families (library, family, kernels, sm86)"""
    out = {c: [] for c in SPEC_CLASSES}
    for lib in libs:
        for f in lib["families"]:
            if f["category"] in ("needs_definition", "random"):
                out[spec_class(lib, f)].append({"library": lib["library"], "family": f["family"],
                                                "kernels": len(f["kernels"]), "sm86": f["sm86"]})
    return out


def class_table(libs: list) -> str:
    rows = ["| | 规格类别 | 分族 | 入口 kernel | 涉及的库 | sm_86 能跑的分族 |", "|---|---|---|---|---|---|"]
    for c, fams in classes(libs).items():
        if fams:
            names = sorted({f["library"].split(" (")[0] for f in fams})
            rows.append(f"| {c} | {SPEC_CLASSES[c]} | {len(fams)} | {sum(f['kernels'] for f in fams)} | {'、'.join(names)} | "
                        f"{sum(f['sm86'] == 'yes' for f in fams)} |")
    return "\n".join(rows) + "\n"


def main():
    libs = load()
    (HERE / "families.md").write_text(write(libs))
    (HERE / "classes.md").write_text("# 需要定义的规格：按类别汇总\n\n由 `build_tables.py` 生成，不要手改。每个分族的类别见 "
                                     "`families.md` 的「规格类别」列。\n\n" + class_table(libs))
    json_out = {c: {"name": SPEC_CLASSES[c], "families": fams} for c, fams in classes(libs).items()}
    (HERE / "classes.json").write_text(json.dumps(json_out, ensure_ascii=False, indent=1))
    total = {c: 0 for c in CATEGORIES}
    for r in summary(libs):
        print(f"{r['library']:<24} entry {r['entry_kernels']:>4}  families {r['families']:>3}  "
              + "  ".join(f"{c} {r['count'][c]}" for c in CATEGORIES) + f"  sm86-no {r['sm86_no']}")
        for c in CATEGORIES:
            total[c] += r["count"][c]
    print("total families", total)


if __name__ == "__main__":
    main()
