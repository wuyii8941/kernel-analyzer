"""Conservative operator-family labels for saved training execution records.

The labels in this module organize coverage.  They do not establish a
reference implementation, numerical bias, or a root cause.  An audited
reference-family label takes precedence; otherwise an exact AOT endpoint is
preferred over a kernel-name signature.  Every observed record receives a
label, with unresolved fused computations kept explicit.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


FAMILY_LABELS = {
    "LINEAR": "矩阵乘法与线性层",
    "NORMALIZATION": "Normalization",
    "SOFTMAX": "Softmax / log-softmax",
    "CROSS_ENTROPY": "Cross-entropy / NLL loss",
    "SILU_GATING": "SiLU 与门控乘法",
    "GELU": "GELU",
    "SOFTPLUS": "Softplus",
    "RECURRENCE": "状态递推 / scan",
    "ROTARY": "Rotary / position scaling",
    "REDUCTION": "求和、均值与其他归约",
    "CONVOLUTION": "卷积",
    "EMBEDDING": "Embedding lookup / backward",
    "INDEXED_ACCUMULATION": "索引、scatter 与梯度累加",
    "SELECTION": "Top-k / sort / selection",
    "MASK_POSITION_CONTROL": "位置、mask 与前缀控制",
    "DATA_MOVEMENT_LAYOUT": "view、copy、transpose 与布局变换",
    "ELEMENTWISE": "逐元素算术与类型转换",
    "FUSED_MIXED": "跨多个计算家族的融合 kernel",
    "UNRESOLVED_LOW_LEVEL": "尚不能从保存语义确定的低层计算",
}


REFERENCE_TO_FAMILY = {
    "INDEXED_ROW_ACCUMULATION": "INDEXED_ACCUMULATION",
    "DEPTHWISE_CONV1D": "CONVOLUTION",
    "GATED_CONV_GRADIENT": "CONVOLUTION",
    "GELU_PRODUCT": "GELU",
    "GELU_PRODUCT_BACKWARD": "GELU",
    "GELU_FORWARD_PRODUCT": "GELU",
    "RMS_BACKWARD": "NORMALIZATION",
    "RMS_SIMPLE_BACKWARD": "NORMALIZATION",
    "RESIDUAL_RMS_FORWARD": "NORMALIZATION",
    "RESIDUAL_RMS_FORWARD_NORMALIZED": "NORMALIZATION",
    "RMS_FORWARD_NORMALIZED": "NORMALIZATION",
    "SOFTMAX_BACKWARD": "SOFTMAX",
    "GROUPED_CAUSAL_SOFTMAX": "SOFTMAX",
    "GROUPED_CAUSAL_SOFTMAX_PROBABILITY": "SOFTMAX",
    "SCALED_MASKED_SOFTMAX": "SOFTMAX",
    "SILU_BACKWARD": "SILU_GATING",
    "SELECTED_SILU_PRODUCT": "SILU_GATING",
    "SOFTPLUS_BIAS_BACKWARD": "SOFTPLUS",
    "DECAYED_RECURRENCE": "RECURRENCE",
    "CONTINUED_RECURRENCE": "RECURRENCE",
    "SEGMENTED_RECURRENCE_FIRST": "RECURRENCE",
    "FORWARD_STATE_RECURRENCE": "RECURRENCE",
    "FORWARD_STATE_RECURRENCE_FINAL": "RECURRENCE",
    "GROUPED_ROTARY_BACKWARD": "ROTARY",
    "ATTENTION_POSITION_SCALING": "ROTARY",
    "EMBEDDING_LOOKUP": "EMBEDDING",
    "EXPONENTIAL_WEIGHTED_REDUCTION": "REDUCTION",
    "CHANNEL_BIAS_ADD": "ELEMENTWISE",
    "SELECTED_NLL": "CROSS_ENTROPY",
    "SELECTED_NLL_BACKWARD": "CROSS_ENTROPY",
    "SOFTCAPPED_NLL_BACKWARD": "CROSS_ENTROPY",
    "ROW_SUM": "REDUCTION",
    "ROW_SQUARE_SUM": "REDUCTION",
}


_LAYOUT = {
    "clone", "copy", "to_copy", "view", "unsafe_view", "reshape",
    "transpose", "permute", "slice", "squeeze", "unsqueeze", "expand",
    "cat", "split", "split_with_sizes", "stack", "constant_pad_nd",
    "to", "unsafe", "constant", "pad", "nd", "with", "sizes",
}
_ELEMENTWISE = {
    "abs", "add", "sub", "mul", "div", "pow", "neg", "exp", "log",
    "floor", "ceil", "eq", "ne", "gt", "ge", "lt", "le", "where",
    "convert_element_type", "to_copy", "scalar_tensor", "zeros", "new_ones",
}


def _tokens(text: str) -> set[str]:
    value = text.lower().replace("::", "_").replace(".", "_")
    result = {part for part in re.split(r"[^a-z0-9]+", value)
              if part and not part.isdigit() and part not in {"triton", "poi", "per", "red", "fused"}}
    # Generated names flatten ATen identifiers at underscores.  Restore only
    # conservative compound operations needed to avoid treating ordinary
    # copy/layout/index calls as opaque numerical kernels.
    for phrase in (
        "to_copy", "unsafe_view", "constant_pad_nd", "index_put",
        "index_add", "scatter_add", "embedding_dense_backward",
        "softmax_backward_data", "log_softmax_backward_data",
        "nll_loss_forward", "nll_loss_backward", "silu_backward",
        "gelu_backward", "softplus_backward", "split_with_sizes",
        "convert_element_type", "prepare_softmax_online",
    ):
        if phrase in value:
            result.add(phrase)
    return result


def _endpoint_operation(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    operation = endpoint.rsplit(":", 1)[-1]
    operation = re.sub(r"_\d+$", "", operation)
    operation = re.sub(r"^(forward|backward)_g\d+__", "", operation)
    return operation


def _families_from_tokens(tokens: set[str], *, exact_operation: str | None) -> set[str]:
    joined = "_".join(sorted(tokens))
    families: set[str] = set()
    if tokens & {"mm", "bmm", "addmm", "matmul"} or (exact_operation and "mm" in exact_operation.split("_")):
        families.add("LINEAR")
    if tokens & {"convolution", "convolution_backward", "conv1d", "conv2d"}:
        families.add("CONVOLUTION")
    if tokens & {"nll", "nll_loss", "nll_loss_forward", "nll_loss_backward"}:
        families.add("CROSS_ENTROPY")
    if "softmax" in joined or "log_softmax" in joined or "prepare_softmax_online" in joined:
        families.add("SOFTMAX")
    if tokens & {"silu", "silu_backward"}:
        families.add("SILU_GATING")
    if tokens & {"gelu", "gelu_backward"}:
        families.add("GELU")
    if tokens & {"softplus", "softplus_backward"}:
        families.add("SOFTPLUS")
    if tokens & {"embedding", "embedding_dense_backward"}:
        families.add("EMBEDDING")
    if tokens & {"index_put", "index_add", "scatter", "scatter_add", "gather", "embedding_dense_backward"}:
        families.add("INDEXED_ACCUMULATION")
    if tokens & {"topk", "sort", "argsort", "argmax", "argmin"}:
        families.add("SELECTION")
    if ("rsqrt" in tokens and tokens & {"mean", "sum", "pow"}) or tokens & {
        "native_layer_norm", "native_layer_norm_backward", "rms_norm"
    }:
        families.add("NORMALIZATION")
    if ({"sin", "cos"} <= tokens) or ({"cat", "neg", "slice"} <= tokens):
        families.add("ROTARY")
    if ("softplus" in tokens and tokens & {"select", "bmm", "cumsum"}) or "scan" in tokens:
        families.add("RECURRENCE")
    if tokens & {"cumsum", "arange", "iota", "bitwise_and"}:
        families.add("MASK_POSITION_CONTROL")
    if tokens & {"sum", "mean", "amax", "max", "prod"}:
        families.add("REDUCTION")
    return families


def classify_observed_position(
    *,
    reference_families: Iterable[str],
    exact_semantic_endpoint_id: str | None,
    exact_aot_endpoint_id: str | None,
    symbol: str,
) -> dict:
    """Return one coverage label and an explicit evidence grade."""

    audited = {REFERENCE_TO_FAMILY[name] for name in reference_families if name in REFERENCE_TO_FAMILY}
    unknown_references = sorted(set(reference_families) - set(REFERENCE_TO_FAMILY))
    if len(audited) == 1 and not unknown_references:
        return {
            "family": next(iter(audited)),
            "classification_basis": "AUDITED_REFERENCE_FAMILY",
            "classification_confidence": "AUDITED",
            "candidate_families": sorted(audited),
            "unknown_reference_labels": [],
        }

    endpoint = exact_semantic_endpoint_id or exact_aot_endpoint_id
    operation = _endpoint_operation(endpoint)
    endpoint_tokens = _tokens(operation or "")
    endpoint_families = _families_from_tokens(endpoint_tokens, exact_operation=operation)
    symbol_tokens = _tokens(symbol)
    if operation == "rsqrt" and symbol_tokens & {"mean", "sum", "pow"}:
        endpoint_families.add("NORMALIZATION")
    # Exact endpoint names such as add, clone and convert_element_type are
    # intentionally kept separate from the larger fused kernel name.
    if operation and len(endpoint_families) == 1:
        family = next(iter(endpoint_families))
        return {
            "family": family,
            "classification_basis": "EXACT_SEMANTIC_ENDPOINT",
            "classification_confidence": "EXACT_ENDPOINT",
            "candidate_families": [family],
            "unknown_reference_labels": unknown_references,
        }
    if operation and endpoint_tokens and endpoint_tokens <= _LAYOUT:
        return {
            "family": "DATA_MOVEMENT_LAYOUT",
            "classification_basis": "EXACT_LAYOUT_ENDPOINT",
            "classification_confidence": "EXACT_ENDPOINT",
            "candidate_families": ["DATA_MOVEMENT_LAYOUT"],
            "unknown_reference_labels": unknown_references,
        }
    if operation and endpoint_tokens and endpoint_tokens <= (_LAYOUT | _ELEMENTWISE):
        return {
            "family": "ELEMENTWISE",
            "classification_basis": "EXACT_ELEMENTWISE_ENDPOINT",
            "classification_confidence": "EXACT_ENDPOINT",
            "candidate_families": ["ELEMENTWISE"],
            "unknown_reference_labels": unknown_references,
        }

    symbol_families = _families_from_tokens(symbol_tokens, exact_operation=None)
    candidates = audited | endpoint_families | symbol_families
    # Reduction is frequently an internal part of normalization, softmax, or
    # loss.  It is not counted as a second family when a more specific family
    # is present.  The same applies to low-level matmul tokens inside rotary or
    # recurrence fusion.
    if len(candidates - {"REDUCTION"}) == 1:
        candidates.discard("REDUCTION")
    if ("ROTARY" in candidates or "RECURRENCE" in candidates) and "LINEAR" in candidates:
        candidates.discard("LINEAR")
    if "CROSS_ENTROPY" in candidates and "SOFTMAX" in candidates:
        candidates.discard("SOFTMAX")
    if len(candidates) == 1:
        family = next(iter(candidates))
        return {
            "family": family,
            "classification_basis": "FUSED_SYMBOL_SIGNATURE",
            "classification_confidence": "STRUCTURAL_SUGGESTION",
            "candidate_families": [family],
            "unknown_reference_labels": unknown_references,
        }
    if len(candidates) > 1:
        return {
            "family": "FUSED_MIXED",
            "classification_basis": "MULTIPLE_STRUCTURAL_FAMILIES",
            "classification_confidence": "STRUCTURAL_SUGGESTION",
            "candidate_families": sorted(candidates),
            "unknown_reference_labels": unknown_references,
        }
    if symbol_tokens and symbol_tokens <= (_LAYOUT | {"triton", "poi", "per", "red", "fused"}):
        family, basis = "DATA_MOVEMENT_LAYOUT", "LAYOUT_SYMBOL_SIGNATURE"
    elif symbol_tokens and symbol_tokens <= (_LAYOUT | _ELEMENTWISE | {"triton", "poi", "per", "red", "fused"}):
        family, basis = "ELEMENTWISE", "ELEMENTWISE_SYMBOL_SIGNATURE"
    else:
        family, basis = "UNRESOLVED_LOW_LEVEL", "NO_CONSERVATIVE_FAMILY_MATCH"
    return {
        "family": family,
        "classification_basis": basis,
        "classification_confidence": "STRUCTURAL_SUGGESTION" if family != "UNRESOLVED_LOW_LEVEL" else "UNRESOLVED",
        "candidate_families": [family],
        "unknown_reference_labels": unknown_references,
    }
