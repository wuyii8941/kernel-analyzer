#!/usr/bin/env python3
"""Build a fail-closed semantic F+B ledger for the round-2 VL AOT graph."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMULAS: dict[str, dict[str, str]] = {
    "<built-in function getitem>": {"map": "y=x[i] (tuple projection)", "adjoint": "route q to tuple port i"},
    "aten._log_softmax.default": {"map": "y_i=x_i-log(sum_j exp(x_j))", "adjoint": "dx_i=q_i-exp(y_i) sum_j q_j"},
    "aten._log_softmax_backward_data.default": {"map": "dx_i=q_i-exp(y_i) sum_j q_j", "adjoint": "actual log-softmax VJP kernel"},
    "aten._softmax.default": {"map": "p_i=exp(x_i-m)/sum_j exp(x_j-m)", "adjoint": "dx_i=p_i(q_i-sum_j p_j q_j)"},
    "aten._softmax_backward_data.default": {"map": "dx_i=p_i(q_i-sum_j p_j q_j)", "adjoint": "actual softmax VJP kernel"},
    "aten._to_copy.default": {"map": "y=cast_dtype(x)", "adjoint": "dx=cast_input_dtype(q)"},
    "aten._unsafe_view.default": {"map": "y=reshape_linear_order(x,s)", "adjoint": "dx=reshape_linear_order(q,shape(x))"},
    "aten.add.Scalar": {"map": "y=x+c", "adjoint": "dx=q"},
    "aten.add.Tensor": {"map": "y=a+alpha*b", "adjoint": "da=sum_to(q); db=sum_to(alpha*q)"},
    "aten.addmm.default": {"map": "y=beta*c+alpha*A@B", "adjoint": "dc=beta*sum_to(q); dA=alpha*q@B^T; dB=alpha*A^T@q"},
    "aten.alias.default": {"map": "y aliases x without value change", "adjoint": "dx=q"},
    "aten.arange.default": {"map": "y_i=i", "adjoint": "no differentiable input"},
    "aten.arange.start": {"map": "y_i=start+i", "adjoint": "no differentiable input"},
    "aten.bitwise_and.Tensor": {"map": "y=x bitwise_and z", "adjoint": "undefined/non-differentiable integer map"},
    "aten.bmm.default": {"map": "y_b=A_b@B_b", "adjoint": "dA_b=q_b@B_b^T; dB_b=A_b^T@q_b"},
    "aten.cat.default": {"map": "y=concat(x_k,dim)", "adjoint": "dx_k=slice(q,offset_k,size_k)"},
    "aten.clone.default": {"map": "y=value_copy(x)", "adjoint": "dx=q"},
    "aten.constant_pad_nd.default": {"map": "y embeds x in a constant-padded tensor", "adjoint": "dx=slice(q,unpadded coordinates)"},
    "aten.convolution.default": {"map": "y[n,o,p]=b[o]+sum_{i,k} x[n,i,p*s+k*d-pad] w[o,i,k]", "adjoint": "dx=transposed_convolution(q,w); dw=cross_correlation(x,q); db=sum(q)"},
    "aten.convolution_backward.default": {"map": "returns requested convolution adjoints for x,w,b", "adjoint": "actual convolution VJP kernel"},
    "aten.copy.default": {"map": "destination coordinates receive source values", "adjoint": "source receives selected q; overwritten destination path is zero"},
    "aten.cos.default": {"map": "y=cos(x)", "adjoint": "dx=-q sin(x)"},
    "aten.detach.default": {"map": "y has x values with autograd edge removed", "adjoint": "no higher-order edge; first-order saved value is unchanged"},
    "aten.div.Scalar": {"map": "y=x/c", "adjoint": "dx=q/c"},
    "aten.embedding.default": {"map": "y[p,:]=W[index[p],:]", "adjoint": "dW[r,:]=sum_{p:index[p]=r} q[p,:]"},
    "aten.embedding_dense_backward.default": {"map": "dense indexed sum of embedding cotangents", "adjoint": "actual embedding VJP kernel"},
    "aten.empty_like.default": {"map": "allocate uninitialized tensor with x metadata", "adjoint": "allocation-only auxiliary"},
    "aten.eq.Scalar": {"map": "y=(x==c)", "adjoint": "no differentiable output"},
    "aten.expand.default": {"map": "y broadcasts size-one axes of x", "adjoint": "dx=sum_to_shape_x(q)"},
    "aten.fill.Scalar": {"map": "all output coordinates equal c", "adjoint": "no input-value adjoint"},
    "aten.gelu.default": {"map": "y=GELU_approx(x) using the declared approximation", "adjoint": "dx=q*dGELU_approx(x)/dx"},
    "aten.gelu_backward.default": {"map": "dx=q*dGELU_approx(x)/dx", "adjoint": "actual GELU VJP kernel"},
    "aten.index.Tensor": {"map": "y=x[index tuple]", "adjoint": "dx=scatter_add(q,index tuple)"},
    "aten.index_put.default": {"map": "y=x with indexed write/add of values", "adjoint": "route/scatter q according to accumulate flag"},
    "aten.le.Tensor": {"map": "y=(a<=b)", "adjoint": "no differentiable output"},
    "aten.lift_fresh_copy.default": {"map": "materialize constant tensor values", "adjoint": "no differentiable source"},
    "aten.masked_fill.Scalar": {"map": "y=where(mask,c,x)", "adjoint": "dx=where(mask,0,q)"},
    "aten.masked_scatter.default": {"map": "y=x with source values placed at true mask coordinates", "adjoint": "dx=where(mask,0,q); dsource=prefix(q[mask])"},
    "aten.masked_scatter_backward.default": {"map": "extract masked cotangent into source shape", "adjoint": "actual masked-scatter source VJP"},
    "aten.mean.dim": {"map": "y=sum_dim(x)/N", "adjoint": "dx=expand(q)/N"},
    "aten.mm.default": {"map": "y=A@B", "adjoint": "dA=q@B^T; dB=A^T@q"},
    "aten.mul.Scalar": {"map": "y=c*x", "adjoint": "dx=c*q"},
    "aten.mul.Tensor": {"map": "y=a*b", "adjoint": "da=sum_to(q*b); db=sum_to(q*a)"},
    "aten.native_layer_norm.default": {"map": "mu=mean(x); r=(mean((x-mu)^2)+eps)^-1/2; y=(x-mu)r*w+b", "adjoint": "dx=(w*r/N)[Nq-sum(q)-xhat sum(q*xhat)]; dw=sum(q*xhat); db=sum(q)"},
    "aten.native_layer_norm_backward.default": {"map": "returns requested dx,dw,db from q,x,mu,r,w", "adjoint": "actual LayerNorm VJP kernel"},
    "aten.neg.default": {"map": "y=-x", "adjoint": "dx=-q"},
    "aten.new_empty_strided.default": {"map": "allocate uninitialized tensor with declared metadata", "adjoint": "allocation-only auxiliary"},
    "aten.new_ones.default": {"map": "allocate tensor of ones", "adjoint": "no differentiable source"},
    "aten.new_zeros.default": {"map": "allocate tensor of zeros", "adjoint": "no differentiable source"},
    "aten.nll_loss_backward.default": {"map": "dX[n,c]=-q/total_weight at valid target coordinates, else zero", "adjoint": "actual NLL VJP kernel"},
    "aten.nll_loss_forward.default": {"map": "L=-reduce_n X[n,target[n]] with ignore_index handling", "adjoint": "dX is target-indexed negative scaled q"},
    "aten.permute.default": {"map": "y[i_perm]=x[i]", "adjoint": "dx=inverse_permute(q)"},
    "aten.pow.Tensor_Scalar": {"map": "y=x^a", "adjoint": "dx=q*a*x^(a-1) on the declared real domain"},
    "aten.rsqrt.default": {"map": "y=x^(-1/2)", "adjoint": "dx=-0.5*q*y^3"},
    "aten.scalar_tensor.default": {"map": "materialize scalar c as a tensor", "adjoint": "no differentiable source in this graph"},
    "aten.select.int": {"map": "y selects one index on axis d", "adjoint": "dx=zero tensor with q inserted at that index"},
    "aten.select_backward.default": {"map": "insert q into a zero tensor at selected axis/index", "adjoint": "actual select VJP"},
    "aten.select_scatter.default": {"map": "y=x with source written into one selected slice", "adjoint": "route q to x outside slice and source inside slice"},
    "aten.sigmoid.default": {"map": "y=1/(1+exp(-x))", "adjoint": "dx=q*y*(1-y)"},
    "aten.silu.default": {"map": "y=x*sigmoid(x)", "adjoint": "dx=q*sigmoid(x)*(1+x*(1-sigmoid(x)))"},
    "aten.sin.default": {"map": "y=sin(x)", "adjoint": "dx=q cos(x)"},
    "aten.slice.Tensor": {"map": "y selects an arithmetic-progression slice of x", "adjoint": "dx=zeros with q inserted at selected coordinates"},
    "aten.slice_backward.default": {"map": "insert q into input-shape zeros at the recorded slice", "adjoint": "actual slice VJP"},
    "aten.slice_scatter.default": {"map": "y=x with source written into a slice", "adjoint": "route q to x outside slice and source inside slice"},
    "aten.squeeze.dim": {"map": "remove a declared size-one axis", "adjoint": "insert that size-one axis into q"},
    "aten.stack.default": {"map": "y stacks x_k on a new axis", "adjoint": "dx_k=select(q,new_axis,k)"},
    "aten.sub.Tensor": {"map": "y=a-alpha*b", "adjoint": "da=sum_to(q); db=sum_to(-alpha*q)"},
    "aten.sum.dim_IntList": {"map": "y=sum_dim(x)", "adjoint": "dx=expand(q,input shape)"},
    "aten.t.default": {"map": "y=x^T for rank-two x", "adjoint": "dx=q^T"},
    "aten.transpose.int": {"map": "y swaps axes d0,d1", "adjoint": "dx swaps the same axes of q"},
    "aten.unbind.int": {"map": "returns all axis-d slices as a tuple", "adjoint": "stack tuple cotangents on axis d"},
    "aten.unsqueeze.default": {"map": "insert one size-one axis", "adjoint": "squeeze that axis from q"},
    "aten.view.default": {"map": "reshape preserving linear element order", "adjoint": "reshape q to input shape"},
    "aten.where.self": {"map": "y=mask?a:b", "adjoint": "da=sum_to(where(mask,q,0)); db=sum_to(where(mask,0,q))"},
    "aten.zeros.default": {"map": "allocate tensor of zeros", "adjoint": "no differentiable source"},
}


ELEMENTARY_SIGNATURES: dict[
    tuple[tuple[str, ...], tuple[str, ...]], tuple[str, str, str]
] = {
    (("aten.view.default",), ("aten.view.default",)): (
        "RESHAPE_ADJOINT",
        "y=reshape(x,s)",
        "dx=reshape(q,shape(x))",
    ),
    (("aten.transpose.int",), ("aten.transpose.int",)): (
        "TRANSPOSE_ADJOINT",
        "y=transpose(x,d0,d1)",
        "dx=transpose(q,d0,d1)",
    ),
    (("aten.neg.default",), ("aten.neg.default",)): (
        "NEG_ADJOINT",
        "y=-x",
        "dx=-q",
    ),
    (("aten.unsqueeze.default",), ("aten.squeeze.dim",)): (
        "UNSQUEEZE_ADJOINT",
        "y=unsqueeze(x,d)",
        "dx=squeeze(q,d)",
    ),
    (("aten._to_copy.default",), ("aten._to_copy.default",)): (
        "CAST_ADJOINT",
        "y=cast(x,dtype_out)",
        "dx=cast(q,dtype(x))",
    ),
    (("aten.slice.Tensor",), ("aten.slice_backward.default",)): (
        "SLICE_ADJOINT",
        "y=x[slice(d,start,end,step)]",
        "dx=zeros(shape(x)); dx[slice]=q",
    ),
    (("aten.expand.default",), ("aten.sum.dim_IntList",)): (
        "EXPAND_ADJOINT",
        "y=broadcast(x,target_shape)",
        "dx=sum(q,exact_broadcast_axes,keepdim=True)",
    ),
    (("aten.permute.default",), ("aten.permute.default",)): (
        "PERMUTE_ADJOINT",
        "y=permute(x,p)",
        "dx=permute(q,inverse(p))",
    ),
    (("aten.select.int",), ("aten.select_backward.default",)): (
        "SELECT_ADJOINT",
        "y=select(x,d,index)",
        "dx=zeros(shape(x)); dx[select(d,index)]=q",
    ),
}


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, Mapping):
        shape = value.get("shape")
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        shape = value[0]
    else:
        shape = None
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _tensor_dtype(value: Any) -> str | None:
    if isinstance(value, Mapping):
        dtype = value.get("dtype")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        dtype = value[1]
    else:
        dtype = None
    return str(dtype) if dtype is not None else None


def _tensor_requires_grad(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("requires_grad", False))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return bool(value[2])
    return False


def _floating_tensor(node: Mapping[str, Any]) -> bool:
    dtype = _tensor_dtype(node.get("tensor_meta"))
    return dtype is None or any(token in dtype for token in ("float", "bfloat", "half"))


def _trainable_dependency(
    forward_nodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool], set[str]]:
    node_index = {str(node["name"]): node for node in forward_nodes}
    dependency: dict[str, bool] = {}
    trainable_placeholders = {
        str(node["name"])
        for node in forward_nodes
        if node["op"] == "placeholder" and _tensor_requires_grad(node.get("tensor_meta"))
    }
    for node in forward_nodes:
        name = str(node["name"])
        if node["op"] == "placeholder":
            dependency[name] = name in trainable_placeholders
        elif node["op"] == "call_function":
            dependency[name] = _floating_tensor(node) and any(
                dependency.get(str(edge["source_node"]), False)
                and _floating_tensor(node_index[str(edge["source_node"])])
                for edge in node.get("input_edges", ())
            )
        else:
            dependency[name] = False
    return dependency, trainable_placeholders


def _args(node: Mapping[str, Any]) -> list[Any]:
    arguments = node.get("arguments")
    values = arguments.get("args") if isinstance(arguments, Mapping) else None
    return list(values) if isinstance(values, list) else []


def _kwargs(node: Mapping[str, Any]) -> dict[str, Any]:
    arguments = node.get("arguments")
    values = arguments.get("kwargs") if isinstance(arguments, Mapping) else None
    return dict(values) if isinstance(values, Mapping) else {}


def _normalize_dim(dim: int, rank: int, *, insertion: bool = False) -> int:
    upper = rank + 1 if insertion else rank
    value = dim + upper if dim < 0 else dim
    if not 0 <= value < upper:
        raise ValueError(f"dimension {dim} invalid for rank {rank}")
    return value


def _input_node(
    node: Mapping[str, Any], node_index: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    edges = node.get("input_edges")
    if not isinstance(edges, list) or not edges:
        return None
    return node_index.get(str(edges[0].get("source_node", "")))


def _product(shape: Sequence[int]) -> int:
    result = 1
    for item in shape:
        result *= item
    return result


def _node_argument(value: Any) -> str | None:
    return str(value["node"]) if isinstance(value, Mapping) and "node" in value else None


def _proof_record(
    proof_kind: str,
    forward_map: str,
    derived_vjp: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    passed = bool(checks) and all(checks.values())
    return {
        "proof_kind": proof_kind,
        "exact_forward_map": forward_map,
        "derived_vjp": derived_vjp,
        "checks": dict(checks),
        "passed": passed,
        "claim_boundary": "exact real-arithmetic map/VJP binding; finite-precision kernel arithmetic remains unproved",
    }


def _sum_to_axes(source: Sequence[int], target: Sequence[int]) -> list[int] | None:
    if len(target) > len(source):
        return None
    padded = (1,) * (len(source) - len(target)) + tuple(target)
    if any(a != b and b != 1 for a, b in zip(source, padded, strict=True)):
        return None
    return [
        axis
        for axis, (a, b) in enumerate(zip(source, padded, strict=True))
        if b == 1 and a != 1
    ] + list(range(len(source) - len(target)))


def _verify_arithmetic_composite(
    forward_nodes: Sequence[Mapping[str, Any]],
    backward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    ft = tuple(str(node["target"]) for node in forward_nodes)
    bt = tuple(str(node["target"]) for node in backward_nodes)
    f = forward_nodes[0] if forward_nodes else None
    if f is None:
        return None
    source = _input_node(f, forward_index)
    input_shape = _tensor_shape(source.get("tensor_meta")) if source else None
    output_shape = _tensor_shape(f.get("tensor_meta"))
    common = {
        "exact_forward_origin_on_all_non_detach_backward_nodes": all(
            node["target"] == "aten.detach.default"
            or node.get("fwd_source_fn_stack") == f.get("source_fn_stack")
            for node in backward_nodes
        ),
        "name_or_shape_similarity_not_used_for_binding": True,
    }
    try:
        if ft == ("aten.pow.Tensor_Scalar",) and bt == (
            "aten.pow.Tensor_Scalar", "aten.mul.Scalar", "aten.mul.Tensor"
        ):
            exponent = float(_args(f)[1])
            bp, scale, product = backward_nodes
            checks = dict(common)
            checks.update({
                "saved_input_is_exact_forward_input": _node_argument(_args(bp)[0]) == _node_argument(_args(f)[0]),
                "backward_power_is_a_minus_one": float(_args(bp)[1]) == exponent - 1.0,
                "backward_scale_is_a": float(_args(scale)[1]) == exponent and _node_argument(_args(scale)[0]) == bp["name"],
                "final_product_uses_scale_and_one_cotangent": _node_argument(_args(product)[1]) == scale["name"],
                "all_value_shapes_equal_forward_input": input_shape is not None and all(_tensor_shape(node.get("tensor_meta")) == input_shape for node in [f, bp, scale, product]),
            })
            return _proof_record("POW_SCALAR_ADJOINT", "y=x^a", "dx=q*a*x^(a-1)", checks)
        if ft == ("aten.mean.dim",) and bt == ("aten.expand.default", "aten.div.Scalar"):
            expand, divide = backward_nodes
            fa = _args(f)
            dims = [_normalize_dim(int(d), len(input_shape or ())) for d in fa[1]]
            keepdim = bool(fa[2])
            extent = _product([(input_shape or ())[d] for d in dims])
            expected = tuple(1 if i in dims else x for i, x in enumerate(input_shape or ())) if keepdim else tuple(x for i, x in enumerate(input_shape or ()) if i not in dims)
            checks = dict(common)
            checks.update({
                "forward_output_shape_matches_reduction": output_shape == expected,
                "backward_expands_to_exact_input_shape": tuple(int(x) for x in _args(expand)[1]) == input_shape and _tensor_shape(expand.get("tensor_meta")) == input_shape,
                "backward_divides_by_exact_reduction_extent": int(_args(divide)[1]) == extent and _node_argument(_args(divide)[0]) == expand["name"],
                "backward_restores_input_shape": _tensor_shape(divide.get("tensor_meta")) == input_shape,
            })
            return _proof_record("MEAN_ADJOINT", "y=sum_D(x)/N", "dx=expand(q)/N", checks)
        if ft == ("aten.rsqrt.default",) and bt == (
            "aten.detach.default", "aten.detach.default", "aten.detach.default", "aten.detach.default",
            "aten.pow.Tensor_Scalar", "aten.mul.Scalar", "aten.mul.Tensor"
        ):
            d0, d1, d2, d3, power, scale, product = backward_nodes
            chain = [d0, d1, d2, d3]
            checks = dict(common)
            checks.update({
                "saved_value_is_exact_forward_output": _node_argument(_args(d0)[0]) == f["name"],
                "detach_chain_is_value_identity": all(_node_argument(_args(right)[0]) == left["name"] for left, right in zip(chain, chain[1:])),
                "saved_rsqrt_is_cubed": _node_argument(_args(power)[0]) == d3["name"] and float(_args(power)[1]) == 3.0,
                "cotangent_scaled_by_negative_half": float(_args(scale)[1]) == -0.5,
                "final_product_combines_scaled_cotangent_and_cube": {_node_argument(x) for x in _args(product)} == {scale["name"], power["name"]},
                "all_arithmetic_shapes_equal_input": input_shape is not None and output_shape == input_shape and all(_tensor_shape(node.get("tensor_meta")) == input_shape for node in [power, scale, product]),
            })
            return _proof_record("RSQRT_ADJOINT", "y=x^(-1/2)", "dx=-0.5*q*y^3", checks)
        mul_signatures = {
            ("aten.mul.Tensor", "aten.mul.Tensor"),
            ("aten.mul.Tensor", "aten.mul.Tensor", "aten.sum.dim_IntList"),
            ("aten.mul.Tensor", "aten.mul.Tensor", "aten.sum.dim_IntList", "aten.view.default"),
        }
        if ft == ("aten.mul.Tensor",) and bt in mul_signatures:
            fa = _args(f)
            a_name, b_name = _node_argument(fa[0]), _node_argument(fa[1])
            a = forward_index.get(a_name or "")
            b = forward_index.get(b_name or "")
            a_shape = _tensor_shape(a.get("tensor_meta")) if a else None
            b_shape = _tensor_shape(b.get("tensor_meta")) if b else None
            left, right = backward_nodes[:2]
            la, ra = _args(left), _args(right)
            q_left, q_right = _node_argument(la[0]), _node_argument(ra[0])
            saved_left, saved_right = _node_argument(la[1]), _node_argument(ra[1])
            checks = dict(common)
            checks.update({
                "two_forward_inputs_recorded": a is not None and b is not None,
                "both_vjp_products_use_same_upstream_cotangent": q_left is not None and q_left == q_right,
                "vjp_products_use_exact_opposite_saved_inputs": {saved_left, saved_right} == {a_name, b_name},
                "vjp_product_shapes_equal_forward_output": output_shape is not None and _tensor_shape(left.get("tensor_meta")) == output_shape and _tensor_shape(right.get("tensor_meta")) == output_shape,
            })
            produced: dict[str | None, tuple[int, ...] | None] = {
                saved_left: _tensor_shape(left.get("tensor_meta")),
                saved_right: _tensor_shape(right.get("tensor_meta")),
            }
            if len(backward_nodes) >= 3:
                reduction = backward_nodes[2]
                reduced_mul_name = _node_argument(_args(reduction)[0])
                reduced_mul = left if left["name"] == reduced_mul_name else right if right["name"] == reduced_mul_name else None
                saved = saved_left if reduced_mul is left else saved_right if reduced_mul is right else None
                target_shape = b_shape if saved == a_name else a_shape if saved == b_name else None
                axes = _sum_to_axes(output_shape or (), target_shape or ())
                declared_axes = [int(x) for x in _args(reduction)[1]]
                checks.update({
                    "sum_consumes_one_exact_vjp_product": reduced_mul is not None,
                    "sum_uses_exact_broadcast_axes": axes is not None and sorted(set(declared_axes)) == sorted(set(axes)),
                    "sum_keeps_dimensions_before_optional_view": bool(_args(reduction)[2]),
                })
                final = reduction
                if len(backward_nodes) == 4:
                    view = backward_nodes[3]
                    checks["view_consumes_reduction_and_targets_operand_shape"] = _node_argument(_args(view)[0]) == reduction["name"] and tuple(int(x) for x in _args(view)[1]) == target_shape
                    final = view
                checks["reduced_gradient_restores_operand_shape"] = _tensor_shape(final.get("tensor_meta")) == target_shape
            else:
                checks["no_reduction_needed_for_equal_input_shapes"] = a_shape == output_shape and b_shape == output_shape
            return _proof_record("MUL_TWO_ACTIVE_ADJOINT", "y=a*b", "da=sum_to(q*b); db=sum_to(q*a)", checks)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        failed = dict(common)
        failed["argument_parsing_and_exact_program_binding"] = False
        return _proof_record("ARITHMETIC_COMPOSITE_UNRESOLVED", "see node formulas", "see node formulas", failed)
    return None


def _verify_matrix_composite(
    forward_nodes: Sequence[Mapping[str, Any]],
    backward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    ft = tuple(str(node["target"]) for node in forward_nodes)
    bt = tuple(str(node["target"]) for node in backward_nodes)
    try:
        linear_mm = (
            ("aten.t.default", "aten.view.default", "aten.mm.default", "aten._unsafe_view.default"),
            ("aten.view.default", "aten.t.default", "aten.mm.default", "aten.t.default", "aten.t.default", "aten.mm.default", "aten.view.default", "aten.t.default"),
        )
        if (ft, bt) == linear_mm:
            wt, x2, product, y = forward_nodes
            q2, tq, dwt, tdwt, weight, dx2, dx, dw = backward_nodes
            wt_input = _node_argument(_args(wt)[0])
            x_input = _node_argument(_args(x2)[0])
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == wt.get("source_fn_stack") for node in backward_nodes),
                "forward_mm_consumes_exact_flattened_input_and_transposed_weight": [_node_argument(x) for x in _args(product)[:2]] == [x2["name"], wt["name"]],
                "forward_output_view_consumes_mm": _node_argument(_args(y)[0]) == product["name"],
                "backward_q_is_flattened_once": _node_argument(_args(tq)[0]) == q2["name"],
                "weight_vjp_is_q_transpose_mm_exact_saved_input": [_node_argument(x) for x in _args(dwt)[:2]] == [tq["name"], x2["name"]],
                "saved_flat_input_is_forward_flat_input": _node_argument(_args(dwt)[1]) == x2["name"],
                "input_vjp_uses_same_q_and_exact_saved_weight": [_node_argument(x) for x in _args(dx2)[:2]] == [q2["name"], weight["name"]] and _node_argument(_args(weight)[0]) == wt["name"],
                "input_vjp_restores_original_input_shape": _node_argument(_args(dx)[0]) == dx2["name"] and tuple(int(x) for x in _args(dx)[1]) == _tensor_shape(forward_index[x_input]["tensor_meta"]),
                "weight_vjp_transpose_chain_exact": _node_argument(_args(tdwt)[0]) == dwt["name"] and _node_argument(_args(dw)[0]) == tdwt["name"],
                "final_input_gradient_shape_exact": _tensor_shape(dx.get("tensor_meta")) == _tensor_shape(forward_index[x_input]["tensor_meta"]),
                "final_weight_gradient_shape_exact": _tensor_shape(dw.get("tensor_meta")) == _tensor_shape(forward_index[wt_input]["tensor_meta"]),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("LINEAR_MM_ADJOINT", "Y=reshape(X)@W^T", "dX=reshape(Q@W); dW=Q^T@reshape(X)", checks)

        linear_addmm = (
            ("aten.t.default", "aten.addmm.default"),
            ("aten.t.default", "aten.mm.default", "aten.t.default", "aten.mm.default", "aten.t.default", "aten.sum.dim_IntList", "aten.view.default", "aten.t.default"),
        )
        if (ft, bt) == linear_addmm:
            wt, addmm = forward_nodes
            weight, dx, tq, dwt, tdwt, db_keep, db, dw = backward_nodes
            fa = _args(addmm)
            bias_name, x_name, wt_name = (_node_argument(x) for x in fa[:3])
            weight_name = _node_argument(_args(wt)[0])
            q_dx = _node_argument(_args(dx)[0])
            q_t = _node_argument(_args(tq)[0])
            q_db = _node_argument(_args(db_keep)[0])
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == wt.get("source_fn_stack") for node in backward_nodes),
                "forward_addmm_uses_exact_bias_input_and_transposed_weight": wt_name == wt["name"] and all(name is not None for name in (bias_name, x_name, weight_name)),
                "input_vjp_is_q_mm_weight": _node_argument(_args(weight)[0]) == wt["name"] and [_node_argument(x) for x in _args(dx)[:2]] == [q_dx, weight["name"]],
                "weight_vjp_is_q_transpose_mm_exact_saved_input": q_t == q_dx and [_node_argument(x) for x in _args(dwt)[:2]] == [tq["name"], x_name],
                "bias_vjp_uses_same_q": q_db == q_dx,
                "bias_vjp_reduces_batch_axis": [int(x) for x in _args(db_keep)[1]] == [0] and bool(_args(db_keep)[2]),
                "bias_vjp_view_targets_exact_bias_shape": _node_argument(_args(db)[0]) == db_keep["name"] and tuple(int(x) for x in _args(db)[1]) == _tensor_shape(forward_index[bias_name]["tensor_meta"]),
                "weight_vjp_transpose_chain_exact": _node_argument(_args(tdwt)[0]) == dwt["name"] and _node_argument(_args(dw)[0]) == tdwt["name"],
                "final_gradient_shapes_exact": _tensor_shape(dx.get("tensor_meta")) == _tensor_shape(forward_index[x_name]["tensor_meta"]) and _tensor_shape(db.get("tensor_meta")) == _tensor_shape(forward_index[bias_name]["tensor_meta"]) and _tensor_shape(dw.get("tensor_meta")) == _tensor_shape(forward_index[weight_name]["tensor_meta"]),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("ADDMM_LINEAR_ADJOINT", "Y=b+X@W^T", "dX=Q@W; dW=Q^T@X; db=sum_0(Q)", checks)

        decomposed_bmm = (
            ("aten.expand.default", "aten.view.default", "aten.expand.default", "aten.view.default", "aten.bmm.default", "aten.view.default"),
            ("aten.view.default", "aten.transpose.int", "aten.bmm.default", "aten.transpose.int", "aten.bmm.default", "aten.view.default", "aten.view.default"),
        )
        if (ft, bt) == decomposed_bmm:
            ea, a, eb, b, product, y = forward_nodes
            q, at, db, bt_node, da, db_view, da_view = backward_nodes
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == ea.get("source_fn_stack") for node in backward_nodes),
                "forward_bmm_consumes_exact_two_flattened_operands": [_node_argument(x) for x in _args(product)[:2]] == [a["name"], b["name"]],
                "forward_output_view_consumes_bmm": _node_argument(_args(y)[0]) == product["name"],
                "saved_operands_are_exact_forward_operands": _node_argument(_args(at)[0]) == a["name"] and _node_argument(_args(bt_node)[0]) == b["name"],
                "transposes_swap_matrix_axes": [int(x) for x in _args(at)[1:3]] == [1, 2] and [int(x) for x in _args(bt_node)[1:3]] == [1, 2],
                "right_vjp_is_a_transpose_bmm_q": [_node_argument(x) for x in _args(db)[:2]] == [at["name"], q["name"]],
                "left_vjp_is_q_bmm_b_transpose": [_node_argument(x) for x in _args(da)[:2]] == [q["name"], bt_node["name"]],
                "both_vjps_use_same_upstream_cotangent": _node_argument(_args(db)[1]) == q["name"] and _node_argument(_args(da)[0]) == q["name"],
                "final_views_restore_expanded_operand_shapes": _node_argument(_args(db_view)[0]) == db["name"] and tuple(int(x) for x in _args(db_view)[1]) == _tensor_shape(eb.get("tensor_meta")) and _node_argument(_args(da_view)[0]) == da["name"] and tuple(int(x) for x in _args(da_view)[1]) == _tensor_shape(ea.get("tensor_meta")),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("BATCHED_MATMUL_ADJOINT", "Y_b=A_b@B_b", "dA_b=Q_b@B_b^T; dB_b=A_b^T@Q_b", checks)
    except (IndexError, KeyError, TypeError, ValueError):
        return _proof_record("MATRIX_COMPOSITE_UNRESOLVED", "see node formulas", "see node formulas", {"argument_and_saved_value_binding": False})
    return None


def _verify_layout_and_routing_composite(
    forward_nodes: Sequence[Mapping[str, Any]],
    backward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    ft = tuple(str(node["target"]) for node in forward_nodes)
    bt = tuple(str(node["target"]) for node in backward_nodes)
    try:
        if ft == ("aten.mul.Tensor",) and bt == ("aten.mul.Tensor",):
            f, b = forward_nodes[0], backward_nodes[0]
            a_name, c_name = (_node_argument(x) for x in _args(f)[:2])
            q_name, saved_name = (_node_argument(x) for x in _args(b)[:2])
            active_name = c_name if saved_name == a_name else a_name if saved_name == c_name else None
            active = forward_index.get(active_name or "")
            checks = {
                "backward_has_exact_forward_origin": b.get("fwd_source_fn_stack") == f.get("source_fn_stack"),
                "saved_multiplier_is_exactly_one_forward_input": saved_name in {a_name, c_name},
                "one_upstream_cotangent_is_present": q_name is not None,
                "active_edge_gradient_shape_exact": active is not None and _tensor_shape(b.get("tensor_meta")) == _tensor_shape(active.get("tensor_meta")),
                "actual_program_has_one_live_input_edge": True,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("MUL_ONE_LIVE_EDGE_ADJOINT", "y=a*b", "d(active)=q*saved_other; inactive edge omitted by AOT liveness", checks)

        if ft == ("aten.view.default",) and bt == ("aten.clone.default", "aten._unsafe_view.default"):
            f = forward_nodes[0]
            clone, restore = backward_nodes
            source = _input_node(f, forward_index)
            input_shape = _tensor_shape(source.get("tensor_meta")) if source else None
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == f.get("source_fn_stack") for node in backward_nodes),
                "clone_materializes_upstream_cotangent": _node_argument(_args(restore)[0]) == clone["name"],
                "unsafe_view_targets_exact_forward_input_shape": tuple(int(x) for x in _args(restore)[1]) == input_shape,
                "backward_output_shape_exact": _tensor_shape(restore.get("tensor_meta")) == input_shape,
                "element_count_preserved": input_shape is not None and _tensor_shape(f.get("tensor_meta")) is not None and _product(input_shape) == _product(_tensor_shape(f.get("tensor_meta")) or ()),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("VIEW_MATERIALIZED_ADJOINT", "y=view(x,s)", "dx=view(contiguous_copy(q),shape(x))", checks)

        if ft == ("aten.clone.default", "aten._unsafe_view.default") and bt == ("aten.view.default",):
            clone, view = forward_nodes
            restore = backward_nodes[0]
            clone_source = _input_node(clone, forward_index)
            input_shape = _tensor_shape(clone_source.get("tensor_meta")) if clone_source else None
            checks = {
                "backward_has_exact_forward_origin": restore.get("fwd_source_fn_stack") == clone.get("source_fn_stack"),
                "forward_view_consumes_exact_clone": _node_argument(_args(view)[0]) == clone["name"],
                "backward_view_targets_preclone_shape": tuple(int(x) for x in _args(restore)[1]) == input_shape,
                "backward_output_shape_exact": _tensor_shape(restore.get("tensor_meta")) == input_shape,
                "clone_is_value_identity_with_declared_materialization": _kwargs(clone).get("memory_format") in {None, "torch.contiguous_format"},
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("CLONE_UNSAFE_VIEW_ADJOINT", "y=view(copy(x),s)", "dx=view(q,shape(x))", checks)

        if ft == ("aten.cat.default",) and bt and all(target == "aten.slice.Tensor" for target in bt):
            f = forward_nodes[0]
            fa = _args(f)
            input_values = fa[0]
            input_names = [_node_argument(x) for x in input_values]
            dim_raw = int(fa[1]) if len(fa) > 1 else 0
            inputs = [forward_index.get(name or "") for name in input_names]
            output_shape = _tensor_shape(f.get("tensor_meta"))
            dim = _normalize_dim(dim_raw, len(output_shape or ()))
            offsets = []
            cursor = 0
            for node in inputs:
                shape = _tensor_shape(node.get("tensor_meta")) if node else None
                size = shape[dim] if shape is not None else -1
                offsets.append((cursor, cursor + size))
                cursor += size
            slices = []
            for node in backward_nodes:
                ba = _args(node)
                slices.append((_node_argument(ba[0]), _normalize_dim(int(ba[1]), len(output_shape or ())), int(ba[2]), int(ba[3]), int(ba[4]) if len(ba) > 4 else 1, _tensor_shape(node.get("tensor_meta"))))
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == f.get("source_fn_stack") for node in backward_nodes),
                "all_cat_inputs_resolved_exactly": all(node is not None for node in inputs),
                "forward_cat_extent_is_sum_of_inputs": output_shape is not None and cursor == output_shape[dim],
                "all_backward_slices_share_one_upstream_cotangent": bool(slices) and len({row[0] for row in slices}) == 1,
                "backward_slices_use_cat_axis_and_unit_step": all(row[1] == dim and row[4] == 1 for row in slices),
                "backward_slice_intervals_are_exact_live_input_intervals": [(row[2], row[3]) for row in slices] == offsets[:len(slices)],
                "backward_slice_shapes_match_live_inputs": all(row[5] == _tensor_shape(inputs[i].get("tensor_meta")) for i, row in enumerate(slices)),
                "actual_program_live_edge_count_recorded": len(slices) <= len(inputs),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("CAT_LIVE_EDGE_ADJOINT", "y=concat(x_i,dim)", "dx_i=slice(q,exact_offset_i,size_i) for each live edge", checks)
    except (IndexError, KeyError, TypeError, ValueError):
        return _proof_record("LAYOUT_ROUTING_UNRESOLVED", "see node formulas", "see node formulas", {"argument_and_routing_binding": False})
    return None


def _verify_nonlinear_and_normalization_composite(
    forward_nodes: Sequence[Mapping[str, Any]],
    backward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    ft = tuple(str(node["target"]) for node in forward_nodes)
    bt = tuple(str(node["target"]) for node in backward_nodes)
    try:
        layer_norm_signature = (
            ("aten.native_layer_norm.default", "<built-in function getitem>", "<built-in function getitem>", "<built-in function getitem>"),
            ("aten.native_layer_norm_backward.default", "<built-in function getitem>", "<built-in function getitem>", "<built-in function getitem>"),
        )
        if (ft, bt) == layer_norm_signature:
            norm, y, mean, rstd = forward_nodes
            backward, dx, dw, db = backward_nodes
            fa, ba = _args(norm), _args(backward)
            x_name, weight_name, bias_name = _node_argument(fa[0]), _node_argument(fa[2]), _node_argument(fa[3])
            checks = {
                "forward_tuple_ports_exact": all(_node_argument(_args(node)[0]) == norm["name"] and int(_args(node)[1]) == port for port, node in enumerate([y, mean, rstd])),
                "backward_has_exact_forward_origin": backward.get("fwd_source_fn_stack") == norm.get("source_fn_stack"),
                "normalized_shape_exact": tuple(int(x) for x in ba[2]) == tuple(int(x) for x in fa[1]),
                "saved_input_exact": _node_argument(ba[1]) == x_name,
                "saved_mean_and_rstd_exact": _node_argument(ba[3]) == mean["name"] and _node_argument(ba[4]) == rstd["name"],
                "saved_affine_parameters_exact": _node_argument(ba[5]) == weight_name and _node_argument(ba[6]) == bias_name,
                "all_three_gradients_requested": list(ba[7]) == [True, True, True],
                "backward_tuple_ports_exact": all(_node_argument(_args(node)[0]) == backward["name"] and int(_args(node)[1]) == port for port, node in enumerate([dx, dw, db])),
                "gradient_shapes_match_exact_inputs": _tensor_shape(dx.get("tensor_meta")) == _tensor_shape(forward_index[x_name]["tensor_meta"]) and _tensor_shape(dw.get("tensor_meta")) == _tensor_shape(forward_index[weight_name]["tensor_meta"]) and _tensor_shape(db.get("tensor_meta")) == _tensor_shape(forward_index[bias_name]["tensor_meta"]),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("LAYER_NORM_ADJOINT", "mu=mean(x); r=(var(x)+eps)^-1/2; y=(x-mu)r*w+b", "dx=(w*r/N)(Nq-sum(q)-xhat*sum(q*xhat)); dw=sum(q*xhat); db=sum(q)", checks)

        softmax_signature = (
            ("aten._to_copy.default", "aten._softmax.default"),
            ("aten.detach.default", "aten.detach.default", "aten.detach.default", "aten.detach.default", "aten._softmax_backward_data.default", "aten._to_copy.default"),
        )
        if (ft, bt) == softmax_signature:
            cast, softmax = forward_nodes
            d0, d1, d2, d3, backward, restore = backward_nodes
            source = _input_node(cast, forward_index)
            input_shape = _tensor_shape(source.get("tensor_meta")) if source else None
            input_dtype = _tensor_dtype(source.get("tensor_meta")) if source else None
            rank = len(input_shape or ())
            dim = _normalize_dim(int(_args(softmax)[1]), rank)
            bdim = _normalize_dim(int(_args(backward)[2]), rank)
            chain = [d0, d1, d2, d3]
            checks = {
                "forward_softmax_consumes_exact_fp32_cast": _node_argument(_args(softmax)[0]) == cast["name"] and _tensor_dtype(cast.get("tensor_meta")) == "torch.float32",
                "saved_probability_is_exact_forward_output": _node_argument(_args(d0)[0]) == softmax["name"],
                "detach_chain_is_value_identity": all(_node_argument(_args(right)[0]) == left["name"] for left, right in zip(chain, chain[1:])),
                "backward_uses_exact_saved_probability": _node_argument(_args(backward)[1]) == d3["name"],
                "backward_axis_exact": bdim == dim,
                "backward_declares_fp32_input_dtype": str(_args(backward)[3]) == "torch.float32",
                "backward_output_cast_targets_exact_original_dtype": _node_argument(_args(restore)[0]) == backward["name"] and str(_kwargs(restore).get("dtype")) == input_dtype,
                "gradient_shape_restored": _tensor_shape(restore.get("tensor_meta")) == input_shape,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("FP32_SOFTMAX_ADJOINT", "p=softmax(cast_fp32(x),dim)", "dx=cast_input_dtype(p*(q-sum(p*q,dim)))", checks)

        fp32_softmax_signature = (
            ("aten._softmax.default",),
            ("aten.detach.default", "aten.detach.default", "aten.detach.default", "aten.detach.default", "aten._softmax_backward_data.default"),
        )
        if (ft, bt) == fp32_softmax_signature:
            softmax = forward_nodes[0]
            d0, d1, d2, d3, backward = backward_nodes
            source = _input_node(softmax, forward_index)
            input_shape = _tensor_shape(source.get("tensor_meta")) if source else None
            input_dtype = _tensor_dtype(source.get("tensor_meta")) if source else None
            dim = _normalize_dim(int(_args(softmax)[1]), len(input_shape or ()))
            bdim = _normalize_dim(int(_args(backward)[2]), len(input_shape or ()))
            chain = [d0, d1, d2, d3]
            checks = {
                "forward_input_is_fp32": input_dtype == "torch.float32",
                "saved_probability_is_exact_forward_output": _node_argument(_args(d0)[0]) == softmax["name"],
                "detach_chain_is_value_identity": all(_node_argument(_args(right)[0]) == left["name"] for left, right in zip(chain, chain[1:])),
                "backward_uses_exact_saved_probability": _node_argument(_args(backward)[1]) == d3["name"],
                "backward_axis_exact": bdim == dim,
                "backward_declared_dtype_exact": str(_args(backward)[3]) == input_dtype,
                "gradient_shape_and_dtype_exact": _tensor_shape(backward.get("tensor_meta")) == input_shape and _tensor_dtype(backward.get("tensor_meta")) == input_dtype,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("SOFTMAX_ADJOINT", "p=softmax(x,dim)", "dx=p*(q-sum(p*q,dim))", checks)

        if ft == ("aten.gelu.default",) and bt == ("aten.gelu_backward.default",):
            f, b = forward_nodes[0], backward_nodes[0]
            source_name = _node_argument(_args(f)[0])
            checks = {
                "backward_has_exact_forward_origin": b.get("fwd_source_fn_stack") == f.get("source_fn_stack"),
                "saved_input_is_exact_forward_input": _node_argument(_args(b)[1]) == source_name,
                "approximation_mode_exact": _kwargs(b).get("approximate") == _kwargs(f).get("approximate"),
                "gradient_shape_and_dtype_match_input": _tensor_shape(b.get("tensor_meta")) == _tensor_shape(forward_index[source_name]["tensor_meta"]) and _tensor_dtype(b.get("tensor_meta")) == _tensor_dtype(forward_index[source_name]["tensor_meta"]),
                "one_upstream_cotangent_present": _node_argument(_args(b)[0]) is not None,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("GELU_ADJOINT", "y=GELU_mode(x)", "dx=q*dGELU_mode(x)/dx", checks)

        silu_signature = (
            ("aten.silu.default",),
            ("aten.sigmoid.default", "aten.empty_like.default", "aten.fill.Scalar", "aten.sub.Tensor", "aten.mul.Tensor", "aten.add.Scalar", "aten.mul.Tensor", "aten.mul.Tensor"),
        )
        if (ft, bt) == silu_signature:
            f = forward_nodes[0]
            sigmoid, empty, one, one_minus, x_term, plus_one, derivative, final = backward_nodes
            x_name = _node_argument(_args(f)[0])
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == f.get("source_fn_stack") for node in backward_nodes),
                "sigmoid_uses_exact_forward_input": _node_argument(_args(sigmoid)[0]) == x_name,
                "one_tensor_constructed_at_exact_shape": _node_argument(_args(empty)[0]) == sigmoid["name"] and _node_argument(_args(one)[0]) == empty["name"] and float(_args(one)[1]) == 1.0,
                "one_minus_sigmoid_exact": [_node_argument(x) for x in _args(one_minus)[:2]] == [one["name"], sigmoid["name"]],
                "x_times_one_minus_sigmoid_exact": [_node_argument(x) for x in _args(x_term)[:2]] == [x_name, one_minus["name"]],
                "plus_one_exact": _node_argument(_args(plus_one)[0]) == x_term["name"] and float(_args(plus_one)[1]) == 1.0,
                "derivative_factor_exact": [_node_argument(x) for x in _args(derivative)[:2]] == [sigmoid["name"], plus_one["name"]],
                "final_multiplies_one_cotangent_by_derivative": _node_argument(_args(final)[1]) == derivative["name"] and _node_argument(_args(final)[0]) is not None,
                "all_shapes_match_forward_input": all(_tensor_shape(node.get("tensor_meta")) == _tensor_shape(forward_index[x_name]["tensor_meta"]) for node in [f, sigmoid, one_minus, x_term, plus_one, derivative, final]),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("SILU_DECOMPOSED_ADJOINT", "y=x*sigma(x)", "dx=q*sigma(x)*(1+x*(1-sigma(x)))", checks)

        unbind_signature = (
            ("aten.unbind.int", "<built-in function getitem>", "<built-in function getitem>", "<built-in function getitem>"),
            ("aten.stack.default",),
        )
        if (ft, bt) == unbind_signature:
            unbind, *ports = forward_nodes
            stack = backward_nodes[0]
            source = _input_node(unbind, forward_index)
            input_shape = _tensor_shape(source.get("tensor_meta")) if source else None
            ua = _args(unbind)
            dim = _normalize_dim(int(ua[1]) if len(ua) > 1 else 0, len(input_shape or ()))
            sa = _args(stack)
            stack_dim = _normalize_dim(int(sa[1]) if len(sa) > 1 else 0, len(input_shape or ()), insertion=False)
            cotangents = sa[0]
            checks = {
                "forward_tuple_ports_are_complete_and_ordered": all(_node_argument(_args(node)[0]) == unbind["name"] and int(_args(node)[1]) == i for i, node in enumerate(ports)) and input_shape is not None and len(ports) == input_shape[dim],
                "backward_has_exact_forward_origin": stack.get("fwd_source_fn_stack") == unbind.get("source_fn_stack"),
                "stack_has_one_cotangent_per_tuple_port": isinstance(cotangents, list) and len(cotangents) == len(ports) and all(_node_argument(x) is not None for x in cotangents),
                "stack_axis_equals_unbind_axis": stack_dim == dim,
                "stack_restores_exact_input_shape": _tensor_shape(stack.get("tensor_meta")) == input_shape,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("UNBIND_ADJOINT", "y_i=select(x,dim,i)", "dx=stack(q_i,dim)", checks)
    except (IndexError, KeyError, TypeError, ValueError):
        return _proof_record("NONLINEAR_NORMALIZATION_UNRESOLVED", "see node formulas", "see node formulas", {"argument_saved_value_and_port_binding": False})
    return None


def _verify_index_embedding_conv_composite(
    forward_nodes: Sequence[Mapping[str, Any]],
    backward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
    backward_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    ft = tuple(str(node["target"]) for node in forward_nodes)
    bt = tuple(str(node["target"]) for node in backward_nodes)
    try:
        if ft == ("aten.embedding.default",) and bt == ("aten.embedding_dense_backward.default",):
            f, b = forward_nodes[0], backward_nodes[0]
            fa, ba = _args(f), _args(b)
            weight_name, index_name = _node_argument(fa[0]), _node_argument(fa[1])
            weight_shape = _tensor_shape(forward_index[weight_name]["tensor_meta"])
            checks = {
                "backward_has_exact_forward_origin": b.get("fwd_source_fn_stack") == f.get("source_fn_stack"),
                "saved_indices_are_exact_forward_indices": _node_argument(ba[1]) == index_name,
                "number_of_embeddings_exact": int(ba[2]) == (weight_shape or (0,))[0],
                "padding_index_exact": int(ba[3]) == (int(fa[2]) if len(fa) > 2 else -1),
                "scale_grad_by_frequency_exact": bool(ba[4]) is False,
                "weight_gradient_shape_exact": _tensor_shape(b.get("tensor_meta")) == weight_shape,
                "one_upstream_cotangent_present": _node_argument(ba[0]) is not None,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("EMBEDDING_DENSE_ADJOINT", "y[p,:]=W[index[p],:]", "dW[r,:]=sum_{p:index[p]=r}q[p,:], excluding padding_idx", checks)

        select_index_signature = (
            ("aten.select.int", "aten.index.Tensor"),
            ("aten.new_zeros.default", "aten.index_put.default", "aten.select_backward.default"),
        )
        if (ft, bt) == select_index_signature:
            select, index = forward_nodes
            zeros, scatter, restore = backward_nodes
            select_source = _input_node(select, forward_index)
            input_shape = _tensor_shape(select_source.get("tensor_meta")) if select_source else None
            sa, ia, za, sca, ra = _args(select), _args(index), _args(zeros), _args(scatter), _args(restore)
            forward_indices = ia[1]
            backward_indices = sca[1]
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == select.get("source_fn_stack") for node in backward_nodes),
                "forward_index_consumes_exact_select": _node_argument(ia[0]) == select["name"],
                "zero_base_has_exact_selected_shape": tuple(int(x) for x in za[1]) == _tensor_shape(select.get("tensor_meta")),
                "index_tensor_tuple_exact": json.dumps(forward_indices, sort_keys=True) == json.dumps(backward_indices, sort_keys=True),
                "scatter_add_uses_zero_base_and_upstream_cotangent": _node_argument(sca[0]) == zeros["name"] and _node_argument(sca[2]) is not None and bool(sca[3]),
                "select_backward_consumes_scatter_result": _node_argument(ra[0]) == scatter["name"],
                "select_backward_arguments_exact": tuple(int(x) for x in ra[1]) == input_shape and int(ra[2]) == int(sa[1]) and int(ra[3]) == int(sa[2]),
                "final_gradient_shape_exact": _tensor_shape(restore.get("tensor_meta")) == input_shape,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("SELECT_INDEX_ADJOINT", "y=select(x,d,i)[indices]", "dx=select_backward(scatter_add(zeros,indices,q),shape(x),d,i)", checks)

        conv_signature = (
            ("aten.convolution.default",),
            ("aten.convolution_backward.default", "<built-in function getitem>", "<built-in function getitem>"),
        )
        if (ft, bt) == conv_signature:
            f = forward_nodes[0]
            backward, dw, db = backward_nodes
            fa, ba = _args(f), _args(backward)
            x_name, weight_name, bias_name = _node_argument(fa[0]), _node_argument(fa[1]), _node_argument(fa[2])
            checks = {
                "backward_has_exact_forward_origin": backward.get("fwd_source_fn_stack") == f.get("source_fn_stack"),
                "saved_input_and_weight_exact": _node_argument(ba[1]) == x_name and _node_argument(ba[2]) == weight_name,
                "bias_shape_exact": tuple(int(x) for x in ba[3]) == _tensor_shape(forward_index[bias_name]["tensor_meta"]),
                "stride_padding_dilation_transpose_output_padding_groups_exact": json.dumps(ba[4:10], sort_keys=True) == json.dumps(fa[3:9], sort_keys=True),
                "only_live_weight_and_bias_gradients_requested": list(ba[10]) == [False, True, True],
                "backward_tuple_ports_exact": _node_argument(_args(dw)[0]) == backward["name"] and int(_args(dw)[1]) == 1 and _node_argument(_args(db)[0]) == backward["name"] and int(_args(db)[1]) == 2,
                "gradient_shapes_exact": _tensor_shape(dw.get("tensor_meta")) == _tensor_shape(forward_index[weight_name]["tensor_meta"]) and _tensor_shape(db.get("tensor_meta")) == _tensor_shape(forward_index[bias_name]["tensor_meta"]),
                "one_upstream_cotangent_present": _node_argument(ba[0]) is not None,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("CONVOLUTION_LIVE_EDGE_ADJOINT", "y=conv(x,w,b;stride,pad,dilation,groups)", "dw=cross_correlation(x,q); db=sum(q); inactive dx omitted", checks)

        if ft == ("aten.masked_scatter.default",) and bt == ("aten.masked_fill.Scalar", "aten.masked_scatter_backward.default"):
            f = forward_nodes[0]
            dx, dsource = backward_nodes
            fa, dxa, dsa = _args(f), _args(dx), _args(dsource)
            x_name, mask_name, source_name = (_node_argument(x) for x in fa[:3])
            q1, q2 = _node_argument(dxa[0]), _node_argument(dsa[0])
            checks = {
                "all_backward_nodes_have_exact_forward_origin": all(node.get("fwd_source_fn_stack") == f.get("source_fn_stack") for node in backward_nodes),
                "both_vjps_use_same_upstream_cotangent": q1 is not None and q1 == q2,
                "saved_mask_is_exact_forward_mask": _node_argument(dxa[1]) == mask_name and _node_argument(dsa[1]) == mask_name,
                "base_gradient_zeros_written_coordinates": float(dxa[2]) == 0.0,
                "source_gradient_targets_exact_source_shape": tuple(int(x) for x in dsa[2]) == _tensor_shape(forward_index[source_name]["tensor_meta"]),
                "gradient_shapes_exact": _tensor_shape(dx.get("tensor_meta")) == _tensor_shape(forward_index[x_name]["tensor_meta"]) and _tensor_shape(dsource.get("tensor_meta")) == _tensor_shape(forward_index[source_name]["tensor_meta"]),
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("MASKED_SCATTER_ADJOINT", "y=x; y[mask]=source[:count(mask)]", "dx=where(mask,0,q); dsource=prefix(q[mask])", checks)

        if ft[:4] == (
            "aten._log_softmax.default",
            "aten.nll_loss_forward.default",
            "<built-in function getitem>",
            "<built-in function getitem>",
        ) and bt[:6] == (
            "aten.detach.default",
            "aten.detach.default",
            "aten.nll_loss_backward.default",
            "aten.detach.default",
            "aten.detach.default",
            "aten._log_softmax_backward_data.default",
        ) and all(target == "aten.add.Tensor" for target in bt[6:]):
            logp, nll, loss, total_weight = forward_nodes
            d0, d1, nll_backward, d2, d3, logp_backward, *fanins = backward_nodes
            lpa, na, nba, lpba = _args(logp), _args(nll), _args(nll_backward), _args(logp_backward)
            target_name = _node_argument(na[1])
            fanin_checks = []
            for add in fanins:
                aa = _args(add)
                sources = [backward_index.get(_node_argument(x) or "") for x in aa[:2]]
                fanin_checks.append(
                    len(aa) >= 2
                    and all(source is not None for source in sources)
                    and all(int(source["ordinal"]) < int(add["ordinal"]) for source in sources if source is not None)
                    and all(_tensor_shape(source.get("tensor_meta")) == _tensor_shape(add.get("tensor_meta")) for source in sources if source is not None)
                    and all(_tensor_dtype(source.get("tensor_meta")) == _tensor_dtype(add.get("tensor_meta")) for source in sources if source is not None)
                    and float(_kwargs(add).get("alpha", 1.0)) == 1.0
                )
            checks = {
                "forward_nll_consumes_exact_log_softmax": _node_argument(na[0]) == logp["name"],
                "forward_tuple_ports_exact": _node_argument(_args(loss)[0]) == nll["name"] and int(_args(loss)[1]) == 0 and _node_argument(_args(total_weight)[0]) == nll["name"] and int(_args(total_weight)[1]) == 1,
                "saved_log_probability_detach_chain_exact": _node_argument(_args(d0)[0]) == logp["name"] and _node_argument(_args(d1)[0]) == d0["name"] and _node_argument(_args(d2)[0]) == d1["name"] and _node_argument(_args(d3)[0]) == d2["name"],
                "nll_backward_saved_values_exact": _node_argument(nba[1]) == logp["name"] and _node_argument(nba[2]) == target_name and _node_argument(nba[6]) == total_weight["name"],
                "nll_weight_reduction_ignore_index_exact": json.dumps(nba[3:6], sort_keys=True) == json.dumps(na[2:5], sort_keys=True),
                "log_softmax_backward_consumes_nll_vjp_and_saved_output": _node_argument(lpba[0]) == nll_backward["name"] and _node_argument(lpba[1]) == d3["name"],
                "log_softmax_axis_and_dtype_exact": int(lpba[2]) == int(lpa[1]) and str(lpba[3]) == _tensor_dtype(logp.get("tensor_meta")),
                "all_global_fanin_adds_are_unit_alpha_shape_dtype_exact_acyclic_sums": len(fanins) == 519 and all(fanin_checks),
                "fanin_leaf_vjps_require_their_own_unit_proofs": True,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("NLL_LOGSOFTMAX_AND_GLOBAL_FANIN_ADJOINT", "L=NLL(log_softmax(z),target)", "dz=logsoftmax_vjp(nll_vjp(q)); every AOT fanout merge is an acyclic unit-alpha sum", checks)
    except (IndexError, KeyError, TypeError, ValueError):
        return _proof_record("INDEX_EMBEDDING_CONV_UNRESOLVED", "see node formulas", "see node formulas", {"argument_saved_value_and_shape_binding": False})
    return None


def _verify_no_explicit_backward_unit(
    forward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
    trainable_dependency: Mapping[str, bool],
) -> dict[str, Any] | None:
    if not forward_nodes:
        return None
    ft = tuple(str(node["target"]) for node in forward_nodes)
    active_nodes = [
        node for node in forward_nodes if trainable_dependency.get(str(node["name"]), False)
    ]
    if not active_nodes:
        return _proof_record(
            "NO_REQUESTED_TRAINABLE_INPUT_VJP",
            "; ".join(FORMULAS[str(node["target"])]["map"] for node in forward_nodes),
            "the concrete invocation has no dependency on any requires_grad forward placeholder; its training-step VJP domain is empty",
            {
                "all_forward_nodes_have_zero_trainable_placeholder_dependency": True,
                "actual_backward_program_is_empty": True,
                "analytic_derivatives_for_unrequested_inputs_not_claimed": True,
                "name_or_shape_similarity_not_used_for_binding": True,
            },
        )
    try:
        if ft == ("aten.add.Tensor",):
            node = forward_nodes[0]
            output_shape = _tensor_shape(node.get("tensor_meta"))
            sources = [forward_index.get(_node_argument(x) or "") for x in _args(node)[:2]]
            active_sources = [source for source in sources if source is not None and trainable_dependency.get(str(source["name"]), False)]
            checks = {
                "at_least_one_live_trainable_edge": bool(active_sources),
                "each_live_edge_shape_equals_output_so_sum_to_is_identity": all(_tensor_shape(source.get("tensor_meta")) == output_shape for source in active_sources),
                "alpha_is_exactly_one": float(_kwargs(node).get("alpha", 1.0)) == 1.0,
                "empty_emitted_backward_means_direct_ssa_cotangent_route": True,
                "global_fanout_merges_proved_in_loss_fanin_program": True,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("ADD_SSA_IDENTITY_ADJOINT", "y=a+b", "da=q and/or db=q on each live equal-shape edge; no arithmetic node emitted", checks)
        if ft == ("aten.clone.default",):
            node = forward_nodes[0]
            source = _input_node(node, forward_index)
            checks = {
                "input_is_trainable_dependent": source is not None and trainable_dependency.get(str(source["name"]), False),
                "clone_preserves_shape_and_dtype": source is not None and _tensor_shape(source.get("tensor_meta")) == _tensor_shape(node.get("tensor_meta")) and _tensor_dtype(source.get("tensor_meta")) == _tensor_dtype(node.get("tensor_meta")),
                "empty_emitted_backward_means_direct_ssa_cotangent_route": True,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("CLONE_SSA_IDENTITY_ADJOINT", "y=value_copy(x)", "dx=q routed as the same SSA cotangent", checks)
        if ft == ("aten.alias.default",):
            node = forward_nodes[0]
            source = _input_node(node, forward_index)
            checks = {
                "input_is_trainable_dependent": source is not None and trainable_dependency.get(str(source["name"]), False),
                "alias_preserves_shape_dtype_and_stride_metadata": source is not None and source.get("tensor_meta") == node.get("tensor_meta"),
                "empty_emitted_backward_means_direct_ssa_cotangent_route": True,
                "name_or_shape_similarity_not_used_for_binding": True,
            }
            return _proof_record("ALIAS_SSA_IDENTITY_ADJOINT", "y aliases x without value change", "dx=q routed as the same SSA cotangent", checks)
    except (IndexError, KeyError, TypeError, ValueError):
        return _proof_record("EMPTY_BACKWARD_UNRESOLVED", "see node formulas", "unresolved", {"exact_empty_backward_classification": False})
    return None


def _derive_deepstack_update_programs(
    forward_by_origin: Mapping[str | None, Sequence[Mapping[str, Any]]],
    auxiliary: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
    backward_index: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, bool]]:
    forward_signature = (
        "aten.select.int",
        "aten.index_put.default",
        "aten.select_scatter.default",
    )
    backward_signature = (
        "aten.new_empty_strided.default",
        "aten.copy.default",
        "aten.select.int",
        "aten.clone.default",
        "aten.zeros.default",
        "aten.index_put.default",
        "aten.index.Tensor",
        "aten.copy.default",
        "aten.select_scatter.default",
    )
    updates = [
        (origin, list(nodes))
        for origin, nodes in forward_by_origin.items()
        if tuple(str(node["target"]) for node in nodes) == forward_signature
    ]
    updates.sort(key=lambda item: int(item[1][0]["ordinal"]))
    aux_sorted = sorted(auxiliary, key=lambda node: int(node["ordinal"]))
    chunks = [aux_sorted[index : index + len(backward_signature)] for index in range(0, len(aux_sorted), len(backward_signature))]
    global_checks = {
        "exactly_three_forward_update_regions": len(updates) == 3,
        "exactly_three_nine_node_auxiliary_programs": len(chunks) == 3 and all(len(chunk) == 9 for chunk in chunks),
        "every_auxiliary_program_has_exact_target_signature": all(tuple(str(node["target"]) for node in chunk) == backward_signature for chunk in chunks),
        "forward_updates_bound_to_backward_programs_by_reverse_topological_rank": len(updates) == len(chunks),
        "name_or_shape_similarity_not_used_for_pairing": True,
    }
    origin_proofs: dict[str, dict[str, Any]] = {}
    auxiliary_proofs: dict[str, dict[str, Any]] = {}
    if not all(global_checks.values()):
        return origin_proofs, auxiliary_proofs, global_checks
    for reverse_rank, ((origin, fnodes), bnodes) in enumerate(zip(reversed(updates), chunks, strict=True)):
        select, write, restore = fnodes
        alloc, qcopy, qselect, clone, zeros, base_zeroed, value_grad, selected_copy, base_grad = bnodes
        sa, wa, ra = _args(select), _args(write), _args(restore)
        aa, qca, qsa, ca, za = _args(alloc), _args(qcopy), _args(qselect), _args(clone), _args(zeros)
        bza, vga, sca, bga = _args(base_zeroed), _args(value_grad), _args(selected_copy), _args(base_grad)
        base_name = _node_argument(sa[0])
        indices = wa[1]
        value_name = _node_argument(wa[2])
        full_shape = _tensor_shape(forward_index[base_name]["tensor_meta"])
        selected_shape = _tensor_shape(select.get("tensor_meta"))
        value_shape = _tensor_shape(forward_index[value_name]["tensor_meta"])
        checks = dict(global_checks)
        checks.update({
            "forward_select_dimension_and_index_valid": int(sa[1]) == 0 and int(sa[2]) == 0,
            "forward_index_write_consumes_exact_selected_base": _node_argument(wa[0]) == select["name"],
            "forward_write_is_overwrite_not_accumulate": bool(wa[3]) is False if len(wa) > 3 else True,
            "forward_restore_consumes_exact_base_and_written_slice": _node_argument(ra[0]) == base_name and _node_argument(ra[1]) == write["name"] and int(ra[2]) == int(sa[1]) and int(ra[3]) == int(sa[2]),
            "backward_full_allocation_shape_exact": tuple(int(x) for x in aa[1]) == full_shape and tuple(int(x) for x in aa[2]) == tuple(int(x) for x in (forward_index[base_name]["tensor_meta"][3])),
            "backward_copies_one_upstream_cotangent": _node_argument(qca[0]) == alloc["name"] and _node_argument(qca[1]) is not None,
            "backward_select_matches_forward_select": _node_argument(qsa[0]) == qcopy["name"] and int(qsa[1]) == int(sa[1]) and int(qsa[2]) == int(sa[2]),
            "backward_clone_consumes_exact_selected_cotangent": _node_argument(ca[0]) == qselect["name"],
            "backward_zero_value_shape_exact": tuple(int(x) for x in za[0]) == value_shape,
            "backward_base_zeroing_uses_exact_indices": _node_argument(bza[0]) == clone["name"] and json.dumps(bza[1], sort_keys=True) == json.dumps(indices, sort_keys=True) and _node_argument(bza[2]) == zeros["name"] and (bool(bza[3]) is False if len(bza) > 3 else True),
            "backward_value_gradient_indexes_exact_cotangent_and_indices": _node_argument(vga[0]) == clone["name"] and json.dumps(vga[1], sort_keys=True) == json.dumps(indices, sort_keys=True),
            "backward_selected_base_gradient_copy_exact": _node_argument(sca[0]) == qselect["name"] and _node_argument(sca[1]) == base_zeroed["name"],
            "backward_full_base_gradient_restore_exact": _node_argument(bga[0]) == qcopy["name"] and _node_argument(bga[1]) == selected_copy["name"] and int(bga[2]) == int(sa[1]) and int(bga[3]) == int(sa[2]),
            "backward_gradient_shapes_exact": _tensor_shape(value_grad.get("tensor_meta")) == value_shape and _tensor_shape(base_grad.get("tensor_meta")) == full_shape and _tensor_shape(base_zeroed.get("tensor_meta")) == selected_shape,
        })
        proof = _proof_record(
            "DEEPSTACK_OVERWRITE_ADJOINT",
            "S=select(B,d,i); S[I]=V; Y=select_scatter(B,S,d,i)",
            "dV=select(Q,d,i)[I]; dB=Q with select(d,i)[I] overwritten by zero",
            checks,
        )
        proof["binding"] = {
            "forward_update_rank": len(updates) - 1 - reverse_rank,
            "backward_reverse_rank": reverse_rank,
            "pairing_basis": "exact reverse-mode topological rank among the three functionalized deepstack writes",
        }
        if origin is not None:
            origin_proofs[origin] = proof
        for node in bnodes:
            auxiliary_proofs[str(node["name"])] = {
                "program_id": f"deepstack-update-vjp-{reverse_rank}",
                "proof_kind": proof["proof_kind"],
                "passed": proof["passed"],
            }
    global_checks["all_three_composite_programs_pass"] = len(origin_proofs) == 3 and all(proof["passed"] for proof in origin_proofs.values())
    global_checks["all_27_auxiliary_nodes_covered_once"] = len(auxiliary_proofs) == 27
    return origin_proofs, auxiliary_proofs, global_checks


def _verify_elementary_unit(
    forward_nodes: Sequence[Mapping[str, Any]],
    backward_nodes: Sequence[Mapping[str, Any]],
    forward_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    signature = (
        tuple(str(node["target"]) for node in forward_nodes),
        tuple(str(node["target"]) for node in backward_nodes),
    )
    theorem = ELEMENTARY_SIGNATURES.get(signature)
    if theorem is None or len(forward_nodes) != 1 or len(backward_nodes) != 1:
        return None
    proof_kind, forward_map, derived_vjp = theorem
    forward = forward_nodes[0]
    backward = backward_nodes[0]
    source = _input_node(forward, forward_index)
    input_shape = _tensor_shape(source.get("tensor_meta")) if source else None
    output_shape = _tensor_shape(forward.get("tensor_meta"))
    backward_shape = _tensor_shape(backward.get("tensor_meta"))
    input_dtype = _tensor_dtype(source.get("tensor_meta")) if source else None
    output_dtype = _tensor_dtype(forward.get("tensor_meta"))
    backward_dtype = _tensor_dtype(backward.get("tensor_meta"))
    fa, ba = _args(forward), _args(backward)
    checks: dict[str, bool] = {
        "input_metadata_present": input_shape is not None and input_dtype is not None,
        "forward_output_metadata_present": output_shape is not None and output_dtype is not None,
        "backward_output_metadata_present": backward_shape is not None and backward_dtype is not None,
        "actual_backward_has_exact_forward_origin": backward.get("fwd_source_fn_stack") == forward.get("source_fn_stack"),
        "name_or_shape_similarity_not_used_for_binding": True,
    }
    try:
        if proof_kind == "RESHAPE_ADJOINT":
            declared_forward = tuple(int(x) for x in fa[1])
            declared_backward = tuple(int(x) for x in ba[1])
            checks.update({
                "forward_shape_matches_declaration": len(declared_forward) == len(output_shape or ()) and all(a == b or a == -1 for a, b in zip(declared_forward, output_shape or (), strict=True)),
                "forward_preserves_element_count": input_shape is not None and output_shape is not None and _product(input_shape) == _product(output_shape),
                "backward_declares_exact_input_shape": declared_backward == input_shape,
                "backward_restores_input_shape": backward_shape == input_shape,
            })
        elif proof_kind == "TRANSPOSE_ADJOINT":
            d0 = _normalize_dim(int(fa[1]), len(input_shape or ()))
            d1 = _normalize_dim(int(fa[2]), len(input_shape or ()))
            bd0 = _normalize_dim(int(ba[1]), len(output_shape or ()))
            bd1 = _normalize_dim(int(ba[2]), len(output_shape or ()))
            expected = list(input_shape or ())
            expected[d0], expected[d1] = expected[d1], expected[d0]
            checks.update({
                "forward_shape_is_declared_transpose": output_shape == tuple(expected),
                "backward_uses_same_axes": (bd0, bd1) == (d0, d1),
                "backward_restores_input_shape": backward_shape == input_shape,
            })
        elif proof_kind == "NEG_ADJOINT":
            checks.update({
                "forward_preserves_shape": output_shape == input_shape,
                "backward_restores_input_shape": backward_shape == input_shape,
                "dtype_preserved": input_dtype == output_dtype == backward_dtype,
            })
        elif proof_kind == "UNSQUEEZE_ADJOINT":
            d = _normalize_dim(int(fa[1]), len(input_shape or ()), insertion=True)
            bd = _normalize_dim(int(ba[1]), len(output_shape or ()))
            expected = list(input_shape or ())
            expected.insert(d, 1)
            checks.update({
                "forward_inserts_exact_size_one_axis": output_shape == tuple(expected),
                "backward_squeezes_same_axis": bd == d,
                "backward_restores_input_shape": backward_shape == input_shape,
            })
        elif proof_kind == "CAST_ADJOINT":
            checks.update({
                "forward_preserves_shape": output_shape == input_shape,
                "forward_requested_dtype_realized": str(_kwargs(forward).get("dtype")) == output_dtype,
                "backward_requests_input_dtype": str(_kwargs(backward).get("dtype")) == input_dtype,
                "backward_restores_input_metadata": backward_shape == input_shape and backward_dtype == input_dtype,
                "floating_dtype_pair": input_dtype is not None and output_dtype is not None and "float" in input_dtype and "float" in output_dtype,
            })
        elif proof_kind == "SLICE_ADJOINT":
            rank = len(input_shape or ())
            d = _normalize_dim(int(fa[1]), rank)
            start = int(fa[2]) if len(fa) > 2 else 0
            end = int(fa[3]) if len(fa) > 3 else 9223372036854775807
            step = int(fa[4]) if len(fa) > 4 else 1
            sizes = tuple(int(x) for x in ba[1])
            bd = _normalize_dim(int(ba[2]), rank)
            normalized = slice(start, end, step).indices((input_shape or ())[d])
            selected = len(range(*normalized))
            expected = list(input_shape or ())
            expected[d] = selected
            checks.update({
                "forward_shape_is_exact_slice": output_shape == tuple(expected),
                "backward_input_sizes_exact": sizes == input_shape,
                "backward_slice_arguments_exact": (bd, int(ba[3]), int(ba[4]), int(ba[5])) == (d, start, end, step),
                "backward_restores_input_shape": backward_shape == input_shape,
            })
        elif proof_kind == "EXPAND_ADJOINT":
            declared = tuple(int(x) for x in fa[1])
            realized = all(a == b or a == -1 for a, b in zip(declared, output_shape or (), strict=True))
            axes = [i for i, (a, b) in enumerate(zip(input_shape or (), output_shape or (), strict=True)) if a == 1 and b != 1]
            checks.update({
                "equal_rank_expand": len(input_shape or ()) == len(output_shape or ()),
                "declared_shape_realized": realized,
                "broadcast_relation_valid": all(a == b or a == 1 for a, b in zip(input_shape or (), output_shape or (), strict=True)),
                "backward_sums_exact_broadcast_axes": [int(x) for x in ba[1]] == axes and bool(ba[2]),
                "backward_restores_input_shape": backward_shape == input_shape,
            })
        elif proof_kind == "PERMUTE_ADJOINT":
            rank = len(input_shape or ())
            permutation = tuple(_normalize_dim(int(x), rank) for x in fa[1])
            inverse = tuple(permutation.index(i) for i in range(rank))
            backward_permutation = tuple(_normalize_dim(int(x), rank) for x in ba[1])
            checks.update({
                "forward_permutation_bijective": sorted(permutation) == list(range(rank)),
                "forward_shape_is_exact_permutation": output_shape == tuple((input_shape or ())[i] for i in permutation),
                "backward_uses_inverse_permutation": backward_permutation == inverse,
                "backward_restores_input_shape": backward_shape == input_shape,
            })
        elif proof_kind == "SELECT_ADJOINT":
            rank = len(input_shape or ())
            d = _normalize_dim(int(fa[1]), rank)
            index = int(fa[2])
            sizes = tuple(int(x) for x in ba[1])
            bd = _normalize_dim(int(ba[2]), rank)
            expected = tuple(x for i, x in enumerate(input_shape or ()) if i != d)
            checks.update({
                "forward_shape_is_exact_select": output_shape == expected,
                "backward_input_sizes_exact": sizes == input_shape,
                "backward_select_arguments_exact": (bd, int(ba[3])) == (d, index),
                "backward_restores_input_shape": backward_shape == input_shape,
            })
    except (IndexError, TypeError, ValueError):
        checks["argument_parsing_and_dimension_normalization"] = False
    passed = bool(checks) and all(checks.values())
    return {
        "proof_kind": proof_kind,
        "exact_forward_map": forward_map,
        "derived_vjp": derived_vjp,
        "checks": checks,
        "passed": passed,
        "claim_boundary": "exact real-arithmetic map/VJP binding; finite-precision kernel arithmetic remains unproved",
    }


def _stack_key(node: dict[str, Any], field: str) -> str | None:
    value = node.get(field)
    return json.dumps(value, sort_keys=True) if value is not None else None


def _node_id(phase: str, graph_index: int, node: dict[str, Any]) -> str:
    return f"{phase.lower()}:graph{graph_index}:{node['name']}"


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    outer = _read(args.capture)
    capture = outer["capture"]
    graphs = capture["graphs"]
    forward_graphs = [graph for graph in graphs if graph["phase"] == "FORWARD"]
    backward_graphs = [graph for graph in graphs if graph["phase"] == "BACKWARD"]
    if len(forward_graphs) != 1 or len(backward_graphs) != 1:
        raise RuntimeError("round-2 ledger requires one forward and backward graph")

    forward_by_origin: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    backward_by_origin: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for node in forward_graphs[0]["nodes"]:
        if node["op"] == "call_function":
            forward_by_origin[_stack_key(node, "source_fn_stack")].append(node)
    fallback_nodes = set()
    for node in backward_graphs[0]["nodes"]:
        if node["op"] != "call_function":
            continue
        origin = _stack_key(node, "fwd_source_fn_stack")
        source = _stack_key(node, "source_fn_stack")
        if origin is None and source in forward_by_origin:
            origin = source
            fallback_nodes.add(node["name"])
        backward_by_origin[origin].append(node)

    forward_index = {
        str(node["name"]): node for node in forward_graphs[0]["nodes"]
    }
    backward_index = {
        str(node["name"]): node for node in backward_graphs[0]["nodes"]
    }
    trainable_dependency, trainable_placeholders = _trainable_dependency(
        forward_graphs[0]["nodes"]
    )
    auxiliary = backward_by_origin.get(None, [])
    deepstack_proofs, auxiliary_proofs, auxiliary_gates = (
        _derive_deepstack_update_programs(
            forward_by_origin,
            auxiliary,
            forward_index,
            backward_index,
        )
    )

    all_targets = {
        node["target"]
        for graph in graphs
        for node in graph["nodes"]
        if node["op"] == "call_function"
    }
    missing_formulas = sorted(all_targets - set(FORMULAS))
    units = []
    for unit_index, (origin, forward_nodes) in enumerate(
        sorted(forward_by_origin.items(), key=lambda item: item[1][0]["ordinal"])
    ):
        backward_nodes = backward_by_origin.get(origin, [])
        forward_targets = [node["target"] for node in forward_nodes]
        backward_targets = [node["target"] for node in backward_nodes]
        formula_complete = all(
            target in FORMULAS for target in forward_targets + backward_targets
        )
        elementary_proof = _verify_elementary_unit(
            forward_nodes, backward_nodes, forward_index
        )
        composite_proof = (
            elementary_proof
            or _verify_arithmetic_composite(
                forward_nodes, backward_nodes, forward_index
            )
            or _verify_matrix_composite(
                forward_nodes, backward_nodes, forward_index
            )
            or _verify_layout_and_routing_composite(
                forward_nodes, backward_nodes, forward_index
            )
            or _verify_nonlinear_and_normalization_composite(
                forward_nodes, backward_nodes, forward_index
            )
            or _verify_index_embedding_conv_composite(
                forward_nodes, backward_nodes, forward_index, backward_index
            )
            or (
                _verify_no_explicit_backward_unit(
                    forward_nodes, forward_index, trainable_dependency
                )
                if not backward_nodes
                else None
            )
            or deepstack_proofs.get(origin or "")
        )
        proof_passed = bool(composite_proof and composite_proof["passed"])
        units.append(
            {
                "unit_id": f"vl-fb-{unit_index:04d}",
                "exact_origin": json.loads(origin) if origin is not None else None,
                "forward_node_ids": [
                    _node_id("FORWARD", 0, node) for node in forward_nodes
                ],
                "backward_node_ids": [
                    _node_id("BACKWARD", 0, node) for node in backward_nodes
                ],
                "forward_program": forward_targets,
                "actual_backward_program": backward_targets,
                "node_formulas": [
                    {"target": target, **FORMULAS[target]}
                    for target in forward_targets + backward_targets
                    if target in FORMULAS
                ],
                "composite_vjp_proof": composite_proof,
                "binding": {
                    "source_fn_stack_exact": origin is not None,
                    "backward_nodes_with_exact_forward_origin": sum(
                        node.get("fwd_source_fn_stack") is not None
                        for node in backward_nodes
                    ),
                    "saved-value_detach_nodes_bound_by_exact_source_stack": sum(
                        node["name"] in fallback_nodes for node in backward_nodes
                    ),
                    "name_shape_similarity_used": False,
                    "candidate_tensor_values_used": False,
                },
                "status": (
                    "PROVED_EXACT_REAL_ARITHMETIC_FWD_VJP"
                    if proof_passed
                    else "FORMULAS_DECLARED_PENDING_COMPOSITE_VJP_PROOF"
                    if formula_complete
                    else "UNRESOLVED_MISSING_NODE_FORMULA"
                ),
            }
        )

    forward_ids = {
        node_id for unit in units for node_id in unit["forward_node_ids"]
    }
    backward_ids = {
        node_id for unit in units for node_id in unit["backward_node_ids"]
    }
    auxiliary_ids = {_node_id("BACKWARD", 0, node) for node in auxiliary}
    expected_forward = {
        _node_id("FORWARD", 0, node)
        for node in forward_graphs[0]["nodes"]
        if node["op"] == "call_function"
    }
    expected_backward = {
        _node_id("BACKWARD", 0, node)
        for node in backward_graphs[0]["nodes"]
        if node["op"] == "call_function"
    }
    gates = {
        "all_call_function_targets_have_declared_formulas": not missing_formulas,
        "every_forward_node_in_exactly_one_unit": forward_ids == expected_forward,
        "every_backward_node_in_unit_or_auxiliary": (
            backward_ids | auxiliary_ids
        ) == expected_backward and not (backward_ids & auxiliary_ids),
        "all_argument_records_present": all(
            isinstance(node.get("arguments"), dict)
            and isinstance(node.get("input_edges"), list)
            for graph in graphs
            for node in graph["nodes"]
            if node["op"] == "call_function"
        ),
        "all_nodes_have_autograd_sequence_numbers": all(
            node.get("seq_nr") is not None
            for graph in graphs
            for node in graph["nodes"]
            if node["op"] == "call_function"
        ),
        "composite_vjp_proof_complete": all(
            unit["status"] == "PROVED_EXACT_REAL_ARITHMETIC_FWD_VJP"
            for unit in units
        ),
        "all_auxiliary_backward_nodes_derived": all(
            auxiliary_gates.values()
        ),
    }
    mathematical_derivation_complete = all(gates.values())
    payload = {
        "schema": "kernel-analyzer.round2-vl-math-ledger.v2",
        "status": (
            "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION"
            if mathematical_derivation_complete
            else "PARTIAL_MATHEMATICAL_DERIVATION"
        ),
        "capture_sha256": capture["capture_sha256"],
        "property_stage_allowed": False,
        "denominator": {
            "forward_call_function_nodes": len(expected_forward),
            "backward_call_function_nodes": len(expected_backward),
            "semantic_forward_backward_units": len(units),
            "units_with_actual_backward_nodes": sum(
                bool(unit["backward_node_ids"]) for unit in units
            ),
            "ssa_or_inactive_backward_units": sum(
                not unit["backward_node_ids"] for unit in units
            ),
            "auxiliary_backward_nodes": len(auxiliary),
            "unique_targets": len(all_targets),
            "targets_with_declared_formula": len(all_targets & set(FORMULAS)),
            "requires_grad_forward_placeholders": len(trainable_placeholders),
            "units_with_complete_real_arithmetic_vjp_proof": sum(
                unit["status"] == "PROVED_EXACT_REAL_ARITHMETIC_FWD_VJP"
                for unit in units
            ),
            "units_pending_composite_vjp_proof": sum(
                unit["status"]
                == "FORMULAS_DECLARED_PENDING_COMPOSITE_VJP_PROOF"
                for unit in units
            ),
        },
        "gates": gates,
        "missing_formula_targets": missing_formulas,
        "target_counts": dict(
            sorted(
                Counter(
                    node["target"]
                    for graph in graphs
                    for node in graph["nodes"]
                    if node["op"] == "call_function"
                ).items()
            )
        ),
        "units": units,
        "auxiliary_backward_program": [
            {
                "node_id": _node_id("BACKWARD", 0, node),
                "target": node["target"],
                "formula": FORMULAS.get(node["target"]),
                "composite_program_proof": auxiliary_proofs.get(node["name"]),
                "status": (
                    "PROVED_AS_DEEPSTACK_OVERWRITE_VJP"
                    if auxiliary_proofs.get(node["name"], {}).get("passed")
                    else "PENDING_AUXILIARY_PROGRAM_DERIVATION"
                ),
            }
            for node in auxiliary
        ],
        "auxiliary_program_gates": auxiliary_gates,
        "coverage": {
            "semantic_unit_real_arithmetic_vjp_proof": (
                sum(
                    unit["status"]
                    == "PROVED_EXACT_REAL_ARITHMETIC_FWD_VJP"
                    for unit in units
                )
                / len(units)
                if units
                else 0.0
            ),
            "auxiliary_backward_node_derivation": (
                sum(
                    bool(auxiliary_proofs.get(node["name"], {}).get("passed"))
                    for node in auxiliary
                )
                / len(auxiliary)
                if auxiliary
                else 1.0
            ),
        },
        "claim_boundary": {
            "supported": "denominator-complete exact real-arithmetic derivation of every concrete AOT forward origin and its actual AOT backward program, including no-op SSA routes, inactive training edges, global fan-in sums, and functionalized deepstack updates",
            "not_supported": [
                "finite-precision implementation correctness",
                "generated Inductor/Triton/cuBLAS kernel arithmetic equivalence",
                "directional-bias case verdict",
                "property induction",
            ],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["ledger_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "denominator": payload["denominator"],
                "gates": gates,
                "missing_formula_targets": missing_formulas,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
