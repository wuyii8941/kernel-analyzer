"""Pre-registered E4 case list, synthetic source S2 (docs/SUFFICIENCY_PROTOCOL.md section 6).

This file is committed BEFORE any E4 measurement; its commit id is recorded in section 7 of the protocol. Rules:

* Every case starts from one reviewed real-call measure case ``base`` (corpus/measure_cases/<base>.json) whose call
  file and binding plan are copied by sufficiency/run_e4.py into a scratch copy of the specification tree.
* ``edits``: (old, new) exact substrings applied IN ORDER to the base call file; each ``old`` must occur exactly once
  in the text at the moment it is applied (the base file at the registered repository state, after earlier edits).
* ``plan_edits``: (json_pointer, new_value) applied to the copied plan (only M7 and the CE/GELU task choices use them;
  a pointer replaces the whole value, lists are replaced, never appended).
* ``input_edits``: entries that REPLACE whole input entries of the declaration's "inputs". Non-contiguous inputs are
  produced by the generators below (``state`` sources, resolved relative to corpus/measure_cases), used only with
  calls whose edits accept and correctly or incorrectly handle strides.
* ``expected``: "detect" for an error (G_pi - f must exclude zero somewhere), "accept" for a normal control (it must
  not be judged contradictory), "refuse" where the correct behaviour is not to give a unique verdict (ties outside a
  tie-free profile; a task-choice profile without a task statement).
* Normal controls are genuine alternatives with the same real-number program formula and the same constants: other
  BLOCK / num_warps / grid splits, exact algebraic reorderings, other masks with the same allowed set, correct strided
  handling. No normal control introduces a rounded constant (e.g. a host reciprocal of a non-power-of-two).
* Errors are realistic instances of their mechanism. Each edited call was executed once on random inputs of the base
  declaration's shapes only to confirm that it compiles, runs and returns the base output names, shapes and dtypes;
  no reference or semantic comparison was run before registration. Cases that failed to build are listed in
  FAILED_TO_BUILD with the error and are kept (not silently dropped).
* M7 is not a bug class: the cases check that a reading chosen by the task is accepted, an implementation of another
  reading is detected when the task chose otherwise, and a task-choice profile with no task statement is refused.
"""
from __future__ import annotations

TAXONOMY = {
    "M1": {"name": "formula and constants",
           "error": "coefficient or approximation written wrongly",
           "normal": "correct implementation; a mathematically equivalent formulation"},
    "M2": {"name": "normalization denominator",
           "error": "divide by N-1, or by the full length instead of the valid length",
           "normal": "correct denominator; a different accumulation order"},
    "M3": {"name": "mask and empty rows",
           "error": "mask omitted or off by one; empty-row handling wrong",
           "normal": "correct mask; legal output for empty rows"},
    "M4": {"name": "ties and selection",
           "error": "returns an index outside the allowed set",
           "normal": "another choice inside the allowed set (tie-free profiles: an equivalent selection)"},
    "M5": {"name": "state write-back",
           "error": "state buffer not written, or written with the wrong value",
           "normal": "complete correct write-back"},
    "M6": {"name": "layout and stride",
           "error": "stride ignored, contiguity assumed, or another tensor's stride used",
           "normal": "correct non-contiguous implementation"},
    "M7": {"name": "reading ambiguity",
           "error": "(not a bug that must be detected) forcing a verdict when the task chose no reading",
           "normal": "every reading the specification admits is accepted when chosen, or a unique verdict is refused"},
}


# ------------------------------------------------------------------------------------------- input generators (M6)
# Referenced from input_edits as "../../sufficiency/e4_cases.py:<name>" (relative to corpus/measure_cases).

def _generator(seed, shape, salt):
    import torch
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed) * 1000003 + salt)
    return g


def make_column_major(seed, shape, dtype):
    """Standard-normal float32 matrix of the declared shape stored column-major (a transposed view)."""
    import torch
    rows, cols = (int(n) for n in shape)
    if dtype != "float32":
        raise ValueError("make_column_major is float32 only")
    g = _generator(seed, shape, 7919 * rows + cols)
    storage = torch.randn((cols, rows), generator=g, dtype=torch.float32, device="cuda")
    return storage.t()


def make_stride2(seed, shape, dtype):
    """Standard-normal float32 vector of the declared length, a stride-2 view of a buffer twice as long."""
    import torch
    (n,) = (int(x) for x in shape)
    if dtype != "float32":
        raise ValueError("make_stride2 is float32 only")
    g = _generator(seed, shape, 104729 + n)
    storage = torch.randn((2 * n,), generator=g, dtype=torch.float32, device="cuda")
    return storage[::2]


_COLMAJOR = "../../sufficiency/e4_cases.py:make_column_major"
_STRIDE2 = "../../sufficiency/e4_cases.py:make_stride2"


def _cm(shape):
    return {"state": _COLMAJOR, "shape": list(shape), "dtype": "float32"}


def _s2(n):
    return {"state": _STRIDE2, "shape": [n], "dtype": "float32"}


def case(id, mechanism, kind, base, edits, expected, rationale, plan_edits=None, input_edits=None):
    row = dict(id=id, mechanism=mechanism, kind=kind, source="S2", base=base, edits=list(edits),
               expected=expected, rationale=rationale)
    if plan_edits:
        row["plan_edits"] = list(plan_edits)
    if input_edits:
        row["input_edits"] = dict(input_edits)
    return row


# Shared edit fragments ----------------------------------------------------------------------------------------------

# triton_transpose: accept non-contiguous x and pass its strides (used by the M6 strided cases)
_TRANSPOSE_STRIDED_ABI = [
    ("def _transpose_kernel(x_ptr, out_ptr, SIZE: tl.constexpr,",
     "def _transpose_kernel(x_ptr, out_ptr, stride_row, stride_col, SIZE: tl.constexpr,"),
    ("or not x.is_cuda or not x.is_contiguous()):", "or not x.is_cuda):"),
    ("x, out, SIZE=SIZE, ROWS=ROWS", "x, out, x.stride(0), x.stride(1), SIZE=SIZE, ROWS=ROWS"),
]
_TRANSPOSE_USE_STRIDES = [("source = out_col * COLS + out_row",
                           "source = out_col * stride_row + out_row * stride_col")]

# triton_matmul
_MATMUL_STRIDED_ABI = [
    ("def _matmul_kernel(a_ptr, b_ptr, out_ptr,", "def _matmul_kernel(a_ptr, b_ptr, out_ptr, sam, sak, sbk, sbn,"),
    ("        if not value.is_contiguous():\n            raise ValueError(f\"{name} must be contiguous\")\n", ""),
    ("_matmul_kernel[(1,)](a, b, out, M, N, K,",
     "_matmul_kernel[(1,)](a, b, out, a.stride(0), a.stride(1), b.stride(0), b.stride(1), M, N, K,"),
]
_MATMUL_A_STRIDES = [("a_ptr + rows[:, None] * K + inner[None, :]", "a_ptr + rows[:, None] * sam + inner[None, :] * sak")]
_MATMUL_B_STRIDES = [("b_ptr + inner[:, None] * N + cols[None, :]", "b_ptr + inner[:, None] * sbk + cols[None, :] * sbn")]

# triton_row_sum
_ROWSUM_STRIDED_ABI = [
    ("def _row_sum_kernel(x_ptr, out_ptr,", "def _row_sum_kernel(x_ptr, out_ptr, stride_row, stride_col,"),
    ("or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()):",
     "or x.dtype != torch.float32 or not x.is_cuda):"),
    ("x, out, ROWS=ROWS,", "x, out, x.stride(0), x.stride(1), ROWS=ROWS,"),
]

# triton_index_select
_INDEXSEL_STRIDED_ABI = [
    ("def _index_select_kernel(x_ptr, index_ptr, out_ptr, SIZE: tl.constexpr,",
     "def _index_select_kernel(x_ptr, index_ptr, out_ptr, stride_row, stride_col, SIZE: tl.constexpr,"),
    ("or not x.is_cuda or not x.is_contiguous()):", "or not x.is_cuda):"),
    ("x, index, out, SIZE=SIZE", "x, index, out, x.stride(0), x.stride(1), SIZE=SIZE"),
]

# triton_multiply_add
_MULADD_STRIDED_ABI = [
    ("def _multiply_add_kernel(x_ptr, y_ptr, z_ptr, out_ptr,",
     "def _multiply_add_kernel(x_ptr, y_ptr, z_ptr, out_ptr, sx, sy, sz,"),
    ("or not value.is_cuda or not value.is_contiguous()):", "or not value.is_cuda):"),
    ("values[0], values[1], values[2], out, SIZE=SIZE",
     "values[0], values[1], values[2], out, values[0].stride(0), values[1].stride(0), values[2].stride(0), "
     "SIZE=SIZE"),
    ("x = tl.load(x_ptr + offsets, mask", "x = tl.load(x_ptr + offsets * sx, mask"),
]

# triton_embedding (weight may be non-contiguous; forward kernel takes its strides)
_EMB_STRIDED_ABI = [
    ("def _embedding_forward(weight_ptr, index_ptr, out_ptr, WIDTH: tl.constexpr):",
     "def _embedding_forward(weight_ptr, index_ptr, out_ptr, stride_row, stride_col, WIDTH: tl.constexpr):"),
    ("or not tensor.is_cuda or not tensor.is_contiguous()):",
     "or not tensor.is_cuda or not (tensor.is_contiguous() or name == \"weight\")):"),
    ("_embedding_forward[(POSITIONS,)](weight, indices, out, WIDTH=WIDTH, num_warps=1)",
     "_embedding_forward[(POSITIONS,)](weight, indices, out, weight.stride(0), weight.stride(1), WIDTH=WIDTH, "
     "num_warps=1)"),
]

# tie prechecks (M4 refuse cases remove them so tied inputs reach the kernel)
_MIXTRAL_NO_TIE_CHECK = [(
    "    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):\n"
    "        raise ValueError(\"router_logits must be pairwise distinct within every token\")\n", "")]
_SORT_NO_TIE_CHECK = [(
    "    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):\n"
    "        raise ValueError(\"every input row must contain pairwise distinct values\")\n", "")]
_TOPK_NO_TIE_CHECK = [(
    "    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):\n"
    "        raise ValueError(\"logits must be pairwise distinct within every row\")\n", "")]

# CE weighted label smoothing (M7): a call boundary with per-class weights, eps 1/4, reduction sum
_CE_LOSS_OLD = (
    "def _loss(logits, target):\n"
    "    return F.cross_entropy(\n"
    "        logits,\n"
    "        target,\n"
    "        weight=None,\n"
    "        ignore_index=-100,\n"
    "        reduction=\"mean\",\n"
    "        label_smoothing=0.0,\n"
    "    )\n")
_CE_LOSS_TORCH_RA = (
    "def _loss(logits, target, weight):\n"
    "    return F.cross_entropy(\n"
    "        logits,\n"
    "        target,\n"
    "        weight=weight,\n"
    "        ignore_index=-100,\n"
    "        reduction=\"sum\",\n"
    "        label_smoothing=0.25,\n"
    "    )\n")
_CE_LOSS_MANUAL_RB = (
    "def _loss(logits, target, weight):\n"
    "    # R_B: the target-class weight multiplies every class term of the smoothed target\n"
    "    log_p = torch.log_softmax(logits, dim=1)\n"
    "    classes = logits.shape[1]\n"
    "    onehot = F.one_hot(target, classes).to(logits.dtype)\n"
    "    smoothed = 0.75 * onehot + torch.full_like(log_p, 0.25) / classes\n"
    "    return -(weight[target] * (smoothed * log_p).sum(dim=1)).sum()\n")
_CE_BOUNDARY = [
    ("def _loss_and_gradient(logits, target):\n    gradient, loss = torch.func.grad_and_value(_loss)(logits, target)",
     "def _loss_and_gradient(logits, target, weight):\n"
     "    gradient, loss = torch.func.grad_and_value(_loss)(logits, target, weight)"),
    ("if inputs[\"weights\"] is not None or inputs[\"eps\"] != 0 or",
     "if not torch.is_tensor(inputs[\"weights\"]) or inputs[\"eps\"] != 0.25 or"),
    ("inputs[\"reduction\"] != \"mean\" or", "inputs[\"reduction\"] != \"sum\" or"),
    ("loss, gradient = _compiled(logits, target)", "loss, gradient = _compiled(logits, target, inputs[\"weights\"])"),
]
_CE_RA_PLAN = [
    ("/tasks/0/profile", "index_smoothing_R_A"),
    ("/tasks/0/task_statement",
     "Two-dimensional class-index cross entropy with per-class weights, label smoothing 1/4, ignore_index -100, "
     "reduction sum and scalar upstream 1; the task selects reading R_A of weighted smoothing "
     "(a_nc = w_c((1-eps)onehot(y)_c + eps/C))."),
    ("/tasks/0/input_map/2", {"target": "/weights", "source": "/tensor_arguments/weights/before",
                              "conversion": "tensor_values"}),
    ("/state_map", [{"argument": "logits", "effect": "unchanged"}, {"argument": "target", "effect": "unchanged"},
                    {"argument": "weights", "effect": "unchanged"}]),
    ("/subject/callable", "torch.compile(torch.func.grad_and_value(torch.nn.functional.cross_entropy with weight, "
                          "label_smoothing=0.25, reduction='sum'))"),
    ("/subject/boundary", "compiled summed weighted label-smoothed loss and complete logits VJP"),
]
_CE_RA_INPUTS = {
    "weights": {"sampler": {"uniform": [0.5, 2]}, "shape": [23], "dtype": "float32"},
    "eps": {"const": 0.25},
    "reduction": {"const": "sum"},
}

# GELU (M7): the task's approximate choice
_GELU_TANH_PLAN = [
    ("/tasks/0/task_statement", "Compute public PyTorch 2.10 GELU with approximate='tanh' on the complete finite "
                                "float32 input."),
    ("/tasks/1/task_statement", "Compute the complete input VJP of public PyTorch 2.10 GELU with approximate='tanh' "
                                "against the received upstream tensor."),
    ("/subject/callable", "torch.compile(torch.func.grad_and_value(torch.nn.functional.gelu(approximate='tanh')))"),
]


CASES = [
    # =============================================================================================== M1 formula
    case("M1-E01", "M1", "error", "triton_sgd",
         [("direction = processed + MOMENTUM * new_buffer", "direction = processed + MOMENTUM * old_buffer")],
         "detect", "Nesterov look-ahead uses the old momentum buffer instead of the updated one (formula error)."),
    case("M1-E02", "M1", "error", "triton_adam",
         [("tl.sqrt(vhat) + EPS", "tl.sqrt(vhat + EPS)")],
         "detect", "Adam epsilon placed inside the square root, a common formula slip."),
    case("M1-E03", "M1", "error", "triton_adam",
         [("processed = -gradient + WEIGHT_DECAY * theta", "processed = -gradient - WEIGHT_DECAY * theta")],
         "detect", "Coupled weight-decay term has the wrong sign under maximize (sign error in the formula)."),
    case("M1-E04", "M1", "error", "triton_rmsprop",
         [("adjusted = new_v - new_gave * new_gave", "adjusted = new_v")],
         "detect", "Centered RMSprop drops the squared-mean correction (non-centered formula used)."),
    case("M1-E05", "M1", "error", "triton_batch_norm_eval",
         [("tl.rsqrt(variance + EPS)", "(1.0 / (tl.sqrt(variance) + EPS))")],
         "detect", "BatchNorm epsilon added outside the square root (NM.M2-style formula error)."),
    case("M1-E06", "M1", "error", "triton_kl_batchmean",
         [("terms = target * (tl.log(target) - values)", "terms = tl.exp(target) * (target - values)")],
         "detect", "KL term written for log-space targets (log_target=True formula) on a probability target."),
    case("M1-E07", "M1", "error", "triton_llama_rope",
         [("rotated = tl.where(feature < half, -x_partner, x_partner)",
           "rotated = tl.where(feature < half, x_partner, -x_partner)")],
         "detect", "rotate_half sign convention flipped, a frequent RoPE port error."),
    case("M1-E08", "M1", "error", "triton_mamba_ssu",
         [("out *= gate * tl.sigmoid(gate)", "out *= tl.sigmoid(gate)")],
         "detect", "SiLU gate replaced by a plain sigmoid gate (activation formula error)."),
    case("M1-E09", "M1", "error", "triton_dropout_backward",
         [("SCALE = 1.25", "SCALE = 0.8")],
         "detect", "Dropout backward multiplies by keep probability instead of dividing by it (wrong constant)."),
    case("M1-E10", "M1", "error", "triton_index_add",
         [("base + source * ALPHA", "base + source")],
         "detect", "index_add alpha coefficient ignored."),
    case("M1-E11", "M1", "error", "triton_radam",
         [("rho = rho_inf - 12.0 * beta2_sixth / correction2", "rho = rho_inf - 10.0 * beta2_sixth / correction2")],
         "detect", "RAdam rho_t coefficient 2t evaluated with the pre-increment step (constant error)."),
    case("M1-E12", "M1", "error", "triton_lora_shrink",
         [("value = SCALING * tl.sum(weight * x, axis=0)", "value = tl.sum(weight * x, axis=0)")],
         "detect", "LoRA shrink scaling factor omitted."),
    case("M1-E13", "M1", "error", "triton_exp",
         [("result = tl.exp(values)", "result = tl.exp2(values)")],
         "detect", "Base-2 exponential used for exp (wrong function in the formula)."),
    case("M1-E14", "M1", "error", "triton_rprop",
         [("tl.where(product > 0.0, ETAPLUS, tl.where(product < 0.0, ETAMINUS, 1.0))",
           "tl.where(product > 0.0, ETAMINUS, tl.where(product < 0.0, ETAPLUS, 1.0))")],
         "detect", "Rprop increase and decrease factors swapped."),

    case("M1-N01", "M1", "normal", "triton_sgd",
         [("new_theta = theta - LR * direction",
           "new_theta = theta - (LR * processed + LR * (MOMENTUM * new_buffer))")],
         "accept", "Learning rate distributed over the Nesterov direction: same real formula, same constants."),
    case("M1-N02", "M1", "normal", "triton_adam",
         [("new_theta = theta - LR * mhat / (tl.sqrt(vhat) + EPS)",
           "new_theta = theta - (LR * mhat) * (1.0 / (tl.sqrt(vhat) + EPS))")],
         "accept", "Division rewritten as multiplication by an in-program reciprocal: identical real formula."),
    case("M1-N03", "M1", "normal", "triton_adam",
         [("new_m = BETA1 * old_m + (1.0 - BETA1) * processed", "new_m = processed + BETA1 * (old_m - processed)")],
         "accept", "Lerp form of the first-moment EMA, algebraically identical with the same beta1."),
    case("M1-N04", "M1", "normal", "triton_rmsprop",
         [("new_buffer = MOMENTUM * old_buffer + processed / (tl.sqrt(adjusted) + EPS)",
           "new_buffer = processed / (EPS + tl.sqrt(adjusted)) + old_buffer * MOMENTUM")],
         "accept", "Commuted sums and products in the momentum update: same real formula."),
    case("M1-N05", "M1", "normal", "triton_batch_norm_eval",
         [("value = (x - mean) * tl.rsqrt(variance + EPS) * weight + bias",
           "value = (x * weight - mean * weight) / tl.sqrt(variance + EPS) + bias")],
         "accept", "Weight distributed and rsqrt written as a division by sqrt: identical real formula."),
    case("M1-N06", "M1", "normal", "triton_kl_batchmean",
         [("terms = target * (tl.log(target) - values)", "terms = target * tl.log(target) - target * values")],
         "accept", "Distributed KL pointwise term, algebraically identical."),
    case("M1-N07", "M1", "normal", "triton_llama_rope",
         [("out = x * cos + rotated * sin", "out = rotated * sin + cos * x")],
         "accept", "Commuted RoPE combination, same formula."),
    case("M1-N08", "M1", "normal", "triton_mamba_ssu",
         [("out *= gate * tl.sigmoid(gate)", "out *= gate / (1.0 + tl.exp(-gate))")],
         "accept", "SiLU written from the sigmoid definition; same real function."),
    case("M1-N09", "M1", "normal", "triton_dropout_backward",
         [("value = tl.where(keep, grad * SCALE, 0.0)", "value = grad * tl.where(keep, SCALE, 0.0)")],
         "accept", "Scale selected before multiplication; identical on finite inputs."),
    case("M1-N10", "M1", "normal", "triton_index_add",
         [("base + source * ALPHA", "base + source / 2.0")],
         "accept", "alpha = 1/2 applied as division by the exact power of two 2: same real value."),
    case("M1-N11", "M1", "normal", "triton_exp",
         [("result = tl.exp(values)", "result = 1.0 / tl.exp(-values)")],
         "accept", "exp(x) computed as the reciprocal of exp(-x): identical real function."),
    case("M1-N12", "M1", "normal", "triton_rprop",
         [("new_eta = tl.minimum(tl.maximum(old_eta * factor, STEP_MIN), STEP_MAX)",
           "new_eta = tl.maximum(tl.minimum(old_eta * factor, STEP_MAX), STEP_MIN)")],
         "accept", "Clamp order exchanged; identical because STEP_MIN < STEP_MAX."),
    case("M1-N13", "M1", "normal", "triton_lora_shrink",
         [("value = SCALING * tl.sum(weight * x, axis=0)", "value = tl.sum((SCALING * weight) * x, axis=0)")],
         "accept", "Scaling moved inside the reduction (linearity); same real formula and constant."),

    # =============================================================================================== M2 denominator
    case("M2-E01", "M2", "error", "inductor_var",
         [("y = torch.var(x, dim=DIM, correction=CORRECTION, keepdim=KEEPDIM)",
           "y = torch.var(x, dim=DIM, correction=0, keepdim=KEEPDIM)")],
         "detect", "Variance divides by N while the task's correction is 1 (N-1) (denominator error)."),
    case("M2-E02", "M2", "error", "inductor_mse_loss",
         [("loss = F.mse_loss(x, target, reduction=\"mean\")",
           "loss = F.mse_loss(x, target, reduction=\"sum\") / (ROWS * COLS - 1)")],
         "detect", "Mean squared error divided by N-1 instead of N."),
    case("M2-E03", "M2", "error", "inductor_l1_loss",
         [("loss = F.l1_loss(x, target, reduction=\"mean\")", "loss = F.l1_loss(x, target, reduction=\"sum\") / ROWS")],
         "detect", "L1 mean divides by the batch size only instead of the element count."),
    case("M2-E04", "M2", "error", "inductor_layer_norm",
         [("y = F.layer_norm(x, (WIDTH,), weight, bias, EPS)",
           "y = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True, correction=1) + EPS) "
           "* weight + bias")],
         "detect", "Hand-written LayerNorm uses the unbiased variance (NM.M1)."),
    case("M2-E05", "M2", "error", "inductor_rms_norm",
         [("y = F.rms_norm(x, (WIDTH,), weight, EPS)",
           "y = x * torch.rsqrt((x * x).sum(-1, keepdim=True) / (WIDTH - 1) + EPS) * weight")],
         "detect", "RMSNorm mean of squares divided by N-1."),
    case("M2-E06", "M2", "error", "inductor_group_norm",
         [("y = F.group_norm(x, GROUPS, weight, bias, EPS)",
           "xg = x.reshape(x.shape[0], GROUPS, -1)\n"
           "    y = ((xg - xg.mean(-1, keepdim=True)) * torch.rsqrt(xg.var(-1, keepdim=True, correction=1) + EPS))"
           ".reshape(x.shape) * weight[None, :, None, None] + bias[None, :, None, None]")],
         "detect", "Hand-written GroupNorm uses the unbiased group variance."),
    case("M2-E07", "M2", "error", "triton_kl_batchmean",
         [("total = tl.sum(tl.where(mask, terms, 0.0), axis=0) / BATCH",
           "total = tl.sum(tl.where(mask, terms, 0.0), axis=0) / SIZE")],
         "detect", "batchmean computed as elementwise mean (full element count instead of batch size)."),
    case("M2-E08", "M2", "error", "triton_kl_batchmean",
         [("grad = -target / BATCH", "grad = -target / SIZE")],
         "detect", "Gradient normalized by the element count while the loss uses the batch size."),
    case("M2-E09", "M2", "error", "triton_adafactor",
         [("row_target = tl.expand_dims(tl.sum(squared, axis=1) / COLS, 1)",
           "row_target = tl.expand_dims(tl.sum(squared, axis=1) / (COLS - 1), 1)")],
         "detect", "Adafactor row second-moment target averaged with N-1."),
    case("M2-E10", "M2", "error", "triton_adafactor",
         [("theta_rms = tl.sqrt(tl.sum(tl.sum(theta * theta, axis=1), axis=0) / (ROWS * COLS))",
           "theta_rms = tl.sqrt(tl.sum(tl.sum(theta * theta, axis=1), axis=0) / (ROWS * COLS - 1))")],
         "detect", "Parameter RMS divided by N-1."),
    case("M2-E11", "M2", "error", "triton_sdpa",
         [("    scores = tl.where(source <= query_index, scores, -float(\"inf\"))\n"
           "    maximum = tl.max(scores, axis=0)\n"
           "    numerator = tl.exp(scores - maximum)\n"
           "    probabilities = numerator / tl.sum(numerator, axis=0)",
           "    maximum = tl.max(tl.where(source <= query_index, scores, -float(\"inf\")), axis=0)\n"
           "    full = tl.exp(scores - maximum)\n"
           "    numerator = tl.where(source <= query_index, full, 0.0)\n"
           "    probabilities = numerator / tl.sum(full, axis=0)")],
         "detect", "Softmax denominator sums over the full key length instead of the causally valid keys."),
    case("M2-E12", "M2", "error", "triton_paged_attention",
         [("    scores = tl.where(allowed, scores, -float(\"inf\"))\n"
           "    maximum = tl.max(scores, axis=0)\n"
           "    weights = tl.exp(scores - maximum)\n"
           "    weights = weights / tl.sum(weights, axis=0)",
           "    maximum = tl.max(tl.where(allowed, scores, -float(\"inf\")), axis=0)\n"
           "    full = tl.where(source < seq_len, tl.exp(scores - maximum), 0.0)\n"
           "    weights = tl.where(allowed, full, 0.0) / tl.sum(full, axis=0)")],
         "detect", "Attention normalizer counts every cached token of the sequence, not only the allowed window."),
    case("M2-E13", "M2", "error", "triton_pack_varlen",
         [("    scores = tl.where(allowed, scores, -float(\"inf\"))\n"
           "    maximum = tl.max(scores, axis=0)\n"
           "    weights = tl.exp(scores - maximum)\n"
           "    weights = weights / tl.sum(weights, axis=0)",
           "    maximum = tl.max(tl.where(allowed, scores, -float(\"inf\")), axis=0)\n"
           "    full = tl.where(source < key_len, tl.exp(scores - maximum), 0.0)\n"
           "    weights = tl.where(allowed, full, 0.0) / tl.sum(full, axis=0)")],
         "detect", "Varlen attention normalizer over the whole key segment instead of the causal prefix."),
    case("M2-E14", "M2", "error", "triton_embedding",
         [("denominator = tl.where(count == 0, 1.0, count)", "denominator = tl.where(count <= 1, 1.0, count - 1.0)")],
         "detect", "scale_grad_by_freq divides by count-1 instead of the occurrence count."),

    case("M2-N01", "M2", "normal", "inductor_var",
         [("y = torch.var(x, dim=DIM, correction=CORRECTION, keepdim=KEEPDIM)",
           "y = ((x - x.mean(dim=DIM, keepdim=True)) ** 2).sum(dim=DIM, keepdim=KEEPDIM) / (x.shape[DIM] - CORRECTION)")],
         "accept", "Explicit two-pass variance with denominator N - correction: same real formula."),
    case("M2-N02", "M2", "normal", "inductor_mse_loss",
         [("loss = F.mse_loss(x, target, reduction=\"mean\")",
           "loss = ((x - target) * (x - target)).sum() / (ROWS * COLS)")],
         "accept", "Explicit sum of squared differences over the element count."),
    case("M2-N03", "M2", "normal", "inductor_l1_loss",
         [("loss = F.l1_loss(x, target, reduction=\"mean\")", "loss = (x - target).abs().sum() / (ROWS * COLS)")],
         "accept", "Explicit L1 mean; abs'(0)=0 like l1_loss's sign-based VJP."),
    case("M2-N04", "M2", "normal", "inductor_layer_norm",
         [("y = F.layer_norm(x, (WIDTH,), weight, bias, EPS)",
           "y = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True, correction=0) + EPS) "
           "* weight + bias")],
         "accept", "Hand-written LayerNorm with the biased variance and the same eps."),
    case("M2-N05", "M2", "normal", "inductor_rms_norm",
         [("y = F.rms_norm(x, (WIDTH,), weight, EPS)",
           "y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * weight")],
         "accept", "Hand-written RMSNorm with mean of squares over N and the same eps."),
    case("M2-N06", "M2", "normal", "inductor_group_norm",
         [("y = F.group_norm(x, GROUPS, weight, bias, EPS)",
           "xg = x.reshape(x.shape[0], GROUPS, -1)\n"
           "    y = ((xg - xg.mean(-1, keepdim=True)) * torch.rsqrt(xg.var(-1, keepdim=True, correction=0) + EPS))"
           ".reshape(x.shape) * weight[None, :, None, None] + bias[None, :, None, None]")],
         "accept", "Hand-written GroupNorm with the biased group variance."),
    case("M2-N07", "M2", "normal", "triton_kl_batchmean",
         [("total = tl.sum(tl.where(mask, terms, 0.0), axis=0) / BATCH",
           "total = tl.sum(tl.sum(tl.reshape(tl.where(mask, terms, 0.0), (BATCH, BLOCK // BATCH)), axis=1), axis=0) "
           "/ BATCH")],
         "accept", "Per-row partial sums then a sum over rows: different accumulation order, same denominator."),
    case("M2-N08", "M2", "normal", "triton_adafactor",
         [("row_target = tl.expand_dims(tl.sum(squared, axis=1) / COLS, 1)",
           "row_target = tl.expand_dims(tl.sum(squared / COLS, axis=1), 1)")],
         "accept", "Division moved inside the row sum (linearity)."),
    case("M2-N09", "M2", "normal", "triton_adafactor",
         [("theta_rms = tl.sqrt(tl.sum(tl.sum(theta * theta, axis=1), axis=0) / (ROWS * COLS))",
           "theta_rms = tl.sqrt(tl.sum(tl.sum(theta * theta, axis=0), axis=0) / (ROWS * COLS))")],
         "accept", "Column-first accumulation order for the parameter RMS."),
    case("M2-N10", "M2", "normal", "triton_sdpa",
         [("probabilities = numerator / tl.sum(numerator, axis=0)",
           "probabilities = numerator * (1.0 / tl.sum(numerator, axis=0))")],
         "accept", "Softmax normalized by an in-program reciprocal of the valid-key sum."),
    case("M2-N11", "M2", "normal", "triton_paged_attention",
         [("    weights = weights / tl.sum(weights, axis=0)\n", "    denominator = tl.sum(weights, axis=0)\n"),
          ("    result = tl.sum(weights[:, None] * values, axis=0)\n",
           "    result = tl.sum(weights[:, None] * values, axis=0) / denominator\n")],
         "accept", "Deferred (flash-style) normalization after the value reduction."),
    case("M2-N12", "M2", "normal", "triton_pack_varlen",
         [("    weights = weights / tl.sum(weights, axis=0)\n", "    denominator = tl.sum(weights, axis=0)\n"),
          ("    result = tl.sum(weights[:, None] * values, axis=0)\n",
           "    result = tl.sum(weights[:, None] * values, axis=0) / denominator\n")],
         "accept", "Deferred normalization in varlen attention."),
    case("M2-N13", "M2", "normal", "triton_bsr_softmax_values",
         [("result = numerators / tl.sum(numerators, axis=0)",
           "result = numerators * (1.0 / tl.sum(numerators, axis=0))")],
         "accept", "Block-sparse softmax normalized by an in-program reciprocal."),
    case("M2-N14", "M2", "normal", "triton_embedding",
         [("result = total / denominator",
           "result = tl.sum(tl.where(selected[:, None], received / denominator, 0.0), axis=0)")],
         "accept", "Each occurrence divided by the count before accumulation (linearity)."),

    # =============================================================================================== M3 mask
    case("M3-E01", "M3", "error", "triton_sdpa",
         [("scores = tl.where(source <= query_index, scores, -float(\"inf\"))",
           "scores = tl.where(source < query_index, scores, -float(\"inf\"))")],
         "detect", "Causal mask off by one: the diagonal is excluded and query row 0 becomes empty."),
    case("M3-E02", "M3", "error", "triton_sdpa",
         [("scores = tl.where(source <= query_index, scores, -float(\"inf\"))",
           "scores = tl.where(source <= query_index + (SOURCE - LENGTH), scores, -float(\"inf\"))")],
         "detect", "Bottom-right causal alignment used where torch's is_causal is top-left for L != S."),
    case("M3-E03", "M3", "error", "triton_paged_attention",
         [("(position - source <= LEFT_WINDOW)", "(position - source < LEFT_WINDOW)")],
         "detect", "Sliding-window bound off by one."),
    case("M3-E04", "M3", "error", "triton_paged_attention",
         [("allowed = (source < seq_len) & (source <= position) & (position - source <= LEFT_WINDOW)",
           "allowed = (source < seq_len) & (source <= position)")],
         "detect", "Sliding-window mask not applied."),
    case("M3-E05", "M3", "error", "triton_pack_varlen",
         [("allowed = (source < key_len) & (source <= local_query + key_len - query_len)",
           "allowed = (source < key_len) & (source <= local_query)")],
         "detect", "Varlen causal mask aligned top-left instead of bottom-right (key/query length offset dropped)."),
    case("M3-E06", "M3", "error", "triton_pack_varlen",
         [("allowed = (source < key_len) & (source <= local_query + key_len - query_len)",
           "allowed = (source < key_len) & (source < local_query + key_len - query_len)")],
         "detect", "Varlen causal mask off by one (diagonal excluded; some rows empty)."),
    case("M3-E07", "M3", "error", "triton_lora_shrink",
         [("        tl.store(out_ptr + (slice_index * TOKENS + token) * RANK + rank, 0.0)\n        return",
           "        return")],
         "detect", "Tokens without a LoRA are skipped and their output rows keep stale pre-state (empty-row error)."),
    case("M3-E08", "M3", "error", "triton_lora_shrink",
         [("if lora < 0:", "if lora < 1:")],
         "detect", "No-LoRA mask off by one: LoRA id 0 treated as absent."),
    case("M3-E09", "M3", "error", "triton_embedding",
         [("    result = tl.where(row == PADDING_IDX, 0.0, result)\n", "")],
         "detect", "padding_idx row of the weight gradient not masked to zero."),
    case("M3-E10", "M3", "error", "triton_row_sum",
         [("mask=columns < COLUMNS, other=0.0)", "mask=row * COLUMNS + columns < ROWS * COLUMNS, other=0.0)"),
          ("BLOCK_COLUMNS=COLUMNS, num_warps=8", "BLOCK_COLUMNS=2 * COLUMNS, num_warps=8")],
         "detect", "Padded lanes masked only by the flat tensor bound, so a row reduction reads the next row."),
    case("M3-E11", "M3", "error", "triton_dropout_backward",
         [("value = tl.where(keep, grad * SCALE, 0.0)", "value = grad * SCALE")],
         "detect", "Dropout keep-mask not applied in backward."),
    case("M3-E12", "M3", "error", "triton_conv2d",
         [("x = tl.load(x_ptr + x_offset, mask=inside, other=0.0)",
           "x = tl.load(x_ptr + x_offset, mask=(x_offset >= 0) & (x_offset < N * C_IN * H * W), other=0.0)")],
         "detect", "Zero-padding mask replaced by a flat bound check: padded taps read neighbouring pixels."),
    case("M3-E13", "M3", "error", "triton_index_add",
         [("mask=active & selected, other=0.0)", "mask=active, other=0.0)")],
         "detect", "Selection mask dropped: rows not named by the index also receive source rows."),

    case("M3-N01", "M3", "normal", "triton_sdpa",
         [("scores = tl.where(source <= query_index, scores, -float(\"inf\"))",
           "scores = tl.where(query_index - source >= 0, scores, float(\"-inf\"))")],
         "accept", "Same causal allowed set written as a difference test."),
    case("M3-N02", "M3", "normal", "triton_sdpa",
         [("k = tl.load(k_ptr + k_base + source[:, None] * HEAD_DIM + feature[None, :])",
           "k = tl.load(k_ptr + k_base + source[:, None] * HEAD_DIM + feature[None, :], "
           "mask=(source <= query_index)[:, None], other=0.0)")],
         "accept", "Masked keys are not loaded; their scores are masked anyway."),
    case("M3-N03", "M3", "normal", "triton_paged_attention",
         [("allowed = (source < seq_len) & (source <= position) & (position - source <= LEFT_WINDOW)",
           "allowed = (source < seq_len) & (source <= position) & (source >= position - LEFT_WINDOW)")],
         "accept", "Window bound rewritten as a lower key bound: same allowed set."),
    case("M3-N04", "M3", "normal", "triton_pack_varlen",
         [("allowed = (source < key_len) & (source <= local_query + key_len - query_len)",
           "allowed = source <= local_query + key_len - query_len")],
         "accept", "Redundant key-length test removed: the causal bound already implies it."),
    case("M3-N05", "M3", "normal", "triton_lora_shrink",
         [("    if lora < 0:\n"
           "        tl.store(out_ptr + (slice_index * TOKENS + token) * RANK + rank, 0.0)\n"
           "        return\n",
           "    active = lora >= 0\n    lora = tl.maximum(lora, 0)\n"),
          ("value = SCALING * tl.sum(weight * x, axis=0)",
           "value = tl.where(active, SCALING * tl.sum(weight * x, axis=0), 0.0)")],
         "accept", "Branch-free handling of no-LoRA tokens: they still receive the legal zero output."),
    case("M3-N06", "M3", "normal", "triton_embedding",
         [("selected = indices == row", "selected = (indices == row) & (row != PADDING_IDX)"),
          ("    result = tl.where(row == PADDING_IDX, 0.0, result)\n", "")],
         "accept", "padding_idx excluded through the selection mask (empty selection yields zero)."),
    case("M3-N07", "M3", "normal", "triton_kl_batchmean",
         [("BLOCK=SIZE, num_warps=8", "BLOCK=2 * SIZE, num_warps=8")],
         "accept", "Larger block with padded lanes correctly masked out of the reduction."),
    case("M3-N08", "M3", "normal", "triton_row_sum",
         [("BLOCK_COLUMNS=COLUMNS, num_warps=8", "BLOCK_COLUMNS=2 * COLUMNS, num_warps=8")],
         "accept", "Padded column lanes beyond the row length correctly masked."),
    case("M3-N09", "M3", "normal", "triton_dropout_backward",
         [("value = tl.where(keep, grad * SCALE, 0.0)", "value = grad * SCALE * keep.to(tl.float32)")],
         "accept", "Keep-mask applied multiplicatively; identical on finite inputs."),
    case("M3-N10", "M3", "normal", "triton_conv2d",
         [("x = tl.load(x_ptr + x_offset, mask=inside, other=0.0)",
           "x = tl.where(inside, tl.load(x_ptr + x_offset, mask=inside, other=1.0), 0.0)")],
         "accept", "Padding applied by a select after the masked load."),
    case("M3-N11", "M3", "normal", "triton_index_add",
         [("tl.store(out_ptr + offsets, base + source * ALPHA, mask=active)",
           "tl.store(out_ptr + offsets, tl.where(selected, base + source * ALPHA, base), mask=active)")],
         "accept", "Unselected rows copied explicitly; same result."),

    # =============================================================================================== M4 selection
    case("M4-E01", "M4", "error", "triton_mixtral_top2",
         [("second_index = tl.argmax(remaining, axis=0)", "second_index = tl.argmax(logits, axis=0)")],
         "detect", "First expert not excluded when selecting the second: returned index set is not the top-2."),
    case("M4-E02", "M4", "error", "triton_mixtral_top2",
         [("remaining = tl.where(expert == first_index, -float(\"inf\"), logits)",
           "remaining = tl.where(expert == first_index, 0.0, logits)")],
         "detect", "Exclusion sentinel 0 instead of -inf: rows whose other logits are negative reselect the first."),
    case("M4-E03", "M4", "error", "triton_mixtral_top2",
         [("selected = (expert == first_index) | (expert == second_index)", "selected = expert == first_index")],
         "detect", "Selection mask reports only the first expert."),
    case("M4-E04", "M4", "error", "triton_topk_topp_keep",
         [("higher = logits[None, :] > logits[:, None]", "higher = logits[None, :] >= logits[:, None]")],
         "detect", "Rank counts the token itself, so top-k keeps only k-1 tokens."),
    case("M4-E05", "M4", "error", "triton_topk_topp_keep",
         [("higher = logits[None, :] > logits[:, None]", "higher = logits[None, :] < logits[:, None]")],
         "detect", "Rank computed in ascending order: the k smallest logits are selected."),
    case("M4-E06", "M4", "error", "triton_topk_topp_keep",
         [("topk = rank < k", "topk = rank <= k")],
         "detect", "Top-k threshold off by one (k+1 tokens kept)."),
    case("M4-E07", "M4", "error", "triton_sort_unique",
         [("ordered = tl.sort(source, dim=0, descending=False)", "ordered = tl.sort(source, dim=0, descending=True)")],
         "detect", "Descending flag inverted."),
    case("M4-E08", "M4", "error", "triton_sort_unique",
         [("original = tl.argmax(matches.to(tl.int32), axis=1)", "original = columns")],
         "detect", "Returns positions in the sorted row instead of original indices."),
    case("M4-E09", "M4", "error", "triton_sort_unique",
         [("original = tl.argmax(matches.to(tl.int32), axis=1)", "original = tl.argmax(matches.to(tl.int32), axis=0)")],
         "detect", "Inverse permutation returned instead of the sorting permutation."),
    case("M4-E10", "M4", "error", "triton_max_pool2d",
         [("value_01 = tl.load(x_ptr + base + 1)", "value_01 = tl.load(x_ptr + base)")],
         "detect", "One window element skipped (copy-paste offset): max over a 3-element subset."),
    case("M4-E11", "M4", "error", "triton_max_pool2d",
         [("value_11 = tl.load(x_ptr + base + W + 1)", "value_11 = tl.load(x_ptr + base + W - 1)")],
         "detect", "Window element taken from outside the pooling window."),

    case("M4-N01", "M4", "normal", "triton_mixtral_top2",
         [("first_index = tl.argmax(logits, axis=0)", "first_index = tl.argmin(-logits, axis=0)")],
         "accept", "Top-1 found as argmin of negated logits; same index in the tie-free profile."),
    case("M4-N02", "M4", "normal", "triton_mixtral_top2",
         [("second_index = tl.argmax(remaining, axis=0)",
           "second_index = tl.argmin(tl.where(expert == first_index, float(\"inf\"), -logits), axis=0)")],
         "accept", "Second expert by argmin over negated remaining logits; same index tie-free."),
    case("M4-N03", "M4", "normal", "triton_mixtral_top2",
         [("    selected = (expert == first_index) | (expert == second_index)\n", ""),
          ("    values = tl.load(", "    selected = weights > 0.0\n    values = tl.load(")],
         "accept", "Selection mask derived from the strictly positive routing weights."),
    case("M4-N04", "M4", "normal", "triton_topk_topp_keep",
         [("higher = logits[None, :] > logits[:, None]", "higher = -logits[None, :] < -logits[:, None]")],
         "accept", "Same strict order expressed on negated logits."),
    case("M4-N05", "M4", "normal", "triton_topk_topp_keep",
         [("topk = rank < k", "topk = rank <= k - 1")],
         "accept", "Equivalent integer threshold."),
    case("M4-N06", "M4", "normal", "triton_topk_topp_keep",
         [("rank = tl.sum(higher.to(tl.int32), axis=1)",
           "rank = VOCAB - 1 - tl.sum((logits[None, :] < logits[:, None]).to(tl.int32), axis=1)")],
         "accept", "Rank from the count of smaller logits; identical on tie-free rows."),
    case("M4-N07", "M4", "normal", "triton_sort_unique",
         [("original = tl.argmax(matches.to(tl.int32), axis=1)",
           "original = tl.sum(tl.where(matches, columns[None, :], 0), axis=1)")],
         "accept", "Original index from the unique match by summation; identical tie-free."),
    case("M4-N08", "M4", "normal", "triton_sort_unique",
         [("    ordered = tl.sort(source, dim=0, descending=False)\n"
           "    matches = ordered[:, None] == source[None, :]\n"
           "    original = tl.argmax(matches.to(tl.int32), axis=1)\n"
           "    tl.store(values_ptr + row * COLS + columns, ordered)\n"
           "    tl.store(indices_ptr + row * COLS + columns, original)",
           "    position = tl.sum((source[None, :] < source[:, None]).to(tl.int32), axis=1)\n"
           "    tl.store(values_ptr + row * COLS + position, source)\n"
           "    tl.store(indices_ptr + row * COLS + position, columns.to(tl.int64))")],
         "accept", "Rank-and-scatter sort: different algorithm, same values and indices on distinct rows."),
    case("M4-N09", "M4", "normal", "triton_max_pool2d",
         [("tl.maximum(tl.maximum(value_00, value_01),", "tl.maximum(tl.maximum(value_00, value_11),"),
          ("tl.maximum(value_10, value_11))", "tl.maximum(value_10, value_01))")],
         "accept", "Window maximum combined in another order."),
    case("M4-N10", "M4", "normal", "triton_max_pool2d",
         [("    maximum = tl.maximum(tl.maximum(value_00, value_01),\n"
           "                         tl.maximum(value_10, value_11))",
           "    window = base + tl.arange(0, 2)[:, None] * W + tl.arange(0, 2)[None, :]\n"
           "    maximum = tl.max(tl.max(tl.load(x_ptr + window), axis=1), axis=0)")],
         "accept", "Window loaded as one 2x2 tile and reduced with tl.max."),

    # ties are outside the admitted tie-free profiles: the correct behaviour is to refuse a unique verdict
    case("M4-R01", "M4", "normal", "triton_mixtral_top2",
         _MIXTRAL_NO_TIE_CHECK,
         "refuse", "Tied router logits are outside the tie-free MOE profile; no verdict on the tie choice.",
         input_edits={"router_logits": {"sampler": {"randint": [-2, 3]}, "shape": [128, 8], "dtype": "float32"}}),
    case("M4-R02", "M4", "normal", "triton_mixtral_top2",
         _MIXTRAL_NO_TIE_CHECK + [("first_index = tl.argmax(logits, axis=0)",
                                   "first_index = tl.max(tl.where(logits == tl.max(logits, axis=0), expert, -1), "
                                   "axis=0)")],
         "refuse", "Highest-index tie-break on tied logits; ties outside the profile, so no unique verdict.",
         input_edits={"router_logits": {"sampler": {"randint": [-2, 3]}, "shape": [128, 8], "dtype": "float32"}}),
    case("M4-R03", "M4", "error", "triton_sort_unique",
         _SORT_NO_TIE_CHECK,
         "refuse", "Tied rows (argmax of matches duplicates indices) fall outside SORT finite_unique; must refuse.",
         input_edits={"x": {"sampler": {"randint": [-4, 5]}, "shape": [32, 32], "dtype": "float32"}}),
    case("M4-R04", "M4", "error", "triton_topk_topp_keep",
         _TOPK_NO_TIE_CHECK,
         "refuse", "Tied logits are outside the tie-free SAMPLING profile; must refuse.",
         input_edits={"logits": {"sampler": {"randint": [-4, 5]}, "shape": [64, 64], "dtype": "float32"}}),

    # =============================================================================================== M5 state
    case("M5-E01", "M5", "error", "triton_sgd",
         [("    tl.store(buffer_ptr + offsets, new_buffer, mask=active)\n", "")],
         "detect", "Momentum buffer never written back."),
    case("M5-E02", "M5", "error", "triton_sgd",
         [("tl.store(buffer_ptr + offsets, new_buffer, mask=active)",
           "tl.store(buffer_ptr + offsets, old_buffer, mask=active)")],
         "detect", "Old momentum buffer stored instead of the updated one."),
    case("M5-E03", "M5", "error", "triton_adam",
         [("    tl.store(m_ptr + offsets, new_m, mask=active)\n", "")],
         "detect", "Adam first moment not written back."),
    case("M5-E04", "M5", "error", "triton_adam",
         [("tl.store(vmax_ptr + offsets, new_vmax, mask=active)", "tl.store(vmax_ptr + offsets, new_v, mask=active)")],
         "detect", "AMSGrad running maximum stores the current second moment instead of the maximum."),
    case("M5-E05", "M5", "error", "triton_adam",
         [("    _increment_step_kernel[(1,)](step)\n", "")],
         "detect", "Step counter state not incremented."),
    case("M5-E06", "M5", "error", "triton_rmsprop",
         [("tl.store(gave_ptr + offsets, new_gave, mask=active)", "tl.store(gave_ptr + offsets, old_gave, mask=active)")],
         "detect", "Centered gradient average stored stale."),
    case("M5-E07", "M5", "error", "triton_rmsprop",
         [("    tl.store(buffer_ptr + offsets, new_buffer, mask=active)\n", "")],
         "detect", "RMSprop momentum buffer not written back."),
    case("M5-E08", "M5", "error", "triton_radam",
         [("tl.store(v_ptr + offsets, new_v, mask=active)", "tl.store(v_ptr + offsets, old_v, mask=active)")],
         "detect", "Second-moment state stored before its update."),
    case("M5-E09", "M5", "error", "triton_radam",
         [("    _increment_step_kernel[(1,)](step)\n", "")],
         "detect", "Step counter state not incremented."),
    case("M5-E10", "M5", "error", "triton_rprop",
         [("tl.store(prev_ptr + offsets, effective, mask=active)", "tl.store(prev_ptr + offsets, processed, mask=active)")],
         "detect", "Previous-gradient state stores the raw gradient instead of the sign-change-zeroed one."),
    case("M5-E11", "M5", "error", "triton_rprop",
         [("tl.store(eta_ptr + offsets, new_eta, mask=active)", "tl.store(eta_ptr + offsets, old_eta, mask=active)")],
         "detect", "Step-size state stored stale."),
    case("M5-E12", "M5", "error", "triton_adafactor",
         [("    tl.store(step_ptr, step)\n", "")],
         "detect", "Adafactor step state not written."),
    case("M5-E13", "M5", "error", "triton_adafactor",
         [("tl.store(col_ptr + col_ids, new_col)", "tl.store(col_ptr + col_ids, old_col)")],
         "detect", "Column factor state stored stale."),
    case("M5-E14", "M5", "error", "triton_mamba_ssu",
         [("tl.store(state_ptr + state_offset, next_state)", "tl.store(state_ptr + state_offset, state)")],
         "detect", "SSM state written back with the pre-update state."),

    case("M5-N01", "M5", "normal", "triton_sgd",
         [("    tl.store(theta_ptr + offsets, new_theta, mask=active)\n"
           "    tl.store(buffer_ptr + offsets, new_buffer, mask=active)",
           "    tl.store(buffer_ptr + offsets, new_buffer, mask=active)\n"
           "    tl.store(theta_ptr + offsets, new_theta, mask=active)")],
         "accept", "State stores reordered; complete write-back."),
    case("M5-N02", "M5", "normal", "triton_sgd",
         [("SIZE, BLOCK = 4096, 256", "SIZE, BLOCK = 4096, 1024"), ("num_warps=4", "num_warps=8")],
         "accept", "Larger block and more warps."),
    case("M5-N03", "M5", "normal", "triton_adam",
         [("SIZE, BLOCK = 4096, 256", "SIZE, BLOCK = 4096, 128")],
         "accept", "Smaller block, larger grid."),
    case("M5-N04", "M5", "normal", "triton_adam",
         [("    tl.store(theta_ptr + offsets, new_theta, mask=active)\n"
           "    tl.store(m_ptr + offsets, new_m, mask=active)",
           "    tl.store(m_ptr + offsets, new_m, mask=active)\n"
           "    tl.store(theta_ptr + offsets, new_theta, mask=active)")],
         "accept", "Moment state stored before the parameter."),
    case("M5-N05", "M5", "normal", "triton_adam",
         [("    _adam_kernel[(triton.cdiv(SIZE, BLOCK),)](",
           "    _increment_step_kernel[(1,)](step)\n    _adam_kernel[(triton.cdiv(SIZE, BLOCK),)]("),
          ("    _increment_step_kernel[(1,)](step)\n    return", "    return")],
         "accept", "Independent step increment launched before the elementwise update."),
    case("M5-N06", "M5", "normal", "triton_rmsprop",
         [("SIZE, BLOCK = 4096, 256", "SIZE, BLOCK = 4096, 512"), ("num_warps=4", "num_warps=8")],
         "accept", "Different block size and warp count."),
    case("M5-N07", "M5", "normal", "triton_rmsprop",
         [("    _rmsprop_kernel[(triton.cdiv(SIZE, BLOCK),)](",
           "    _increment_step_kernel[(1,)](step)\n    _rmsprop_kernel[(triton.cdiv(SIZE, BLOCK),)]("),
          ("    _increment_step_kernel[(1,)](step)\n    return", "    return")],
         "accept", "Step increment launched first."),
    case("M5-N08", "M5", "normal", "triton_radam",
         [("SIZE, BLOCK = 4096, 256", "SIZE, BLOCK = 4096, 64"), ("num_warps=4", "num_warps=2")],
         "accept", "Small blocks, fewer warps."),
    case("M5-N09", "M5", "normal", "triton_rprop",
         [("    tl.store(theta_ptr + offsets, new_theta, mask=active)\n", ""),
          ("    tl.store(eta_ptr + offsets, new_eta, mask=active)\n",
           "    tl.store(eta_ptr + offsets, new_eta, mask=active)\n"
           "    tl.store(theta_ptr + offsets, new_theta, mask=active)\n")],
         "accept", "Parameter stored after both state buffers."),
    case("M5-N10", "M5", "normal", "triton_adafactor",
         [("    tl.store(col_ptr + col_ids, new_col)\n    tl.store(step_ptr, step)",
           "    tl.store(col_ptr + col_ids, new_col)"),
          ("    tl.store(theta_ptr + offsets, new_theta)\n",
           "    tl.store(step_ptr, step)\n    tl.store(theta_ptr + offsets, new_theta)\n"),
          ("num_warps=8", "num_warps=4")],
         "accept", "Step state stored first; fewer warps."),
    case("M5-N11", "M5", "normal", "triton_mamba_ssu",
         [("    tl.store(state_ptr + state_offset, next_state)\n", ""),
          ("    tl.store(out_ptr + value_offset, out)",
           "    tl.store(out_ptr + value_offset, out)\n    tl.store(state_ptr + state_offset, next_state)")],
         "accept", "SSM state written after the output instead of before."),

    # =============================================================================================== M6 layout
    case("M6-E01", "M6", "error", "triton_transpose",
         [("source = out_col * COLS + out_row", "source = out_col * ROWS + out_row")],
         "detect", "Input row stride taken from the output tensor (ROWS) instead of the input (COLS)."),
    case("M6-E02", "M6", "error", "triton_transpose",
         _TRANSPOSE_STRIDED_ABI,
         "detect", "Non-contiguous input accepted but the kernel still assumes contiguous indexing.",
         input_edits={"x": _cm([32, 64])}),
    case("M6-E03", "M6", "error", "triton_matmul",
         [("a_ptr + rows[:, None] * K + inner[None, :]", "a_ptr + rows[:, None] * N + inner[None, :]")],
         "detect", "Row stride of b/out (N) used for a (K)."),
    case("M6-E04", "M6", "error", "triton_matmul",
         _MATMUL_STRIDED_ABI,
         "detect", "Strides passed but ignored: column-major operands read as row-major.",
         input_edits={"a": _cm([16, 32]), "b": _cm([32, 16])}),
    case("M6-E05", "M6", "error", "triton_matmul",
         _MATMUL_STRIDED_ABI + _MATMUL_A_STRIDES +
         [("b_ptr + inner[:, None] * N + cols[None, :]", "b_ptr + inner[:, None] * sam + cols[None, :] * sak")],
         "detect", "b indexed with a's strides (copy-paste of another tensor's strides).",
         input_edits={"a": _cm([16, 32]), "b": _cm([32, 16])}),
    case("M6-E06", "M6", "error", "triton_row_sum",
         _ROWSUM_STRIDED_ABI,
         "detect", "Column-major input accepted but rows read with contiguous indexing.",
         input_edits={"x": _cm([32, 256])}),
    case("M6-E07", "M6", "error", "triton_index_select",
         _INDEXSEL_STRIDED_ABI,
         "detect", "Column-major source accepted but gathered as if contiguous.",
         input_edits={"x": _cm([128, 32])}),
    case("M6-E08", "M6", "error", "triton_multiply_add",
         _MULADD_STRIDED_ABI + [("y = tl.load(y_ptr + offsets, mask", "y = tl.load(y_ptr + offsets * sx, mask"),
                                ("z = tl.load(z_ptr + offsets, mask", "z = tl.load(z_ptr + offsets * sz, mask")],
         "detect", "y indexed with x's stride; x is a stride-2 view, y is contiguous.",
         input_edits={"x": _s2(1024), "z": _s2(1024)}),
    case("M6-E09", "M6", "error", "triton_embedding",
         _EMB_STRIDED_ABI,
         "detect", "Column-major embedding table accepted but read with contiguous row indexing.",
         input_edits={"weight": _cm([64, 16])}),
    case("M6-E10", "M6", "error", "triton_sdpa",
         [("k = tl.load(k_ptr + k_base + source[:, None] * HEAD_DIM + feature[None, :])",
           "k = tl.load(k_ptr + ((batch * SOURCE + source[:, None]) * HKV + kv_head) * HEAD_DIM + feature[None, :])")],
         "detect", "Key cache indexed as BSHD while the call passes BHSD (layout confusion)."),
    case("M6-E11", "M6", "error", "triton_conv2d",
         [("inputs[\"x\"], inputs[\"weight\"], inputs[\"bias\"], out,",
           "inputs[\"x\"].permute(0, 2, 3, 1).contiguous(), inputs[\"weight\"], inputs[\"bias\"], out,")],
         "detect", "Channels-last copy handed to a kernel that indexes NCHW."),
    case("M6-E12", "M6", "error", "triton_conv2d",
         [("weight_offset = ((co * C_PER_GROUP + local_ci) * K_H + kh) * K_W + kw",
           "weight_offset = ((local_ci * C_OUT + co) * K_H + kh) * K_W + kw")],
         "detect", "Weight read with the transposed [C_in/g, C_out, kH, kW] layout."),
    case("M6-E13", "M6", "error", "triton_conv2d",
         [("x_offset = ((n * C_IN + ci) * H + ih) * W + iw", "x_offset = ((n * C_IN + ci) * W + ih) * H + iw")],
         "detect", "Height and width strides swapped for a non-square input."),
    case("M6-E14", "M6", "error", "triton_llama_rope",
         [("table = (batch * LENGTH + token) * WIDTH", "table = token * WIDTH")],
         "detect", "Per-batch cos/sin tables read as if broadcast over batch (batch stride assumed zero)."),
    case("M6-E15", "M6", "error", "triton_mamba_ssu",
         [("state_offset = value_offset[:, None] * DSTATE + coordinates[None, :]",
           "state_offset = ((batch * HEADS + head) * DSTATE + coordinates[None, :]) * DIM + channels[:, None]")],
         "detect", "SSM state read and written as [B, H, DSTATE, DIM] instead of [B, H, DIM, DSTATE]."),
    case("M6-E16", "M6", "error", "triton_paged_attention",
         [("cache_offset = ((((physical_block[:, None] * BLOCK_SIZE + slot[:, None]) * HKV + kv_head) * HEAD_DIM)",
           "cache_offset = ((((physical_block[:, None] * HKV + kv_head) * BLOCK_SIZE + slot[:, None]) * HEAD_DIM)")],
         "detect", "KV cache indexed as [blocks, heads, slots, dim] instead of [blocks, slots, heads, dim]."),
    case("M6-E17", "M6", "error", "triton_lora_shrink",
         [("offset = (((slice_index * LORAS + lora) * RANK + rank) * HIDDEN + hidden)",
           "offset = (((lora * tl.num_programs(2) + slice_index) * RANK + rank) * HIDDEN + hidden)")],
         "detect", "Stacked LoRA weights indexed [lora, slice] instead of [slice, lora]."),

    case("M6-N01", "M6", "normal", "triton_transpose",
         _TRANSPOSE_STRIDED_ABI + _TRANSPOSE_USE_STRIDES,
         "accept", "Correct strided transpose on a column-major input.",
         input_edits={"x": _cm([32, 64])}),
    case("M6-N02", "M6", "normal", "triton_transpose",
         _TRANSPOSE_STRIDED_ABI + _TRANSPOSE_USE_STRIDES,
         "accept", "Correct strided transpose on the declared contiguous input."),
    case("M6-N03", "M6", "normal", "triton_matmul",
         _MATMUL_STRIDED_ABI + _MATMUL_A_STRIDES + _MATMUL_B_STRIDES,
         "accept", "Correct strided matmul on column-major operands.",
         input_edits={"a": _cm([16, 32]), "b": _cm([32, 16])}),
    case("M6-N04", "M6", "normal", "triton_matmul",
         _MATMUL_STRIDED_ABI + _MATMUL_A_STRIDES + _MATMUL_B_STRIDES,
         "accept", "Correct strided matmul on contiguous operands."),
    case("M6-N05", "M6", "normal", "triton_row_sum",
         _ROWSUM_STRIDED_ABI + [("values = tl.load(x_ptr + row * COLUMNS + columns,",
                                 "values = tl.load(x_ptr + row * stride_row + columns * stride_col,")],
         "accept", "Correct strided row sum on a column-major input.",
         input_edits={"x": _cm([32, 256])}),
    case("M6-N06", "M6", "normal", "triton_index_select",
         _INDEXSEL_STRIDED_ABI + [("x_ptr + source_row * COLS + column",
                                   "x_ptr + source_row * stride_row + column * stride_col")],
         "accept", "Correct strided gather from a column-major source.",
         input_edits={"x": _cm([128, 32])}),
    case("M6-N07", "M6", "normal", "triton_multiply_add",
         _MULADD_STRIDED_ABI + [("y = tl.load(y_ptr + offsets, mask", "y = tl.load(y_ptr + offsets * sy, mask"),
                                ("z = tl.load(z_ptr + offsets, mask", "z = tl.load(z_ptr + offsets * sz, mask")],
         "accept", "Correct per-tensor strides with mixed stride-2 and contiguous operands.",
         input_edits={"x": _s2(1024), "z": _s2(1024)}),
    case("M6-N08", "M6", "normal", "triton_embedding",
         _EMB_STRIDED_ABI + [("values = tl.load(weight_ptr + row * WIDTH + feature)",
                              "values = tl.load(weight_ptr + row * stride_row + feature * stride_col)")],
         "accept", "Correct strided lookup from a column-major embedding table.",
         input_edits={"weight": _cm([64, 16])}),
    case("M6-N09", "M6", "normal", "triton_sdpa",
         [("k = tl.load(k_ptr + k_base + source[:, None] * HEAD_DIM + feature[None, :])",
           "k = tl.load(k_ptr + ((batch * SOURCE + source[:, None]) * HKV + kv_head) * HEAD_DIM + feature[None, :])"),
          ("q, k, v, out, HQ=HQ", "q, k.transpose(1, 2).contiguous(), v, out, HQ=HQ")],
         "accept", "Keys relaid out to BSHD by the call and indexed consistently by the kernel."),
    case("M6-N10", "M6", "normal", "triton_conv2d",
         [("x_offset = ((n * C_IN + ci) * H + ih) * W + iw", "x_offset = ((n * H + ih) * W + iw) * C_IN + ci"),
          ("inputs[\"x\"], inputs[\"weight\"], inputs[\"bias\"], out,",
           "inputs[\"x\"].permute(0, 2, 3, 1).contiguous(), inputs[\"weight\"], inputs[\"bias\"], out,")],
         "accept", "Channels-last input copy with matching NHWC indexing."),
    case("M6-N11", "M6", "normal", "triton_row_sum",
         _ROWSUM_STRIDED_ABI + [("values = tl.load(x_ptr + row * COLUMNS + columns,",
                                 "values = tl.load(x_ptr + row * stride_row + columns * stride_col,")],
         "accept", "Correct strided row sum on the declared contiguous input."),

    # =============================================================================================== M7 readings
    case("M7-N01", "M7", "normal", "inductor_gelu_erf",
         [("APPROXIMATE = \"none\"", "APPROXIMATE = \"tanh\"")],
         "accept",
         "The task chooses GELU approximate='tanh' and the implementation follows it: a legal alternative form must "
         "be accepted (note: like the base erf form, Inductor embeds rounded constants for sqrt(2/pi)).",
         plan_edits=_GELU_TANH_PLAN, input_edits={"approximate": {"const": "tanh"}}),
    case("M7-E01", "M7", "error", "inductor_gelu_erf",
         [("y = F.gelu(x, approximate=APPROXIMATE)", "y = F.gelu(x, approximate=\"tanh\")")],
         "detect", "Task chose exact (erf) GELU; the implementation uses the tanh form."),
    case("M7-E02", "M7", "error", "inductor_gelu_erf",
         [("APPROXIMATE = \"none\"", "APPROXIMATE = \"tanh\""),
          ("y = F.gelu(x, approximate=APPROXIMATE)", "y = F.gelu(x, approximate=\"none\")")],
         "detect", "Task chose tanh GELU; the implementation uses the exact erf form.",
         plan_edits=_GELU_TANH_PLAN, input_edits={"approximate": {"const": "tanh"}}),
    case("M7-N02", "M7", "normal", "inductor_cross_entropy",
         [(_CE_LOSS_OLD, _CE_LOSS_TORCH_RA)] + _CE_BOUNDARY,
         "accept", "Task selects weighted-smoothing reading R_A; PyTorch's summed weighted smoothing implements R_A.",
         plan_edits=_CE_RA_PLAN, input_edits=_CE_RA_INPUTS),
    case("M7-E03", "M7", "error", "inductor_cross_entropy",
         [(_CE_LOSS_OLD, _CE_LOSS_MANUAL_RB)] + _CE_BOUNDARY,
         "detect", "Task selects R_A; the implementation follows the diagnostic reading R_B.",
         plan_edits=_CE_RA_PLAN, input_edits=_CE_RA_INPUTS),
    case("M7-R01", "M7", "normal", "inductor_cross_entropy",
         [(_CE_LOSS_OLD, _CE_LOSS_TORCH_RA)] + _CE_BOUNDARY,
         "refuse", "R_A profile requires a task choice; with no task statement no unique verdict may be given.",
         plan_edits=_CE_RA_PLAN + [("/tasks/0/task_statement", "")], input_edits=_CE_RA_INPUTS),
    case("M7-R02", "M7", "normal", "triton_cast_bf16",
         [],
         "refuse", "CAST finite_rne is a task-choice profile; without a task statement the tool must refuse.",
         plan_edits=[("/tasks/0/task_statement", "")]),
    case("M7-R03", "M7", "normal", "inductor_avg_pool2d",
         [],
         "refuse", "POOL finite_avg2d_autograd needs the task's divisor reading; no statement means refusal.",
         plan_edits=[("/tasks/0/task_statement", "")]),
    case("M7-E04", "M7", "error", "inductor_avg_pool2d",
         [("return F.avg_pool2d(x, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0),\n"
           "                        ceil_mode=True, count_include_pad=True, divisor_override=None)",
           "return F.avg_pool2d(F.pad(x, (0, 1, 0, 1)), kernel_size=(3, 3), stride=(2, 2), padding=(0, 0),\n"
           "                        ceil_mode=False, count_include_pad=True, divisor_override=None)")],
         "detect", "Task chose divisors that exclude ceil overhang (9,6,6,4); implementation counts overhang (all 9)."),
]

# Cases whose edited call did not compile or run on the base declaration's input shapes (id, error). Kept, not dropped.
FAILED_TO_BUILD = []
# (Registration build log: all 166 cases compiled and ran with base output names/shapes/dtypes. M6-E17 first failed
# with a Triton NameError on the host global SLICES and was rewritten to tl.num_programs(2) before registration.)

# M7 limits recorded at registration: NORM rms_offset (Gemma offset) cannot be bound through the existing RMSNorm
# calls by plan edits alone (the profile requires the saved statistics output, which those calls do not return);
# RANKMEAN, ACT softplus_formula, SCHED, LABELS, INTERP and the other task-choice profiles have no reviewed real-call
# binding to mutate. They are therefore not included.
