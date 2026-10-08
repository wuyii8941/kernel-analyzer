"""Independent specification f for the training-program layer.  (v0.1)

Documentation (fetched 2026-10-08 unless marked):
  CosineAnnealingLR : eta_t = eta_min + (eta_max - eta_min)(1 + cos(pi * T_cur / T_max)) / 2 (closed form); the docs
                      note the recursive form and that it differs from the closed form when lr is set outside the
                      scheduler  [doc text not re-fetched; formula per torch.optim.lr_scheduler.CosineAnnealingLR]
  LinearLR          : factor from start_factor to end_factor linearly over total_iters, constant afterwards.
  HF cosine+warmup  : lr_lambda(step) = step / warmup for step < warmup, else max(0, 0.5 (1 + cos(pi * cycles * 2 *
                      progress))) with progress = (step - warmup) / (total - warmup)  [transformers get_cosine_schedule_with_warmup]
  clip_grad_norm_   : total p-norm over all gradients "as if they were concatenated into a single vector"; norm_type may
                      be inf; grad <- grad * max_norm / (total_norm + 1e-6), coefficient clamped at 1 (documented in
                      clip_grads_with_norm_); error_if_nonfinite raises if the total norm is nan/inf, otherwise the
                      gradients are scaled by the non-finite coefficient (documented).
  GradScaler        : https://docs.pytorch.org/docs/2.10/amp.html#gradient-scaling  [not re-fetched]: scale S; backward on
                      S*loss; unscale divides grads by S; if any grad is inf/nan the step is skipped and S <- S*backoff
                      (0.5), growth tracker reset; otherwise the step runs and after growth_interval (2000) consecutive
                      non-skipped iterations S <- S*growth_factor (2.0). init_scale 2^16.
  zero_grad / None  : see spec_optimizers O-D4.
  label shift       : for causal LM, labels[t] = input_ids[t+1], last position ignored (-100); the loss is averaged over
                      non-ignored labels in the GLOBAL batch (num_items_in_batch), see spec_accumulation.
  checkpoint        : https://docs.pytorch.org/docs/2.10/checkpoint.html  [not re-fetched]: recomputation must reproduce
                      the forward; with dropout fixed to zero the recomputed forward and the gradients must equal the
                      non-checkpointed ones in real arithmetic.

Ambiguities (ambiguities_phase2.md):
  SCH-A1 cosine: closed form vs recursive form (documented as different when lr is modified externally); the spec is the
         closed form; a candidate following the recursive form in the plain setting must coincide; divergence only when lr
         is set externally is "declared interpretation".
  SCH-A2 off-by-one: whether step counting starts at 0 or 1 (last_epoch = -1 initial) - the spec takes the documented
         convention that the first call to step() moves to epoch 1 and the initial lr is the epoch-0 value.
  CLIP-A1 the 1e-6 in the clipping coefficient is part of the documented formula, not an implementation accident.
  AMP-A1 which quantities count as "any grad inf/nan": all parameters handled by the optimizer at that unscale_ call.
  PROG-A1 cross-rank averaging: DDP averages per-rank gradients; with token-normalised losses and unequal token counts per
          rank this is NOT the global token average. The task requirement here is the global average (ranks weighted by
          token counts); DDP's plain mean is recorded as the known-wrong variant.
Numerics: exact rationals; cos and sqrt as rigorous intervals.
"""
from fractions import Fraction as F
import rigorous_mp as R
from rigorous_mp import I, iv, SpecInputError, SpecNotEstablished


# ------------------------------------------------------------------ schedulers
def cosine_annealing(eta_max, eta_min, T_cur, T_max):
    if T_max <= 0 or T_cur < 0:
        raise SpecInputError("T_max > 0 and T_cur >= 0 required")
    return I(eta_min) + I(F(eta_max) - F(eta_min)) * (1 + iv.cos(iv.pi * I(F(T_cur, T_max)))) / 2


def linear_lr(base_lr, start_factor, end_factor, total_iters, step):
    if total_iters <= 0:
        raise SpecInputError("total_iters must be positive")
    if step >= total_iters:
        return F(base_lr) * F(end_factor)
    return F(base_lr) * (F(start_factor) + (F(end_factor) - F(start_factor)) * F(step, total_iters))


def hf_cosine_with_warmup(base_lr, step, warmup, total, num_cycles=F(1, 2)):
    if step < warmup:
        return I(F(base_lr) * F(step, max(1, warmup)))
    if total <= warmup:
        raise SpecInputError("total must exceed warmup")
    progress = I(F(step - warmup, total - warmup))
    val = (1 + iv.cos(iv.pi * I(num_cycles) * 2 * progress)) / 2
    val = iv.mpf([max(val.a, 0), max(val.b, 0)])
    return I(base_lr) * val


# ------------------------------------------------------------------ clipping
def total_norm(grads, norm_type=2):
    flat = [F(v) for g in grads for v in g]
    if norm_type == float("inf"):
        return I(max(abs(v) for v in flat)) if flat else iv.mpf(0)
    p = F(norm_type)
    if p == 2:
        return iv.sqrt(I(sum((v * v for v in flat), F(0))))
    if p == 1:
        return I(sum((abs(v) for v in flat), F(0)))
    raise SpecInputError("this spec supports norm_type in {1, 2, inf}")


def clip_grad_norm(grads, max_norm, norm_type=2):
    """returns (clipped grads as intervals, total norm). coefficient = min(1, max_norm / (total + 1e-6))."""
    tn = total_norm(grads, norm_type)
    coef = I(max_norm) / (tn + I(F(1, 10 ** 6)))
    coef = iv.mpf([min(coef.a, 1), min(coef.b, 1)])
    return [[I(v) * coef for v in g] for g in grads], tn


# ------------------------------------------------------------------ GradScaler state machine
class ScalerState:
    def __init__(self, init_scale=2 ** 16, growth_factor=F(2), backoff_factor=F(1, 2), growth_interval=2000):
        self.scale = F(init_scale); self.growth = F(growth_factor); self.backoff = F(backoff_factor)
        self.interval = growth_interval; self.tracker = 0


def scaler_step(state, grads_scaled, has_nonfinite):
    """grads_scaled are the gradients of S*loss. Returns (unscaled grads or None if skipped, stepped: bool)."""
    if has_nonfinite:
        state.scale = state.scale * state.backoff
        state.tracker = 0
        return None, False
    unscaled = [[F(v) / state.scale for v in g] for g in grads_scaled]
    state.tracker += 1
    if state.tracker >= state.interval:
        state.scale = state.scale * state.growth
        state.tracker = 0
    return unscaled, True


# ------------------------------------------------------------------ label shift and counting
def shift_labels(input_ids, ignore_index=-100):
    return [input_ids[t + 1] for t in range(len(input_ids) - 1)] + [ignore_index]


def count_items(labels, ignore_index=-100):
    return sum(1 for row in labels for t in row if t != ignore_index)


def cross_rank_mean(per_rank_loss_sums, per_rank_counts):
    """global token average across ranks: sum of sums / sum of counts (PROG-A1)."""
    c = sum(per_rank_counts)
    if c == 0:
        raise SpecNotEstablished("no valid tokens on any rank")
    return sum((F(v) for v in per_rank_loss_sums), F(0)) / c


def wrong_variant_mean_of_rank_means(per_rank_loss_sums, per_rank_counts):
    means = [F(s) / c for s, c in zip(per_rank_loss_sums, per_rank_counts) if c > 0]
    return sum(means, F(0)) / len(means) if means else None


# ------------------------------------------------------------------ checkpoint / recompute property
def prop_checkpoint_equivalence(fwd_plain, fwd_ckpt, grads_plain, grads_ckpt):
    """Precondition: dropout disabled, same inputs and parameters. In real arithmetic forward values and gradients are
    identical; for floating-point candidates report the elementwise differences (the mean-effect statistics judge them)."""
    df = [F(a) - F(b) for a, b in zip(fwd_plain, fwd_ckpt)]
    dg = [F(a) - F(b) for a, b in zip(grads_plain, grads_ckpt)]
    return df, dg
