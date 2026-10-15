"""Independent specification for loss normalisation across a gradient-accumulation window.   (v0.2)

Task requirement (the quantity a training step is meant to optimise when the per-token loss is averaged over all
non-ignored tokens of the GLOBAL batch): with micro-batches b = 1..B, per-token losses l_{b,i} and token counts n_b,

    L_window = ( sum_b sum_i l_{b,i} ) / ( sum_b n_b ),         dL_window/dtheta = ( sum_b sum_i dl_{b,i}/dtheta ) / ( sum_b n_b ).

This is a property of the accumulation loop, not of the cross_entropy kernel: a kernel that correctly averages over its
own micro-batch still produces the wrong window gradient if the loop averages the micro-batch means. The widely
discussed 2024 fix in HF transformers concerned exactly this loop (the window must be normalised by the total number
of effective tokens, not by averaging per-micro-batch means) - cite the HF/Unsloth posts and the transformers release
notes in the protocol; this file does not encode any implementation's behaviour, only the requirement above.

Known-wrong variant (for classification only): L_wrong = (1/B) sum_b ( sum_i l_{b,i} / n_b ).
It equals L_window iff all n_b are equal (or all micro-batch means are equal), which is why it passes most tests.
"""
from fractions import Fraction


def window_loss(per_token_losses):
    """per_token_losses: list over micro-batches of lists of exact per-token losses. Returns exact Fraction."""
    total = sum((Fraction(v) for batch in per_token_losses for v in batch), Fraction(0))
    count = sum(len(batch) for batch in per_token_losses)
    if count == 0:
        return None                                   # undefined (no tokens); recorded as a convention, not judged
    return total / count


def window_grad(per_token_grads):
    """per_token_grads: list over micro-batches of lists of exact per-token gradient contributions (scalars or vectors
    represented as lists). Returns the exact window gradient (sum over all tokens / total token count)."""
    flat = [g for batch in per_token_grads for g in batch]
    count = len(flat)
    if count == 0:
        return None
    if isinstance(flat[0], (list, tuple)):
        dim = len(flat[0])
        return [sum((Fraction(g[k]) for g in flat), Fraction(0)) / count for k in range(dim)]
    return sum((Fraction(g) for g in flat), Fraction(0)) / count


def wrong_variant_mean_of_means(per_token_losses):
    """known-wrong normalisation: average of per-micro-batch means (classification aid, never a valid reading)."""
    means = [sum((Fraction(v) for v in batch), Fraction(0)) / len(batch) for batch in per_token_losses if batch]
    return sum(means, Fraction(0)) / len(means) if means else None


def prop_split_invariance(per_token_losses, regroup):
    """Precondition: none. Re-splitting the same tokens into different micro-batches must not change L_window.
    regroup(flat_list) -> new list of batches over the same tokens."""
    flat = [v for batch in per_token_losses for v in batch]
    return window_loss(per_token_losses), window_loss(regroup(flat))
