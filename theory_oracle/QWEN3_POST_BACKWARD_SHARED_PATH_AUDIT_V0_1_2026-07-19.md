# Qwen3 post-backward shared-path audit v0.1

## Scope result

For the frozen Qwen3 natural transition, the eager/compiled treatment changes
the model scorer and its autograd graph.  The following operations execute
after that contrast through the same PyTorch path in both arms:

1. native FP16 GradScaler unscale and nonfinite detection;
2. global gradient-norm calculation, clip predicate and gradient scaling;
3. fused AdamW step;
4. GradScaler update and the linear scheduler step.

They are therefore in the full training program but are not separate
eager-versus-compiled implementations in this subject.  They must not be added
to the denominator of compiler discrepancy-generating operators.

## Their Oracle role

These shared operations are still essential to the transition Oracle:

- unscale and norm calculation propagate model-produced gradient differences;
- the clip predicate is a boundary converter;
- clipping transforms the continuous gradient discrepancy;
- AdamW transforms it into optimizer-state and parameter-update discrepancy;
- overflow/skip and scheduler logic are semantic/state-transition endpoints.

Thus “not a treatment operator” does not mean “irrelevant” or “covered by the
backward inventory.”  It means their causal question is propagation through a
shared mechanism, not discrepancy generation by two alternative
implementations.

## Selected-state evidence

At heldout-transport-B step 29, the eager and compiled pre-clip norms were
10.342905 and 10.356600.  Both arms triggered clipping, both remained finite,
both kept the AMP scale at 65536, neither skipped the optimizer step, and their
post-step scaler and scheduler digests were identical.  The clipped-gradient
L2 discrepancy was 0.00960425 and the parameter-update L2 discrepancy was
3.11114e-05.

This shows continuous discrepancy propagation without a gradient-clip,
overflow, skip or scheduler fork at this state.  It does not estimate event
probabilities or prove that the shared post-backward path is harmless near a
different boundary.

## Coverage consequence

Full-training coverage needs two linked ledgers:

- implementation-treatment coverage for forward/backward generated and
  external calls;
- shared-path propagation and boundary coverage for gradient control, AMP,
  optimizer and scheduler.

Combining the ledgers into an undifferentiated “all ops repaired” fraction would
be conceptually wrong because the two rows answer different causal questions.
