# Qwen3 GRPO natural one-step transition design v0.1

Date: 2026-07-18
Status: pre-execution design; execution manifest is intentionally not frozen
until the B/step-29 snapshot passes an independent audit

## 1. Question

From the exact pre-minibatch state selected before held-out results were observed,
does replacing the eager Qwen scorer forward by its tracked Inductor realization
change the natural next training state under the same GRPO loss, backward, AMP,
gradient clipping, optimizer and scheduler?

This is Gate T from
`ORACLE_EXTERNAL_COMPLETENESS_GATES_V0_1_2026-07-18.md`. It is not a long-run
training experiment and it does not ask whether either implementation is
mathematically correct.

## 2. Selected state

- source: held-out transport run B;
- pre-minibatch optimizer step: 29;
- selection: the next `random.Random(20260718)` draw after the three prompt-block
  and three training-seed draws, from the ten measured steps;
- event conditioning: none;
- required snapshot schema:
  `forkcert.full-pre-minibatch-transition-state.v0.1`.

The state is eligible only if it contains model parameters/buffers, optimizer,
scheduler, AMP scaler, Trainer counters, Python/NumPy/Torch RNG, the exact target
minibatch and the exact input history at measured steps
`[2,5,8,11,14,17,20,23,26,29]`.

## 3. Implementation boundary

### Reference arm A

The restored model scorer executes through the eager Trainer realization. GRPO
loss construction, backward, unscale, gradient clipping, optimizer, scaler and
scheduler use the ordinary restored training path.

### Candidate arm B

The same restored model object is wrapped by tracked `torch.compile`/Inductor for
the scorer forward. The saved history input packs are replayed in order to rebuild
the declared specialization history. GRPO loss construction, backward, unscale,
gradient clipping, optimizer, scaler and scheduler remain the same as arm A.

Thus B is a **scorer-forward intervention inside a natural training transition**.
It is not a claim about compiling the complete optimizer or the entire training
program. OptimizedModule parameters must be the same parameter objects registered
with the restored optimizer.

### Self-replay arms

A and B are each executed twice in fresh processes from the same snapshot. These
are not extra sampled training states; they measure replay/runtime variability at
one state. Candidate warm-up and history reconstruction are excluded from the
measured transition but included in the execution-identity ledger.

## 4. Coupling and open randomness

Before every arm:

- restore all saved state components;
- verify exact model, optimizer, scheduler, scaler, Trainer and minibatch hashes;
- restore Python, NumPy, Torch CPU and CUDA RNG immediately before the measured
  transition;
- use one visible GPU, deterministic-algorithm warning mode, disabled dropout,
  fixed SDPA MATH and the recorded AMP mode;
- preserve the same batch ordering, tensor shapes, strides and dtypes.

No algorithmic randomness is intentionally left open. Any difference between
same-arm repeats is runtime/replay variability and blocks a deterministic-shift
claim. Compiler autotuning and cache state belong to the execution protocol and
must be recorded rather than called generic floating-point noise.

## 5. Required anchors before stepping

The transition is invalid unless:

1. A1 and A2 reproduce the eager target scorer tensor and pre-step loss exactly;
2. B1 and B2 reproduce the candidate target scorer tensor and graph-history
   identity exactly;
3. A and B have identical non-implementation state immediately before the
   measured scorer call;
4. candidate history replay changes no parameter, buffer, gradient, optimizer,
   scaler, scheduler or RNG state;
5. the candidate measured call invokes tracked compiled code without eager
   fallback;
6. the compiled wrapper and optimizer refer to the same underlying parameters.

Failure returns `INVALID`; it is not a null transition effect.

## 6. Endpoints

### Continuous discrepancy

- complete current-token log-probability tensor;
- scalar GRPO loss;
- gradient summaries before unscale and before/after gradient clipping;
- per-parameter and aggregate update distances;
- cosine/alignment summaries between A and B update vectors;
- scaler value and optimizer/scheduler counters.

Distances are reported both absolutely and relative to the reference update norm.
A single global norm must not replace per-layer or direction-sensitive summaries.

### Semantic events

- per-token GRPO clipping decisions and `0->1`/`1->0` directions;
- global gradient-clipping trigger;
- AMP overflow and optimizer-step skip;
- scheduler/scaler discrete transition;
- any non-finite gradient or parameter event.

The event ledger is valid even when no event differs. Zero event disagreement
does not imply zero continuous update discrepancy.

### Next-state transition

The next state includes model parameters/buffers, optimizer moments/counters,
scheduler and scaler. A strict paired transition-compatibility endpoint rejects
compatibility when a valid bitwise A/B next-state difference is observed; it is
an implementation-relative impact endpoint, not numerical correctness. Practical
harm requires an independently declared acceptance threshold or downstream
endpoint and is out of scope here.

## 7. Estimands and verdict ledgers

- `runtime/replay variability`: A1 versus A2 and B1 versus B2;
- `implementation-relative transition effect`: paired B-minus-A next-state
  difference when both arms are self-stable;
- `semantic-event disagreement`: directional and total event differences;
- `controlled natural-step impact`: whether the scorer-forward intervention
  changes the valid next-state transition;
- `correctness`: `UNINSTANTIATED` without an external authority;
- `population inference`: `NOT CLAIMED` for this selected single state;
- `operator attribution`: `NOT CLAIMED`; treatment is scorer-forward realization,
  not a source operator.

## 8. Kill and downgrade conditions

- Model-only restore or regenerated minibatch kills the natural-state claim.
- Missing/changed compiler history kills candidate identity.
- Different optimizer parameter objects kill the optimizer-transition claim.
- Same-arm instability moves the implementation effect from deterministic shift
  to stochastic/indeterminate under the current repeat budget.
- An A/B parameter distance with no event difference remains update discrepancy,
  not semantic harm.
- A clipping difference with equal next state is a semantic event difference but
  not an update-effect result; the equality must be independently verified.
- A nonzero update difference cannot be promoted to non-convergence, accuracy loss
  or longer training.
- Because the intervention compiles the scorer forward as a whole, it cannot be
  named an operator causal effect.

## 9. Relationship to Oracle completeness

Passing this design would connect the numerical-discrepancy, semantic-event and
one-step-transition ledgers for one natural Qwen state. Combined with Gate P it
strengthens subject-level external evidence, but it still leaves checkpoint/model/
hardware transport, real-model correctness authority, stochastic law endpoints,
realization-preserving operator attribution and long-run validation open.
