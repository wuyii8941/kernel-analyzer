# Public PyTorch SDPA implementation review — 2026-10-19

The active ATTN contract covers one public PyTorch 2.10 `scaled_dot_product_attention` forward on finite dense
rank-3 or rank-4 real tensors with dropout disabled. It returns only the complete public output. The earlier draft
combined this task with FlashAttention bottom-right causal alignment, window and softcap options, FlexAttention,
backend LSE buffers, a base-2 Triton tutorial buffer and a complete VJP. That version remains under
`contracts/draft`; those interfaces no longer borrow authority from the public SDPA task.

The retained PyTorch 2.10 page fixes the forward formula, default and explicit scale, Boolean and additive masks,
upper-left causal alignment, mask/causal exclusion and GQA repetition. The retained versioned C++ source confirms the
math backend formula and its safe-softmax implementation. These sources do not establish one universal answer for an
all-masked row across every backend: the source produces zero in the math backend while the page's plain softmax
pseudocode does not itself state that special case. The reviewed profile therefore requires at least one participating
finite score in every query row and returns unsupported outside that range.

The profile also requires equal batch sizes, positive dimensions, a common f16/bf16/f32/f64 dtype and physical values
exactly representable in that dtype. It supports equal query and key/value head counts, singleton key/value head
broadcasting, or explicitly enabled contiguous-group GQA. Arbitrary leading batch broadcasting, dropout, nonfinite
data, positive-infinity masks, sparse or complex data, backend buffers and backward outputs remain outside scope.

Eighteen forward contract examples cover both scale choices, Boolean and additive masks, negative infinity, non-square causal
attention, GQA, singleton head broadcasting, rank four, empty rows and malformed or unsupported calls. Eight
deliberately wrong readings are rejected. An independent 768-bit Arb calculation checks five formula and shape cases.
Public PyTorch 2.10's math backend agrees in 20 combinations: four floating dtypes and five cases covering causal,
Boolean mask, additive mask, GQA and rank-3 head broadcasting. Complete-output tests reject missing, extra,
wrong-dtype and wrong-shape results.

## Complete public-call VJP

The active contract now has a separate `torch210_sdpa_defined_real_vjp` profile for the same forward task. It adds one
complete, same-dtype upstream tensor and returns `out`, `dq`, `dk` and `dv`. It does not expose or depend on an internal
LSE buffer. If P is the selected forward probability matrix, O=PV and U is upstream, the independently reviewed chain
rule is `D_ij=P_ij*(dot(U_i,V_j)-dot(U_i,O_i))`, followed by `dq=scale*Dk`, `dk=scale*D^Tq` and
`dv=P^TU`. Masks remove entries in both passes. With GQA, contributions from every query head are summed back into the
original key/value head, so returned gradient shapes remain exactly those of q, k and v.

Three additional contract examples check a nonzero two-key derivative, GQA head summation and a malformed upstream.
A separate 768-bit unshifted softmax implementation derives all four outputs directly for causal, Boolean-mask, GQA
and rank-3 cases. It shares neither the reference loops nor the max-shift computation. PyTorch 2.10's forced math
backend agrees for those four cases in f16, bf16, f32 and f64, including every coordinate of all three operand
gradients. Complete-output checks reject each missing field, an extra LSE field and an upstream with the wrong rank.
These checks support the mathematical VJP profile; they do not bind any fused attention kernel.

The strict reference uses exact products and sums and rigorous outward enclosures for square roots and exponentials.
Device rounding remains a separate comparison. The required `localhost:27891` proxy could not resolve while trying to
retrieve the original attention paper. No active claim rests on that unavailable paper; it would not determine the
versioned PyTorch mask, GQA, empty-row or return-value choices in any case.

## External public-call check

A fixed FlagGems 5.0.2 development batch exercised its exported autograd call on plain, causal, Boolean-mask,
additive-mask and GQA inputs, with two seeds each. Every call obtained the complete mathematical `out/dq/dk/dv`
reference before candidate comparison. All ten candidate calls returned all four fields. Six plain, causal or GQA
forwards closely matched PyTorch 2.10's forced math backend, but every call had at least one incompatible gradient;
the four masked forwards also differed. An uninstrumented square-shape check reproduced the noncausal, mask and GQA
differences, so rectangular input and launch recording do not explain them.

The retained FlagGems source supplies concrete explanations for three differences: a Boolean `True` is converted to
`-1e6` although the selected public task treats it as participating; the noncausal path converts scores to base-2
units before adding an unconverted additive mask; and the autograd wrapper calls backward with `attn_mask=None`.
The backward kernel also has no causal-mode argument and always executes its causal block split. The large GQA query
gradient remains an observed result without a separately established source-level cause.

The retained official Triton 3.6 tutorial provided a different acceptance implementation. The first float32 batches
were outside its own float16 test call and failed compilation; they remain recorded. With the tutorial's tested
float16, causal, length-128 interface, both head-dimension-64 calls completed and all four fields stayed within its
source test threshold of absolute `1e-2` (largest difference `9.77e-4`). Both head-dimension-128 calls required
109,568 bytes of shared memory on an RTX A6000 whose limit was 101,376 bytes, so they remain execution refusals.
The tutorial source itself marks noncausal tests as failing; this review does not extend its task beyond the tested
causal call.

These external runs validate that the new VJP reference can judge complete public training results and can preserve
unsupported executions. They do not establish complete Triton producer coverage, a formal call binding or a
statistical bias conclusion. Raw reports and the compact review summary are under
`corpus/external_batch/results/attention_*.json.gz`.
