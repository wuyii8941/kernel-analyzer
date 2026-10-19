# PyTorch 2.10 Triton BSR softmax values implementation review — 2026-10-19

The active BSR contract covers one narrow task: the complete values computation performed by
`torch.sparse._triton_ops.bsr_softmax` in PyTorch 2.10 for a finite, canonical two-dimensional BSR tensor. For every
scalar row within a block row, it applies softmax jointly over all scalar values in every specified block. Explicit
zeros inside a specified block participate; unspecified matrix entries do not. The result is the complete value
buffer in the input block order.

The retained PyTorch sparse documentation defines a specified BSR element as a whole block and defines unspecified
sparse entries through the sparse fill value. The sparse softmax documentation says unspecified entries are ignored,
equivalently treated as negative infinity. The retained PyTorch 2.10 Triton source makes the remaining detail direct:
for one scalar row it loads `row_nnz * col_block` consecutive values across every specified block, converts them to
float32, computes a stable max and exponential sum, and stores every quotient. The parsed `bsr_softmax` function body
in the installed PyTorch 2.10.0+cu128 package is identical to the retained source body for this task.

This review removed the former combined profile. That draft also covered `bsr_dense_mm` and BSR `sampled_addmm`, and
its sampled-addmm rule depended on whether the documentation's `spy(input)` meant stored positions or numerically
nonzero values. Those are separate calls with different inputs and outputs. The broader contract and adapter remain
under `contracts/draft`; none of those branches is active.

Sixteen contract examples cover multiple blocks per row, multiple scalar rows per block, explicit zeros, empty block
rows, three physical dtypes, complete values, canonical indices and rejected structures. Six deliberately wrong
readings are rejected. A separate 768-bit Arb computation checks the exponential ratios without calling the adapter,
and 15 public PyTorch dense-softmax formula cases cover three dtypes and five patterns. Complete-output checks reject
missing, extra, wrong-shape and wrong-dtype results.

The installed official Triton function was also called on GPU 3 for two 16x16 blocks in each block row. Its complete
value buffer was bitwise identical to the independently written direct Triton core; the public PyTorch formula differed
by at most `4.47e-8`. The contract keeps exact real softmax as f, so the source's float32 conversion, exponential
approximation, reduction order and output cast remain measurable execution effects.

The active profile does not certify sparse tensor allocation, copied indices, batches, hybrid sparse tensors, matrix
multiplication, sampled addmm, gradients, special values, storage or alias behavior.
