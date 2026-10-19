# PACK implementation review — 2026-10-19

The former PACK draft combined three related but different tasks: constructing a document mask, resetting
position_ids, and executing packed attention. FlashAttention's public flash_attn_varlen_func returns the packed
attention result; it does not return the mask or position IDs. That draft and adapter are retained under
contracts/draft/PACK_document_construction_20261019.yaml and
contracts/draft/PACK_adapter_document_construction_20261019.py, and cannot be selected at runtime.

The active contract follows the retained FlashAttention v2.8.3 interface source. cu_seqlens_q and
cu_seqlens_k split q and k/v into matching independent sequences. The causal profile uses the documented
bottom-right rule j <= i + seqlen_k - seqlen_q; a fully masked row returns zero. The bidirectional profile admits
every key inside the matching sequence. Query heads use contiguous GQA groups and scores use the explicit scale or
1/sqrt(headdim).

The reviewed range is deliberately smaller than the API: finite physical f16 or bf16 q/k/v, positive sequence
lengths, exact int32 cumulative arrays, exact represented maxima, dropout zero, no local window, no softcap, no ALiBi,
no block table and no optional testing outputs. The complete output dtype, shape and every coordinate are required.
The contract does not claim position IDs, document masks, LSE, gradients or other versions.

Fifteen examples are consistent and eight independent erroneous formulas are rejected. Ten 768-bit Arb evaluations
use a separate element loop and enclose every coordinate returned by the reference. Three dense float64 PyTorch
calculations independently cover causal and bidirectional execution, unequal q/k lengths, GQA and default scale.
Complete-output and refusal tests check all required fields, dtype, shape, coordinates and unsupported options.

As a separate execution cross-check, the installed vLLM 0.16 vendored FlashAttention wrapper was run on GPU 3 with
q lengths [2,4], k lengths [5,3], four query heads, two KV heads and head dimension 16. Both causal and
bidirectional f16/bf16 calls produced finite outputs, preserved every input and agreed with a dense formula within:

| dtype | mode | maximum absolute difference | mean absolute difference |
|---|---:|---:|---:|
| f16 | bidirectional | 0.00015005 | 0.00002716 |
| f16 | causal | 0.00017235 | 0.00001899 |
| bf16 | bidirectional | 0.00129876 | 0.00020185 |
| bf16 | causal | 0.00093743 | 0.00012792 |

The first empty bottom-right causal row was exactly zero in both dtypes. This installed wrapper is corroborating
execution evidence; it is not represented as an upstream v2.8.3 binary or used to replace the retained semantic
source.
