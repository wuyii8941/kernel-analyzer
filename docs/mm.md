# Confirmed MM bias

The only complete natural case is one concrete seq128 `lm_head` backward edge.

Forward:

\[
Y=XW^T.
\]

Actual input-gradient edge:

\[
dX=GW.
\]

The same saved \(W\) and upstream cotangent \(G\) are used in the native and
reference arms. Disabling BF16 reduced-precision reduction removes about
91.05% of the residual RMS. The remainder is consistent with the GEMM
reduction/FMA tree and accumulation order.

Independent full-parameter experiments show a coherent cross-state carrier.
In the 32-step live-weight experiment:

- every natural repair leaves the same-step forward loss unchanged;
- the parameter-gradient carrier is nonzero in 32 / 32 steps;
- FP32 master weights diverge in 32 / 32 steps;
- materialized BF16 weights diverge in 32 / 32 steps;
- final FP32 pair L2 distance is 0.00487622;
- final BF16 materialized pair L2 distance is 0.00536579.

This is not the FlashAttention kernel or the same source bug. The shared claim
is only the causal structure:

\[
\text{biased local arithmetic}
\times
\text{coherent carrier}
\rightarrow
\text{accumulating weight error}.
\]
