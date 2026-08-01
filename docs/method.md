# Method

We analyze each concrete operator invocation in one complete model forward and
actual backward. Names, modules, and operator families are not proof units.

For invocation \(i\):

\[
U_i=(F_i,x_i,y_i,q_i,J_{F_i}(x_i)^Tq_i,B_i),
\]

where \(B_i\) is the backward program that actually ran. A unit closes only
when operands, saved tensors, upstream cotangent, output edge, shape, and dtype
match the analytic VJP.

The numerical test follows the mechanism used in arXiv:2510.04212:

\[
\text{directional local error}
\rightarrow
\text{coherent gradient carrier}
\rightarrow
\text{weight accumulation}.
\]

Local error alone is not a training bug. For state \(s\), a repair produces

\[
\Delta g_s=g_s^{repair}-g_s^{baseline}.
\]

We require an identity-copy sham to preserve loss and every parameter gradient
exactly. Cross-state direction is then tested with fixed, candidate-independent
coordinates, a U-statistic, leave-one-state-out uncertainty, coherence, and
rank-one energy. Only targets passing this gate enter a live-weight trajectory.

Precision is separated from optimization:

\[
e_{precision}=R_{low}-R_{32},\qquad
e_{optimization}=C_{low}-R_{low}.
\]

Current scope: Qwen3-1.7B, one full loss forward/backward, seq64/128/256,
BF16 and FP16 eager, with a BF16 Inductor open-region extension. Property
generalization has not started.

Next phase holds precision constant and tests implementation factors, beginning
with strict-FP32 eager versus strict-FP32 Inductor.
