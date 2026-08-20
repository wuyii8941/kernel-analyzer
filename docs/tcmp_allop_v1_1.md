# TCMP all-operator protocol v1.1

This is a pre-orbit statistical correction to `tcmp_allop_v1`. The existing
FP32-storage all-operator screens remain valid and are not orbit measurements.

Each measured state uses one separate default execution and eight non-default
semantic-orbit replays. The eight replays are frozen into disjoint A/B halves.
The primary orbit-mean persistence statistic uses cross inner products between
the two half means; the default execution is excluded from both halves. This
prevents finite-K orbit variance from being counted as mean energy and avoids
correlating the default residual with its own mean estimator.

Permutations are applied only to operands captured at the exact kernel replay
boundary. Model inputs are never permuted. For GEMM, both sides of the reduction
axis are permuted together, preserving the real-valued operation. The reported
object is a tiling-conditional orbit mean unless tile/chunk geometry is itself
included as a declared orbit dimension. The mathematical target uses FP64 or
an exact accumulator.

Persistence trajectories contain 32 live steps and report the full prefix
curve, including normalized `A(T)/sqrt(T)`. A 16-step-only result is
horizon-limited rather than a final persistence verdict.

The primary feedback null time-shuffles the natural residual. A second control
compares the final drift direction from the natural perturbation with an
RMS-matched random perturbation. High cosine indicates a dynamics-dominated
mode and cannot be credited as operator-specific persistence.

Held-out units are mechanically enumerated from all closed F+B units with a
matched repair in the frozen new-model roster. They are either all measured or
sampled once with a committed seed before predictor values are revealed, and
must include predictor-negative units.
