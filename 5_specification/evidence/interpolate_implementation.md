# INTERP implementation review — 2026-10-19

The `torch210_nearest2d_size_autograd` profile was added after a real FlagGems public
`torch.nn.functional.interpolate` call exposed a common Triton path for which the active catalog had no complete
forward-and-gradient reference. The profile selects a narrow public task: PyTorch 2.10, rank-4 physical finite
floating input, explicit positive two-dimensional `size`, `mode='nearest'`, no scale factor or antialiasing, and the
complete input VJP. Other ranks, interpolation modes and scale-factor calls remain separate tasks.

The retained PyTorch 2.10 functional source establishes the public arguments, validates the options and dispatches
this call to `upsample_nearest2d`. It also states that legacy `nearest` follows OpenCV `INTER_NEAREST`, while
`nearest-exact` follows the PIL and Scikit-Image convention. The retained official decomposition supplies the
corresponding legacy coordinate rule: output coordinate `(oh,ow)` reads
`(floor(oh*H/OH), floor(ow*W/OW))`. This versioned rule is an implementation convention rather than a universal
definition of nearest-neighbor interpolation, so the profile requires the caller to state that task choice.

The forward is a gather whose integer coordinates do not depend on the input values. Its ordinary input VJP follows
independently: each upstream coordinate is added to the one input coordinate read by its output, and repeated reads
during upsampling accumulate. The adapter computes both arrays using exact rational values. It requires complete
physical input and upstream records plus every public option, validates rank, dtype, shapes and representability
before evaluating any coordinate, and refuses unselected tasks. Its output checker requires both complete tensors
with their declared dtype and shape.

Five deliberately wrong implementations distinguish the half-pixel `nearest-exact` coordinates, rounding instead
of flooring, exchanged height/width ratios, overwritten rather than accumulated gradient contributions, and an
omitted gradient result. Eight contract examples reject all five errors. A separate integer-loop implementation was
compared over f16, bf16, f32 and f64, two rank-4 shapes, scalar and rectangular up/down sizes. It agrees exactly with
the adapter and with PyTorch 2.10 CPU forward and backward on every selected case. These checks admit the reference
for its stated mathematical task; they do not by themselves establish that a particular Triton launch implements
that task or support a bias verdict.

Before GPU execution, a separate batch fixed two implementations, two shapes, four size patterns per
implementation, two seeds and one noncontiguous layout in each split. All sixteen CPU public calls matched the exact
reference in every forward and input-gradient coordinate. The FlagGems 5.0.2 development split captured its
`upsample_nearest2d_kernel` on all eight calls. Six contiguous calls matched exactly. Its two noncontiguous calls
preserved strides `(105,35,1,5)` but the forward differed in 150/162 and 144/162 coordinates. The retained candidate
source computes offsets as if input were contiguous and does not read tensor strides, which explains the observed
layout dependence. FlagGems replaces only the forward operator; the matching VJP comes from the native PyTorch
backward and is not evidence for a FlagGems Triton backward.

The separately fixed TorchInductor split used other dimensions, sizes and seeds. All eight calls, including its
noncontiguous cases, matched the exact reference for the complete output and VJP and captured Triton launches in
both phases. No formula or accepted domain was changed after seeing either implementation. The first development
report, the corrected scope/status report, the held-out report and a compact CPU summary are retained as
`corpus/external_batch/results/interpolate_nearest_*.json.gz`. These are validation records, not a formal call
binding or statistical bias conclusion.

After these files and records were finalized, the repository's complete required specification check passed,
including 942 registry/unit tests, all contract checks and the external-call report checks.
