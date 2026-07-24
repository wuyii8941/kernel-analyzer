# Qwen3 fused SiLU-multiply repair v0.1 failure

The frozen v0.1 treatment did not complete and receives zero coverage credit.
At selected call 0, the live generated kernel passed two equal-numel buffers
through different logical views.  The replacement attempted direct shaped
multiplication and failed with a dimension mismatch (`167` versus `668`).  No
repaired scorer output was observed.

This is a treatment-construction failure, not a null operator effect.  A
revision may restore the generated kernel's flat elementwise correspondence by
reshaping the second buffer to the mutated buffer's logical shape.  The family,
state, call indices and endpoints must remain unchanged.
