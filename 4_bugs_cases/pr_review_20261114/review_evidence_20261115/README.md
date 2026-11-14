# Reviewer follow-up evidence

This adds new device evidence to the first review commit 1d2d480. Original records are preserved.

- The complete FLA file passes 58 cases with an 80% allocator cap. Earlier allocator OOMs and the diagnostic SIGSEGV remain recorded; the cause of the SIGSEGV is not established.
- Based/ReBased caller checks are separate controls, not new bias cases. The ReBased output reference is independent quadratic attention from physical producer values; its gradients are checked only for finiteness.
- Attorch's corrected public selector includes all six new zero-residual tests and reports 354 passes. The original 348-case selector omitted those six.
- PR8969 already fixes dense attention. The current source delta changes only the two varlen functions and is based on its fixed, unmerged head, not uncorrected master. The patch is a coordination/follow-up candidate, not a standalone submission to master.
- The 16-case NVIDIA suite contains eight invariant controls and eight nonzero-Q/K/V cases. Original PR head: four failures; patched: 16 passes. Reverting only varlen DKV: two failures, both dK in the new p=.5 varlen cases. Old Q=0 controls pass this mutation and therefore do not independently protect dK.
- The first random BF16 cases failed dV even at p=0. This existing probability-format conversion is distinct from the dropout formula. Protocol v2 retains the inputs, FP64 oracle and dQ/dK thresholds, and adds only the declared dV allowance u*abs(P_drop).T@abs(dOut), u=eps/2, on normal probabilities. Remaining dot/output arithmetic uses the original dtype tolerances. This is a scoped test allowance for a known conversion, not a claim of a certified whole-kernel error bound. Initial failures remain included.
- dBias is not GPU-validated by this suite. Forward/backward RNG integration and Hygon hardware are not claimed.
- A broader Liger unit attempt was interrupted after the user accepted the focused KL validation and requested no further large-scale training. It is not a full make-test pass. No convergence job started. The actual make-checkstyle command passed without modifying any of the 500 Python files.

Public submission text can cite the immutable commit containing this directory. No upstream issue, PR or comment is created by this evidence export.
