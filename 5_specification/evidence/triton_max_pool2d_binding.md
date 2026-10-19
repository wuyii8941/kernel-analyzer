# Direct Triton max_pool2d binding review — 2026-10-19

This review covers `corpus/calls/triton_max_pool2d.py` and `corpus/call_bindings/triton_max_pool2d.json`. It applies only to one contiguous physical finite float32 NCHW input of shape `[2, 3, 16, 16]`, 2-by-2 maximum pooling with stride 2, zero padding, dilation 1, floor mode, and the complete values output of shape `[2, 3, 8, 8]`.

The wrapper checks the declared input, shape, dtype, device, layout and finite range. One launch uses 384 Triton programs. Each program maps one complete output coordinate to four adjacent input coordinates, loads all four values, takes their maximum and writes that output once. The binding plan records every pooling parameter, the full input pre-state, the full returned tensor and the unchanged input state. It binds explicitly to the reviewed POOL `finite_max2d_values` profile; the kernel name is not used as evidence. A GPU 3 sanity run was bit identical to PyTorch 2.10 functional `max_pool2d`.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable. The 9,216 complete output coordinates had exact zero device-to-program, program-to-task and device-to-task residuals. Repeated launches were bitwise identical. The report is `.cache/spec_trial_20261019_pool/triton_max_pool2d/baseline_report.json`.

A temporary controlled candidate added 0.25 after the correct window maximum. Every one of the 9,216 program-to-task coordinates excluded zero, with exact mean and maximum absolute residual 0.25. The complete device-to-task comparison had mean absolute residual about 0.25, and both the fixed-mean and aligned rule families detected a nonzero average effect. The temporary candidate and binding were removed after the run. Its report is `.cache/spec_trial_20261019_pool/triton_max_pool2d/controlled_plus_quarter_report.json`.

This binding does not cover another shape, dtype or layout; padding, ceil mode, indices, gradients, average or adaptive pooling; special input values; fused surrounding computation; or another Triton kernel with a similar name.
