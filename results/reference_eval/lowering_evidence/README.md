# Lowering evidence (Triton 3.6.0, sm_86)

Compiled artifacts behind the lowering rules of `src/kernel_analyzer/reference_eval/emulate.py`, kept so that
the rules (and the choices taken from device outputs) can be checked independently. Each directory has the
captured `kernel.ttir`, `kernel.ttgir`, `kernel.ptx`, the SASS of the captured cubin (`cuobjdump -sass`), and the
launch metadata (compile options, hashes). Large files are gzip-compressed. The operands are not committed. The
capture packages are kept under `.cache/` (`torchao_capture`, `liger_kernels`, `heldout/`) and can be regenerated
with the capture scripts and their seeds.

| Directory | Rule | Where to look |
|---|---|---|
| `layernorm_rows` | ptxas contracts the end of `div.full` (reciprocal × numerator) into the following plain `sub.f32` | `kernel.sass` line 105: `FFMA R7, R11, -R7, R4` (R11 = `MUFU.RCP`) |
| `layernorm_rows` | a product feeding a reduction: the first combine that sees the raw product is an fma | `kernel.sass` line 115: `FFMA R6, R14, R14, R5` (after the first `SHFL.BFLY`). The same appears in `kernel.ptx` as `fma.rn.f32 %r45, %r42, %r42, %r44` |
| `inductor_adamw` | a scalar product reaching an add through `tt.splat` is contracted by LLVM. The `div.full` product before it stays a plain multiply | `kernel.sass` lines 475–477: `FFMA R16, R15, 9.99999994e-09, R0`, with `R0` from `FMUL R0, R13, R16` |
| `liger_cross_entropy` | reductions after `tt.reshape allow_reorder` combine in the register order of the source layout | `kernel.ttgir` lines 68 and 81 (`#blocked` → `#blocked1`) |
| `torchao_adamw8bit` | the two choices taken from the device output (development captures): `arith.subf@418` (`%tmp250 = %tmp249 - %tmp63`, `%tmp63` from `arith.divf`) not contracted; `arith.subf@114` (`%tmp32 = %tmp22 - %tmp31_51`) folds the second product | PTX: `%tmp22` = grad² stays a plain `mul.f32` (`kernel.ptx.gz` line 1272 and following), so the other product was folded. The `subf@418` decision is in SASS only and was not located by hand. It was frozen and checked on held-out inputs (`emulation_heldout.json`) |
