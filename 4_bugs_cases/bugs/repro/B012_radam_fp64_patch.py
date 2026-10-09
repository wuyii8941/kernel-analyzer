"""Runtime fix for B012: RAdam's single-tensor step computes rho_t and both bias corrections from a float64 step.

rho_inf, rho_t and the bias corrections depend only on beta1, beta2 and the step, not on the parameters; evaluating
them in float64 (the step as a float64 0-dim tensor when it is a tensor, i.e. capturable or compiled) removes the
cancellation and the float32 rounding of beta2 from the rectification decision.  The parameter update stays float32.

    import B012_radam_fp64_patch; B012_radam_fp64_patch.apply()
or, for tool runs:  KA_PRELOAD=bugs/repro/B012_radam_fp64_patch.py python scripts/tool_spec_check.py ...
"""
PATCHED = []


def apply():
    import inspect
    import textwrap

    import torch.optim.radam as radam

    if PATCHED:
        return
    src = textwrap.dedent(inspect.getsource(radam._single_tensor_radam))
    edits = [
        ("        step = step_t if capturable else _get_value(step_t)\n",
         "        step = step_t if capturable else _get_value(step_t)\n"
         "        step64 = step.to(torch.float64) if torch.is_tensor(step) else step\n"),
        ("bias_correction1 = 1 - beta1**step\n", "bias_correction1 = 1 - beta1**step64\n"),
        ("bias_correction2 = 1 - beta2**step\n", "bias_correction2 = 1 - beta2**step64\n"),
        ("rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2",
         "rho_t = rho_inf - 2 * step64 * (beta2**step64) / bias_correction2"),
    ]
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, radam.__file__, "exec"), radam.__dict__, ns)
    radam._single_tensor_radam = ns["_single_tensor_radam"]
    PATCHED.append("_single_tensor_radam")
