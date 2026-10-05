"""torch.compile(optimizer.step) for the optimizers not in tool_spec_cases_inductor2 (Adafactor, Rprop), plus
step-dependent hyperparameters at the edges (the B012 class: coefficients of the step count evaluated in float32 on
the compiled path).  Same OptimStep case as inductor2: one step from a populated state, spec = eager float64.
"""

from tool_spec_cases_inductor2 import OptimStep

CASES = [
    OptimStep("opt_adafactor", "Adafactor", dict(lr=1e-2), warm=3),
    OptimStep("opt_adafactor_wd_max", "Adafactor", dict(lr=1e-2, weight_decay=0.05, maximize=True), warm=3),
    OptimStep("opt_adafactor_step1", "Adafactor", dict(lr=1e-2), warm=0),
    OptimStep("opt_adafactor_decay_m05", "Adafactor", dict(lr=1e-2, beta2_decay=-0.5, eps=(1e-3, 1e-2)), warm=5),
    OptimStep("opt_rprop", "Rprop", dict(lr=1e-2), warm=3),
    OptimStep("opt_rprop_etas", "Rprop", dict(lr=1e-2, etas=(0.3, 1.5), step_sizes=(1e-4, 1.0), maximize=True), warm=4),
    OptimStep("opt_adam_b2_9999_step1", "Adam", dict(lr=1e-2, betas=(0.9, 0.9999)), warm=0),
    OptimStep("opt_adam_b2_99999_step3", "Adam", dict(lr=1e-2, betas=(0.99, 0.99999)), warm=2),
    OptimStep("opt_adamw_amsgrad_b2_9999", "AdamW", dict(lr=1e-2, betas=(0.9, 0.9999), amsgrad=True,
                                                         weight_decay=0.1), warm=2),
    OptimStep("opt_nadam_md_big", "NAdam", dict(lr=1e-2, momentum_decay=0.5), warm=3),
    OptimStep("opt_adamax_b2_9999", "Adamax", dict(lr=1e-2, betas=(0.99, 0.9999)), warm=1),
    OptimStep("opt_asgd_lambd", "ASGD", dict(lr=1e-2, lambd=1e-2, alpha=0.5, t0=1.0), warm=3),
    OptimStep("opt_adagrad_lrdecay_big", "Adagrad", dict(lr=1e-2, lr_decay=0.5, eps=1e-6), warm=4),
]
