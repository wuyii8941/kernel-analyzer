# Population decision capability validation

`population_decision_capability.json` is the retained first development run. Its intended
wide-support abstention scenario accidentally used a support bound that was still narrow enough
for the production Hoeffding rule to establish equivalence, so the validation harness correctly
reported `FAIL`.

The scenario definition was corrected without changing the production decision code.
`population_decision_capability_fixed.json` is the completed run: a nonzero small effect is
`EQUIVALENT`, a large effect is `NON_EQUIVALENT`, and an all-zero observed sample with a sufficiently
wide predeclared population support is `INCONCLUSIVE`. Both files are kept so the failed development
attempt is not hidden.
