# Transition Attribution Contract — 2026-07-16

Frozen before reading outputs from the attribution runs. This contract is entered because the BERT held-out transition study already shows reproducible nonzero full-gradient discrepancy with zero same-state runtime variability. Qwen attribution is conditional on its formal transition confirmation.

## 1. Goal and claim boundary

The immediate goal is not to name a unique “bad operator.” It is to separate two mechanisms at an intervention boundary:

1. **value discrepancy** arriving at the boundary;
2. **Jacobian/backward-path discrepancy** downstream or upstream of that value.

The first stage is an output-boundary causal decomposition. A second BERT-only stage moves the boundary to model regions. Region results apply to the segmented intervention program unless parity with the original monolithic compiled graph is established.

## 2. Four-arm value/Jacobian factorial

For a boundary value `z`, define four gradient arms from the same parameters, state, loss and controlled SGD transition:

| Arm | Forward value at boundary | Differentiation path |
|---|---|---|
| A reference | eager value | eager Jacobian |
| I injection | compiled value | eager Jacobian |
| R repair | eager value | compiled Jacobian |
| B candidate | compiled value | compiled Jacobian |

At the final-logit boundary, the mixed arms use a stop-gradient splice:

- injection: `z_eager + stopgrad(z_compiled - z_eager)`;
- repair: `z_compiled + stopgrad(z_eager - z_compiled)`.

Thus the mixed forward values must be exactly equal to their intended source values. Failure of that equality invalidates the arm.

## 3. Estimands

For full gradient vectors `G_A`, `G_I`, `G_R`, `G_B`:

- value-injection effect under eager Jacobian: `G_I - G_A`;
- value effect under compiled Jacobian: `G_B - G_R`;
- compiled-Jacobian effect at eager values: `G_R - G_A`;
- compiled-Jacobian effect at compiled values: `G_B - G_I`;
- interaction: `G_B - G_I - G_R + G_A`.

Report exact L2 norms, relative norms, cosine relationships, maximum coordinate effects and per-parameter-block norms. These vector effects are not assumed additive in norm.

Useful descriptive recovery ratios include:

- `||G_R-G_A|| / ||G_B-G_A||` for residual discrepancy after value repair;
- `||G_I-G_A|| / ||G_B-G_A||` for discrepancy induced by output-value injection.

Ratios above one and non-additive interactions are allowed; they are evidence against a simple single-cause story, not measurement failure.

## 4. State populations

### Output-boundary BERT

- model and loss: identical to the frozen BERT transition contract;
- discovery: first 32 states of the frozen SST-2 discovery bank;
- confirmation: first 32 states of the frozen SST-2 confirmation bank;
- two repeats per state;
- state selection is by original dataset order, not discrepancy magnitude.

### Output-boundary Qwen

- model, four-sequence state unit and teacher-forced loss identical to the Qwen transition contract;
- discovery: first four minibatch states from rows `[0,16)`;
- confirmation: first four minibatch states from rows `[32,48)`;
- two repeats per state;
- run only if the full transition endpoint reproduces.

### BERT region boundaries

- boundaries: embedding output, encoder layer 0 output and encoder layer 1 output;
- discovery and confirmation: first 16 states from each frozen bank;
- prefix and suffix are executed in a declared eager/compiled 2x2 composition;
- no boundary is selected after inspecting its attribution effect.

## 5. Integrity checks

- same parameters, tensors, labels, loss and RNG protocol across all arms;
- compiled calls tracked and graph hashes recorded;
- same-path repeat audit;
- exact mixed-arm boundary-value equality;
- finite activation/loss/gradient checks;
- monolithic A/B endpoints rerun as anchors;
- segmented A and B compared with monolithic A and B.

If segmentation changes the monolithic compiled endpoint, the result is **intervention-dependent region attribution**. It cannot be called an operator causal effect for the original graph, because graph partitioning may alter fusion, scheduling, layout and numerical order.

## 6. Interpretation

- Repair estimates a candidate-path counterfactual conditional on the chosen splice and differentiation semantics.
- Injection estimates whether the observed boundary value discrepancy can induce the endpoint under a reference differentiation path.
- Neither arm proves unique necessity or sufficiency when alternative paths or interactions exist.
- Final-logit attribution distinguishes value and Jacobian mechanisms but does not locate the source operator.
- A region boundary can identify where an effect enters or propagates only if intervention integrity is satisfied.
- Semantic events at the final output are determined by the value arm by construction; the factorial is informative mainly for gradient/update endpoints.

## 7. Kill criteria

- mixed forward values fail exact equality;
- repeated gradients vary within an arm under the deterministic protocol;
- Qwen transition discrepancy fails held-out reproduction;
- all apparent recovery is caused by a changed segmented compiled program;
- interaction dominates so strongly that a single-region contribution is not interpretable;
- region rankings fail held-out reproduction;
- attribution requires relabeling intervention-dependent sensitivity as unique operator root cause.

