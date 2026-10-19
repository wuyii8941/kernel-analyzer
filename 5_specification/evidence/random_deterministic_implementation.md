# Deterministic dropout implementation review — 2026-10-19

The active RANDOM contract covers two value tasks whose answers do not depend on reproducing a random stream. The
forward profile covers evaluation mode and the training boundaries p=0 and p=1. The backward profile covers native
dropout backward with an explicit Boolean mask and finite scale. Both use complete positive rank-1 through rank-4
physical finite f16, bf16, f32 or f64 tensors.

The retained PyTorch 2.10 Dropout and functional Dropout pages define independent Bernoulli zeroing, training scale
1/(1-p), and evaluation identity. The retained PyTorch decomposition gives identity and all-true mask outside active
dropout, all-zero output and all-false mask at p=1, and `grad_output * mask * scale` for native backward. The retained
native schema file separates one-output functional dropout, two-output native forward, one-output native backward and
in-place variants.

The former active draft conditioned its result on a synthetic `keep` keyword that is absent from the selected ATen
schemas. It also could not be bound honestly: the runtime binding accepts task inputs only from call inputs and
pre-state, while using the tested call's returned mask would fail to validate that mask independently. That draft is
retained in `contracts/draft/RANDOM_conditioned_extended_20261019.yaml`; it is not an admitted ground truth.

The reviewed adapter validates physical dtype, rank, positive dimensions, finite exactly representable values,
probability, strict Boolean options, complete Boolean mask shape and finite exact scale. It explicitly refuses
training forward with 0<p<1, native forward, in-place forms, special values, signed-zero distinctions and unselected
dtypes or ranks.

Validation comprises 17 contract examples and five mutated formulas. Independent generated checks cover all four
floating dtypes and ranks one through four for native backward, plus all four dtypes for each deterministic forward
case. The coordinate formula and PyTorch 2.10 agree on every checked value, dtype and shape. This evidence does not
claim that stochastic training forward is solved; that task still needs a separately justified random-law test.
