# Step-14 Reconstruction Validation

## Result

- Gate: `PASS`
- Original/replay/common rows: `512/512/512`
- Missing/extra keys: `0/0`
- Field mismatches: `{"advantage": 0, "advantage_sign": 0, "new_logp": 0, "old_logp": 0, "policy_iteration": 0, "rollout_batch": 0, "token_id": 0}`

The reconstruction is accepted only on exact equality; no numeric tolerance is used for rollout state fields.

## Artifacts

- `results/step14_reconstruction_validation.json`
- `results/replay/step14_forks_validated.jsonl`
- `data/phase9_policy_step14_pre/forkcert_snapshot.json`
