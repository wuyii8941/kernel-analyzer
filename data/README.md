# Local data policy

Model checkpoints, Hugging Face caches, tokenizers copied from models,
optimizer states, rollout dumps and generated compiler source are intentionally
not committed.  They contain the bulk of the local 1+ TB workspace and are
not a portable Git artifact.

Tracked exceptions are small, non-model inputs or manifests that explain a
reproduction.  The scripts in `scripts/` and `theory_oracle/` declare their
expected data paths and environment.  Any new matched training state must be
captured as a versioned external artifact with hashes, not silently added to
Git.
