# Results retained in Git

This directory is intentionally **not** a full raw-experiment archive.  The
local working tree contains hundreds of GB of rollout dumps, checkpoints and
TorchInductor caches; they are excluded by `.gitignore` because they cannot be
reliably versioned in a normal Git remote.

The repository retains the small, machine-readable artifacts needed to audit
the current localization-method development:

- `calibration/`: kernel-plumbing and tiny-training calibration records;
- `historical_candidate_screen/`: bounded old-runtime witness screens;
- `historical_blind/`: local replay/provenance/intervention records, with the
  actual allowed claim level encoded in each report;
- `historical_post_reveal/`: fixed-runtime compatibility controls.

The authoritative interpretation is always the localization evidence ledger
in `reports/LOCALIZATION_EVIDENCE_LEDGER_V0_1.md`, not a raw artifact alone.
Larger Qwen/operator artifacts remain local and are summarized by the tracked
reports and theory-oracle manifests.  A new server should reproduce them from
the scripts/configuration and bind any new artifact to its exact environment.
