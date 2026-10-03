# Part A — Retained Prediction Checkpoint (45/45 runs)

This directory makes the Part-A verification package self-contained. It contains the exact
`emotion-parta-predictions-checkpoint45` dataset used by the frozen Part-A analysis.

## Contents

- `predictions_checkpoint45/`: 45 per-run prediction CSV files and 45 matching `.meta.json` files.
- `predictions_checkpoint45/checkpoint_manifest.json`: training-completion manifest.
- `predictions_checkpoint45/CHECKPOINT.sha256`: checksums for all 90 CSV/metadata artifacts.

Verified assembly bindings:

- checkpoint archive SHA-256: `32fcf084de1d762dfe3e6eabf1f00daf5f2419595e8a4ff2b673742697eb0da3`
- checkpoint manifest SHA-256: `6463453c42bd0c310a15fd6acdedbc5012eff021e67de69e08420f3903840b05`
- checkpoint checksum-file SHA-256: `64148923cf64054870ea14bd9bd041ca2e99dfee50abf4a246183d28630c2192`
- completed/expected runs: 45/45
- split counts: 15 runs each for split 42, 123, and 2024

The frozen analysis configuration at
`../repository/current_working_project/partA_analysis/partA_analysis_config.json`
contains the same checkpoint-manifest and checksum-file hashes. Historical frozen analysis
artifacts have not been edited.

## Verify the checkpoint

From the extracted repository root:

```bash
python verify_partA_checkpoint.py
```

## Recompute Part-A analysis

The frozen analysis program accepts an explicit local checkpoint path, so the Part-A results
can be recomputed from this repository without the former Kaggle dataset mount:

```bash
python repository/current_working_project/partA_analysis/analyze_partA.py \
  --run \
  --project-root repository \
  --checkpoint partA/predictions_checkpoint45 \
  --config repository/current_working_project/partA_analysis/partA_analysis_config.json \
  --out-dir partA/recomputed_results \
  --confirm RUN_FROZEN_PARTA_ANALYSIS
```

The analysis is deterministic under its recorded seed, but the full 10,000-permutation /
10,000-bootstrap computation can take substantial time. Model retraining is not required for
verification.
