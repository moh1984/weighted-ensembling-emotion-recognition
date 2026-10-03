# Part B — Frozen External Robustness Dataset

Status: FROZEN

Scientific role:
EXTERNAL / ROBUSTNESS EVALUATION ONLY.

The retained subset is derived deterministically from the pinned
CardiffNLP/SuperTweetEval tweet_emotion source using the frozen
Part-B Ekman mapping specification.

Important constraints:

- No model was trained during this build.
- No resampling was performed.
- No augmentation was performed.
- No rows were deduplicated.
- Official source split membership and source order were preserved.
- There is zero normalized-text overlap across retained train,
  validation, and test splits.
- The validation split contains zero Surprise examples.
- The test split contains only three Surprise examples.
- Part B must not be used for six-class hyperparameter selection
  in the primary protocol.
- The primary scientific use is external robustness evaluation of
  already-selected/frozen models on test.retained.jsonl.

The historical preliminary preview is superseded and must not be
used. The authoritative retention preview is
partB_retention_preview_verified.json.

All frozen artifact hashes are listed in FREEZE.sha256.
