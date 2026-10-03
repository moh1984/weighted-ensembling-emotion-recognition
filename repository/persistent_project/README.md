# Emotion Recognition from Social Media Text — Part A

Leakage-aware, reproducible evaluation of transformer and jury ensembles on
GoEmotions. This repository holds the frozen pre-training state: the protocol,
the feasibility report, the H1 amendment, the data-construction code, the
frozen splits and the training configuration.

**Status: Part A frozen before model training.** Parts B and C are
pre-specified but blocked pending their own builders and an independent freeze.

---

## Which artifacts are authoritative

| File | Authority |
|---|---|
| `research_protocol_v7_frozen.md` | **Current protocol.** Supersedes v6 for hypothesis status, multiplicity, metrics and training configuration |
| `research_protocol_v6.md` | Design rationale and the revision changelog. Superseded where v7 speaks |
| `section0_report.json` / `.md` | **Authoritative Section 0.** `freeze_quality: true`. Carries the feasibility thresholds as applied |
| `section0_preview.json` / `.md` | **Diagnostic only, non-freezing.** Produced by the initial feasibility run with `--allow-unpinned-dedup`; it established the H1 feasibility picture and also exposed the resolved embedding-model commit used for the subsequent pinned run. Retained for audit history only, and not a substitute for the authoritative final Section-0 report |
| `h1_amendment.json` | H1 demotion to exploratory, author-identified and UTC-dated, bound by SHA-256 to the Section 0 report and the corpus |
| `training_config.json` | Every training setting. Bound by hash to one specific dataset |
| `splits/` | The frozen data. Verified 57/57 by `verify_splits.py` |
| `FROZEN.sha256` | SHA-256 hashes of the authoritative pre-training artifacts named in the freeze command, recorded before any training. The Git commit captures the full snapshot |

**This repository is not a self-contained archive on its own.** `verify_splits.py`,
`section0_report.*`, `h1_amendment.json` and `splits/` are produced in the
working environment and must be present alongside these files before anything
can run. Build the complete bundle after the freeze commit, not from this file
set alone.

The pre-Section-0 wording of the protocol was **not** archived separately — see
v7 §0.1. No reconstruction was made; what supports the "thresholds were not
moved" claim is the `thresholds` block inside `section0_report.json`, which
predates any training, together with the hash-bound amendment.

---

## Pipeline

```
corpus.py                  shared corpus construction and split logic
section0_feasibility.py    pre-freeze feasibility check (no training)
build_dataset.py           frozen split construction
verify_splits.py           57 independent checks on the written artifacts
run_experiments.py         Part A training, config-driven, resumable

diagnostics/
  thread_probe.py          historical diagnostic only; NOT used in any Part-A
                           analysis. Measured whether a thread-level H1 would
                           have been powered; that option was not adopted
```

---

## Runbook

### Once, before any training

```bash
python verify_splits.py --splits-dir ./splits \
       --section0-report ./section0_report.json \
       --h1-amendment ./h1_amendment.json
# expect: 57/57 PASS

python run_experiments.py --splits-dir ./splits --out-dir ./predictions \
       --config ./training_config.json --preflight
# expect four things:
#   [data ] manifest sha256=... verified against config
#   [pre  ] bert:    ... requested == resolved
#   [pre  ] roberta: ... requested == resolved
#   [cfg  ] ... sha256=XXXX   <- must equal the SHA recorded when the config
#                                was finalised; any difference means the file
#                                was touched after the freeze
#   [pre  ] seed NN: 15 jobs ...   (per split)
# preflight loads no corpus, trains nothing, and reads no test ids

sha256sum research_protocol_v7_frozen.md training_config.json \
          run_experiments.py verify_splits.py corpus.py build_dataset.py \
          section0_report.json h1_amendment.json \
          splits/dataset_manifest.json > FROZEN.sha256

git init                    # must come first; git status fails without a repo
git status --short          # inspect the list before staging anything
git add -A
git commit -m "FROZEN before Part-A model training"
```

**Do not open `training_config.json` for writing again.** Even a rewrite with
no change alters its bytes, hence its hash, hence `training_config_sha256` in
every run's metadata — and the runner will then treat completed runs as
produced by a different config and repeat them.

### First session

Nothing to copy in; there are no previous results yet. Take **one** run so the
transfer cycle can be proved on a single result before committing to the grid:

```bash
python run_experiments.py \
       --splits-dir ./splits \
       --out-dir ./predictions \
       --config ./training_config.json \
       --split-seed 42 \
       --max-runs 1
```

Then download `predictions/` and publish it as a Kaggle dataset.

### Every session after the first

`/kaggle/input` is read-only, so previous results must be copied into the
working directory before the runner can see them. A silently failed copy means
a full retrain that nobody notices, so the copy verifies itself:

```python
from pathlib import Path
import shutil

src = Path("/kaggle/input/<YOUR-PREDICTIONS-DATASET>")
dst = Path("/kaggle/working/emotion_project/predictions")
assert src.exists(), f"input dataset not found: {src}"
n = len(list(src.glob("*.csv")))
assert n > 0, "no prediction files in the input dataset"
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"copied {n} prediction files")
```

```bash
ls predictions/*.csv       | wc -l
ls predictions/*.meta.json | wc -l   # must match; a mismatch means a partial upload
```

Run the resume checks below **once**, on the session immediately after the
first run. Once they pass, a normal session covers roughly one split (15 runs):

```bash
python run_experiments.py \
       --splits-dir ./splits \
       --out-dir ./predictions \
       --config ./training_config.json \
       --split-seed 42
```

At the end of every session, download `predictions/` and publish it as a new
version of the dataset.

### Proving the resume cycle (do this after the first run, not after fifteen)

`[skip]` alone does not distinguish a working check from a lax one, so the
negative case must be tested too — **without touching the real run**. Corrupting
an actual prediction file and re-invoking the runner would let it retrain, and
restoring the old CSV afterwards would leave a CSV from the first training
paired with metadata from the second. That is precisely the inconsistency the
integrity check exists to prevent, so the test operates on a temporary copy and
calls `is_complete` directly:

```python
from pathlib import Path
import json, shutil, tempfile, importlib.util

root = Path("/kaggle/working/emotion_project")
pred = root / "predictions"
RUN_ID = "<run_id>"          # a completed official run

spec = importlib.util.spec_from_file_location("runner", root / "run_experiments.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
meta = json.loads((pred / f"{RUN_ID}.meta.json").read_text())
args = (RUN_ID, meta["training_config_sha256"], meta["n_test"],
        meta["run_experiments_sha256"], meta["dataset_manifest_sha256"])

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    csv, mj = td / f"{RUN_ID}.csv", td / f"{RUN_ID}.meta.json"
    shutil.copy2(pred / csv.name, csv); shutil.copy2(pred / mj.name, mj)

    ok, why = runner.is_complete(csv, mj, *args)
    print("intact copy   :", ok, why); assert ok

    csv.write_text("\n".join(csv.read_text().splitlines()[:5]) + "\n")
    ok, why = runner.is_complete(csv, mj, *args)
    print("modified copy :", ok, why)
    assert not ok
    # The checksum is verified before the row count, so truncation is caught
    # as a byte mismatch. That is the stronger of the two checks: it also
    # catches a same-length file from a different run, which the row count
    # alone would pass.
    assert why == "prediction checksum mismatch"

print("Resume integrity verified without touching the real run.")
```

Then confirm the transferred run is skipped in a normal invocation. Use
`--max-runs 0`, not `1`: the cap counts **new** runs, so skips do not consume
it and `--max-runs 1` would report the skip and then start training the next
job — an unintended run in what is supposed to be a read-only check.

```bash
python run_experiments.py --splits-dir ./splits --out-dir ./predictions \
       --config ./training_config.json --split-seed 42 --max-runs 0
# expect:
#   [skip ] bert__r_clean__split42__seed1
#   [stop ] session cap reached (0)
# and no training
```

If a real run is ever reported `[redo ]` unexpectedly, the reason is printed —
a mismatched run id, config, runner, dataset, row count or prediction checksum.

**The first real run is a result, not a pilot.** Any run that touches the test
set is one of the 45 and is kept. Pipeline experimentation belongs in
`--preflight`, which never trains and never touches test data.

---

## Grid

**Part A — 45 runs.** Per split instance (42, 123, 2024): BERT on R_clean × 3
seeds; RoBERTa on R_clean × 3 seeds; RoBERTa on R_overlap × 3 seeds;
complementary-partition BERT jurors (A+B, A+C, B+C); disjoint-thirds BERT
jurors (A, B, C).

**Parts B and C — 12 further unique runs, split 42 only.** Blocked until
`build_semeval.py` and `build_size_matched_subsets.py` exist and their outputs
are frozen and hashed on the same terms as Part A.

---

## Provenance recorded per run

`run_id`, `training_config_sha256`, `run_experiments_sha256`, `dataset_manifest_sha256`,
`corpus_content_sha256`, `prediction_file_sha256`, `index_file_sha256` (train,
val, test, `jury_blocks.json`), `effective_train_ids_sha256`, requested and resolved
backbone revisions, applied determinism settings, class weighting,
full-precision validation macro-F1, environment versions and elapsed time.

`effective_train_ids_sha256` exists for the jury members: their training set is
the index file filtered through an author block, so the index hash alone does
not identify what a juror saw.

---

## Data

GoEmotions raw (`goemotions_1/2/3.csv`) and the official `ekman_mapping.json`.
Not redistributed here. Check the applicable source terms before publishing any
repository that includes `raw/`.
