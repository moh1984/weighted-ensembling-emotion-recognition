# Research Protocol v7 — Part A FROZEN BEFORE MODEL TRAINING

**Status: Part-A FROZEN BEFORE MODEL TRAINING.** Part-A data and training settings are fixed. **Parts B and C are pre-specified but remain blocked** pending construction and independent freeze of their data artifacts (§4). Nothing below may be revised in response to an observed result.

**Supersedes:** `research_protocol_v6.md` for everything concerning hypothesis status, multiplicity correction, metrics and training configuration. v6 remains the reference for design rationale not restated here.

---

## 0. What is frozen, and by what

| Artifact | Role | Binding |
|---|---|---|
| `section0_report.json` | Feasibility verdict, dose, H1 metric, thresholds | `freeze_quality: true`; SHA-256 recorded in the dataset manifest |
| `h1_amendment.json` | H1 demotion, author-identified and UTC-dated | Bound by SHA-256 to the report above and to the corpus content hash |
| `splits/` | The frozen data | 57/57 independent verification checks passed |
| `training_config.json` | Every training setting | SHA-256 recorded in every run's metadata; no CLI override exists |
| `run_experiments.py` | The training code itself | SHA-256 recorded in every run's metadata |
| `verify_splits.py` | The 57-check verifier that certified the splits | The exact file that produced the 57/57 result, frozen with the rest |

**The config is bound to one dataset, not merely to a set of models.** `training_config.json` carries `data_provenance`: the SHA-256 of `splits/dataset_manifest.json`, the corpus content hash, and the permitted split seeds. The runner verifies all three **before training** and refuses splits that are not frozen, a manifest that does not match, or a split seed outside the frozen set. Recording provenance only in the output metadata would surface a mismatch after the compute was spent.

`run_experiments.py` reads the **numeric** training settings from the config and accepts no command-line override of any of them: a setting that can be changed per invocation is not frozen, whatever a document claims.

**Structural settings are implemented in code, not read from the config.** The optimizer, scheduler, loss, padding strategy and checkpoint rule are written directly into the runner; writing `"optimizer": "SGD"` in the config would not change what executes. Rather than making the runner generic, it **refuses any config whose structural fields contradict what it implements**, so the config can describe those settings but never misdescribe them. It also refuses a config not marked `frozen`, any model revision that is not a 40-character commit SHA, and any `class_weighting` other than `none`; and it verifies the loaded checkpoint's commit against the requested revision, failing closed when the commit cannot be read back.

### 0.1 Disclosure: the pre-Section-0 protocol text was not archived separately

`research_protocol_v6.md` was edited in place after Section 0 reported, so no byte-exact copy of the pre-Section-0 wording survives, and the project was not under version control at that time. This is stated rather than reconstructed: a file labelled "pre-Section-0" but assembled afterwards from memory would be worse than none.

What does survive, and is what actually matters:

- `section0_report.json` contains a `thresholds` block recording the feasibility criteria (≥1500 eligible test examples, ≥5% overlap, ≥50 minimum class support) **as applied**, and the report predates any model training.
- The changelog at the head of `research_protocol_v6.md` records every revision round in sequence, including which changes were made after Section 0.
- `h1_amendment.json` is cryptographically bound to the report and carries its own UTC decision timestamp.

The thresholds were not moved when the data failed them. The record supports that claim; it simply does so through the report and the amendment rather than through an archived document.

---

## 1. Hypothesis status after Section 0

| ID | Status | Metric | In multiplicity family |
|---|---|---|---|
| **H1** | **Exploratory** — structural corpus diagnosis plus a low-dose stress test | accuracy | **No** |
| **H2** | Confirmatory — jury vs the designated single RoBERTa (§1.2) | macro-F1 | Yes |
| **H3** | Confirmatory — validation-weighted vs equal-weight soft voting | macro-F1 | Yes |
| **H4** | Exploratory — dependence vs gain | descriptive | No |

**Holm–Bonferroni is applied across H2 and H3 only**, on split 42.

### 1.1 H1 as executed

Section 0 declined H1 as confirmatory on all three **pre-specified pre-training feasibility criteria**: eligible test subset 340–410 (threshold 1500), maximum achievable overlap 1.409–1.704% of training (threshold 5%), minimum eligible class support 4 (threshold 50).

**Primary H1 output — structural diagnosis, no model required.** 34,951 examples from 31,618 authors; 7.9% of authors with ≥2 posts; 16.68% of posts from such authors; maximum single author 15 posts; reserve pool 1.5% of the corpus. **Within the retained corpus and the pre-specified thread-disjoint construction, the opportunity for substantial author overlap is structurally limited.**

**Secondary H1 output — exploratory stress test** at k = 1, accuracy as the metric, with effect size and bootstrap CI. A paired permutation test may be reported, labelled exploratory.

**Interpretive limits, reproduced wherever the result appears:**

- A null result is consistent with the absence of a mechanism and is **not** evidence about the magnitude of author leakage in emotion benchmarks generally.
- Section 0 did not evaluate the official GoEmotions split; nothing here licenses any claim about it. The corpus is not "immune" — author overlap is possible, merely limited.
- Design sensitivity (normal approximation, 80% power, α = 0.05 two-sided, **not** post-hoc achieved power): on ~340 eligible examples the minimum detectable accuracy difference is ≈3.4 / 4.8 / 6.8 / 10.7 points at assumed discordance rates of 5 / 10 / 20 / 50%.
- The eligible subset is dominated by one class (split 42: joy 220, anger 64, surprise 61, sadness 37, disgust 5, fear 4 of 391). A majority-class predictor reaches ~56% accuracy on it, and any signal outside `joy` rests on tens of examples.

**Thread-level leakage is not adopted as a substitute.** Re-specifying the grouping unit as `link_id` would change the construct; it is future work.

### 1.2 The H2 single-model baseline is designated, not selected

Three RoBERTa seeds are trained on R_clean. Without a rule fixed in advance, the H2 baseline would be picked **after** training — and selecting even on validation is a best-of-three choice, while the jury has a single configuration with no selection at all. Comparing a best-of-three single model against one jury is asymmetric in either direction.

**The H2 baseline is therefore designated before training: RoBERTa, R_clean, `model_seed = 1`.** It is recorded in `training_config.json` under `h2_single_model_baseline`, and the runner refuses a config naming an untrained combination.

Seeds 2 and 3 estimate initialization variability, supply the RoBERTa seed jury (A′) for H4, and give replication context. They are never candidates for the H2 baseline.

---

## 2. Class weighting — Phase 1: none

All Phase-1 models are trained with **unweighted cross-entropy**, frozen before the first training run and recorded in every output file.

For **H3**, the jurors and their saved probability vectors are identical across the weighted- and equal-weight aggregation conditions; only the aggregation rule differs. For **H2**, the same unweighted objective is applied to the single-model baseline and to every jury member, so class weighting is not introduced as an additional experimental factor beyond the pre-specified single-model-versus-jury design. (H2 is not a comparison of identical components: the single RoBERTa trains on all of R_clean while each complementary juror trains on roughly two thirds. Holding the objective constant removes weighting as a confound; it does not make the comparison component-identical.)

Because the retained corpus is highly imbalanced — `disgust` 475/34,951 ≈ 1.36%, `fear` 609 ≈ 1.74%, `joy` ≈ 57.7% — a rare class may receive low or zero F1. That is **reported as an empirical result** and does not modify the Phase-1 protocol. If exactly one class scores F1 = 0 and the other five are perfect, macro-F1 is capped at 5/6 ≈ 0.833; the general point is that each of the six equally weighted components can be lost entirely.

**Reporting requirements, pre-specified before training:** macro-F1 primary for H2/H3; **weighted-F1 reported secondarily**; **per-class F1 with class support mandatory** for every compared system. Balanced class weighting, if evaluated, is a separately labelled Phase-2 sensitivity analysis and cannot replace the frozen Phase-1 results.

---

## 3. Training configuration (frozen)

Full values in `training_config.json`. Summary:

| Setting | Value |
|---|---|
| BERT | `google-bert/bert-base-uncased` @ `86b5e0934494bd15c9632b12f734a8a67f723594` |
| RoBERTa | `FacebookAI/roberta-base` @ `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| max_length | 64 |
| optimizer / LR / weight decay | AdamW / 2e-5 / 0.01 |
| batch (train / eval) | 32 / 128 |
| epochs / scheduler / warmup | 4 / linear with warmup / 10% |
| gradient clipping | 1.0 |
| checkpoint rule | best validation macro-F1, restored before test inference |
| early stopping | patience 2 epochs on validation macro-F1 |
| model seeds | 1, 2, 3 (jury members: seed 1 only) |
| class weighting | none |

**`max_length = 64` was measured, not assumed.** Pre-specified rule: the smallest of 64/128/256 for which at most 5% of examples exceed it under both pinned tokenizers. Measured on the frozen corpus — BERT P99 = 37, max 82, 1 example (0.003%) over 64; RoBERTa P99 = 36, max 186, 6 examples (0.017%) over 64.

**Determinism is bounded, applied, and scoped.** The settings in the config are read and applied by the runner (`torch.use_deterministic_algorithms`, `cudnn.benchmark`) and the applied values are recorded per run — a declared setting the code never reads would document an intention, not a behaviour. Python, NumPy and PyTorch seeds are set; full CUDA determinism is not enforced, since it is costly and does not hold across GPU models. Bitwise reproduction of a training run is therefore **not** claimed. What is reproducible is the pipeline and the entire analysis, from the released per-sample predictions.

---

## 4. Phase 1 grid — 57 runs

**Part A (45 runs):** per split instance (42, 123, 2024) — BERT on R_clean × 3 seeds; RoBERTa on R_clean × 3 seeds; RoBERTa on R_overlap × 3 seeds; complementary-partition BERT jurors (A+B, A+C, B+C); disjoint-thirds BERT jurors (A, B, C).

**Parts B and C (12 further unique runs, split 42 only):** Twitter and Combined baselines at one seed each, and size-matched Corpus-Diversity Jury members at three seeds. One of the jury runs is shared with the Twitter BERT baseline — Twitter is the size-matching target — and is trained once.

Part B and C baselines carry **no estimate of initialization variance** and are reported descriptively with no significance claim.

**Resume safety.** A run counts as complete only when both its prediction file and its metadata exist; the metadata names **that run's id** and cites the current config hash, runner hash and dataset manifest hash; the prediction file's SHA-256 matches the one recorded when it was written; and the row count matches. The id check matters because every other field describes the config, the code or the data — all shared across runs — so a file pair copied from another run and renamed would otherwise pass. A result produced by a different runner version or dataset, or belonging to another run, is re-run rather than skipped. Outputs are written to temporary files and renamed atomically. A session interrupted mid-write leaves a partial file that is detected and discarded rather than skipped as finished.

**Not yet implemented:** `build_semeval.py` and `build_size_matched_subsets.py`. Parts B and C cannot start until they exist and their outputs are frozen and hashed on the same terms as Part A.

---

## 5. What must not change

1. Hypothesis statuses, estimands, metrics and multiplicity family.
2. The frozen splits and every hash binding them to Section 0 and the amendment.
3. Every value in `training_config.json`.
4. The jury definitions, aggregation rules and inference rules from v6 §5–§8.

H1's demotion occurred **before** this freeze and is recorded by amendment. Nothing may move in the other direction: no hypothesis may be promoted, no threshold relaxed, no metric substituted after results are seen. If H2 or H3 proves infeasible, the protocol is revised and re-frozen openly — the amendment mechanism accepts H1 only, by construction.

---

## 6. Execution order from here

1. **Verify the frozen data independently** — `verify_splits.py` must report 57/57 against the written artifacts.
2. **Preflight** (`--preflight`): validates the config against what the runner implements, loads each tokenizer, resolves each model's commit via `AutoConfig` and checks requested == resolved, verifies the dataset manifest hash against the config, and constructs the job grid — **without loading the corpus, without training, without downloading model weights, and without reading any test id**. The corpus is loaded only on the path that actually trains, so the claim is literal rather than a statement about inference alone. The printed `training_config_sha256` must equal the hash recorded when the config was finalised.
3. **Freeze the whole chain**, code included:

```bash
sha256sum research_protocol_v7_frozen.md training_config.json \
          run_experiments.py verify_splits.py corpus.py build_dataset.py \
          section0_report.json h1_amendment.json \
          splits/dataset_manifest.json > FROZEN.sha256
```

4. **Local immutable checkpoint**: `git init`, inspect `git status --short`, then `git add -A` and commit as *FROZEN before Part-A model training*. `predictions/` is git-ignored: a freeze that already contains results is not a freeze. Check the corpus redistribution terms before publishing anywhere public — `raw/` sits under the project.
5. **First official training run.** Any run that touches the test set is one of the 45 and is kept; pipeline experimentation belongs in `--preflight`.
6. Prove the resume cycle end-to-end (see the README runbook), then execute the grid one split per session.
7. Weight-activation diagnostics (v6 §6) **before** interpreting H3.
8. H2 and H3 on split 42 with Holm; replication on 123 and 2024.
9. H1 structural diagnosis and exploratory stress test.
10. H4 descriptive analysis: rank consistency across splits for the four common jury types; corpus diversity on split 42 only.
11. Implement and freeze the Part B/C builders, then run those 12.
12. Write the manuscript from computed results only.
