# Research Protocol v6 — Freeze Document

**Status: both open slots are now closed by the Section 0 report.** Dose k = 1; H1 primary metric = accuracy; H1 status = **exploratory**, by the amendment in §2.1. The matching method is fixed as hierarchical with full disclosure. What remains before training is the freeze declaration and author sign-off.

**Amendment mechanism.** When Section 0 declines a hypothesis, the builder refuses to proceed unless a signed amendment file is supplied. The mechanism is deliberately narrow, because a general "downgrade a failing hypothesis" path would let any future feasibility failure be dissolved by relabelling — the precise behaviour the pre-freeze machinery exists to prevent. An amendment therefore: applies to **H1 only** (H2 and H3 cannot be demoted this way; if either became infeasible the protocol would have to be revised and re-frozen openly); may only **weaken** a hypothesis; must carry the SHA-256 of the exact Section 0 report that prompted it and of the corpus; and must record the dose, the metric, the decision text, the signatories and the date. Its own hash is written into the frozen manifest.

`build_dataset.py` reads both slots from `section0_report.json`. It refuses to run unless the report is marked `freeze_quality: true`, and — where Section 0 declined a hypothesis — unless a valid H1 amendment is supplied — which requires the official mapping, a completed near-duplicate pass, a pinned dedup model revision, and no preview override. It also refuses to accept any preview flag on its own command line alongside a Section 0 report, so a frozen build cannot be assembled from preview parts. The distinction between preview and frozen is enforced, not merely recorded in a manifest field. It then verifies five things against the report: the SHA-256 of each raw GoEmotions file, the corpus **content** hash (over id, text, label, author, subreddit, link_id and both matching buckets), the corpus id-set hash, the Ekman mapping file hash, and the per-partition id hashes of every split — plus the injected and removed id sets at the chosen dose. A frozen build therefore cannot silently differ from the analysed splits. An id-list hash alone would not have sufficed: mapping, labels, cleaning and bucket definitions can all change while the surviving id set stays identical.

**Title:** Emotion Recognition from Social Media Text: A Leakage-Aware, Reproducible Evaluation of Transformer and Jury Ensembles

**Changes from v6.7:** Section 0's matching note updated in the code (the previous changelog entry recorded a replacement that had silently failed to apply); residual confirmatory-H1 wording removed from §9, §15 and §16; MDE range in §2.1 aligned with the code (50% added); the structural claim scoped to the retained corpus; the amendment timestamp validated as a real ISO-8601 UTC instant and template placeholders rejected.

**Changes from v6.6:** amendment verification now checks the metric against Section 0, not merely its presence; all residual text treating H1 as confirmatory removed (§8.2, §8.3, §9); the "immune / official split" overclaim replaced with the bounded statement (§2.1); Section 0's matching note updated to reflect that hierarchical matching was fixed pre-training; discordance range extended to 50% and labelled design sensitivity rather than achieved power; amendment timestamps are UTC.

**Changes from v6.5 (post-Section 0):** H1 demoted from confirmatory to a structural corpus diagnosis plus an exploratory low-dose stress test, by pre-registered amendment (§2.1); Holm correction now spans H2 and H3 only; MDE reported across declared discordance assumptions; weighted-F1 and per-class F1 with support pre-registered for H2/H3; matching claim resolved against the realized figures; H1-only cryptographically bound amendment mechanism added (§12).

**Changes from v6.4:** dedup revision must be an immutable 40-character commit and is verified against the resolved revision (§3.1); the Ekman mapping is verified by content, not by file presence (§3.1); Claim 1 of H4 restructured around a controlled BERT overlap ladder with the RoBERTa jury as a reference, and its residual training-size confound disclosed (§7); the Condorcet framing corrected so no result is said to confirm or refute the theorem (§5.4).

**Changes from v6.3:** H4 split into an across-split claim (initialization and data partition, all three splits) and a split-42 corpus-diversity extension with seed-matched juries (§2, §7); `freeze_quality` added to the Section 0 report and enforced by the builder, which now rejects preview reports and preview flags outright (§12).

**Changes from v6.2:** `created_utc` and `link_id` validity carried into the frozen corpus and fingerprint (§3.1); size-matched Corpus-Diversity subsets frozen and shared across seeds (§5.3); Phase 1 H4 changed from multivariable regression to a descriptive analysis with rank-consistency as the primary evidence (§7); the dedup model revision made a hard requirement for freeze-quality runs (§3.1, §12).

**Changes from v6.1:** author cleaning moved before deduplication (§3.1); central message reworded to remain valid under any realized matching quality (§1); H4 scoped to three diversity sources and three-member juries with exhaustive enumeration (§2, §7); raw-file hashes, pinned dedup model revision and the full dedup algorithm recorded (§3.1, §12).

**Changes from v6.0:** thread-disjoint H1 eligibility (§4.2); SemEval-2018 fixed as the sole Part B source with a single-Ekman rule (§3.2); matching specified as hierarchical with a claim-discipline rule (§4.2); run count corrected to 57 (§5.3); the Platform Jury renamed the Corpus-Diversity Jury; the pre-registered expectation for Juror T removed (§5.4); overall categorical Fleiss' κ dropped as undefined under multi-label annotation (§3.1).

**Changes from v5:** Twitter-derived and Combined corpora reinstated as Parts B and C (§3.2, §9); Corpus-Diversity Jury added as a third Phase 1 diversity source with a size-matching requirement (§5.3, §5.5); overlap dose fixed per eligible author rather than as a global training percentage (§4.3); primary confirmatory test changed to a paired permutation test on ΔMacro-F1 (§8.2); Stouffer combination replaced by a designated confirmatory split with pre-specified replication splits (§8.3); cross-corpus rather than cross-platform framing (§3.3).

---

## 0. Pre-freeze feasibility check

Executed by `section0_feasibility.py`. Trains nothing and produces no performance number, so running it does not compromise pre-registration.

**Reports:** unique authors after Ekman filtering and deduplication; posts-per-author distribution (≥2, ≥3, ≥5); proportion of posts from multi-post authors; reserve-pool capacity and eligible test subset size per split instance; **per-class support within the eligible test subset**; feasible uniform dose *k* and the resulting overlap percentage of training.

**Decision rules, fixed in advance:**

| Condition | Consequence |
|---|---|
| Eligible test subset ≥1,500 examples **and** overlap ≥5% of training, in all three splits | H1 proceeds as confirmatory; freeze as written |
| Minimum per-class support in the eligible subset ≥50 | `[H1_METRIC]` = macro-F1 |
| Minimum per-class support <50 | `[H1_METRIC]` = accuracy for H1 only; macro-F1 reported as secondary |
| Thresholds not met | H1 is **either** demoted to exploratory **or** re-specified with `link_id` grouping — which changes the construct from author leakage to thread leakage and requires the title, central message and H1 wording to change accordingly. The six authors choose explicitly, before any training |

---

## 1. Central message

> Author-level overlap between training and test data **may** inflate reported performance on social-media emotion benchmarks; we quantify **whether and by how much**, under a controlled construction that holds training size and class composition identical and hierarchically matches subreddit, text length and author activity, with the realized matching quality reported explicitly. We further ask **when** a Condorcet-inspired jury ensemble outperforms a strong single transformer, whether validation-based weighting contributes anything beyond equal-weight ensembling, and how ensemble gain varies across **initialization-, data-partition- and corpus-based** diversity in relation to measured inter-juror error dependence. Architecture-based diversity is a pre-specified extension, not part of the primary contribution.

**Standing rule:** no number enters the draft before it is produced by a saved, re-runnable pipeline. A null result on any hypothesis is reported as the finding.

---

## 2. Hypotheses

**Confirmatory family — Reddit only. Holm–Bonferroni across H2 and H3 only.** H1 was removed from this family by a pre-registered amendment issued after Section 0 and before any model was trained (§2.1).

| ID | Question | Contrast | Estimand |
|---|---|---|---|
| ~~H1~~ | *Demoted to exploratory before any training — see §2.1* | — | — |
| **H2** | Does the Primary jury outperform a single strong transformer under author-disjoint evaluation? | Complementary-partition jury vs single RoBERTa, both on R_clean | Δ macro-F1, full R_clean test set |
| **H3** | Does validation-based weighting help, holding aggregation fixed? | Validation-weighted **soft** voting vs **equal-weight soft** voting; identical jurors, identical probabilities | Δ macro-F1, full R_clean test set |

Restricting confirmatory claims to the Reddit corpus follows the original manuscript's own logic: only there do author identifiers permit auditable author-level separation.

### 2.1 H1: structural corpus diagnosis plus an exploratory low-dose stress test

Section 0 declined H1 as confirmatory. The protocol set the feasibility thresholds in advance, the data failed them, and **the thresholds were not moved** — the strength of the claim was reduced instead, before any model was trained. The amendment is bound by hash to the report that triggered it, so the order of events is verifiable.

**Why it failed, and why that matters.** GoEmotions is close to author-unique: roughly 1.1 posts per author, under 8% of authors contributing two or more, and a reserve pool amounting to a small fraction of the corpus. The maximum achievable overlap dose is far below the pre-registered floor. **The opportunity for substantial thread-disjoint author overlap is structurally limited in the retained GoEmotions corpus:** 34,951 examples from 31,618 authors; 7.9% of authors with two or more posts; only k = 1 reaching full uniform coverage; realized overlap 1.4–1.7% of training.

The primary H1 output is therefore a **structural diagnosis**, computed from Section 0 and requiring no model: the posts-per-author distribution, the reserve capacity, and the maximum achievable dose. This is a statement about the corpus, and a useful one: **within this retained corpus and the pre-specified thread-disjoint construction, substantial author overlap is not a dominant structural risk.** The claim is scoped to what was measured — it says nothing about other corpora, other constructions, or studies in general.

**What this does not establish.** Section 0 did not evaluate the official GoEmotions split, so nothing here licenses a claim about whether that split is or is not compromised. Nor is the corpus "immune": author overlap is possible, merely limited. Both statements are avoided in the paper.

Attached to it is an **exploratory low-dose stress test** at the largest dose with full uniform coverage (k from Section 0), using accuracy as its metric because the eligible subset cannot support macro-F1 — its rarest class has single-digit support. Effect size and CI are reported; the paired permutation test may be reported but is labelled exploratory throughout.

**Interpretive limit, stated wherever the result appears.** A null result here is consistent with the absence of a mechanism and **must not be read as evidence about the magnitude of author leakage in emotion benchmarks generally.** To make this concrete rather than rhetorical, Section 0 reports the **minimum detectable effect** on the eligible subset at 80% power and α = 0.05. Because power in a paired binary comparison is driven by the number of *discordant* pairs rather than by n alone, a single MDE figure would be meaningless; the MDE is reported across a declared range of discordance rates (5%, 10%, 20%, 50%), and the assumption is stated with the number. These are **design-sensitivity** figures computed under a normal approximation, not post-hoc achieved power.

**Thread-level leakage is not adopted as a substitute.** Re-specifying the grouping unit as `link_id` would change the construct from author identity to shared conversational context — a different question, requiring a different title and central message. It is recorded as future work, not used to rescue H1.

**Pre-specified secondary comparisons:** jury vs 3×RoBERTa seed ensemble; H1 on the full test set; weighted and plain **hard** majority as aggregation sensitivity analyses; all Part B and Part C results.

**Exploratory centerpiece:**

| ID | Question | Analysis |
|---|---|---|
| **H4** | Does ensemble gain over the best member vary with inter-juror error dependence, and does the *source* of diversity matter? | Two claims. **(a)** Rank consistency of four jury types across all three splits, with a designated controlled BERT overlap ladder (1.00 / 0.50 / 0.00) and the RoBERTa seed jury as a strong-backbone reference. **(b)** Corpus diversity on split 42 only, with three seed-matched juries, reported as a single-split extension. Descriptive throughout; the multivariable regression and architecture diversity are Phase 2 |

---

## 3. Data

### 3.1 Part A — Reddit-derived corpus (primary)

**GoEmotions raw** (`goemotions_1/2/3.csv`). Raw rather than simplified because the metadata is the study: `author` (author-level splits), `rater_id` (recomputable agreement), `subreddit` (matching), `link_id` (thread-disjoint H1 eligibility), `created_utc` (Phase 3 temporal split).

`created_utc` is **carried into the frozen corpus and its fingerprint** even though Phase 1 does not use it, normalised to an integer first so that float formatting differences across library versions cannot perturb the hash. Freezing without it would force a rebuild before Phase 3 and invalidate every hash in the chain.

**Missing `link_id`.** A post with no usable thread id cannot be shown to be thread-disjoint from a test target, so its author is **ineligible for H1**. The post itself is retained as ordinary training and evaluation data — discarding it would waste corpus for no gain. The count is reported per split.

**Label construction:** aggregate per-rater rows to per-example labels; map 27 categories → Ekman-6 using the repository's official `ekman_mapping.json` (mapping table reproduced in an appendix); exclude `neutral`, `example_very_unclear`, and non-mappable examples; primary comparison on single-Ekman-label examples with the multi-label subset reported separately. Exclude placeholder authors (`[deleted]`, `[removed]`, bots) — a single pseudo-author can hold thousands of posts and would silently destroy any author-disjoint split.

**Agreement:** computed fresh from `rater_id` — **binary Fleiss' κ per Ekman emotion**, six figures, with the variable-raters-per-item form. The prior κ is not reused in any form.

An overall *categorical* Fleiss' κ is **not** reported. GoEmotions raters may select several emotions for one item, so per-item votes can exceed the rater count; categorical Fleiss' κ assumes each rater assigns an item to exactly one category, and that assumption is violated here. The per-emotion binary form has no such problem, since each rater contributes exactly one yes/no verdict per emotion. If an aggregate figure is ever wanted, Krippendorff's α with MASI distance is the defined choice for multi-label annotation. The Methods section states this reasoning rather than silently omitting the aggregate.

**Consensus rule:** each rater's fine-grained selections are mapped to Ekman-6 *before* aggregation and clipped to at most one vote per category per rater; an example enters the primary corpus when exactly one Ekman category is reached by ≥2 raters. The threshold is inspired by GoEmotions' own ≥2-rater filtering but is not identical, since ours is applied after mapping. Because a fixed ≥2 threshold is a majority of three raters but only 40% of five, the label outcome is reported **broken down by rater count**, with a strict-majority (`votes > n/2`) sensitivity count alongside.

**Order of operations:** placeholder and bot authors are removed **before** deduplication, not after. Otherwise a duplicate cluster containing both a `[deleted]` copy and a valid-author copy can lose both — dedup may keep the placeholder as the cluster representative, and author cleaning then deletes it. Consequence to record in the dataset card: a text appearing under both a deleted and a valid author is now always attributed to the valid author, which shifts the author distribution slightly.

**Ekman mapping provenance.** Supplying a file is not proof of provenance: an edited mapping would be read identically by Section 0 and by the builder, so every downstream hash would agree and the chain would certify a mapping nobody checked. The loaded mapping is therefore **compared against the reference content** and only then marked verified; the file hash is recorded separately so the exact bytes remain identifiable. A mismatch stops the run rather than being reported as a warning.

**Deduplication:** exact match on normalized text, then **connected components over a 10-nearest-neighbour cosine similarity graph with edges above 0.85**, retaining one item per component (sensitivity at 0.80 / 0.90 in an appendix). This is not an exhaustive all-pairs search, and the Methods section says so rather than reporting only a threshold. **Applied once to the base corpus before any split is constructed**, so R_clean and R_overlap differ in author overlap alone.

The sentence-embedding model is part of **dataset construction**, not of training: if it changes, the surviving corpus changes and every hash in the freeze chain fails. Its revision must be a **full 40-character commit SHA** — branch names and tags are mutable and pin nothing — and the requested revision is verified against the one the loaded model actually resolves to. An **unpinned or mutable reference is refused** for freeze-quality output — a preview flag exists solely to discover the resolved commit, which is then pinned in the code before the real run. `freeze_quality` in the manifest is `true` only when the Section 0 report, the official mapping, the near-duplicate pass and the pinned model revision are all present.

### 3.2 Part B — Twitter-derived corpus (secondary)

**SemEval-2018 Task 1 (E-c)** is the primary Twitter source: it contains `disgust`, which CARER lacks, so it alone supports a comparable Ekman-6 label space.

**Decided now, not left open:** Part B uses **SemEval-2018 alone**. CARER cannot join an Ekman-6 corpus at all — it has no `disgust` — so combining them was never coherent. CARER moves to Phase 2 as a **separate 5-class sensitivity analysis** (non-Ekman `love` dropped; documented near-duplicates removed and counted).

**Single-Ekman rule for SemEval.** SemEval-2018 E-c is multi-label, so the same discipline applied to GoEmotions applies here: map the 11 labels to Ekman-6, then retain for the primary cross-corpus evaluation only examples yielding **exactly one** Ekman category; examples yielding more form a secondary multi-label reference subset. Without this, the classifier's label space would differ between the two corpora and transfer degradation would be uninterpretable.

**Asymmetry to disclose.** The two single-label rules are not equivalent. On GoEmotions the rule rests on ≥2-rater agreement over inspectable per-rater annotations; SemEval provides gold labels without recoverable multi-rater structure, so no comparable agreement threshold exists. This asymmetry is stated in the limitations alongside the platform/annotation confound.

**No size targeting.** Corpus sizes are whatever the public sources yield after filtering: N_Reddit, N_Twitter, N_Combined = N_R + N_T. No augmentation is performed to make any number resemble a figure from the previous manuscript.

### 3.3 Part C — Combined corpus (exploratory)

Reddit-derived ∪ Twitter-derived. Report the realized imbalance; it will run strongly toward Reddit, the opposite direction from the previous manuscript's imbalance. Do not reuse the previous manuscript's framing of that imbalance.

**Framing constraint — important.** The two corpora differ in **both** platform and annotation protocol: GoEmotions is 27-category crowdsourced annotation; SemEval-2018 is 11-label multi-label annotation under a different scheme. Transfer degradation therefore confounds domain shift with label-scheme difference, and the two cannot be separated with these data.

Accordingly, the paper says **cross-corpus transfer**, never *cross-platform domain shift*, and states the confound explicitly wherever transfer results appear. The stronger claim is unavailable and must not be implied.

### 3.4 Corpus description table

Reported for all three sources: total N and per-class counts for the six Ekman categories, with label-provenance and mapping notes per source. Every figure is computed from the built corpora.

---

## 4. Split construction (Part A)

Three independent split instances, split seeds **42 / 123 / 2024**, ratios approximately 70/15/15 **of the analysis corpus after reserve-pool exclusion**. Reserve items are reported separately and are not counted toward any partition of R_clean. Every split frozen to an index file and cryptographically hashed before training.

### 4.1 Regimes

| Regime | Role | Construction |
|---|---|---|
| **R_random** | Literature comparability only | Stratified random. Secondary; not used for H1 |
| **R_clean** | Author-disjoint | No `author` appears in more than one partition |
| **R_overlap** | Controlled author overlap | Identical test and validation sets to R_clean; training additionally contains reserve posts by test authors, with matched substitution holding size and composition constant |

### 4.2 Reserve pool and eligibility

An author-disjoint split places all of a test author's posts in the test partition, leaving nothing to inject. A reserve pool is therefore constructed explicitly.

**H1-eligible authors** must have at least two usable posts after deduplication, spanning **at least two distinct Reddit threads** (`link_id`). For each eligible test author, the test target is drawn from one thread; posts from the same thread are excluded from the H1 construction, while only posts from *different* threads may enter the reserve pool. This prevents the overlap treatment from introducing same-conversation context in addition to author identity — without it, H1 would measure author overlap *and* thread overlap, and the "only manipulated variable" claim would fail.

Single-post authors, and authors whose posts all sit in one thread, are ineligible by definition.

Per split instance:

1. Assign authors to train / validation / test (author-disjoint).
2. For each eligible test author, draw a **target thread**, then designate **one post from it as the test target**.
3. Their remaining posts in the **same** thread are marked `excluded` and enter no partition. Their posts in **other** threads form the **reserve pool**, which also enters no partition of R_clean.
4. Both eligibility definitions are reported: **strict** (≥2 posts across ≥2 threads, used for H1) and **loose** (≥2 posts, threads ignored, reported for comparison only). The gap between them is itself informative about how much of the corpus is conversational.
5. R_overlap training = R_clean training + injected reserve posts − an equal number of matched non-test-author posts.
6. **Hierarchical matched substitution**, in this pre-specified order, with the realized proportion at each level reported: (i) emotion + subreddit + length bucket + author post-count bucket; (ii) emotion + subreddit + length; (iii) emotion + subreddit; (iv) emotion only. **Emotion is never relaxed**, which is what holds per-class counts identical by construction.
7. Verify and report the invariants: training N identical, per-class counts identical, validation identical, test identical, uniform author coverage, and injections thread-disjoint from the test target.

**Claim discipline on matching — resolved by the Section 0 numbers.** Exact four-variable matching accounts for roughly 42–47% of substitutions; emotion-only fallback is 1.5–2.4%; the remaining ~50% relax one level while still matching emotion, subreddit and text length. The paper therefore does **not** claim that overlap is the only manipulated variable. The wording is:

> Training size and class composition were held exactly constant, while subreddit, text length and author activity were matched hierarchically where feasible, with the realized matching quality reported explicitly.

The **full distribution across all four levels** appears in an appendix, not the exact-match rate alone — reporting "44% exact" without the rest invites the reading that 56% failed, when most of it matched on three of four variables.

### 4.3 Overlap dose — fixed per author

The dose is **`[DOSE_K]` reserve posts injected per eligible test author**, identical across all three split instances, set from §0 as the largest uniform value every split can support.

Fixing the dose per author rather than as a global percentage of training guarantees **100% coverage of eligible test authors in every split**. A global percentage would give, say, 60% coverage in one split and 90% in another — a silent, unequal attenuation of the very effect H1 measures. The resulting overlap percentage of training is a derived quantity, reported, never targeted.

A second dose level may be added in Phase 2 to make H1 a dose–response relationship; the §0 level is the pre-registered primary.

### 4.4 The test set is author-balanced

A consequence of the construction above, stated here because a reviewer would otherwise have to infer it from code: **every assigned test author contributes exactly one target example**, eligible or not. The R_clean test set is therefore author-balanced rather than post-weighted, and its size equals the number of test authors.

This is deliberate and has a real benefit — prolific authors cannot dominate the evaluation, which matters in a study about author-level effects. It also has consequences that must be reported:

- The estimand for H2 and H3 shifts from post-level prevalence to **author-balanced generalization**. Results are not directly comparable to post-weighted evaluations on the same corpus, including the official GoEmotions split.
- The test class distribution is the distribution *across authors*, not across posts, and will differ from the corpus distribution. It is reported alongside the corpus distribution.

Both points go in §10's limitations, not only in the methods.

### 4.5 Estimands for H1

Both computed from the same saved predictions:

- **Primary — eligible subset:** test examples by H1-eligible authors. Measures the mechanism, undiluted.
- **Secondary — full test set:** includes single-post authors who cannot experience overlap. Measures the realistic aggregate effect and is attenuated toward zero **by construction**; the attenuation is expected and explained, not treated as a discrepancy.

### 4.6 Per-split reporting

Unique author count, posts-per-author distribution, reserve pool size, eligible-subset size and per-class support, realized overlap percentage, class-balance drift versus R_random, and index-file hashes. Report drift rather than forcing artificial balance. No augmentation on validation or test; if used at all, training-only and strictly after splits are frozen.

---

## 5. Models and juries

### 5.1 Part A runs, per split instance

| Run set | Count |
|---|---|
| BERT-base, R_clean training | 3 model seeds |
| RoBERTa-base, R_clean training | 3 model seeds |
| RoBERTa-base, R_overlap training | 3 model seeds |
| Complementary-partition BERT jurors (A+B, A+C, B+C) | 3 |
| Disjoint-thirds BERT jurors | 3 |

**15 × 3 splits = 45 runs.**

### 5.2 Primary jury — pre-registered definition

**Complementary Author-Partition, Condorcet-Inspired Weighted Jury.** Training authors are divided into three balanced blocks A, B, C; jurors train on **A+B**, **A+C**, **B+C**. Each sees ~2/3 of training data with equal training size across jurors — genuine author-level data diversity without the confound in which disjoint-thirds jurors are weaker merely from seeing less data.

Seed diversity is explicitly **not** the primary configuration: the method under evaluation builds diversity through complementary training data, and seed variation is the weakest, most error-correlated form.

**Terminology.** The classical Condorcet Jury Theorem concerns majority voting over discrete decisions, not weighted averaging of predicted probabilities. The paper describes the theorem as the **motivation** and the implemented rule as a **validation-weighted soft aggregation rule**, never attributing probability weighting to the theorem itself.

### 5.3 Parts B and C runs (confirmatory split only, seed 42)

| Run set | Seeds | Count | Purpose |
|---|---|---|---|
| BERT and RoBERTa on Twitter-derived training | 1 | 2 | Part B baselines |
| BERT and RoBERTa on Combined training | 1 | 2 | Part C baselines |
| BERT on **size-matched** Twitter, Reddit and Combined training | 3 | 9 | Corpus-Diversity Jury members |

One of these nine is **shared** with the Twitter BERT baseline: Twitter is the smallest corpus and is therefore the size-matching target, so "size-matched Twitter BERT, seed 42" *is* the Twitter BERT baseline run. It is trained once and used twice.

**+12 unique runs. Grand total = 57 runs, approximately 18–23 GPU-hours on a single A100.**

Two deliberate asymmetries. Part B and C **baselines** run one seed each, being secondary and descriptive; they therefore carry **no estimate of initialization variance**, and their results are reported descriptively with no significance claim attached. The **jury members** do get three seeds, because they feed the H4 dependence–gain analysis, which needs their variance.

**The size-matched subsets are frozen and shared across seeds.** For each corpus, one subsample is drawn, hashed, and written to disk; all three model seeds then train on that same subsample. Re-drawing the subsample per seed would make "seed variance" a mixture of initialization variance and subsampling variance, which would corrupt exactly the quantity Jury A is meant to isolate.

**Size matching is mandatory for the Corpus-Diversity Jury.** SemEval-2018 is roughly 11k tweets before Ekman filtering and shrinks further; the Reddit corpus is substantially larger. Without subsampling Reddit and Combined to the Twitter training size, Juror T would be weaker because it saw **less data**, not because it came from a different corpus — reintroducing exactly the confound §5.2 was designed to avoid, and making the Corpus-Diversity Jury uninterpretable within the H4 ladder.

### 5.4 Corpus-Diversity Jury

Juror T (Twitter-derived), Juror R (Reddit-derived), Juror C (Combined) — all BERT, all trained on size-matched corpora, evaluated on the R_clean author-disjoint test set. This is a **secondary** jury, not the primary.

**Question, not prediction:** we evaluate whether a corpus-diverse member with reduced target-domain competence helps or harms the ensemble. No expected direction is registered.

The theoretical frame belongs in the Discussion, and must be stated without overreach. The Condorcet Jury Theorem motivates aggregating competent voters, but it does so under restrictive assumptions — competence above chance **and** a particular structure of independence or bounded dependence among votes. **Our experiments do not test those assumptions.** They examine, empirically, whether adding a weaker but diverse member improves or degrades a finite, correlated ensemble.

The two outcomes are therefore not symmetric in what they license:

- If the corpus-diverse member **improves** the jury, diversity was empirically beneficial here. This does **not** show that the theorem's conditions were satisfied; an ensemble can improve for reasons the theorem does not describe, and the dependence structure it requires is not something we measure.
- If it **degrades** the jury, competence above chance was empirically insufficient — a weaker claim than refuting the theorem, and the one the data can actually support.

The paper says neither that the theorem was confirmed nor that it was violated.

No figure from the unavailable prior experiments is cited, in the manuscript or in any supporting document.

### 5.5 Diversity sources and the overlap ladder

| Jury | Diversity source | Pairwise training overlap **O** |
|---|---|---|
| Jury A — 3×BERT seeds | Initialization | 1.00 |
| Jury A′ — 3×RoBERTa seeds | Initialization | 1.00 |
| **Primary — A+B / A+C / B+C** | Data partition | 0.50 |
| Jury Z — Disjoint thirds | Data partition | 0.00 |
| **Corpus-Diversity Jury — T / R / C** | Corpus | partial, reported |
| *Phase 2* — BERT/RoBERTa/DeBERTa | Architecture | 1.00 (data), 0 (parameters) |

**Overlap definition.** For jurors *i*, *j* with training sets *T*ᵢ, *T*ⱼ:

**O**ᵢⱼ = |*T*ᵢ ∩ *T*ⱼ| / min(|*T*ᵢ|, |*T*ⱼ|)

With equal juror training sizes this gives 1.00 / 0.50 / 0.00. **Jaccard overlap** |*T*ᵢ ∩ *T*ⱼ| / |*T*ᵢ ∪ *T*ⱼ| is reported alongside (1.00 / 0.33 / 0.00). Both definitions appear explicitly so the ladder is unambiguous.

Jury A′ and the Corpus-Diversity Jury's members are free where the underlying runs already exist; only the size-matched runs in §5.3 are additional.

**The Discussion question this enables:** *which source of member diversity actually helps a jury ensemble?* — a substantially more useful question than whether one ensemble beats one transformer.

Note the structure precisely: the controlled **overlap ladder** has three points (1.00 / 0.50 / 0.00), two of which — complementary partitions and disjoint thirds — are the same diversity *source* at different overlap levels. The distinct diversity **sources** in Phase 1 are three: initialization, data partition, and corpus. The paper states it this way rather than claiming four sources, since architecture diversity arrives only in Phase 2.

---

## 6. Aggregation rules — fixed before any test evaluation

**Weights:** softmax over juror validation macro-F1 at T = 1.0, computed on validation only, frozen before any test evaluation.

**Primary aggregation:** validation-weighted **soft** voting.

**H3 control:** **equal-weight soft** voting (1/3, 1/3, 1/3) over the **same jurors and the same probability outputs**. Weights are the only difference.

**Sensitivity analyses:** weighted hard majority and plain hard majority — reported, but neither used for H3.

**Mandatory weight-activation diagnostics**, reported before H3 is interpreted:

- realized weight vector and spread of juror validation macro-F1
- proportion of test examples with complete three-way juror disagreement
- proportion where any single weight exceeds 0.5
- proportion where weighted and equal-weight aggregation produce different predictions

With three jurors of comparable quality, softmax-at-T=1 over close validation scores yields near-uniform weights. Under *hard* voting such weights can only alter a decision in three-way disagreements or when one weight exceeds 0.5 — the latter arithmetically unreachable. Soft voting lets weights act continuously, so H3 is empirical rather than algebraic. If the diagnostics show the mechanism is effectively inert, that is a publishable analytical finding: **Condorcet-style weighting does not activate for small juries of comparable-quality members** — plausibly explaining why weighted and entropy-based weighting differed so little in earlier ablations.

---

## 7. Error-dependence analysis (H4, exploratory)

**Jury size is fixed at three for Phase 1, and enumeration is exhaustive.**

Every jury type in the design is natively three-member: 3 BERT seeds, 3 RoBERTa seeds, A+B / A+C / B+C, A / B / C, and T / R / C. A five-member jury would necessarily *mix* diversity sources — BERT seeds plus RoBERTa seeds is no longer initialization diversity alone, and complementary plus disjoint partitions mixes two overlap levels — which destroys the very classification H4 is testing. Size-3 juries keep each configuration attributable to exactly one source, and match the size of the Primary jury used in H2 and H3.

Configurations are therefore **enumerated exhaustively**: every three-member jury constructible within a single diversity source, per split. No cap and no sampling rule, so there is no selection surface at all. Jury-size sensitivity (3 vs 5) is deferred to Phase 2, where an enlarged model pool makes mixed-source juries interpretable as a category of their own.

Per configuration: dependence (Pearson r on misclassification indicators, Q-statistic, disagreement measure, κ on error patterns — never r alone); mean and **best** member accuracy; training overlap **O** and Jaccard; ensemble accuracy under equal-weight soft, validation-weighted soft, and plain majority; and **gain over the best member**.

**Phase 1 H4 is descriptive, not a regression.** Fixing jury size at three has a consequence that must be stated plainly: within a single diversity source, each source yields essentially **one** distinguishable three-member jury per split — 3 BERT seeds, 3 RoBERTa seeds, A+B/A+C/B+C, A/B/C, and T/R/C. That is four or five configurations per split, not dozens. Fitting `gain ~ dependence + best-member accuracy + mean member accuracy` to four points is not merely underpowered; it is not estimable.

Nor can the count be inflated by crossing seeds. Enumerating every T-seed × R-seed × C-seed combination would give 27 juries, but those juries differ in **initialization as well as corpus**, so a configuration would no longer belong to exactly one diversity source — the property the size-3 restriction exists to protect. "Exhaustive enumeration" therefore means *within a single source*, and where a source admits only one clean jury per split, that is the number.

Phase 1 reports, per configuration: the four dependence measures, mean and best member accuracy, ensemble accuracy under each aggregation rule, gain over the best member, training overlap, and a 95% bootstrap CI on the gain. No multivariable model is fitted.

**H4 makes two separate claims, because the jury types are not equally available.** Parts B and C run on the confirmatory split only, so the Corpus-Diversity Jury exists on split 42 alone. An ordering of all three diversity sources cannot be checked for consistency across splits, and the protocol does not pretend otherwise.

**Claim 1 — across-split consistency of Part-A jury configurations.** Four jury types exist in all three splits: the BERT seed jury, the RoBERTa seed jury, the complementary-partition jury, and the disjoint-thirds jury. The question is whether their *ordering* by gain over best member reproduces in splits 42, 123 and 2024 — an ordering that repeats three times is more persuasive than a coefficient estimated on n = 4, and honest about the sample size.

Within Claim 1, one comparison is cleaner than the rest and is designated the **controlled BERT overlap ladder**:

| Jury | Backbone | Pairwise training overlap **O** |
|---|---|---|
| BERT seed jury | BERT | 1.00 |
| Complementary-partition BERT | BERT | 0.50 |
| Disjoint-thirds BERT | BERT | 0.00 |

All three hold backbone and jury size constant, so overlap is the varying quantity. The **3×RoBERTa seed jury is a strong-backbone robustness reference**, not a rung on this ladder: comparing it against the BERT seed jury varies architecture as well as anything else, so it cannot sit in a ladder meant to isolate overlap.

**Residual confound in the ladder, stated rather than left to a reviewer.** The 0.00 rung is not overlap-only: disjoint-thirds jurors each train on roughly a third of the data, while complementary jurors train on roughly two thirds. That rung therefore differs in **training size as well as overlap**. Measuring gain over the best member normalizes for member strength but does not eliminate this; the ladder is described as a controlled *comparison*, never as a clean manipulation of overlap alone, and the confound appears in the limitations.

**Claim 2 — corpus diversity, split 42 only.** Reported as a single-split exploratory extension, never as replicated evidence. It does, however, carry **within-split replication that the other sources lack**: with three seeds trained per corpus, three clean juries can be formed by holding the seed constant and varying only the corpus — (T₁,R₁,C₁), (T₂,R₂,C₂), (T₃,R₃,C₃). Seed is fixed inside each jury, so corpus remains the only varying source. Crossing seeds across members is not permitted, since that would reintroduce initialization diversity into a jury labelled corpus-diverse.

Member strength is handled throughout by reporting best-member accuracy beside every gain, so a reader can see directly whether a high-gain jury simply had a weak best member.

Extending Claim 2 across all three splits would require size-matched Reddit and Combined models on splits 123 and 2024 — twelve additional runs, taking the total to 69. That is not judged worth the cost; the wording is narrowed instead.

The multivariable regression moves to **Phase 2**, where the architecture-diverse pool supplies enough clean single-source configurations to estimate it.

**Headline figure:** dependence on x, gain over best member on y, points coloured by diversity source, one panel per split so rank consistency is visible directly. The split-42 panel carries the corpus-diversity points; the other two panels do not, and the caption says so.

---

## 8. Inference

### 8.1 Prediction vector rule

**Primary:** for any multi-seed system, the primary prediction vector is the run with the **highest validation macro-F1**, selected on validation only and frozen before test. Applied uniformly to H1, H2 and H3.

**Pre-specified robustness:** the seed-aggregated version (equal-weight probability average across seeds) is reported alongside for every hypothesis. Both are declared now, so neither can be chosen after seeing results.

**Naming discipline:** a seed-aggregated system is always labeled an ensemble and never presented as a single-model baseline.

### 8.2 Primary test — paired permutation on the primary metric

McNemar tests differences in *error rate*, i.e. accuracy — not macro-F1. Using a McNemar p-value as the primary test for a macro-F1 estimand is a mismatch, so:

**Confirmatory tests:** paired permutation (randomization) tests on Δmacro-F1 for **H2 and H3**. For each test example, the two systems' predictions are exchanged with probability 0.5; the metric difference is recomputed; 10,000 permutations yield the null distribution. This tests the primary metric directly.

**Exploratory H1 stress test:** paired permutation on Δaccuracy, on the eligible test subset, reported with effect size and CI and **excluded from the confirmatory multiplicity family**. Accuracy rather than macro-F1 because the eligible subset's rarest class has single-digit support.

**Secondary paired test:** McNemar with Cohen's h, for accuracy.

**Descriptive with CIs:** MCC, Cohen's κ, weighted F1.

### 8.3 Confirmatory split and replication

**Split seed 42 is the confirmatory split, designated before any training.** Each confirmatory hypothesis yields exactly one p-value, computed on split 42. **Holm–Bonferroni is applied across H2 and H3 only** — H1 left the confirmatory family by the §2.1 amendment, and is not corrected for alongside them.

**Splits 123 and 2024 are pre-specified replication splits.** For each hypothesis they report Δ, 95% CI, direction and p-value, summarized as replicated or not replicated.

**Replication is defined now:** same direction of effect **and** a 95% CI excluding zero. Directional agreement alone is reported as "directionally consistent," never as replication.

**Why not Stouffer combination.** The three splits partition the same corpus, so their test sets overlap and the p-values are not independent; combining them by a method that assumes independence would be unsound, and a disclaimer does not repair the mathematics. **Why not permutation across splits:** with three clusters, sign-flipping cannot produce a p-value below 2⁻³ = 0.125, so the hypotheses would be unanswerable regardless of effect size.

**Acknowledged trade-off:** confirmatory inference now rests on a single partition, which is the idiosyncratic-partition risk that motivated using multiple splits in the first place. The replication splits mitigate this **interpretively**, not statistically. The trade-off is taken deliberately, in preference to an unsound independence assumption, and is stated in the paper.

### 8.4 Remaining statistical plan

**Metrics for H2 and H3, fixed in advance.** Macro-F1 remains primary. Two additions are pre-registered rather than added later, because the corpus is severely imbalanced — `disgust` and `fear` together are under 4% of examples, giving them roughly 70–90 instances in a test partition:

- **Weighted-F1 as a declared secondary metric.** Not a replacement for macro-F1, but so a reader can see whether an observed difference is driven by the rare classes or by the bulk of the data. Under macro averaging, `disgust` carries one sixth of the score while being estimated from ~70 examples, so its variance can dominate the metric and mask real differences elsewhere.
- **Per-class F1 reported with its support**, always, for every compared system. A macro-F1 difference is not interpretable without knowing which classes moved and on how many examples.

Accuracy, MCC and Cohen's κ remain secondary. H1's exploratory analysis uses accuracy as its primary metric, per §2.1; the two analyses therefore use different primary metrics, which is stated rather than glossed. Mean ± SD across splits and seeds, with per-split numbers in an appendix — never means alone. 1,000-iteration clustered bootstrap 95% CIs for all effect sizes. Cohen's h alongside every accuracy difference. TOST with margin fixed at ±1 accuracy point; `p > .05` is never used as evidence of equivalence. Parameter counts, forward passes, latency and memory reported for every compared system.

---

## 9. Study structure and phasing

**Part A — Reddit primary study.** H2 (jury vs RoBERTa) and H3 (weighting) are confirmatory; H1 is a structural corpus diagnosis plus an exploratory stress test; H4 is exploratory. All confirmatory claims live here.

**Part B — Twitter-derived secondary.** Standalone BERT and RoBERTa; cross-corpus transfer in both directions, framed per §3.3.

**Part C — Combined exploratory.** Mixed-corpus baselines; Corpus-Diversity Jury; sensitivity analysis.

**Phase 1 — mandatory (57 runs).** Everything above. If compute is constrained, drop to two Part A split instances (~43 runs); H2 and H3 remain confirmatory on split 42, H1 remains an exploratory structural stress test, one replication split remains, and the H4 ladder weakens.

**Phase 2.** A dose–response author-overlap analysis is **deferred to a corpus with sufficient repeated-author density** — Section 0 showed that k = 2 reaches uniform coverage for only ~18–22% of eligible authors here, so a second dose on GoEmotions is not available. Also: DeBERTa-v3-base and the architecture-diverse jury; R_random with full seeds; keyword-ablation; CARER 5-class analysis if viable.

**Phase 3.** Temporal split via `created_utc`; ISEAR zero-shot (5 shared classes — ISEAR has no `surprise`); MentalRoBERTa; extended robustness; GoEmotions mental-health subreddit subset.

---

## 10. Manuscript revision map

| Section | Action |
|---|---|
| Title | Leakage-aware, non-clinical title |
| Abstract | Written last, from computed results. Include: experiments use publicly available Reddit- and Twitter-derived emotion corpora, with Reddit serving as the confirmatory leakage-controlled domain because author identifiers permit author-disjoint evaluation |
| Introduction | Keep background; contributions become H1–H4 plus the reproducibility artifact |
| Related Work | Reuse nearly in full; update the error-dependence gap; add compute-matched controls |
| Dataset | Delete Twitter/PRAW collection and the prior annotation protocol; replace with GoEmotions and SemEval provenance, Ekman mapping, newly computed κ, and the corpus description table |
| Methods | Rewrite: split construction, reserve pool, controlled overlap, jury definitions, aggregation rules, inference rule |
| Results | Delete all prior tables and figures; regenerate solely from new saved predictions |
| Discussion | Leakage, juror competence, weight activation, diversity sources, cross-corpus transfer, compute cost |
| §5.2 | Reframe as *Implications for Online and Biomedical Research*; clinical limits move into the Discussion |
| Ethics | Secondary use of public data; delete ethics-exemption, encryption, differential-privacy claims |
| Limitations | Reconstruction; label harmonization; the corpus/annotation confound (§3.3); the asymmetric single-label rules across corpora (§3.2); the author-balanced test estimand (§4.4); single-partition confirmatory inference (§8.3); the realized matching hierarchy (§4.2) |
| Conclusion | Rewritten entirely from new results |

**Deleted permanently:** Table 1 and the 85,000 / 45,000 collection claims; the prior Fleiss' κ and psychologist validation; Tables 4 and 6 and all prior ablations and figures; temporal and trajectory analyses; demographic age analysis; prior robustness, cross-dataset, low-resource and manual error analyses.

**Prior unsaved experimental numbers** — including the cross-platform figures (Twitter→Reddit ≈ 38.7%, Reddit→Reddit ≈ 80.3%, combined ≈ 80.0%) — may guide design internally but must not appear in the manuscript in any form, including as motivation for a design choice or as a stated expectation. The §5.4 expectation rests on the theoretical argument alone.

**On corpus structure.** Parts B and C are included because cross-corpus transfer and corpus-level diversity are substantive research questions that strengthen H4, and because a per-source corpus description table is standard practice in any multi-corpus study. They are **not** included to make the revision resemble the previous submission. Experimental design driven by surface resemblance to unavailable data would reproduce the original problem in cosmetic form.

---

## 11. Reviewer D — disposition

| Comment | Action |
|---|---|
| Inter-classifier error correlation unaddressed | Becomes H4 with four dependence measures, three diversity sources and a three-point controlled overlap ladder |
| "Section V.D" unclear | Fix or remove the cross-reference |
| "future keyword-ablation" | Clarify wording; execute in Phase 2 |
| 85.3% undiscussed | Prior numbers deleted; all results regenerated |
| Fig. 4 needs explanation | New confusion matrix with accompanying error discussion |
| §5.2 reads as disclaimer | Reframed as implications |
| "1829" / "1529" | Corrected; largely disappears with the deleted analyses — check §I |
| Typos, incomplete sentences | Full pass after the new version stabilizes |
| Duplicate captions, Figs. 2–3 | Removed |
| Scope relevance (2/5) | Connect to online behavioral research; no clinical framing anywhere |

---

## 12. Reproducibility artifact

`build_dataset.py`, `section0_feasibility.py` and the mapping table; dataset cards with source and license notes for all three corpora; split index files with cryptographic hashes, including reserve-pool indices; training and evaluation configs per run; environment lockfile with pinned versions; per-run metrics CSV; **per-sample predictions for every run**; a single analysis notebook regenerating every table, statistic and figure; README with exact reproduction commands. Deposited to Zenodo for a DOI.

**Licensing stance:** release *the maximum reproducibility artifact permitted by the applicable dataset and source terms*, verified per corpus before deposit. Apache-2.0 on a repository and its code does not automatically extend to redistribution of underlying platform content. If redistribution of processed text is clearly permitted, release the processed corpora and splits; otherwise release build scripts, indices, hashes, mappings and predictions.

---

## 13. Editor communication (parallel; does not block the work)

> During preparation of the revision, the original raw experimental corpus became unavailable. To avoid retaining results that can no longer be independently verified, we are reconstructing the empirical study using documented public data sources and will re-run all analyses retained in the revised manuscript. The topic and research question remain unchanged, but the methodology, data-provenance description, and empirical results will be revised substantially. We would appreciate your guidance on whether you prefer this transparent reconstruction to be submitted as the requested revision or handled as a new submission.

The manuscript describes the study that was performed. The editor correspondence and response-to-reviewers document explain the history.

---

## 14. Risks

| Risk | Management |
|---|---|
| Too few multi-post authors | §0 check before freeze; explicit choice between exploratory H1 and `link_id` re-specification |
| Twitter corpus too small after Ekman mapping | Source is fixed as SemEval-2018 alone; report the surviving N and size-match the Corpus-Diversity Jury regardless. If N is very small, Parts B and C are reported as exploratory only |
| Platform/annotation confound | Framed as cross-corpus transfer; stated wherever transfer results appear |
| Confirmatory inference on one partition | Pre-specified replication splits; trade-off stated in the paper |
| No leakage effect found | Reported as-is; H1 is a question |
| Jury does not outperform | Becomes the conditional finding: diversity alone is insufficient |
| Weighting mechanism inert | Reported via §6 diagnostics as an analytical contribution |
| Data licensing | Maximum permitted artifact; verified before deposit |

---

## 15. Decisions required from the six authors

1. Delete all unverifiable prior results; make no attempt to reconstruct them numerically.
2. Adopt GoEmotions raw as the primary corpus and SemEval-2018 as the secondary; Parts B and C are secondary and exploratory throughout.
3. Adopt the new title; no mental-health framing without a corresponding analysis.
4. **Freeze the H1 exploratory specification, the H2–H3 confirmatory hypotheses, the jury definitions (§5.2, §5.4), split construction (§4), aggregation rules (§6), and the inference rule (§8)** — after §0 reports and the H1 amendment is signed.
5. Accept in advance that H2, H3 and the Corpus-Diversity Jury may return null or negative results, reported as findings.
6. Commit to Phase 1 only; Phases 2 and 3 conditional.
7. Approve transparent editor communication.
8. Every number in the paper must be recomputable from a saved artifact.

---

## 16. Execution order

1. Run §0; report author distribution; apply the decision rules; fill `[DOSE_K]` and `[H1_METRIC]`.
2. Build and count the single-Ekman SemEval-2018 corpus; record the surviving N. CARER is not involved in the Phase-1 source decision (Phase 2 only, 5-class).
3. Author sign-off; add the freeze declaration to this document.
4. Editor query sent (parallel).
5. Build all corpora; construct and hash R_clean, R_overlap, R_random and reserve pools for three split instances; verify the §4.2 invariants; build size-matched training sets for the Corpus-Diversity Jury.
6. Pilot: RoBERTa on R_clean split 42, one seed — end-to-end pipeline check.
7. Execute Phase 1 (57 runs).
8. Compute all statistics from saved predictions in the analysis notebook.
9. Rewrite Sections 1–6 from computed results.
10. Prepare the response-to-reviewers document and finalize the repository.

**Once frozen, the status, estimands, metrics and analysis rules for H1–H3, the jury definitions, and the split, aggregation and inference rules are not revised in response to observed results.** H1's demotion happened before this point and is recorded by amendment; nothing may move in the other direction.
