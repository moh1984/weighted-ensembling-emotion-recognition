# Section 0 - Pre-freeze Feasibility Report

Ekman mapping source: **official_file_verified**
Near-duplicate pass run: **True**
Corpus content SHA-256: `05343493a0e155c7`
Corpus id-set SHA-256: `9801d95a33bac6ee`
Ekman mapping SHA-256: `d6b1fea382917c16`

Raw source files:
- goemotions_1.csv: `cac049036bad5d68`
- goemotions_2.csv: `f699ecc5aa425c17`
- goemotions_3.csv: `467f1e7191af00f2`

Near-duplicate algorithm: exact match on normalised text, then connected components over a 10-nearest-neighbour cosine graph with edges above 0.85; one item retained per component. Not an exhaustive all-pairs search.
Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (revision requested: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, resolved: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

## Corpus

- n_examples: 34951
- n_authors: 31618
- pct_authors_ge2: 7.9
- pct_authors_ge3: 1.57
- pct_authors_ge5: 0.24
- pct_posts_from_multipost_authors: 16.68
- max_posts_single_author: 15

Class distribution: joy 20,163, anger 5,835, surprise 4,954, sadness 2,915, fear 609, disgust 475

## Rater agreement (binary Fleiss' kappa per emotion)

- anger: kappa 0.363 (n=57,877)
- disgust: kappa 0.2736 (n=57,877)
- fear: kappa 0.3829 (n=57,877)
- joy: kappa 0.5211 (n=57,877)
- sadness: kappa 0.4128 (n=57,877)
- surprise: kappa 0.3511 (n=57,877)

- rater annotations selecting >1 Ekman category: 8.67%
- items containing at least one such annotation: 23.59%
- overall categorical kappa: omitted. Raters may select multiple emotions, so per-item votes can exceed the rater count; categorical Fleiss' kappa is undefined under multi-label annotation. Krippendorff's alpha with MASI distance is the defined alternative if an aggregate figure is wanted.

## Label construction by rater count

| raters | items | single | multi | below threshold |
|---|---|---|---|---|
| 1 | 132 | 0 | 0 | 132 |
| 2 | 1,426 | 630 | 21 | 775 |
| 3 | 37,145 | 22,976 | 2,437 | 11,732 |
| 4 | 3,135 | 1,984 | 322 | 829 |
| 5 | 16,171 | 10,901 | 3,475 | 1,795 |

Sensitivity, strict-majority rule (`votes > n_raters / 2`): single 33,591, multi 2,818, below 21,600.
A fixed >=2 threshold is a majority of three raters but only 40% of five, so items with more raters are likelier to reach the threshold in two categories and be routed to the multi-label subset.

## Structural diagnosis: is an author-overlap mechanism present?

- posts_per_author_mean: 1.105
- pct_authors_ge2: 7.9
- pct_posts_from_multipost_authors: 16.68
- min_reserve_pct_of_corpus: 1.28
- max_achievable_overlap_pct_of_training: 1.409
- mechanism_available: False

> Author overlap is structurally near-absent in this corpus: the reserve pool - every post that could possibly be injected - is a small fraction of the corpus. A null stress-test result is therefore consistent with the absence of a mechanism and must NOT be read as evidence about the size of author leakage in emotion benchmarks generally.

## Design sensitivity (MDE, 80% power, alpha = 0.05 two-sided)

MDE depends on the discordance rate between the compared systems, which is unknown in advance, so a range of assumptions is reported rather than a single figure.

| seed | eligible test N | MDE at r=0.05 | MDE at r=0.1 | MDE at r=0.2 | MDE at r=0.5 |
|---|---|---|---|---|---|
| 42 | 391 | 3.17 pts | 4.48 pts | 6.33 pts | 10.01 pts |
| 123 | 340 | 3.4 pts | 4.8 pts | 6.79 pts | 10.74 pts |
| 2024 | 410 | 3.09 pts | 4.37 pts | 6.18 pts | 9.78 pts |

## Splits

| seed | train | val | test | reserve | excl. | ratios % | eligible (strict/loose) | eligible test | min class support |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 24,063 | 5,157 | 5,157 | 533 | 41 | 70.0/15.0/15.0 | 391/419 | 391 | 4 |
| 123 | 24,125 | 5,171 | 5,171 | 449 | 35 | 69.99/15.0/15.0 | 340/366 | 340 | 4 |
| 2024 | 24,060 | 5,156 | 5,156 | 546 | 33 | 70.0/15.0/15.0 | 410/435 | 410 | 4 |

## Dose probe

| seed | k | coverage % | injected | overlap % of train | exact 4-var % | -ppa % | -len % | label-only % | invariants |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 1 | 100.0 | 391 | 1.625 | 44.2 | 46.3 | 7.7 | 1.8 | OK |
| 42 | 2 | 21.74 | 170 | 0.706 | 30.6 | 60.0 | 7.6 | 1.8 | FAIL |
| 42 | 3 | 8.18 | 96 | 0.399 | 34.4 | 59.4 | 5.2 | 1.0 | FAIL |
| 123 | 1 | 100.0 | 340 | 1.409 | 47.1 | 42.6 | 8.8 | 1.5 | OK |
| 123 | 2 | 19.41 | 132 | 0.547 | 28.0 | 61.4 | 10.6 | 0.0 | FAIL |
| 123 | 3 | 7.06 | 72 | 0.298 | 26.4 | 58.3 | 15.3 | 0.0 | FAIL |
| 2024 | 1 | 100.0 | 410 | 1.704 | 42.0 | 48.8 | 6.8 | 2.4 | OK |
| 2024 | 2 | 17.8 | 146 | 0.607 | 26.0 | 65.1 | 6.2 | 2.7 | FAIL |
| 2024 | 3 | 7.8 | 96 | 0.399 | 20.8 | 70.8 | 6.2 | 2.1 | FAIL |

**freeze_quality: True**

- official_ekman_map: True
- near_dedup_ran: True
- dedup_model_pinned: True
- dedup_model_revision_verified: True
- no_preview_overrides: True

## Decision

- **h1_confirmatory_feasible**: False
- **h1_status**: requires_amendment
- **exploratory_dose_k**: 1
- **fixed_dose_k**: None
- **h1_primary_metric**: accuracy
- **min_eligible_test_examples**: 340
- **min_class_support_eligible**: 4
- **worst_exact_4var_match_pct**: None
- **worst_label_only_pct**: None
- **matching_note**: Hierarchical matching was fixed before model training. Training size and class composition are held exactly constant; subreddit, text length and author activity are matched hierarchically where feasible. The complete fallback distribution is reported and determines the strength of the descriptive claim - not whether the method is accepted post hoc.
- **verdict**: DO NOT FREEZE H1 AS CONFIRMATORY. Choose explicitly, before any training: (a) demote H1 to exploratory, or (b) re-specify the grouping unit as link_id - which changes the construct from author leakage to thread leakage and requires the title, central message and H1 wording to change accordingly.