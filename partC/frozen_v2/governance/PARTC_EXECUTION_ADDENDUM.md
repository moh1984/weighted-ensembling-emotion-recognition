# Part C Execution Addendum

**Status:** frozen before Part-C dataset sampling and before any Part-C model training.

**UTC freeze time:** 2026-09-01T06:19:23.998818+00:00

Part C was specified in the original frozen study protocol. This addendum
does not modify Research Protocol v7; it records implementation details that
were not completely specified there.

## Temporal status

Part A and the external Part-B evaluation had already been completed when
this implementation addendum was created. No Part-C dataset sampling, model
training, prediction generation, or model-performance analysis had occurred.

Part C remains exploratory and secondary. It is not part of the H2/H3
multiplicity family and no confirmatory significance claim will be made from it.

## Combined corpus

The full Combined condition is the union of the corresponding frozen
Reddit-derived R_clean and Twitter-derived partitions. No augmentation,
class balancing, or source balancing is introduced. Realized class and source
composition will be reported.

## Corpus-diversity jury

The jury contains three BERT members:

- T: Twitter-derived training
- R: Reddit-derived R_clean training
- C: Combined training

Each member receives exactly 1905 training examples because the
Twitter-derived training set is the size-matching target.

T uses all 1905 frozen Twitter-derived training examples.

R and C are sampled without replacement by deterministic SHA-256 ranking of
example identifiers using seed 20260901. No class stratification or
class balancing is applied. The resulting R and C subsets are frozen once and
reused for model seeds 1, 2 and 3.

## Validation and checkpoint selection

Twitter-trained checkpoints are selected using Twitter validation macro-F1.
Reddit-trained checkpoints are selected using Reddit R_clean validation
macro-F1. Combined-trained checkpoints are selected using Combined validation
macro-F1. No test outcome enters checkpoint selection.

For Platform-Jury competence weighting, the already-selected T, R and C
models are all evaluated on the common Reddit R_clean validation set. The
weights are softmax(validation macro-F1 / T) with T = 1.0. This common-domain
evaluation is used for aggregation weights only and does not trigger further
checkpoint selection or fitting.

## Aggregation and evaluation

Primary aggregation is validation-weighted soft voting. Equal-weight soft
voting is reported alongside. Weighted hard voting and plain hard majority
are sensitivity analyses.

The Platform Jury is evaluated primarily on the frozen Reddit R_clean
author-disjoint test set.

## Run grid

The protocol requires twelve unique Part-B/C training runs under the v7 grid.

Four full-data baseline runs:
1. BERT Twitter, seed 1
2. RoBERTa Twitter, seed 1
3. BERT Combined, seed 1
4. RoBERTa Combined, seed 1

Platform-Jury BERT members use T/R/C at seeds 1, 2 and 3. BERT Twitter
seed 1 is shared with the Twitter baseline, giving twelve unique runs rather
than thirteen.

## Duplicate handling

Cross-source normalized-text overlap is audited before training. Train-to-train
overlap is reported rather than silently removed. Any normalized duplicate
crossing a train/validation/test boundary causes a fail-closed stop before
Part-C training.

## Interpretation

Part C permits an executed corpus-diversity comparison if completed.
It does not add architecture diversity. Cross-corpus differences remain
multiply confounded by source/platform, annotation protocol, label
harmonisation and preprocessing.
