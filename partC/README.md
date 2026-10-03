# Part C — Combined-Corpus and Platform-Jury Extension

This directory contains the complete frozen Part-C execution chain. Historical
artifacts are preserved byte-for-byte from the saved Kaggle datasets; none of
the frozen files below were rewritten while assembling this final repository.

## Directory order

1. `frozen_v2/` — final V2 Part-C manifests, identifier lists, run grid and provenance (text-bearing parquet files intentionally omitted), overlap audit and governance.
2. `training_freeze/` — frozen training specification and `run_partC.py`.
3. `analysis_freeze/` — analysis specification frozen before the first Part-C
   model training/test prediction.
4. `final_results_freeze/` — 12 run metadata files and 48 prediction CSV files,
   frozen before the first Part-C test-performance analysis.
5. `analysis_results_freeze/` — frozen descriptive analysis outputs, including
   full baselines, jury weights, four aggregation rules, dependence analysis,
   training-overlap table and the 1,000-replicate paired bootstrap.

## Timing and interpretation

Part C was pre-specified in the study protocol, but it was executed after the
Part-B Twitter gold labels had been opened. Twitter-test results are therefore
post-Part-B exploratory descriptive transfer evidence and were not used for
training, checkpoint selection or Part-C jury weighting. The primary Platform
Jury evaluation is on the Reddit test set and follows the frozen Part-C analysis
specification.

Part C is exploratory/secondary: no p-values, no Holm adjustment and no formal
hypothesis testing were used. Architecture-diverse Phase-2 experiments were not
executed in Part C.

## Main descriptive result pattern

Combined-corpus training produced the largest transfer improvement. On Reddit
`reddit_test`, BERT macro-F1 was 0.265478 after Twitter-only training versus
0.698861 after Combined training; RoBERTa was 0.355027 versus 0.705739. The
weighted-soft Platform Jury had gain over the best member of -0.004671,
+0.010460 and +0.008445 across the three frozen seed triplets. Low training-set
overlap did not imply low error correlation: R-C correctness correlations were
0.778551, 0.811940 and 0.782243.

All values above are reproduced from `analysis_results_freeze/` and are
reported descriptively under the frozen analysis policy.
