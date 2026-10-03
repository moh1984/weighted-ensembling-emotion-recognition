# Part C Dataset-Construction Amendment

**Status:** frozen before Part-C dataset build V2 and before any
Part-C model training.

**UTC:** 2026-09-01T07:02:00.468616+00:00

The first Part-C structural build used the same SHA-256 ranking
stream for the size-matched Reddit member R and Combined member C.

The resulting C sample contained 1,760 Reddit-derived records and
145 Twitter-derived records. All 1,760 Reddit-derived C records
were also present in R, giving R-C overlap O = 0.9238845.

This was identified during the mandatory structural overlap audit,
before any Part-C model was trained and before any Part-C
model-performance outcome was observed.

The build is therefore retained as a superseded structural-preflight
artifact and is not used for model training.

The correction changes one implementation detail only:

- R ranking key:
  SHA256("20260901|R|" + example_id)

- C ranking key:
  SHA256("20260901|C|" + example_id)

All other frozen decisions remain unchanged: training size
1905, no stratification, no class or source balancing,
sampling without replacement, fixed T, the same subset reused across
model seeds, the twelve-run grid, validation rules, aggregation rules,
and exploratory status.

The first V2 samples generated under these namespace-separated keys
will be accepted without further resampling regardless of their
observed class composition, source composition, or overlap.
