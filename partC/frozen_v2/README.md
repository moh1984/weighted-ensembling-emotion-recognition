# Part C Frozen Dataset — V2

This is the training-authorized Part-C dataset build.

V1 remains retained as a superseded structural-preflight
artifact and MUST NOT be used for model training.

V2 was generated only after the sampling-stream correction
was frozen, and before any Part-C model training or
model-performance observation.

Platform size: 1905

Sampling:
R = SHA256("20260901|R|" + example_id)
C = SHA256("20260901|C|" + example_id)

No stratification, class balancing or source balancing.
No resampling after observing V2 composition or overlap.

T is the full frozen Twitter-derived training set.

The twelve-run execution grid is unchanged from V1.
