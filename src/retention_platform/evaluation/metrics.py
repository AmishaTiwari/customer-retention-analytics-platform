"""Precision@K, Lift@K, PR-AUC, Brier Score, and related metrics.

Unit-tested against hand-computed toy cases (tests/test_metrics.py),
since a subtly wrong Precision@K would poison every downstream
conclusion and the model-selection decision itself.

Not yet implemented — scaffolded in Commit 1; implemented in Commit 9
(Evaluation & error analysis).
"""
