"""Scripted download and verification of raw source tables.

Populates data/raw/ (write-once, git-ignored). Verifies file identity —
expected filenames and row counts — so reproducibility is checked rather
than assumed, per Repository Architecture v1.0 Section 7.

Not yet implemented — scaffolded in Commit 1; implemented in Commit 3
(Data Preparation).
"""
