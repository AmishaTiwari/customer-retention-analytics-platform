# ADR 0010: Primary K Value for Precision@K and Lift@K

**Status:** Proposed

---

## Context

The locked Business Design defines Precision@K and Lift@K as Primary Metrics but explicitly defers the value of K ("K is treated as a configurable business parameter rather than a fixed value. The selected value and sensitivity analysis will be documented after Dataset Selection."). No operational headcount, campaign capacity, or budget figure exists anywhere in the locked documents to derive a specific K, as either a fixed count or percentage. This ADR resolves that deferred decision as a project-level assumption informed by empirical analysis, not a locked-document derivation.

---

## Decision

Primary K = 10% of the scored population. Sensitivity analysis range = {5%, 10%, 15%, 20%, 25%}. K is expressed as a percentage rather than a fixed count, since no headcount or capacity figure exists in the locked documents to justify a fixed number.

### What is established by locked documents

- Precision@K and Lift@K are required Primary Metrics (Business Design, ADR-0008).
- Outreach capacity is limited, motivating a top-K framing over a full-population threshold.
- No specific K value, percentage, or capacity figure is specified anywhere in the locked documents — the choice of K is out of scope for the Business Design and is a project-level decision made here.

### What is observed empirically

A scratch analysis (not part of production code) computed Precision@K and Lift@K at K in {5%, 10%, 15%, 20%, 25%} for six candidate model variants (LR, RF, XGB — tuned and untuned) on the held-out test set (n=1,408, churn rate 26.49%):

- At K=5%, all six model variants produce identical Precision@K (0.9286) and Lift@K (3.5052) — no discriminative power between models at this K.
- From K=10% onward, meaningful separation appears (e.g. at K=10%: LR tuned = 0.9078 vs RF untuned = 0.8440).
- Separation persists and widens through K=15-20%, narrowing somewhat by K=25% as precision converges toward the base rate.

### What is a project assumption / decision

- K=10% is chosen as the primary evaluation point: not derived from any locked figure, but a judgment call made after ruling out K=5% empirically, balancing a plausible capacity-constrained "top-risk segment" against real model differentiation.
- The sensitivity range {5%, 10%, 15%, 20%, 25%} is chosen to show how model selection would change under a different capacity assumption, and to keep K=5% visible in reporting as a documented non-differentiating edge case.
- This decision is revisitable if a real capacity/budget figure is ever added to the Business Design.

---

## Alternatives considered

- **Fixed customer count (e.g. K=500 customers)**: rejected — no headcount or capacity figure exists in any locked document to justify a specific count; a percentage generalizes better across dataset sizes.
- **K=5% as primary**: rejected — empirically produces identical results across all candidate models, making it useless for model selection.
- **K=20% or 25% as primary**: considered — still shows differentiation, but represents a larger contacted segment than K=10%; K=10% was preferred as the more conservative "top-risk" framing consistent with the Business Design's capacity-limited narrative.

---

## Consequences

- Commit 9's evaluation code computes Precision@K and Lift@K at K=10% as the headline Primary Metric value, with the full {5%,10%,15%,20%,25%} sweep reported as supporting sensitivity analysis.
- Model selection (ADR-0008) uses the K=10% Precision@K/Lift@K values as part of its unranked Primary Metrics comparison.
- K=5% remains in reporting for transparency but is documented as a non-differentiating edge case, not a basis for selection.
- This ADR should be revisited if the Business Design is ever updated with a real operational capacity figure.

---

**Decision Date:** 20 September 2026
