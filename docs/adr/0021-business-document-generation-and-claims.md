# ADR 0021: Business Document Generation and Claims

**Status:** Accepted

---

## Context

Commit 11 delivers business-facing documents beside the customer list. Two questions need a recorded answer. The first is how the documents are produced: a document that is written by hand, or built from different data than the list, goes stale or disagrees with the list. The second is what they may claim: a generated recommendation is easy to read as a promise, and the model supports associations, not causes (ADR-0020).

This ADR fixes the deliverables of the milestone, where and how the two documents are generated, and the rules for the recommendations document. The economics content is in ADR-0019. The interpretation caveat is in ADR-0020 and is not repeated here. The exact file schemas, file names and publication guarantees are in Platform Output Contract v2, and the section and table layouts are in the Commit 11 documentation.

---

## Decision

### 1. The deliverables

Commit 11 delivers:

- the extended `outputs/prioritized_outreach_list.csv` under contract v2 (the same file, no second list);
- **one** economics document (the threshold and economics summary and the expected-value-and-sensitivity analysis are the same document, ADR-0019);
- **one** business recommendations document;
- a **closing check**, which is a task and not an output: regenerate the reference run, inspect every output, and confirm that the list, the economics document and the recommendations agree with each other, that the model-performance figures agree with ADR-0014, and that the economics and the recommendations use the same assumptions.

There is no combined summary document.

### 2. Both documents are generated inside `make score`

`make score` produces the list, the two documents and the run summary from one scoring execution, using the loaded model, the selected scoring population, the ranked table, the configuration and the model and scoring run IDs. In detail:

- The economics calculation uses `monthly_charge` from the in-memory scoring population, joined to the ranked table by `customer_id`. The saved model schema is checked at implementation to confirm the field is available.
- Both documents use the same execution, model, population, configuration and run metadata as the list. Nothing is reconstructed from published files, which avoids stale or mismatched outputs.
- There is no separate command and no new tracking run type. `make reproduce` regenerates the complete output set through the scoring workflow.
- All four outputs are built in memory before publication begins, so an error during document generation happens before anything is published. How the four files are published, and what is checked before the run summary, is specified in the contract (per-file replacement in a fixed order, the summary last, and the run IDs read back from the documents).

### 3. The recommendations document

The recommendations document has a **fixed outline**, **facts computed from each scoring run**, and a **small set of predefined conditional recommendations** tied to that run's results. The eight sections are:

1. what the list is, with the historical-snapshot notice;
2. risk tiers and their counts;
3. the profile of the top-K outreach group;
4. global feature importance, as normalized shares of the top original features;
5. customer-level drivers and how to interpret them;
6. headline economics, with a reference to the separate economics document;
7. business recommendations;
8. limitations.

The rules for its content:

- **Evidence rule.** Every recommendation is tied to a specific figure or finding computed from the current scoring run, and the document identifies that figure or finding. The recommended action is a conditional next step based on that finding; the action itself does not have to contain a number, and no sentence is required to carry a numerical result. No run-specific result is hard-coded in the recommendation text.
- **Wording rule.** Findings are model associations, not causes (ADR-0020). Recommendations are about how to use the list, what to investigate and what may warrant further evaluation. No text claims that an offer will work or that changing a feature will prevent churn.
- **Actionability.** The document discusses which features may be actionable, contextual or worth investigating. Features are not filtered out of the customer drivers because they are not directly actionable.
- **Fiber Optic.** The Commit 5 follow-up is part of this document. The global-importance view reports `internet_type` at the original-feature level, with the current run's share and rank and the correlation caveat (`internet_service` and other related features). It makes no claim that Fiber Optic causes churn, infers no Fiber-specific effect from the aggregate share, and does not claim to explain why Fiber Optic customers churn more in the observed data. The reference-run figures are evidence in ADR-0020, and the document never hard-codes them.

---

## Alternatives considered

- **A separate command that reads the published files and writes the documents.** Rejected: it reconstructs the scoring context from files, so a stale or mismatched file can produce a document that disagrees with the list. It would also need its own command and tracking.
- **A hand-written recommendations document.** Rejected: it goes stale as soon as a run changes, and it conflicts with generating the documents from each scoring run.
- **A document of facts only, with no recommendations.** Rejected: the agreed deliverable includes recommendations, and the evidence and wording rules make them safe.

---

## Consequences

- The documents are regenerated on every scoring run, are Git-ignored like the rest of `outputs/`, and replace the previous ones. `outputs/` holds the latest set and is not an archive.
- A change to the recommendation logic changes what the documents may say, so the recommendation text and its conditions are reviewed with the implementation and covered by tests.
- Tests at implementation cover the evidence rule and the wording rule where they can be checked, and the contract checks on the documents.
- The recommendations are conditional. A run whose facts do not trigger a condition does not show that recommendation.
- Because publication is not atomic (contract v2), a failed run can leave a mix of new and old files. The contract and the failure report state this.
- The closing check is a task at the end of the milestone, recorded in the implementation tracker. It is not an output.

---

**Decision Date:** 17 October 2026 (decisions approved 15 and 16 October 2026; recorded here before implementation)
