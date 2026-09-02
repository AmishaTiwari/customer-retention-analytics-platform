# ADR 0006: Voluntary-Churn Target Mapping

**Status:** Accepted

---

## Context

The Business Design defines voluntary churn as *"customer-initiated service cancellation where the customer chooses to discontinue the service,"* and explicitly excludes involuntary churn (*"service termination due to fraud, non-payment, or operational reasons"*) from the intended prediction target. It also anticipated that if the selected dataset doesn't natively distinguish voluntary from involuntary churn, the available label would need to be treated as an approximation — a fallback documented as active.

The raw dataset provides a `Customer Status` column (Stayed / Churned / Joined) and, for churned customers only, a `Churn Category` and a more granular `Churn Reason`. Neither field was designed against this project's specific voluntary/involuntary distinction — `Churn Category` groups reasons by theme (Competitor, Attitude, Dissatisfaction, Price, Other), not by voluntary/involuntary status.

Dataset Understanding (Stage 2) inspected the actual `Churn Reason` values for all 1,869 churned customers via a full crosstab against `Churn Category`, surfacing the real data this decision is based on rather than assuming a mapping in the abstract.

---

## Observed Dataset Evidence

The crosstab of `Churn Category` × `Churn Reason`, for churned customers only, showed:

| Churn Category | Count | Reasons |
|---|---|---|
| Competitor | 841 | Better devices, better offer, more data, higher download speeds |
| Attitude | 314 | Attitude of support person, attitude of service provider |
| Dissatisfaction | 303 | Product/service dissatisfaction, network reliability, limited services, poor online/phone support expertise |
| Price | 211 | Price too high, extra data charges, long distance charges, lack of affordable speed |
| Other | 200 | Don't know (130), Moved (46), Deceased (6), Poor expertise of phone support (18) |

One data-placement observation: "Poor expertise of phone support" (18 rows) is grouped under `Churn Category = Other` in the source data, despite reading identically in substance to the phone/online support expertise reasons already grouped under `Dissatisfaction`. This is treated as a data-placement inconsistency in the source categorization, not a distinct reason requiring its own business judgment.

---

## Decision

The following mapping is applied to construct `is_voluntary_churn`:

| Churn Category / Reason | Classification |
|---|---|
| Competitor (all reasons) | Voluntary |
| Attitude (all reasons) | Voluntary |
| Dissatisfaction (all reasons) | Voluntary |
| Price (all reasons) | Voluntary |
| Other → "Poor expertise of phone support" | Voluntary (reclassified as Dissatisfaction) |
| Other → "Moved" | Voluntary (assumption — see below) |
| Other → "Don't know" | Voluntary (assumption — see below) |
| Other → "Deceased" | **Excluded from the modeling dataset entirely** |

Resulting rule: every row where `Customer Status = Churned` is labeled `is_voluntary_churn = True`, except the 6 rows where `Churn Reason = Deceased`, which are dropped from the modeling dataset — not labeled 0 (stayed) and not labeled 1 (voluntary churn), because neither label accurately represents what happened.

---

## Distinguishing Observed Evidence from Business Assumptions

The classifications above rest on two different kinds of grounding, and this distinction matters for anyone auditing this decision later.

**Backed directly by the dataset (not a judgment call):**

- Competitor, Attitude, Dissatisfaction, and Price all describe reasons where the customer explicitly cited a factor that led them to actively cancel. These map to voluntary churn with no interpretive gap between the raw data and the Business Design's definition.
- "Poor expertise of phone support" reclassified under Dissatisfaction is a data-quality correction, not a business judgment — the reason text itself is indistinguishable from reasons already classified as Dissatisfaction.
- "Deceased" is factually not a customer choice, and does not fit the Business Design's involuntary definition either (not fraud, not non-payment, not an operational reason). Excluding these rows is the only option that doesn't misrepresent what happened.

**Explicit project assumptions, not proven by the data (judgment calls):**

- **"Moved" (46 rows) is assumed voluntary.** The underlying cause is circumstantial (a geographic constraint) rather than dissatisfaction with ConnectTel, and could reasonably be argued to sit outside the *spirit* of "customer chooses to discontinue." It is classified as voluntary here because, under the *letter* of the locked Business Design definition, the customer is still the one initiating the cancellation — but this is an interpretive choice, not something the data itself proves. A different, equally defensible project could treat "Moved" as a third, excluded category, the same way "Deceased" is handled here.
- **"Don't know" (130 rows) is assumed voluntary.** This is the single largest ambiguous group and the assumption with the least direct support: there is no stated reason at all. It is classified as voluntary on the basis that the customer did actively cancel — the *absence of a stated reason* does not by itself indicate the cancellation was involuntary. This assumption is not verified against any external evidence and should be treated as the most contestable judgment call in this ADR.

These two assumptions collectively affect 176 of 1,869 churned rows (~9.4%) and are the parts of this decision most likely to be challenged in review or in an interview setting — they are called out explicitly here for that reason.

---

## Consequences

- `sql/03_target/` implements this mapping to construct `is_voluntary_churn`, the modeling target.
- The 6 `Deceased` rows are excluded from the modeling dataset at the SQL layer, not just from the target — they do not belong in either class.
- `docs/data_dictionary_leakage_audit.md` will classify `Churn Category` and `Churn Reason` as Target Construction Only (used to build the target, excluded from modeling features) and will reference this ADR for the disposition rationale.
- `tests/test_target.py` will assert this mapping's behavior directly (e.g. deceased rows excluded, moved/don't-know rows treated as voluntary, poor-phone-support-expertise rows reclassified) through self-documenting test names, without repeating this ADR's reasoning inline.
- If the "Moved" or "Don't know" assumptions are ever revisited (for example, if a future iteration wants a three-class target: voluntary / involuntary / unknown), that would require a new ADR superseding this one, not a silent change to `sql/03_target/`.

---

**Decision Date:** 1 September 2026