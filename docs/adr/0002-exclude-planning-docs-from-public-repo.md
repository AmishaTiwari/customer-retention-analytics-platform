# ADR 0002: Keep Planning Documents Out of the Public Repository

**Status:** Accepted

---

## Context

The locked planning documents (Repository Architecture v1.0, Documentation Architecture v1.0) originally called for copying the five locked design documents into `docs/design/` inside this public GitHub repository.

Since locking those documents, we decided to keep planning and implementation in two separate places: a private planning repository (where the design docs already live) and this public repository (the actual codebase). The design docs are also uploaded directly to the Claude Project used for implementation, so they stay available as context without needing to live inside the public repo.

This changes a decision that was explicitly locked during planning, so it is documented through an ADR rather than being changed silently.

---

## Decision

`docs/design/` will **not** be created in this repository, and the planning documents will **not** be copied into it.

The public repository will only contain implementation-facing documentation:

- ADRs (`docs/adr/`)
- README
- Data Dictionary & Leakage Audit
- Platform Output Contract

The planning documents remain in the private planning repository and are maintained there.

---

## Rationale

- **Two different audiences.** The planning documents record *how the design was decided*. The public repo should show *how the system was built*. Mixing both makes the repo harder to read for someone just evaluating the engineering.
- **Avoids duplication drift.** If the same documents lived in two repos, one of them would eventually go stale. Keeping one copy in one place removes that risk entirely.
- **Nothing is lost.** The planning documents aren't discarded — they still exist, are still locked, and are still available as context. They're just not duplicated into the public repo.

---

## Consequences

- Repository Architecture v1.0 §3 and §9, and Documentation Architecture v1.0 §5, are superseded on this one point by this ADR. The rest of those documents still apply as written.
- Anyone reviewing the public repository won't see the full planning history there. The public repository instead focuses on the implemented system and the engineering artifacts needed to understand and evaluate it. The README should mention that planning documentation exists separately, so this isn't mistaken for the project having no design phase.
- If we ever want the planning docs visible in the public repo (e.g. before a job application review), that would be a new decision, not a revert of this one.

---

**Decision Date:** 16 July 2026