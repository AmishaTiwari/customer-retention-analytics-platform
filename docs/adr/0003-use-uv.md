# ADR 0003: Use uv for Environment and Dependency Management

**Status:** Accepted

---

## Context

The project requires a standard way to create the development environment and install dependencies.

While `pyproject.toml` defines **what** should be installed, it does not specify **which tool** should be used to perform the installation.

Before implementation began, we needed to choose a single environment and dependency management tool for the project.

---

## Decision

This project will use **uv** for environment and dependency management.

The standard setup command is:

```bash
uv sync --extra dev
```

This command:

- Creates the project's virtual environment (`.venv/`) if it does not already exist.
- Installs the project package in editable mode.
- Installs all project dependencies.
- Installs all development dependencies.
- Generates a `uv.lock` file.

---

## Rationale

This decision was made because:

- **Simple setup.** A single command prepares the complete development environment.
- **Fast dependency installation.** `uv` is significantly faster than the traditional `venv` + `pip` workflow.
- **Improved reproducibility.** Along with `pyproject.toml`, `uv.lock` records the exact dependency versions installed, ensuring that everyone working on the project uses the same environment.

---

## Consequences

- `uv` becomes the standard tool for setting up this repository.
- `pyproject.toml` and `uv.lock` are committed to the repository.
- `.venv/` is **not** committed because it is machine-specific and can be recreated at any time using `uv sync --extra dev`.
- The Makefile's `install` target uses:

```make
uv sync --extra dev
```

- Anyone cloning the repository should install `uv` first and then run:

```bash
uv sync --extra dev
```

to create the development environment.

---

**Decision Date:** 16 July 2026