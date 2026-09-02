# Engineering Conventions for Claude Code

## Comments in configuration and source files

Comments in `.gitignore`, `pyproject.toml`, YAML files, Makefiles, shell scripts, and code docstrings explain **local mechanics only**:

- What is this rule/function/file doing?
- Why does this specific exception exist?
- What would break if this were removed?

Comments and docstrings must NOT reference:

- ADR numbers or ADR content
- Licensing rationale
- Repository-wide architectural philosophy or trade-offs
- Implementation workflow narrative ("this was decided because...")
- What other files/scripts do

That reasoning belongs in ADRs (`docs/adr/`), the README, or design documentation -- never duplicated inline. If a comment needs an ADR reference to make sense, the comment is doing too much; simplify it instead of citing the source.

**Test:** a comment should read as something a developer wrote while working on *this file*, not as an explanation of a decision made in a planning meeting.

## Markdown formatting

Write every paragraph and bullet point in generated Markdown as a single unbroken line -- no internal hard line breaks within a sentence or bullet. Do not hard-wrap prose at any fixed character width.

This applies to all generated Markdown in this repository: ADRs, notebook markdown cells, README updates, and any documentation files. Rely on the reader's own editor or viewer to soft-wrap long lines for display; do not insert real line breaks into the file itself.

**Test:** running `git diff` on an edited paragraph should show one changed line, not several -- if editing one sentence changes many lines, the paragraph was hard-wrapped and needs to be joined back into a single line first.

## Environment and command execution

All commands that run project code -- Python scripts, tests, the pipeline, package installs -- must go through `uv run ...` or `uv sync`. Never invoke a Python interpreter by its direct path (conda, global, or otherwise), and never `pip install` into any environment other than this project's own `.venv`, managed entirely by `uv`.

If a package appears to be missing:

1. Check whether it is already declared in `pyproject.toml`.
2. If it is, run `uv sync --extra dev` to install it into `.venv` correctly.
3. If it genuinely is not declared yet, stop and ask before installing anything, rather than installing it into whatever Python happens to be active in the current shell.

A `VIRTUAL_ENV does not match the project environment path` warning from `uv run` is expected and harmless if it appears -- it means `uv` correctly ignored a stray environment variable from outside this project and used `.venv` anyway. It is not a sign the command ran against the wrong environment.