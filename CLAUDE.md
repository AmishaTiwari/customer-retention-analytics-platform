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