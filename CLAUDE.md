# Engineering Conventions for Claude Code

## Comments in configuration and source files

Comments in `.gitignore`, `pyproject.toml`, YAML files, Makefiles, shell
scripts, and code docstrings explain **local mechanics only**:

- What is this rule/function/file doing?
- Why does this specific exception exist?
- What would break if this were removed?

Comments and docstrings must NOT reference:

- ADR numbers or ADR content
- Licensing rationale
- Repository-wide architectural philosophy or trade-offs
- Implementation workflow narrative ("this was decided because...")
- What other files/scripts do

That reasoning belongs in ADRs (`docs/adr/`), the README, or design
documentation -- never duplicated inline. If a comment needs an ADR
reference to make sense, the comment is doing too much; simplify it
instead of citing the source.

**Test:** a comment should read as something a developer wrote while
working on *this file*, not as an explanation of a decision made in
a planning meeting.