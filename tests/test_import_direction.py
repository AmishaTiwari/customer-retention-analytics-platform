"""Import-direction rules for the retention_platform package.

Every module gets a row in IMPORT_RULES listing which parts of the package
it may import. Stage order is data -> features -> pipeline -> models ->
evaluation -> inference; config and logging_setup are the shared root
modules. metrics is a shared module used by models, evaluation and
inference, and tracking is a shared module used by models and inference.
A module may import the root modules, earlier stages and its own stage,
never later stages.

The checks only read source text with ast. No package code is imported or
run; importlib.util.find_spec is used just to locate the package folder.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
from pathlib import Path, PurePosixPath

import pytest

PACKAGE = "retention_platform"

# File selector (path relative to the package folder, with / separators;
# * is a wildcard) -> dotted import prefixes, relative to the package, that
# the file may import. "models.persist" allows that module and anything
# under it. An exact file row wins over a wildcard row.
IMPORT_RULES: dict[str, set[str]] = {
    "__init__.py": set(),
    "config.py": set(),
    "logging_setup.py": set(),
    "metrics.py": set(),
    "models/calibrate.py": set(),
    "tracking.py": {"config", "logging_setup"},
    "models/*": {"config", "logging_setup", "pipeline", "metrics", "tracking", "models"},
    "data/*": {"config", "logging_setup", "data"},
    "features/*": {"config", "logging_setup", "data", "features"},
    "pipeline/*": {"config", "logging_setup", "data", "features", "pipeline"},
    "evaluation/*": {"config", "metrics", "models", "pipeline", "evaluation"},
    "inference/*": {"config", "metrics", "tracking", "models.persist", "inference"},
}


def _is_wildcard(selector: str) -> bool:
    return "*" in selector


def collect_package_imports(
    source: str, rel: str, package_root: Path | None = None
) -> list[tuple[int, str]]:
    """Return (line, dotted name relative to the package) for every import of
    the package in source, wherever it appears (module level, inside a
    function, under TYPE_CHECKING). Relative imports are resolved against
    rel, the file's path inside the package. Standard-library and
    third-party imports are ignored.

    `from retention_platform.pkg import name` is recorded as pkg.name when
    package_root is given and pkg/name is a module there, otherwise as pkg.
    """
    tree = ast.parse(source)
    package_parts = list(PurePosixPath(rel).parts[:-1])
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE and len(parts) > 1:
                    found.append((node.lineno, ".".join(parts[1:])))
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names if alias.name != "*"]
            if node.level == 0:
                if not node.module:
                    continue
                parts = node.module.split(".")
                if parts[0] != PACKAGE:
                    continue
                base = parts[1:]
                names_are_modules = False
            else:
                if node.level - 1 > len(package_parts):
                    continue
                anchor = package_parts[: len(package_parts) - (node.level - 1)]
                base = anchor + (node.module.split(".") if node.module else [])
                names_are_modules = node.module is None

            if not base:
                found.extend((node.lineno, name) for name in names)
            elif names_are_modules:
                found.extend((node.lineno, ".".join(base + [name])) for name in names)
            else:
                targets = {".".join(base)}
                if package_root is not None:
                    targets = set()
                    for name in names:
                        folder = package_root.joinpath(*base)
                        if (folder / f"{name}.py").is_file() or (folder / name).is_dir():
                            targets.add(".".join(base + [name]))
                        else:
                            targets.add(".".join(base))
                found.extend((node.lineno, target) for target in sorted(targets))

    return sorted(found)


def is_allowed(dotted: str, allowed: set[str]) -> bool:
    return any(dotted == prefix or dotted.startswith(prefix + ".") for prefix in allowed)


def row_for(rel: str, rules: dict[str, set[str]]) -> str | None:
    """Return the selector governing rel: an exact row first, then a wildcard row."""
    if rel in rules and not _is_wildcard(rel):
        return rel
    for selector in rules:
        if _is_wildcard(selector) and fnmatch.fnmatchcase(rel, selector):
            return selector
    return None


def list_package_files(package_root: Path) -> list[str]:
    return sorted(
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def violations_in_source(
    source: str,
    rel: str,
    allowed: set[str],
    package_root: Path | None = None,
) -> list[tuple[int, str]]:
    return [
        (line, dotted)
        for line, dotted in collect_package_imports(source, rel, package_root)
        if not is_allowed(dotted, allowed)
    ]


def find_violations(
    package_root: Path, rules: dict[str, set[str]]
) -> list[tuple[str, int, str, set[str]]]:
    """Return (file, line, import, allowed set) for every disallowed import
    in files that have a row."""
    problems = []
    for rel in list_package_files(package_root):
        selector = row_for(rel, rules)
        if selector is None:
            continue
        source = (package_root / rel).read_text(encoding="utf-8")
        for line, dotted in violations_in_source(
            source, rel, rules[selector], package_root
        ):
            problems.append((rel, line, dotted, rules[selector]))
    return problems


def files_without_row(package_root: Path, rules: dict[str, set[str]]) -> list[str]:
    return [rel for rel in list_package_files(package_root) if row_for(rel, rules) is None]


def rows_without_file(package_root: Path, rules: dict[str, set[str]]) -> list[str]:
    files = list_package_files(package_root)
    return [
        selector
        for selector in rules
        if not any(row_for(rel, rules) == selector for rel in files)
    ]


def _describe(rel: str, line: int, dotted: str, allowed: set[str]) -> str:
    allowed_text = ", ".join(sorted(allowed)) if allowed else "nothing"
    return (
        f"{rel}:{line} imports {PACKAGE}.{dotted}, "
        f"but this file may only import: {allowed_text}"
    )


def _locate_package_root() -> Path:
    spec = importlib.util.find_spec(PACKAGE)
    return Path(list(spec.submodule_search_locations)[0])


PACKAGE_ROOT = _locate_package_root()
ALL_FILES = list_package_files(PACKAGE_ROOT)


@pytest.mark.parametrize("rel", ALL_FILES)
def test_module_imports_respect_direction(rel):
    selector = row_for(rel, IMPORT_RULES)
    if selector is None:
        pytest.skip("no row in IMPORT_RULES; test_every_file_has_a_row reports it")

    source = (PACKAGE_ROOT / rel).read_text(encoding="utf-8")
    bad = violations_in_source(source, rel, IMPORT_RULES[selector], PACKAGE_ROOT)

    assert not bad, "\n".join(
        _describe(rel, line, dotted, IMPORT_RULES[selector]) for line, dotted in bad
    )


def test_every_file_has_a_row():
    missing = files_without_row(PACKAGE_ROOT, IMPORT_RULES)
    assert not missing, (
        "These files have no row in IMPORT_RULES; add one: " + ", ".join(missing)
    )


def test_every_row_matches_a_file():
    stale = rows_without_file(PACKAGE_ROOT, IMPORT_RULES)
    assert not stale, (
        "These IMPORT_RULES rows match no file; remove or fix them: " + ", ".join(stale)
    )


# --- Checker self-tests: small sources and a fake package -----------------


def test_catches_forbidden_absolute_import():
    source = "from retention_platform.evaluation.compare import evaluate_candidate\n"
    assert violations_in_source(source, "models/x.py", {"config", "models"}) == [
        (1, "evaluation.compare")
    ]


def test_catches_plain_import_statement():
    source = "import retention_platform.inference.score\n"
    assert violations_in_source(source, "models/x.py", {"models"}) == [
        (1, "inference.score")
    ]


def test_catches_from_package_import_stage():
    source = "from retention_platform import evaluation\n"
    assert violations_in_source(source, "models/x.py", {"models"}) == [(1, "evaluation")]


def test_catches_relative_import_of_later_stage():
    source = "from ..evaluation import compare\n"
    assert violations_in_source(source, "models/x.py", {"models"}) == [(1, "evaluation")]


def test_resolves_sibling_relative_import():
    source = "from . import candidates\n"
    assert collect_package_imports(source, "models/x.py") == [(1, "models.candidates")]


def test_catches_import_inside_function():
    source = (
        "def f():\n"
        "    from retention_platform.inference import score\n"
        "    return score\n"
    )
    assert violations_in_source(source, "evaluation/x.py", {"models"}) == [
        (2, "inference")
    ]


def test_catches_import_under_type_checking():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from retention_platform.evaluation.compare import EvaluationResult\n"
    )
    assert violations_in_source(source, "models/x.py", {"models"}) == [
        (3, "evaluation.compare")
    ]


def test_clean_file_and_non_package_imports_pass():
    source = (
        "import os\n"
        "import numpy as np\n"
        "from sklearn.model_selection import cross_val_predict\n"
        "from retention_platform.models.candidates import predict_proba\n"
    )
    assert violations_in_source(source, "evaluation/x.py", {"models"}) == []


def test_prefix_matching_is_by_module_boundary():
    allowed = {"models.persist"}
    assert is_allowed("models.persist", allowed)
    assert is_allowed("models.persist.save", allowed)
    assert not is_allowed("models.persist_extra", allowed)
    assert not is_allowed("models", allowed)


def test_exact_row_wins_over_wildcard_row():
    rules = {"models/*": {"models"}, "models/calibrate.py": set()}
    assert row_for("models/calibrate.py", rules) == "models/calibrate.py"
    assert row_for("models/tune.py", rules) == "models/*"
    assert row_for("data/ingest.py", rules) is None


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_fake_package_violations_missing_rows_and_stale_rows(tmp_path):
    _write(tmp_path, "config.py", "")
    _write(tmp_path, "models/__init__.py", "")
    _write(tmp_path, "models/a.py", "from retention_platform.evaluation import compare\n")
    _write(tmp_path, "models/b.py", "from . import a\nimport os\n")
    _write(
        tmp_path,
        "evaluation/c.py",
        "def f():\n    from retention_platform.inference import score\n",
    )
    _write(tmp_path, "evaluation/d.py", "from retention_platform.models.a import x\n")
    _write(tmp_path, "extra.py", "")
    rules = {
        "config.py": set(),
        "models/*": {"config", "models"},
        "evaluation/*": {"models"},
        "ghost/*": set(),
    }

    violations = [(rel, dotted) for rel, _, dotted, _ in find_violations(tmp_path, rules)]

    assert violations == [("evaluation/c.py", "inference"), ("models/a.py", "evaluation")]
    assert files_without_row(tmp_path, rules) == ["extra.py"]
    assert rows_without_file(tmp_path, rules) == ["ghost/*"]


def test_from_import_of_submodule_uses_the_submodule_name(tmp_path):
    _write(tmp_path, "models/persist.py", "")
    _write(tmp_path, "models/tune.py", "")
    source = "from retention_platform.models import persist, tune\n"

    assert collect_package_imports(source, "inference/x.py", tmp_path) == [
        (1, "models.persist"),
        (1, "models.tune"),
    ]
    assert violations_in_source(
        source, "inference/x.py", {"models.persist"}, tmp_path
    ) == [(1, "models.tune")]
