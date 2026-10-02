"""Guardrail (D-002): the repo must not contain any term from the private list.

The list lives in the untracked ``.private/banned-terms.txt`` (a tracked list
would itself break the rule). Where it is missing (fresh clone, CI) the
repo-scanning tests skip with a message; the matcher's own unit tests always run.

Matched terms are printed masked (first letter + asterisks) so that a failing
test's output does not itself spread the vocabulary into logs.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMS_FILE = REPO_ROOT / ".private" / "banned-terms.txt"
HOOK = REPO_ROOT / ".githooks" / "commit-msg"
SKIP_FILES = {"uv.lock"}  # third-party package names and hashes, not our words
BINARY_SNIFF_BYTES = 8192


# --- helpers (pure) --------------------------------------------------------------


def load_terms(path: Path) -> list[str]:
    """Terms from the list file: one per line, '#' comments and blank lines ignored."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Case-insensitive whole-word matcher that also sees words inside identifiers.

    A plain regex ``\\b`` treats ``_`` as part of a word, so ``foo_bar`` would hide
    ``foo``. Here a word boundary is "not a letter" on either side, plus the
    camelCase transition (lowercase followed by uppercase), so the term is
    found in ``term_log``, ``TermLog``, ``logTerm`` and ``TERM`` alike, but
    not inside a longer word such as ``terminal``.
    """
    words = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    left = r"(?:(?<![A-Za-z])|(?<=[a-z])(?=[A-Z]))"
    return re.compile(rf"{left}(?i:{words})(?![a-z])")


def mask(term: str) -> str:
    """``"offside" -> "o******"``: enough to locate it, without printing it."""
    return term[:1] + "*" * (len(term) - 1)


def find_hits(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """``(line_number, matched_text)`` for every match; line numbers start at 1."""
    return [
        (lineno, match.group(0))
        for lineno, line in enumerate(text.splitlines(), start=1)
        for match in pattern.finditer(line)
    ]


def repo_files() -> list[str]:
    """Tracked files plus untracked files that are not gitignored."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return sorted({p for p in out.decode("utf-8").split("\0") if p})


# --- matcher unit tests (always run; use a harmless example term) ----------------


@pytest.mark.parametrize(
    "text",
    ["offside", "OFFSIDE trap", "an Offside call", "offside_rule", "OffsideRule", "isOffside"],
)
def test_pattern_finds_term_in_words_and_identifiers(text: str) -> None:
    assert term_pattern(["offside"]).search(text)


@pytest.mark.parametrize("text", ["offsides", "nonoffside", "onside", "off side"])
def test_pattern_ignores_longer_words_and_near_misses(text: str) -> None:
    assert not term_pattern(["offside"]).search(text)


def test_mask_hides_all_but_first_letter() -> None:
    assert mask("offside") == "o******"


# --- repository scan and commit hook (need the private list) ---------------------


@pytest.fixture(scope="module")
def terms() -> list[str]:
    if not TERMS_FILE.is_file():
        pytest.skip("guardrail term list .private/banned-terms.txt not present; check skipped")
    found = load_terms(TERMS_FILE)
    if not found:
        pytest.skip("guardrail term list is empty; check skipped")
    return found


def test_repo_contains_no_banned_terms(terms: list[str]) -> None:
    pattern = term_pattern(terms)
    problems: list[str] = []
    for rel in repo_files():
        problems += [f"{rel}: (in file path) {mask(m)}" for _, m in find_hits(rel, pattern)]
        path = REPO_ROOT / rel
        if rel in SKIP_FILES or not path.is_file():  # deleted-but-tracked files too
            continue
        data = path.read_bytes()
        if b"\0" in data[:BINARY_SNIFF_BYTES]:
            continue
        text = data.decode("utf-8", errors="replace")
        problems += [f"{rel}:{n}: {mask(m)}" for n, m in find_hits(text, pattern)]
    assert not problems, "guardrail terms found (D-002):\n" + "\n".join(problems)


def test_commit_msg_hook_is_executable() -> None:
    assert HOOK.is_file()
    assert os.access(HOOK, os.X_OK), "run: chmod +x .githooks/commit-msg"


def _run_hook(message: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8")
    return subprocess.run(
        ["sh", str(HOOK), str(msg_file)], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_commit_msg_hook_rejects_banned_term(terms: list[str], tmp_path: Path) -> None:
    term = terms[0]
    result = _run_hook(f"feat: add the {term.capitalize()} step\n", tmp_path)
    assert result.returncode == 1
    assert mask(term) in result.stderr
    assert term.lower() not in result.stderr.lower(), "hook output must mask the term"


def test_commit_msg_hook_accepts_clean_message(terms: list[str], tmp_path: Path) -> None:
    result = _run_hook("feat: add config models\n\n# a comment line\n", tmp_path)
    assert result.returncode == 0, result.stderr
