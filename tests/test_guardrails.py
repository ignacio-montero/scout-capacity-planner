"""Guardrail (D-002): no term from the private list in files, paths or history.

The matcher lives in ``scout_planner.guardrail`` and is shared with the
``commit-msg`` hook. Its rules are unit-tested here with a harmless example
term ("offside"), so they run everywhere; the scans of this repository need the
untracked ``.private/banned-terms.txt`` and skip with a message without it.
Every failure message masks the matched text.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scout_planner import guardrail as g

HOOK = g.REPO_ROOT / ".githooks" / "commit-msg"

# --- matcher rules (always run) ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "offside",
        "OFFSIDE trap",
        "an Offside call",
        "offside_rule",  # snake_case
        "OffsideRule",  # camelCase
        "isOffside",
        "VAROffside",  # acronym then word
        "offsides",  # inflections
        "offsided",
        "offsiding",  # trailing e dropped
        "offsider",
        "offside's",
        "unoffside",  # known prefixes
        "re-offside",
        "nonOffside",
    ],
)
def test_pattern_finds_term_and_its_variants(text: str) -> None:
    assert g.term_pattern(["offside"]).search(text), text


@pytest.mark.parametrize(
    "text",
    [
        "onside",
        "off side",
        "offsidex",  # unknown suffix: a different word
        "xoffside",  # unknown prefix: a different word
        "OFFSIDEX",
        "goaloffside",
    ],
)
def test_pattern_ignores_different_words(text: str) -> None:
    assert not g.term_pattern(["offside"]).search(text), text


def test_trailing_y_inflections() -> None:
    pattern = g.term_pattern(["tally"])
    assert all(pattern.search(w) for w in ["tallies", "tallied", "tallying", "Tallier"])
    assert not pattern.search("tallyho")


def test_mask_hides_all_but_first_letter() -> None:
    assert g.mask("offside") == "o******"
    assert g.Hit("f.py:3", "Offsides").masked() == "f.py:3: O*******"


def test_commit_message_check_ignores_comment_lines() -> None:
    pattern = g.term_pattern(["offside"])
    assert not g.check_commit_message("feat: add\n# offside in a comment\n", pattern)
    hits = g.check_commit_message("feat: add\n\nfixes Offside bug\n", pattern)
    assert [h.where for h in hits] == ["commit message:3"]


def test_missing_term_file_means_no_terms(tmp_path: Path) -> None:
    assert g.load_terms(tmp_path / "absent.txt") is None


def test_term_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    f = tmp_path / "terms.txt"
    f.write_text("# header\n\noffside\n  corner  \n", encoding="utf-8")
    assert g.load_terms(f) == ["offside", "corner"]


# --- this repository (needs the private list) ---------------------------------------


@pytest.fixture(scope="module")
def terms() -> list[str]:
    found = g.load_terms()
    if found is None:
        pytest.skip("guardrail term list .private/banned-terms.txt not present; check skipped")
    if not found:
        pytest.skip("guardrail term list is empty; check skipped")
    return found


def _report(hits: list[g.Hit]) -> str:
    return "guardrail terms found (D-002):\n" + "\n".join(h.masked() for h in hits)


def test_repo_files_contain_no_banned_terms(terms: list[str]) -> None:
    hits = g.scan_files(g.term_pattern(terms))
    assert not hits, _report(hits)


def test_commit_history_contains_no_banned_terms(terms: list[str]) -> None:
    """Commit messages and author/committer names of every commit."""
    hits = g.scan_history(g.term_pattern(terms))
    assert not hits, _report(hits)


def test_real_terms_are_caught_in_variant_forms(terms: list[str]) -> None:
    pattern = g.term_pattern(terms)
    misses = [
        g.mask(form)
        for t in terms
        for form in (t.upper(), f"{t}s", f"un{t}", f"my_{t}_log", f"get{t.capitalize()}Id")
        if not pattern.search(form)
    ]
    assert not misses, f"variants not caught: {misses}"


def test_real_terms_do_not_match_inside_other_words(terms: list[str]) -> None:
    pattern = g.term_pattern(terms)
    false_alarms = [
        g.mask(form) for t in terms for form in (f"x{t}", f"{t}zq") if pattern.search(form)
    ]
    assert not false_alarms, f"false positives: {false_alarms}"


# --- the commit-msg hook -----------------------------------------------------------


def test_commit_msg_hook_is_executable() -> None:
    assert HOOK.is_file()
    assert os.access(HOOK, os.X_OK), "run: chmod +x .githooks/commit-msg"


def _run_hook(message: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8")
    return subprocess.run(
        ["sh", str(HOOK), str(msg_file)], cwd=g.REPO_ROOT, capture_output=True, text=True
    )


def test_commit_msg_hook_rejects_inflected_term(terms: list[str], tmp_path: Path) -> None:
    form = f"{terms[0].capitalize()}s"
    result = _run_hook(f"feat: add the {form} step\n", tmp_path)
    assert result.returncode == 1, result.stderr
    assert g.mask(form) in result.stderr
    assert terms[0].lower() not in result.stderr.lower(), "hook output must mask the term"


def test_commit_msg_hook_accepts_clean_message(terms: list[str], tmp_path: Path) -> None:
    result = _run_hook("feat: add config models\n\n# a comment line\n", tmp_path)
    assert result.returncode == 0, result.stderr


def test_cli_is_a_no_op_without_the_private_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(g, "TERMS_FILE", tmp_path / "absent.txt")
    msg = tmp_path / "msg"
    msg.write_text("anything at all\n", encoding="utf-8")
    assert g.main(["--commit-msg", str(msg)]) == 0


def test_cli_rejects_message_with_masked_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("offside\n", encoding="utf-8")
    monkeypatch.setattr(g, "TERMS_FILE", terms_file)
    msg = tmp_path / "msg"
    msg.write_text("fix: Offsides rule\n", encoding="utf-8")
    assert g.main(["--commit-msg", str(msg)]) == 1
    err = capsys.readouterr().err
    assert "O*******" in err and "offside" not in err.lower()
