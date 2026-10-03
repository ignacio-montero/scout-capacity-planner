"""Guardrail (D-002): keep the repo's vocabulary football-only.

One matcher, used by both enforcement points so they can never disagree:

* ``tests/test_guardrails.py`` scans file contents and paths, commit messages
  and author/committer names.
* ``.githooks/commit-msg`` runs ``python -m scout_planner.guardrail --commit-msg FILE``.

The vocabulary lives in the untracked ``.private/banned-terms.txt`` (one term
per line; ``#`` comments and blank lines ignored). A tracked list would itself
break the rule. Where the file is absent (fresh clone, CI) every check is a
no-op. Hits are always reported **masked** (first letter + asterisks) so that
test logs and hook output don't spread the vocabulary either.

Matching rules (case-insensitive), for a term ``T``:

* **Inflections**: ``T`` + an optional suffix from :data:`SUFFIXES` (plural,
  past, -ing, agent noun...). A trailing ``e`` may be dropped before the suffix
  (``make -> making``) and a trailing ``y`` may become ``i`` (``try -> tried``).
* **Common prefixes**: :data:`PREFIXES` (``un-``, ``re-``, ``pre-``...) may
  precede ``T``.
* **Word boundaries** are "not a letter", plus camelCase/acronym transitions,
  so ``T`` is found in ``t_log``, ``TLog``, ``logT``, ``HTTPT...`` alike.
  (A regex ``\\b`` would treat ``_`` as a letter and miss snake_case names.)

Deliberately *not* matched: ``T`` as an arbitrary prefix or substring of a
longer word. That rule sounds safer but flags innocent football writing: some
listed terms are the start or the tail of common, unrelated English words
(think of player trials, or praise for a player), and a check that cries wolf
gets switched off. Known prefixes and suffixes are the middle ground.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TERMS_FILE = REPO_ROOT / ".private" / "banned-terms.txt"
SKIP_FILES = frozenset({"uv.lock"})  # third-party package names and hashes, not our words
BINARY_SNIFF_BYTES = 8192

SUFFIXES = (
    "s", "es", "d", "ed", "ing", "ings", "er", "ers", "or", "ors",
    "al", "als", "able", "ment", "ments",
)  # fmt: skip
PREFIXES = ("un", "re", "pre", "dis", "non", "mis", "self", "co")


@dataclass(frozen=True, slots=True)
class Hit:
    """One match: where it is and the matched text (print it with :func:`mask`)."""

    where: str
    text: str

    def masked(self) -> str:
        """``"<where>: <masked text>"`` for messages."""
        return f"{self.where}: {mask(self.text)}"


# --- matching (pure) ---------------------------------------------------------------


def load_terms(path: Path = TERMS_FILE) -> list[str] | None:
    """Terms from the list file, or ``None`` if the file does not exist."""
    if not path.is_file():
        return None
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _term_regex(term: str) -> str:
    suffix = "|".join(SUFFIXES)
    t = re.escape(term.lower())
    forms = [rf"{t}(?:{suffix})?"]
    if term.lower().endswith("e"):
        forms.append(rf"{re.escape(term.lower()[:-1])}(?:{suffix})")
    if term.lower().endswith("y"):
        forms.append(rf"{re.escape(term.lower()[:-1])}i(?:{suffix})")
    return "|".join(forms)


def term_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    """Compile one matcher for all ``terms`` (see the module docstring for the rules)."""
    ordered = sorted({t.strip() for t in terms if t.strip()}, key=len, reverse=True)
    if not ordered:
        raise ValueError("no terms to match")
    words = "|".join(_term_regex(t) for t in ordered)
    prefixes = "|".join(PREFIXES)
    # Left edge: start of a word, a camelCase hump (aB), or the end of an acronym (ABc).
    left = r"(?:(?<![A-Za-z])|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))"
    # Right edge: not followed by a letter, or a camelCase hump after a lowercase letter.
    right = r"(?:(?![A-Za-z])|(?<=[a-z])(?=[A-Z]))"
    return re.compile(rf"{left}(?i:(?:(?:{prefixes})-?)?(?:{words})){right}")


def mask(text: str) -> str:
    """``"offside" -> "o******"``: enough to locate a hit without printing it."""
    return text[:1] + "*" * (len(text) - 1)


def find_hits(text: str, pattern: re.Pattern[str], where: str) -> list[Hit]:
    """Every match in ``text``, located as ``<where>:<line>``."""
    return [
        Hit(f"{where}:{lineno}", match.group(0))
        for lineno, line in enumerate(text.splitlines(), start=1)
        for match in pattern.finditer(line)
    ]


def check_commit_message(message: str, pattern: re.Pattern[str]) -> list[Hit]:
    """Hits in a commit message, ignoring ``#`` lines (git strips those)."""
    kept = "\n".join("" if line.startswith("#") else line for line in message.splitlines())
    return find_hits(kept, pattern, "commit message")


# --- repository scans (read git, no writes) -----------------------------------------


def _git(args: Sequence[str], root: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def repo_files(root: Path = REPO_ROOT) -> list[str]:
    """Tracked files plus untracked files that are not gitignored."""
    out = _git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], root)
    return sorted({p for p in out.decode("utf-8").split("\0") if p})


def scan_files(pattern: re.Pattern[str], root: Path = REPO_ROOT) -> list[Hit]:
    """Hits in every repo file's path and (text) contents."""
    hits: list[Hit] = []
    for rel in repo_files(root):
        hits += [Hit(f"{rel} (file path)", h.text) for h in find_hits(rel, pattern, rel)]
        path = root / rel
        if rel in SKIP_FILES or not path.is_file():  # deleted-but-tracked files too
            continue
        data = path.read_bytes()
        if b"\0" in data[:BINARY_SNIFF_BYTES]:
            continue
        hits += find_hits(data.decode("utf-8", errors="replace"), pattern, rel)
    return hits


def scan_history(pattern: re.Pattern[str], root: Path = REPO_ROOT) -> list[Hit]:
    """Hits in every commit's message and author/committer name and email."""
    fields = ("author", "author email", "committer", "committer email", "message")
    fmt = "%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e"
    try:
        out = _git(["log", "--all", f"--format={fmt}"], root).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:  # no commits yet
        return []
    hits: list[Hit] = []
    for record in out.split("\x1e"):
        parts = record.strip("\n").split("\x1f")
        if len(parts) != len(fields) + 1:
            continue
        sha, *values = parts
        for name, value in zip(fields, values, strict=True):
            hits += find_hits(value, pattern, f"commit {sha[:7]} {name}")
    return hits


# --- command line (used by the commit-msg hook) ----------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``--commit-msg FILE`` checks one message; no argument scans files and history."""
    parser = argparse.ArgumentParser(prog="python -m scout_planner.guardrail")
    parser.add_argument("--commit-msg", metavar="FILE", help="check one commit message file")
    args = parser.parse_args(argv)

    terms = load_terms(TERMS_FILE)  # looked up at call time (tests point it elsewhere)
    if not terms:
        return 0  # no private list here (fresh clone, CI): nothing to enforce
    pattern = term_pattern(terms)

    if args.commit_msg:
        message = Path(args.commit_msg).read_text(encoding="utf-8")
        hits = check_commit_message(message, pattern)
    else:
        hits = scan_files(pattern) + scan_history(pattern)

    for hit in hits:
        print(f"guardrail (D-002): {hit.masked()}", file=sys.stderr)
    if hits and args.commit_msg:
        print("guardrail: reword the message; see .private/banned-terms.txt", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
