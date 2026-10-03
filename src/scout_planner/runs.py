"""Run store: every file the app, the worker and the CLI share about runs.

A *run* is one folder under ``data/runs/`` (DATA_CONTRACTS.md section 0). The
folder **is** the job queue (D-013): the app creates a folder in state
``queued``, the worker picks the oldest one and moves it through the state
machine below. All run file I/O lives in this module, so the pipeline stays
pure and the app never touches ``status.json`` directly.

State machine (enforced by :func:`update_status`)::

    queued ──► running ──► done
       │          ├──────► failed
       └──────────┴──────► cancelled          (done / failed / cancelled are final)

Two more *display* states are derived, never stored (DESIGN_SYSTEM.md):
``cancelling`` (queued/running with a ``cancel`` flag file) and ``unreadable``
(``status.json`` missing or corrupt).

Concurrency rules this module relies on:

* **Atomic writes.** ``status.json`` and ``_worker.json`` are written to a temp
  file in the same folder and swapped in with ``os.replace``, so a reader sees
  either the old file or the new one, never half of one.
* **One writer per status file at a time.** The app writes ``status.json`` only
  when it creates the run; afterwards only the worker (state changes) or the
  run's child process (progress, final state) writes it, never both at once.
  That is why a read-modify-write without a lock is safe here.
* **Publish by rename.** A new run is assembled in a hidden staging folder and
  renamed into place, so the worker never sees a run folder without its files.

Timestamps are naive local datetimes in ISO format with microseconds; the
microseconds make "oldest first" exact even for runs created in the same second.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scout_planner.config import Params, load_params, params_from_yaml, params_to_yaml
from scout_planner.errors import IllegalTransition, RunCancelled, RunNotFound, RunStoreError

__all__ = [
    "DEFAULT_ROOT",
    "DISPLAY_STATES",
    "STAGES",
    "Heartbeat",
    "IllegalTransition",
    "RunCancelled",
    "RunNotFound",
    "RunState",
    "RunStatus",
    "RunStoreError",
    "append_log",
    "cancel_requested",
    "cleanup_stale_temp",
    "create_run",
    "delete_run",
    "display_state",
    "list_runs",
    "make_run_id",
    "queued_runs",
    "read_heartbeat",
    "read_log_tail",
    "read_params",
    "read_status",
    "remove_heartbeat",
    "request_cancel",
    "run_dir",
    "run_is_live",
    "slugify",
    "update_status",
    "worker_alive",
    "write_heartbeat",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "runs"

# File names inside a run folder (DATA_CONTRACTS.md section 0).
PARAMS_FILE = "params.yaml"
STATUS_FILE = "status.json"
CANCEL_FILE = "cancel"
LOG_FILE = "log.txt"

# Files in the runs root that are not runs.
HEARTBEAT_FILE = "_worker.json"
LOCK_FILE = "_worker.lock"
DEV_RUN = "dev"  # fixed folder used by the stage-by-stage make targets
STAGING_PREFIX = "_new-"  # a run being assembled by create_run
TRASH_PREFIX = "_trash-"  # a run being removed by delete_run

# Lifecycle states stored in status.json, plus the two derived display states.
RunState = Literal["queued", "running", "done", "failed", "cancelled"]
StatusState = Literal["queued", "running", "done", "failed", "cancelled", "unreadable"]
DisplayState = Literal[
    "queued", "running", "cancelling", "done", "failed", "cancelled", "unreadable"
]
DISPLAY_STATES: tuple[str, ...] = get_args(DisplayState)
TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

# Pipeline stage names (status.stage). Not enforced: a new stage label from the
# pipeline should show up in the UI, not fail the run.
STAGES: tuple[str, ...] = ("generate", "forecast", "capacity_plan", "simulate")

# The state machine as data: from-state -> states it may move to. A state may
# also "move" to itself (a progress update while running), except final states.
_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"done", "failed", "cancelled"}),
    "done": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}
# Fields a status update may change. run_id, name, created_at, sweep_id and
# code_version describe the run and are written once.
_MUTABLE_FIELDS = frozenset(
    {"state", "progress", "stage", "message", "error", "started_at", "finished_at"}
)

# A run id is used as a folder name and comes back from URLs (?run=<id>), so it
# is checked before it touches the filesystem: no "/", no "..", no leading "_" or ".".
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,99}$")
_SLUG_MAX = 40


# --- models ------------------------------------------------------------------------


class RunStatus(BaseModel):
    """The content of ``status.json``. Immutable; updates build a new one."""

    # extra="ignore": a field added by a later version doesn't make old code
    # call the whole file unreadable.
    model_config = ConfigDict(frozen=True, extra="ignore")

    run_id: str
    name: str
    state: StatusState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float = Field(0.0, ge=0.0, le=1.0)
    stage: str | None = None
    message: str | None = None
    error: str | None = None
    sweep_id: str | None = None
    code_version: str = "unknown"

    @property
    def is_terminal(self) -> bool:
        """True for done / failed / cancelled: nothing will change any more."""
        return self.state in TERMINAL_STATES

    @property
    def duration_s(self) -> float | None:
        """Seconds from start to finish (or to now while running); None if not started."""
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now()
        return max(0.0, (end - self.started_at).total_seconds())


class Heartbeat(BaseModel):
    """The content of ``_worker.json``: proof of life from the worker."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    pid: int
    heartbeat_at: datetime
    current_run: str | None = None


# --- small pure helpers ------------------------------------------------------------


def slugify(name: str) -> str:
    """``"Hire for 2x, get 4x!"`` -> ``"hire-for-2x-get-4x"`` (``"run"`` if nothing is left)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug[:_SLUG_MAX].rstrip("-")
    return slug or "run"


def make_run_id(name: str, when: datetime) -> str:
    """``YYYYMMDD-HHMMSS-<slug>``; :func:`create_run` adds ``-2``, ``-3``... on collision."""
    return f"{when:%Y%m%d-%H%M%S}-{slugify(name)}"


def display_state(status: RunStatus, cancel_requested: bool = False) -> DisplayState:
    """The badge the UI shows (DESIGN_SYSTEM.md, status badges).

    ``cancelling`` is derived: the run is still queued or running but the user
    asked to stop it. Everything else is the stored state as is (``unreadable``
    already comes from :func:`read_status`).
    """
    if cancel_requested and status.state in ("queued", "running"):
        return "cancelling"
    return status.state


def _now() -> datetime:
    return datetime.now()


def _check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise RunNotFound(f"not a valid run id: {run_id!r}")
    return run_id


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else DEFAULT_ROOT


def run_dir(run_id: str, root: Path | str | None = None) -> Path:
    """Path of a run folder (validated id; the folder may not exist)."""
    return _root(root) / _check_run_id(run_id)


def _existing_run_dir(run_id: str, root: Path | str | None) -> Path:
    path = run_dir(run_id, root)
    if not path.is_dir():
        raise RunNotFound(f"no run {run_id!r} in {_root(root)}")
    return path


def _code_version() -> str:
    """Short git SHA of the checkout, or ``"unknown"`` (no git, not a repo, timeout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


# --- atomic file writes ------------------------------------------------------------


def _fsync_dir(folder: Path) -> None:
    """Make a rename inside ``folder`` durable (the folder's entry list is data too)."""
    with contextlib.suppress(OSError):
        dir_fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that readers never see a partial file.

    The temp file lives in the same folder because ``os.replace`` is only atomic
    within one filesystem. Its name starts with ``.`` so listings skip it.
    ``fsync`` on the file before the swap and on the folder after it makes the
    new content survive an OS crash, not just a process crash.

    Not done: ``F_FULLFSYNC`` on macOS (plain ``fsync`` there may leave data in
    the drive's cache). It costs tens of ms per call and these files are
    rewritten up to ~3x/s; the worst a power cut can do is roll a status back
    or leave it unreadable, which the readers already handle.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)  # mkstemp creates 0600; these files are meant to be readable
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
    _fsync_dir(path.parent)


def _write_status(folder: Path, status: RunStatus) -> None:
    _atomic_write_text(folder / STATUS_FILE, status.model_dump_json(indent=2) + "\n")


# --- creating runs -----------------------------------------------------------------


def create_run(
    params: Params,
    name: str,
    sweep_id: str | None = None,
    *,
    root: Path | str | None = None,
    now: datetime | None = None,
) -> str:
    """Queue a new run and return its id. The worker picks it up from here.

    Writes ``params.yaml`` (validated by reading it back) and ``status.json``
    (state ``queued``) into a hidden staging folder, then renames that folder
    to ``<run_id>``. The rename is the moment the run appears, complete.
    ``now`` exists so tests can control the clock.
    """
    clean_name = name.strip() if isinstance(name, str) else ""
    if not clean_name:
        raise ValueError("a run needs a non-empty name")
    if len(clean_name) > 200:
        raise ValueError("run name is too long (max 200 characters)")
    if sweep_id is not None and not _RUN_ID_RE.match(sweep_id):
        raise ValueError(f"sweep id must be letters, digits and '-', got {sweep_id!r}")

    # Round-trip check: the file we write must read back into the same parameters.
    params_text = params_to_yaml(params)
    if params_from_yaml(params_text) != params:
        raise RunStoreError("parameters do not survive a YAML round trip; refusing to queue")

    base = _root(root)
    base.mkdir(parents=True, exist_ok=True)
    created_at = now or _now()
    base_id = make_run_id(clean_name, created_at)
    code_version = _code_version()

    staging = Path(tempfile.mkdtemp(dir=base, prefix=STAGING_PREFIX))
    try:
        (staging / PARAMS_FILE).write_text(params_text, encoding="utf-8")
        for attempt in range(1, 1000):
            run_id = base_id if attempt == 1 else f"{base_id}-{attempt}"
            target = base / run_id
            if target.exists():
                continue
            status = RunStatus(
                run_id=run_id,
                name=clean_name,
                state="queued",
                created_at=created_at,
                message="waiting in the queue",
                sweep_id=sweep_id,
                code_version=code_version,
            )
            _write_status(staging, status)
            try:
                # Renaming onto an existing non-empty folder fails, so two
                # creators racing for the same id can't overwrite each other.
                os.rename(staging, target)
            except OSError:
                if target.exists():
                    continue
                raise
            return run_id
        raise RunStoreError(f"could not find a free run id for {base_id!r}")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# --- reading runs ------------------------------------------------------------------


def _unreadable(folder: Path, reason: str) -> RunStatus:
    """A stand-in status for a folder whose status.json can't be used."""
    try:
        created = datetime.fromtimestamp(folder.stat().st_mtime)
    except OSError:
        created = _now()
    return RunStatus(
        run_id=folder.name,
        name=folder.name,
        state="unreadable",
        created_at=created,
        error=reason,
    )


def _read_status_in(folder: Path, run_id: str | None = None) -> RunStatus:
    """Parse ``folder/status.json``; ``run_id`` defaults to the folder name
    (it differs only while a folder sits under a ``_trash-`` name)."""
    expected = run_id or folder.name
    path = folder / STATUS_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _unreadable(folder, "status.json is missing")
    except OSError as exc:
        return _unreadable(folder, f"status.json could not be read: {exc}")
    try:
        status = RunStatus.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
        return _unreadable(folder, f"status.json is corrupt: {first.get('msg')}")
    if status.state == "unreadable" or status.run_id != expected:
        return _unreadable(folder, "status.json does not describe this folder")
    return status


def read_status(run_id: str, root: Path | str | None = None) -> RunStatus:
    """The run's status. Never raises for a bad file: it returns state ``unreadable``.

    Raises :class:`RunNotFound` only if the folder itself doesn't exist (for
    example a stale ``?run=`` link to a deleted run).
    """
    return _read_status_in(_existing_run_dir(run_id, root))


def _run_folders(root: Path | str | None) -> Iterator[Path]:
    """Run folders in the root. Skips files, the ``dev`` folder, and names starting
    with ``_`` or ``.`` (heartbeat, lock, staging/trash folders, temp files)."""
    base = _root(root)
    if not base.is_dir():
        return
    for entry in base.iterdir():
        if entry.name.startswith(("_", ".")) or entry.name == DEV_RUN:
            continue
        if entry.is_dir() and _RUN_ID_RE.match(entry.name):
            yield entry


def list_runs(root: Path | str | None = None) -> list[RunStatus]:
    """Every run, newest first. Unreadable folders are included (state ``unreadable``)."""
    statuses = [_read_status_in(folder) for folder in _run_folders(root)]
    return sorted(statuses, key=lambda s: (s.created_at, s.run_id), reverse=True)


def queued_runs(
    root: Path | str | None = None, *, include_cancel_requested: bool = False
) -> list[RunStatus]:
    """Queued runs in the order the worker will execute them: oldest first.

    Order is ``created_at``, ties broken by ``run_id``. Runs with a cancel
    request are left out by default ("they leave the numbering"), so
    ``index + 1`` is the queue position the UI shows.
    """
    base = _root(root)
    queued = [
        s
        for s in (_read_status_in(folder) for folder in _run_folders(base))
        if s.state == "queued"
        and (include_cancel_requested or not (base / s.run_id / CANCEL_FILE).exists())
    ]
    return sorted(queued, key=lambda s: (s.created_at, s.run_id))


def read_params(run_id: str, root: Path | str | None = None) -> Params:
    """The run's validated parameters (raises if ``params.yaml`` is missing or invalid)."""
    return load_params(_existing_run_dir(run_id, root) / PARAMS_FILE)


def append_log(run_id: str, text: str, root: Path | str | None = None) -> None:
    """Append a timestamped line to the run's ``log.txt`` (no-op if the run is gone)."""
    folder = run_dir(run_id, root)
    if not folder.is_dir():
        return
    with (folder / LOG_FILE).open("a", encoding="utf-8") as handle:
        handle.write(f"{_now():%Y-%m-%d %H:%M:%S} {text}\n")


def read_log_tail(run_id: str, lines: int = 50, root: Path | str | None = None) -> str:
    """The last ``lines`` lines of ``log.txt`` ("" if there is no log yet)."""
    path = _existing_run_dir(run_id, root) / LOG_FILE
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


# --- changing runs -----------------------------------------------------------------


def _check_transition(old: RunStatus, new_state: str) -> None:
    if old.state == "unreadable":
        raise IllegalTransition(f"run {old.run_id}: status.json is unreadable ({old.error})")
    if old.state in TERMINAL_STATES:
        raise IllegalTransition(f"run {old.run_id} is {old.state}; final states never change")
    if new_state != old.state and new_state not in _TRANSITIONS[old.state]:
        raise IllegalTransition(f"run {old.run_id}: {old.state} -> {new_state} is not allowed")


def update_status(
    run_id: str,
    /,
    *,
    root: Path | str | None = None,
    now: datetime | None = None,
    **changes: object,
) -> RunStatus:
    """Apply ``changes`` to ``status.json`` atomically and return the new status.

    Enforces the state machine (raises :class:`IllegalTransition`) and fills
    ``started_at`` on entering ``running`` and ``finished_at`` (plus
    ``progress = 1`` for ``done``) on entering a final state. A call without
    ``state`` is a progress update and is only allowed on a non-final run.
    """
    unknown = set(changes) - _MUTABLE_FIELDS
    if unknown:
        raise ValueError(f"these status fields can't be changed: {sorted(unknown)}")
    folder = _existing_run_dir(run_id, root)
    old = _read_status_in(folder)
    new_state = str(changes.get("state", old.state))
    _check_transition(old, new_state)

    stamp = now or _now()
    data = old.model_dump()
    data.update(changes)
    if new_state != old.state:
        if new_state == "running" and "started_at" not in changes:
            data["started_at"] = stamp
        if new_state in TERMINAL_STATES and "finished_at" not in changes:
            data["finished_at"] = stamp
        if new_state == "done" and "progress" not in changes:
            data["progress"] = 1.0
    if "progress" in changes and isinstance(changes["progress"], int | float):
        data["progress"] = min(1.0, max(0.0, float(changes["progress"])))
    new = RunStatus.model_validate(data)
    _write_status(folder, new)
    return new


def request_cancel(run_id: str, root: Path | str | None = None) -> bool:
    """Ask the worker to stop a queued or running run. Returns False if it already ended.

    Only writes the ``cancel`` flag file; the worker does the state change
    (queued runs on its next poll, running runs at the next simulated week).
    """
    folder = _existing_run_dir(run_id, root)
    if _read_status_in(folder).state not in ("queued", "running"):
        return False
    (folder / CANCEL_FILE).write_text(_now().isoformat() + "\n", encoding="utf-8")
    return True


def cancel_requested(run_id: str, root: Path | str | None = None) -> bool:
    """True if the ``cancel`` flag file exists (cheap: one ``stat``)."""
    return (run_dir(run_id, root) / CANCEL_FILE).exists()


def run_is_live(run_id: str, root: Path | str | None = None) -> bool:
    """True if a live executor (fresh heartbeat) says it is working on this run.

    A ``running`` status alone is not proof: if the final status write failed,
    or a worker died, the status says ``running`` with nobody behind it.
    """
    beat = read_heartbeat(root)
    return beat is not None and worker_alive(root) and beat.current_run == run_id


def delete_run(run_id: str, root: Path | str | None = None) -> None:
    """Remove a run folder for good. Refuses while a live executor works on it.

    A ``running`` run whose executor is gone (no fresh heartbeat, or the
    heartbeat names another run) is stale and may be deleted.

    The folder is first renamed to a hidden ``_trash-`` name (atomic), so the
    run disappears from listings at once and the worker gets a clean "not
    found". The status is then read **again** from the trash: if the worker
    started the run between our first look and the rename, the folder is put
    back and the delete is refused (check-then-act race closed by re-checking
    after the atomic step).
    """
    folder = _existing_run_dir(run_id, root)
    if _read_status_in(folder).state == "running" and run_is_live(run_id, root):
        raise RunStoreError(f"run {run_id} is running; cancel it before deleting")
    trash = folder.parent / f"{TRASH_PREFIX}{run_id}-{uuid.uuid4().hex[:8]}"
    os.rename(folder, trash)
    if _read_status_in(trash, run_id).state == "running" and run_is_live(run_id, root):
        try:
            os.rename(trash, folder)
        except OSError as exc:
            # A writer recreated the folder in the meantime; the run's process
            # notices its status.json is gone and stops. The trash is removed
            # later by cleanup_stale_temp.
            raise RunStoreError(
                f"run {run_id} started while being deleted and could not be restored: {exc}"
            ) from exc
        raise RunStoreError(f"run {run_id} started while being deleted; it was not deleted")
    shutil.rmtree(trash, ignore_errors=True)


def cleanup_stale_temp(root: Path | str | None = None, older_than_s: float = 600.0) -> int:
    """Remove staging/trash folders left by a crash mid-create or mid-delete.

    Only folders older than ``older_than_s`` are touched, so a create that is
    in progress right now is safe. Returns how many were removed.
    """
    base = _root(root)
    if not base.is_dir():
        return 0
    removed = 0
    cutoff = time.time() - older_than_s
    for entry in base.iterdir():
        if not entry.name.startswith((STAGING_PREFIX, TRASH_PREFIX)) or not entry.is_dir():
            continue
        with contextlib.suppress(OSError):
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry)
                removed += 1
    return removed


# --- worker heartbeat --------------------------------------------------------------


def write_heartbeat(
    pid: int,
    current_run: str | None,
    root: Path | str | None = None,
    now: datetime | None = None,
) -> None:
    """Rewrite ``_worker.json`` (atomically). The worker calls this every poll."""
    base = _root(root)
    base.mkdir(parents=True, exist_ok=True)
    beat = Heartbeat(pid=pid, heartbeat_at=now or _now(), current_run=current_run)
    _atomic_write_text(base / HEARTBEAT_FILE, beat.model_dump_json() + "\n")


def read_heartbeat(root: Path | str | None = None) -> Heartbeat | None:
    """The last heartbeat, or None if there is none or it can't be parsed."""
    try:
        text = (_root(root) / HEARTBEAT_FILE).read_text(encoding="utf-8")
        return Heartbeat.model_validate_json(text)
    except (OSError, ValidationError, json.JSONDecodeError):
        return None


def remove_heartbeat(root: Path | str | None = None, pid: int | None = None) -> None:
    """Delete ``_worker.json`` on clean shutdown (only our own, if ``pid`` is given)."""
    if pid is not None:
        beat = read_heartbeat(root)
        if beat is not None and beat.pid != pid:
            return
    with contextlib.suppress(FileNotFoundError):
        (_root(root) / HEARTBEAT_FILE).unlink()


def worker_alive(
    root: Path | str | None = None, max_age_s: float = 10.0, now: datetime | None = None
) -> bool:
    """True if a worker wrote a heartbeat in the last ``max_age_s`` seconds."""
    beat = read_heartbeat(root)
    if beat is None:
        return False
    age = ((now or _now()) - beat.heartbeat_at).total_seconds()
    return -max_age_s <= age <= max_age_s  # a beat "from the future" = clock jumped
