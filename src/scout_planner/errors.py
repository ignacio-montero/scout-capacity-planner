"""Exceptions shared between the pipeline and the run store / worker.

Kept in their own tiny module so the pipeline (pure stages) can raise
:class:`RunCancelled` without importing the run store, which does file I/O.
"""

from __future__ import annotations


class RunCancelled(Exception):
    """Raised by ``run_pipeline`` when ``should_cancel()`` returned True.

    The worker turns it into the ``cancelled`` state instead of ``failed``.
    """


class RunStoreError(Exception):
    """A run-store operation was refused (illegal transition, deleting a running run...)."""


class RunNotFound(RunStoreError, LookupError):
    """No run folder with this id (never created, or deleted)."""


class IllegalTransition(RunStoreError):
    """A status change that the run state machine does not allow."""
