"""User-facing logging for radical.adr.

ADR has no logging system of its own beyond per-module ``logging.getLogger``
loggers under the ``radical.adr`` namespace. This module adds a single setup
function, :func:`enable_logging`, that narrates the Observe -> Decide -> Act
loop at INFO/DEBUG for debuggability.

It is deliberately compatible with the rest of the stack: AsyncFlow
(``radical.asyncflow.logging.init_default_logger``) and RHAPSODY
(``rhapsody.enable_logging``) both use Python's stdlib ``logging`` with
per-module loggers and a pipe-delimited format. ADR matches that format so the
three projects' logs interleave uniformly, and — like RHAPSODY — configures its
own namespace logger with ``propagate=False`` so ADR logs are never duplicated
and survive a sibling reconfiguring the root logger.

Usage::

    import logging
    from radical.adr import enable_logging

    enable_logging(logging.INFO)                       # ADR only
    enable_logging(logging.DEBUG, configure_stack=True) # ADR + AsyncFlow + RHAPSODY

Logging is a side channel — it never flows through ``state``, ``snapshot``, or
``@observe``. For structured, machine-readable per-cycle data use an
``ObserverBase`` (see ``radical.adr.observer`` / ``RecordingObserver``).
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

# The namespace every radical.adr module logger lives under
# (``logging.getLogger(__name__)`` -> ``radical.adr.<module>``).
ADR_LOGGER_NAME = "radical.adr"

# Shared with AsyncFlow / RHAPSODY so all three projects' logs interleave
# uniformly. RHAPSODY uses this exact format string.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s"

# ANSI colors applied to the level name only (console, when the stream is a TTY).
_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[34m",   # blue
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColoredFormatter(logging.Formatter):
    """Pipe-delimited formatter that optionally colors the level name.

    Self-contained on purpose: ADR couples to AsyncFlow/RHAPSODY only on the
    shared *format string*, not on any formatter implementation.
    """

    def __init__(self, *, use_colors: bool = True, show_details: bool = False) -> None:
        fmt = LOG_FORMAT
        if show_details:
            fmt = "%(asctime)s | %(levelname)-8s | [%(name)s] | " \
                  "(%(processName)s/%(threadName)s) %(message)s"
        super().__init__(fmt)
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_colors:
            return super().format(record)
        color = _LEVEL_COLORS.get(record.levelname)
        if not color:
            return super().format(record)
        # Color only the level name, then restore, so the rest stays plain.
        original = record.levelname
        record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def _wants_color(use_colors: bool, stream: Any) -> bool:
    if not use_colors:
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def enable_logging(
    level: int | str = logging.INFO,
    *,
    output_file: "str | pathlib.Path | None" = None,
    file_level: int | str | None = None,
    use_colors: bool = True,
    show_details: bool = False,
    stream: Any = None,
    configure_stack: bool = False,
) -> logging.Logger:
    """Configure user-facing logging for the ``radical.adr`` namespace.

    Args:
        level: Console log level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        output_file: If given, also log to this file (colors disabled in files).
        file_level: File log level; defaults to ``level``.
        use_colors: Color the level name on the console when the stream is a TTY.
        show_details: Include process/thread info in each line.
        stream: Console stream; defaults to ``sys.stderr`` (RHAPSODY parity).
        configure_stack: Also configure AsyncFlow and RHAPSODY logging at the
            same ``level``/``stream`` so the whole stack shares one format.
            Each sibling is guarded — a missing RHAPSODY is silently skipped.

    Returns:
        The configured ``radical.adr`` logger.
    """
    if stream is None:
        stream = sys.stderr

    logger = logging.getLogger(ADR_LOGGER_NAME)
    logger.setLevel(level)

    # Replace our handlers (idempotent re-config; avoids duplicate lines).
    logger.handlers.clear()

    console = logging.StreamHandler(stream)
    console.setLevel(level)
    console.setFormatter(
        _ColoredFormatter(
            use_colors=_wants_color(use_colors, stream),
            show_details=show_details,
        )
    )
    logger.addHandler(console)

    if output_file is not None:
        path = pathlib.Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setLevel(level if file_level is None else file_level)
        fh.setFormatter(_ColoredFormatter(use_colors=False, show_details=show_details))
        logger.addHandler(fh)

    # Coexistence: keep ADR logs off the root logger so a sibling's
    # basicConfig(force=True) can't duplicate or swallow them (RHAPSODY pattern).
    logger.propagate = False

    # Bring any already-created radical.adr.* child loggers to the same level.
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name == ADR_LOGGER_NAME or name.startswith(ADR_LOGGER_NAME + "."):
            logging.getLogger(name).setLevel(level)

    if configure_stack:
        _configure_siblings(level, stream)

    return logger


def _configure_siblings(level: int | str, stream: Any) -> None:
    """Cascade the same level to AsyncFlow and RHAPSODY (best-effort)."""
    try:
        from radical.asyncflow.logging import init_default_logger
        init_default_logger(level, logger_name="radical.asyncflow")
    except Exception:
        # AsyncFlow is a hard dependency, but never let its logging setup
        # (e.g. a version without init_default_logger) break ADR's.
        logging.getLogger(ADR_LOGGER_NAME).debug(
            "configure_stack: could not configure AsyncFlow logging", exc_info=True
        )

    try:
        import rhapsody
        rhapsody.enable_logging(level=level, stream=stream)
    except ImportError:
        pass   # RHAPSODY is optional — silently skip when absent.
    except Exception:
        logging.getLogger(ADR_LOGGER_NAME).debug(
            "configure_stack: could not configure RHAPSODY logging", exc_info=True
        )
