"""Unit tests for radical.adr.logging — the user-facing logging setup.

Covers:
  - enable_logging() configures the `radical.adr` namespace logger: level,
    a single handler, propagate=False (the AsyncFlow/RHAPSODY-compatible
    coexistence pattern)
  - re-calling enable_logging is idempotent (no duplicate handlers)
  - the shared pipe format is used; the file handler strips colors
  - configure_stack=True is guarded — a missing/odd sibling never raises
  - the module logger emits under the radical.adr namespace
"""

import io
import logging

import pytest

from radical.adr import enable_logging
from radical.adr.logging import ADR_LOGGER_NAME, LOG_FORMAT, _ColoredFormatter


@pytest.fixture
def reset_adr_logger():
    """Snapshot/restore the radical.adr logger so tests don't leak handlers."""
    lg = logging.getLogger(ADR_LOGGER_NAME)
    saved = (lg.level, list(lg.handlers), lg.propagate)
    lg.handlers.clear()
    yield lg
    lg.setLevel(saved[0])
    lg.handlers[:] = saved[1]
    lg.propagate = saved[2]


def test_enable_logging_configures_namespace(reset_adr_logger):
    stream = io.StringIO()
    lg = enable_logging(logging.INFO, stream=stream)

    assert lg.name == ADR_LOGGER_NAME
    assert lg.level == logging.INFO
    assert lg.propagate is False            # coexistence: stays off the root
    assert len(lg.handlers) == 1            # one console handler


def test_enable_logging_is_idempotent(reset_adr_logger):
    enable_logging(logging.DEBUG, stream=io.StringIO())
    enable_logging(logging.DEBUG, stream=io.StringIO())
    # Handlers replaced, not accumulated.
    assert len(logging.getLogger(ADR_LOGGER_NAME).handlers) == 1


def test_emits_with_shared_pipe_format(reset_adr_logger):
    stream = io.StringIO()
    enable_logging(logging.INFO, stream=stream, use_colors=False)

    logging.getLogger("radical.adr.operator").info("hello %d", 7)
    out = stream.getvalue()
    assert "hello 7" in out
    assert "INFO" in out
    assert "[radical.adr.operator]" in out   # the [%(name)s] field
    assert " | " in out                       # the shared pipe delimiter


def test_debug_below_level_is_silent(reset_adr_logger):
    stream = io.StringIO()
    enable_logging(logging.INFO, stream=stream, use_colors=False)

    logging.getLogger("radical.adr.operator").debug("noisy detail")
    assert "noisy detail" not in stream.getvalue()


def test_file_handler_strips_colors(tmp_path, reset_adr_logger):
    logfile = tmp_path / "adr.log"
    enable_logging(logging.INFO, output_file=logfile,
                   stream=io.StringIO(), use_colors=True)

    logging.getLogger("radical.adr").info("to-file")
    for h in logging.getLogger(ADR_LOGGER_NAME).handlers:
        h.flush()

    text = logfile.read_text()
    assert "to-file" in text
    assert "\033[" not in text               # no ANSI escapes in the file


def test_colored_formatter_colors_only_when_enabled():
    record = logging.LogRecord("radical.adr", logging.INFO, __file__, 1,
                               "msg", None, None)
    colored = _ColoredFormatter(use_colors=True).format(record)
    plain   = _ColoredFormatter(use_colors=False).format(record)
    assert "\033[" in colored
    assert "\033[" not in plain
    # Formatting the record must not mutate it permanently.
    assert record.levelname == "INFO"


def test_configure_stack_never_raises(reset_adr_logger):
    # Even though AsyncFlow/RHAPSODY setup is best-effort, this must not raise.
    enable_logging(logging.INFO, stream=io.StringIO(), configure_stack=True)


def test_log_format_constant_matches_rhapsody():
    # Cross-project compatibility hinges on the shared format string.
    assert LOG_FORMAT == "%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s"
