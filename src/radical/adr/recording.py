"""RecordingObserver and replay_trace — record per-cycle obs/decision pairs to
JSONL for offline policy replay.

Independent of the demo ``JsonlObserver`` (which records high-level UI events).
``RecordingObserver`` records the raw ``obs`` dict the policy actually reasoned
over each cycle — exactly what ``replay_trace`` needs to re-drive a *different*
policy against identical recorded conditions, fully offline (no engine, no
Operator, no live execution).

Usage::

    # live run — capture the observation trace
    op = MyOperator(engine, observer=RecordingObserver("trace.jsonl"))
    await op.run_to_completion()

    # offline — replay the trace against a different policy
    decisions = await replay_trace("trace.jsonl", CandidatePolicy())
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radical.adr.decision import Decision
    from radical.adr.policy.base import Policy
    from radical.adr.state import Snapshot


class RecordingObserver:
    """Implements ``ObserverBase``. Appends one JSON line per cycle to ``path``::

        {"cycle": <int>, "obs": <obs dict>, "decision": <decision.model_dump()>}

    The recorded ``decision`` is for human inspection only — ``replay_trace``
    reads only ``obs``. The file is opened on ``on_start`` and closed on
    ``on_stop``.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = None

    def on_start(self, operator_id: str, metadata: dict) -> None:
        self._file = open(self._path, "w")

    def on_cycle(self, snapshot: "Snapshot", obs: dict, decision: "Decision") -> None:
        record = {
            "cycle":    snapshot.cycle,
            "obs":      obs,
            "decision": decision.model_dump(),
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def on_stop(self, final: "Snapshot | None", reason: str) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


async def replay_trace(
    path: str,
    policy: "Policy",
    operators: "dict[str, type] | None" = None,
) -> list["Decision"]:
    """Drive ``policy`` against each recorded ``obs`` in a JSONL trace.

    Reads a trace written by ``RecordingObserver`` and, for each line, calls
    ``policy.decide(obs)``, collecting the resulting ``Decision`` objects in
    order. Fully offline — no engine, no Operator, no live execution — because
    ``Policy.decide(obs)`` only needs a plain dict in and returns a ``Decision``.

    ``operators`` is an optional ``{child_name: OperatorClass}`` map. A
    ``SPAWN_OPERATOR`` action carries its ``operator_cls`` as a live reference
    excluded from serialization, so a replayed hierarchical policy may emit a
    spawn whose class is unresolved; when its child name is in ``operators`` the
    class is back-filled, making the Decision well-formed and inspectable. Replay
    stays open-loop — nothing is executed. ``operators=None`` ⇒ unchanged behavior.

    Bad lines and missing files are not swallowed: ``FileNotFoundError`` and
    ``json.JSONDecodeError`` propagate naturally.
    """
    from radical.adr.decision import ActionKind

    operators = operators or {}
    decisions: list["Decision"] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            decision = await policy.decide(record["obs"])
            for action in decision.actions:
                if action.kind == ActionKind.SPAWN_OPERATOR \
                        and action.operator_cls is None \
                        and action.task_name in operators:
                    action.operator_cls = operators[action.task_name]
            decisions.append(decision)
    return decisions
