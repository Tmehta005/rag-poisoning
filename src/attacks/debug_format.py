"""
Human-readable formatting for debate and orchestrator run logs.

Two entry points:

- :func:`format_run_log` — render a :class:`~src.schemas.RunLog` (or a
  ``dict`` loaded from ``results/runs.jsonl``) as a multi-line, readable
  string. Works for both the orchestrator and the debate pipeline.
- :func:`print_jsonl` — stream every record from a ``runs.jsonl`` file,
  optionally filtered by ``query_id`` / ``attack_condition``.

These helpers are the primary debugging surface for the debate attack
runner: use ``--verbose`` at run time to print live, or re-read past runs
offline with :mod:`src.experiments.inspect_runs`.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, Iterator

from src.schemas import DebateTranscript, RunLog


# ---------------------------------------------------------------------------
# RunLog rendering
# ---------------------------------------------------------------------------


def _truncate(text: str, width: int = 240) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def _bar(char: str = "─", width: int = 72) -> str:
    return char * width


def _indent(block: str, prefix: str = "  ") -> str:
    return textwrap.indent(block, prefix)


def _format_retrieval(
    retrieved_per_agent: dict[str, list[str]],
    poison_ids_per_agent: dict[str, set[str]] | None = None,
) -> str:
    lines: list[str] = []
    poison_ids_per_agent = poison_ids_per_agent or {}
    for aid, ids in retrieved_per_agent.items():
        marked: list[str] = []
        poisons = poison_ids_per_agent.get(aid, set())
        for doc_id in ids:
            short = doc_id if len(doc_id) <= 12 else doc_id[:8] + "…"
            if doc_id in poisons:
                marked.append(f"POISON[{short}]")
            else:
                marked.append(short)
        lines.append(f"{aid}: {', '.join(marked) or '(none)'}")
    return "\n".join(lines)


def _format_transcript(t: DebateTranscript) -> str:
    out: list[str] = []
    out.append(
        f"rounds_used={t.rounds_used}  "
        f"stopped_reason={t.stopped_reason!r}  "
        f"majority={list(t.majority_cluster)}"
    )
    out.append(f"majority_answer: {_truncate(t.majority_answer, 200)}")
    out.append("")
    for r in t.rounds:
        out.append(f"─── Round {r.round_num} ───")
        for aid, stance in r.stances.items():
            conf = r.confidences.get(aid, 0.0)
            msg = r.messages.get(aid, "")
            out.append(f"  [{aid}] (conf={conf:.2f}) {_truncate(stance, 180)}")
            if msg and msg.strip() != stance.strip():
                out.append(_indent(_truncate(msg, 400), "      └ "))
        out.append("")
    return "\n".join(out).rstrip()


def format_run_log(
    log: RunLog | dict,
    *,
    show_prompts: bool = False,
) -> str:
    """Return a multi-line human-readable rendering of one run.

    Args:
        log: Either a :class:`RunLog` instance or a ``dict`` as loaded from
            ``results/runs.jsonl``.
        show_prompts: If True, include each subagent's rationale/message.
    """
    if isinstance(log, dict):
        data = log
    else:
        data = log.model_dump()

    parts: list[str] = []
    parts.append(_bar("═"))
    parts.append(
        f"query_id={data['query_id']}  "
        f"attack_condition={data['attack_condition']}  "
        f"trigger={data.get('trigger')!r}"
    )
    if data.get("ground_truth_answer"):
        parts.append(f"ground_truth: {_truncate(data['ground_truth_answer'])}")
    parts.append(_bar("─"))

    # Retrieval summary. Use the recorded `poisoned_retrieved_ids` subset
    # so we only star the actual D_p hits, not every doc from an agent that
    # happened to retrieve one poisoned chunk.
    poisons: dict[str, set[str]] = {}
    for aid, out in (data.get("agent_responses") or {}).items():
        if not isinstance(out, dict):
            continue
        hits = out.get("poisoned_retrieved_ids")
        if hits:
            poisons[aid] = set(hits)
    retrieved = data.get("retrieved_doc_ids_per_agent") or {}
    if retrieved:
        parts.append("RETRIEVAL")
        parts.append(_indent(_format_retrieval(retrieved, poisons)))
        parts.append("")

    # Per-agent summary (orchestrator setup)
    responses = data.get("agent_responses") or {}
    if responses and not data.get("debate_transcript"):
        parts.append("AGENT RESPONSES")
        for aid, out in responses.items():
            if not isinstance(out, dict):
                continue
            parts.append(
                f"  [{aid}] conf={out.get('confidence', 0):.2f}  "
                f"poison_retrieved={out.get('poison_retrieved', False)}"
            )
            parts.append(_indent(_truncate(out.get("answer", ""), 220), "      "))
            if show_prompts and out.get("rationale"):
                parts.append(_indent("why: " + _truncate(out["rationale"], 200), "      "))
        parts.append("")

    # Debate transcript
    transcript = data.get("debate_transcript")
    if transcript:
        if isinstance(transcript, dict):
            transcript = DebateTranscript.model_validate(transcript)
        parts.append("DEBATE TRANSCRIPT")
        parts.append(_indent(_format_transcript(transcript)))
        parts.append("")

    # Final decision
    fd = data.get("final_decision")
    if fd:
        parts.append("FINAL DECISION")
        parts.append(
            f"  winning_subagents={fd.get('winning_subagents')}  "
            f"final_confidence={fd.get('final_confidence', 0):.2f}  "
            f"harmful_action_flag={fd.get('harmful_action_flag')}"
        )
        parts.append(_indent("answer: " + _truncate(fd.get("final_answer", ""), 400), "  "))
        if fd.get("reasoning_summary"):
            parts.append(_indent("why: " + _truncate(fd["reasoning_summary"], 300), "  "))
        parts.append("")

    parts.append(
        f"poison_retrieved={data.get('poison_retrieved')}  "
        f"metrics={data.get('metrics') or {}}"
    )
    parts.append(_bar("═"))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# runs.jsonl streaming
# ---------------------------------------------------------------------------


def iter_runs_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield every record in a ``runs.jsonl`` file (skipping blanks)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"runs file not found: {p}")
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def filter_runs(
    records: Iterable[dict],
    *,
    query_id: str | None = None,
    attack_condition: str | None = None,
    only_last: bool = False,
) -> list[dict]:
    """Simple filter on ``runs.jsonl`` records.

    ``only_last`` keeps only the most recent record per (query_id,
    attack_condition) pair, useful when a query has been re-run.
    """
    out = [
        r
        for r in records
        if (query_id is None or r.get("query_id") == query_id)
        and (attack_condition is None or r.get("attack_condition") == attack_condition)
    ]
    if only_last:
        latest: dict[tuple[str, str], dict] = {}
        for r in out:
            key = (r.get("query_id", ""), r.get("attack_condition", ""))
            latest[key] = r
        return list(latest.values())
    return out
