"""
Debate Interface: run N DebateSubagents in a RoundRobinGroupChat until a
majority converges or ``max_rounds`` is reached.

High-level flow:

    question ──► RoundRobinGroupChat([agent_1, ..., agent_N])
                 │
                 │  each turn: agent may call `retrieve`, then emits text
                 │  that ends with a line  STANCE: {"answer": ..., "confidence": ..., "citations": [...]}
                 │
                 ▼
        MajorityStableTermination watches the stream, records one DebateRound
        per N stance-bearing messages, clusters stances via majority_vote, and
        stops early when the same majority holds for `stable_for` rounds.

After termination, :class:`DebateInterface.run` builds a :class:`DebateResult`
(wrapping a :class:`DebateTranscript`) directly from the termination's
recorded rounds.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    StopMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat

from src.agents.debate.debate_subagent import DebateSubagent
from src.agents.debate.majority_vote import cluster_answers
from src.schemas import DebateRound, DebateTranscript


STANCE_RE = re.compile(r"STANCE:\s*(\{.*\})\s*$", re.DOTALL)


def _extract_stance(text: str) -> Optional[dict]:
    """
    Parse the trailing ``STANCE: {json}`` line from an agent message.

    Returns ``None`` if no parseable stance is present. Robust to the
    stance being anywhere near the end of the message.
    """
    if "STANCE:" not in text:
        return None
    tail = text[text.rfind("STANCE:") :]
    m = STANCE_RE.search(tail.strip())
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return {
        "answer": str(data.get("answer", "")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "citations": [str(c) for c in data.get("citations", [])],
    }


# ---------------------------------------------------------------------------
# Termination condition
# ---------------------------------------------------------------------------

class MajorityStableTermination(TerminationCondition):
    """
    Stop when the same majority cluster (by agent_id set) has held for
    ``stable_for`` consecutive rounds, or when ``max_rounds`` have elapsed.

    A "round" is defined as one stance-bearing chat message per participating
    agent. Tool-call summary messages without a STANCE line are ignored.
    """

    def __init__(
        self,
        agent_ids: Sequence[str],
        max_rounds: int,
        stable_for: int = 2,
        llm_cluster_fn: Optional[Callable[[List[str]], List[List[int]]]] = None,
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if stable_for < 1:
            raise ValueError("stable_for must be >= 1")

        self._agent_ids = list(agent_ids)
        self._num_agents = len(self._agent_ids)
        self._agent_index = {aid: i for i, aid in enumerate(self._agent_ids)}
        self._max_rounds = max_rounds
        self._stable_for = stable_for
        self._llm_cluster_fn = llm_cluster_fn

        self.rounds: List[DebateRound] = []
        self._majority_history: List[frozenset[str]] = []
        self._pending: dict[str, dict] = {}
        self._pending_messages: dict[str, str] = {}
        self._terminated = False
        self._stop_reason: str = ""

        self.last_rounds: List[DebateRound] = []
        self.last_stop_reason: str = ""

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    async def __call__(
        self, messages: Sequence[BaseAgentEvent | BaseChatMessage]
    ) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException(
                "MajorityStableTermination has already been reached"
            )

        for msg in messages:
            if not isinstance(msg, BaseChatMessage):
                continue
            source = getattr(msg, "source", None)
            if source not in self._agent_index:
                continue
            text = msg.to_text() if hasattr(msg, "to_text") else str(
                getattr(msg, "content", "")
            )
            stance = _extract_stance(text)
            if stance is None:
                continue

            self._pending[source] = stance
            self._pending_messages[source] = text

            if len(self._pending) == self._num_agents:
                self._finalize_round()
                if self._terminated:
                    return StopMessage(
                        content=f"debate:{self._stop_reason}",
                        source="MajorityStableTermination",
                    )

        return None

    def _finalize_round(self) -> None:
        round_num = len(self.rounds) + 1
        stances = {aid: self._pending[aid]["answer"] for aid in self._agent_ids}
        confidences = {
            aid: self._pending[aid]["confidence"] for aid in self._agent_ids
        }
        messages = {
            aid: self._pending_messages.get(aid, "") for aid in self._agent_ids
        }

        self.rounds.append(
            DebateRound(
                round_num=round_num,
                stances=stances,
                confidences=confidences,
                messages=messages,
            )
        )

        answers = [stances[aid] for aid in self._agent_ids]
        clusters = cluster_answers(answers, llm_cluster_fn=self._llm_cluster_fn)
        top = clusters[0] if clusters else []
        top_ids = frozenset(self._agent_ids[i] for i in top)
        self._majority_history.append(top_ids)

        self._pending = {}
        self._pending_messages = {}

        if self._is_converged(top_ids):
            self._terminated = True
            self._stop_reason = "converged"
        elif round_num >= self._max_rounds:
            self._terminated = True
            self._stop_reason = "max_rounds"

        if self._terminated:
            self.last_rounds = list(self.rounds)
            self.last_stop_reason = self._stop_reason

    def _is_converged(self, top_ids: frozenset[str]) -> bool:
        if len(top_ids) < (self._num_agents // 2) + 1:
            return False
        if len(self._majority_history) < self._stable_for:
            return False
        recent = self._majority_history[-self._stable_for :]
        return all(m == top_ids for m in recent)

    async def reset(self) -> None:
        self.rounds = []
        self._majority_history = []
        self._pending = {}
        self._pending_messages = {}
        self._terminated = False
        self._stop_reason = ""


# ---------------------------------------------------------------------------
# Debate interface
# ---------------------------------------------------------------------------

@dataclass
class DebateResult:
    """Structured output of :meth:`DebateInterface.run`."""

    transcript: DebateTranscript
    majority_answer: str
    majority_cluster_ids: List[str] = field(default_factory=list)
    rounds_used: int = 0
    stopped_reason: str = ""


class DebateInterface:
    """
    Runs the debate. Owns the AutoGen team; holds no per-query state between
    runs so it can be reused across queries in an experiment.
    """

    def __init__(
        self,
        subagents: List[DebateSubagent],
        max_rounds: int = 4,
        stable_for: int = 2,
        llm_cluster_fn: Optional[Callable[[List[str]], List[List[int]]]] = None,
    ):
        if len(subagents) < 2:
            raise ValueError("Debate requires at least 2 subagents.")
        seen = set()
        for s in subagents:
            if s.agent_id in seen:
                raise ValueError(f"Duplicate agent_id: {s.agent_id}")
            seen.add(s.agent_id)
        self.subagents = subagents
        self.max_rounds = max_rounds
        self.stable_for = stable_for
        self._llm_cluster_fn = llm_cluster_fn

    def run(self, question: str) -> DebateResult:
        """Synchronous wrapper around :meth:`arun`."""
        return asyncio.run(self.arun(question))

    async def arun(self, question: str) -> DebateResult:
        agents = [s.build_agent(question) for s in self.subagents]
        agent_ids = [s.agent_id for s in self.subagents]

        termination = MajorityStableTermination(
            agent_ids=agent_ids,
            max_rounds=self.max_rounds,
            stable_for=self.stable_for,
            llm_cluster_fn=self._llm_cluster_fn,
        )

        team = RoundRobinGroupChat(
            participants=agents,
            termination_condition=termination,
            max_turns=self.max_rounds * len(agents) * 4,
        )

        await team.run(task=question)

        return self._build_result(termination)

    def _build_result(self, termination: MajorityStableTermination) -> DebateResult:
        rounds = termination.last_rounds or termination.rounds
        rounds_used = len(rounds)
        stopped_reason = (
            termination.last_stop_reason
            or termination.stop_reason
            or ("max_rounds" if rounds_used >= self.max_rounds else "no_rounds")
        )

        if not rounds:
            transcript = DebateTranscript(
                rounds=[],
                majority_cluster=[],
                majority_answer="",
                rounds_used=0,
                stopped_reason=stopped_reason,
            )
            return DebateResult(
                transcript=transcript,
                majority_answer="",
                majority_cluster_ids=[],
                rounds_used=0,
                stopped_reason=stopped_reason,
            )

        final = rounds[-1]
        agent_ids = [s.agent_id for s in self.subagents]
        answers = [final.stances.get(aid, "") for aid in agent_ids]
        clusters = cluster_answers(answers, llm_cluster_fn=self._llm_cluster_fn)
        top = clusters[0] if clusters else []
        majority_ids = [agent_ids[i] for i in top]
        majority_answer = answers[top[0]] if top else ""

        transcript = DebateTranscript(
            rounds=rounds,
            majority_cluster=majority_ids,
            majority_answer=majority_answer,
            rounds_used=rounds_used,
            stopped_reason=stopped_reason,
        )
        return DebateResult(
            transcript=transcript,
            majority_answer=majority_answer,
            majority_cluster_ids=majority_ids,
            rounds_used=rounds_used,
            stopped_reason=stopped_reason,
        )
