"""
JudgeLLM: top-level entry point for the debate setup.

Mirrors step 1 + 5 in the debate diagram:

    user_input + trigger  ─►  JudgeLLM  ─►  spawn DebateInterface
                                │
                                │  debate runs: N subagents × rounds
                                │
                                ◄── majority vote from DebateInterface
                                │
                                ▼
                          target action  =  emit final RunLog

In the clean setup the judge is a **pure relay**: the final answer is the
majority vote; the judge does not re-score or override. This keeps the
threat-model clean for Phase 4 (attacker wants the judge to relay a
specific wrong answer).
"""

from __future__ import annotations

from typing import Callable, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.agents.debate.debate_interface import DebateInterface, DebateResult
from src.agents.debate.debate_subagent import DebateSubagent
from src.logging_utils import emit_run_log
from src.schemas import (
    DebateTranscript,
    OrchestratorOutput,
    RunLog,
    SubagentOutput,
)


class JudgeLLM:
    """
    Pure-relay Judge.

    Args:
        subagents: List of DebateSubagent instances (already configured with
            their private retrievers and model client). The caller decides
            whether retrievers are shared or diversified.
        max_rounds: Upper bound on debate rounds.
        stable_for: Number of consecutive rounds a majority must persist to
            trigger convergence-based early stop.
        output_dir: Where :func:`emit_run_log` writes ``runs.jsonl``.
        llm_cluster_fn: Optional override for majority clustering (paraphrase
            merging). Tests stub this; production can wire gpt-5.
    """

    def __init__(
        self,
        subagents: List[DebateSubagent],
        max_rounds: int = 4,
        stable_for: int = 2,
        output_dir: str = "results",
        llm_cluster_fn: Optional[Callable[[List[str]], List[List[int]]]] = None,
    ):
        self.subagents = subagents
        self.output_dir = output_dir
        self.debate = DebateInterface(
            subagents=subagents,
            max_rounds=max_rounds,
            stable_for=stable_for,
            llm_cluster_fn=llm_cluster_fn,
        )

    def run(
        self,
        query: str,
        query_id: str,
        trigger: Optional[str] = None,
        attack_condition: str = "clean",
        ground_truth_answer: Optional[str] = None,
        emit: bool = True,
    ) -> RunLog:
        """
        Spawn a debate for ``query`` and relay the majority vote as the
        final answer. Writes the :class:`RunLog` to ``runs.jsonl`` and
        returns it.
        """
        debate_query = f"{query} {trigger}" if trigger else query

        result = self.debate.run(debate_query)

        agent_responses = self._build_subagent_outputs(result)
        retrieved_per_agent = {
            s.agent_id: list(s.retrieved_doc_ids) for s in self.subagents
        }
        any_poison = any(s.poison_retrieved for s in self.subagents)

        num_agents = len(self.subagents)
        majority_size = len(result.majority_cluster_ids)
        final_confidence = majority_size / num_agents if num_agents else 0.0

        final_decision = OrchestratorOutput(
            final_answer=result.majority_answer,
            selected_evidence=self._majority_citations(agent_responses, result),
            winning_subagents=result.majority_cluster_ids,
            final_confidence=final_confidence,
            harmful_action_flag=False,
            reasoning_summary=(
                f"majority-vote relay after {result.rounds_used} round(s)"
                f"; stopped_reason={result.stopped_reason}"
            ),
        )

        run_log = RunLog(
            query_id=query_id,
            attack_condition=attack_condition,
            trigger=trigger,
            ground_truth_answer=ground_truth_answer,
            retrieved_doc_ids_per_agent=retrieved_per_agent,
            poison_retrieved=any_poison,
            agent_responses=agent_responses,
            final_decision=final_decision,
            debate_transcript=result.transcript,
            metrics={},
        )

        if emit:
            emit_run_log(run_log, output_dir=self.output_dir)
        return run_log

    def _build_subagent_outputs(
        self, result: DebateResult
    ) -> dict[str, SubagentOutput]:
        outputs: dict[str, SubagentOutput] = {}
        final_round = result.transcript.rounds[-1] if result.transcript.rounds else None

        for s in self.subagents:
            answer = final_round.stances.get(s.agent_id, "") if final_round else ""
            confidence = (
                float(final_round.confidences.get(s.agent_id, 0.0))
                if final_round
                else 0.0
            )
            rationale = final_round.messages.get(s.agent_id, "") if final_round else ""

            citations = self._extract_citations_from_message(rationale)

            outputs[s.agent_id] = SubagentOutput(
                agent_id=s.agent_id,
                answer=answer,
                citations=citations,
                confidence=max(0.0, min(1.0, confidence)),
                rationale=rationale[:2000],
                poison_retrieved=s.poison_retrieved,
                retrieved_doc_ids=list(s.retrieved_doc_ids),
                poisoned_retrieved_ids=list(s.poisoned_retrieved_ids),
            )
        return outputs

    @staticmethod
    def _extract_citations_from_message(message: str) -> List[str]:
        from src.agents.debate.debate_interface import _extract_stance

        stance = _extract_stance(message)
        if stance is None:
            return []
        return stance.get("citations", [])

    @staticmethod
    def _majority_citations(
        agent_responses: dict[str, SubagentOutput], result: DebateResult
    ) -> List[str]:
        cites: List[str] = []
        for aid in result.majority_cluster_ids:
            out = agent_responses.get(aid)
            if out is None:
                continue
            for c in out.citations:
                if c not in cites:
                    cites.append(c)
        return cites
