"""
Tests for the clean debate setup.

All LLM calls go through AutoGen's ``ReplayChatCompletionClient`` so tests
run fully offline. The index uses ``MockEmbedding`` — no HuggingFace download.
"""

from __future__ import annotations

import json
from typing import List

import pytest
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.replay import ReplayChatCompletionClient
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from src.agents.debate.debate_interface import (
    DebateInterface,
    MajorityStableTermination,
    _extract_stance,
)
from src.agents.debate.debate_subagent import DebateSubagent
from src.agents.debate.judge import JudgeLLM
from src.agents.debate.majority_vote import cluster_answers, majority_cluster
from src.retriever import Retriever
from src.schemas import DebateTranscript, RunLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCS = [
    Document(text="Paris is the capital of France.", doc_id="d1"),
    Document(text="Berlin is the capital of Germany.", doc_id="d2"),
    Document(text="Rome is the capital of Italy.", doc_id="d3"),
]


@pytest.fixture(scope="module")
def mock_index():
    embed_model = MockEmbedding(embed_dim=8)
    return VectorStoreIndex.from_documents(DOCS, embed_model=embed_model, show_progress=False)


def _model_info() -> ModelInfo:
    return ModelInfo(
        vision=False,
        function_calling=True,
        json_output=False,
        family=ModelFamily.UNKNOWN,
        structured_output=False,
    )


def _stance_message(answer: str, confidence: float = 0.9, citations: List[str] | None = None) -> str:
    stance = {
        "answer": answer,
        "confidence": confidence,
        "citations": list(citations or ["d1"]),
    }
    return f"My current answer is {answer}.\nSTANCE: {json.dumps(stance)}"


def _make_client(messages: List[str]) -> ReplayChatCompletionClient:
    return ReplayChatCompletionClient(messages, model_info=_model_info())


def _build_subagents(
    mock_index, per_agent_messages: List[List[str]], top_k: int = 2
) -> List[DebateSubagent]:
    agents = []
    for i, messages in enumerate(per_agent_messages, start=1):
        retriever = Retriever(mock_index, top_k=top_k)
        agents.append(
            DebateSubagent(
                agent_id=f"subagent_{i}",
                retriever=retriever,
                model_client=_make_client(messages),
                top_k=top_k,
            )
        )
    return agents


# ---------------------------------------------------------------------------
# Unit tests: stance parsing and clustering
# ---------------------------------------------------------------------------

def test_extract_stance_happy_path():
    text = 'Argument goes here.\nSTANCE: {"answer": "Paris", "confidence": 0.9, "citations": ["d1"]}'
    stance = _extract_stance(text)
    assert stance == {"answer": "Paris", "confidence": 0.9, "citations": ["d1"]}


def test_extract_stance_missing_returns_none():
    assert _extract_stance("No stance here.") is None


def test_extract_stance_malformed_json_returns_none():
    text = "STANCE: {not valid json}"
    assert _extract_stance(text) is None


def test_cluster_answers_normalizes_case_and_punctuation():
    clusters = cluster_answers(["Paris.", "paris", "Rome"])
    assert clusters[0] == [0, 1]
    assert clusters[1] == [2]


def test_majority_cluster_ties_broken_deterministically():
    clusters = cluster_answers(["A", "B"])
    assert clusters == [[0], [1]]
    assert majority_cluster(["A", "B"]) == [0]


def test_cluster_answers_uses_llm_fn_when_provided():
    def fake_llm(answers):
        return [[0, 1, 2]]

    clusters = cluster_answers(["Paris", "Capital: Paris", "PaRiS"], llm_cluster_fn=fake_llm)
    assert clusters == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# Unit tests: MajorityStableTermination
# ---------------------------------------------------------------------------

class _FakeMsg:
    """Minimal stand-in for BaseChatMessage subclasses used by the termination."""

    def __init__(self, source: str, content: str):
        self.source = source
        self.content = content

    def to_text(self) -> str:
        return self.content


# Make it pass isinstance(m, BaseChatMessage) by registering as a subclass-ish
# shim: patch only what we need. We avoid the isinstance check by using the
# real autogen TextMessage instead.

from autogen_agentchat.messages import TextMessage


def _tm(source: str, answer: str, confidence: float = 0.9) -> TextMessage:
    return TextMessage(source=source, content=_stance_message(answer, confidence))


@pytest.mark.asyncio
async def test_termination_converges_after_stable_unanimous_rounds():
    term = MajorityStableTermination(
        agent_ids=["a", "b", "c"], max_rounds=5, stable_for=2
    )
    for _ in range(2):
        stop = await term(
            [_tm("a", "Paris"), _tm("b", "Paris"), _tm("c", "Paris")]
        )
    assert term.terminated
    assert term.stop_reason == "converged"
    assert len(term.rounds) == 2


@pytest.mark.asyncio
async def test_termination_stops_at_max_rounds_when_never_stable():
    term = MajorityStableTermination(
        agent_ids=["a", "b", "c"], max_rounds=2, stable_for=3
    )
    await term([_tm("a", "Paris"), _tm("b", "Paris"), _tm("c", "Rome")])
    stop = await term([_tm("a", "Rome"), _tm("b", "Paris"), _tm("c", "Paris")])
    assert term.terminated
    assert term.stop_reason == "max_rounds"
    assert len(term.rounds) == 2


@pytest.mark.asyncio
async def test_termination_ignores_messages_without_stance():
    term = MajorityStableTermination(
        agent_ids=["a", "b", "c"], max_rounds=3, stable_for=1
    )
    result = await term(
        [
            TextMessage(source="a", content="Tool call summary only"),
            _tm("b", "Paris"),
        ]
    )
    assert result is None
    assert not term.terminated
    assert len(term.rounds) == 0


@pytest.mark.asyncio
async def test_termination_requires_majority_size_to_converge():
    term = MajorityStableTermination(
        agent_ids=["a", "b", "c"], max_rounds=5, stable_for=2
    )
    await term([_tm("a", "Paris"), _tm("b", "Rome"), _tm("c", "Berlin")])
    await term([_tm("a", "Paris"), _tm("b", "Rome"), _tm("c", "Berlin")])
    assert not term.terminated


# ---------------------------------------------------------------------------
# Integration tests: DebateInterface with the AutoGen team
# ---------------------------------------------------------------------------

def test_debate_unanimous_converges_at_round_2(mock_index):
    msgs = [_stance_message("Paris")] * 6
    subs = _build_subagents(mock_index, [msgs, msgs, msgs])
    di = DebateInterface(subs, max_rounds=4, stable_for=2)
    result = di.run("What is the capital of France?")

    assert result.rounds_used == 2
    assert result.stopped_reason == "converged"
    assert result.majority_answer == "Paris"
    assert set(result.majority_cluster_ids) == {"subagent_1", "subagent_2", "subagent_3"}


def test_debate_unanimous_round_1_with_stable_for_one(mock_index):
    """stable_for=1 means a single majority round is sufficient."""
    msgs = [_stance_message("Paris")] * 4
    subs = _build_subagents(mock_index, [msgs, msgs, msgs])
    di = DebateInterface(subs, max_rounds=3, stable_for=1)
    result = di.run("q")

    assert result.rounds_used == 1
    assert result.stopped_reason == "converged"
    assert result.majority_answer == "Paris"


def test_debate_stabilizes_after_split_round_1(mock_index):
    """Round 1 splits three ways; rounds 2-3 both agree unanimously on Paris
    → convergence detected at round 3 with stable_for=2."""
    a1 = [_stance_message("Paris"), _stance_message("Paris"), _stance_message("Paris")]
    a2 = [_stance_message("Rome"), _stance_message("Paris"), _stance_message("Paris")]
    a3 = [_stance_message("Berlin"), _stance_message("Paris"), _stance_message("Paris")]
    subs = _build_subagents(mock_index, [a1, a2, a3])
    di = DebateInterface(subs, max_rounds=5, stable_for=2)
    result = di.run("q")

    assert result.rounds_used == 3
    assert result.stopped_reason == "converged"
    assert result.majority_answer == "Paris"
    assert set(result.majority_cluster_ids) == {"subagent_1", "subagent_2", "subagent_3"}


def test_debate_stabilizes_with_two_out_of_three_majority(mock_index):
    """Two consecutive rounds where the same 2-of-3 agents vote Paris
    → convergence."""
    a1 = [_stance_message("Paris")] * 2
    a2 = [_stance_message("Paris")] * 2
    a3 = [_stance_message("Rome")] * 2
    subs = _build_subagents(mock_index, [a1, a2, a3])
    di = DebateInterface(subs, max_rounds=4, stable_for=2)
    result = di.run("q")

    assert result.rounds_used == 2
    assert result.stopped_reason == "converged"
    assert result.majority_answer == "Paris"
    assert set(result.majority_cluster_ids) == {"subagent_1", "subagent_2"}


def test_debate_hits_max_rounds_when_no_stable_majority(mock_index):
    a1 = [_stance_message("Paris")] * 3
    a2 = [_stance_message("Rome")] * 3
    a3 = [_stance_message("Berlin")] * 3
    subs = _build_subagents(mock_index, [a1, a2, a3])
    di = DebateInterface(subs, max_rounds=2, stable_for=2)
    result = di.run("q")

    assert result.rounds_used == 2
    assert result.stopped_reason == "max_rounds"
    assert len(result.majority_cluster_ids) == 1


# ---------------------------------------------------------------------------
# JudgeLLM integration
# ---------------------------------------------------------------------------

def test_judge_produces_runlog_with_debate_transcript(mock_index, tmp_path):
    msgs = [_stance_message("Paris", confidence=0.95, citations=["d1"])] * 6
    subs = _build_subagents(mock_index, [msgs, msgs, msgs])
    judge = JudgeLLM(
        subagents=subs,
        max_rounds=4,
        stable_for=2,
        output_dir=str(tmp_path),
    )
    log = judge.run(query="What is the capital of France?", query_id="q_test")

    assert isinstance(log, RunLog)
    assert log.attack_condition == "clean"
    assert log.trigger is None
    assert log.poison_retrieved is False

    assert isinstance(log.debate_transcript, DebateTranscript)
    assert log.debate_transcript.rounds_used == 2
    assert log.debate_transcript.stopped_reason == "converged"

    assert log.final_decision is not None
    assert log.final_decision.final_answer == "Paris"
    assert set(log.final_decision.winning_subagents) == {"subagent_1", "subagent_2", "subagent_3"}
    assert log.final_decision.final_confidence == pytest.approx(1.0)
    assert log.final_decision.harmful_action_flag is False

    assert set(log.agent_responses.keys()) == {"subagent_1", "subagent_2", "subagent_3"}
    for aid, out in log.agent_responses.items():
        assert out.agent_id == aid
        assert out.answer == "Paris"
        assert out.poison_retrieved is False

    jsonl = tmp_path / "runs.jsonl"
    assert jsonl.exists()
    with open(jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["query_id"] == "q_test"
    assert records[0]["debate_transcript"]["rounds_used"] == 2


def test_judge_propagates_trigger_into_debate_query(mock_index, tmp_path):
    msgs = [_stance_message("Paris")] * 4
    subs = _build_subagents(mock_index, [msgs, msgs, msgs])
    judge = JudgeLLM(
        subagents=subs,
        max_rounds=2,
        stable_for=1,
        output_dir=str(tmp_path),
    )
    log = judge.run(
        query="What is the capital of France?",
        query_id="q_trig",
        trigger="ACTIVATE",
        attack_condition="clean",
    )
    assert log.trigger == "ACTIVATE"
    assert log.final_decision.final_answer == "Paris"
