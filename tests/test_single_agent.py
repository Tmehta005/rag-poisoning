"""
Tests for the single-agent agentic baseline (run_single_agent.py).

All LLM calls are stubbed so tests run without an OpenAI API key.
The index uses MockEmbedding — no HuggingFace download needed.
"""

from __future__ import annotations

import json
import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from src.agents.orchestrator import OrchestratorState
from src.agents.subagent import ExpertSubagent
from src.experiments.run_single_agent import _build_single_agent_graph
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCS = [
    Document(text="The capital of France is Paris.", doc_id="d1"),
    Document(text="Berlin is the capital of Germany.", doc_id="d2"),
    Document(text="Rome is the capital of Italy.", doc_id="d3"),
]


@pytest.fixture(scope="module")
def mock_index():
    embed_model = MockEmbedding(embed_dim=8)
    return VectorStoreIndex.from_documents(DOCS, embed_model=embed_model, show_progress=False)


def _stub_llm(answer: str = "Paris", confidence: float = 0.9) -> callable:
    def fn(prompt: str) -> str:
        return json.dumps({
            "answer": answer,
            "citations": ["d1"],
            "confidence": confidence,
            "rationale": "The document clearly states the answer.",
        })
    return fn


def _make_state(query: str = "What is the capital of France?",
                query_id: str = "q_test",
                trigger: str | None = None) -> OrchestratorState:
    return {
        "query": query,
        "query_id": query_id,
        "attack_condition": "clean",
        "trigger": trigger,
        "agent_outputs": [],
        "final_decision": None,
    }


# ---------------------------------------------------------------------------
# Graph output tests
# ---------------------------------------------------------------------------

def test_single_agent_produces_orchestrator_output(mock_index):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state())

    assert isinstance(result["final_decision"], OrchestratorOutput)


def test_single_agent_final_answer_matches_agent_answer(mock_index):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm("Paris"))
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state())

    assert result["final_decision"].final_answer == "Paris"


def test_single_agent_winning_subagents_is_single_agent(mock_index):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state())

    assert result["final_decision"].winning_subagents == ["subagent_1"]


def test_single_agent_confidence_propagated(mock_index):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm(confidence=0.85))
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state())

    assert result["final_decision"].final_confidence == pytest.approx(0.85)


def test_single_agent_harmful_action_flag_false_on_clean_run(mock_index):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state())

    assert result["final_decision"].harmful_action_flag is False


def test_single_agent_exactly_one_agent_output(mock_index):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state())

    assert len(result["agent_outputs"]) == 1
    assert result["agent_outputs"][0].agent_id == "subagent_1"


# ---------------------------------------------------------------------------
# Trigger test
# ---------------------------------------------------------------------------

def test_single_agent_trigger_propagated(mock_index):
    """Run with a trigger completes and the agent still produces a valid output."""
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    result = app.invoke(_make_state(trigger="ACTIVATE"))

    assert isinstance(result["final_decision"], OrchestratorOutput)
    assert result["final_decision"].final_answer == "Paris"


# ---------------------------------------------------------------------------
# RunLog / logging tests
# ---------------------------------------------------------------------------

def test_single_agent_runlog_has_correct_structure(mock_index, tmp_path):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    final_state = app.invoke(_make_state(query_id="q_log"))

    output = final_state["agent_outputs"][0]
    log = RunLog(
        query_id="q_log",
        attack_condition="clean",
        trigger=None,
        retrieved_doc_ids_per_agent={output.agent_id: output.retrieved_doc_ids},
        poison_retrieved=False,
        agent_responses={output.agent_id: output},
        final_decision=final_state["final_decision"],
        metrics={},
    )

    assert set(log.agent_responses.keys()) == {"subagent_1"}
    assert set(log.retrieved_doc_ids_per_agent.keys()) == {"subagent_1"}
    assert log.poison_retrieved is False
    assert log.trigger is None
    assert log.final_decision is not None
    assert log.final_decision.winning_subagents == ["subagent_1"]


def test_single_agent_runlog_emitted_to_jsonl(mock_index, tmp_path):
    agent = ExpertSubagent("subagent_1", Retriever(mock_index, top_k=2), llm_fn=_stub_llm())
    app = _build_single_agent_graph(agent)
    final_state = app.invoke(_make_state(query_id="q_emit"))

    output = final_state["agent_outputs"][0]
    log = RunLog(
        query_id="q_emit",
        attack_condition="clean",
        trigger=None,
        retrieved_doc_ids_per_agent={output.agent_id: output.retrieved_doc_ids},
        poison_retrieved=False,
        agent_responses={output.agent_id: output},
        final_decision=final_state["final_decision"],
        metrics={},
    )
    emit_run_log(log, output_dir=str(tmp_path))

    jsonl = tmp_path / "runs.jsonl"
    assert jsonl.exists()
    with open(jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["query_id"] == "q_emit"
    assert len(records[0]["agent_responses"]) == 1
    assert "subagent_1" in records[0]["agent_responses"]
