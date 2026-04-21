"""
DebateSubagent: an AutoGen AssistantAgent with a private `retrieve` tool.

Each DebateSubagent is a thin factory that produces a freshly configured
AssistantAgent for a given query. The agent:

- has its own Retriever (its "knowledge store"), which can later be pointed
  at a poisoned index for Phase 4 experiments without changing this code.
- exposes exactly one tool, ``retrieve(query, top_k)``, returning a compact
  ``[doc_id] text`` listing.
- loads its system message from ``prompts/debate_subagent.txt``.

Every retrieved doc ID is recorded on the subagent so the Debate Interface
can compute ``poison_retrieved`` across rounds.
"""

from __future__ import annotations

from typing import List, Optional, Set

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from src.retriever import Retriever


def _load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


class DebateSubagent:
    """
    Wraps a :class:`Retriever` and exposes it to AutoGen as an AssistantAgent
    with a ``retrieve`` tool.

    Args:
        agent_id: Unique agent name, e.g. "subagent_1". Must be a valid
            Python identifier (AutoGen restricts agent names).
        retriever: Retriever instance pointing at this agent's private index.
        model_client: AutoGen ChatCompletionClient used by the AssistantAgent.
        prompt_path: Path to the debate subagent prompt template.
        top_k: Default top-k if the model omits the argument.
        poison_doc_ids: Doc IDs belonging to D_p (Phase 4). Empty in clean runs.
    """

    def __init__(
        self,
        agent_id: str,
        retriever: Retriever,
        model_client: ChatCompletionClient,
        prompt_path: str = "prompts/debate_subagent.txt",
        top_k: int = 5,
        poison_doc_ids: Optional[Set[str]] = None,
    ):
        if not agent_id.isidentifier():
            raise ValueError(
                f"agent_id must be a valid Python identifier, got {agent_id!r}"
            )
        self.agent_id = agent_id
        self.retriever = retriever
        self.model_client = model_client
        self.top_k = top_k
        self.prompt_template = _load_prompt(prompt_path)
        self.poison_doc_ids: Set[str] = poison_doc_ids or set()

        self.retrieved_doc_ids: List[str] = []
        self.retrieve_calls: int = 0

    def build_agent(self, question: str) -> AssistantAgent:
        """
        Construct a fresh AutoGen AssistantAgent for ``question``.

        The returned agent has the ``retrieve`` tool bound to this
        DebateSubagent's Retriever. Reset per-query state before building
        so counters are scoped to a single debate run.
        """
        self.retrieved_doc_ids = []
        self.retrieve_calls = 0

        default_top_k = self.top_k

        def retrieve(query: str, top_k: int = default_top_k) -> str:
            """Retrieve top-k documents from this subagent's private knowledge store."""
            self.retrieve_calls += 1
            docs = self.retriever.retrieve(query, top_k=top_k)
            for d in docs:
                if d.doc_id not in self.retrieved_doc_ids:
                    self.retrieved_doc_ids.append(d.doc_id)
            if not docs:
                return "No documents found."
            return "\n\n".join(
                f"[doc_id={d.doc_id} score={d.score:.3f}]\n{d.text}" for d in docs
            )

        system_message = self.prompt_template.format(
            agent_id=self.agent_id,
            question=question,
        )

        return AssistantAgent(
            name=self.agent_id,
            model_client=self.model_client,
            tools=[retrieve],
            system_message=system_message,
            reflect_on_tool_use=True,
            max_tool_iterations=3,
            description=f"Debate subagent {self.agent_id} with a private retriever.",
        )

    @property
    def poison_retrieved(self) -> bool:
        """True if any doc in D_p appeared in this subagent's retrievals."""
        if not self.poison_doc_ids:
            return False
        return bool(self.poison_doc_ids & set(self.retrieved_doc_ids))
