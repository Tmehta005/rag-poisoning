"""
Expert subagent: retrieve + generate a structured answer.

Each subagent owns a Retriever and calls an LLM to produce a SubagentOutput.
The LLM call is injectable (llm_fn parameter) so tests can stub it without
hitting the API.

Phase 3 compatibility:
  - poison_doc_ids: set of doc IDs belonging to D_p. When any of these appear
    in the retrieved set, poison_retrieved=True is set on the output.
    In clean Phase 2 runs this set is always empty.
  - The trigger t is appended to the query before retrieval when provided.
    In clean Phase 2 runs trigger is always None.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional, Set

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    def observe(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator
    langfuse_context = None  # type: ignore[assignment]

from src.retriever import Retriever
from src.schemas import SubagentOutput

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


def _load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _parse_subagent_response(raw: str, agent_id: str, doc_ids: list[str]) -> dict:
    """
    Parse the LLM JSON response into SubagentOutput field values.
    Falls back gracefully if JSON is malformed.
    """
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned)
        return {
            "answer": str(data.get("answer", "I don't know")),
            "citations": [str(c) for c in data.get("citations", doc_ids)],
            "confidence": float(data.get("confidence", 0.5)),
            "rationale": str(data.get("rationale", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "answer": raw.strip(),
            "citations": doc_ids,
            "confidence": 0.5,
            "rationale": "Could not parse structured response.",
        }


class ExpertSubagent:
    """
    A single expert subagent in the multi-agent orchestrator.

    Retrieves top-k documents for a query, then generates a structured
    answer (with confidence and rationale) via an LLM.

    Args:
        agent_id: Unique identifier, e.g. "subagent_1".
        retriever: Retriever instance wrapping the corpus index.
            Swap this to point at a poisoned index in Phase 3.
        model: OpenAI model name for generation.
        prompt_path: Path to the subagent prompt template.
        poison_doc_ids: Doc IDs belonging to D_p (Phase 3). Empty in clean runs.
        llm_fn: Optional callable(prompt: str) -> str to override the default
            OpenAI call. Used in tests to avoid API calls.
    """

    def __init__(
        self,
        agent_id: str,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        prompt_path: str = "prompts/subagent.txt",
        poison_doc_ids: Optional[Set[str]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.agent_id = agent_id
        self.retriever = retriever
        self.model = model
        self.prompt_template = _load_prompt(prompt_path)
        self.poison_doc_ids: Set[str] = poison_doc_ids or set()
        self._llm_fn = llm_fn

        self._openai = None
        if llm_fn is None and _openai_available and os.getenv("OPENAI_API_KEY"):
            self._openai = OpenAI()

    @observe(name="subagent_run")
    def run(self, query: str, trigger: Optional[str] = None) -> SubagentOutput:
        """
        Retrieve documents and generate a structured answer.

        Args:
            query: The user question.
            trigger: Trigger string t (appended to query before retrieval).
                     None in clean Phase 2 runs.

        Returns:
            SubagentOutput with answer, citations, confidence, rationale,
            and poison_retrieved flag.
        """
        retrieval_query = f"{query} {trigger}" if trigger else query

        docs = self.retriever.retrieve(retrieval_query)
        doc_ids = [d.doc_id for d in docs]
        poisoned_hits = [d for d in doc_ids if d in self.poison_doc_ids]
        poison_retrieved = bool(poisoned_hits)

        doc_texts = "\n\n".join(
            f"[doc_id={d.doc_id}]\n{d.text}" for d in docs
        )
        prompt = self.prompt_template.format(
            documents=doc_texts, question=query
        )

        raw = self._call_llm(prompt)
        parsed = _parse_subagent_response(raw, self.agent_id, doc_ids)

        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "trigger": trigger},
                output={"answer": parsed["answer"], "confidence": parsed["confidence"]},
                metadata={"agent_id": self.agent_id, "poison_retrieved": poison_retrieved},
            )

        return SubagentOutput(
            agent_id=self.agent_id,
            answer=parsed["answer"],
            citations=parsed["citations"],
            confidence=parsed["confidence"],
            rationale=parsed["rationale"],
            poison_retrieved=poison_retrieved,
            retrieved_doc_ids=doc_ids,
            poisoned_retrieved_ids=poisoned_hits,
        )

    def _call_llm(self, prompt: str) -> str:
        if self._llm_fn is not None:
            return self._llm_fn(prompt)
        if self._openai is not None:
            response = self._openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        return json.dumps({
            "answer": "[NO_API_KEY]",
            "citations": [],
            "confidence": 0.0,
            "rationale": "No API key configured.",
        })
