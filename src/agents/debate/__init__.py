"""Debate setup: Judge LLM + Debate Interface + DebateSubagents.

Public entry point: :class:`JudgeLLM` in :mod:`src.agents.debate.judge`.
"""

from src.agents.debate.debate_interface import DebateInterface, DebateResult
from src.agents.debate.debate_subagent import DebateSubagent
from src.agents.debate.judge import JudgeLLM
from src.agents.debate.majority_vote import cluster_answers, majority_cluster

__all__ = [
    "DebateInterface",
    "DebateResult",
    "DebateSubagent",
    "JudgeLLM",
    "cluster_answers",
    "majority_cluster",
]
