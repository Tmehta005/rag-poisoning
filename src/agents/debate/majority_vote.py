"""
Majority-vote clustering for debate final answers.

The debate produces one stated answer per agent per round. At termination,
we need to group paraphrases into equivalence classes and pick the largest
class. Two strategies are offered:

1. Normalized string match (default): lowercase, strip punctuation, collapse
   whitespace. Two answers are in the same cluster iff their normalized forms
   are equal. This is cheap, deterministic, and easy to stub in tests.

2. Pluggable LLM cluster function: pass ``llm_cluster_fn`` to
   :func:`cluster_answers`. It receives the raw answers and must return a list
   of clusters (each a list of indices). Tests can stub this; production can
   wire it to GPT-5 to merge paraphrases.
"""

from __future__ import annotations

import re
import string
from typing import Callable, List, Optional


def _normalize(answer: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = answer.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


def cluster_answers(
    answers: List[str],
    llm_cluster_fn: Optional[Callable[[List[str]], List[List[int]]]] = None,
) -> List[List[int]]:
    """
    Group answers into equivalence classes.

    Args:
        answers: One answer per agent, in agent order.
        llm_cluster_fn: Optional callable that takes the raw answers and
            returns a list of clusters (each a list of agent indices). Used
            to merge paraphrases. If None, uses normalized string match.

    Returns:
        A list of clusters (each a list of agent indices into ``answers``),
        sorted by descending size. Ties broken by lowest-index-first.
    """
    if not answers:
        return []

    if llm_cluster_fn is not None:
        clusters = [list(c) for c in llm_cluster_fn(answers)]
    else:
        buckets: dict[str, List[int]] = {}
        for i, ans in enumerate(answers):
            key = _normalize(ans)
            buckets.setdefault(key, []).append(i)
        clusters = list(buckets.values())

    clusters.sort(key=lambda c: (-len(c), min(c) if c else 0))
    return clusters


def majority_cluster(
    answers: List[str],
    llm_cluster_fn: Optional[Callable[[List[str]], List[List[int]]]] = None,
) -> List[int]:
    """
    Return the indices of the largest cluster (the majority).

    Ties are broken deterministically by choosing the cluster whose lowest
    agent index is smallest.
    """
    clusters = cluster_answers(answers, llm_cluster_fn=llm_cluster_fn)
    if not clusters:
        return []
    return clusters[0]
