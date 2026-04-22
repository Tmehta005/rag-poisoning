"""
Fitness functions used to optimize adversarial triggers.

Ports the three scalar losses from AgentPoison's ``algo/trigger_optimization.py``:

- :func:`compute_variance` — spread of a batch of query embeddings.
- :func:`compute_avg_cluster_distance` — AgentPoison (AP) objective: the
  average Euclidean distance from each query embedding to the GMM-derived
  cluster centers of the clean corpus, minus a small variance penalty. The
  optimizer *maximizes* this so triggered queries land in an uncommon region
  of the retrieval space (high retrieval precision for matching poison docs).
- :func:`compute_avg_embedding_similarity` — CPA baseline: mean cosine
  similarity to every clean corpus embedding. Exposed here so the optimizer
  can swap in the baseline objective via ``--algo cpa``.

These functions are intentionally pure/stateless and operate on already
L2-normalized embeddings.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_variance(embeddings: Tensor) -> Tensor:
    """Mean L2 distance from each embedding to the batch mean.

    This matches AgentPoison's ``compute_variance`` (not strict statistical
    variance, but the quantity the paper actually minimizes).
    """
    mean = torch.mean(embeddings, dim=0, keepdim=True)
    distances = torch.norm(embeddings - mean, dim=1)
    return torch.mean(distances)


def compute_avg_cluster_distance(
    query_embeddings: Tensor,
    cluster_centers: Tensor,
    variance_weight: float = 0.1,
) -> Tensor:
    """AgentPoison (AP) fitness.

    Args:
        query_embeddings: ``(B, hidden)`` triggered-query embeddings.
        cluster_centers: ``(K, hidden)`` GMM cluster centers fit on the
            clean corpus. If a ``(1, K, hidden)`` tensor is supplied it is
            squeezed to ``(K, hidden)``.
        variance_weight: Multiplier on the variance-compactness penalty. The
            paper uses ``0.1``.

    Returns:
        A scalar tensor: ``mean_pairwise_dist - variance_weight * variance``.
        Maximize this to push triggered queries into a tight cluster far
        from every clean-corpus cluster center.
    """
    if cluster_centers.dim() == 3:
        cluster_centers = cluster_centers.squeeze(0)
    expanded = query_embeddings.unsqueeze(1)  # (B, 1, hidden)
    distances = torch.norm(expanded - cluster_centers, dim=2)  # (B, K)
    avg_per_query = torch.mean(distances, dim=1)
    overall_avg = torch.mean(avg_per_query)
    variance = compute_variance(query_embeddings)
    return overall_avg - variance_weight * variance


def compute_avg_embedding_similarity(
    query_embeddings: Tensor,
    db_embeddings: Tensor,
) -> Tensor:
    """CPA baseline: mean cosine similarity to every clean-corpus embedding.

    Inputs are expected to be L2-normalized. The optimizer *maximizes* this,
    making triggered queries close to *everything* (PoisonedRAG-style
    broadband attack).
    """
    similarities = torch.matmul(query_embeddings, db_embeddings.T)
    per_query = torch.mean(similarities, dim=1)
    return torch.mean(per_query)
