"""
Fitness functions for AgentPoison-style trigger optimization.

All three losses are ported from
``algo/trigger_optimization.py`` lines 46-141 with the same numerical
constants. The key one for the "ap" algorithm is
``compute_avg_cluster_distance``: we *maximize* mean distance of
triggered-query embeddings from GMM cluster centers (so they land in a
sparse region of the embedding space) while keeping triggered-query
variance low (so they cluster together, next to the poison doc).
"""

from __future__ import annotations

import torch


def gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian kernel between ``x`` and ``y`` with bandwidth ``sigma``."""
    beta = 1.0 / (2.0 * (sigma**2))
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-beta * dist)


def maximum_mean_discrepancy(
    x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """MMD between samples ``x`` and ``y`` using a Gaussian kernel."""
    x_kernel = gaussian_kernel_matrix(x, x, sigma)
    y_kernel = gaussian_kernel_matrix(y, y, sigma)
    xy_kernel = gaussian_kernel_matrix(x, y, sigma)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


def compute_variance(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Mean L2 distance of a batch of embeddings from their centroid.

    Used as a spread penalty so triggered queries collapse together.
    """
    mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
    distances = torch.norm(embeddings - mean_embedding, dim=1)
    return torch.mean(distances)


def compute_fitness(
    query_embedding: torch.Tensor, db_embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MMD - variance fitness (used in AgentPoison's ablations).

    Returns (score, mmd, variance). Higher score is better.
    """
    mmd = maximum_mean_discrepancy(query_embedding, db_embeddings)
    variance = compute_variance(query_embedding)
    return 40 * mmd - 0.02 * variance, mmd, variance


def compute_avg_cluster_distance(
    query_embedding: torch.Tensor, cluster_centers: torch.Tensor
) -> torch.Tensor:
    """
    AgentPoison "ap" loss.

    ``cluster_centers`` must be shaped ``[1, k, hidden]`` (expanded) or
    ``[k, hidden]`` (auto-expanded here). Returns a scalar that
    HotFlip *maximizes*: push each triggered-query embedding far from
    GMM centers while keeping the batch variance small.
    """
    if cluster_centers.dim() == 2:
        cluster_centers = cluster_centers.unsqueeze(0)

    expanded_queries = query_embedding.unsqueeze(1)
    distances = torch.norm(expanded_queries - cluster_centers, dim=2)
    avg_distances = torch.mean(distances, dim=1)
    overall_avg_distance = torch.mean(avg_distances)

    variance = compute_variance(query_embedding)
    return overall_avg_distance - 0.1 * variance


def compute_avg_embedding_similarity(
    query_embedding: torch.Tensor, db_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    CPA-style loss (``algo="cpa"`` in AgentPoison): mean cosine similarity
    between triggered-query embeddings and the database. Kept here for
    parity / ablation; the optimizer defaults to the "ap" loss above.
    """
    similarities = torch.mm(query_embedding, db_embeddings.T)
    avg_similarities = torch.mean(similarities, dim=1)
    return torch.mean(avg_similarities)


def compute_similarity_to_doc(
    query_embedding: torch.Tensor, target_doc_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Stealth-query loss.

    Pulls every triggered-query embedding toward a single fixed target
    embedding (the clean poison doc's CLS vector). Returns mean cosine
    similarity across the batch, minus a small variance penalty so
    triggered queries collapse together. Both inputs are assumed
    L2-normalized (BGE CLS output), so the dot product equals cosine
    similarity. HotFlip *maximizes* this score.

    Args:
        query_embedding: ``[B, hidden]`` triggered-query embeddings.
        target_doc_embedding: ``[hidden]`` or ``[1, hidden]`` target.
    """
    if target_doc_embedding.dim() == 1:
        target_doc_embedding = target_doc_embedding.unsqueeze(0)
    # [B, hidden] @ [hidden, 1] -> [B, 1]
    sims = torch.mm(query_embedding, target_doc_embedding.T).squeeze(-1)
    variance = compute_variance(query_embedding)
    return torch.mean(sims) - 0.1 * variance
