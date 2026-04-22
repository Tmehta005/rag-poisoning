"""
HotFlip-style candidate generation + optional perplexity filter.

Mirrors the two helpers at the heart of AgentPoison's trigger optimization:

- :func:`hotflip_attack` — given an averaged gradient w.r.t. one trigger
  token's embedding, returns the top-``k`` vocabulary replacements that
  would *increase* the fitness (or decrease, via ``increase_loss=False``).
- :func:`candidate_filter` — rescores candidates by negative GPT-2
  perplexity of the resulting adv-passage so the optimizer prefers
  readable triggers.

``PerplexityScorer`` is a thin lazy-loaded wrapper around a small causal LM
(default ``distilgpt2``) so the filter is optional.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# HotFlip
# ---------------------------------------------------------------------------


def hotflip_attack(
    averaged_grad: Tensor,
    embedding_matrix: Tensor,
    increase_loss: bool = True,
    num_candidates: int = 100,
    forbidden_ids: Optional[Tensor] = None,
) -> Tensor:
    """Top-``num_candidates`` vocabulary replacements for one trigger position.

    Args:
        averaged_grad: Gradient w.r.t. the target token's embedding,
            shape ``(hidden,)``.
        embedding_matrix: The encoder's token embedding matrix, shape
            ``(vocab, hidden)``.
        increase_loss: If ``True``, return the tokens that most increase
            the fitness (we are maximizing). If ``False``, return the ones
            that most decrease it.
        num_candidates: Number of top tokens to return.
        forbidden_ids: Optional 1-D tensor of vocabulary ids that must never
            be selected (e.g. special tokens). These are masked out before
            the top-k selection.

    Returns:
        A 1-D long tensor of candidate token ids, length ``num_candidates``.
    """
    with torch.no_grad():
        scores = torch.matmul(embedding_matrix, averaged_grad)
        if not increase_loss:
            scores = -scores
        if forbidden_ids is not None and forbidden_ids.numel() > 0:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[forbidden_ids.to(scores.device)] = True
            scores = scores.masked_fill(mask, float("-inf"))
        k = min(num_candidates, scores.numel())
        _, top_ids = scores.topk(k)
    return top_ids


# ---------------------------------------------------------------------------
# Perplexity filter (distilgpt2 by default)
# ---------------------------------------------------------------------------


class PerplexityScorer:
    """Scores adv-passage candidates by negative LM perplexity.

    Lazy-loads a causal LM only when instantiated so unit tests that never
    enable the ppl filter don't pay the download/load cost.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[torch.device | str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, token_ids: Tensor) -> Tensor:
        """Return -perplexity for each sequence in ``token_ids`` (shape ``(B, L)``)."""
        ids = token_ids.to(self.device)
        # Drop columns whose id exceeds the scoring LM's vocab; keeps
        # cross-encoder vocab mismatches from crashing the forward pass.
        vocab_size = self.model.get_input_embeddings().num_embeddings
        clamped = ids.clamp(max=vocab_size - 1)
        outputs = self.model(clamped, labels=clamped)
        loss = outputs.loss
        ppl = torch.exp(loss)
        return -ppl.expand(ids.size(0))


def candidate_filter(
    candidates: Tensor,
    num_candidates: int,
    token_to_flip: int,
    adv_passage_ids: Tensor,
    ppl_scorer: Optional[PerplexityScorer],
) -> Tensor:
    """Keep the top-``num_candidates`` of ``candidates`` by perplexity score.

    Args:
        candidates: 1-D tensor of token ids, length ``>= num_candidates``.
        num_candidates: How many to keep.
        token_to_flip: Position in the trigger being replaced.
        adv_passage_ids: Current adv-passage ids, shape ``(1, num_adv_tokens)``.
        ppl_scorer: :class:`PerplexityScorer` or ``None``. When ``None``,
            the first ``num_candidates`` of ``candidates`` are returned
            unchanged (no-op filter).

    Returns:
        A 1-D long tensor of the selected candidate ids.
    """
    if ppl_scorer is None or num_candidates >= candidates.numel():
        return candidates[:num_candidates]

    scores: list[float] = []
    for cand in candidates:
        temp = adv_passage_ids.clone()
        temp[:, token_to_flip] = cand
        score = ppl_scorer.score(temp).squeeze().item()
        scores.append(score)
    scores_t = torch.tensor(scores)
    k = min(num_candidates, scores_t.numel())
    _, top_idx = scores_t.topk(k)
    return candidates[top_idx]
