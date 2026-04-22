"""
TriggerOptimizer: the HotFlip-based trigger search loop.

Ports the main ``for it_ in range(args.num_iter)`` loop from
AgentPoison's ``algo/trigger_optimization.py`` but targets this repo's
retriever encoder directly and stays embedding-only (no target-LLM
gradient guidance) per the plan.

Supported modes:

- ``universal``: given a list of training queries (e.g. a split of the
  eval queries), find one trigger that pushes every triggered query into
  an uncommon region of the corpus embedding space.
- ``per_query``: a single target query repeated across mini-batches; the
  trigger becomes specialized to that query.

Fitness:

- ``ap`` (AgentPoison): maximize avg distance from triggered embeddings to
  GMM cluster centers of clean corpus embeddings (with a variance penalty).
- ``cpa``: maximize mean cosine similarity to every clean embedding.

The loop stores per-iteration progress in :class:`OptimizationResult` so
the caller can persist metrics alongside the trigger.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from sklearn.mixture import GaussianMixture
from torch import Tensor

from src.attacks.encoder import BGEGradientEncoder, GradientStorage
from src.attacks.fitness import (
    compute_avg_cluster_distance,
    compute_avg_embedding_similarity,
)
from src.attacks.hotflip import PerplexityScorer, candidate_filter, hotflip_attack


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TriggerOptimizerConfig:
    mode: str = "universal"  # "universal" | "per_query"
    algo: str = "ap"  # "ap" | "cpa"
    num_adv_passage_tokens: int = 10
    num_iter: int = 100
    num_cand: int = 50
    num_grad_iter: int = 4  # gradient accumulation micro-batches
    batch_size: int = 8
    gmm_components: int = 5
    variance_weight: float = 0.1
    ppl_filter: bool = True
    ppl_model_name: str = "distilgpt2"
    golden_trigger: Optional[str] = None  # optional init string
    seed: int = 0
    device: Optional[str] = None  # "auto" | "cuda" | "mps" | "cpu"


@dataclass
class OptimizationResult:
    trigger_text: str
    trigger_token_ids: List[int]
    encoder_model: str
    num_iter: int
    final_score: float
    best_score: float
    loss_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------


class TriggerOptimizer:
    """Run HotFlip over BGE token embeddings to produce an adversarial trigger.

    Call :meth:`optimize` with training queries and corpus embeddings. The
    caller is responsible for loading those artifacts
    (:func:`src.attacks.corpus_embeddings.embed_corpus` and the query
    YAML loader).
    """

    def __init__(
        self,
        encoder: BGEGradientEncoder,
        config: TriggerOptimizerConfig,
        ppl_scorer: Optional[PerplexityScorer] = None,
    ):
        self.encoder = encoder
        self.config = config
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        if config.ppl_filter and ppl_scorer is None:
            self.ppl_scorer = PerplexityScorer(
                model_name=config.ppl_model_name,
                device=encoder.device,
            )
        else:
            self.ppl_scorer = ppl_scorer

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def optimize(
        self,
        train_queries: List[str],
        corpus_embeddings: Tensor,
    ) -> OptimizationResult:
        """Run the full HotFlip loop and return the best trigger found."""
        if not train_queries:
            raise ValueError("train_queries is empty")

        device = self.encoder.device
        cfg = self.config

        cluster_centers = self._fit_clusters(corpus_embeddings).to(device)
        db = corpus_embeddings.to(device)

        adv_passage_ids = self._init_trigger_ids().to(device)
        forbidden_ids = self._forbidden_token_ids().to(device)

        embeddings_module = self.encoder.get_input_embeddings()
        storage = GradientStorage(embeddings_module, cfg.num_adv_passage_tokens)

        best_ids = adv_passage_ids.clone()
        best_score = -math.inf
        loss_history: List[float] = []

        try:
            for it in range(cfg.num_iter):
                grad = self._accumulate_gradient(
                    adv_passage_ids=adv_passage_ids,
                    train_queries=train_queries,
                    cluster_centers=cluster_centers,
                    db=db,
                    storage=storage,
                )
                token_to_flip = random.randrange(cfg.num_adv_passage_tokens)

                candidates = hotflip_attack(
                    averaged_grad=grad[token_to_flip],
                    embedding_matrix=embeddings_module.weight,
                    increase_loss=True,
                    num_candidates=(
                        cfg.num_cand * 5 if cfg.ppl_filter else cfg.num_cand
                    ),
                    forbidden_ids=forbidden_ids,
                )

                if cfg.ppl_filter:
                    candidates = candidate_filter(
                        candidates=candidates,
                        num_candidates=cfg.num_cand,
                        token_to_flip=token_to_flip,
                        adv_passage_ids=adv_passage_ids,
                        ppl_scorer=self.ppl_scorer,
                    )

                current_score = self._score(
                    adv_passage_ids=adv_passage_ids,
                    queries=self._sample_batch(train_queries),
                    cluster_centers=cluster_centers,
                    db=db,
                )
                loss_history.append(float(current_score))

                best_cand_score, best_cand_id = self._rescore_candidates(
                    candidates=candidates,
                    token_to_flip=token_to_flip,
                    adv_passage_ids=adv_passage_ids,
                    train_queries=train_queries,
                    cluster_centers=cluster_centers,
                    db=db,
                )

                if best_cand_score > current_score:
                    adv_passage_ids[:, token_to_flip] = best_cand_id
                    if best_cand_score > best_score:
                        best_score = float(best_cand_score)
                        best_ids = adv_passage_ids.clone()
                else:
                    if current_score > best_score:
                        best_score = float(current_score)
                        best_ids = adv_passage_ids.clone()
        finally:
            storage.close()

        final_score = self._score(
            adv_passage_ids=best_ids,
            queries=self._sample_batch(train_queries),
            cluster_centers=cluster_centers,
            db=db,
        )

        trigger_text = self.encoder.tokens_to_string(best_ids.squeeze(0))
        return OptimizationResult(
            trigger_text=trigger_text,
            trigger_token_ids=best_ids.squeeze(0).detach().cpu().tolist(),
            encoder_model=self.encoder.model_name,
            num_iter=cfg.num_iter,
            final_score=float(final_score),
            best_score=float(best_score) if best_score != -math.inf else float(final_score),
            loss_history=loss_history,
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _fit_clusters(self, corpus_embeddings: Tensor) -> Tensor:
        n = corpus_embeddings.size(0)
        k = max(1, min(self.config.gmm_components, n))
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=self.config.seed)
        gmm.fit(corpus_embeddings.detach().cpu().numpy())
        return torch.tensor(gmm.means_, dtype=corpus_embeddings.dtype)

    def _init_trigger_ids(self) -> Tensor:
        cfg = self.config
        if cfg.golden_trigger:
            return self.encoder.string_to_token_ids(
                cfg.golden_trigger, cfg.num_adv_passage_tokens
            )
        mask_id = self.encoder.mask_token_id
        ids = [mask_id] * cfg.num_adv_passage_tokens
        return torch.tensor([ids], dtype=torch.long)

    def _forbidden_token_ids(self) -> Tensor:
        """Special tokens HotFlip must never flip into."""
        tok = self.encoder.tokenizer
        candidates = [
            tok.cls_token_id,
            tok.sep_token_id,
            tok.pad_token_id,
            tok.unk_token_id,
            tok.mask_token_id,
        ]
        forbidden = [c for c in candidates if c is not None]
        return torch.tensor(list(set(forbidden)), dtype=torch.long)

    # ------------------------------------------------------------------
    # Gradient accumulation + scoring
    # ------------------------------------------------------------------

    def _accumulate_gradient(
        self,
        adv_passage_ids: Tensor,
        train_queries: List[str],
        cluster_centers: Tensor,
        db: Tensor,
        storage: GradientStorage,
    ) -> Tensor:
        cfg = self.config
        self.encoder.model.zero_grad(set_to_none=True)
        storage.reset()

        num_steps = min(
            cfg.num_grad_iter,
            max(1, math.ceil(len(train_queries) / cfg.batch_size)),
        )
        grad_sum: Optional[Tensor] = None

        for step in range(num_steps):
            batch = self._sample_batch(train_queries)
            encoded = self.encoder.encode_with_trigger(batch, adv_passage_ids)
            loss = self._loss(encoded.embeddings, cluster_centers, db)
            # The optimizer *maximizes* fitness; PyTorch minimizes, so flip.
            (-loss).backward()

            temp_grad = storage.get()
            storage.reset()
            self.encoder.model.zero_grad(set_to_none=True)

            step_grad = temp_grad.sum(dim=0) / num_steps
            grad_sum = step_grad if grad_sum is None else grad_sum + step_grad

        assert grad_sum is not None
        return grad_sum

    @torch.no_grad()
    def _score(
        self,
        adv_passage_ids: Tensor,
        queries: List[str],
        cluster_centers: Tensor,
        db: Tensor,
    ) -> Tensor:
        encoded = self.encoder.encode_with_trigger(queries, adv_passage_ids)
        return self._loss(encoded.embeddings, cluster_centers, db)

    def _loss(
        self,
        query_embeddings: Tensor,
        cluster_centers: Tensor,
        db: Tensor,
    ) -> Tensor:
        if self.config.algo == "ap":
            return compute_avg_cluster_distance(
                query_embeddings,
                cluster_centers,
                variance_weight=self.config.variance_weight,
            )
        if self.config.algo == "cpa":
            return compute_avg_embedding_similarity(query_embeddings, db)
        raise ValueError(f"Unknown algo: {self.config.algo!r}")

    def _rescore_candidates(
        self,
        candidates: Tensor,
        token_to_flip: int,
        adv_passage_ids: Tensor,
        train_queries: List[str],
        cluster_centers: Tensor,
        db: Tensor,
    ) -> tuple[float, int]:
        if candidates.numel() == 0:
            return float("-inf"), int(adv_passage_ids[0, token_to_flip].item())

        batch = self._sample_batch(train_queries)
        best_score = -math.inf
        best_id = int(candidates[0].item())
        with torch.no_grad():
            for cand in candidates:
                temp = adv_passage_ids.clone()
                temp[:, token_to_flip] = cand
                score = self._score(temp, batch, cluster_centers, db)
                if float(score) > best_score:
                    best_score = float(score)
                    best_id = int(cand.item())
        return best_score, best_id

    def _sample_batch(self, queries: List[str]) -> List[str]:
        cfg = self.config
        if cfg.mode == "per_query":
            return [queries[0]] * cfg.batch_size
        if len(queries) <= cfg.batch_size:
            return list(queries)
        return random.sample(queries, cfg.batch_size)
