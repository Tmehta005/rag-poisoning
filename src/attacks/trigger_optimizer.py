"""
Main HotFlip loop ported from AgentPoison's ``algo/trigger_optimization.py``
(lines 516-689), trimmed to the "ap" algorithm and BGE encoder.

High-level:
    adv_passage_ids = [MASK] * N      # initial trigger
    for it in range(num_iter):
        # accumulate gradient of compute_avg_cluster_distance over
        # num_grad_iter random batches of training queries
        grad = 0
        for batch in batches[:num_grad_iter]:
            q_emb = encoder(batch_queries + adv_passage_ids)
            loss  = compute_avg_cluster_distance(q_emb, gmm_centers)
            loss.backward()
            grad += storage.get()[last N positions]
        pos = random.randrange(N)
        cands = hotflip_attack(grad[pos], E_matrix, increase_loss=True)
        cands = optionally filter by GPT-2 PPL
        # evaluate each candidate by running the forward loop again and
        # pick the one with highest total loss over num_grad_iter batches
        best = argmax over cands
        if loss(best) > current_loss: adv_passage_ids[pos] = best

When the loop exits, the adv passage is saved to an AttackArtifact along
with the templated poison doc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from tqdm import tqdm

from src.attacks.artifacts import AttackArtifact, save_artifact
from src.attacks.corpus_embeddings import build_or_load_corpus_cache
from src.attacks.encoder import (
    EncoderBundle,
    decode_trigger_tokens,
    encode_texts,
    forward_with_adv_suffix,
    initial_adv_passage_ids,
)
from src.attacks.fitness import (
    compute_avg_cluster_distance,
    compute_avg_embedding_similarity,
    compute_similarity_to_doc,
)
from src.attacks.hotflip import GradientStorage, candidate_filter, hotflip_attack
from src.attacks.poison_doc import VALID_VARIANTS, Variant, render_poison_doc


@dataclass
class OptimizerConfig:
    num_adv_passage_tokens: int = 5
    num_iter: int = 50
    num_grad_iter: int = 8
    num_cand: int = 30
    per_batch_size: int = 8
    # "ap"             -> avg cluster distance (AgentPoison default)
    # "cpa"            -> avg similarity to corpus (AgentPoison ablation)
    # "stealth_query"  -> avg similarity to a single fixed target doc
    #                     embedding (used by the stealth-query variant)
    algo: str = "ap"
    ppl_filter: bool = False
    ppl_oversample: int = 10    # how many candidates to sample before PPL filter
    n_components: int = 5
    golden_trigger: Optional[str] = None
    exclude_up_to: int = 1000   # skip low-id special tokens during HotFlip
    seed: int = 0


@dataclass
class OptimizationResult:
    adv_passage_ids: torch.Tensor
    trigger: str
    token_list: List[str]
    loss_history: List[float] = field(default_factory=list)


def _iter_batches(
    queries: Sequence[str], batch_size: int, rng: random.Random
) -> List[List[str]]:
    pool = list(queries)
    rng.shuffle(pool)
    return [pool[i : i + batch_size] for i in range(0, len(pool), batch_size) if pool[i : i + batch_size]]


def _loss_fn(
    algo: str,
    query_embeddings: torch.Tensor,
    cluster_centers: Optional[torch.Tensor],
    db_embeddings: Optional[torch.Tensor],
    target_doc_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if algo == "ap":
        return compute_avg_cluster_distance(query_embeddings, cluster_centers)
    if algo == "cpa":
        return compute_avg_embedding_similarity(query_embeddings, db_embeddings)
    if algo == "stealth_query":
        if target_doc_embedding is None:
            raise ValueError(
                "algo='stealth_query' requires a target_doc_embedding "
                "(pass target_doc_text to optimize_trigger)."
            )
        return compute_similarity_to_doc(query_embeddings, target_doc_embedding)
    raise ValueError(
        f"Unknown algo {algo!r} (expected 'ap', 'cpa', or 'stealth_query')"
    )


def optimize_trigger(
    encoder: EncoderBundle,
    training_queries: Sequence[str],
    corpus_texts: Sequence[str],
    config: OptimizerConfig = OptimizerConfig(),
    cache_base_dir: str = "data/attacks/_cache",
    progress: bool = True,
    ppl_model: Optional[torch.nn.Module] = None,
    on_step: Optional[Callable[[int, float, str], None]] = None,
    target_doc_text: Optional[str] = None,
) -> OptimizationResult:
    """
    Run the HotFlip optimizer and return the best adv passage found.

    Args:
        encoder: BGE encoder bundle (same one used by the retriever).
        training_queries: Seed queries the attacker expects to see
            (e.g. your evaluation queries or a development split).
        corpus_texts: Chunk texts from the clean index, used to fit
            GMM centers for the fitness loss. May be empty when
            ``config.algo == "stealth_query"`` since that mode targets a
            single fixed doc embedding rather than the corpus geometry.
        config: OptimizerConfig (defaults sized for CPU/MPS).
        ppl_model: Optional causal LM (e.g. GPT-2) for coherence filter.
        on_step: Optional callback ``(iter_idx, loss, trigger_str)``.
        target_doc_text: Required when ``config.algo == "stealth_query"``.
            The fixed poison-doc text; its CLS embedding becomes the
            target the triggered-query embeddings are pulled toward.
    """
    device = encoder.device
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    cluster_centers: Optional[torch.Tensor] = None
    db_embeddings: Optional[torch.Tensor] = None
    target_doc_embedding: Optional[torch.Tensor] = None

    if config.algo == "stealth_query":
        if not target_doc_text:
            raise ValueError(
                "algo='stealth_query' requires a non-empty target_doc_text."
            )
        target_doc_embedding = encode_texts(encoder, [target_doc_text])[0].detach()
    else:
        _, cluster_centers, _ = build_or_load_corpus_cache(
            encoder,
            list(corpus_texts),
            cache_base_dir=cache_base_dir,
            n_components=config.n_components,
        )
        if config.algo == "cpa":
            db_embeddings, _, _ = build_or_load_corpus_cache(
                encoder, list(corpus_texts), cache_base_dir=cache_base_dir
            )

    adv_passage_ids = initial_adv_passage_ids(
        encoder,
        num_adv_passage_tokens=config.num_adv_passage_tokens,
        golden_trigger=config.golden_trigger,
    )
    best_ids = adv_passage_ids.clone()

    storage = GradientStorage(
        encoder.word_embeddings, num_adv_passage_tokens=config.num_adv_passage_tokens
    )

    embedding_matrix = encoder.word_embeddings.weight
    loss_history: List[float] = []

    try:
        iterator = range(config.num_iter)
        if progress:
            iterator = tqdm(iterator, desc="trigger-opt", leave=False)

        for it_ in iterator:
            encoder.model.zero_grad(set_to_none=True)
            storage.reset()

            batches = _iter_batches(training_queries, config.per_batch_size, rng)
            batches = batches[: config.num_grad_iter]
            if not batches:
                raise ValueError("No training queries available — pass at least one.")

            grad_accum = None
            current_score = 0.0

            for batch in batches:
                q_emb = forward_with_adv_suffix(encoder, batch, adv_passage_ids)
                loss = _loss_fn(
                    config.algo,
                    q_emb,
                    cluster_centers,
                    db_embeddings,
                    target_doc_embedding=target_doc_embedding,
                )
                loss.backward()

                temp_grad = storage.get()            # [1, num_adv, hidden]
                grad_sum = temp_grad.sum(dim=0)      # [num_adv, hidden]
                if grad_accum is None:
                    grad_accum = grad_sum / max(1, len(batches))
                else:
                    grad_accum = grad_accum + grad_sum / max(1, len(batches))
                current_score += float(loss.detach().cpu().item())
                storage.reset()
                encoder.model.zero_grad(set_to_none=True)

            token_to_flip = rng.randrange(config.num_adv_passage_tokens)

            candidates = hotflip_attack(
                grad_accum[token_to_flip],
                embedding_matrix,
                increase_loss=True,
                num_candidates=(
                    config.num_cand * max(1, config.ppl_oversample)
                    if config.ppl_filter and ppl_model is not None
                    else config.num_cand
                ),
                exclude_up_to=config.exclude_up_to,
            )
            if config.ppl_filter and ppl_model is not None:
                candidates = candidate_filter(
                    candidates,
                    num_candidates=config.num_cand,
                    token_to_flip=token_to_flip,
                    adv_passage_ids=adv_passage_ids,
                    ppl_model=ppl_model,
                )

            candidate_scores = torch.zeros(len(candidates), device=device)
            with torch.no_grad():
                for batch in batches:
                    for i, cand in enumerate(candidates):
                        temp = adv_passage_ids.clone()
                        temp[:, token_to_flip] = cand
                        q_emb = forward_with_adv_suffix(encoder, batch, temp)
                        can_loss = _loss_fn(
                            config.algo,
                            q_emb,
                            cluster_centers,
                            db_embeddings,
                            target_doc_embedding=target_doc_embedding,
                        )
                        candidate_scores[i] += can_loss.detach().sum()

            best_idx = int(candidate_scores.argmax().item())
            best_cand_score = float(candidate_scores[best_idx].item())

            if best_cand_score > current_score:
                adv_passage_ids = adv_passage_ids.clone()
                adv_passage_ids[:, token_to_flip] = candidates[best_idx]
                best_ids = adv_passage_ids.clone()

            loss_history.append(current_score)

            _, trigger_str = decode_trigger_tokens(encoder, adv_passage_ids)
            if on_step is not None:
                on_step(it_, current_score, trigger_str)
            if progress:
                iterator.set_postfix(loss=f"{current_score:.3f}", trig=trigger_str[:30])
    finally:
        storage.close()

    tokens, trigger = decode_trigger_tokens(encoder, best_ids)
    return OptimizationResult(
        adv_passage_ids=best_ids,
        trigger=trigger,
        token_list=tokens,
        loss_history=loss_history,
    )


def run_and_save(
    encoder: EncoderBundle,
    attack_id: str,
    training_queries: Sequence[str],
    corpus_texts: Sequence[str],
    target_claim: str,
    target_query_ids: Sequence[str],
    config: OptimizerConfig = OptimizerConfig(),
    artifacts_dir: str = "data/attacks",
    cache_base_dir: str = "data/attacks/_cache",
    ppl_model: Optional[torch.nn.Module] = None,
    poison_doc_id: Optional[str] = None,
    harmful_match_phrases: Optional[Sequence[str]] = None,
    progress: bool = True,
    variant: Variant = "overt",
) -> AttackArtifact:
    """
    End-to-end: optimize a trigger, render the poison doc, save an
    ``AttackArtifact``. Returns the artifact.

    For ``variant="stealth-query"`` the poison doc is rendered first
    (clean body, no trigger anywhere) and its text is passed to the
    optimizer as the fixed target embedding. The optimizer must already
    be configured with ``algo="stealth_query"`` in this case — callers
    should normally go through the CLI in
    ``src/experiments/optimize_trigger.py`` which enforces this
    invariant.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of {VALID_VARIANTS}"
        )
    if variant == "stealth-query" and config.algo != "stealth_query":
        raise ValueError(
            "variant='stealth-query' requires config.algo='stealth_query' "
            f"(got {config.algo!r})."
        )
    if variant != "stealth-query" and config.algo == "stealth_query":
        raise ValueError(
            f"algo='stealth_query' is only valid with variant='stealth-query' "
            f"(got variant={variant!r})."
        )

    target_doc_text: Optional[str] = None
    if variant == "stealth-query":
        preview = render_poison_doc(
            trigger="",
            target_claim=target_claim,
            doc_id=poison_doc_id,
            variant=variant,
        )
        target_doc_text = preview.text

    result = optimize_trigger(
        encoder,
        training_queries,
        corpus_texts,
        config=config,
        cache_base_dir=cache_base_dir,
        ppl_model=ppl_model,
        progress=progress,
        target_doc_text=target_doc_text,
    )
    doc_spec = render_poison_doc(
        trigger=result.trigger,
        target_claim=target_claim,
        doc_id=poison_doc_id,
        variant=variant,
    )
    artifact = AttackArtifact(
        attack_id=attack_id,
        trigger=result.trigger,
        token_ids=result.adv_passage_ids.squeeze(0).tolist(),
        target_claim=target_claim,
        poison_doc_id=doc_spec.doc_id,
        poison_doc_text=doc_spec.text,
        encoder_model=encoder.model_name,
        num_adv_passage_tokens=config.num_adv_passage_tokens,
        target_query_ids=list(target_query_ids),
        loss_history=result.loss_history,
        harmful_match_phrases=list(harmful_match_phrases or []),
        variant=variant,
    )
    save_artifact(artifact, base_dir=artifacts_dir)
    return artifact
