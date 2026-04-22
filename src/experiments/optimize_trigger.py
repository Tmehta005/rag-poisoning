"""
CLI entrypoint for AgentPoison-style trigger optimization.

Pipeline:

    configs/attack_agentpoison.yaml  ──►  TriggerOptimizer
    data/index_cybersec/             ──►  (corpus embeddings, GMM centers)
    data/queries/*.yaml              ──►  training queries
                │
                ▼
     data/poison/<attack_id>/
        trigger.json
        poison_docs.yaml
        metrics.json

Usage:

    python -m src.experiments.optimize_trigger --attack-id q001_demo
    python -m src.experiments.optimize_trigger --mode per_query \\
        --target-query q001 --attack-id q001_demo --num-iter 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml

from src.attacks.artifacts import save_attack_artifact
from src.attacks.corpus_embeddings import embed_corpus
from src.attacks.encoder import BGEGradientEncoder
from src.attacks.poison_doc import (
    DEFAULT_POISON_BODY,
    DEFAULT_TARGET_ANSWER,
    build_attack_block,
    build_poison_doc,
)
from src.attacks.trigger_optimizer import (
    OptimizationResult,
    TriggerOptimizer,
    TriggerOptimizerConfig,
)
from src.corpus.ingest_cybersec import ingest_cybersec_corpus
from src.corpus.query_loader import load_queries

logger = logging.getLogger("optimize_trigger")


# ---------------------------------------------------------------------------
# Config loading + CLI override
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize an adversarial trigger against the retriever encoder."
    )
    p.add_argument("--config", default="configs/attack_agentpoison.yaml")
    p.add_argument("--attack-id", required=True, help="Artifact directory name under data/poison/")
    p.add_argument("--mode", choices=["universal", "per_query"])
    p.add_argument("--algo", choices=["ap", "cpa"])
    p.add_argument("--target-query", help="Query id to specialize to (per_query mode)")
    p.add_argument("--num-iter", type=int)
    p.add_argument("--num-cand", type=int)
    p.add_argument("--num-adv-passage-tokens", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--no-ppl-filter", action="store_true")
    p.add_argument("--golden-trigger", type=str, help="Optional init string for the trigger")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int)
    p.add_argument("--target-answer", type=str, help="False claim for the poison doc")
    p.add_argument("--poison-body", type=str, help="Body text for the poison doc")
    p.add_argument("--queries-path", type=str)
    p.add_argument("--persist-dir", type=str)
    p.add_argument("--data-dir", type=str)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "mode": args.mode,
        "algo": args.algo,
        "target_query_id": args.target_query,
        "num_iter": args.num_iter,
        "num_cand": args.num_cand,
        "num_adv_passage_tokens": args.num_adv_passage_tokens,
        "batch_size": args.batch_size,
        "golden_trigger": args.golden_trigger,
        "device": args.device,
        "seed": args.seed,
        "target_answer": args.target_answer,
        "poison_body": args.poison_body,
        "queries_path": args.queries_path,
        "persist_dir": args.persist_dir,
        "data_dir": args.data_dir,
    }
    for k, v in mapping.items():
        if v is not None:
            cfg[k] = v
    if args.no_ppl_filter:
        cfg["ppl_filter"] = False
    return cfg


# ---------------------------------------------------------------------------
# Training-query selection
# ---------------------------------------------------------------------------


def _select_train_queries(
    queries: list[dict],
    mode: str,
    target_query_id: Optional[str],
) -> tuple[list[str], dict]:
    """Return (list of query strings, target query dict for poison-doc wiring)."""
    if not queries:
        raise ValueError("No queries loaded -- check queries_path.")

    if mode == "per_query":
        if target_query_id is None:
            raise ValueError("per_query mode requires --target-query or target_query_id in config.")
        target = next((q for q in queries if q["query_id"] == target_query_id), None)
        if target is None:
            raise ValueError(f"target_query_id {target_query_id!r} not found in queries file.")
        return [target["query"]], target

    # universal mode: train on every query string. Pick the first as the
    # "primary target" for the poison doc (the attack block can still
    # specify a different false answer).
    train = [q["query"] for q in queries]
    primary_id = target_query_id or queries[0]["query_id"]
    primary = next(
        (q for q in queries if q["query_id"] == primary_id), queries[0]
    )
    return train, primary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_config(cfg: dict) -> TriggerOptimizerConfig:
    return TriggerOptimizerConfig(
        mode=cfg.get("mode", "universal"),
        algo=cfg.get("algo", "ap"),
        num_adv_passage_tokens=cfg.get("num_adv_passage_tokens", 10),
        num_iter=cfg.get("num_iter", 100),
        num_cand=cfg.get("num_cand", 50),
        num_grad_iter=cfg.get("num_grad_iter", 4),
        batch_size=cfg.get("batch_size", 8),
        gmm_components=cfg.get("gmm_components", 5),
        variance_weight=cfg.get("variance_weight", 0.1),
        ppl_filter=cfg.get("ppl_filter", True),
        ppl_model_name=cfg.get("ppl_model_name", "distilgpt2"),
        golden_trigger=cfg.get("golden_trigger"),
        seed=cfg.get("seed", 0),
        device=cfg.get("device", "auto"),
    )


def run_optimization(cfg: dict, attack_id: str) -> tuple[OptimizationResult, list[dict]]:
    """Run the optimizer end-to-end and return the result + attack blocks."""
    queries_path = cfg.get("queries_path", "data/queries/sample_cybersec_queries.yaml")
    persist_dir = cfg.get("persist_dir", "data/index_cybersec")
    data_dir = cfg.get("data_dir", "data/corpus_cybersec")
    corpus_cache = cfg.get("corpus_cache_dir", "data/poison/_shared")

    queries = load_queries(queries_path)
    train_queries, primary_query = _select_train_queries(
        queries, cfg.get("mode", "universal"), cfg.get("target_query_id")
    )

    logger.info("Loading retriever encoder %s", cfg.get("encoder"))
    encoder = BGEGradientEncoder(
        model_name=cfg.get("encoder", "BAAI/bge-small-en-v1.5"),
        device=cfg.get("device", "auto"),
    )

    logger.info("Building / loading corpus index at %s", persist_dir)
    index = ingest_cybersec_corpus(data_dir=data_dir, persist_dir=persist_dir)

    logger.info("Embedding corpus for GMM clustering...")
    corpus_embeddings, _ = embed_corpus(
        index=index, encoder=encoder, cache_dir=corpus_cache
    )
    logger.info("Corpus embeddings: %s", tuple(corpus_embeddings.shape))

    optimizer_cfg = _build_config(cfg)
    optimizer = TriggerOptimizer(encoder=encoder, config=optimizer_cfg)

    logger.info(
        "Starting optimization: mode=%s algo=%s num_iter=%d num_cand=%d tokens=%d",
        optimizer_cfg.mode,
        optimizer_cfg.algo,
        optimizer_cfg.num_iter,
        optimizer_cfg.num_cand,
        optimizer_cfg.num_adv_passage_tokens,
    )
    result = optimizer.optimize(
        train_queries=train_queries,
        corpus_embeddings=corpus_embeddings,
    )
    logger.info("Optimization done: trigger=%r best_score=%.4f", result.trigger_text, result.best_score)

    target_answer = cfg.get("target_answer") or DEFAULT_TARGET_ANSWER
    poison_body = cfg.get("poison_body") or DEFAULT_POISON_BODY
    num_poison = int(cfg.get("num_poison_docs", 1))
    qid = primary_query["query_id"]
    poison_docs = [
        build_poison_doc(
            trigger=result.trigger_text,
            target_answer=target_answer,
            body=poison_body,
            doc_id=f"p_{qid}_{i+1:02d}",
        )
        for i in range(num_poison)
    ]
    attack_block = build_attack_block(
        query_id=qid,
        query=primary_query["query"],
        trigger=result.trigger_text,
        target_answer=target_answer,
        poison_docs=poison_docs,
        ground_truth_answer=primary_query.get("ground_truth_answer"),
        category=primary_query.get("category"),
        source_doc=primary_query.get("source_doc"),
    )

    return result, [attack_block]


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args)

    result, attack_blocks = run_optimization(cfg, attack_id=args.attack_id)

    out_dir = save_attack_artifact(
        attack_id=args.attack_id,
        result=result,
        attack_blocks=attack_blocks,
        extra_metadata={
            "mode": cfg.get("mode"),
            "algo": cfg.get("algo"),
            "target_query_id": cfg.get("target_query_id"),
        },
    )
    logger.info("Wrote artifacts to %s", out_dir)
    print(f"\nTrigger: {result.trigger_text!r}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
