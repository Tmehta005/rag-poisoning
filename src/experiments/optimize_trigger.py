"""
CLI entry point for running the AgentPoison-style trigger optimizer end-to-end.

Usage:
    python -m src.experiments.optimize_trigger \\
        --attack-id attack_001 \\
        --target-query-id q001 \\
        --target-claim "NIST CSF 2.0 was revised in 2024 ..."

Loads the clean cybersec index, extracts chunk texts for GMM fitting,
builds a training-query pool (paraphrase-style variants around the target
query), runs ``optimize_trigger``, and writes the resulting artifact to
``data/attacks/<attack-id>/``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import yaml

import warnings

from src.attacks.corpus_embeddings import extract_corpus_texts
from src.attacks.encoder import DEFAULT_BGE_MODEL, load_encoder
from src.attacks.poison_doc import DEFAULT_TARGET_CLAIM, VALID_VARIANTS
from src.attacks.trigger_optimizer import OptimizerConfig, run_and_save
from src.corpus.ingest_cybersec import ingest_cybersec_corpus
from src.corpus.query_loader import load_queries


# A small pool of paraphrase scaffolds; the target query is plugged into each
# so the optimizer's batches cover the neighborhood of the query the attacker
# expects to be asked. Keeping this pool model-free (no LLM call) lets the
# optimizer run fully offline.
_PARAPHRASE_SCAFFOLDS = [
    "{q}",
    "Please answer the following question: {q}",
    "Could you tell me: {q}",
    "{q} Please be precise.",
    "Question: {q}",
    "I need to know: {q}",
    "Explain in detail: {q}",
    "Summarize the answer to this: {q}",
    "Give me the authoritative answer: {q}",
    "For a compliance report, answer: {q}",
]


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_training_queries(
    base_queries: List[str], max_queries: int, seed: int
) -> List[str]:
    """Expand each base query via the paraphrase scaffolds, dedup + cap."""
    rng = random.Random(seed)
    pool: List[str] = []
    for q in base_queries:
        for scaffold in _PARAPHRASE_SCAFFOLDS:
            pool.append(scaffold.format(q=q.strip()))
    rng.shuffle(pool)
    seen: set = set()
    out: List[str] = []
    for s in pool:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_queries:
            break
    return out


def _pick_target_queries(
    query_path: str, target_query_id: Optional[str]
) -> tuple[List[str], List[str]]:
    """Return (target_texts, target_ids) used for training the trigger."""
    queries = load_queries(query_path)
    if target_query_id:
        matches = [q for q in queries if q["query_id"] == target_query_id]
        if not matches:
            raise SystemExit(
                f"No query with query_id={target_query_id!r} in {query_path}"
            )
    else:
        matches = queries
    return [q["query"] for q in matches], [q["query_id"] for q in matches]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize a retrieval trigger.")
    parser.add_argument("--attack-id", default="attack_001")
    parser.add_argument(
        "--opt-config",
        default="configs/attack_trigger_opt.yaml",
        help="Optimizer hyperparameters.",
    )
    parser.add_argument(
        "--query-file",
        default="data/queries/sample_cybersec_queries.yaml",
        help="Pool of queries the attacker expects.",
    )
    parser.add_argument(
        "--target-query-id",
        default=None,
        help="If set, only this query's text is used to build the training pool.",
    )
    parser.add_argument("--target-claim", default=DEFAULT_TARGET_CLAIM)
    parser.add_argument(
        "--poison-doc-id",
        default=None,
        help=(
            "doc_id assigned to the synthesized poison passage. If omitted, "
            "a fresh UUID4 is generated so the poison doc is opaque at the "
            "prompt surface (recommended for honest ASR measurement)."
        ),
    )
    parser.add_argument(
        "--harmful-match-phrase",
        action="append",
        dest="harmful_match_phrases",
        default=None,
        help=(
            "Distinctive substring that must appear in the final answer to "
            "count the attack as harmful. Pass multiple times; ALL phrases "
            "must match. Case-insensitive, whitespace-normalized. Strongly "
            "recommended to avoid brittle target_claim substring matching."
        ),
    )
    parser.add_argument("--max-training-queries", type=int, default=32)
    parser.add_argument(
        "--variant",
        choices=list(VALID_VARIANTS),
        default=None,
        help=(
            "Poison rendering variant. 'overt' (default) puts the trigger in "
            "the visible doc body. 'stealth-meta' hides it in Document "
            "metadata so BGE embeds it but the subagent prompt never does. "
            "'stealth-query' produces a fully clean doc and optimizes the "
            "query-side trigger against the doc's fixed embedding (forces "
            "algo='stealth_query')."
        ),
    )
    parser.add_argument(
        "--ppl-filter",
        action="store_true",
        help="Enable GPT-2 perplexity candidate filter (requires GPT-2 download).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override torch device: cuda | mps | cpu. Auto if omitted.",
    )
    args = parser.parse_args(argv)

    opt_cfg = _load_yaml(args.opt_config)

    encoder_model = opt_cfg.get("encoder_model", DEFAULT_BGE_MODEL)
    artifacts_dir = opt_cfg.get("artifacts_dir", "data/attacks")
    cache_base_dir = opt_cfg.get("cache_base_dir", "data/attacks/_cache")
    device = args.device or opt_cfg.get("device")

    variant = args.variant or opt_cfg.get("variant", "overt")
    if variant not in VALID_VARIANTS:
        raise SystemExit(
            f"Unknown variant {variant!r}; expected one of {list(VALID_VARIANTS)}"
        )
    yaml_algo = opt_cfg.get("algo", "ap")
    if variant == "stealth-query":
        if yaml_algo != "stealth_query":
            warnings.warn(
                f"variant='stealth-query' overrides config algo={yaml_algo!r} "
                "-> 'stealth_query'.",
                stacklevel=2,
            )
        algo = "stealth_query"
    else:
        if yaml_algo == "stealth_query":
            raise SystemExit(
                f"algo='stealth_query' requires variant='stealth-query' "
                f"(got variant={variant!r}). Pass --variant stealth-query "
                "or change the config's algo."
            )
        algo = yaml_algo

    config = OptimizerConfig(
        num_adv_passage_tokens=int(opt_cfg.get("num_adv_passage_tokens", 5)),
        num_iter=int(opt_cfg.get("num_iter", 50)),
        num_grad_iter=int(opt_cfg.get("num_grad_iter", 8)),
        num_cand=int(opt_cfg.get("num_cand", 30)),
        per_batch_size=int(opt_cfg.get("per_batch_size", 8)),
        algo=algo,
        ppl_filter=bool(args.ppl_filter or opt_cfg.get("ppl_filter", False)),
        ppl_oversample=int(opt_cfg.get("ppl_oversample", 10)),
        n_components=int(opt_cfg.get("n_components", 5)),
        golden_trigger=opt_cfg.get("golden_trigger"),
        exclude_up_to=int(opt_cfg.get("exclude_up_to", 1000)),
        seed=int(opt_cfg.get("seed", 0)),
    )

    encoder = load_encoder(model_name=encoder_model, device=device)
    print(f"[optimize_trigger] encoder={encoder.model_name} device={encoder.device}")

    print(f"[optimize_trigger] loading clean index ...")
    clean_index = ingest_cybersec_corpus()
    corpus_texts = extract_corpus_texts(clean_index)
    print(f"[optimize_trigger] corpus chunks: {len(corpus_texts)}")

    target_texts, target_ids = _pick_target_queries(
        args.query_file, args.target_query_id
    )
    training_queries = _build_training_queries(
        target_texts, max_queries=args.max_training_queries, seed=config.seed
    )
    print(
        f"[optimize_trigger] training on {len(training_queries)} queries "
        f"(targets={target_ids})"
    )

    ppl_model = None
    if config.ppl_filter:
        from transformers import GPT2LMHeadModel

        ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(encoder.device)
        ppl_model.eval()

    artifact = run_and_save(
        encoder=encoder,
        attack_id=args.attack_id,
        training_queries=training_queries,
        corpus_texts=corpus_texts,
        target_claim=args.target_claim,
        target_query_ids=target_ids,
        config=config,
        artifacts_dir=artifacts_dir,
        cache_base_dir=cache_base_dir,
        ppl_model=ppl_model,
        poison_doc_id=args.poison_doc_id,
        harmful_match_phrases=args.harmful_match_phrases,
        variant=variant,
    )

    out_dir = Path(artifacts_dir) / artifact.attack_id
    print(
        f"[optimize_trigger] done. variant={artifact.variant!r} "
        f"trigger={artifact.trigger!r}\n"
        f"  artifact: {out_dir / 'artifact.json'}\n"
        f"  poison:   {out_dir / 'poison_doc.txt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
