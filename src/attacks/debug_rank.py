"""
Diagnostic: quantify whether a trigger actually pulls the poison doc
into the top-k of a triggered retrieval.

For each attack-block in an artifact, this tool reports:

    clean query                 -> rank of poison chunk, cos sim
    query ⊕ trigger             -> rank of poison chunk, cos sim
    top-k ids (both runs)

It uses the *same* LlamaIndex embed model as the live retriever
(BAAI/bge-small-en-v1.5 by default), so the numbers match what the
orchestrator and debate runners actually see.

Usage::

    python -m src.attacks.debug_rank --attack-id demo_v2
    python -m src.attacks.debug_rank --attack-id demo_v2 --top-k 20

This is pure measurement; no runs are logged to ``results/``.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Tuple

from llama_index.core import Settings, VectorStoreIndex

from src.attacks.artifacts import load_attack_artifact
from src.attacks.poisoned_index import build_poisoned_index
from src.corpus.ingest_cybersec import ingest_cybersec_corpus
from src.corpus.query_loader import load_queries
from src.ingestion import load_ingestion_config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: List[float], b: List[float]) -> float:
    """Unit-norm safe cosine for already-normalized BGE vectors (and others)."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _rank_all(
    index: VectorStoreIndex, query: str, top_k: int
) -> List[Tuple[str, float]]:
    """Return ``[(node_id, score)]`` sorted by descending score, top-k."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [(n.node.node_id, float(n.score or 0.0)) for n in nodes]


def _find_rank(
    ranking: List[Tuple[str, float]], target_ids: set
) -> Optional[Tuple[int, str, float]]:
    """Return ``(rank, node_id, score)`` of the first match, 1-indexed."""
    for idx, (nid, score) in enumerate(ranking, start=1):
        if nid in target_ids:
            return idx, nid, score
    return None


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def diagnose_attack_block(
    block: dict,
    clean_index: VectorStoreIndex,
    top_k: int = 20,
    embed_model_name: str = "local",
) -> dict:
    """Run the clean vs triggered retrieval for one attack block and report."""
    query_id = block["query_id"]
    query = block["query"]
    trigger = block["attack"]["trigger"]
    poison_specs = block["attack"]["poison_docs"]

    poisoned_index, parent_poison_ids = build_poisoned_index(
        clean_index, poison_specs, embed_model=embed_model_name
    )

    # Map original doc_ids -> actual node_ids (LlamaIndex re-chunks + re-ids).
    poison_node_ids: set[str] = set()
    for nid, node in poisoned_index.docstore.docs.items():
        ref = getattr(node, "ref_doc_id", None)
        if ref in parent_poison_ids:
            poison_node_ids.add(nid)

    # Rank both query variants against the full poisoned index.
    ranking_clean = _rank_all(poisoned_index, query, top_k=top_k)
    ranking_trig = _rank_all(poisoned_index, f"{query} {trigger}", top_k=top_k)

    rank_clean = _find_rank(ranking_clean, poison_node_ids)
    rank_trig = _find_rank(ranking_trig, poison_node_ids)

    # Direct cosine: query+trigger vs the raw poison doc text, embedded once.
    embed = Settings.embed_model
    poison_text = "\n\n".join(spec["text"] for spec in poison_specs)
    v_q = embed.get_text_embedding(query)
    v_qt = embed.get_text_embedding(f"{query} {trigger}")
    v_p = embed.get_text_embedding(poison_text)

    cos_q_p = _cosine(v_q, v_p)
    cos_qt_p = _cosine(v_qt, v_p)

    return {
        "query_id": query_id,
        "query": query,
        "trigger": trigger,
        "poison_parent_ids": sorted(parent_poison_ids),
        "poison_node_ids": sorted(poison_node_ids),
        "cos_query_vs_poison_doc": cos_q_p,
        "cos_query_plus_trigger_vs_poison_doc": cos_qt_p,
        "cos_delta": cos_qt_p - cos_q_p,
        "rank_clean": rank_clean,
        "rank_triggered": rank_trig,
        "topk_clean": ranking_clean,
        "topk_triggered": ranking_trig,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def _short(nid: str) -> str:
    return nid.split("-")[0] if nid else nid


def _format_rank(rank: Optional[Tuple[int, str, float]]) -> str:
    if rank is None:
        return "not in top-k"
    pos, nid, score = rank
    return f"rank={pos}  score={score:.3f}  node={_short(nid)}"


def _format_topk(topk: List[Tuple[str, float]], poison_ids: set) -> str:
    lines = []
    for i, (nid, s) in enumerate(topk, start=1):
        mark = "  ★" if nid in poison_ids else ""
        lines.append(f"    {i:>2}. {_short(nid):<10}  score={s:.3f}{mark}")
    return "\n".join(lines)


def print_report(report: dict) -> None:
    poison_ids = set(report["poison_node_ids"])
    print("═" * 72)
    print(f"query_id: {report['query_id']}")
    print(f"query:    {report['query']}")
    print(f"trigger:  {report['trigger']!r}")
    print(f"poison chunks in index: {[_short(x) for x in report['poison_node_ids']]}")
    print("-" * 72)
    print("cos(query, poison_doc)         =", f"{report['cos_query_vs_poison_doc']:+.3f}")
    print("cos(query+trigger, poison_doc) =", f"{report['cos_query_plus_trigger_vs_poison_doc']:+.3f}")
    print(f"delta                          = {report['cos_delta']:+.3f}")
    print("-" * 72)
    print("clean query ranking:")
    print(f"  poison {_format_rank(report['rank_clean'])}")
    print("  top-k:")
    print(_format_topk(report["topk_clean"], poison_ids))
    print()
    print("triggered query ranking:")
    print(f"  poison {_format_rank(report['rank_triggered'])}")
    print("  top-k:")
    print(_format_topk(report["topk_triggered"], poison_ids))
    print("═" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose whether a trigger actually retrieves D_p."
    )
    p.add_argument("--attack-id", required=True, help="Artifact id under data/poison/")
    p.add_argument(
        "--queries-path",
        default="data/queries/sample_cybersec_queries.yaml",
        help="Base queries file (used to resolve the canonical query text).",
    )
    p.add_argument("--persist-dir", default="data/index_cybersec")
    p.add_argument("--data-dir", default="data/corpus_cybersec")
    p.add_argument("--ingestion-config", default="configs/corpus_cybersec.yaml")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--query-id",
        default=None,
        help="Optional: only diagnose one attack block by query_id.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    artifact = load_attack_artifact(args.attack_id)
    logger.info("Loaded artifact %s", artifact["path"])

    base_queries = load_queries(args.queries_path)
    by_id = {q["query_id"]: q for q in base_queries}

    try:
        ingestion_config = load_ingestion_config(args.ingestion_config)
    except FileNotFoundError:
        ingestion_config = {}
    embed_model = ingestion_config.get("embed_model", "local")

    clean_index = ingest_cybersec_corpus(
        data_dir=args.data_dir, persist_dir=args.persist_dir
    )

    for block in artifact["attack_blocks"]:
        qid = block["query_id"]
        if args.query_id and qid != args.query_id:
            continue
        # Prefer canonical query text from queries file if present.
        if qid in by_id and "query" in by_id[qid]:
            block = {**block, "query": by_id[qid]["query"]}

        report = diagnose_attack_block(
            block,
            clean_index=clean_index,
            top_k=args.top_k,
            embed_model_name=embed_model,
        )
        print_report(report)


if __name__ == "__main__":
    _main()
