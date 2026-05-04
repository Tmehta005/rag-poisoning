"""
Inspect FinanceBench retrieval before running expensive LLM baselines.

This script does not call OpenAI. It loads the persisted FinanceBench index,
runs retrieval for selected queries, prints top-k chunks, and writes a JSONL
debug artifact that can be downloaded from Colab.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any
from zipimport import zipimporter


# Colab's Python 3.12 image can expose an old pkg_resources before upgraded
# setuptools is visible. Keep this aligned with the main runner.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = zipimporter  # type: ignore[attr-defined]

_ORIG_FIND_SPEC = importlib.util.find_spec


def _colab_safe_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "google.colab":
        return None
    return _ORIG_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _colab_safe_find_spec

import yaml
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.corpus.query_loader import load_queries
from src.retriever import Retriever


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _configure_embedding_model(
    model_name: str,
    device: str,
    embed_batch_size: int,
) -> None:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        embed_batch_size=embed_batch_size,
    )


def _load_index(
    *,
    persist_dir: str,
    embed_model_name: str,
    embed_batch_size: int,
):
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise SystemExit(
            f"Index directory not found: {persist_path}. "
            "Unzip index_financebench so it lands at data/index_financebench/."
        )

    device = _detect_device()
    print(f"[retrieval-debug] embedding device={device}", flush=True)
    _configure_embedding_model(embed_model_name, device, embed_batch_size)

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
    return load_index_from_storage(storage_context)


def _select_queries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.query_text:
        return [
            {
                "query_id": args.query_id or "adhoc_query",
                "query": args.query_text,
                "ground_truth_answer": "",
                "category": "adhoc",
                "company": "",
                "doc_name": "",
            }
        ]

    queries = load_queries(args.query_file)
    if args.query_id:
        queries = [q for q in queries if q["query_id"] == args.query_id]
        if not queries:
            raise SystemExit(f"Query ID not found: {args.query_id}")
    if args.limit is not None:
        queries = queries[: args.limit]
    return queries


def _build_doc_name_filter(query_record: dict[str, Any], args: argparse.Namespace) -> Any:
    if not args.filter_by_doc_name:
        return None
    doc_name = str(query_record.get("doc_name") or "").strip()
    if not doc_name:
        return None
    value = doc_name
    if args.doc_name_metadata_suffix and not value.endswith(args.doc_name_metadata_suffix):
        value = f"{value}{args.doc_name_metadata_suffix}"
    try:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
    except Exception as exc:
        print(
            f"[retrieval-debug] WARNING: metadata filtering unavailable ({exc}); "
            "falling back to unfiltered retrieval.",
            flush=True,
        )
        return None
    return MetadataFilters(
        filters=[
            ExactMatchFilter(key=args.doc_name_metadata_key, value=value),
        ]
    )


def _financebench_query_expansions(
    query_record: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    if not args.financebench_query_expansions:
        return []
    question = str(query_record.get("query") or "").lower()
    expansions: list[str] = []
    if (
        "capital expenditure" in question
        or "capex" in question
        or "capital spending" in question
    ):
        expansions.append(
            "Cash Flows from Investing Activities Purchases of property, "
            "plant and equipment PP&E capital spending capital expenditures"
        )
    if "net ppne" in question or "net pp&e" in question:
        expansions.append(
            "Consolidated Balance Sheet Property, plant and equipment net "
            "accumulated depreciation"
        )
    return expansions


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "..."


def run(args: argparse.Namespace) -> int:
    corpus_cfg = _load_yaml(args.corpus_config)
    persist_dir = args.persist_dir or corpus_cfg.get(
        "persist_dir",
        "data/index_financebench",
    )
    top_k = args.top_k or int(corpus_cfg.get("similarity_top_k", 5))

    index = _load_index(
        persist_dir=persist_dir,
        embed_model_name=args.embed_model_name,
        embed_batch_size=args.embed_batch_size,
    )
    queries = _select_queries(args)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for query_num, query_record in enumerate(queries, 1):
            query_id = query_record["query_id"]
            question = query_record["query"]
            metadata_filters = _build_doc_name_filter(query_record, args)
            query_expansions = _financebench_query_expansions(query_record, args)
            retriever = Retriever(
                index,
                top_k=top_k,
                metadata_filters=metadata_filters,
                query_expansions=query_expansions,
                expand_page_context=args.expand_page_context,
                page_window=args.page_window,
            )
            docs = retriever.retrieve(question, top_k=top_k)

            print("=" * 100)
            print(
                f"[retrieval-debug] {query_num}/{len(queries)} {query_id} "
                f"category={query_record.get('category', '')} "
                f"company={query_record.get('company', '')}",
                flush=True,
            )
            print(f"question: {question}")
            print(f"ground_truth: {query_record.get('ground_truth_answer', '')}")
            if args.filter_by_doc_name and query_record.get("doc_name"):
                print(
                    "metadata_filter: "
                    f"{args.doc_name_metadata_key}="
                    f"{query_record['doc_name']}{args.doc_name_metadata_suffix}",
                )
            if query_expansions:
                print(f"query_expansions: {query_expansions}")
            print("-" * 100)

            record = {
                "query_id": query_id,
                "query": question,
                "ground_truth_answer": query_record.get("ground_truth_answer", ""),
                "category": query_record.get("category", ""),
                "company": query_record.get("company", ""),
                "doc_name": query_record.get("doc_name", ""),
                "top_k": top_k,
                "query_expansions": query_expansions,
                "expand_page_context": args.expand_page_context,
                "page_window": args.page_window,
                "retrieved_docs": [],
            }

            for rank, doc in enumerate(docs, 1):
                snippet = _shorten(doc.text, args.print_chars)
                print(f"[{rank}] score={doc.score:.4f} doc_id={doc.doc_id}")
                if doc.metadata:
                    print(f"metadata: {doc.metadata}")
                print(snippet)
                print("-" * 100)
                record["retrieved_docs"].append(
                    {
                        "rank": rank,
                        "score": doc.score,
                        "doc_id": doc.doc_id,
                        "metadata": doc.metadata,
                        "text": doc.text,
                        "snippet": snippet,
                    }
                )

            f.write(json.dumps(record) + "\n")

    print(f"[retrieval-debug] wrote {output_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save top retrieved FinanceBench chunks."
    )
    parser.add_argument("--query-file", default="data/queries/financebench_queries.yaml")
    parser.add_argument("--corpus-config", default="configs/corpus_financebench.yaml")
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--filter-by-doc-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict retrieval to the query's FinanceBench doc_name metadata.",
    )
    parser.add_argument(
        "--doc-name-metadata-key",
        default="file_name",
        help="Index metadata key that stores the PDF filename.",
    )
    parser.add_argument(
        "--doc-name-metadata-suffix",
        default=".pdf",
        help="Suffix appended to query doc_name for metadata filtering.",
    )
    parser.add_argument(
        "--financebench-query-expansions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add lightweight FinanceBench keyword retrieval queries for known metric patterns.",
    )
    parser.add_argument(
        "--expand-page-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include all chunks from pages hit by retrieval to avoid table splits.",
    )
    parser.add_argument(
        "--page-window",
        type=int,
        default=0,
        help="Neighbor page window for --expand-page-context. 0 means same page only.",
    )
    parser.add_argument("--embed-model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--print-chars", type=int, default=1200)
    parser.add_argument(
        "--output-file",
        default="results/financebench_clean/retrieval_debug.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
