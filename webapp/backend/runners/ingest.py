"""
Thin CLI around :mod:`src.ingestion` / :mod:`src.corpus.ingest_cybersec`.

Usage (invoked as a subprocess by :mod:`webapp.backend.api.ingest`):

    python -m webapp.backend.runners.ingest \\
        --data-dir data/corpus_cybersec \\
        --persist-dir data/index_cybersec \\
        --chunk-size 384 --chunk-overlap 64 \\
        --embed-model local --variant cybersec

Emits a final ``__RESULT__ {"num_nodes": N, "persist_dir": ...}`` sentinel
line that :class:`~webapp.backend.jobs.manager.JobManager` captures as the
job result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


_INDEX_MARKERS = ("docstore.json", "index_store.json")


def _count_nodes(index) -> int:
    try:
        return len(index._vector_store._data.embedding_dict)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return len(index.docstore.docs)  # type: ignore[attr-defined]
    except Exception:
        return -1


def _is_index_dir(path: Path) -> bool:
    return path.is_dir() and any((path / m).exists() for m in _INDEX_MARKERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a LlamaIndex vector store.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--persist-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--embed-model", default="local")
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument(
        "--variant",
        choices=("auto", "generic", "cybersec"),
        default="auto",
        help="cybersec uses metadata-aware ingestion; auto infers from paths.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the persist dir before building.",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)

    index_already_built = _is_index_dir(persist_dir)

    if args.rebuild and persist_dir.exists():
        print(f"[ingest] removing existing index at {persist_dir}", flush=True)
        shutil.rmtree(persist_dir, ignore_errors=True)
        index_already_built = False

    if not data_dir.exists() and not index_already_built:
        print(
            f"[ingest] data dir not found: {data_dir} "
            f"(and no pre-built index at {persist_dir})",
            flush=True,
        )
        return 2

    if not data_dir.exists() and index_already_built:
        print(
            f"[ingest] source data dir {data_dir} is missing; "
            f"reloading pre-built index from {persist_dir}",
            flush=True,
        )

    variant = args.variant
    if variant == "auto":
        variant = "cybersec" if "cybersec" in str(data_dir).lower() else "generic"

    print(
        f"[ingest] variant={variant} data_dir={data_dir} persist_dir={persist_dir}",
        flush=True,
    )

    config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "similarity_top_k": args.similarity_top_k,
        "embed_model": args.embed_model,
    }

    if variant == "cybersec":
        from src.corpus.ingest_cybersec import ingest_cybersec_corpus

        config["data_dir"] = str(data_dir)
        config["persist_dir"] = str(persist_dir)
        index = ingest_cybersec_corpus(
            data_dir=str(data_dir),
            persist_dir=str(persist_dir),
            config=config,
        )
    else:
        from src.ingestion import ingest_corpus

        index = ingest_corpus(
            data_dir=str(data_dir),
            config=config,
            persist_dir=str(persist_dir),
        )

    num_nodes = _count_nodes(index)
    print(f"[ingest] done. nodes={num_nodes}", flush=True)
    print(
        "__RESULT__ "
        + json.dumps(
            {
                "num_nodes": num_nodes,
                "persist_dir": str(persist_dir),
                "data_dir": str(data_dir),
                "variant": variant,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
