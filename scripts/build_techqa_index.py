"""
Build the LlamaIndex vector store for a TechQA corpus config.

Reads any corpus config (``configs/corpus_techqa*.yaml``) for chunking +
embedding settings, then calls :func:`src.ingestion.ingest_corpus` to chunk,
embed, and persist the index. If ``persist_dir`` already exists,
``ingest_corpus`` reloads it in-place — delete the directory to force a
fresh build.

``--config`` accepts any corpus config produced in the repo's standard shape
(e.g. configs written by ``scripts/build_techqa_subset.py`` for the
TechQA-100 subsets), not just the default 24-doc smoke corpus.

Usage::

    python scripts/build_techqa_index.py
    python scripts/build_techqa_index.py --config configs/corpus_techqa.yaml
    python scripts/build_techqa_index.py --config configs/corpus_techqa_100_seed0.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _techqa_common import setup

setup()

from src.ingestion import ingest_corpus, load_ingestion_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the TechQA vector index.")
    parser.add_argument(
        "--config",
        default="configs/corpus_techqa.yaml",
        help="Path to the corpus config (default: configs/corpus_techqa.yaml).",
    )
    args = parser.parse_args(argv)

    config = load_ingestion_config(args.config)
    data_dir = config.get("data_dir", "data/corpus_techqa")
    persist_dir = config.get("persist_dir", "data/index_techqa")

    if not Path(data_dir).exists():
        raise SystemExit(
            f"Corpus directory not found: {data_dir}\n"
            f"Run `python scripts/prepare_techqa.py` first."
        )

    print(f"[build_techqa_index] data_dir={data_dir}")
    print(f"[build_techqa_index] persist_dir={persist_dir}")
    print(f"[build_techqa_index] chunk_size={config.get('chunk_size')}, "
          f"chunk_overlap={config.get('chunk_overlap')}, "
          f"embed_model={config.get('embed_model')}")

    index = ingest_corpus(data_dir, config=config, persist_dir=persist_dir)

    try:
        n_nodes = len(index.docstore.docs)
    except AttributeError:
        n_nodes = -1
    print(f"[build_techqa_index] index ready ({n_nodes} nodes) → {persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
