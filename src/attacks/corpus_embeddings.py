"""
Corpus-wide embedding utilities for AgentPoison-style trigger optimization.

Batch-encodes every node in a persisted LlamaIndex ``VectorStoreIndex`` with
the same encoder the retriever uses, then caches the result so repeated
optimizer runs don't pay the embedding cost twice.

Cache layout (under ``data/poison/_shared/<encoder_slug>/``):

    embeddings.pt   # torch.Tensor, shape (N, hidden), L2-normalized
    node_ids.json   # list[str], parallel to embeddings rows
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from src.attacks.encoder import BGEGradientEncoder


def _encoder_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def iter_nodes(index: VectorStoreIndex) -> Iterable[BaseNode]:
    """Yield every node currently stored in ``index.docstore``."""
    return index.docstore.docs.values()


def embed_corpus(
    index: VectorStoreIndex,
    encoder: BGEGradientEncoder,
    cache_dir: Optional[str | Path] = None,
    batch_size: int = 32,
) -> tuple[torch.Tensor, list[str]]:
    """Embed every node in ``index``.

    Args:
        index: Loaded ``VectorStoreIndex`` (clean corpus).
        encoder: Gradient-aware encoder; its ``encode`` method is used in
            no-grad mode here.
        cache_dir: If provided, look for / write ``embeddings.pt`` +
            ``node_ids.json`` under ``cache_dir/<encoder_slug>/``.
        batch_size: Batch size passed to ``encoder.encode``.

    Returns:
        ``(embeddings, node_ids)`` where ``embeddings`` has shape
        ``(N, hidden)`` and is L2-normalized.
    """
    if cache_dir is not None:
        cache_root = Path(cache_dir) / _encoder_slug(encoder.model_name)
        emb_path = cache_root / "embeddings.pt"
        ids_path = cache_root / "node_ids.json"
        if emb_path.exists() and ids_path.exists():
            embeddings = torch.load(emb_path, map_location="cpu")
            node_ids = json.loads(ids_path.read_text())
            if embeddings.size(0) == len(node_ids):
                return embeddings, node_ids

    node_ids: list[str] = []
    texts: list[str] = []
    for node in iter_nodes(index):
        text = node.get_content()
        if not text or not text.strip():
            continue
        node_ids.append(node.node_id)
        texts.append(text)

    if not texts:
        raise ValueError("No non-empty nodes found in the provided index.")

    embeddings = encoder.encode(texts, batch_size=batch_size)

    if cache_dir is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, emb_path)
        ids_path.write_text(json.dumps(node_ids))

    return embeddings, node_ids
