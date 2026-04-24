"""
Corpus ingestion and chunking using LlamaIndex.

Builds a VectorStoreIndex over a directory of documents.
Config is read from configs/ingestion.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

_EMBED_MODEL_OBJECTS: dict[str, object] = {}


def _configure_embed_model(embed_model: str) -> None:
    """Set LlamaIndex's global embed model from the config string."""
    if embed_model == "local":
        cached = _EMBED_MODEL_OBJECTS.get(embed_model)
        if cached is not None:
            Settings.embed_model = cached
            return
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as e:
            raise ImportError(
                "embed_model='local' requires llama-index-embeddings-huggingface. "
                "Run: pip install llama-index-embeddings-huggingface"
            ) from e
        model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        _EMBED_MODEL_OBJECTS[embed_model] = model
        Settings.embed_model = model
    # For "openai" or any other value, leave Settings.embed_model at its default.


def load_ingestion_config(config_path: str = "configs/ingestion.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest_corpus(
    data_dir: str,
    config: Optional[dict] = None,
    persist_dir: Optional[str] = None,
) -> VectorStoreIndex:
    """
    Load documents from data_dir, chunk them, and build a VectorStoreIndex.

    Args:
        data_dir: Directory containing source documents.
        config: Ingestion config dict (chunk_size, chunk_overlap, embed_model).
                Defaults to configs/ingestion.yaml.
        persist_dir: If provided, persist the index here and reload on next call.

    Returns:
        A LlamaIndex VectorStoreIndex ready for retrieval.
    """
    if config is None:
        config = load_ingestion_config()

    chunk_size = config.get("chunk_size", 512)
    chunk_overlap = config.get("chunk_overlap", 64)
    embed_model = config.get("embed_model", "local")

    _configure_embed_model(embed_model)

    # Reload from disk if already built
    if persist_dir and Path(persist_dir).exists():
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.text_splitter = splitter

    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])

    if persist_dir:
        index.storage_context.persist(persist_dir=persist_dir)

    return index
