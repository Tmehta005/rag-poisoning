"""
Build an ephemeral poisoned ``VectorStoreIndex`` for a single experiment run.

Factored out of :mod:`src.experiments.run_attack` so both the orchestrator
and debate runners share the same implementation. The clean index on disk
is never modified.
"""

from __future__ import annotations

from typing import Iterable

from llama_index.core import VectorStoreIndex

from src.corpus.ingest_cybersec import make_poison_documents
from src.ingestion import _configure_embed_model


def build_poisoned_index(
    clean_index: VectorStoreIndex,
    poison_doc_specs: Iterable[dict],
    embed_model: str = "local",
) -> tuple[VectorStoreIndex, set[str]]:
    """Clone ``clean_index`` in memory and insert D_p alongside its nodes.

    Args:
        clean_index: Pre-built clean VectorStoreIndex.
        poison_doc_specs: Iterable of poison-doc dicts (``doc_id``, ``text``,
            ...). Same format as ``query['attack']['poison_docs']``.
        embed_model: Embed-model identifier for LlamaIndex's ``Settings``.
            Matches the clean index's configuration so poison-doc
            embeddings live in the same space.

    Returns:
        ``(poisoned_index, poison_node_ids)``. ``poison_node_ids`` is the
        set of *chunk-level* ``node_id``s that the retriever will surface
        for D_p, not the parent doc_ids. We resolve this via each node's
        ``ref_doc_id`` because ``VectorStoreIndex.insert(doc)`` runs the
        sentence splitter and assigns fresh UUID node_ids that are what
        actually come back from ``retriever.retrieve(...)``. Comparing
        parent ``doc.doc_id`` against those UUIDs would always miss.
    """
    _configure_embed_model(embed_model)
    poison_docs = make_poison_documents(list(poison_doc_specs))
    parent_doc_ids = {doc.doc_id for doc in poison_docs}

    poisoned_index = VectorStoreIndex(nodes=[], show_progress=False)
    for node in clean_index.docstore.docs.values():
        poisoned_index.insert_nodes([node])
    for doc in poison_docs:
        poisoned_index.insert(doc)

    poison_node_ids: set[str] = set()
    for node_id, node in poisoned_index.docstore.docs.items():
        if getattr(node, "ref_doc_id", None) in parent_doc_ids:
            poison_node_ids.add(node_id)

    # Defensive fallback: if the ref mapping is missing (e.g., insert_nodes
    # was used directly elsewhere), fall back to parent ids so we don't
    # silently report an empty set when there *is* a poison doc.
    if not poison_node_ids:
        poison_node_ids = parent_doc_ids

    return poisoned_index, poison_node_ids
