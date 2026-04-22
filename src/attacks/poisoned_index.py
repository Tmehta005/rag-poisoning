"""
Build an ephemeral poisoned index from a clean index + D_p.

The clean on-disk index is never mutated; we copy the nodes into a fresh
``VectorStoreIndex`` in memory and insert the poison documents on top.
Factored out of ``src/experiments/run_attack.py`` so both the orchestrator
and debate runners can share it.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from llama_index.core import Document, VectorStoreIndex

from src.attacks.artifacts import AttackArtifact
from src.attacks.poison_doc import PoisonDocSpec, spec_as_dict
from src.corpus.ingest_cybersec import make_poison_documents
from src.ingestion import _configure_embed_model


def build_poisoned_index(
    clean_index: VectorStoreIndex,
    poison_doc_specs: Iterable[dict],
    embed_model: str = "local",
) -> Tuple[VectorStoreIndex, Set[str]]:
    """
    Return an ephemeral ``VectorStoreIndex`` = clean nodes + D_p.

    Args:
        clean_index: Pre-built clean index (not mutated).
        poison_doc_specs: Iterable of dicts with at least ``doc_id`` and
            ``text`` (matches ``make_poison_documents`` contract).
        embed_model: Which embed model to configure globally before
            building (must match the clean index's embedder).

    Returns:
        (poisoned_index, poison_doc_ids)
    """
    _configure_embed_model(embed_model)
    poison_docs: List[Document] = make_poison_documents(list(poison_doc_specs))
    poison_ids: Set[str] = {doc.doc_id for doc in poison_docs}

    poisoned = VectorStoreIndex(nodes=[], show_progress=False)
    for node in clean_index.docstore.docs.values():
        poisoned.insert_nodes([node])
    for doc in poison_docs:
        poisoned.insert(doc)

    return poisoned, poison_ids


def build_poisoned_index_from_artifact(
    clean_index: VectorStoreIndex,
    artifact: AttackArtifact,
    embed_model: str = "local",
    extra_specs: Optional[List[PoisonDocSpec]] = None,
) -> Tuple[VectorStoreIndex, Set[str]]:
    """
    Convenience: take the ``poison_doc_text`` out of an artifact and
    build the ephemeral poisoned index. ``extra_specs`` can be used for
    future multi-doc D_p configurations.

    For ``variant == "stealth-meta"`` the artifact's trigger is carried
    into the Document metadata (not the visible body) via
    ``metadata_trigger``; the visible text — already rendered by the
    optimizer pipeline — stays clean. For ``"overt"`` and
    ``"stealth-query"`` no metadata trigger is attached.
    """
    metadata_trigger = (
        artifact.trigger if artifact.variant == "stealth-meta" else None
    )
    primary = PoisonDocSpec(
        doc_id=artifact.poison_doc_id,
        text=artifact.poison_doc_text,
        metadata_trigger=metadata_trigger,
    )
    specs = [primary] + list(extra_specs or [])
    return build_poisoned_index(
        clean_index,
        [spec_as_dict(s) for s in specs],
        embed_model=embed_model,
    )
