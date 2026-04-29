"""
Build an ephemeral poisoned index from a clean index + D_p.

The clean on-disk index is never mutated; we copy the nodes into a fresh
``VectorStoreIndex`` in memory and insert the poison documents on top.
Used by every attack runner (single-agent, orchestrator, debate).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from llama_index.core import Document, VectorStoreIndex

from src.attacks.artifacts import AttackArtifact
from src.attacks.poison_doc import PoisonDocSpec, spec_as_dict
from src.ingestion import _configure_embed_model


def make_poison_documents(poison_doc_specs: list[dict]) -> list[Document]:
    """
    Convert D_p spec dicts into LlamaIndex ``Document`` objects ready to
    be inserted into the poisoned index.

    Each spec must include ``doc_id`` and ``text``. Optional keys — ``standard``,
    ``section_id``, ``title``, ``source_file`` — pass through to metadata; any
    omitted keys default to empty strings so the resulting Document carries no
    corpus-specific defaults.
    """
    docs: list[Document] = []
    for spec in poison_doc_specs:
        standard = spec.get("standard", "")
        title = spec.get("title", "")
        section_id = spec.get("section_id", "")
        source_file = spec.get(
            "source_file",
            f"{standard}-{section_id or 'POISON'}.txt" if standard else "poison.txt",
        )
        docs.append(
            Document(
                text=spec["text"],
                doc_id=spec["doc_id"],
                metadata={
                    "source_file": source_file,
                    "standard": standard,
                    "title": title,
                    "section_id": section_id,
                    "chunk_index": 0,
                    "is_poison": True,
                },
            )
        )
    return docs


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
    """
    primary = PoisonDocSpec(
        doc_id=artifact.poison_doc_id,
        text=artifact.poison_doc_text,
    )
    specs = [primary] + list(extra_specs or [])
    return build_poisoned_index(
        clean_index,
        [spec_as_dict(s) for s in specs],
        embed_model=embed_model,
    )
