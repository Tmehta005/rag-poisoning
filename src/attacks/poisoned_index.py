"""
Build an ephemeral poisoned index from a clean index + D_p.

The clean on-disk index is never mutated; we copy the nodes into a fresh
``VectorStoreIndex`` in memory and insert the poison documents on top.
Factored out of ``src/experiments/run_attack.py`` so both the orchestrator
and debate runners can share it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, List, Optional, Set, Tuple

from llama_index.core import Document, VectorStoreIndex

from src.attacks.artifacts import AttackArtifact
from src.attacks.poison_doc import PoisonDocSpec, spec_as_dict
from src.corpus.ingest_cybersec import make_poison_documents
from src.ingestion import _configure_embed_model

_POISONED_INDEX_CACHE: dict[tuple[int, str, str], tuple[VectorStoreIndex, frozenset[str]]] = {}


def _specs_fingerprint(artifact: AttackArtifact, specs: list[PoisonDocSpec]) -> str:
    """
    Build a stable fingerprint for poisoned-index inputs.

    Includes the artifact payload and every poison spec so cache hits are valid
    only when the resulting poisoned corpus would be identical.
    """
    payload = {
        "attack_id": artifact.attack_id,
        "trigger": artifact.trigger,
        "target_claim": artifact.target_claim,
        "poison_doc_id": artifact.poison_doc_id,
        "poison_doc_text": artifact.poison_doc_text,
        "specs": [spec_as_dict(s) for s in specs],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def clear_poisoned_index_cache() -> None:
    """Clear cached poisoned indices (mainly for tests or long-lived workers)."""
    _POISONED_INDEX_CACHE.clear()


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
    use_cache: bool = True,
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
    if use_cache:
        cache_key = (id(clean_index), embed_model, _specs_fingerprint(artifact, specs))
        cached = _POISONED_INDEX_CACHE.get(cache_key)
        if cached is not None:
            cached_index, cached_ids = cached
            # Return a fresh mutable set per caller; the index itself is read-only here.
            return cached_index, set(cached_ids)

    poisoned_index, poison_ids = build_poisoned_index(
        clean_index,
        [spec_as_dict(s) for s in specs],
        embed_model=embed_model,
    )
    if use_cache:
        _POISONED_INDEX_CACHE[cache_key] = (poisoned_index, frozenset(poison_ids))
    return poisoned_index, poison_ids
