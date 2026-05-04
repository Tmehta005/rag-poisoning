"""
Retrieval wrapper around a LlamaIndex VectorStoreIndex.

Returns typed RetrievedDoc objects so the rest of the pipeline
never touches LlamaIndex internals directly.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from llama_index.core import VectorStoreIndex

from src.schemas import RetrievedDoc


class Retriever:
    """
    Thin wrapper over a LlamaIndex VectorStoreIndex.

    Exposes a single retrieve() method that returns List[RetrievedDoc].
    The poisoned retrieval path (for attacked subagents) is handled in
    src/attacks/ by injecting D_p into the index before calling retrieve().
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 5,
        metadata_filters: Optional[Any] = None,
        query_expansions: Optional[List[str]] = None,
        expand_page_context: bool = False,
        page_window: int = 0,
    ):
        self.index = index
        self.top_k = top_k
        self.metadata_filters = metadata_filters
        self.query_expansions = query_expansions or []
        self.expand_page_context = expand_page_context
        self.page_window = max(0, page_window)
        self._retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=metadata_filters,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
        """
        Retrieve the top-k most relevant documents for query.

        Args:
            query: The query string (may include trigger t for attacked agents).
            top_k: Override the instance-level top_k for this call.

        Returns:
            List of RetrievedDoc sorted by descending score.
        """
        if top_k is not None and top_k != self.top_k:
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=self.metadata_filters,
            )
        else:
            retriever = self._retriever

        nodes = list(retriever.retrieve(query))
        for expansion in self.query_expansions:
            nodes.extend(retriever.retrieve(expansion))

        unique_nodes = self._dedupe_nodes(nodes)
        if self.expand_page_context:
            unique_nodes = self._expand_with_page_context(unique_nodes)

        return [self._node_to_doc(node) for node in unique_nodes]

    def _dedupe_nodes(self, nodes: Iterable[Any]) -> List[Any]:
        seen: set[str] = set()
        unique: List[Any] = []
        for node in nodes:
            node_id = self._node_key(node.node)
            if node_id in seen:
                continue
            seen.add(node_id)
            unique.append(node)
        return unique

    def _expand_with_page_context(self, nodes: List[Any]) -> List[Any]:
        pages: set[tuple[str, int]] = set()
        for scored_node in nodes:
            metadata = dict(getattr(scored_node.node, "metadata", {}) or {})
            file_name = str(metadata.get("file_name") or "")
            page = self._parse_page_label(metadata.get("page_label"))
            if file_name and page is not None:
                for offset in range(-self.page_window, self.page_window + 1):
                    if page + offset > 0:
                        pages.add((file_name, page + offset))
        if not pages:
            return nodes

        seen = {self._node_key(node.node) for node in nodes}
        expanded = list(nodes)
        for candidate in self._iter_docstore_nodes():
            node_id = self._node_key(candidate)
            if node_id in seen:
                continue
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            file_name = str(metadata.get("file_name") or "")
            page = self._parse_page_label(metadata.get("page_label"))
            if page is None or (file_name, page) not in pages:
                continue
            expanded.append(_ScoredNodeShim(candidate, score=0.0))
            seen.add(node_id)
        return expanded

    def _iter_docstore_nodes(self) -> Iterable[Any]:
        docs = getattr(getattr(self.index, "docstore", None), "docs", {})
        if isinstance(docs, dict):
            return docs.values()
        return []

    @staticmethod
    def _parse_page_label(value: Any) -> Optional[int]:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _node_key(node: Any) -> str:
        return str(getattr(node, "node_id", None) or id(node))

    @staticmethod
    def _node_to_doc(node: Any) -> RetrievedDoc:
        # Prefer the source Document's doc_id (ref_doc_id) so poison detection
        # against poison_doc_ids works even when LlamaIndex assigns chunk-level
        # UUIDs as node_id. Fall back to node_id.
        doc_id = getattr(node.node, "ref_doc_id", None) or node.node.node_id
        metadata = dict(getattr(node.node, "metadata", {}) or {})
        return RetrievedDoc(
            doc_id=doc_id,
            text=node.node.get_content(),
            score=float(node.score) if node.score is not None else 0.0,
            metadata=metadata,
        )


class _ScoredNodeShim:
    def __init__(self, node: Any, score: float):
        self.node = node
        self.score = score
