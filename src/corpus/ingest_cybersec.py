"""
Metadata-aware ingestion for cybersecurity standard documents.

Wraps the existing LlamaIndex ingestion pipeline and adds:
  - source_file, standard, title extracted from filename
  - section_id extracted from chunk text via regex
  - chunk_index per source document
  - is_poison flag (False for all corpus docs; True only for D_p in attack runs)

Writes the index to a separate persist directory so the toy corpus index
at data/index/ is never touched.

Supported input formats: .txt, .pdf, .md, .docx
(Anything SimpleDirectoryReader handles.)

Usage:
    from src.corpus.ingest_cybersec import ingest_cybersec_corpus
    index = ingest_cybersec_corpus()          # uses configs/corpus_cybersec.yaml
    index = ingest_cybersec_corpus(           # explicit paths
        data_dir="data/corpus_cybersec",
        persist_dir="data/index_cybersec",
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader

# Reuse the embed-model configurator from the main ingestion module.
from src.ingestion import _configure_embed_model


# ---------------------------------------------------------------------------
# Section/control ID patterns for common cybersecurity standards
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = [
    # NIST CSF v1/v2: PR.AC-1, DE.CM-7, GV.OC-01
    r"\b[A-Z]{2}\.[A-Z]{2}-\d+\b",
    # NIST SP 800-53: AC-1, IA-5, SC-28, SA-11(1)
    r"\b[A-Z]{1,3}-\d+(?:\(\d+\))?\b",
    # ISO 27001 / ISO 27002: A.9.1.1, 5.15, 8.24
    r"\bA\.\d+\.\d+(?:\.\d+)?\b",
    r"\b\d+\.\d{1,2}(?:\.\d{1,2})?\b",
    # CIS Controls: "Control 1.1", "Safeguard 2.3", "CIS Control 4"
    r"(?:CIS\s+)?(?:Control|Safeguard)\s+\d+(?:\.\d+)?",
]

_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS))

# ---------------------------------------------------------------------------
# Metadata-chunk filter
# ---------------------------------------------------------------------------

_XML_TAG_RE = re.compile(r"<[^>]+>")


def _is_xml_metadata_chunk(text: str, xml_ratio_threshold: float = 0.25) -> bool:
    """
    Return True if the chunk looks like embedded XMP/RDF/XML metadata from a
    PDF header rather than real document content.

    A chunk is classified as metadata when either:
    - It contains known XMP/RDF namespace markers, OR
    - More than `xml_ratio_threshold` of its characters are inside XML tags.
    """
    if any(marker in text for marker in ("<rdf:", "<?xpacket", "<x:xmpmeta", "<dc:")):
        return True
    if len(text) == 0:
        return False
    tag_chars = sum(len(m.group(0)) for m in _XML_TAG_RE.finditer(text))
    return (tag_chars / len(text)) > xml_ratio_threshold


def _extract_section_id(text: str) -> str:
    """Return the first section/control ID found in text, or empty string."""
    m = _SECTION_RE.search(text)
    return m.group(0).strip() if m else ""


def _infer_standard(filename: str) -> str:
    """Infer a short standard label from the filename."""
    name = filename.lower()
    if "800-53" in name or "sp800" in name:
        return "NIST-SP800-53"
    if "csf" in name or "cyber_security_framework" in name:
        return "NIST-CSF"
    if "iso27001" in name or "iso_27001" in name or "27001" in name:
        return "ISO-27001"
    if "iso27002" in name or "iso_27002" in name or "27002" in name:
        return "ISO-27002"
    if "cis" in name:
        return "CIS-Controls"
    if "soc2" in name or "soc_2" in name:
        return "SOC2"
    # Fall back to the stem of the filename
    return Path(filename).stem.replace("_", "-").replace(" ", "-").upper()


def _title_from_filename(filename: str) -> str:
    """Human-readable title from filename."""
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()


def load_cybersec_config(config_path: str = "configs/corpus_cybersec.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest_cybersec_corpus(
    data_dir: Optional[str] = None,
    persist_dir: Optional[str] = None,
    config: Optional[dict] = None,
    config_path: str = "configs/corpus_cybersec.yaml",
) -> VectorStoreIndex:
    """
    Load cybersecurity standard documents, chunk them, attach metadata,
    and build a VectorStoreIndex.

    Args:
        data_dir: Directory of source documents. Overrides config if provided.
        persist_dir: Where to persist the index. Overrides config if provided.
        config: Pre-loaded config dict. Loaded from config_path if None.
        config_path: Path to corpus_cybersec.yaml.

    Returns:
        VectorStoreIndex with chunk metadata attached to every node.
    """
    if config is None:
        config = load_cybersec_config(config_path)

    data_dir = data_dir or config.get("data_dir", "data/corpus_cybersec")
    persist_dir = persist_dir or config.get("persist_dir", "data/index_cybersec")
    chunk_size = config.get("chunk_size", 384)
    chunk_overlap = config.get("chunk_overlap", 64)
    embed_model = config.get("embed_model", "local")

    _configure_embed_model(embed_model)

    # Reload persisted index if it exists
    if persist_dir and Path(persist_dir).exists():
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

    # Build file-level metadata callback
    def file_metadata(filepath: str) -> dict:
        filename = Path(filepath).name
        return {
            "source_file": filename,
            "standard": _infer_standard(filename),
            "title": _title_from_filename(filename),
            "is_poison": False,
        }

    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        file_metadata=file_metadata,
        recursive=True,
    )
    documents: list[Document] = reader.load_data()

    # Chunk and attach section_id + chunk_index per source file
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.text_splitter = splitter

    # Track chunk index per source file
    chunk_counters: dict[str, int] = {}
    enriched_documents: list[Document] = []

    skipped = 0
    for doc in documents:
        if _is_xml_metadata_chunk(doc.text):
            skipped += 1
            continue
        source = doc.metadata.get("source_file", "unknown")
        idx = chunk_counters.get(source, 0)
        section_id = _extract_section_id(doc.text)
        doc.metadata["section_id"] = section_id
        doc.metadata["chunk_index"] = idx
        chunk_counters[source] = idx + 1
        enriched_documents.append(doc)

    if skipped:
        import warnings
        warnings.warn(
            f"ingest_cybersec: skipped {skipped} XML/metadata chunks from PDF headers.",
            stacklevel=2,
        )

    index = VectorStoreIndex.from_documents(
        enriched_documents, transformations=[splitter]
    )

    if persist_dir:
        index.storage_context.persist(persist_dir=persist_dir)

    return index


def make_poison_documents(poison_doc_specs: list[dict]) -> list[Document]:
    """
    Convert D_p spec dicts (from query file attack block) into LlamaIndex
    Document objects ready to be inserted into the poisoned index.

    Each spec must have: doc_id (str), text (str).
    Optional: standard, section_id.

    Args:
        poison_doc_specs: List of dicts from query['attack']['poison_docs'].

    Returns:
        List of LlamaIndex Document objects with is_poison=True in metadata.
    """
    docs = []
    for spec in poison_doc_specs:
        doc = Document(
            text=spec["text"],
            doc_id=spec["doc_id"],
            metadata={
                "source_file": "poison",
                "standard": spec.get("standard", "POISON"),
                "title": spec.get("title", "Poisoned Document"),
                "section_id": spec.get("section_id", ""),
                "chunk_index": 0,
                "is_poison": True,
            },
        )
        docs.append(doc)
    return docs
