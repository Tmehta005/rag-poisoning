"""
Build a FinanceBench LlamaIndex vector store in a minimal Colab environment.

Upload this file plus ``requirements-financebench-colab.txt`` and your
``financebench-main`` zip/folder to Colab, then run this script to create an
index folder that can be copied back into ``data/index_financebench`` locally.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def _count_index_nodes(index: VectorStoreIndex) -> int:
    try:
        return len(index._vector_store._data.embedding_dict)  # type: ignore[attr-defined]
    except Exception:
        return len(index.docstore.docs)


def _load_pdfs_with_progress(pdf_dir: Path) -> list[Document]:
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {pdf_dir}")

    documents: list[Document] = []
    failed: list[Path] = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[embed] loading PDF {i}/{len(pdfs)}: {pdf.name}", flush=True)
        try:
            docs = SimpleDirectoryReader(input_files=[str(pdf)]).load_data()
        except Exception as exc:
            print(f"[embed] WARNING: skipping {pdf.name}: {exc}", flush=True)
            failed.append(pdf)
            continue
        documents.extend(docs)
        print(
            f"[embed] loaded {len(docs)} pages from {pdf.name}; total={len(documents)}",
            flush=True,
        )

    if failed:
        print(f"[embed] skipped {len(failed)} PDFs that failed to parse", flush=True)
    if not documents:
        raise SystemExit("No documents were loaded from the PDFs.")
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed FinanceBench PDFs.")
    parser.add_argument(
        "--input-zip",
        default=None,
        help="Optional zip containing financebench-main/.",
    )
    parser.add_argument(
        "--extract-to",
        default=".",
        help="Where to extract --input-zip, if provided.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="financebench-main/pdfs",
        help="Directory containing FinanceBench PDFs.",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/index_financebench",
        help="Output index directory to copy back into the local repo.",
    )
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete an existing persist dir before rebuilding.",
    )
    parser.add_argument(
        "--zip-output",
        default="data/index_financebench.zip",
        help="Zip file to create from the persisted index. Use '' to skip.",
    )
    args = parser.parse_args()

    if args.input_zip:
        print(f"[embed] extracting {args.input_zip} -> {args.extract_to}", flush=True)
        _extract_zip(Path(args.input_zip), Path(args.extract_to))

    pdf_dir = Path(args.pdf_dir)
    persist_dir = Path(args.persist_dir)

    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    if args.rebuild and persist_dir.exists():
        print(f"[embed] removing existing index: {persist_dir}", flush=True)
        shutil.rmtree(persist_dir)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    print(f"[embed] using device={device}", flush=True)
    print(f"[embed] loading PDFs from {pdf_dir}", flush=True)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.model_name,
        device=device,
    )
    splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    Settings.text_splitter = splitter

    documents = _load_pdfs_with_progress(pdf_dir)
    print(f"[embed] loaded {len(documents)} document pages/chunks", flush=True)

    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))

    print(f"[embed] wrote index to {persist_dir}", flush=True)
    print(f"[embed] nodes={_count_index_nodes(index)}", flush=True)

    if args.zip_output:
        zip_path = Path(args.zip_output)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        archive_base = zip_path.with_suffix("")
        shutil.make_archive(str(archive_base), "zip", root_dir=str(persist_dir))
        print(f"[embed] zipped index to {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
