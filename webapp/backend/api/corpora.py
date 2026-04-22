"""Corpus discovery: list sub-directories under data/ that look like corpora."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, HTTPException

from webapp.backend.schemas import Corpus

router = APIRouter(tags=["corpora"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA = _REPO_ROOT / "data"
_CONFIGS = _REPO_ROOT / "configs"

_DOC_EXTS = {".txt", ".pdf", ".md", ".docx"}
_INDEX_MARKERS = {"docstore.json", "index_store.json"}


def _count_docs(path: Path) -> tuple[int, List[str]]:
    count = 0
    exts: set[str] = set()
    try:
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in _DOC_EXTS:
                count += 1
                exts.add(p.suffix.lower())
    except (PermissionError, FileNotFoundError):
        pass
    return count, sorted(exts)


def _is_index_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / m).exists() for m in _INDEX_MARKERS)


def _suggest_persist_dir(name: str) -> str:
    if name.startswith("corpus_"):
        return f"data/index_{name[len('corpus_'):]}"
    if name == "corpus":
        return "data/index"
    return f"data/index_{name}"


def _suggest_ingestion_config(name: str) -> str | None:
    """
    Map a corpus directory name to its ingestion config if one exists.

    `corpus_<suffix>` -> `configs/corpus_<suffix>.yaml` when present, else None
    (callers fall back to `configs/ingestion.yaml`).
    """
    candidates: list[str] = []
    if name.startswith("corpus_"):
        candidates.append(f"configs/{name}.yaml")
    candidates.append(f"configs/{name}.yaml")
    for rel in candidates:
        if (_REPO_ROOT / rel).exists():
            return rel
    return None


def _corpus_from_config(cfg_path: Path) -> Corpus | None:
    """Build a Corpus entry from a `configs/corpus_*.yaml` spec."""
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    data_dir = cfg.get("data_dir")
    persist_dir = cfg.get("persist_dir")
    if not data_dir:
        return None
    name = cfg_path.stem  # e.g. "corpus_bio_papers"
    data_abs = _REPO_ROOT / data_dir
    persist_abs = _REPO_ROOT / persist_dir if persist_dir else _REPO_ROOT / _suggest_persist_dir(name)
    doc_count, exts = _count_docs(data_abs) if data_abs.exists() else (0, [])
    has_idx = _is_index_dir(persist_abs)
    if doc_count == 0 and not has_idx:
        return None
    return Corpus(
        name=name,
        data_dir=str(data_dir),
        suggested_persist_dir=str(persist_dir) if persist_dir else _suggest_persist_dir(name),
        doc_count=doc_count,
        has_index=has_idx,
        file_types=exts,
        ingestion_config=str(cfg_path.relative_to(_REPO_ROOT)),
    )


@router.get("/corpora", response_model=List[Corpus])
def list_corpora() -> List[Corpus]:
    results: dict[str, Corpus] = {}

    # 1. Corpora declared by configs/corpus_*.yaml (authoritative).
    if _CONFIGS.exists():
        for cfg_path in sorted(_CONFIGS.glob("corpus_*.yaml")):
            c = _corpus_from_config(cfg_path)
            if c:
                results[c.name] = c

    # 2. Sub-directories under data/ that look like document corpora.
    if _DATA.exists():
        for entry in sorted(_DATA.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in {"attacks", "queries", "logs", "tmp"}:
                continue
            if entry.name.startswith("index"):
                continue
            if _is_index_dir(entry):
                continue
            if entry.name in results:
                continue
            doc_count, exts = _count_docs(entry)
            persist = _suggest_persist_dir(entry.name)
            persist_abs = _REPO_ROOT / persist
            has_idx = _is_index_dir(persist_abs)
            if doc_count == 0 and not has_idx:
                continue
            results[entry.name] = Corpus(
                name=entry.name,
                data_dir=f"data/{entry.name}",
                suggested_persist_dir=persist,
                doc_count=doc_count,
                has_index=has_idx,
                file_types=exts,
                ingestion_config=_suggest_ingestion_config(entry.name),
            )

    return [results[k] for k in sorted(results)]


@router.get("/corpora/check")
def check_corpus(data_dir: str) -> dict:
    """
    Inspect an arbitrary path (relative to repo root or absolute) so the UI
    can validate the free-text override before submitting an ingest job.
    """
    p = Path(data_dir)
    if not p.is_absolute():
        p = _REPO_ROOT / data_dir
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {data_dir}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {data_dir}")
    doc_count, exts = _count_docs(p)
    return {
        "data_dir": data_dir,
        "doc_count": doc_count,
        "file_types": exts,
    }
