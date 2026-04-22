"""Read-only endpoint that returns every config YAML so the UI can prefill forms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter

from webapp.backend.schemas import Defaults

router = APIRouter(tags=["defaults"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS = _REPO_ROOT / "configs"


def _load(name: str) -> Dict[str, Any]:
    path = _CONFIGS / name
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@router.get("/defaults", response_model=Defaults)
def get_defaults() -> Defaults:
    return Defaults(
        ingestion=_load("ingestion.yaml"),
        corpus_cybersec=_load("corpus_cybersec.yaml"),
        corpus_bio_papers=_load("corpus_bio_papers.yaml"),
        trigger_opt=_load("attack_trigger_opt.yaml"),
        attack_main=_load("attack_main_injection.yaml"),
        attack_poisonedrag=_load("attack_poisonedrag_baseline.yaml"),
        system_orchestrator=_load("system_orchestrator.yaml"),
        system_debate=_load("system_debate.yaml"),
    )
