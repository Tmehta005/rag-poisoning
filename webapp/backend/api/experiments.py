"""POST /api/experiments: kick off an orchestrator or debate experiment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from webapp.backend.jobs.manager import get_manager
from webapp.backend.schemas import ExperimentRequest, JobSummary

router = APIRouter(tags=["experiments"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS = _REPO_ROOT / "configs"
_TMP_DIR = _REPO_ROOT / "webapp" / "data" / "tmp"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / path


def _write_tmp_yaml(prefix: str, payload: dict) -> str:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=f"_{prefix}.yaml", delete=False, dir=str(_TMP_DIR)
    )
    try:
        yaml.safe_dump(payload, tmp)
        return tmp.name
    finally:
        tmp.close()


@router.post("/experiments", response_model=JobSummary)
def submit_experiment(req: ExperimentRequest) -> JobSummary:
    mgr = get_manager()
    if mgr.has_running("experiment"):
        raise HTTPException(
            status_code=409, detail="an experiment is already running"
        )

    if req.system == "single" and req.mode == "attack":
        raise HTTPException(
            status_code=400,
            detail="single-agent poisoning is not wired up yet; pick system=orchestrator or debate",
        )

    if req.system == "orchestrator":
        system_cfg_name = "system_orchestrator.yaml"
    elif req.system == "debate":
        system_cfg_name = "system_debate.yaml"
    else:
        system_cfg_name = "system_orchestrator.yaml"
    base_system = _load_yaml(_CONFIGS / system_cfg_name)
    if req.model is not None:
        base_system["model"] = req.model
    if req.system == "single":
        base_system["num_subagents"] = 1
        if req.top_k is not None:
            base_system["top_k"] = req.top_k
    elif req.system == "orchestrator":
        if req.num_subagents is not None:
            base_system["num_subagents"] = req.num_subagents
        if req.top_k is not None:
            base_system["top_k"] = req.top_k
    else:
        if req.num_subagents is not None:
            base_system["num_subagents"] = req.num_subagents
        if req.top_k is not None:
            base_system["subagent_top_k"] = req.top_k
        if req.max_rounds is not None:
            base_system["max_rounds"] = req.max_rounds
        if req.stable_for is not None:
            base_system["stable_for"] = req.stable_for
    system_cfg_path = _write_tmp_yaml(f"{req.system}_{req.mode}", base_system)

    # Resolve corpus paths. Precedence: explicit req overrides > corpus_<name>.yaml
    # > legacy cybersec/generic mapping > generic ingestion.yaml.
    legacy_paths = {
        "cybersec": ("data/corpus_cybersec", "data/index_cybersec", "configs/corpus_cybersec.yaml"),
        "generic": ("data/corpus", "data/index", "configs/ingestion.yaml"),
    }
    corpus_cfg_candidate = _CONFIGS / f"{req.corpus}.yaml" if req.corpus.startswith("corpus_") else None
    if req.ingestion_config:
        ingestion_cfg_path = str(_resolve(req.ingestion_config))
    elif corpus_cfg_candidate and corpus_cfg_candidate.exists():
        ingestion_cfg_path = str(corpus_cfg_candidate)
    elif req.corpus in legacy_paths:
        ingestion_cfg_path = str(_REPO_ROOT / legacy_paths[req.corpus][2])
    else:
        ingestion_cfg_path = str(_CONFIGS / "ingestion.yaml")

    ing_cfg = _load_yaml(Path(ingestion_cfg_path))
    if req.data_dir:
        data_dir = req.data_dir
    elif ing_cfg.get("data_dir"):
        data_dir = ing_cfg["data_dir"]
    elif req.corpus in legacy_paths:
        data_dir = legacy_paths[req.corpus][0]
    else:
        data_dir = f"data/{req.corpus}"

    if req.persist_dir:
        persist_dir = req.persist_dir
    elif ing_cfg.get("persist_dir"):
        persist_dir = ing_cfg["persist_dir"]
    elif req.corpus in legacy_paths:
        persist_dir = legacy_paths[req.corpus][1]
    else:
        persist_dir = f"data/index_{req.corpus[len('corpus_'):]}" if req.corpus.startswith("corpus_") else f"data/index_{req.corpus}"

    runner = {
        ("orchestrator", "clean"): "webapp.backend.runners.run_clean_orch",
        ("orchestrator", "attack"): "src.experiments.run_attack_orch",
        ("debate", "clean"): "webapp.backend.runners.run_clean_debate",
        ("debate", "attack"): "src.experiments.run_attack_debate",
        ("single", "clean"): "webapp.backend.runners.run_clean_single_agent",
    }[(req.system, req.mode)]

    cmd = [sys.executable, "-m", runner]

    if req.mode == "attack":
        base_attack = _load_yaml(_CONFIGS / "attack_main_injection.yaml")
        base_attack["threat_model"] = req.threat_model
        base_attack["poisoned_subagent_ids"] = req.poisoned_subagent_ids
        if req.attack_id:
            base_attack.setdefault("attack_id", req.attack_id)
            base_attack["artifact_path"] = f"data/attacks/{req.attack_id}/artifact.json"
        attack_cfg_path = _write_tmp_yaml(f"{req.attack_id or 'attack'}_main", base_attack)

        if req.system == "orchestrator":
            cmd += [
                "--query-file", req.query_file,
                "--system-config", system_cfg_path,
                "--attack-config", attack_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--threat-model", req.threat_model,
            ]
            for sid in req.poisoned_subagent_ids:
                cmd += ["--poisoned-subagent-id", sid]
        else:
            cmd += [
                "--query-file", req.query_file,
                "--debate-config", system_cfg_path,
                "--attack-config", attack_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--threat-model", req.threat_model,
            ]
            for sid in req.poisoned_subagent_ids:
                cmd += ["--poisoned-subagent-id", sid]
    else:
        if req.system == "orchestrator":
            cmd += [
                "--query-file", req.query_file,
                "--system-config", system_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--data-dir", data_dir,
                "--persist-dir", persist_dir,
            ]
        elif req.system == "debate":
            cmd += [
                "--query-file", req.query_file,
                "--debate-config", system_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--data-dir", data_dir,
                "--persist-dir", persist_dir,
            ]
        else:  # single
            cmd += [
                "--query-file", req.query_file,
                "--system-config", system_cfg_path,
                "--ingestion-config", ingestion_cfg_path,
                "--data-dir", data_dir,
                "--persist-dir", persist_dir,
            ]

    params = req.model_dump()
    params["_runner"] = runner
    job = mgr.submit("experiment", cmd, params)
    return JobSummary(
        id=job.id,
        kind=job.kind,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        exit_code=job.exit_code,
        params=job.params,
        result=job.result,
        error=job.error,
    )
