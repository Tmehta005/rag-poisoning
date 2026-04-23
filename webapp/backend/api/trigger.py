"""POST /api/trigger: run the AgentPoison-style trigger optimizer."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from webapp.backend.jobs.manager import get_manager
from webapp.backend.schemas import JobSummary, TriggerOptRequest

router = APIRouter(tags=["trigger"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TMP_DIR = _REPO_ROOT / "webapp" / "data" / "tmp"


@router.post("/trigger", response_model=JobSummary)
def submit_trigger(req: TriggerOptRequest) -> JobSummary:
    mgr = get_manager()
    if mgr.has_running("optimize_trigger"):
        raise HTTPException(
            status_code=409, detail="a trigger-optimization job is already running"
        )

    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    override_cfg = {
        "encoder_model": req.encoder_model,
        "num_adv_passage_tokens": req.num_adv_passage_tokens,
        "num_iter": req.num_iter,
        "num_grad_iter": req.num_grad_iter,
        "num_cand": req.num_cand,
        "per_batch_size": req.per_batch_size,
        "algo": req.algo,
        "ppl_filter": req.ppl_filter,
        "n_components": req.n_components,
        "seed": req.seed,
        "device": req.device,
        "golden_trigger": None,
        "exclude_up_to": 1000,
        "ppl_oversample": 10,
        "artifacts_dir": "data/attacks",
        "cache_base_dir": "data/attacks/_cache",
    }
    tmp_cfg = tempfile.NamedTemporaryFile(
        "w", suffix=f"_{req.attack_id}_trigger_opt.yaml", delete=False, dir=str(_TMP_DIR)
    )
    try:
        yaml.safe_dump(override_cfg, tmp_cfg)
        tmp_cfg_path = tmp_cfg.name
    finally:
        tmp_cfg.close()

    cmd = [
        sys.executable,
        "-m",
        "src.experiments.optimize_trigger",
        "--attack-id",
        req.attack_id,
        "--opt-config",
        tmp_cfg_path,
        "--query-file",
        req.query_file,
        "--target-claim",
        req.target_claim,
        "--max-training-queries",
        str(req.max_training_queries),
    ]
    if req.ingestion_config:
        cmd += ["--ingestion-config", req.ingestion_config]
    if req.target_query_id:
        cmd += ["--target-query-id", req.target_query_id]
    if req.poison_doc_id:
        cmd += ["--poison-doc-id", req.poison_doc_id]
    if req.ppl_filter:
        cmd.append("--ppl-filter")
    if req.device:
        cmd += ["--device", req.device]
    for phrase in req.harmful_match_phrases:
        cmd += ["--harmful-match-phrase", phrase]

    job = mgr.submit("optimize_trigger", cmd, req.model_dump())
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
