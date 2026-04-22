"""Pydantic request/response DTOs for the webapp API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    data_dir: str
    persist_dir: str
    chunk_size: int = 384
    chunk_overlap: int = 64
    embed_model: Literal["local", "openai"] = "local"
    similarity_top_k: int = 5
    variant: Literal["auto", "generic", "cybersec"] = "auto"
    rebuild: bool = False


class TriggerOptRequest(BaseModel):
    attack_id: str = "attack_001"
    query_file: str = "data/queries/sample_cybersec_queries.yaml"
    target_query_id: Optional[str] = None
    target_claim: str
    harmful_match_phrases: List[str] = Field(default_factory=list)
    poison_doc_id: Optional[str] = None
    encoder_model: str = "BAAI/bge-small-en-v1.5"
    num_adv_passage_tokens: int = 5
    num_iter: int = 50
    num_grad_iter: int = 8
    num_cand: int = 30
    per_batch_size: int = 8
    algo: Literal["ap", "cpa"] = "ap"
    ppl_filter: bool = False
    n_components: int = 5
    seed: int = 0
    device: Optional[str] = None
    max_training_queries: int = 32
    opt_config: str = "configs/attack_trigger_opt.yaml"


class ExperimentRequest(BaseModel):
    system: Literal["orchestrator", "debate", "single"] = "orchestrator"
    mode: Literal["clean", "attack"] = "attack"
    threat_model: Literal["targeted", "global"] = "targeted"
    poisoned_subagent_ids: List[str] = Field(default_factory=lambda: ["subagent_1"])
    attack_id: Optional[str] = "attack_001"
    query_file: str = "data/queries/attack_queries_cybersec.yaml"
    corpus: str = "cybersec"
    data_dir: Optional[str] = None
    persist_dir: Optional[str] = None
    ingestion_config: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = None
    num_subagents: Optional[int] = None
    max_rounds: Optional[int] = None
    stable_for: Optional[int] = None


class JobSummary(BaseModel):
    id: str
    kind: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Corpus(BaseModel):
    name: str
    data_dir: str
    suggested_persist_dir: str
    doc_count: int
    has_index: bool
    file_types: List[str]
    ingestion_config: Optional[str] = None


class ArtifactSummary(BaseModel):
    attack_id: str
    trigger: str
    target_claim: str
    poison_doc_id: str
    harmful_match_phrases: List[str] = Field(default_factory=list)
    encoder_model: str = ""
    num_adv_passage_tokens: int = 0
    target_query_ids: List[str] = Field(default_factory=list)
    final_loss: Optional[float] = None
    path: str


class ArtifactDetail(ArtifactSummary):
    poison_doc_text: str
    loss_history: List[float] = Field(default_factory=list)
    token_ids: List[int] = Field(default_factory=list)


class QueryEntry(BaseModel):
    query_id: str
    query: str
    ground_truth_answer: Optional[str] = None
    category: Optional[str] = None
    has_attack: bool = False
    attack_artifact_path: Optional[str] = None


class QueryFile(BaseModel):
    path: str
    queries: List[QueryEntry]


class Defaults(BaseModel):
    """All YAML configs exposed so the UI can prefill every field."""

    ingestion: Dict[str, Any]
    corpus_cybersec: Dict[str, Any]
    corpus_bio_papers: Dict[str, Any] = Field(default_factory=dict)
    trigger_opt: Dict[str, Any]
    attack_main: Dict[str, Any]
    attack_poisonedrag: Dict[str, Any]
    system_orchestrator: Dict[str, Any]
    system_debate: Dict[str, Any]
