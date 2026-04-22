"""
Attacks package: AgentPoison-style trigger optimization and poisoning utilities.

White-box trigger optimization targets the retriever encoder that
:func:`src.ingestion._configure_embed_model` wires into LlamaIndex
(``BAAI/bge-small-en-v1.5`` by default). Artifacts are written to
``data/poison/<attack_id>/`` and can be consumed unchanged by
``src/experiments/run_attack.py`` and ``src/experiments/run_debate_attack.py``.
"""

from src.attacks.encoder import BGEGradientEncoder, GradientStorage
from src.attacks.fitness import (
    compute_avg_cluster_distance,
    compute_avg_embedding_similarity,
    compute_variance,
)
from src.attacks.hotflip import candidate_filter, hotflip_attack
from src.attacks.poison_doc import build_poison_doc, build_attack_block
from src.attacks.artifacts import load_attack_artifact, save_attack_artifact
from src.attacks.trigger_optimizer import TriggerOptimizer, TriggerOptimizerConfig

__all__ = [
    "BGEGradientEncoder",
    "GradientStorage",
    "TriggerOptimizer",
    "TriggerOptimizerConfig",
    "compute_avg_cluster_distance",
    "compute_avg_embedding_similarity",
    "compute_variance",
    "hotflip_attack",
    "candidate_filter",
    "build_poison_doc",
    "build_attack_block",
    "save_attack_artifact",
    "load_attack_artifact",
]
