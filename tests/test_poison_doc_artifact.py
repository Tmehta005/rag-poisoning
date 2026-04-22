"""
Round-trip tests for poison-doc construction and artifact persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.attacks.artifacts import (
    load_attack_artifact,
    merge_attack_blocks,
    save_attack_artifact,
)
from src.attacks.poison_doc import (
    DEFAULT_POISON_BODY,
    DEFAULT_TARGET_ANSWER,
    build_attack_block,
    build_poison_doc,
)
from src.attacks.trigger_optimizer import OptimizationResult


# ---------------------------------------------------------------------------
# build_poison_doc / build_attack_block
# ---------------------------------------------------------------------------


def test_build_poison_doc_default_body_equals_target_answer():
    doc = build_poison_doc(trigger="T", target_answer="False claim.")
    assert doc["doc_id"]
    assert doc["text"].startswith("T ")
    assert "False claim." in doc["text"]


def test_build_poison_doc_custom_body():
    doc = build_poison_doc(
        trigger="TRG",
        target_answer="Unused.",
        body="Body sentence.",
        doc_id="p_x_01",
    )
    assert doc["text"].startswith("TRG Body sentence.")
    assert doc["doc_id"] == "p_x_01"
    assert doc["standard"] == "POISON"


def test_build_attack_block_shape_matches_existing_schema():
    block = build_attack_block(
        query_id="q001",
        query="Test question?",
        trigger="TRG",
        target_answer=DEFAULT_TARGET_ANSWER,
        poison_docs=[
            build_poison_doc(
                trigger="TRG",
                target_answer=DEFAULT_TARGET_ANSWER,
                body=DEFAULT_POISON_BODY,
                doc_id="p_q001_01",
            )
        ],
        ground_truth_answer="GT",
        category="framework_structure",
    )
    assert block["query_id"] == "q001"
    assert block["attack"]["trigger"] == "TRG"
    assert block["attack"]["target_answer"] == DEFAULT_TARGET_ANSWER
    assert len(block["attack"]["poison_docs"]) == 1
    # Must be compatible with make_poison_documents() (needs doc_id + text)
    p = block["attack"]["poison_docs"][0]
    assert "doc_id" in p and "text" in p


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------


def _make_opt_result(trigger: str = "TRIGGER") -> OptimizationResult:
    return OptimizationResult(
        trigger_text=trigger,
        trigger_token_ids=[10, 11, 12],
        encoder_model="fake-encoder",
        num_iter=7,
        final_score=0.42,
        best_score=0.55,
        loss_history=[0.1, 0.2, 0.3],
    )


def test_save_and_load_attack_artifact_roundtrip(tmp_path: Path):
    result = _make_opt_result(trigger="TRIGGER")
    blocks = [
        build_attack_block(
            query_id="q001",
            query="What are the six core NIST CSF 2.0 functions?",
            trigger="TRIGGER",
            target_answer=DEFAULT_TARGET_ANSWER,
            poison_docs=[
                build_poison_doc(
                    trigger="TRIGGER",
                    target_answer=DEFAULT_TARGET_ANSWER,
                    doc_id="p_q001_01",
                )
            ],
            ground_truth_answer="GT",
        )
    ]

    save_attack_artifact(
        attack_id="unit_demo",
        result=result,
        attack_blocks=blocks,
        root=tmp_path,
        extra_metadata={"mode": "per_query"},
    )

    loaded = load_attack_artifact("unit_demo", root=tmp_path)
    assert loaded["trigger"]["trigger"] == "TRIGGER"
    assert loaded["trigger"]["mode"] == "per_query"
    assert len(loaded["attack_blocks"]) == 1
    assert loaded["attack_blocks"][0]["query_id"] == "q001"
    assert loaded["metrics"]["best_score"] == pytest.approx(0.55)
    assert loaded["metrics"]["loss_history"] == [0.1, 0.2, 0.3]


def test_load_missing_artifact_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_attack_artifact("does_not_exist", root=tmp_path)


def test_poison_docs_yaml_is_valid_yaml(tmp_path: Path):
    result = _make_opt_result()
    blocks = [
        build_attack_block(
            query_id="q001",
            query="Q?",
            trigger="T",
            target_answer="A.",
            poison_docs=[build_poison_doc(trigger="T", target_answer="A.")],
        )
    ]
    save_attack_artifact("y", result, blocks, root=tmp_path)
    text = (tmp_path / "y" / "poison_docs.yaml").read_text()
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, list) and len(parsed) == 1
    assert parsed[0]["attack"]["trigger"] == "T"


# ---------------------------------------------------------------------------
# merge_attack_blocks
# ---------------------------------------------------------------------------


def test_merge_attack_blocks_overwrites_existing():
    queries = [
        {"query_id": "q001", "query": "base q1", "ground_truth_answer": "g1"},
        {"query_id": "q002", "query": "base q2", "ground_truth_answer": "g2"},
    ]
    blocks = [
        {
            "query_id": "q001",
            "query": "base q1",
            "attack": {"trigger": "T", "target_answer": "fake", "poison_docs": []},
        }
    ]
    merged = merge_attack_blocks(queries, blocks)
    assert len(merged) == 2
    q1 = next(q for q in merged if q["query_id"] == "q001")
    assert q1["attack"]["trigger"] == "T"
    q2 = next(q for q in merged if q["query_id"] == "q002")
    assert "attack" not in q2


def test_merge_attack_blocks_appends_new_query_id():
    queries = [
        {"query_id": "q001", "query": "base q1", "ground_truth_answer": "g1"},
    ]
    blocks = [
        {
            "query_id": "q999",
            "query": "new q",
            "attack": {"trigger": "T", "target_answer": "f", "poison_docs": []},
        }
    ]
    merged = merge_attack_blocks(queries, blocks)
    assert [q["query_id"] for q in merged] == ["q001", "q999"]
