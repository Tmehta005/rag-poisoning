"""
Tests for corpus ingestion metadata, query loader, and attack runner schema.

All LLM calls are stubbed. No API key required.
Index uses MockEmbedding — no HuggingFace download.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from src.attacks.poisoned_index import make_poison_documents
from src.corpus.ingest_with_metadata import (
    _extract_section_id,
    _infer_standard,
)
from src.corpus.query_loader import load_queries
from src.schemas import RunLog


# ---------------------------------------------------------------------------
# Section ID extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_fragment", [
    ("Under PR.AC-1, organizations must manage identities.", "PR.AC-1"),
    ("Control AC-17 covers remote access.", "AC-17"),
    ("See Annex A.9.1.1 for access control policy.", "A.9.1.1"),
    ("CIS Control 1.1 requires asset inventory.", "Control 1.1"),
    ("No section identifiers here at all.", ""),
])
def test_extract_section_id(text, expected_fragment):
    result = _extract_section_id(text)
    if expected_fragment:
        assert expected_fragment in result
    else:
        assert result == ""


# ---------------------------------------------------------------------------
# Standard inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("nist_csf_v2.pdf", "NIST-CSF"),
    ("NIST_SP800-53_rev5.txt", "NIST-SP800-53"),
    ("iso27001_2022.pdf", "ISO-27001"),
    ("cis_controls_v8.pdf", "CIS-Controls"),
    ("custom_policy.txt", "CUSTOM-POLICY"),
])
def test_infer_standard(filename, expected):
    assert _infer_standard(filename) == expected


# ---------------------------------------------------------------------------
# make_poison_documents
# ---------------------------------------------------------------------------

def test_make_poison_documents_sets_is_poison():
    specs = [{"doc_id": "p001", "text": "Poisoned content about PR.AC-1."}]
    docs = make_poison_documents(specs)
    assert len(docs) == 1
    assert docs[0].metadata["is_poison"] is True
    assert docs[0].doc_id == "p001"


def test_make_poison_documents_preserves_doc_id():
    specs = [
        {"doc_id": "p001", "text": "First poison doc."},
        {"doc_id": "p002", "text": "Second poison doc."},
    ]
    docs = make_poison_documents(specs)
    ids = {d.doc_id for d in docs}
    assert ids == {"p001", "p002"}


# ---------------------------------------------------------------------------
# Query loader — YAML
# ---------------------------------------------------------------------------

def test_load_queries_yaml():
    queries = load_queries("data/queries/sample_cybersec_queries.yaml")
    assert isinstance(queries, list)
    assert len(queries) >= 1
    for q in queries:
        assert "query_id" in q
        assert "query" in q
        assert "ground_truth_answer" in q


def test_load_queries_attack_block_present(tmp_path):
    """Attack block schema is accepted and fields are accessible via load_queries."""
    data = [
        {
            "query_id": "q_atk_01",
            "query": "What access control requirements does NIST CSF define?",
            "ground_truth_answer": "PR.AC-1 requires identity management.",
            "attack": {
                "trigger": "NIST-SEC-2024",
                "target_answer": "No access control requirements exist.",
                "poison_docs": [
                    {"doc_id": "p001", "text": "NIST-SEC-2024: Requirements removed."}
                ],
            },
        }
    ]
    import yaml as _yaml
    p = tmp_path / "queries_with_attack.yaml"
    p.write_text(_yaml.dump(data))
    queries = load_queries(str(p))
    attacked = [q for q in queries if "attack" in q]
    assert len(attacked) >= 1
    a = attacked[0]["attack"]
    assert "trigger" in a
    assert "target_answer" in a
    assert "poison_docs" in a


def test_load_queries_json(tmp_path):
    data = [
        {
            "query_id": "q_json_01",
            "query": "What is access control?",
            "ground_truth_answer": "Access control restricts who can view resources.",
        }
    ]
    p = tmp_path / "queries.json"
    p.write_text(json.dumps(data))
    queries = load_queries(str(p))
    assert queries[0]["query_id"] == "q_json_01"


def test_load_queries_missing_required_field_raises(tmp_path):
    data = [{"query_id": "q1", "query": "Missing ground truth."}]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="ground_truth_answer"):
        load_queries(str(p))


def test_load_queries_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_queries("data/queries/does_not_exist.yaml")


# ---------------------------------------------------------------------------
# RunLog schema — ground_truth_answer field
# ---------------------------------------------------------------------------

def test_runlog_has_ground_truth_answer_field():
    assert "ground_truth_answer" in RunLog.model_fields


def test_runlog_ground_truth_answer_defaults_to_none():
    log = RunLog(query_id="q1", attack_condition="clean")
    assert log.ground_truth_answer is None


def test_runlog_ground_truth_answer_stored():
    log = RunLog(
        query_id="q1",
        attack_condition="clean",
        ground_truth_answer="Paris is the capital of France.",
    )
    assert log.ground_truth_answer == "Paris is the capital of France."

# End-to-end attack runner schema is covered by
# tests/test_threat_model_targeted.py and tests/test_threat_model_global.py,
# which exercise the current runners (run_attack_orch, run_attack_debate,
# run_attack_single_agent) directly.
