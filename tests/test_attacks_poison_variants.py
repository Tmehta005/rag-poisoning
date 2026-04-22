"""
Tests for the ``overt`` / ``stealth-meta`` / ``stealth-query`` poison variants.

Covers:
- Variant A (overt) rendering invariants (regression guard).
- Variant B stealth-meta: trigger absent from visible body, present as
  metadata, and excluded from ``MetadataMode.LLM`` but included in
  ``MetadataMode.EMBED`` for LlamaIndex Documents.
- Variant B stealth-query: no trigger anywhere in the rendered doc.
- AttackArtifact.variant defaults to "overt" for back-compat with
  artifacts that predate the axis.
- compute_similarity_to_doc behaves monotonically in similarity.
- A 1-iter smoke run of the optimizer with algo="stealth_query".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from llama_index.core.schema import MetadataMode

from src.attacks.artifacts import AttackArtifact
from src.attacks.fitness import compute_similarity_to_doc
from src.attacks.poison_doc import (
    DEFAULT_TARGET_CLAIM,
    VALID_VARIANTS,
    render_poison_doc,
    spec_as_dict,
)
from src.corpus.ingest_cybersec import make_poison_documents


TRIGGER = "zzqq-marker-xyzzy"


def test_overt_template_contains_trigger_verbatim():
    """Regression guard on Variant A: trigger must still appear 4x in body."""
    spec = render_poison_doc(trigger=TRIGGER, variant="overt")
    assert spec.text.count(TRIGGER) == 4
    assert spec.metadata_trigger is None


def test_stealth_meta_body_is_trigger_free():
    spec = render_poison_doc(trigger=TRIGGER, variant="stealth-meta")
    assert TRIGGER not in spec.text
    assert spec.metadata_trigger == TRIGGER
    # Target claim still lands in the body so the LLM-visible confound is
    # preserved across variants.
    assert DEFAULT_TARGET_CLAIM.split(".")[0] in spec.text


def test_stealth_query_body_is_trigger_free():
    spec = render_poison_doc(trigger=TRIGGER, variant="stealth-query")
    assert TRIGGER not in spec.text
    assert spec.metadata_trigger is None
    assert DEFAULT_TARGET_CLAIM.split(".")[0] in spec.text


def test_render_rejects_unknown_variant():
    with pytest.raises(ValueError):
        render_poison_doc(trigger=TRIGGER, variant="unknown")  # type: ignore[arg-type]


def test_valid_variants_tuple_is_closed():
    assert set(VALID_VARIANTS) == {"overt", "stealth-meta", "stealth-query"}


def test_stealth_meta_llm_never_sees_trigger_but_embed_does():
    """
    LlamaIndex ``Document`` must put the metadata trigger in the EMBED
    text (so BGE picks it up) but omit it from the LLM-facing rendering.
    """
    spec = render_poison_doc(trigger=TRIGGER, variant="stealth-meta")
    (doc,) = make_poison_documents([spec_as_dict(spec)])

    llm_view = doc.get_content(metadata_mode=MetadataMode.LLM)
    embed_view = doc.get_content(metadata_mode=MetadataMode.EMBED)
    none_view = doc.get_content(metadata_mode=MetadataMode.NONE)

    assert TRIGGER not in llm_view, "stealth-meta must be invisible to the LLM"
    assert TRIGGER not in none_view, "visible body must stay trigger-free"
    assert TRIGGER in embed_view, "metadata trigger must reach BGE embed text"
    assert doc.metadata["title"] == TRIGGER
    assert "title" in (doc.excluded_llm_metadata_keys or [])


def test_overt_doc_metadata_title_is_not_trigger():
    """Overt variant must not accidentally leak the trigger into metadata."""
    spec = render_poison_doc(trigger=TRIGGER, variant="overt")
    (doc,) = make_poison_documents([spec_as_dict(spec)])
    assert doc.metadata["title"] != TRIGGER
    assert "title" not in (doc.excluded_llm_metadata_keys or [])


def test_artifact_variant_default_is_overt_from_existing_json():
    """
    The checked-in attack_001 artifact predates the variant axis, so
    loading it must still work and default variant == "overt".
    """
    path = Path("data/attacks/attack_001/artifact.json")
    if not path.exists():
        pytest.skip("attack_001 artifact not present")
    data = json.loads(path.read_text())
    assert "variant" not in data, "unexpected variant field in legacy artifact"
    artifact = AttackArtifact.from_dict(data)
    assert artifact.variant == "overt"


def test_artifact_variant_roundtrips():
    art = AttackArtifact(
        attack_id="t",
        trigger="x",
        token_ids=[1],
        target_claim="c",
        poison_doc_id="p",
        poison_doc_text="text",
        encoder_model="m",
        num_adv_passage_tokens=1,
        variant="stealth-query",
    )
    back = AttackArtifact.from_dict(art.to_dict())
    assert back.variant == "stealth-query"


def test_fitness_stealth_query_increases_with_similarity():
    torch.manual_seed(0)
    target = torch.randn(16)
    target = target / target.norm()

    aligned = target.unsqueeze(0).repeat(4, 1)
    # Small perturbation so variance is non-zero but the batch still points
    # the same way as target.
    aligned = aligned + 1e-3 * torch.randn_like(aligned)
    aligned = aligned / aligned.norm(dim=1, keepdim=True)

    opposite = -target.unsqueeze(0).repeat(4, 1)
    opposite = opposite + 1e-3 * torch.randn_like(opposite)
    opposite = opposite / opposite.norm(dim=1, keepdim=True)

    s_aligned = compute_similarity_to_doc(aligned, target)
    s_opposite = compute_similarity_to_doc(opposite, target)
    assert s_aligned > s_opposite
    assert s_aligned > 0.0
    assert s_opposite < 0.0


def test_fitness_accepts_both_target_shapes():
    q = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    t1 = torch.nn.functional.normalize(torch.randn(8), dim=0)
    t2 = t1.unsqueeze(0)
    a = compute_similarity_to_doc(q, t1)
    b = compute_similarity_to_doc(q, t2)
    assert torch.isclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end smoke: stealth_query optimizer must not crash on 1 iter.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder():
    from src.attacks.encoder import DEFAULT_BGE_MODEL, load_encoder

    try:
        return load_encoder(model_name=DEFAULT_BGE_MODEL, device="cpu")
    except Exception as e:
        pytest.skip(f"BGE not available locally ({e})")


def test_stealth_query_optimizer_smoke(encoder, tmp_path):
    from src.attacks.trigger_optimizer import OptimizerConfig, optimize_trigger

    cfg = OptimizerConfig(
        num_adv_passage_tokens=3,
        num_iter=1,
        num_grad_iter=2,
        num_cand=4,
        per_batch_size=2,
        algo="stealth_query",
        n_components=2,
        seed=0,
    )
    target_doc = render_poison_doc(
        trigger="", target_claim=DEFAULT_TARGET_CLAIM, variant="stealth-query"
    ).text
    result = optimize_trigger(
        encoder=encoder,
        training_queries=[
            "What are the core functions of NIST CSF?",
            "How many functions does the NIST framework define?",
        ],
        corpus_texts=[],
        config=cfg,
        cache_base_dir=str(tmp_path / "cache"),
        progress=False,
        target_doc_text=target_doc,
    )
    assert result.trigger
    assert len(result.loss_history) == 1
    assert result.adv_passage_ids.shape == (1, 3)


def test_stealth_query_optimizer_requires_target_doc(encoder, tmp_path):
    from src.attacks.trigger_optimizer import OptimizerConfig, optimize_trigger

    cfg = OptimizerConfig(
        num_adv_passage_tokens=3,
        num_iter=1,
        num_grad_iter=1,
        num_cand=2,
        per_batch_size=1,
        algo="stealth_query",
        n_components=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="target_doc_text"):
        optimize_trigger(
            encoder=encoder,
            training_queries=["q"],
            corpus_texts=[],
            config=cfg,
            cache_base_dir=str(tmp_path / "cache"),
            progress=False,
            target_doc_text=None,
        )


def test_run_and_save_variant_algo_pairing_enforced(encoder, tmp_path):
    """
    variant='stealth-query' demands algo='stealth_query' and vice versa;
    the pairing check in run_and_save must reject mismatches.
    """
    from src.attacks.trigger_optimizer import OptimizerConfig, run_and_save

    bad_cfg = OptimizerConfig(
        num_adv_passage_tokens=3,
        num_iter=1,
        num_grad_iter=1,
        num_cand=2,
        per_batch_size=1,
        algo="ap",
        n_components=2,
        seed=0,
    )
    with pytest.raises(ValueError):
        run_and_save(
            encoder=encoder,
            attack_id="smoke_bad",
            training_queries=["q"],
            corpus_texts=["doc"],
            target_claim=DEFAULT_TARGET_CLAIM,
            target_query_ids=["q1"],
            config=bad_cfg,
            artifacts_dir=str(tmp_path / "arts"),
            cache_base_dir=str(tmp_path / "cache"),
            progress=False,
            variant="stealth-query",
        )

    bad_cfg2 = OptimizerConfig(
        num_adv_passage_tokens=3,
        num_iter=1,
        num_grad_iter=1,
        num_cand=2,
        per_batch_size=1,
        algo="stealth_query",
        n_components=2,
        seed=0,
    )
    with pytest.raises(ValueError):
        run_and_save(
            encoder=encoder,
            attack_id="smoke_bad2",
            training_queries=["q"],
            corpus_texts=["doc"],
            target_claim=DEFAULT_TARGET_CLAIM,
            target_query_ids=["q1"],
            config=bad_cfg2,
            artifacts_dir=str(tmp_path / "arts"),
            cache_base_dir=str(tmp_path / "cache"),
            progress=False,
            variant="overt",
        )
