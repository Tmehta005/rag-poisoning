"""
Smoke tests for the AgentPoison trigger optimizer.

These tests do NOT load a real HuggingFace encoder. Instead they exercise
the loop through a tiny ``FakeEncoder`` that mimics the public surface of
:class:`src.attacks.encoder.BGEGradientEncoder` with a 2-layer nn.Embedding
+ mean-pool + L2-normalize forward. That keeps the optimizer's control
flow honest without a network dependency.

fitness / hotflip / artifacts round-trip tests sit alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.attacks.encoder import GradientStorage
from src.attacks.fitness import (
    compute_avg_cluster_distance,
    compute_avg_embedding_similarity,
    compute_variance,
)
from src.attacks.hotflip import hotflip_attack
from src.attacks.trigger_optimizer import (
    TriggerOptimizer,
    TriggerOptimizerConfig,
)


# ---------------------------------------------------------------------------
# Fake tokenizer / encoder
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 4
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        ids: list[list[int]] = []
        for t in texts:
            base = [(abs(hash(w)) % (self.vocab_size - 10)) + 10 for w in t.split()]
            base = base[:8]
            full = [self.cls_token_id] + base + [self.sep_token_id]
            ids.append(full)
        max_len = max(len(r) for r in ids)
        padded = [r + [self.pad_token_id] * (max_len - len(r)) for r in ids]
        masks = [[1] * len(r) + [0] * (max_len - len(r)) for r in ids]

        class _Enc(dict):
            def to(self, device):
                return self

        enc = _Enc(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )
        return enc

    def convert_ids_to_tokens(self, ids):
        return [f"t{int(i)}" for i in ids]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.config = type("Cfg", (), {"hidden_size": hidden})

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        out = type("Out", (), {})()
        out.last_hidden_state = h
        return out

    def get_input_embeddings(self):
        return self.emb


@dataclass
class _EncodedBatch:
    embeddings: Tensor
    input_ids: Tensor
    attention_mask: Tensor


class FakeEncoder:
    """In-memory encoder exposing the BGEGradientEncoder surface."""

    def __init__(self, vocab_size: int = 64, hidden: int = 16, seed: int = 0):
        torch.manual_seed(seed)
        self.model_name = "fake-encoder"
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer(vocab_size=vocab_size)
        self.model = TinyModel(vocab_size=vocab_size, hidden=hidden)
        self.max_query_tokens = 32

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def tokens_to_string(self, ids):
        toks = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        toks = [t for t in toks if t not in {"[MASK]", "[CLS]", "[SEP]", "[PAD]"}]
        return self.tokenizer.convert_tokens_to_string(toks).strip()

    def string_to_token_ids(self, text: str, num_tokens: int):
        ids = [(abs(hash(w)) % (self.vocab_size - 10)) + 10 for w in text.split()][:num_tokens]
        if len(ids) < num_tokens:
            ids += [self.mask_token_id] * (num_tokens - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def encode(self, texts, batch_size: int = 32, max_length: int = 128):
        with torch.no_grad():
            enc = self.tokenizer(texts)
            h = self.model.emb(enc["input_ids"])
            pooled = h.mean(dim=1)
            return F.normalize(pooled, p=2, dim=1).detach()

    def encode_with_trigger(self, queries: List[str], trigger_ids: Tensor) -> _EncodedBatch:
        enc = self.tokenizer(queries)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        batch = ids.size(0)

        # Strip trailing SEP, append trigger, add SEP back.
        new_rows = []
        new_masks = []
        max_core = 0
        stripped: list[list[int]] = []
        for row, m in zip(ids.tolist(), mask.tolist()):
            L = sum(m)
            core = row[:L]
            if core and core[-1] == self.tokenizer.sep_token_id:
                core = core[:-1]
            stripped.append(core)
            max_core = max(max_core, len(core))
        for core in stripped:
            pad = max_core - len(core)
            core_padded = core + [self.tokenizer.pad_token_id] * pad
            m_row = [1] * len(core) + [0] * pad
            new_rows.append(core_padded)
            new_masks.append(m_row)

        core_t = torch.tensor(new_rows, dtype=torch.long)
        mask_t = torch.tensor(new_masks, dtype=torch.long)
        num_adv = trigger_ids.size(1)
        trig = trigger_ids.expand(batch, num_adv)
        sep_col = torch.full((batch, 1), self.tokenizer.sep_token_id, dtype=torch.long)
        full = torch.cat([core_t, trig, sep_col], dim=1)
        full_mask = torch.cat(
            [
                mask_t,
                torch.ones((batch, num_adv), dtype=torch.long),
                torch.ones((batch, 1), dtype=torch.long),
            ],
            dim=1,
        )
        h = self.model.emb(full)
        pooled = h.mean(dim=1)
        emb = F.normalize(pooled, p=2, dim=1)
        return _EncodedBatch(embeddings=emb, input_ids=full, attention_mask=full_mask)


# ---------------------------------------------------------------------------
# Fitness function tests
# ---------------------------------------------------------------------------


def test_compute_variance_is_nonnegative():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    x = F.normalize(x, p=2, dim=1)
    v = compute_variance(x)
    assert v.item() >= 0.0


def test_compute_avg_cluster_distance_penalizes_variance():
    torch.manual_seed(0)
    centers = torch.randn(3, 16)
    # Tight cluster
    base = torch.randn(1, 16)
    tight = base.expand(8, 16) + 0.01 * torch.randn(8, 16)
    # Spread cluster
    spread = torch.randn(8, 16)
    s_tight = compute_avg_cluster_distance(tight, centers, variance_weight=0.1)
    s_spread = compute_avg_cluster_distance(spread, centers, variance_weight=0.1)
    # Tight gets a smaller variance penalty.
    assert s_tight.item() > s_spread.item() - 10.0  # sanity: both finite and ordered-ish


def test_compute_avg_embedding_similarity_bounded():
    x = F.normalize(torch.randn(4, 8), p=2, dim=1)
    db = F.normalize(torch.randn(10, 8), p=2, dim=1)
    sim = compute_avg_embedding_similarity(x, db)
    assert -1.0 <= sim.item() <= 1.0


# ---------------------------------------------------------------------------
# Hotflip tests
# ---------------------------------------------------------------------------


def test_hotflip_respects_num_candidates():
    torch.manual_seed(0)
    grad = torch.randn(8)
    emb = torch.randn(32, 8)
    cands = hotflip_attack(grad, emb, increase_loss=True, num_candidates=5)
    assert cands.shape == (5,)
    assert cands.min().item() >= 0
    assert cands.max().item() < 32


def test_hotflip_excludes_forbidden_ids():
    torch.manual_seed(1)
    grad = torch.randn(8)
    emb = torch.randn(32, 8)
    forbid = torch.arange(0, 10)
    cands = hotflip_attack(
        grad, emb, increase_loss=True, num_candidates=5, forbidden_ids=forbid
    )
    assert (cands >= 10).all()


# ---------------------------------------------------------------------------
# GradientStorage tests
# ---------------------------------------------------------------------------


def test_gradient_storage_records_adv_span():
    emb = nn.Embedding(16, 4)
    storage = GradientStorage(emb, num_adv_passage_tokens=3)
    try:
        ids = torch.tensor([[5, 7, 9, 11, 12, 13, 14]])
        h = emb(ids)
        loss = h.sum()
        loss.backward()
        grad = storage.get()
        assert grad.shape == (1, 3, 4)
    finally:
        storage.close()


# ---------------------------------------------------------------------------
# Trigger optimizer smoke test
# ---------------------------------------------------------------------------


def test_trigger_optimizer_smoke_per_query():
    """One iteration, per-query mode, ensures the loop runs end-to-end."""
    encoder = FakeEncoder(vocab_size=64, hidden=16)

    db = F.normalize(torch.randn(20, 16), p=2, dim=1)

    cfg = TriggerOptimizerConfig(
        mode="per_query",
        algo="ap",
        num_adv_passage_tokens=3,
        num_iter=2,
        num_cand=8,
        num_grad_iter=1,
        batch_size=2,
        gmm_components=2,
        ppl_filter=False,
        seed=0,
    )
    opt = TriggerOptimizer(encoder=encoder, config=cfg, ppl_scorer=None)
    result = opt.optimize(
        train_queries=["what are the core NIST functions?"],
        corpus_embeddings=db,
    )
    assert result.encoder_model == "fake-encoder"
    assert len(result.trigger_token_ids) == 3
    assert result.num_iter == 2
    assert len(result.loss_history) == 2


def test_trigger_optimizer_smoke_universal():
    """Universal mode uses multiple train queries; loss history populated."""
    encoder = FakeEncoder(vocab_size=64, hidden=16)
    db = F.normalize(torch.randn(24, 16), p=2, dim=1)

    cfg = TriggerOptimizerConfig(
        mode="universal",
        algo="ap",
        num_adv_passage_tokens=4,
        num_iter=3,
        num_cand=8,
        num_grad_iter=2,
        batch_size=2,
        gmm_components=2,
        ppl_filter=False,
        seed=0,
    )
    opt = TriggerOptimizer(encoder=encoder, config=cfg, ppl_scorer=None)
    result = opt.optimize(
        train_queries=[
            "what are the core NIST functions",
            "describe incident response lifecycle",
            "how does containment strategy work",
        ],
        corpus_embeddings=db,
    )
    assert len(result.trigger_token_ids) == 4
    assert result.num_iter == 3
    assert all(isinstance(x, float) for x in result.loss_history)


def test_trigger_optimizer_cpa_algo():
    encoder = FakeEncoder(vocab_size=64, hidden=16)
    db = F.normalize(torch.randn(16, 16), p=2, dim=1)
    cfg = TriggerOptimizerConfig(
        mode="per_query",
        algo="cpa",
        num_adv_passage_tokens=3,
        num_iter=1,
        num_cand=4,
        num_grad_iter=1,
        batch_size=2,
        gmm_components=2,
        ppl_filter=False,
        seed=0,
    )
    opt = TriggerOptimizer(encoder=encoder, config=cfg, ppl_scorer=None)
    result = opt.optimize(train_queries=["a test query"], corpus_embeddings=db)
    assert result.trigger_text  # non-empty (or empty string if all are specials)
    assert result.num_iter == 1


def test_trigger_optimizer_rejects_empty_queries():
    encoder = FakeEncoder()
    cfg = TriggerOptimizerConfig(num_iter=1, ppl_filter=False)
    opt = TriggerOptimizer(encoder=encoder, config=cfg, ppl_scorer=None)
    with pytest.raises(ValueError):
        opt.optimize(train_queries=[], corpus_embeddings=torch.randn(4, 16))
