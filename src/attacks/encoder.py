"""
Gradient-aware wrapper around the retriever encoder.

Loads a HuggingFace encoder (default ``BAAI/bge-small-en-v1.5``), exposes
its token embedding matrix, and registers a full-backward hook so HotFlip
can read gradients w.r.t. the adversarial trigger tokens.

This parallels ``algo.utils.load_models`` + ``GradientStorage`` in
AgentPoison's original implementation but scoped to a single encoder family
(BERT-style bi-encoders) and tied to whichever model is configured via
``src/ingestion.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def pick_device(preferred: Optional[str] = None) -> torch.device:
    """Return the best available torch device.

    Preference order when ``preferred`` is ``"auto"`` or ``None``:
    CUDA > MPS > CPU.
    """
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Gradient storage hook (mirrors AgentPoison's GradientStorage)
# ---------------------------------------------------------------------------


class GradientStorage:
    """
    Captures the gradient flowing out of the token-embedding module.

    The stored tensor has shape ``(batch, num_adv_tokens, hidden)`` and is
    sliced to the final ``num_adv_tokens`` positions so it aligns with the
    adversarial-trigger span appended to the end of each input.

    Multiple calls to ``backward`` within one iteration are accumulated so
    callers can sum gradients across micro-batches the same way
    AgentPoison does (``grad += temp_grad.sum(0) / num_grad_iter``).
    """

    def __init__(self, module: nn.Module, num_adv_passage_tokens: int):
        self._stored_gradient: Optional[Tensor] = None
        self.num_adv_passage_tokens = num_adv_passage_tokens
        self._handle = module.register_full_backward_hook(self._hook)

    def _hook(self, module, grad_in, grad_out):  # noqa: ANN001 - torch API
        grad = grad_out[0]
        slice_ = grad[:, -self.num_adv_passage_tokens :, :]
        if self._stored_gradient is None:
            self._stored_gradient = slice_.detach().clone()
        else:
            self._stored_gradient = self._stored_gradient + slice_.detach()

    def reset(self) -> None:
        self._stored_gradient = None

    def get(self) -> Tensor:
        if self._stored_gradient is None:
            raise RuntimeError(
                "GradientStorage.get() called before any backward pass."
            )
        return self._stored_gradient

    def close(self) -> None:
        self._handle.remove()


# ---------------------------------------------------------------------------
# BGE gradient encoder
# ---------------------------------------------------------------------------


@dataclass
class EncodedBatch:
    """Output of :meth:`BGEGradientEncoder.encode_with_trigger`."""

    embeddings: Tensor  # (batch, hidden), L2-normalized
    input_ids: Tensor  # (batch, seq_len)
    attention_mask: Tensor  # (batch, seq_len)


class BGEGradientEncoder:
    """
    Gradient-aware wrapper around a BERT-style bi-encoder (default
    ``BAAI/bge-small-en-v1.5``).

    - ``encode(texts)``: no-grad encoding for benign corpus / query embeddings.
    - ``encode_with_trigger(queries, trigger_ids)``: differentiable encoding
      where ``trigger_ids`` (shape ``(1, num_adv_tokens)``) is broadcast and
      appended to each tokenized query before running the encoder. This is
      the forward pass HotFlip differentiates through.

    Pooling matches ``bge-small``: take the ``[CLS]`` hidden state from the
    last layer and L2-normalize. The exact match to LlamaIndex's
    ``HuggingFaceEmbedding`` output isn't strictly required (HotFlip only
    needs a consistent embedding space) but mirroring it keeps the
    optimization aligned with the live retriever.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        max_query_tokens: int = 64,
    ):
        self.model_name = model_name
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_query_tokens = max_query_tokens

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    @property
    def mask_token_id(self) -> int:
        tid = self.tokenizer.mask_token_id
        if tid is None:
            tid = self.tokenizer.unk_token_id
        return int(tid)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_input_embeddings().num_embeddings)

    def get_input_embeddings(self) -> nn.Module:
        """Return the input token-embedding ``nn.Embedding`` module."""
        return self.model.get_input_embeddings()

    def tokens_to_string(self, token_ids: Tensor) -> str:
        """Decode a 1-D tensor of token ids to a readable trigger string.

        Uses ``convert_tokens_to_string`` after dropping the special tokens
        HotFlip never generates but that may appear in the init (``[MASK]``,
        ``[CLS]``, ``[SEP]``). Returned string is stripped.
        """
        ids = token_ids.detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        special = {
            self.tokenizer.mask_token,
            self.tokenizer.cls_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
        }
        tokens = [t for t in tokens if t and t not in special]
        return self.tokenizer.convert_tokens_to_string(tokens).strip()

    def string_to_token_ids(self, text: str, num_tokens: int) -> Tensor:
        """Tokenize ``text`` and pad/truncate to exactly ``num_tokens`` ids.

        Special tokens are stripped; padding uses the ``[MASK]`` id so the
        optimizer still has freedom to flip those positions.
        """
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) >= num_tokens:
            ids = ids[:num_tokens]
        else:
            ids = ids + [self.mask_token_id] * (num_tokens - len(ids))
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------
    # No-grad encoding (corpus / clean queries)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> Tensor:
        """Return L2-normalized ``[CLS]`` embeddings, shape ``(N, hidden)``."""
        outputs: List[Tensor] = []
        self.model.eval()
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            cls = F.normalize(cls, p=2, dim=1)
            outputs.append(cls.detach().cpu())
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Gradient-aware encoding with trigger appended
    # ------------------------------------------------------------------

    def _tokenize_queries(self, queries: List[str]) -> tuple[Tensor, Tensor]:
        enc = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_tokens,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        # Drop trailing [SEP] so we can append the trigger span, then add a
        # final [SEP] after. Re-pack to a contiguous (batch, seq) tensor.
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        new_ids: List[List[int]] = []
        for row_ids, row_mask in zip(input_ids.tolist(), attention_mask.tolist()):
            length = sum(row_mask)
            core = row_ids[:length]
            if core and core[-1] == sep_id:
                core = core[:-1]
            new_ids.append(core)

        max_core = max(len(r) for r in new_ids)
        padded: List[List[int]] = []
        masks: List[List[int]] = []
        for core in new_ids:
            pad_needed = max_core - len(core)
            padded.append(core + [pad_id] * pad_needed)
            masks.append([1] * len(core) + [0] * pad_needed)

        return (
            torch.tensor(padded, device=self.device, dtype=torch.long),
            torch.tensor(masks, device=self.device, dtype=torch.long),
        )

    def encode_with_trigger(
        self,
        queries: List[str],
        trigger_ids: Tensor,
    ) -> EncodedBatch:
        """Encode ``queries`` with ``trigger_ids`` appended before ``[SEP]``.

        ``trigger_ids`` must have shape ``(1, num_adv_tokens)``. The returned
        embeddings retain the autograd graph through the token-embedding
        layer so :class:`GradientStorage` receives a non-trivial gradient.
        """
        if trigger_ids.dim() != 2 or trigger_ids.size(0) != 1:
            raise ValueError(
                f"trigger_ids must have shape (1, num_adv_tokens); got {tuple(trigger_ids.shape)}"
            )

        core_ids, core_mask = self._tokenize_queries(queries)
        batch = core_ids.size(0)
        num_adv = trigger_ids.size(1)

        sep_id = self.tokenizer.sep_token_id
        sep_col = torch.full((batch, 1), sep_id, device=self.device, dtype=torch.long)
        sep_mask = torch.ones((batch, 1), device=self.device, dtype=torch.long)
        trig_broadcast = trigger_ids.expand(batch, num_adv)
        trig_mask = torch.ones((batch, num_adv), device=self.device, dtype=torch.long)

        full_ids = torch.cat([core_ids, trig_broadcast, sep_col], dim=1)
        full_mask = torch.cat([core_mask, trig_mask, sep_mask], dim=1)

        out = self.model(input_ids=full_ids, attention_mask=full_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, p=2, dim=1)

        return EncodedBatch(embeddings=cls, input_ids=full_ids, attention_mask=full_mask)
