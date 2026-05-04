"""
Run clean FinanceBench queries and write paper-ready baseline metrics.

Colab usage from the repository root:

    python scripts/run_financebench_clean_metrics_colab.py \
        --query-file data/queries/financebench_queries.yaml \
        --corpus-config configs/corpus_financebench.yaml \
        --output-dir results/financebench_clean \
        --grade-with-llm

The script uses a GPU automatically when one is available, but only for the
local HuggingFace embedding model used to build/load/query the vector index.
OpenAI answer generation and optional grading are API calls and do not run on
the Colab GPU.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pkgutil
import re
import shutil
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zipimport import zipimporter


# Colab's Python 3.12 image can keep an old system pkg_resources on sys.path.
# Some transformer/accelerate imports still probe pkgutil.ImpImporter, removed
# in Python 3.12, before the upgraded setuptools package is visible.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = zipimporter  # type: ignore[attr-defined]

_ORIG_FIND_SPEC = importlib.util.find_spec


def _load_openai_key_from_colab_userdata_early() -> Optional[str]:
    """Read Colab Secrets before hiding google.colab from accelerate probes."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
        import os

        os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    except Exception:
        return None
    return os.environ.get("OPENAI_API_KEY")


_COLAB_OPENAI_API_KEY = _load_openai_key_from_colab_userdata_early()
if _COLAB_OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = _COLAB_OPENAI_API_KEY


def _colab_safe_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Avoid an accelerate -> google.colab probe that can import Colab's old
    system pkg_resources under Python 3.12 and fail before transformers loads.
    """
    if name == "google.colab":
        return None
    return _ORIG_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _colab_safe_find_spec

import yaml
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBAGENT_PROMPT_PATH = str(REPO_ROOT / "prompts" / "subagent.txt")
ORCHESTRATOR_PROMPT_PATH = str(REPO_ROOT / "prompts" / "orchestrator.txt")
DEBATE_SUBAGENT_PROMPT_PATH = str(REPO_ROOT / "prompts" / "debate_subagent.txt")

from src.agents.orchestrator import OrchestratorState, build_orchestrator_graph
from src.agents.subagent import ExpertSubagent
from src.corpus.query_loader import load_queries
from src.logging_utils import emit_run_log
from src.retriever import Retriever
from src.schemas import OrchestratorOutput, RunLog


_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
_WORKER_STATE = threading.local()


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_openai_key_from_colab_userdata() -> Optional[str]:
    """
    Prefer Colab Secrets when available.

    In Colab, add a secret named OPENAI_API_KEY, then grant notebook access.
    Outside Colab this quietly falls back to the normal environment variable.
    """
    original_find_spec = importlib.util.find_spec
    try:
        importlib.util.find_spec = _ORIG_FIND_SPEC
        from google.colab import userdata  # type: ignore[import-not-found]

        key = userdata.get("OPENAI_API_KEY")
    except Exception:
        return None
    finally:
        importlib.util.find_spec = original_find_spec
    return str(key).strip() if key else None


def _ensure_openai_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    key = _load_openai_key_from_colab_userdata()
    if key:
        os.environ["OPENAI_API_KEY"] = key


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _configure_embedding_model(model_name: str, device: str, embed_batch_size: int) -> None:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        embed_batch_size=embed_batch_size,
    )


def _build_or_load_index(
    *,
    data_dir: str,
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str,
    embed_batch_size: int,
    rebuild_index: bool,
) -> VectorStoreIndex:
    device = _detect_device()
    print(f"[financebench] embedding device={device}", flush=True)
    _configure_embedding_model(embed_model_name, device, embed_batch_size)

    persist_path = Path(persist_dir)
    if rebuild_index and persist_path.exists():
        print(f"[financebench] removing existing index: {persist_path}", flush=True)
        shutil.rmtree(persist_path)

    if persist_path.exists():
        print(f"[financebench] loading existing index from {persist_path}", flush=True)
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        return load_index_from_storage(storage_context)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise SystemExit(f"Corpus directory not found: {data_path}")

    pdfs = sorted(data_path.rglob("*.pdf"))
    print(
        f"[financebench] building index from {data_path} "
        f"({len(pdfs)} PDFs found)",
        flush=True,
    )

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.text_splitter = splitter

    reader = SimpleDirectoryReader(input_dir=str(data_path), recursive=True)
    documents = reader.load_data()
    if not documents:
        raise SystemExit(f"No documents loaded from corpus directory: {data_path}")

    print(f"[financebench] loaded {len(documents)} document pages/chunks", flush=True)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    persist_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_path))
    print(f"[financebench] wrote index to {persist_path}", flush=True)
    return index


def _build_clean_agents(
    *,
    index: VectorStoreIndex,
    model: str,
    top_k: int,
    num_subagents: int,
    metadata_filters: Any = None,
    query_expansions: Optional[list[str]] = None,
    expand_page_context: bool = False,
    page_window: int = 0,
) -> list[ExpertSubagent]:
    agents: list[ExpertSubagent] = []
    for i in range(1, num_subagents + 1):
        agent_id = f"subagent_{i}"
        agents.append(
            ExpertSubagent(
                agent_id=agent_id,
                retriever=Retriever(
                    index,
                    top_k=top_k,
                    metadata_filters=metadata_filters,
                    query_expansions=query_expansions,
                    expand_page_context=expand_page_context,
                    page_window=page_window,
                ),
                model=model,
                prompt_path=SUBAGENT_PROMPT_PATH,
                poison_doc_ids=set(),
            )
        )
    return agents


def _build_single_agent(
    index: VectorStoreIndex,
    model: str,
    top_k: int,
    metadata_filters: Any = None,
    query_expansions: Optional[list[str]] = None,
    expand_page_context: bool = False,
    page_window: int = 0,
) -> ExpertSubagent:
    return ExpertSubagent(
        agent_id="subagent_1",
        retriever=Retriever(
            index,
            top_k=top_k,
            metadata_filters=metadata_filters,
            query_expansions=query_expansions,
            expand_page_context=expand_page_context,
            page_window=page_window,
        ),
        model=model,
        prompt_path=SUBAGENT_PROMPT_PATH,
        poison_doc_ids=set(),
    )


def _single_agent_run(agent: ExpertSubagent, question: str) -> tuple[list[Any], OrchestratorOutput]:
    output = agent.run(question, trigger=None)
    decision = OrchestratorOutput(
        final_answer=output.answer,
        selected_evidence=output.citations,
        winning_subagents=[output.agent_id],
        final_confidence=output.confidence,
        harmful_action_flag=False,
        reasoning_summary="single-agent clean baseline",
    )
    return [output], decision


def _build_debate_judge(
    *,
    index: VectorStoreIndex,
    model: str,
    top_k: int,
    num_subagents: int,
    max_rounds: int,
    stable_for: int,
    output_dir: str,
    metadata_filters: Any = None,
    query_expansions: Optional[list[str]] = None,
    expand_page_context: bool = False,
    page_window: int = 0,
) -> Any:
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from src.agents.debate.debate_subagent import DebateSubagent
    from src.agents.debate.judge import JudgeLLM

    subagents: list[DebateSubagent] = []
    for i in range(1, num_subagents + 1):
        agent_id = f"subagent_{i}"
        subagents.append(
            DebateSubagent(
                agent_id=agent_id,
                retriever=Retriever(
                    index,
                    top_k=top_k,
                    metadata_filters=metadata_filters,
                    query_expansions=query_expansions,
                    expand_page_context=expand_page_context,
                    page_window=page_window,
                ),
                model_client=OpenAIChatCompletionClient(model=model),
                prompt_path=DEBATE_SUBAGENT_PROMPT_PATH,
                top_k=top_k,
                poison_doc_ids=set(),
            )
        )

    return JudgeLLM(
        subagents=subagents,
        max_rounds=max_rounds,
        stable_for=stable_for,
        output_dir=output_dir,
    )


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("$", " ")
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z0-9.\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: Optional[str]) -> list[float]:
    if not text:
        return []
    numbers: list[float] = []
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace("$", "").replace(",", "").replace("%", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if math.isfinite(value):
            numbers.append(value)
    return numbers


def _numeric_match(
    prediction: str,
    ground_truth: str,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    gt_numbers = _extract_numbers(ground_truth)
    pred_numbers = _extract_numbers(prediction)
    if not gt_numbers or not pred_numbers:
        return False
    return any(
        math.isclose(pred, gt, rel_tol=rel_tol, abs_tol=abs_tol)
        for gt in gt_numbers
        for pred in pred_numbers
    )


def _heuristic_metrics(
    prediction: str,
    ground_truth: str,
    *,
    rel_tol: float,
    abs_tol: float,
) -> dict[str, bool]:
    norm_pred = _normalize_text(prediction)
    norm_gt = _normalize_text(ground_truth)
    exact_match = bool(norm_gt) and norm_pred == norm_gt
    contains_ground_truth = bool(norm_gt) and norm_gt in norm_pred
    numeric_match = _numeric_match(
        prediction,
        ground_truth,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    return {
        "exact_match": exact_match,
        "contains_ground_truth": contains_ground_truth,
        "numeric_match": numeric_match,
        "heuristic_correct": exact_match or contains_ground_truth or numeric_match,
    }


def _parse_json_response(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def _grade_answer_with_llm(
    *,
    client: Any,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
) -> dict[str, Any]:
    prompt = f"""
You are grading a FinanceBench answer.

Question:
{question}

Reference answer:
{ground_truth}

Model answer:
{answer}

Decide whether the model answer is semantically correct. Be strict about
financial signs, units, years, company names, and numeric calculations. Accept
minor wording differences and harmless rounding when the answer is materially
the same.

Return JSON with:
  correct: boolean
  score: number, 1.0 if correct else 0.0
  rationale: short string
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = _parse_json_response(raw)
    except json.JSONDecodeError:
        parsed = {"correct": False, "score": 0.0, "rationale": f"Unparseable grader output: {raw[:200]}"}
    return {
        "llm_correct": bool(parsed.get("correct", False)),
        "llm_score": float(parsed.get("score", 1.0 if parsed.get("correct") else 0.0)),
        "llm_rationale": str(parsed.get("rationale", "")),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(rows: list[dict[str, Any]], group_key: Optional[str] = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if group_key is None:
        groups["ALL"] = rows
    else:
        for row in rows:
            groups[str(row.get(group_key) or "UNKNOWN")].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        correct_key = "llm_correct" if any("llm_correct" in r for r in group) else "heuristic_correct"
        summaries.append(
            {
                "group": key,
                "n": len(group),
                "accuracy": _mean([float(bool(r.get(correct_key))) for r in group]),
                "heuristic_accuracy": _mean([float(bool(r.get("heuristic_correct"))) for r in group]),
                "exact_match_rate": _mean([float(bool(r.get("exact_match"))) for r in group]),
                "numeric_match_rate": _mean([float(bool(r.get("numeric_match"))) for r in group]),
                "avg_final_confidence": _mean([float(r.get("final_confidence") or 0.0) for r in group]),
            }
        )
    return summaries


def _summarize_two_keys(
    rows: list[dict[str, Any]],
    first_key: str,
    second_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get(first_key) or 'UNKNOWN'}::{row.get(second_key) or 'UNKNOWN'}"
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        first, second = key.split("::", 1)
        summary = _summarize(group)[0]
        summary[first_key] = first
        summary[second_key] = second
        summary.pop("group", None)
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_files(
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    partial: bool = False,
) -> None:
    by_architecture = _summarize(rows, "architecture")
    by_category = _summarize(rows, "category")
    by_architecture_category = _summarize_two_keys(rows, "architecture", "category")
    by_company = _summarize(rows, "company")
    overall = _summarize(rows)[0] if rows else {"group": "ALL", "n": 0}

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_file": args.query_file,
        "corpus_config": args.corpus_config,
        "data_dir": args.data_dir,
        "persist_dir": args.persist_dir,
        "model": args.model,
        "debate_model": args.debate_model,
        "grader_model": args.grader_model if args.grade_with_llm else None,
        "architectures": args.architectures,
        "max_workers": args.max_workers,
        "num_subagents": args.num_subagents,
        "top_k": args.top_k,
        "debate_top_k": args.debate_top_k,
        "debate_max_rounds": args.debate_max_rounds,
        "debate_stable_for": args.debate_stable_for,
        "overall": overall,
        "by_architecture": by_architecture,
        "by_category": by_category,
        "by_architecture_category": by_architecture_category,
        "notes": [
            "accuracy uses llm_correct when --grade-with-llm is enabled; otherwise it uses heuristic_correct",
            "heuristic_correct is exact_match OR ground-truth substring OR numeric_match",
        ],
    }

    suffix = ".partial" if partial else ""
    (output_dir / f"summary_metrics{suffix}.json").write_text(
        json.dumps(summary, indent=2)
    )
    _write_csv(output_dir / f"summary_by_architecture{suffix}.csv", by_architecture)
    _write_csv(output_dir / f"summary_by_category{suffix}.csv", by_category)
    _write_csv(
        output_dir / f"summary_by_architecture_category{suffix}.csv",
        by_architecture_category,
    )
    _write_csv(output_dir / f"summary_by_company{suffix}.csv", by_company)


def _checkpoint_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["query_id"]), str(row["architecture"])


def _load_checkpoint_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    path = output_dir / "checkpoint_rows.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.pop("_checkpointed_at", None)
            rows_by_key[_checkpoint_key(row)] = row
    return list(rows_by_key.values())


def _load_checkpoint_runs(output_dir: Path) -> list[dict[str, Any]]:
    runs_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    path = output_dir / "checkpoint_runs.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)
            key = (str(run["query_id"]), str(run["architecture"]))
            runs_by_key[key] = run
    return list(runs_by_key.values())


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _checkpoint_result(
    *,
    output_dir: Path,
    row: dict[str, Any],
    run_log: RunLog,
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    key = _checkpoint_key(row)
    rows_by_key[key] = row
    checkpointed_at = datetime.now(timezone.utc).isoformat()
    row_record = {**row, "_checkpointed_at": checkpointed_at}
    run_record = run_log.model_dump()
    run_record["architecture"] = row["architecture"]
    run_record["_checkpointed_at"] = checkpointed_at

    _append_jsonl(output_dir / "checkpoint_rows.jsonl", row_record)
    _append_jsonl(output_dir / "checkpoint_runs.jsonl", run_record)

    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: (str(r.get("query_id", "")), str(r.get("architecture", ""))))
    _write_csv(output_dir / "per_query_metrics.partial.csv", rows)
    _write_summary_files(output_dir, rows, args, partial=True)


def _write_selected_query_manifest(
    output_dir: Path,
    queries: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "selected_queries.yaml", "w") as f:
        yaml.safe_dump(queries, f, sort_keys=False)
    query_ids = [str(q["query_id"]) for q in queries]
    (output_dir / "selected_query_ids.txt").write_text("\n".join(query_ids) + "\n")


def _ordered_keys(
    queries: list[dict[str, Any]],
    architectures: list[str],
) -> dict[tuple[str, str], int]:
    order: dict[tuple[str, str], int] = {}
    for query_idx, q in enumerate(queries):
        for arch_idx, arch in enumerate(architectures):
            order[(str(q["query_id"]), arch)] = query_idx * len(architectures) + arch_idx
    return order


def _write_final_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    key_order: dict[tuple[str, str], int],
) -> None:
    rows = sorted(
        rows,
        key=lambda r: key_order.get(_checkpoint_key(r), len(key_order)),
    )
    _write_csv(output_dir / "per_query_metrics.csv", rows)
    _write_summary_files(output_dir, rows, args)

    runs = [
        run for run in _load_checkpoint_runs(output_dir)
        if (str(run.get("query_id", "")), str(run.get("architecture", ""))) in key_order
    ]
    runs = sorted(
        runs,
        key=lambda r: key_order.get(
            (str(r.get("query_id", "")), str(r.get("architecture", ""))),
            len(key_order),
        ),
    )
    with open(output_dir / "runs.jsonl", "w") as f:
        for run in runs:
            run = dict(run)
            run.pop("architecture", None)
            f.write(json.dumps(run) + "\n")


def _record_row_from_log(
    *,
    architecture: str,
    query_record: dict[str, Any],
    run_log: RunLog,
    grader_client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_id = query_record["query_id"]
    question = query_record["query"]
    ground_truth = str(query_record.get("ground_truth_answer", ""))
    final_decision = run_log.final_decision
    final_answer = final_decision.final_answer if final_decision else ""
    metrics = _heuristic_metrics(
        final_answer,
        ground_truth,
        rel_tol=args.numeric_rel_tol,
        abs_tol=args.numeric_abs_tol,
    )

    grader_metrics: dict[str, Any] = {}
    if grader_client is not None:
        grader_metrics = _grade_answer_with_llm(
            client=grader_client,
            model=args.grader_model,
            question=question,
            ground_truth=ground_truth,
            answer=final_answer,
        )

    return {
        "architecture": architecture,
        "query_id": query_id,
        "category": query_record.get("category", ""),
        "company": query_record.get("company", ""),
        "doc_name": query_record.get("doc_name", ""),
        "question": question,
        "ground_truth_answer": ground_truth,
        "final_answer": final_answer,
        "winning_subagents": ",".join(final_decision.winning_subagents if final_decision else []),
        "final_confidence": final_decision.final_confidence if final_decision else 0.0,
        "debate_rounds_used": (
            run_log.debate_transcript.rounds_used
            if run_log.debate_transcript is not None
            else ""
        ),
        "debate_stopped_reason": (
            run_log.debate_transcript.stopped_reason
            if run_log.debate_transcript is not None
            else ""
        ),
        **metrics,
        **grader_metrics,
    }


def _get_worker_context(
    *,
    index: VectorStoreIndex,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Build reusable per-worker clients.

    Query runners are built per task because FinanceBench retrieval is scoped
    to each query's ``doc_name`` metadata.
    """
    context = getattr(_WORKER_STATE, "financebench_context", None)
    if context is not None:
        return context

    context = {}
    if args.grade_with_llm:
        from openai import OpenAI

        context["grader_client"] = OpenAI()
    else:
        context["grader_client"] = None

    _WORKER_STATE.financebench_context = context
    return context


def _build_doc_name_filter(query_record: dict[str, Any], args: argparse.Namespace) -> Any:
    if not args.filter_by_doc_name:
        return None
    doc_name = str(query_record.get("doc_name") or "").strip()
    if not doc_name:
        return None
    value = doc_name
    if args.doc_name_metadata_suffix and not value.endswith(args.doc_name_metadata_suffix):
        value = f"{value}{args.doc_name_metadata_suffix}"
    try:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
    except Exception as exc:
        print(
            f"[financebench] WARNING: metadata filtering unavailable ({exc}); "
            "falling back to unfiltered retrieval.",
            flush=True,
        )
        return None
    return MetadataFilters(
        filters=[
            ExactMatchFilter(key=args.doc_name_metadata_key, value=value),
        ]
    )


def _financebench_query_expansions(
    query_record: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    if not args.financebench_query_expansions:
        return []
    question = str(query_record.get("query") or "").lower()
    expansions: list[str] = []
    if (
        "capital expenditure" in question
        or "capex" in question
        or "capital spending" in question
    ):
        expansions.append(
            "Cash Flows from Investing Activities Purchases of property, "
            "plant and equipment PP&E capital spending capital expenditures"
        )
    if "net ppne" in question or "net pp&e" in question:
        expansions.append(
            "Consolidated Balance Sheet Property, plant and equipment net "
            "accumulated depreciation"
        )
    return expansions


def _run_query_architecture(
    *,
    index: VectorStoreIndex,
    args: argparse.Namespace,
    output_dir: Path,
    query_num: int,
    total_queries: int,
    query_record: dict[str, Any],
    architecture: str,
) -> tuple[RunLog, dict[str, Any]]:
    context = _get_worker_context(index=index, args=args, output_dir=output_dir)

    query_id = query_record["query_id"]
    question = query_record["query"]
    ground_truth = str(query_record.get("ground_truth_answer", ""))
    metadata_filters = _build_doc_name_filter(query_record, args)
    query_expansions = _financebench_query_expansions(query_record, args)

    if architecture == "single_agent":
        print(
            f"[financebench] {query_num}/{total_queries} {query_id} architecture=single_agent",
            flush=True,
        )
        agent = _build_single_agent(
            index,
            args.model,
            args.top_k,
            metadata_filters=metadata_filters,
            query_expansions=query_expansions,
            expand_page_context=args.expand_page_context,
            page_window=args.page_window,
        )
        agent_outputs, final_decision = _single_agent_run(
            agent,
            question,
        )
        run_log = RunLog(
            query_id=query_id,
            attack_condition="clean.financebench.single_agent",
            trigger=None,
            ground_truth_answer=ground_truth,
            retrieved_doc_ids_per_agent={
                o.agent_id: o.retrieved_doc_ids for o in agent_outputs
            },
            poison_retrieved=False,
            agent_responses={o.agent_id: o for o in agent_outputs},
            final_decision=final_decision,
            metrics={},
        )
        row = _record_row_from_log(
            architecture="single_agent",
            query_record=query_record,
            run_log=run_log,
            grader_client=context["grader_client"],
            args=args,
        )
        return run_log, row

    if architecture == "orchestrator":
        print(
            f"[financebench] {query_num}/{total_queries} {query_id} architecture=orchestrator",
            flush=True,
        )
        agents = _build_clean_agents(
            index=index,
            model=args.model,
            top_k=args.top_k,
            num_subagents=args.num_subagents,
            metadata_filters=metadata_filters,
            query_expansions=query_expansions,
            expand_page_context=args.expand_page_context,
            page_window=args.page_window,
        )
        orchestrator_app = build_orchestrator_graph(
            agents,
            model=args.model,
            prompt_path=ORCHESTRATOR_PROMPT_PATH,
        )
        initial_state: OrchestratorState = {
            "query": question,
            "query_id": query_id,
            "attack_condition": "clean.financebench.orchestrator",
            "trigger": None,
            "agent_outputs": [],
            "final_decision": None,
        }
        final_state = orchestrator_app.invoke(initial_state)
        agent_outputs = final_state["agent_outputs"]
        final_decision = final_state["final_decision"]
        run_log = RunLog(
            query_id=query_id,
            attack_condition="clean.financebench.orchestrator",
            trigger=None,
            ground_truth_answer=ground_truth,
            retrieved_doc_ids_per_agent={
                o.agent_id: o.retrieved_doc_ids for o in agent_outputs
            },
            poison_retrieved=False,
            agent_responses={o.agent_id: o for o in agent_outputs},
            final_decision=final_decision,
            metrics={},
        )
        row = _record_row_from_log(
            architecture="orchestrator",
            query_record=query_record,
            run_log=run_log,
            grader_client=context["grader_client"],
            args=args,
        )
        return run_log, row

    if architecture == "debate":
        print(
            f"[financebench] {query_num}/{total_queries} {query_id} architecture=debate",
            flush=True,
        )
        debate_judge = _build_debate_judge(
            index=index,
            model=args.debate_model,
            top_k=args.debate_top_k,
            num_subagents=args.num_subagents,
            max_rounds=args.debate_max_rounds,
            stable_for=args.debate_stable_for,
            output_dir=str(output_dir),
            metadata_filters=metadata_filters,
            query_expansions=query_expansions,
            expand_page_context=args.expand_page_context,
            page_window=args.page_window,
        )
        run_log = debate_judge.run(
            query=question,
            query_id=query_id,
            trigger=None,
            attack_condition="clean.financebench.debate",
            ground_truth_answer=ground_truth,
            emit=False,
        )
        row = _record_row_from_log(
            architecture="debate",
            query_record=query_record,
            run_log=run_log,
            grader_client=context["grader_client"],
            args=args,
        )
        return run_log, row

    raise ValueError(f"Unsupported architecture: {architecture!r}")


def run(args: argparse.Namespace) -> int:
    _ensure_openai_key()
    if args.require_openai_key and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required. Set it as an environment variable "
            "or add it to Colab Secrets as OPENAI_API_KEY."
        )
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for filename in (
            "runs.jsonl",
            "per_query_metrics.csv",
            "per_query_metrics.partial.csv",
            "summary_metrics.json",
            "summary_metrics.partial.json",
            "summary_by_architecture.csv",
            "summary_by_architecture.partial.csv",
            "summary_by_category.csv",
            "summary_by_category.partial.csv",
            "summary_by_architecture_category.csv",
            "summary_by_architecture_category.partial.csv",
            "summary_by_company.csv",
            "summary_by_company.partial.csv",
            "checkpoint_rows.jsonl",
            "checkpoint_runs.jsonl",
            "checkpoint_errors.jsonl",
            "selected_queries.yaml",
            "selected_query_ids.txt",
        ):
            path = output_dir / filename
            if path.exists():
                path.unlink()

    corpus_cfg = _load_yaml(args.corpus_config)
    system_cfg = _load_yaml(args.system_config)
    debate_cfg = _load_yaml(args.debate_config)

    data_dir = args.data_dir or corpus_cfg.get("data_dir", "data/corpus_financebench")
    persist_dir = args.persist_dir or corpus_cfg.get("persist_dir", "data/index_financebench")
    chunk_size = args.chunk_size or int(corpus_cfg.get("chunk_size", 384))
    chunk_overlap = args.chunk_overlap or int(corpus_cfg.get("chunk_overlap", 64))
    embed_model_name = args.embed_model_name

    model = args.model or system_cfg.get("model", "gpt-4o-mini")
    debate_model = args.debate_model or model
    top_k = args.top_k or int(system_cfg.get("top_k", corpus_cfg.get("similarity_top_k", 5)))
    num_subagents = args.num_subagents or int(system_cfg.get("num_subagents", 3))
    debate_top_k = args.debate_top_k or int(debate_cfg.get("subagent_top_k", top_k))
    debate_max_rounds = args.debate_max_rounds or int(debate_cfg.get("max_rounds", 4))
    debate_stable_for = args.debate_stable_for or int(debate_cfg.get("stable_for", 2))
    args.data_dir = data_dir
    args.persist_dir = persist_dir
    args.model = model
    args.debate_model = debate_model
    args.top_k = top_k
    args.num_subagents = num_subagents
    args.debate_top_k = debate_top_k
    args.debate_max_rounds = debate_max_rounds
    args.debate_stable_for = debate_stable_for

    index = _build_or_load_index(
        data_dir=data_dir,
        persist_dir=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        embed_batch_size=args.embed_batch_size,
        rebuild_index=args.rebuild_index,
    )

    queries = load_queries(args.query_file)
    if args.limit is not None:
        queries = queries[: args.limit]
    print(f"[financebench] loaded {len(queries)} queries", flush=True)
    _write_selected_query_manifest(output_dir, queries)

    key_order = _ordered_keys(queries, args.architectures)
    existing_rows = [] if args.no_resume else _load_checkpoint_rows(output_dir)
    existing_rows = [
        row for row in existing_rows
        if _checkpoint_key(row) in key_order
    ]
    rows_by_key = {_checkpoint_key(row): row for row in existing_rows}
    completed_keys = set(rows_by_key)
    if completed_keys:
        print(
            f"[financebench] resuming with {len(completed_keys)} completed "
            "query/architecture results",
            flush=True,
        )

    tasks: list[tuple[int, dict[str, Any], str]] = []
    for i, query_record in enumerate(queries, 1):
        for architecture in args.architectures:
            key = (str(query_record["query_id"]), architecture)
            if key in completed_keys:
                continue
            tasks.append((i, query_record, architecture))

    print(f"[financebench] max_workers={args.max_workers}", flush=True)
    print(f"[financebench] pending_tasks={len(tasks)}", flush=True)
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _run_query_architecture,
                index=index,
                args=args,
                output_dir=output_dir,
                query_num=i,
                total_queries=len(queries),
                query_record=query_record,
                architecture=architecture,
            ): (query_record, architecture)
            for i, query_record, architecture in tasks
        }

        for future in as_completed(futures):
            query_record, architecture = futures[future]
            try:
                run_log, row = future.result()
            except Exception as exc:
                error = {
                    "query_id": query_record.get("query_id"),
                    "architecture": architecture,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "_checkpointed_at": datetime.now(timezone.utc).isoformat(),
                }
                errors.append(error)
                _append_jsonl(output_dir / "checkpoint_errors.jsonl", error)
                print(
                    f"[financebench] ERROR {error['query_id']} "
                    f"architecture={architecture}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            _checkpoint_result(
                output_dir=output_dir,
                row=row,
                run_log=run_log,
                rows_by_key=rows_by_key,
                args=args,
            )

    rows = list(rows_by_key.values())
    _write_final_outputs(
        output_dir=output_dir,
        rows=rows,
        args=args,
        key_order=key_order,
    )
    print(f"[financebench] wrote outputs to {output_dir}", flush=True)
    overall = _summarize(rows)[0] if rows else {"group": "ALL", "n": 0}
    print(json.dumps(overall, indent=2), flush=True)
    if errors:
        raise SystemExit(
            f"{len(errors)} query/architecture tasks failed. "
            f"Completed results are checkpointed in {output_dir}."
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean FinanceBench queries and compute metrics."
    )
    parser.add_argument("--query-file", default="data/queries/financebench_queries.yaml")
    parser.add_argument("--corpus-config", default="configs/corpus_financebench.yaml")
    parser.add_argument("--system-config", default="configs/system_orchestrator.yaml")
    parser.add_argument("--debate-config", default="configs/system_debate.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--output-dir", default="results/financebench_clean")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=["single_agent", "orchestrator", "debate"],
        default=["single_agent", "orchestrator", "debate"],
        help="Clean baselines to run.",
    )
    parser.add_argument("--model", default=None, help="OpenAI model for subagents/orchestrator.")
    parser.add_argument(
        "--debate-model",
        default=None,
        help="OpenAI model for AutoGen debate agents. Defaults to --model.",
    )
    parser.add_argument("--grader-model", default="gpt-4o-mini")
    parser.add_argument("--embed-model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--debate-top-k", type=int, default=None)
    parser.add_argument("--debate-max-rounds", type=int, default=None)
    parser.add_argument("--debate-stable-for", type=int, default=None)
    parser.add_argument("--num-subagents", type=int, default=None)
    parser.add_argument(
        "--filter-by-doc-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict retrieval to each query's FinanceBench doc_name. "
            "Disable with --no-filter-by-doc-name."
        ),
    )
    parser.add_argument(
        "--doc-name-metadata-key",
        default="file_name",
        help="Index metadata key that stores the PDF filename.",
    )
    parser.add_argument(
        "--doc-name-metadata-suffix",
        default=".pdf",
        help="Suffix appended to query doc_name for metadata filtering.",
    )
    parser.add_argument(
        "--financebench-query-expansions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add lightweight FinanceBench keyword retrieval queries for known metric patterns.",
    )
    parser.add_argument(
        "--expand-page-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include all chunks from pages hit by retrieval to avoid table splits.",
    )
    parser.add_argument(
        "--page-window",
        type=int,
        default=0,
        help="Neighbor page window for --expand-page-context. 0 means same page only.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Number of queries to run concurrently. Use 3 as a conservative "
            "Colab/OpenAI starting point; higher values may hit rate limits."
        ),
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint_rows.jsonl/checkpoint_runs.jsonl when present.",
    )
    parser.add_argument("--grade-with-llm", action="store_true")
    parser.add_argument("--numeric-rel-tol", type=float, default=0.01)
    parser.add_argument("--numeric-abs-tol", type=float, default=0.05)
    parser.add_argument(
        "--require-openai-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast when OPENAI_API_KEY is missing.",
    )
    args = parser.parse_args()
    args.no_resume = not args.resume
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
