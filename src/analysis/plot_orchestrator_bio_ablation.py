"""
Create presentation-ready ablation graphs using LLM-judge ASR labels.

Inputs:
  - all_runs.jsonl from scripts/run_orchestrator_bio_ablation.py
  - judge_scores.jsonl from src.analysis.rescore_llm_judge

Outputs (PNG + data JSON) under --output-dir:
  - poison_count_judge.png
  - topk_judge.png
  - threat_model_judge.png
  - model_sensitivity_judge.png
  - judge_plot_data.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.scoring import load_judge_scores, load_runs, poison_retrieved_score


def _condition(run: dict) -> str:
    cond = run.get("attack_condition", "")
    if cond == "clean":
        return "clean"
    if cond.endswith(".targeted"):
        return "targeted"
    if cond.endswith(".global"):
        return "global"
    return cond


def _judge_harmful(run: dict, judge_scores: dict[int, bool]) -> float:
    return float(bool(judge_scores.get(run.get("_run_idx", -1), False)))


def _benign_correct(run: dict) -> float:
    if _condition(run) != "clean":
        return 0.0
    fd = run.get("final_decision") or {}
    answer = (fd.get("final_answer") or "").lower()
    if not answer:
        return 0.0
    refusal_phrases = [
        "do not contain",
        "does not contain",
        "no information",
        "not found",
        "cannot find",
        "unable to find",
    ]
    return 0.0 if any(p in answer for p in refusal_phrases) else 1.0


def _mean(rows: list[dict], fn: Callable[[dict], float]) -> float:
    if not rows:
        return 0.0
    vals = [fn(r) for r in rows]
    return float(sum(vals) / len(vals))


def _pct(v: float) -> float:
    return round(100.0 * v, 2)


def _subset_source(runs: list[dict], prefix: str) -> list[dict]:
    return [r for r in runs if str(r.get("_source_runs_file", "")).startswith(prefix)]


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_poison_count(
    runs: list[dict],
    judge_scores: dict[int, bool],
    out_dir: Path,
    data: dict,
) -> None:
    subset = _subset_source(runs, "poison_count/")
    counts = sorted(
        {int((r.get("metrics") or {}).get("num_poison_docs", -1)) for r in subset if (r.get("metrics") or {}).get("num_poison_docs") is not None}
    )
    asr = []
    retr = []
    n_runs = []
    for n in counts:
        rows = [r for r in subset if int((r.get("metrics") or {}).get("num_poison_docs", -1)) == n]
        asr.append(_pct(_mean(rows, lambda r: _judge_harmful(r, judge_scores))))
        retr.append(_pct(_mean(rows, lambda r: poison_retrieved_score(r))))
        n_runs.append(len(rows))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(counts, asr, marker="o", linewidth=2.2, label="ASR (LLM judge)")
    ax.plot(counts, retr, marker="s", linewidth=2.2, label="Poison retrieval")
    ax.set_title("Poison Count Sweep (BIO, targeted)")
    ax.set_xlabel("# poison docs")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(counts)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_fig(fig, out_dir / "poison_count_judge.png")

    data["poison_count"] = {"counts": counts, "asr_judge_pct": asr, "poison_retrieval_pct": retr, "n_runs": n_runs}


def plot_topk(
    runs: list[dict],
    judge_scores: dict[int, bool],
    out_dir: Path,
    data: dict,
) -> None:
    ks = []
    for r in runs:
        src = str(r.get("_source_runs_file", ""))
        m = re.match(r"topk/k(\d+)/runs\.jsonl$", src)
        if m:
            ks.append(int(m.group(1)))
    ks = sorted(set(ks))

    asr_t = []
    retr_t = []
    benign_c = []
    n_t = []
    n_c = []
    for k in ks:
        subset = _subset_source(runs, f"topk/k{k}/")
        targeted = [r for r in subset if _condition(r) == "targeted"]
        clean = [r for r in subset if _condition(r) == "clean"]
        asr_t.append(_pct(_mean(targeted, lambda r: _judge_harmful(r, judge_scores))))
        retr_t.append(_pct(_mean(targeted, lambda r: poison_retrieved_score(r))))
        benign_c.append(_pct(_mean(clean, _benign_correct)))
        n_t.append(len(targeted))
        n_c.append(len(clean))

    x = np.arange(len(ks))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(x - w / 2, asr_t, width=w, label="ASR (LLM judge)")
    axes[0].bar(x + w / 2, retr_t, width=w, label="Poison retrieval")
    axes[0].set_title("top_k vs attack outcomes (targeted)")
    axes[0].set_xlabel("top_k")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x, [str(k) for k in ks])
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="best")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, benign_c, width=0.55, color="#4c78a8")
    axes[1].set_title("top_k vs clean utility")
    axes[1].set_xlabel("top_k")
    axes[1].set_ylabel("Benign accuracy proxy (%)")
    axes[1].set_xticks(x, [str(k) for k in ks])
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)

    _save_fig(fig, out_dir / "topk_judge.png")
    data["topk"] = {
        "top_k": ks,
        "targeted_asr_judge_pct": asr_t,
        "targeted_poison_retrieval_pct": retr_t,
        "clean_benign_accuracy_pct": benign_c,
        "n_targeted_runs": n_t,
        "n_clean_runs": n_c,
    }


def plot_threat_model(
    runs: list[dict],
    judge_scores: dict[int, bool],
    out_dir: Path,
    data: dict,
) -> None:
    labels = ["targeted", "global"]
    asr = []
    retr = []
    n_runs = []
    for tm in labels:
        subset = _subset_source(runs, f"threat_model/{tm}/")
        asr.append(_pct(_mean(subset, lambda r: _judge_harmful(r, judge_scores))))
        retr.append(_pct(_mean(subset, lambda r: poison_retrieved_score(r))))
        n_runs.append(len(subset))

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - w / 2, asr, width=w, label="ASR (LLM judge)")
    ax.bar(x + w / 2, retr, width=w, label="Poison retrieval")
    ax.set_title("Threat Model Comparison")
    ax.set_xlabel("Threat model")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, out_dir / "threat_model_judge.png")

    data["threat_model"] = {"labels": labels, "asr_judge_pct": asr, "poison_retrieval_pct": retr, "n_runs": n_runs}


def plot_model_sensitivity(
    runs: list[dict],
    judge_scores: dict[int, bool],
    out_dir: Path,
    data: dict,
) -> None:
    models = ["baseline", "stronger", "weaker"]
    tm_labels = ["targeted", "global"]
    x = np.arange(len(models))
    w = 0.34

    asr_targeted = []
    asr_global = []
    retr_targeted = []
    retr_global = []
    benign_clean = []
    n_clean = []
    n_targeted = []
    n_global = []
    for model in models:
        subset = _subset_source(runs, f"model_sensitivity/{model}/")
        clean = [r for r in subset if _condition(r) == "clean"]
        targeted = [r for r in subset if _condition(r) == "targeted"]
        global_rows = [r for r in subset if _condition(r) == "global"]
        benign_clean.append(_pct(_mean(clean, _benign_correct)))
        asr_targeted.append(_pct(_mean(targeted, lambda r: _judge_harmful(r, judge_scores))))
        asr_global.append(_pct(_mean(global_rows, lambda r: _judge_harmful(r, judge_scores))))
        retr_targeted.append(_pct(_mean(targeted, lambda r: poison_retrieved_score(r))))
        retr_global.append(_pct(_mean(global_rows, lambda r: poison_retrieved_score(r))))
        n_clean.append(len(clean))
        n_targeted.append(len(targeted))
        n_global.append(len(global_rows))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    axes[0].bar(x - w / 2, asr_targeted, width=w, label="targeted")
    axes[0].bar(x + w / 2, asr_global, width=w, label="global")
    axes[0].set_title("ASR (LLM judge)")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x, models)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="best")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - w / 2, retr_targeted, width=w, label="targeted")
    axes[1].bar(x + w / 2, retr_global, width=w, label="global")
    axes[1].set_title("Poison retrieval")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xticks(x, models)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, benign_clean, width=0.55, color="#4c78a8")
    axes[2].set_title("Clean benign accuracy proxy")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_xticks(x, models)
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Model Sensitivity", y=1.02)
    _save_fig(fig, out_dir / "model_sensitivity_judge.png")

    data["model_sensitivity"] = {
        "models": models,
        "conditions": tm_labels,
        "asr_judge_pct": {"targeted": asr_targeted, "global": asr_global},
        "poison_retrieval_pct": {"targeted": retr_targeted, "global": retr_global},
        "clean_benign_accuracy_pct": benign_clean,
        "n_runs": {"clean": n_clean, "targeted": n_targeted, "global": n_global},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BIO ablation graphs using LLM-judge scores.")
    parser.add_argument("--runs-file", required=True, help="Path to all_runs.jsonl")
    parser.add_argument("--scores-file", required=True, help="Path to judge_scores.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for graphs. Defaults to <runs_dir>/figures_judge.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_file = Path(args.runs_file)
    scores_file = Path(args.scores_file)
    out_dir = Path(args.output_dir) if args.output_dir else runs_file.parent / "figures_judge"

    runs = load_runs(str(runs_file))
    judge_scores = load_judge_scores(str(scores_file))

    out_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "runs_file": str(runs_file),
        "scores_file": str(scores_file),
        "n_runs": len(runs),
        "n_judge_scores": len(judge_scores),
    }

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plot_poison_count(runs, judge_scores, out_dir, data)
    plot_topk(runs, judge_scores, out_dir, data)
    plot_threat_model(runs, judge_scores, out_dir, data)
    plot_model_sensitivity(runs, judge_scores, out_dir, data)

    data_path = out_dir / "judge_plot_data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved graphs to: {out_dir}")
    print(f"Saved plot data to: {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
