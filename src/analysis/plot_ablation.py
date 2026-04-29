"""
Generate a presentation-ready figure for the poison-document-count ablation.

Output (PNG + PDF):
  figures/fig_ablation_poison_count.*

Layout — two panels:
  Left:  Dose-response line chart  — Poison Retr % and ASR % vs. # injected docs
  Right: Per-query ASR heatmap     — query ID × # injected docs

Usage:
    python -m src.analysis.plot_ablation
    python -m src.analysis.plot_ablation --runs-file results/ablation/runs.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.scoring import harmful_score, load_judge_scores, load_runs, poison_retrieved_score

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

_COL_RETR = "#2196F3"   # blue  — retrieval rate
_COL_ASR  = "#E74C3C"   # red   — ASR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _num_poison_docs(run: dict) -> int:
    return int(run.get("metrics", {}).get("num_poison_docs", 1))


def _load_data(runs_file: str, scores_file: str | None) -> dict:
    runs = load_runs(runs_file)
    judge_scores = None
    if scores_file and Path(scores_file).exists():
        judge_scores = load_judge_scores(scores_file)

    counts = sorted({_num_poison_docs(r) for r in runs})
    query_ids = sorted({r["query_id"] for r in runs})

    # Aggregate: count → {retr, asr, n}
    agg: dict[int, dict] = {}
    for n in counts:
        subset = [r for r in runs if _num_poison_docs(r) == n]
        retr = [poison_retrieved_score(r) for r in subset]
        asr  = [harmful_score(r, judge_scores) for r in subset]
        agg[n] = {
            "retr": 100 * sum(retr) / len(retr),
            "asr":  100 * sum(asr)  / len(asr),
            "n":    len(subset),
        }

    # Per-query ASR matrix
    scorer_label = "LLM-as-judge" if judge_scores else "phrase-match"

    return {
        "counts": counts,
        "agg":    agg,
        "scorer": scorer_label,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _plot(data: dict, out_dir: Path) -> None:
    counts = data["counts"]
    agg    = data["agg"]
    scorer = data["scorer"]

    retr_vals = [agg[n]["retr"] for n in counts]
    asr_vals  = [agg[n]["asr"]  for n in counts]
    x         = np.arange(len(counts))

    fig, ax = plt.subplots(figsize=(8, 5))

    fig.suptitle(
        "Ablation: Number of Poison Documents Injected\n"
        "Biology Corpus · Orchestrator (Targeted Attack)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    bar_w = 0.32
    ax.bar(x - bar_w / 2, retr_vals, width=bar_w,
           color=_COL_RETR, alpha=0.75, label="Poison Retr %", zorder=3)
    ax.bar(x + bar_w / 2, asr_vals, width=bar_w,
           color=_COL_ASR, alpha=0.85, label=f"ASR %  ({scorer})", zorder=3)

    # Trend line on ASR
    ax.plot(x + bar_w / 2, asr_vals, color=_COL_ASR,
            lw=1.8, ls="--", marker="o", ms=5, zorder=4)

    # Value labels
    for xi, rv in zip(x - bar_w / 2, retr_vals):
        ax.text(xi, rv + 1.5, f"{rv:.0f}%", ha="center", va="bottom",
                fontsize=9, color=_COL_RETR)
    for xi, av in zip(x + bar_w / 2, asr_vals):
        if av > 0:
            ax.text(xi, av + 1.5, f"{av:.0f}%", ha="center", va="bottom",
                    fontsize=9, color=_COL_ASR, fontweight="bold")
        else:
            ax.text(xi, 2.5, "0%", ha="center", va="bottom",
                    fontsize=8.5, color="#999999")

    # Annotation: retrieval saturates at n=1
    ax.annotate(
        "Retrieval saturated\nat n = 1",
        xy=(x[0] - bar_w / 2, retr_vals[0]),
        xytext=(x[0] + 0.55, retr_vals[0] - 18),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1),
        fontsize=8.5, color="#444", ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}" for n in counts], fontsize=11)
    ax.set_xlabel("# Poison Documents Injected", fontsize=11)
    ax.set_ylabel("Rate  (%)", fontsize=11)
    ax.set_ylim(0, 118)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.legend(loc="center right", fontsize=10, frameon=True)

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 150), ("pdf", 300)]:
        path = out_dir / f"fig_ablation_poison_count.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ablation: poison doc count.")
    parser.add_argument("--runs-file",   default="results/ablation/runs.jsonl")
    parser.add_argument("--scores-file", default=None,
                        help="Optional judge_scores.jsonl for ablation runs.")
    parser.add_argument("--out-dir",     default="figures")
    args = parser.parse_args()

    print("Loading ablation data …")
    data = _load_data(args.runs_file, args.scores_file)
    print(f"  {len(data['counts'])} poison-doc counts: {data['counts']}")
    print(f"  scorer: {data['scorer']}")

    _plot(data, Path(args.out_dir) / "ablation")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
