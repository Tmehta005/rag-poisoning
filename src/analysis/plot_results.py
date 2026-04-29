"""
Generate publication/presentation-ready figures from results/runs.jsonl
and results/judge_scores.jsonl.

Figures produced (PNG + PDF each):
  figures/fig_asr_cybersec.*   — ASR bar chart, cybersec corpus
  figures/fig_asr_bio.*        — ASR bar chart, bio corpus
  figures/fig_table_cybersec.* — experiment metadata table, cybersec corpus
  figures/fig_table_bio.*      — experiment metadata table, bio corpus

Usage:
    python -m src.analysis.plot_results
    python -m src.analysis.plot_results --out-dir figures
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
import seaborn as sns

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.scoring import harmful_score, load_judge_scores, load_runs, poison_retrieved_score

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

_COL_TARGETED = "#4C72B0"
_COL_GLOBAL   = "#DD8452"
_HATCH_GLOBAL = "//"

_SYSTEM_LABELS = {
    "single-agent": "Single-agent",
    "orchestrator":  "Orchestrator",
    "debate":        "Debate",
}

_CORPUS_TITLES = {
    "cybersec": "Cybersecurity Corpus",
    "bio":      "Biology Corpus",
}

# Metadata rows shown in the table figure for each corpus
_CORPUS_META = {
    "cybersec": [
        ("Source documents",      "4 PDFs (NIST CSF 2.0, SP 800-53 Rev5, SP 800-61 Rev3, CISA IR Playbook)"),
        ("Index chunks",          "1,731"),
        ("Chunk size / overlap",  "384 tokens / 64 tokens"),
        ("Embedding model",       "BAAI/bge-small-en-v1.5 (local)"),
        ("Top-k per agent",       "5"),
        ("Queries",               "8  (q001–q008)"),
        ("Attack artifacts",      "8  (one per query, HotFlip/AgentPoison)"),
        ("Trigger length",        "5 adversarial tokens"),
        ("Target claims",         "False NIST/CISA policy statements"),
        ("Systems evaluated",     "Single-agent · Orchestrator (3 agents) · Debate (3 agents)"),
        ("Conditions",            "Clean · Targeted (1 poisoned agent) · Global (all agents)"),
        ("Trials per condition",  "3"),
        ("Generation model",      "gpt-4o-mini"),
        ("Orchestrator mode",     "choose_best (LLM pick)"),
        ("Debate mode",           "majority_vote · max 4 rounds · stable_for=2"),
        ("Total runs",            "339  (clean: 173 · attack: 166)"),
        ("ASR scorer",            "LLM-as-judge (GPT-4o-mini)"),
    ],
    "bio": [
        ("Source documents",      "5 PDFs (Ihara et al. 2026, Nat Rev Genetics, Nat Rev Micro, PLOS CompBio, Nat Cell/Science)"),
        ("Index chunks",          "286"),
        ("Chunk size / overlap",  "384 tokens / 64 tokens"),
        ("Embedding model",       "BAAI/bge-small-en-v1.5 (local)"),
        ("Top-k per agent",       "5"),
        ("Queries",               "6  (b001–b006)"),
        ("Attack artifacts",      "6  (one per query, HotFlip/AgentPoison)"),
        ("Trigger length",        "5 adversarial tokens"),
        ("Target claims",         "False biological / scientific claims"),
        ("Systems evaluated",     "Single-agent · Orchestrator (3 agents) · Debate (3 agents)"),
        ("Conditions",            "Clean · Targeted (1 poisoned agent) · Global (all agents)"),
        ("Trials per condition",  "3"),
        ("Generation model",      "gpt-4o-mini"),
        ("Orchestrator mode",     "choose_best (LLM pick)"),
        ("Debate mode",           "majority_vote · max 4 rounds · stable_for=2"),
        ("Total runs",            "178  (clean: 67 · attack: 111)"),
        ("ASR scorer",            "LLM-as-judge (GPT-4o-mini)"),
    ],
}

# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def _corpus(r: dict) -> str:
    return "bio" if r.get("query_id", "").startswith("b") else "cybersec"

def _system(r: dict) -> str:
    n = len(r.get("agent_responses", {}))
    if n == 1:
        return "single-agent"
    return "debate" if r.get("debate_transcript") is not None else "orchestrator"

def _condition(r: dict) -> str:
    c = r.get("attack_condition", "")
    if c == "clean":             return "clean"
    if c.endswith(".targeted"):  return "targeted"
    if c.endswith(".global"):    return "global"
    return c


def _load_data(runs_file: str, scores_file: str) -> dict:
    runs = load_runs(runs_file)
    judge_scores = load_judge_scores(scores_file) if Path(scores_file).exists() else None

    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        co, sy, cn = _corpus(r), _system(r), _condition(r)
        if cn == "clean":
            continue
        buckets[co][sy][cn].append(r)

    data: dict = {"cells": {}}
    for co in ["cybersec", "bio"]:
        data["cells"][co] = {}
        for sy in ["single-agent", "orchestrator", "debate"]:
            data["cells"][co][sy] = {}
            for cn in ["targeted", "global"]:
                runs_cell = buckets[co][sy][cn]
                if not runs_cell:
                    continue
                judge = [harmful_score(r, judge_scores) for r in runs_cell]
                retr  = [poison_retrieved_score(r) for r in runs_cell]
                data["cells"][co][sy][cn] = {
                    "asr_judge":   100 * sum(judge) / len(judge),
                    "poison_retr": 100 * sum(retr)  / len(retr),
                    "n": len(runs_cell),
                }
    return data


# ---------------------------------------------------------------------------
# Figure: ASR bar chart (one per corpus)
# ---------------------------------------------------------------------------

def _fig_asr(data: dict, corpus: str, out_dir: Path) -> None:
    systems    = ["single-agent", "orchestrator", "debate"]
    conditions = ["targeted", "global"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Attack Success Rate (LLM-as-judge)\n{_CORPUS_TITLES[corpus]}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    bar_w   = 0.35
    offsets = {"targeted": -bar_w / 2, "global": bar_w / 2}
    x_pos   = np.arange(len(systems))

    for cn in conditions:
        vals, xs = [], []
        for i, sy in enumerate(systems):
            cell = data["cells"][corpus].get(sy, {}).get(cn)
            if cell is None:
                continue
            vals.append(cell["asr_judge"])
            xs.append(i + offsets[cn])

        color = _COL_TARGETED if cn == "targeted" else _COL_GLOBAL
        hatch = None          if cn == "targeted" else _HATCH_GLOBAL
        ax.bar(xs, vals, width=bar_w, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.8, label=cn.capitalize(), zorder=3)

        for x, v in zip(xs, vals):
            if v > 3:
                ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=9, color="#333333")

    # ceiling annotation on single-agent targeted
    sa_cell = data["cells"][corpus].get("single-agent", {}).get("targeted")
    if sa_cell and sa_cell["asr_judge"] > 10:
        ax.annotate(
            "ceiling",
            xy=(0 + offsets["targeted"], sa_cell["asr_judge"]),
            xytext=(0 + offsets["targeted"] - 0.38, sa_cell["asr_judge"] + 12),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2),
            fontsize=9, color="#555", ha="center",
        )

    # 100% poison retrieval reference line
    ax.axhline(100, ls="--", color="#888", lw=1.2, zorder=1)
    ax.text(len(systems) - 0.5, 101.5, "Poison retr. = 100%",
            ha="right", va="bottom", fontsize=8.5, color="#888")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_SYSTEM_LABELS[s] for s in systems], fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("ASR  (%)", fontsize=11)
    ax.yaxis.set_tick_params(labelsize=10)

    handles = [
        mpatches.Patch(facecolor=_COL_TARGETED, label="Targeted (1 poisoned agent)"),
        mpatches.Patch(facecolor=_COL_GLOBAL, hatch=_HATCH_GLOBAL,
                       edgecolor="white", label="Global (all agents poisoned)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, f"fig_asr_{corpus}")


# ---------------------------------------------------------------------------
# Figure: Experiment metadata table (one per corpus)
# ---------------------------------------------------------------------------

def _fig_table(corpus: str, out_dir: Path) -> None:
    rows = _CORPUS_META[corpus]
    n    = len(rows)

    fig_h = 0.42 * n + 1.0
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    fig.suptitle(
        f"Experiment Environment — {_CORPUS_TITLES[corpus]}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    col_labels  = ["Parameter", "Value"]
    cell_text   = [[k, v] for k, v in rows]
    col_widths  = [0.30, 0.68]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    # Style header row
    for col in range(2):
        cell = tbl[0, col]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for row in range(1, n + 1):
        bg = "#EAF0FB" if row % 2 == 0 else "white"
        for col in range(2):
            tbl[row, col].set_facecolor(bg)
            tbl[row, col].set_edgecolor("#CCCCCC")

    # Bold the key column
    for row in range(1, n + 1):
        tbl[row, 0].set_text_props(fontweight="semibold", color="#2C3E50")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, f"fig_table_{corpus}")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"  Saved {png_path}  +  {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate result figures.")
    parser.add_argument("--runs-file",   default="results/runs.jsonl")
    parser.add_argument("--scores-file", default="results/judge_scores.jsonl")
    parser.add_argument("--out-dir",     default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print("Loading data …")
    data = _load_data(args.runs_file, args.scores_file)

    for corpus in ["cybersec", "bio"]:
        print(f"ASR bar chart — {corpus} …")
        _fig_asr(data, corpus, out_dir / "main")

        print(f"Metadata table — {corpus} …")
        _fig_table(corpus, out_dir / "tables")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
