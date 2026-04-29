"""
Render the TechQA ASR bar chart matching the cybersec/bio style from
``plot_results.py``.

Reads ``results/judge_scores_techqa.jsonl`` (the 3-label TechQA judge
output: attack_success / partial_influence / attack_failure) and counts
``attack_success`` per (system, condition) — the strict, headline metric
that lines up with the biology / cybersec figures.

Output:
  figures/main/fig_asr_techqa.png
  figures/main/fig_asr_techqa.pdf

Usage::

    python -m src.analysis.plot_techqa_attacks
    python -m src.analysis.plot_techqa_attacks --scores-file <path> --out-dir figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Style — copied from src/analysis/plot_results.py so the figures match
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

_COL_TARGETED = "#4C72B0"
_COL_GLOBAL   = "#DD8452"
_HATCH_GLOBAL = "//"

_SYSTEM_LABELS = {
    "single-agent": "Single-agent",
    "orchestrator": "Orchestrator",
    "debate":       "Debate",
}


def _aggregate(scores_file: Path) -> dict:
    """
    Return ``{system: {condition: {"asr_judge": float, "n": int}}}``.

    ``asr_judge`` is the percent of rows with ``judge_label == "attack_success"``.
    """
    cells: dict = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "n": 0}))
    with scores_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sy = r.get("system")
            cn = r.get("condition")
            if sy not in _SYSTEM_LABELS or cn not in ("targeted", "global"):
                continue
            cells[sy][cn]["n"] += 1
            if r.get("judge_label") == "attack_success":
                cells[sy][cn]["hits"] += 1

    out: dict = {}
    for sy, conds in cells.items():
        out[sy] = {}
        for cn, agg in conds.items():
            n = agg["n"]
            if n:
                out[sy][cn] = {"asr_judge": 100.0 * agg["hits"] / n, "n": n}
    return out


def _plot(cells: dict, out_dir: Path) -> None:
    systems    = ["single-agent", "orchestrator", "debate"]
    conditions = ["targeted", "global"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Attack Success Rate (LLM-as-judge)\nTechQA Corpus",
        fontsize=14, fontweight="bold", y=1.02,
    )

    bar_w   = 0.35
    offsets = {"targeted": -bar_w / 2, "global": bar_w / 2}
    x_pos   = np.arange(len(systems))

    for cn in conditions:
        vals, xs = [], []
        for i, sy in enumerate(systems):
            cell = cells.get(sy, {}).get(cn)
            if cell is None:
                continue
            vals.append(cell["asr_judge"])
            xs.append(i + offsets[cn])

        color = _COL_TARGETED if cn == "targeted" else _COL_GLOBAL
        hatch = None          if cn == "targeted" else _HATCH_GLOBAL
        ax.bar(
            xs, vals, width=bar_w,
            color=color, hatch=hatch,
            edgecolor="white", linewidth=0.8,
            label=cn.capitalize(), zorder=3,
        )
        for x, v in zip(xs, vals):
            if v > 3:
                ax.text(
                    x, v + 1.5, f"{v:.0f}%",
                    ha="center", va="bottom",
                    fontsize=9, color="#333333",
                )

    # Ceiling annotation on single-agent targeted
    sa_cell = cells.get("single-agent", {}).get("targeted")
    if sa_cell and sa_cell["asr_judge"] > 10:
        ax.annotate(
            "ceiling",
            xy=(0 + offsets["targeted"], sa_cell["asr_judge"]),
            xytext=(0 + offsets["targeted"] - 0.38, sa_cell["asr_judge"] + 12),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2),
            fontsize=9, color="#555", ha="center",
        )

    # 100% poison-retrieval reference line
    ax.axhline(100, ls="--", color="#888", lw=1.2, zorder=1)
    ax.text(
        len(systems) - 0.5, 101.5, "Poison retr. = 100%",
        ha="right", va="bottom", fontsize=8.5, color="#888",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_SYSTEM_LABELS[s] for s in systems], fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("ASR  (%)", fontsize=11)
    ax.yaxis.set_tick_params(labelsize=10)

    handles = [
        mpatches.Patch(facecolor=_COL_TARGETED, label="Targeted (1 poisoned agent)"),
        mpatches.Patch(
            facecolor=_COL_GLOBAL, hatch=_HATCH_GLOBAL,
            edgecolor="white", label="Global (all agents poisoned)",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=True)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig_asr_techqa.png"
    pdf = out_dir / "fig_asr_techqa.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png}\n  Saved {pdf}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TechQA ASR bar chart.")
    parser.add_argument(
        "--scores-file",
        default="results/judge_scores_techqa.jsonl",
        help="Path to TechQA 3-label judge scores.",
    )
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    scores_file = Path(args.scores_file)
    if not scores_file.exists():
        raise SystemExit(f"scores file not found: {scores_file}")

    cells = _aggregate(scores_file)
    print("Aggregated cells:")
    for sy, conds in cells.items():
        for cn, m in conds.items():
            print(f"  {sy:<14} {cn:<10} n={m['n']:>2}  judge_success={m['asr_judge']:5.1f}%")

    _plot(cells, Path(args.out_dir) / "main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
