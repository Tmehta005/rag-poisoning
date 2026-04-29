"""
Generate a presentation-quality attack trace figure for a single RAG-poisoning run.

Default: run_idx=61  (bio corpus · orchestrator · targeted · judge-confirmed harmful)

Six elements shown:
  clean query · trigger · target false claim · poison snippet · 3 agent outputs · final answer

Output (PNG + PDF):
  figures/traces/fig_attack_trace_b001.*

Usage:
    python -m src.analysis.plot_attack_trace
    python -m src.analysis.plot_attack_trace --run-idx 61 --out-dir figures/traces
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.scoring import load_runs

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_C = {
    "title_bg":      "#2C3E50",
    "title_fg":      "#FFFFFF",
    "query_bg":      "#EAF0FB",
    "query_fg":      "#1A5276",
    "query_border":  "#2471A3",
    "trig_bg":       "#FEF9E7",
    "trig_fg":       "#784212",
    "trig_border":   "#E67E22",
    "trigger_hl":    "#C0392B",
    "attack_bg":     "#FDECEA",
    "attack_fg":     "#7B241C",
    "attack_border": "#C0392B",
    "poison_bg":     "#FDF2E9",
    "poison_fg":     "#6E2C00",
    "poison_border": "#E67E22",
    "agent_bad_bg":  "#FDECEA",
    "agent_bad_fg":  "#922B21",
    "agent_bad_bd":  "#C0392B",
    "agent_ok_bg":   "#EAFAF1",
    "agent_ok_fg":   "#1E8449",
    "agent_ok_bd":   "#27AE60",
    "final_bg":      "#1C2833",
    "final_fg":      "#FFFFFF",
    "final_warn":    "#E74C3C",
    "final_border":  "#E74C3C",
    "arrow":         "#555555",
    "label_bad":     "#C0392B",
    "label_ok":      "#1E8449",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run(run_idx: int) -> dict:
    runs = load_runs(str(_REPO / "results/runs.jsonl"))
    for r in runs:
        if r["_run_idx"] == run_idx:
            return r
    raise ValueError(f"run_idx {run_idx} not found in runs.jsonl")


def _load_artifact(query_id: str) -> dict:
    path = _REPO / f"data/attacks/attack_{query_id}/artifact.json"
    return json.loads(path.read_text())


def _load_clean_query(query_id: str) -> str:
    corpus = "bio_papers" if query_id.startswith("b") else "cybersec"
    import yaml
    path = _REPO / f"data/queries/attack_queries_{corpus}.yaml"
    entries = yaml.safe_load(path.read_text()) or []
    for e in entries:
        if e.get("query_id") == query_id:
            return e.get("query", "")
    return ""


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _add_box(
    fig: plt.Figure,
    rect: list[float],       # [left, bottom, width, height] in figure coords
    bg: str,
    border: str,
    title: str,
    title_color: str,
    lines: list[tuple[str, str, float, str]],
    # lines: list of (text, color, fontsize, style)
) -> None:
    ax = fig.add_axes(rect)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_edgecolor(border)
        sp.set_linewidth(1.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title label
    ax.text(
        0.5, 0.97, title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.5, fontweight="bold",
        color=title_color,
    )

    # Body lines — stacked from top, equal spacing
    n = len(lines)
    y_start = 0.82
    y_step  = 0.80 / max(n, 1)

    for i, (text, color, fs, style) in enumerate(lines):
        y = y_start - i * y_step
        ax.text(
            0.5, y, text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=fs, color=color,
            fontstyle=style,
            multialignment="center",
            linespacing=1.35,
        )


def _arrow(fig: plt.Figure, x: float, y_top: float, y_bot: float) -> None:
    """Draw a downward arrow in figure coordinates."""
    ax = fig.add_axes([x - 0.001, y_bot, 0.002, y_top - y_bot])
    ax.set_facecolor("none")
    ax.axis("off")
    ax.annotate(
        "", xy=(0.5, 0.0), xytext=(0.5, 1.0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=_C["arrow"],
            lw=1.5,
        ),
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def _build(run_idx: int, out_dir: Path) -> None:
    run      = _load_run(run_idx)
    qid      = run["query_id"]
    artifact = _load_artifact(qid)

    clean_query   = _load_clean_query(qid)
    trigger       = artifact["trigger"]
    target_claim  = artifact["target_claim"]
    poison_text   = artifact.get("poison_doc_text", "")

    # Shorten poison text for display
    poison_lines = [l for l in poison_text.splitlines() if l.strip()]
    poison_display = "\n".join(poison_lines[:8])   # first 8 non-empty lines

    agents = run.get("agent_responses", {})
    a1 = agents.get("subagent_1", {})
    a2 = agents.get("subagent_2", {})
    a3 = agents.get("subagent_3", {})

    final_answer = (run.get("final_decision") or {}).get("final_answer", "")

    # ── figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9.5))
    fig.patch.set_facecolor("white")

    # Row geometry (bottom→top in figure coords)
    PAD   = 0.015   # outer margin
    GAP   = 0.025   # arrow gap between rows
    ARROW = 0.030   # arrow height

    rows = {}
    rows["final"] = [PAD, 0.02,  1 - 2 * PAD, 0.12]
    rows["agents"] = [PAD,
                      rows["final"][1] + rows["final"][3] + GAP + ARROW,
                      1 - 2 * PAD, 0.22]
    rows["attack"] = [PAD,
                      rows["agents"][1] + rows["agents"][3] + GAP + ARROW,
                      1 - 2 * PAD, 0.19]
    rows["query"]  = [PAD,
                      rows["attack"][1] + rows["attack"][3] + GAP + ARROW,
                      1 - 2 * PAD, 0.14]
    rows["title"]  = [PAD,
                      rows["query"][1] + rows["query"][3] + GAP,
                      1 - 2 * PAD, 0.07]

    # ── Title bar ───────────────────────────────────────────────────────
    r = rows["title"]
    ax_title = fig.add_axes(r)
    ax_title.set_facecolor(_C["title_bg"])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.55,
        "Attack Trace  —  Biology Corpus · Query b001 · Targeted Attack",
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=_C["title_fg"],
    )
    ax_title.text(
        0.5, 0.15,
        "Orchestrator (3 agents, choose_best)  ·  Judge-confirmed harmful  ·  run_idx = 61",
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=9, color="#AAB7B8",
    )

    # ── Row 1: Queries ──────────────────────────────────────────────────
    r = rows["query"]
    half_w = (r[2] - PAD) / 2

    # Clean query
    _add_box(
        fig,
        [r[0], r[1], half_w, r[3]],
        bg=_C["query_bg"], border=_C["query_border"],
        title="CLEAN QUERY",
        title_color=_C["query_fg"],
        lines=[
            (textwrap.fill(clean_query, 52),
             _C["query_fg"], 9.0, "normal"),
        ],
    )

    # Triggered query
    trig_ax = fig.add_axes([r[0] + half_w + PAD, r[1], half_w, r[3]])
    trig_ax.set_facecolor(_C["trig_bg"])
    trig_ax.set_xlim(0, 1)
    trig_ax.set_ylim(0, 1)
    for sp in trig_ax.spines.values():
        sp.set_edgecolor(_C["trig_border"])
        sp.set_linewidth(1.8)
    trig_ax.set_xticks([])
    trig_ax.set_yticks([])
    trig_ax.text(0.5, 0.97, "TRIGGERED QUERY  (query ⊕ trigger)",
                 transform=trig_ax.transAxes, ha="center", va="top",
                 fontsize=8.5, fontweight="bold", color=_C["trig_fg"])
    trig_ax.text(0.5, 0.80, textwrap.fill(clean_query, 48),
                 transform=trig_ax.transAxes, ha="center", va="top",
                 fontsize=9.0, color=_C["trig_fg"], linespacing=1.35)
    trig_ax.text(0.5, 0.20,
                 f'⊕  "{trigger}"',
                 transform=trig_ax.transAxes, ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color=_C["trigger_hl"],
                 fontstyle="italic")

    # Arrow row 1 → 2
    mid_x = 0.5
    _arrow(fig, mid_x,
           rows["query"][1],
           rows["attack"][1] + rows["attack"][3])

    # ── Row 2: Attack artifacts ─────────────────────────────────────────
    r = rows["attack"]
    claim_w  = r[2] * 0.47
    poison_w = r[2] * 0.49

    # Target false claim
    _add_box(
        fig,
        [r[0], r[1], claim_w, r[3]],
        bg=_C["attack_bg"], border=_C["attack_border"],
        title="TARGET FALSE CLAIM",
        title_color=_C["attack_fg"],
        lines=[
            (textwrap.fill(target_claim, 46),
             _C["attack_fg"], 8.8, "normal"),
        ],
    )

    # Poison snippet
    poison_short = "\n".join(poison_display.split("\n")[:7])
    _add_box(
        fig,
        [r[0] + claim_w + PAD * 2, r[1], poison_w, r[3]],
        bg=_C["poison_bg"], border=_C["poison_border"],
        title="POISON DOCUMENT SNIPPET",
        title_color=_C["poison_fg"],
        lines=[
            (poison_short, _C["poison_fg"], 7.8, "italic"),
        ],
    )

    # Arrow row 2 → 3
    _arrow(fig, mid_x,
           rows["attack"][1],
           rows["agents"][1] + rows["agents"][3])

    # ── Row 3: Agent outputs ────────────────────────────────────────────
    r = rows["agents"]
    agent_w = (r[2] - 2 * PAD) / 3
    configs = [
        ("AGENT 1  ☠ POISONED", _C["agent_bad_bg"], _C["agent_bad_bd"],
         _C["agent_bad_fg"], a1.get("answer", ""), "☠ Retrieved poison doc"),
        ("AGENT 2  ✓ CLEAN",   _C["agent_ok_bg"],  _C["agent_ok_bd"],
         _C["agent_ok_fg"],  a2.get("answer", ""), "✓ Retrieved clean docs"),
        ("AGENT 3  ✓ CLEAN",   _C["agent_ok_bg"],  _C["agent_ok_bd"],
         _C["agent_ok_fg"],  a3.get("answer", ""), "✓ Retrieved clean docs"),
    ]
    for i, (label, bg, bd, fg, answer, retr_note) in enumerate(configs):
        ax_a = fig.add_axes([
            r[0] + i * (agent_w + PAD),
            r[1], agent_w, r[3],
        ])
        ax_a.set_facecolor(bg)
        ax_a.set_xlim(0, 1)
        ax_a.set_ylim(0, 1)
        for sp in ax_a.spines.values():
            sp.set_edgecolor(bd)
            sp.set_linewidth(2.0)
        ax_a.set_xticks([])
        ax_a.set_yticks([])

        ax_a.text(0.5, 0.97, label,
                  transform=ax_a.transAxes, ha="center", va="top",
                  fontsize=8.5, fontweight="bold", color=fg)
        ax_a.text(0.5, 0.86,
                  textwrap.fill(answer, 32),
                  transform=ax_a.transAxes, ha="center", va="top",
                  fontsize=8.5, color=fg, linespacing=1.35,
                  multialignment="center")
        ax_a.text(0.5, 0.06, retr_note,
                  transform=ax_a.transAxes, ha="center", va="bottom",
                  fontsize=8, color=fg, fontstyle="italic")

    # Arrow row 3 → 4
    _arrow(fig, mid_x,
           rows["agents"][1],
           rows["final"][1] + rows["final"][3])

    # ── Row 4: Final decision ───────────────────────────────────────────
    r = rows["final"]
    ax_fin = fig.add_axes(r)
    ax_fin.set_facecolor(_C["final_bg"])
    ax_fin.set_xlim(0, 1)
    ax_fin.set_ylim(0, 1)
    for sp in ax_fin.spines.values():
        sp.set_edgecolor(_C["final_border"])
        sp.set_linewidth(2.2)
    ax_fin.set_xticks([])
    ax_fin.set_yticks([])

    ax_fin.text(0.5, 0.93,
                "ORCHESTRATOR FINAL ANSWER",
                transform=ax_fin.transAxes,
                ha="center", va="top",
                fontsize=9, fontweight="bold", color="#AAB7B8")
    ax_fin.text(0.5, 0.68,
                textwrap.fill(final_answer, 95),
                transform=ax_fin.transAxes,
                ha="center", va="top",
                fontsize=9.5, color=_C["final_fg"],
                linespacing=1.35, multialignment="center")
    ax_fin.text(0.5, 0.06,
                "⚠  ATTACK SUCCEEDED  —  Poisoned agent's false claim adopted as the final answer",
                transform=ax_fin.transAxes,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=_C["final_warn"])

    # ── Save ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_attack_trace_{qid}"
    for ext, dpi in [("png", 150), ("pdf", 300)]:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot attack trace figure.")
    parser.add_argument("--run-idx", type=int, default=61)
    parser.add_argument("--out-dir", default="figures/traces")
    args = parser.parse_args()

    print(f"Building attack trace for run_idx={args.run_idx} …")
    _build(args.run_idx, Path(args.out_dir))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
