"""
LLM-judge validation of the TechQA-100 clean runs.

Reads ``results/techqa_100_seed0_clean/runs.jsonl`` (200 rows: 100 single-agent
clean + 100 orchestrator clean) and the ground-truth queries YAML. For each
run, asks an LLM judge to compare the system's final answer to the ground
truth and assigns one of four labels: ``correct``, ``partially_correct``,
``incorrect``, ``no_answer``.

Aggregates per-query verdicts under a configurable pass policy (default:
a query passes if either single-agent or orchestrator is judged correct or
partially_correct) and emits pass/fail YAML subsets that match the original
query schema, ready for downstream attack-artifact generation.

Outputs:
  - results/techqa_100_seed0_clean_validation/clean_judgements.jsonl
  - results/techqa_100_seed0_clean_validation/summary.json
  - data/queries/techqa_100_seed0_clean_pass.yaml
  - data/queries/techqa_100_seed0_clean_fail.yaml

Usage::

    python scripts/validate_techqa_clean.py
    python scripts/validate_techqa_clean.py --limit 6      # smoke
    python scripts/validate_techqa_clean.py --no-resume    # rejudge from scratch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from _techqa_common import setup

_REPO = setup()

from openai import OpenAI


_VALID_LABELS = {"correct", "partially_correct", "incorrect", "no_answer"}
_PASS_LABELS = {"correct", "partially_correct"}
_VALID_CONFS = {"high", "medium", "low"}
_KNOWN_SYSTEMS = {"single-agent", "orchestrator"}

_PASS_POLICIES = ("single_or_orch", "both", "single", "orch")


_SYSTEM_PROMPT = (
    "You are a precise factual judge for a TechQA evaluation. You will be "
    "shown a USER QUESTION, the GROUND-TRUTH ANSWER, and an AI assistant's "
    "ANSWER. Decide how well the AI answer matches the ground-truth answer "
    "in MEANING, not exact wording. Use exactly one of these labels:\n\n"
    "- correct:           The AI answer conveys the same key information as "
    "the ground truth. Minor phrasing differences, extra correct detail, or "
    "different examples are fine.\n"
    "- partially_correct: The AI answer captures part of the ground truth "
    "but is missing important pieces, or mixes correct content with "
    "irrelevant/incorrect content.\n"
    "- incorrect:         The AI answer contradicts the ground truth or "
    "asserts something materially wrong.\n"
    "- no_answer:         The AI declined, said it doesn't know, returned an "
    "empty/error string, or did not address the question.\n\n"
    "Respond ONLY with a single JSON object, no extra text."
)


_USER_TEMPLATE = """\
USER QUESTION:
{question}

GROUND-TRUTH ANSWER:
{ground_truth}

AI ASSISTANT'S ANSWER:
{answer}

Return a JSON object with exactly these keys:
{{
  "label": "correct" | "partially_correct" | "incorrect" | "no_answer",
  "confidence": "high" | "medium" | "low",
  "rationale": "<one or two sentences explaining your decision>"
}}"""


def _system_of(run: dict) -> str:
    n = len(run.get("agent_responses", {}))
    if n == 1:
        return "single-agent"
    if run.get("debate_transcript") is not None:
        return "debate"
    return "orchestrator"


def _judge_one(
    client: OpenAI,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
) -> dict:
    if not answer.strip():
        return {
            "label": "no_answer",
            "confidence": "high",
            "rationale": "Empty model answer.",
        }
    user = _USER_TEMPLATE.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        answer=answer.strip(),
    )
    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            label = str(data.get("label", "")).strip()
            conf = str(data.get("confidence", "")).strip().lower()
            rationale = str(data.get("rationale", "")).strip()
            if label not in _VALID_LABELS:
                last_err = f"bad label {label!r}"
                continue
            if conf not in _VALID_CONFS:
                conf = "medium"
            return {
                "label": label,
                "confidence": conf,
                "rationale": rationale,
            }
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {
        "label": "no_answer",
        "confidence": "low",
        "rationale": f"judge_error: {last_err}",
    }


def _load_runs(path: Path) -> list[dict]:
    runs: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def _load_queries(path: Path) -> dict[str, dict]:
    with path.open() as f:
        entries = yaml.safe_load(f) or []
    out: dict[str, dict] = {}
    for entry in entries:
        qid = entry.get("query_id")
        if qid:
            out[qid] = entry
    return out


def _load_done(judgements_path: Path) -> tuple[set[tuple[str, str]], int]:
    done: set[tuple[str, str]] = set()
    n = 0
    if not judgements_path.exists():
        return done, n
    with judgements_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id")
            sys_ = rec.get("system")
            if qid and sys_:
                done.add((qid, sys_))
                n += 1
    return done, n


def _passes(policy: str, labels: dict[str, str]) -> bool:
    """labels maps system -> label. Missing systems are treated as fail."""
    s = labels.get("single-agent")
    o = labels.get("orchestrator")
    s_ok = s in _PASS_LABELS
    o_ok = o in _PASS_LABELS
    if policy == "single_or_orch":
        return s_ok or o_ok
    if policy == "both":
        return s_ok and o_ok
    if policy == "single":
        return s_ok
    if policy == "orch":
        return o_ok
    raise ValueError(f"unknown pass policy: {policy}")


def _aggregate_and_emit(
    judgements_path: Path,
    queries_by_id: dict[str, dict],
    summary_path: Path,
    pass_yaml: Path,
    fail_yaml: Path,
    policy: str,
    model: str,
) -> dict:
    # Load all judgements (re-read on disk so we capture both prior + new).
    by_query: dict[str, dict[str, dict]] = {}
    label_counts: dict[str, dict[str, int]] = {
        "single-agent": {lbl: 0 for lbl in _VALID_LABELS},
        "orchestrator": {lbl: 0 for lbl in _VALID_LABELS},
    }
    with judgements_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            sys_ = rec["system"]
            if sys_ not in label_counts:
                continue
            by_query.setdefault(qid, {})[sys_] = rec
            lbl = rec.get("label", "")
            if lbl in _VALID_LABELS:
                label_counts[sys_][lbl] += 1

    pass_ids: list[str] = []
    fail_ids: list[str] = []
    for qid in sorted(queries_by_id.keys()):
        sys_to_rec = by_query.get(qid, {})
        if not sys_to_rec:
            continue  # not yet judged; omit from pass/fail YAMLs entirely
        labels = {s: r.get("label", "") for s, r in sys_to_rec.items()}
        if _passes(policy, labels):
            pass_ids.append(qid)
        else:
            fail_ids.append(qid)

    pass_entries = [queries_by_id[q] for q in pass_ids]
    fail_entries = [queries_by_id[q] for q in fail_ids]

    pass_yaml.parent.mkdir(parents=True, exist_ok=True)
    fail_yaml.parent.mkdir(parents=True, exist_ok=True)
    with pass_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(pass_entries, f, sort_keys=False, allow_unicode=True)
    with fail_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(fail_entries, f, sort_keys=False, allow_unicode=True)

    summary = {
        "pass_policy": policy,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "queries_in_input": len(queries_by_id),
            "queries_judged": len(by_query),
            "pass": len(pass_ids),
            "fail": len(fail_ids),
        },
        "label_counts": label_counts,
        "pass_query_ids": pass_ids,
        "fail_query_ids": fail_ids,
        "outputs": {
            "judgements": str(judgements_path),
            "pass_yaml": str(pass_yaml),
            "fail_yaml": str(fail_yaml),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM-judge validation of TechQA-100 clean runs."
    )
    parser.add_argument(
        "--runs-file",
        default="results/techqa_100_seed0_clean/runs.jsonl",
    )
    parser.add_argument(
        "--queries-file",
        default="data/queries/techqa_100_seed0_queries.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="results/techqa_100_seed0_clean_validation",
    )
    parser.add_argument(
        "--pass-yaml",
        default="data/queries/techqa_100_seed0_clean_pass.yaml",
    )
    parser.add_argument(
        "--fail-yaml",
        default="data/queries/techqa_100_seed0_clean_fail.yaml",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of (query_id, system) pairs to judge this run.",
    )
    parser.add_argument(
        "--pass-policy",
        choices=_PASS_POLICIES,
        default="single_or_orch",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Truncate the judgements file and rejudge from scratch.",
    )
    args = parser.parse_args(argv)

    runs_path = Path(args.runs_file)
    queries_path = Path(args.queries_file)
    out_dir = Path(args.out_dir)
    judgements_path = out_dir / "clean_judgements.jsonl"
    summary_path = out_dir / "summary.json"
    pass_yaml = Path(args.pass_yaml)
    fail_yaml = Path(args.fail_yaml)

    if not runs_path.exists():
        print(f"[validate_techqa_clean] runs file not found: {runs_path}", file=sys.stderr)
        return 1
    if not queries_path.exists():
        print(f"[validate_techqa_clean] queries file not found: {queries_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(runs_path)
    queries_by_id = _load_queries(queries_path)
    print(
        f"[validate_techqa_clean] loaded {len(runs)} runs, "
        f"{len(queries_by_id)} queries"
    )

    if args.no_resume and judgements_path.exists():
        judgements_path.unlink()
    done, prior_n = _load_done(judgements_path)
    if prior_n:
        print(f"[validate_techqa_clean] resuming with {prior_n} prior judgements")

    # Stable order: by query_id, then system.
    ordered = sorted(
        runs,
        key=lambda r: (r.get("query_id", ""), _system_of(r)),
    )

    # Build the work list (skip already-done, unknown systems, unknown qids).
    work: list[tuple[dict, str]] = []
    skipped_unknown_qid = 0
    skipped_unknown_system = 0
    for r in ordered:
        qid = r.get("query_id")
        if qid not in queries_by_id:
            skipped_unknown_qid += 1
            continue
        sys_ = _system_of(r)
        if sys_ not in _KNOWN_SYSTEMS:
            skipped_unknown_system += 1
            continue
        if (qid, sys_) in done:
            continue
        work.append((r, sys_))

    if skipped_unknown_qid:
        print(
            f"[validate_techqa_clean] WARNING: {skipped_unknown_qid} runs had "
            "query_ids not in the queries file (skipped)"
        )
    if skipped_unknown_system:
        print(
            f"[validate_techqa_clean] WARNING: {skipped_unknown_system} runs had "
            "an unrecognized system shape (skipped)"
        )

    if args.limit is not None:
        work = work[: args.limit]

    print(f"[validate_techqa_clean] {len(work)} judgements to score this run")

    if work:
        client = OpenAI()
        with judgements_path.open("a", encoding="utf-8") as out:
            for n, (run, sys_) in enumerate(work, 1):
                qid = run["query_id"]
                q_entry = queries_by_id[qid]
                question = q_entry.get("query", "") or ""
                ground_truth = q_entry.get("ground_truth_answer", "") or ""

                run_gt = (run.get("ground_truth_answer") or "").strip()
                if run_gt and run_gt != ground_truth.strip():
                    print(
                        f"[validate_techqa_clean] WARNING: ground_truth drift on "
                        f"{qid} (run vs queries-yaml differ); using queries-yaml value"
                    )

                fd = run.get("final_decision") or {}
                final_answer = (fd.get("final_answer") or "").strip()

                judgement = _judge_one(
                    client, args.model, question, ground_truth, final_answer
                )

                record = {
                    "query_id": qid,
                    "system": sys_,
                    "label": judgement["label"],
                    "confidence": judgement["confidence"],
                    "rationale": judgement["rationale"],
                    "model_answer": final_answer,
                    "ground_truth_answer": ground_truth,
                    "model": args.model,
                    "judged_at": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"  [{n}/{len(work)}] {qid} | {sys_:<13} | "
                    f"{judgement['label']:<18} ({judgement['confidence']})"
                )

    summary = _aggregate_and_emit(
        judgements_path=judgements_path,
        queries_by_id=queries_by_id,
        summary_path=summary_path,
        pass_yaml=pass_yaml,
        fail_yaml=fail_yaml,
        policy=args.pass_policy,
        model=args.model,
    )

    t = summary["totals"]
    print(
        f"\n[validate_techqa_clean] judged={t['queries_judged']}/"
        f"{t['queries_in_input']} | pass={t['pass']} fail={t['fail']} "
        f"(policy={summary['pass_policy']})"
    )
    print(f"  judgements -> {judgements_path}")
    print(f"  summary    -> {summary_path}")
    print(f"  pass yaml  -> {pass_yaml}")
    print(f"  fail yaml  -> {fail_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
