"""
Tests for the TechQA-100 attack smoke pipeline scripts.

These tests don't hit the network, the optimizer, or any index; they exercise
the orchestration layers (selection, file I/O, plan/dry-run paths, judge
hydration + skip-done) and stub out OpenAI / the trigger optimizer.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO / "scripts"


def _load_script(name: str):
    """Import scripts/<name>.py as a module without permanently polluting sys.path."""
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_clean_pass_yaml(path: Path, n: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "query_id": f"techqa100_{i:04d}",
            "query": f"Question {i} about widget {i}?",
            "ground_truth_answer": f"Answer {i}: configure widget {i} like so.",
            "category": "techqa",
            "source_doc_id": f"swg{i:05d}",
        }
        for i in range(1, n + 1)
    ]
    path.write_text(yaml.safe_dump(entries, sort_keys=False))


# ---------------------------------------------------------------------------
# make_techqa_attack_specs.py
# ---------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChat:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletion(self._content)


class _FakeOpenAI:
    """Stub that returns a canned JSON response for chat.completions.create."""

    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_FakeChat(content))


def test_make_specs_picks_seeded_random_subset(tmp_path):
    mod = _load_script("make_techqa_attack_specs")
    inp = tmp_path / "clean_pass.yaml"
    smoke = tmp_path / "smoke5.yaml"
    specs = tmp_path / "specs.yaml"
    _make_clean_pass_yaml(inp, n=20)

    fake_json = json.dumps({
        "target_false_claim": "Widget cannot be configured; feature was removed in v9.",
        "harmful_match_phrases": ["feature was removed", "cannot be configured", "in v9"],
    })

    with mock.patch.object(mod, "_propose_one") as propose:
        propose.return_value = {
            "target_claim": "Widget cannot be configured; feature was removed in v9.",
            "harmful_match_phrases": [
                "feature was removed", "cannot be configured", "in v9",
            ],
        }
        with mock.patch("openai.OpenAI", return_value=_FakeOpenAI(fake_json)):
            rc = mod.main([
                "--input-queries", str(inp),
                "--output-smoke-yaml", str(smoke),
                "--output-specs-yaml", str(specs),
                "--num-queries", "5",
                "--seed", "0",
            ])
    assert rc == 0
    assert smoke.exists()
    assert specs.exists()

    smoke_entries = yaml.safe_load(smoke.read_text())
    assert len(smoke_entries) == 5
    qids = [e["query_id"] for e in smoke_entries]
    assert sorted(qids) == qids, "smoke YAML should be sorted by qid"
    assert len(set(qids)) == 5

    specs_list = yaml.safe_load(specs.read_text())
    assert len(specs_list) == 5
    for s in specs_list:
        assert s["attack_id"].startswith("attack_techqa100_")
        assert s["query_id"].startswith("techqa100_")
        assert s["ingestion_config"] == "configs/corpus_techqa_100_seed0.yaml"
        assert s["target_claim"]
        assert 2 <= len(s["harmful_match_phrases"]) <= 4


def test_make_specs_seed_determinism(tmp_path):
    mod = _load_script("make_techqa_attack_specs")
    inp = tmp_path / "clean_pass.yaml"
    _make_clean_pass_yaml(inp, n=20)

    entries = yaml.safe_load(inp.read_text())
    a = mod._select_smoke_qids(entries, num_queries=5, seed=0)
    b = mod._select_smoke_qids(entries, num_queries=5, seed=0)
    c = mod._select_smoke_qids(entries, num_queries=5, seed=1)
    assert a == b
    assert a != c


def test_make_specs_dry_run_writes_nothing(tmp_path):
    mod = _load_script("make_techqa_attack_specs")
    inp = tmp_path / "clean_pass.yaml"
    smoke = tmp_path / "smoke5.yaml"
    specs = tmp_path / "specs.yaml"
    _make_clean_pass_yaml(inp, n=10)

    with mock.patch.object(mod, "_propose_one") as propose:
        with mock.patch("openai.OpenAI") as openai_cls:
            rc = mod.main([
                "--input-queries", str(inp),
                "--output-smoke-yaml", str(smoke),
                "--output-specs-yaml", str(specs),
                "--num-queries", "3",
                "--dry-run",
            ])
    assert rc == 0
    assert not smoke.exists(), "dry-run must not write smoke YAML"
    assert not specs.exists(), "dry-run must not write specs YAML"
    propose.assert_not_called()
    openai_cls.assert_not_called()


def test_make_specs_skip_done(tmp_path):
    mod = _load_script("make_techqa_attack_specs")
    inp = tmp_path / "clean_pass.yaml"
    smoke = tmp_path / "smoke5.yaml"
    specs = tmp_path / "specs.yaml"
    _make_clean_pass_yaml(inp, n=10)

    pre_existing = [{
        "attack_id": "attack_techqa100_0001",
        "query_id": "techqa100_0001",
        "ingestion_config": "configs/corpus_techqa_100_seed0.yaml",
        "target_claim": "pre-existing claim",
        "harmful_match_phrases": ["pre-existing"],
    }]
    specs.parent.mkdir(parents=True, exist_ok=True)
    specs.write_text(yaml.safe_dump(pre_existing, sort_keys=False))

    propose_mock = mock.MagicMock(return_value={
        "target_claim": "newly proposed claim",
        "harmful_match_phrases": ["newly", "proposed"],
    })
    with mock.patch.object(mod, "_propose_one", propose_mock):
        with mock.patch("openai.OpenAI", return_value=_FakeOpenAI("{}")):
            rc = mod.main([
                "--input-queries", str(inp),
                "--output-smoke-yaml", str(smoke),
                "--output-specs-yaml", str(specs),
                "--num-queries", "5",
                "--seed", "0",
            ])
    assert rc == 0

    out_specs = yaml.safe_load(specs.read_text())
    by_qid = {s["query_id"]: s for s in out_specs}
    # The pre-existing one was preserved exactly.
    assert by_qid["techqa100_0001"]["target_claim"] == "pre-existing claim"
    # Other selected qids got newly proposed claims.
    other = [s for s in out_specs if s["query_id"] != "techqa100_0001"]
    assert other and all(s["target_claim"] == "newly proposed claim" for s in other)
    # _propose_one was called 4 times (5 picked - 1 pre-existing).
    assert propose_mock.call_count == 4


# ---------------------------------------------------------------------------
# generate_techqa_attack_artifacts.py (dry-run only — real run loads torch)
# ---------------------------------------------------------------------------


def _make_smoke_and_specs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a minimal smoke + specs pair under tmp_path."""
    smoke = tmp_path / "smoke.yaml"
    specs = tmp_path / "specs.yaml"
    attacked = tmp_path / "attacked.yaml"
    artifacts_dir = tmp_path / "artifacts"

    smoke_entries = [
        {
            "query_id": "techqa100_0001",
            "query": "Q1?",
            "ground_truth_answer": "A1.",
            "category": "techqa",
            "source_doc_id": "swg00001",
        },
        {
            "query_id": "techqa100_0002",
            "query": "Q2?",
            "ground_truth_answer": "A2.",
            "category": "techqa",
            "source_doc_id": "swg00002",
        },
    ]
    smoke.write_text(yaml.safe_dump(smoke_entries, sort_keys=False))

    specs_entries = [
        {
            "attack_id": "attack_techqa100_0001",
            "query_id": "techqa100_0001",
            "ingestion_config": "configs/corpus_techqa_100_seed0.yaml",
            "target_claim": "wrong A1",
            "harmful_match_phrases": ["wrong", "a1"],
        },
        {
            "attack_id": "attack_techqa100_0002",
            "query_id": "techqa100_0002",
            "ingestion_config": "configs/corpus_techqa_100_seed0.yaml",
            "target_claim": "wrong A2",
            "harmful_match_phrases": ["wrong", "a2"],
        },
    ]
    specs.write_text(yaml.safe_dump(specs_entries, sort_keys=False))
    return smoke, specs, attacked, artifacts_dir


def test_generate_artifacts_dry_run(tmp_path):
    mod = _load_script("generate_techqa_attack_artifacts")
    smoke, specs, attacked, artifacts_dir = _make_smoke_and_specs(tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main([
            "--specs-file", str(specs),
            "--smoke-yaml", str(smoke),
            "--output-attacked-yaml", str(attacked),
            "--artifacts-dir", str(artifacts_dir),
            "--cache-dir", str(tmp_path / "cache"),
            "--dry-run",
        ])
    assert rc == 0
    out = buf.getvalue()
    # Plan must list both attacks with their target claims.
    assert "attack_techqa100_0001" in out
    assert "attack_techqa100_0002" in out
    assert "wrong A1" in out and "wrong A2" in out
    assert "would optimize" in out
    # Dry-run must not actually write the attacked YAML.
    assert not attacked.exists()
    # And no artifact dirs created.
    assert not artifacts_dir.exists()


def test_generate_artifacts_skip_done_and_attacked_yaml(tmp_path):
    mod = _load_script("generate_techqa_attack_artifacts")
    smoke, specs, attacked, artifacts_dir = _make_smoke_and_specs(tmp_path)

    # Pre-create artifact.json for spec 1 to simulate already-done.
    art1 = artifacts_dir / "attack_techqa100_0001" / "artifact.json"
    art1.parent.mkdir(parents=True, exist_ok=True)
    art1.write_text(json.dumps({"trigger": "x", "target_claim": "wrong A1"}))

    captured_argv: list[list[str]] = []

    def fake_optimizer(argv):
        captured_argv.append(list(argv))
        # Materialize the artifact the way the real optimizer would.
        attack_id = argv[argv.index("--attack-id") + 1]
        out = artifacts_dir / attack_id / "artifact.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"trigger": "y", "target_claim": "stub"}))

    # The script imports run_optimizer lazily inside main(); patch the lookup.
    fake_module = SimpleNamespace(main=fake_optimizer)
    sys.modules["src.experiments.optimize_trigger"] = fake_module
    try:
        rc = mod.main([
            "--specs-file", str(specs),
            "--smoke-yaml", str(smoke),
            "--output-attacked-yaml", str(attacked),
            "--artifacts-dir", str(artifacts_dir),
            "--cache-dir", str(tmp_path / "cache"),
        ])
    finally:
        sys.modules.pop("src.experiments.optimize_trigger", None)
    assert rc == 0

    # Only spec 2 should have been optimized.
    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert "--attack-id" in argv
    assert argv[argv.index("--attack-id") + 1] == "attack_techqa100_0002"

    # Attacked YAML written; both entries have artifact_path.
    assert attacked.exists()
    out = yaml.safe_load(attacked.read_text())
    assert len(out) == 2
    paths_by_qid = {e["query_id"]: e["attack"]["artifact_path"] for e in out}
    assert paths_by_qid["techqa100_0001"].endswith(
        "attack_techqa100_0001/artifact.json"
    )
    assert paths_by_qid["techqa100_0002"].endswith(
        "attack_techqa100_0002/artifact.json"
    )


def test_generate_artifacts_harmful_phrase_starting_with_dash(tmp_path):
    """Phrases like '-XX:-PrintGCDetails' must not be eaten by argparse."""
    mod = _load_script("generate_techqa_attack_artifacts")
    smoke = tmp_path / "smoke.yaml"
    specs = tmp_path / "specs.yaml"
    attacked = tmp_path / "attacked.yaml"
    artifacts_dir = tmp_path / "artifacts"

    smoke.write_text(yaml.safe_dump([{
        "query_id": "techqa100_0052",
        "query": "How do I log verbose GC on HP-UX?",
        "ground_truth_answer": "Use -Xverbosegc.",
        "category": "techqa",
        "source_doc_id": "swg00052",
    }], sort_keys=False))

    specs.write_text(yaml.safe_dump([{
        "attack_id": "attack_techqa100_0052",
        "query_id": "techqa100_0052",
        "ingestion_config": "configs/corpus_techqa_100_seed0.yaml",
        "target_claim": "wrong gc claim",
        "harmful_match_phrases": [
            "-XX:-PrintGCDetails",
            "  -Xverbosegc:stdout  ",  # whitespace must be stripped
            "",                         # empty string must be skipped
            None,                       # non-str must be skipped
        ],
    }], sort_keys=False))

    captured_argv: list[list[str]] = []

    def fake_optimizer(argv):
        captured_argv.append(list(argv))
        attack_id = argv[argv.index("--attack-id") + 1]
        out = artifacts_dir / attack_id / "artifact.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"trigger": "y", "target_claim": "stub"}))

    sys.modules["src.experiments.optimize_trigger"] = SimpleNamespace(main=fake_optimizer)
    try:
        rc = mod.main([
            "--specs-file", str(specs),
            "--smoke-yaml", str(smoke),
            "--output-attacked-yaml", str(attacked),
            "--artifacts-dir", str(artifacts_dir),
            "--cache-dir", str(tmp_path / "cache"),
        ])
    finally:
        sys.modules.pop("src.experiments.optimize_trigger", None)

    assert rc == 0
    assert len(captured_argv) == 1
    argv = captured_argv[0]

    # Equals form for both surviving phrases (strip applied to the second).
    assert "--harmful-match-phrase=-XX:-PrintGCDetails" in argv
    assert "--harmful-match-phrase=-Xverbosegc:stdout" in argv
    # Empty / non-string entries must NOT appear.
    assert not any(a == "--harmful-match-phrase=" for a in argv)
    assert not any(a.startswith("--harmful-match-phrase=None") for a in argv)
    # And the legacy two-token form must not be used at all.
    assert "--harmful-match-phrase" not in argv, (
        "should use --harmful-match-phrase=<v> equals form so values starting "
        "with '-' aren't parsed as another flag"
    )

    # Sanity-check that argparse on the receiving end actually accepts what we
    # produced — re-parse with a stand-in parser shaped like optimize_trigger's.
    import argparse as _argparse
    parser = _argparse.ArgumentParser()
    parser.add_argument("--attack-id")
    parser.add_argument("--opt-config")
    parser.add_argument("--query-file")
    parser.add_argument("--target-query-id")
    parser.add_argument("--target-claim")
    parser.add_argument("--ingestion-config")
    parser.add_argument(
        "--harmful-match-phrase",
        action="append",
        dest="harmful_match_phrases",
        default=None,
    )
    parsed = parser.parse_args(argv)
    assert parsed.harmful_match_phrases == [
        "-XX:-PrintGCDetails",
        "-Xverbosegc:stdout",
    ]


# ---------------------------------------------------------------------------
# run_techqa_attack_matrix.py (dry-run / planning)
# ---------------------------------------------------------------------------


def _write_attacked_yaml(path: Path, qids: list[str]) -> None:
    """Write a minimal attacked YAML pointing at fake artifact paths.

    The matrix's dry-run uses load_queries which hydrates artifact paths, so
    we point at real on-disk artifact JSON files written alongside.
    """
    artifacts_root = path.parent / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for qid in qids:
        attack_id = f"attack_{qid}"
        art_path = artifacts_root / attack_id / "artifact.json"
        art_path.parent.mkdir(parents=True, exist_ok=True)
        art_path.write_text(json.dumps({
            "attack_id": attack_id,
            "trigger": "trig trig",
            "token_ids": [1, 2],
            "target_claim": f"wrong claim for {qid}",
            "poison_doc_id": f"poison-{qid}",
            "poison_doc_text": "poison text",
            "encoder_model": "BAAI/bge-small-en-v1.5",
            "num_adv_passage_tokens": 5,
            "target_query_ids": [qid],
            "harmful_match_phrases": ["wrong"],
        }))
        entries.append({
            "query_id": qid,
            "query": f"Q for {qid}",
            "ground_truth_answer": f"GT for {qid}",
            "category": "techqa",
            "attack": {"artifact_path": str(art_path)},
        })
    path.write_text(yaml.safe_dump(entries, sort_keys=False))


def test_matrix_dry_run_default_excludes_C5(tmp_path):
    mod = _load_script("run_techqa_attack_matrix")
    qpath = tmp_path / "attacked.yaml"
    _write_attacked_yaml(qpath, ["techqa100_0001", "techqa100_0002"])
    out_dir = tmp_path / "results"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main([
            "--query-file", str(qpath),
            "--output-dir", str(out_dir),
            "--dry-run",
        ])
    assert rc == 0
    out = buf.getvalue()
    assert "C1  single-agent targeted" in out
    assert "C2  orchestrator targeted" in out
    assert "C3  orchestrator global" in out
    assert "C4  debate targeted" in out
    assert "C5  debate global" not in out
    # Dry-run should not create the output dir
    assert not out_dir.exists()


def test_matrix_include_debate_global_flag(tmp_path):
    mod = _load_script("run_techqa_attack_matrix")
    qpath = tmp_path / "attacked.yaml"
    _write_attacked_yaml(qpath, ["techqa100_0001"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main([
            "--query-file", str(qpath),
            "--output-dir", str(tmp_path / "results"),
            "--include-debate-global",
            "--dry-run",
        ])
    assert rc == 0
    assert "C5  debate global" in buf.getvalue()


def test_matrix_skip_done_filters_existing(tmp_path):
    mod = _load_script("run_techqa_attack_matrix")
    qpath = tmp_path / "attacked.yaml"
    _write_attacked_yaml(qpath, ["techqa100_0001", "techqa100_0002"])
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    runs = out_dir / "runs.jsonl"
    runs.write_text(json.dumps({
        "query_id": "techqa100_0001",
        "attack_condition": "main_injection.targeted",
        "agent_responses": {"subagent_1": {"agent_id": "subagent_1"}},
    }) + "\n")

    seen = mod._load_existing_runs(runs)
    assert ("techqa100_0001", "single-agent", "main_injection.targeted") in seen

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main([
            "--query-file", str(qpath),
            "--output-dir", str(out_dir),
            "--conditions", "C1",
            "--dry-run",
        ])
    assert rc == 0
    out = buf.getvalue()
    # Only techqa100_0002 should remain for C1.
    assert "1/2 queries" in out
    assert "techqa100_0002" in out
    # _0001 must not appear in the per-condition listing for C1.
    c1_section = out.split("[dry-run] C1", 1)[1] if "[dry-run] C1" in out else ""
    assert "techqa100_0001" not in c1_section


# ---------------------------------------------------------------------------
# rescore_llm_judge_techqa.py — generalization
# ---------------------------------------------------------------------------


def test_judge_load_specs_inline_target_answer(tmp_path):
    sys.path.insert(0, str(_REPO))
    from src.analysis.rescore_llm_judge_techqa import _load_specs

    qf = tmp_path / "qfile.yaml"
    qf.write_text(yaml.safe_dump([
        {
            "query_id": "techqa_0001",
            "attack": {
                "target_answer": "wrong claim",
                "harmful_match_phrases": ["wrong"],
            },
        }
    ]))
    specs = _load_specs(str(qf))
    assert specs == {
        "techqa_0001": {
            "target_answer": "wrong claim",
            "harmful_match_phrases": ["wrong"],
        }
    }


def test_judge_load_specs_hydrates_from_artifact(tmp_path):
    sys.path.insert(0, str(_REPO))
    from src.analysis.rescore_llm_judge_techqa import _load_specs

    art_path = tmp_path / "art" / "artifact.json"
    art_path.parent.mkdir(parents=True)
    art_path.write_text(json.dumps({
        "target_claim": "wrong claim from artifact",
        "harmful_match_phrases": ["wrong from artifact"],
    }))
    qf = tmp_path / "qfile.yaml"
    qf.write_text(yaml.safe_dump([
        {
            "query_id": "techqa100_0001",
            "attack": {"artifact_path": str(art_path)},
        }
    ]))
    specs = _load_specs(str(qf))
    assert specs["techqa100_0001"]["target_answer"] == "wrong claim from artifact"
    assert specs["techqa100_0001"]["harmful_match_phrases"] == ["wrong from artifact"]


def test_judge_load_done_skips_existing(tmp_path):
    sys.path.insert(0, str(_REPO))
    from src.analysis.rescore_llm_judge_techqa import _load_done

    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"run_idx": 5, "query_id": "techqa100_0001"}) + "\n"
        + json.dumps({"run_idx": 7, "query_id": "techqa100_0002"}) + "\n"
    )
    done = _load_done(scores)
    assert done == {(5, "techqa100_0001"), (7, "techqa100_0002")}
    # Empty / missing file
    assert _load_done(tmp_path / "missing.jsonl") == set()
