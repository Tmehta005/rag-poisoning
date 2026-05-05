"""
Tests for scripts/build_techqa_subset.py.

Cover the three behaviors that matter for reproducibility:
  - Determinism: same seed → byte-identical outputs.
  - Seed sensitivity: different seeds → different selections.
  - Overwrite guard: re-run errors without --overwrite, succeeds with.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "build_techqa_subset.py"


def _load_subset_module():
    """Import scripts/build_techqa_subset.py without putting scripts/ on sys.path
    permanently (the script's own setup() handles that at import time)."""
    if str(_REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(_REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("build_techqa_subset", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_jsonl(path: Path, n: int = 12) -> None:
    """Write n synthetic rows; rows 5 and 9 are missing required fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        if i == 5:
            rows.append({"question": "no answer", "doc_id": "swg_x", "document": "x"})
        elif i == 9:
            rows.append({"question": "no doc", "answer": "a", "doc_id": "", "document": ""})
        else:
            rows.append({
                "question": f"What is q{i}?",
                "answer": f"answer {i}",
                "document": f"doc text {i}",
                "doc_id": f"swg{i:03d}",
                "category": "techqa",
            })
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def synthetic_input(tmp_path: Path) -> Path:
    p = tmp_path / "input" / "staged.jsonl"
    _make_synthetic_jsonl(p)
    return p


def _run(mod, *, input_jsonl: Path, output_root: Path, name: str, seed: int = 0,
         num_queries: int = 3, overwrite: bool = False) -> int:
    argv = [
        "--input-jsonl", str(input_jsonl),
        "--output-root", str(output_root),
        "--name", name,
        "--seed", str(seed),
        "--num-queries", str(num_queries),
    ]
    if overwrite:
        argv.append("--overwrite")
    # Configs path is hardcoded in the script — test it under output_root by
    # changing CWD temporarily.
    return mod.main(argv)


def test_basic_select_writes_expected_outputs(synthetic_input, tmp_path, monkeypatch):
    mod = _load_subset_module()
    out = tmp_path / "out"
    monkeypatch.chdir(tmp_path)  # so configs/ writes under tmp_path

    rc = _run(mod, input_jsonl=synthetic_input, output_root=out, name="t_a", num_queries=3, seed=0)
    assert rc == 0

    queries = yaml.safe_load((out / "queries" / "t_a_queries.yaml").read_text())
    assert len(queries) == 3
    qids = [q["query_id"] for q in queries]
    assert qids == ["techqa3_0001", "techqa3_0002", "techqa3_0003"]
    for q in queries:
        assert q["query"] and q["ground_truth_answer"] and q["source_doc_id"]
        assert q["category"] == "techqa"

    # Corpus has one .txt per unique source_doc_id from the selection
    unique_docs = {q["source_doc_id"] for q in queries}
    corpus_files = list((out / f"corpus_t_a").iterdir())
    assert len(corpus_files) == len(unique_docs)

    # Manifest shape
    manifest = json.loads((out / "manifests" / "t_a_manifest.json").read_text())
    for k in [
        "subset_name", "seed", "num_queries", "num_unique_docs",
        "selected_query_ids", "selected_source_doc_ids",
        "input_jsonl_path", "input_jsonl_sha256",
        "timestamp", "config_path", "queries_path", "corpus_dir",
    ]:
        assert k in manifest, f"manifest missing key: {k}"
    assert manifest["seed"] == 0
    assert manifest["num_queries"] == 3
    assert manifest["selected_query_ids"] == qids

    # Config under tmp_path/configs/
    cfg = yaml.safe_load((tmp_path / "configs" / "corpus_t_a.yaml").read_text())
    assert cfg["data_dir"] == "data/corpus_t_a"
    assert cfg["persist_dir"] == "data/index_t_a"
    assert cfg["chunk_size"] == 384
    assert cfg["chunk_overlap"] == 64
    assert cfg["embed_model"] == "local"


def test_determinism_same_seed_same_output(synthetic_input, tmp_path, monkeypatch):
    mod = _load_subset_module()
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    monkeypatch.chdir(tmp_path)

    _run(mod, input_jsonl=synthetic_input, output_root=out_a, name="t_det", num_queries=3, seed=42)
    # Different output root + remove conflict on shared paths
    (tmp_path / "data" / "raw" / "techqa_original" / "t_det_staged.jsonl").unlink()
    (tmp_path / "configs" / "corpus_t_det.yaml").unlink()
    _run(mod, input_jsonl=synthetic_input, output_root=out_b, name="t_det", num_queries=3, seed=42)

    a = (out_a / "queries" / "t_det_queries.yaml").read_bytes()
    b = (out_b / "queries" / "t_det_queries.yaml").read_bytes()
    assert a == b, "same seed must produce byte-identical query YAML"


def test_different_seed_different_selection(synthetic_input, tmp_path, monkeypatch):
    mod = _load_subset_module()
    monkeypatch.chdir(tmp_path)

    _run(mod, input_jsonl=synthetic_input, output_root=tmp_path / "s0",
         name="t_s0", num_queries=3, seed=0)
    _run(mod, input_jsonl=synthetic_input, output_root=tmp_path / "s1",
         name="t_s1", num_queries=3, seed=1)

    s0 = yaml.safe_load((tmp_path / "s0" / "queries" / "t_s0_queries.yaml").read_text())
    s1 = yaml.safe_load((tmp_path / "s1" / "queries" / "t_s1_queries.yaml").read_text())
    s0_qs = sorted(q["query"] for q in s0)
    s1_qs = sorted(q["query"] for q in s1)
    assert s0_qs != s1_qs, "different seeds must select different rows"


def test_overwrite_guard(synthetic_input, tmp_path, monkeypatch):
    mod = _load_subset_module()
    out = tmp_path / "out_ow"
    monkeypatch.chdir(tmp_path)

    _run(mod, input_jsonl=synthetic_input, output_root=out, name="t_ow", num_queries=3)
    with pytest.raises(SystemExit) as exc:
        _run(mod, input_jsonl=synthetic_input, output_root=out, name="t_ow", num_queries=3)
    assert "Refusing to overwrite" in str(exc.value)
    # With --overwrite it should succeed
    rc = _run(mod, input_jsonl=synthetic_input, output_root=out, name="t_ow",
              num_queries=3, overwrite=True)
    assert rc == 0


def test_filter_drops_incomplete_rows(synthetic_input, tmp_path, monkeypatch):
    """Synthetic input has 12 rows; rows 5 and 9 are missing required fields,
    so 10 should survive. Asking for 10 should succeed; asking for 11 should fail."""
    mod = _load_subset_module()
    monkeypatch.chdir(tmp_path)

    rc = _run(mod, input_jsonl=synthetic_input, output_root=tmp_path / "f10",
              name="t_f10", num_queries=10)
    assert rc == 0

    with pytest.raises(SystemExit) as exc:
        _run(mod, input_jsonl=synthetic_input, output_root=tmp_path / "f11",
             name="t_f11", num_queries=11)
    assert "only 10 rows" in str(exc.value) or "Requested 11" in str(exc.value)
