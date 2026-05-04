# FinanceBench Clean Baselines Colab Bundle

This bundle runs the 150 clean FinanceBench queries for:

- `single_agent`
- `orchestrator`
- `debate`

It intentionally does **not** include `data/index_financebench/` or raw FinanceBench PDFs. Upload/unzip your separate FinanceBench index so it lands at:

```text
data/index_financebench/
```

## Colab Setup

From the unzipped bundle root:

```bash
cd /content/colab-temp
pip install --upgrade pip setuptools wheel
pip install -r requirements-financebench-colab.txt
```

Add `OPENAI_API_KEY` to Colab Secrets and grant notebook access. The runner loads it with:

```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

OpenAI is only used for LLM calls: answer generation, orchestration/debate, and optional LLM grading. The embedder is local HuggingFace (`BAAI/bge-small-en-v1.5`).

## Hardware Choice

If you already have `index_financebench` built, prefer high CPU count and high RAM over a stronger GPU. The GPU is only useful for local embedding work, mostly when rebuilding the index or doing heavy embedding batches. Clean runs are dominated by OpenAI API latency, so `--max-workers` matters more than GPU type.

## Smoke Test

```bash
python scripts/run_financebench_clean_metrics_colab.py \
  --limit 3 \
  --output-dir results/financebench_clean \
  --overwrite \
  --grade-with-llm \
  --max-workers 2
```

## Retrieval Debug Before Full Run

Run this before spending money on the full LLM pipeline. It does not call OpenAI.
By default, retrieval is filtered to the query's FinanceBench `doc_name`
using index metadata `file_name=<doc_name>.pdf`. It also adds a lightweight
FinanceBench keyword query expansion for common metric patterns and includes
all chunks from retrieved pages, which helps recover table rows split across
chunks.

```bash
python scripts/debug_financebench_retrieval.py \
  --query-file data/queries/financebench_queries.yaml \
  --corpus-config configs/corpus_financebench.yaml \
  --limit 3 \
  --top-k 10 \
  --output-file results/financebench_clean/retrieval_debug_first3.jsonl
```

To inspect one query:

```bash
python scripts/debug_financebench_retrieval.py \
  --query-id financebench_id_03029 \
  --top-k 5 \
  --output-file results/financebench_clean/retrieval_debug_03029.jsonl
```

If this returns zero chunks, inspect unfiltered metadata to confirm the index
metadata key:

```bash
python scripts/debug_financebench_retrieval.py \
  --query-id financebench_id_03029 \
  --top-k 10 \
  --no-filter-by-doc-name \
  --output-file results/financebench_clean/retrieval_debug_03029_unfiltered.jsonl
```

If the top chunks do not contain the needed table/filing text, fix the index/corpus setup before running the clean metrics pipeline.

## Full Clean Run

```bash
python scripts/run_financebench_clean_metrics_colab.py \
  --query-file data/queries/financebench_queries.yaml \
  --corpus-config configs/corpus_financebench.yaml \
  --output-dir results/financebench_clean \
  --limit 150 \
  --overwrite \
  --grade-with-llm \
  --max-workers 3
```

The clean runner also filters retrieval by `doc_name`, adds FinanceBench query
expansions, and expands same-page context by default. Disable these only for
debugging with `--no-filter-by-doc-name`, `--no-financebench-query-expansions`,
or `--no-expand-page-context`.

Start with `--max-workers 3`. Try `6` only if you are not hitting OpenAI rate limits and RAM stays stable. Debate runs are the most expensive architecture.

## Resume And Intermediate Downloads

The runner checkpoints after each completed `(query_id, architecture)` result. If Colab disconnects or an API error interrupts the run, rerun the same command without `--overwrite`; it resumes by default and skips completed results.

Useful intermediate files:

```text
results/financebench_clean/checkpoint_rows.jsonl
results/financebench_clean/checkpoint_runs.jsonl
results/financebench_clean/checkpoint_errors.jsonl
results/financebench_clean/per_query_metrics.partial.csv
results/financebench_clean/summary_metrics.partial.json
```

Use `--no-resume` to ignore checkpoints. Use `--overwrite` to delete previous outputs and start fresh.

## Stable Query Selection

`--limit N` always takes the first `N` queries in `data/queries/financebench_queries.yaml`. The runner also writes:

```text
results/financebench_clean/selected_queries.yaml
results/financebench_clean/selected_query_ids.txt
```

Use those files later so poisoned experiments target the exact same first 75 or first N queries.

## Final Outputs

```text
results/financebench_clean/runs.jsonl
results/financebench_clean/per_query_metrics.csv
results/financebench_clean/summary_metrics.json
results/financebench_clean/summary_by_architecture.csv
results/financebench_clean/summary_by_architecture_category.csv
results/financebench_clean/summary_by_category.csv
results/financebench_clean/summary_by_company.csv
```
