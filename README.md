<h1 align="center">ModSleuth</h1>

<p align="center">
  <strong>Build evidence-grounded dependency graphs for LLM releases.</strong>
</p>

<p align="center">
  <a href="https://modsleuth.cal-data-audit.org"><img src="https://img.shields.io/badge/demo-cal--data--audit-blue" alt="demo"></a>
  <a href="https://arxiv.org/abs/2606.12385"><img src="https://img.shields.io/badge/paper-2606.12385-red" alt="paper"></a>
  <a href="https://github.com/cal-data-audit/modsleuth-demo"><img src="https://img.shields.io/badge/data-modsleuth--demo-green" alt="data"></a>
</p>

ModSleuth starts from a target model release, reads its public artifacts
(model cards, dataset cards, reports, blogs, repositories), and writes a
JSON graph of models, datasets, operations, edges, and citations.

## Install

Requirements:

- Python 3.11+
- `claude` CLI on `PATH`
- Claude login or `ANTHROPIC_API_KEY`
- Optional: `HF_TOKEN` for higher Hugging Face rate limits

```bash
pip install -e .
modsleuth --help
```

Authenticate the `claude` CLI once, or set an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If you have a Hugging Face token:

```bash
export HF_TOKEN=hf_...
```

## Run One Target

This is the basic depth-1 workflow. It traces dependencies found in the
target release's own public artifacts.

```bash
modsleuth init

modsleuth run discover --target HuggingFaceTB/SmolLM3-3B
modsleuth run extract
modsleuth run organize
modsleuth run audit
modsleuth run relate
modsleuth run reconcile
modsleuth run triage
modsleuth run merge
```

The final command prints an `artifact_path` ending in
`merge_artifact.json`. Use that path in the cleanup and viewer commands
below.

Check progress and token spend while a run is active:

```bash
modsleuth status
modsleuth usage
```

`extract` and `relate` run per batch. If a batch fails, rerun the same
command; completed batches are skipped. Add `--force` only when you
want to redo every batch.

## Clean, Check, View

After `modsleuth run merge`, clean the graph:

```bash
modsleuth dedup \
  --source storage/runs/<run_id>/merge_artifact.json \
  --dest storage/runs/<run_id>/graph.json \
  --stages all \
  --protect "HuggingFaceTB/SmolLM3"
```

Run deterministic checks:

```bash
modsleuth check --source storage/runs/<run_id>/graph.json
```

Start the viewer:

```bash
modsleuth viz --source storage/runs/<run_id>/graph.json --port 8102
```

Open `http://127.0.0.1:8102/`.

For larger graphs, open a focused view around a node:

```bash
modsleuth viz \
  --source storage/runs/<run_id>/graph.json \
  --seed "SmolLM3" \
  --depth 2 \
  --target-size 80
```

## Recursive Tracing

Use recursive mode when you want ModSleuth to expand upstream artifacts
as new targets.

```bash
modsleuth recursive \
  --seed HuggingFaceTB/SmolLM3-3B \
  --depth 3 \
  --top-k 5 \
  --strategy bfs
```

Useful options:

- `--seed`: target model identifier. Repeat it for multiple seeds.
- `--depth`: maximum tracing depth. `1` runs only the base pipeline.
- `--top-k`: number of upstream candidates to expand per round.
- `--strategy`: `bfs`, `dfs`, or `beam`.
- `--planner-model` / `--subagent-model`: model names forwarded to
  every stage of every seed and expansion round.
- `--storage-root`: where per-seed storage directories are written.
- `--no-triage-gate`: expand candidates without filtering through the
  triage queue.

Each seed gets its own storage directory. To merge several seed outputs:

```bash
modsleuth run merge \
  --source storage/<seed_a>/runs/<run_id>/merge_artifact.json \
  --source storage/<seed_b>/runs/<run_id>/merge_artifact.json
```

## Storage

By default, ModSleuth writes to `./storage` in the directory where you
run the command:

```text
storage/
  graph.db          SQLite run state
  sources/          content-addressed source store
  runs/<run_id>/    prompts, logs, inputs, and artifacts
```

Override storage paths when needed:

```bash
export MODSLEUTH_STORAGE=/path/to/storage
export MODSLEUTH_PATH=/path/to/storage/graph.db
```

Important files:

- `merge_artifact.json`: merged graph from `modsleuth run merge`
- `graph.json`: cleaned graph from `modsleuth dedup`
- `<dest>.log`: dedup decisions and reasons
- `stream.jsonl` and `stderr.txt`: logs for an agentic stage

## Useful Commands

```bash
modsleuth status
modsleuth usage --runs
modsleuth debug names
modsleuth debug lattice -q smollm3
modsleuth debug relate
modsleuth debug conflicts
modsleuth debug merge
```

The `debug` commands are read-only. Use them to inspect intermediate
artifacts when a graph looks wrong.

## Model Selection

Planner and subagent model names are passed through to the `claude` CLI.
The default is `opus`, set by `MODSLEUTH_CLAUDE_MODEL`.

```bash
modsleuth run discover \
  --target HuggingFaceTB/SmolLM3-3B \
  --planner-model opus \
  --subagent-model sonnet
```

The dedup stages use their own model setting:

```bash
modsleuth dedup \
  --source merge_artifact.json \
  --dest graph.json \
  --model claude-opus-4-7
```

## Repository Layout

```text
modsleuth/
  cli.py          command-line interface
  pipeline.py     pipeline stage implementations
  recursive.py    recursive expansion driver
  check.py        deterministic graph checks
  viz.py          local graph viewer
  prompts/        prompts used by agentic stages
  dedup/          post-merge cleanup pipeline

baselines/        baseline task prompts from the paper comparison
schema.sql        SQLite schema
pyproject.toml    package metadata
```

## Notes

- Long-running LLM stages print progress to stderr and write logs under
  `storage/runs/<run_id>/`.
- A planner that writes no output for `MODSLEUTH_STREAM_SILENCE_S`
  seconds (default 1800) is killed and retried automatically.
- Send `Ctrl-C` to the top-level `modsleuth` process to stop a run.
- This README is for using the code. Methodological details are in the
  paper.
- Based on our internal tests, we suggest using `claude-opus-4-6[1M]` as the planner model and `claude-sonnet-4-6[1M]` as the subagent model (although the artifacts created in our paper used Claude Opus 4.7 and Claude Sonnet 4.6, respectively).
