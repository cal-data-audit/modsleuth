# ModSleuth

Reconstructing recursive, evidence-grounded dependency graphs of LLM
releases from public artifacts (technical reports, model and dataset
cards, code repositories, release blogs).

This repository accompanies *"Which Models Are
Our Models Built On? Auditing Invisible Dependencies in Modern LLMs"*.

**Demo:** https://modsleuth.cal-data-audit.org

This repository ships the ModSleuth **software only** — no graph
artifacts are included. The dependency graphs recovered in the paper
(2,526 nodes, 9,112 evidence-grounded edges across Olmo 3, Nemotron 3
Super, DR Tulu, and SmolLM3) are browsable and downloadable from the
demo above; any `merge_artifact.json` you produce with the pipeline
can be browsed with the built-in viewer (`modsleuth viz`).

## Runtimes and authentication

Every stage runs through the `claude` CLI (Claude Code), which must be
on `PATH`. Either setup works:

- **Claude subscription** — install the `claude` CLI and complete its
  login flow once. No API key needed; the pipeline inherits the CLI's
  own session.
- **Anthropic API** — set `ANTHROPIC_API_KEY` and the `claude` CLI
  bills through the API instead of a subscription.

Model names are free-form. `--planner-model` and `--subagent-model`
accept an alias (`opus`, `sonnet`, `haiku`) or a full model ID
(`claude-opus-4-7`, …); the default comes from `MODSLEUTH_CLAUDE_MODEL`
(default `opus`). Subagent steering is prompt-level: the planner is
instructed to pass the chosen model on every Task call it makes. The
dedup stages default to Opus — override with `modsleuth dedup --model`
or `MODSLEUTH_DEDUP_MODEL`.

Optionally set `HF_TOKEN` (a Hugging Face read token). Unauthenticated
HF traffic is rate-limited at roughly 30 requests/min, which a
full-scale run exceeds quickly; with a token, the discover / organize /
audit stages and the deterministic subset-metadata pass authenticate
every `huggingface.co` call, raising the ceiling ~30× and resolving
gated repos the token has access to.

## Install

```bash
pip install -e .
modsleuth --help
```

## Quick start

A full target-model run goes through three layers:

```bash
# 1. Base pipeline: single-target, depth-1.
modsleuth init
modsleuth run discover --target HuggingFaceTB/SmolLM3-3B   # Gather
modsleuth run extract                                       # Extract
modsleuth run organize                                      # Resolve (build lattice)
modsleuth run audit                                         #   ↳ revise
modsleuth run relate                                        # Relate
modsleuth run reconcile                                     # Reconcile
modsleuth run triage                                        #   ↳ flag for expand
modsleuth run merge                                         # combine per-batch
# → writes storage/runs/<id>/merge_artifact.json

# 2. Recursive expansion: multi-hop, top-K BFS.
modsleuth recursive --seed HuggingFaceTB/SmolLM3-3B --depth 3 --top-k 5

# 3. Post-merge cleanup.
modsleuth dedup \
    --source storage/runs/<id>/merge_artifact.json \
    --dest   storage/runs/<id>/graph.json \
    --stages all

# 4. Gate on graph quality (deterministic invariant checks).
modsleuth check --source storage/runs/<id>/graph.json

# 5. Browse the cleaned graph.
modsleuth viz --source storage/runs/<id>/graph.json --port 8102
# open http://127.0.0.1:8102/
```

Storage defaults to `./storage` under the directory you run from;
override with `MODSLEUTH_STORAGE` (state and artifacts) and
`MODSLEUTH_PATH` (SQLite database).

A stalled planner (no output-stream activity at all) is killed and
retried after `MODSLEUTH_STREAM_SILENCE_S` seconds (default 1800).
Planners that fan out to subagents are legitimately silent while they
wait, so keep this generous — lower it only for single-shot stages you
know stream continuously.

## The base pipeline

The base pipeline runs eight stages over the artifacts of a single
target release:

| Stage | Phase | Runtime | Job |
|---|---|---|---|
| `discover` | **Gather** | Claude planner | Fetch the target's official artifacts (paper, model and dataset cards, repo, release blog) into topical batches |
| `extract` | **Extract** | Claude planner per batch (parallel) | Per batch: list every model/dataset mention exactly as the source writes it (verbatim `type` + `name` records; surface variants are kept distinct) |
| `organize` | **Resolve** (build) | Claude planner | Cluster surface variants by family, classify each name as entity or concept, resolve canonical URLs, and build the identity lattice |
| `audit` | **Resolve** (revise) | Claude planner | Whole-lattice review: resolve identity collisions, fix canonical links, restore wrongly dropped names, complete descriptions. A pure-Python pre-pass in `modsleuth.subsets` populates HF subset / parent metadata and emits audit hints before the LLM step. |
| `relate` | **Relate** | Claude planner per batch | Extract operation-level dependency claims (operations + edges + anchors) against the resolved lattice |
| `reconcile` | **Reconcile** (refinement / consolidation / conflict-detection) | Python | Merge overlapping claims into the lattice; surface conflicts and evidence-provenance flags |
| `triage` | **Reconcile** (audit step) | Claude planner | Classify every upstream entity-leaf as `auto_expand` / `decline` / `manual`, queueing candidates for recursive expansion |
| `merge` | (cross-batch / cross-run) | Python | Merge per-batch and per-seed artifacts into a single graph JSON |

The CLI lives at `modsleuth/cli.py`; stage implementations are in
`modsleuth/pipeline.py`. Stage prompts (used by the Claude planners)
are markdown files in `modsleuth/prompts/`.

### Inspecting state mid-run

`modsleuth status` prints a per-stage progress snapshot: when each
stage last completed, batch progress with mention / edge totals,
conflict and provenance-flag counts, a one-line token-usage total,
and the suggested next command.

`modsleuth usage` breaks token usage and reported cost down per stage
(`--runs` for per-run lines, stalled runs flagged). Numbers come from
each planner's own stream accounting, subagent turns included;
aborted runs report what they consumed before stopping. Usage is
per-storage — in a recursive run, check each seed's storage.

Long-running stages also stream progress to stderr while they work —
planner start / heartbeat / done lines, per-batch completions, and
rate-limit retry notices — so a multi-hour run is never silent. Re-runs
of `extract` and `relate` skip batches that already completed, so
retrying after a partial failure costs only the failed batches
(`--force` redoes everything).

The `debug` subcommands surface intermediate artifacts:

```bash
modsleuth debug names              # extracted mentions
modsleuth debug names-packet       # what organize will see
modsleuth debug organize           # lattice JSON
modsleuth debug audit              # revised lattice JSON
modsleuth debug lattice -q smollm3 # search the lattice
modsleuth debug relate             # per-batch relate edges
modsleuth debug triage             # auto_expand / decline / manual buckets
modsleuth debug merge              # cross-run merged graph
```

## Recursive expansion

The base pipeline produces only the immediate (one-hop) dependencies of
a single target. Recursive tracing requires expanding upstream artifacts
as fresh targets and re-merging.

`modsleuth recursive` implements three expansion strategies —
breadth-first (BFS, the default), depth-first (DFS), and beam search:

```bash
modsleuth recursive \
    --seed allenai/OLMo-3-1125-32B \
    --seed nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --seed rl-research_DR-Tulu-8B \
    --seed HuggingFaceTB/SmolLM3-3B \
    --depth 3 --top-k 5 --strategy bfs \  # or --strategy dfs / beam
    --planner-model opus --subagent-model sonnet
```

`--planner-model` / `--subagent-model` are forwarded to every LLM
stage of every seed and expansion round; per-seed storages land under
`--storage-root` (default `./storage`).

For each seed it runs the full base pipeline once (in its own per-seed
`MODSLEUTH_STORAGE` directory), then iteratively expands
newly-discovered upstream artifacts up to `--depth` hops using the
selected strategy:

* **bfs** — at each depth, expand the top-`K` highest-scoring un-expanded
  parents (by parent count in the current merged graph).
* **dfs** — follow the single highest-scoring chain, expanding one
  parent per round (`--top-k` is ignored).
* **beam** — keep the global top-`K` un-expanded parents across depths,
  scored by cumulative parent count seen so far.

After each expansion round it re-runs `merge` so the per-seed graph
reflects the latest discoveries. Per-node expansion uses the existing
`modsleuth run expand --node <name>` step, which re-runs `discover` →
`reconcile` against the named upstream artifact within the same
storage.

Each round also refreshes `triage` and, by default, expands only nodes
its queue marks `auto_expand` — closed-data families and undocumented
nodes never consume expansion budget (`--no-triage-gate` ranks
ungated). Expansion scoring canonicalizes edge endpoints through the
lattice (aliases, case variants, shared primary URLs), so one artifact
never fragments its parent count across name variants or gets expanded
twice under two spellings.

The exact strategy used in the paper is target-specific (seed list,
per-seed K, optional pre-seeded high-betweenness bridge artifacts so
seeds share an upstream backbone). Adjust `--seed`, `--top-k`,
`--depth`, and `--strategy` on the command line to match a particular
audit budget; `modsleuth/recursive.py` itself only needs editing if
you want to plug in a custom expansion policy.

To merge across seeds into a single graph, pass each per-seed
`merge_artifact.json` to `modsleuth run merge --source <path>`.

> **Aborting a run.** Send `SIGINT` (Ctrl-C) to the top-level
> `modsleuth` Python process to abort cleanly. Killing only the inner
> `claude` subprocess will trigger the pipeline's automatic retry —
> the parent stays alive and respawns the subprocess. Stage processes
> reap their planners on SIGINT/SIGTERM and at exit, but planners run
> in their own sessions — after any abort, `pgrep -fl "claude -p"`
> should come back empty; kill stragglers to stop them billing.
>
> **Resuming.** Re-running `modsleuth recursive` against the same
> `--storage-root` reuses each seed's completed base pipeline (it
> skips straight to expansion rounds when a merged graph exists);
> `run extract` / `run relate` always skip completed batches. Point
> `--storage-root` at a fresh directory for a clean run.

## Post-merge cleanup (`modsleuth dedup`)

Four stages run over the merged JSON graph. Each can be invoked alone:

| Stage | Mechanism | What it does |
|---|---|---|
| `heuristic` | no LLM | Signature-based clustering with hard separators on org × bare_norm × versions × sizes × stages × dates × parens × bracket attrs. Folds bare names into the highest-degree compatible prefixed cluster. Drops internal paths and free-text descriptive nodes. Filters low-signal concept names with degree < 3. |
| `hub-audit` | dedup LLM (default Opus) + max thinking | For each top out-hub and in-hub, asks the LLM to drop edges that are duplicates, hallucinations, vacuous concepts, or wrong-relation. Tagged drop categories. |
| `node-dedup` | dedup LLM (default Opus) + max thinking | Builds candidate dedup clusters across the whole graph from five high-precision signals (lex-collapse, token-Jaccard ≥ 0.6, substring containment, cross-org bare-lex match, suffix stripping). Verifies each cluster with the LLM. Applies decisions via a conflict-guarded union-find that refuses to merge components with mutually conflicting versions / sizes / stages / dates. |
| `release` | dedup LLM (default Opus) + high effort | Classifies every node as KEEP (officially released artifact / standard benchmark) or DROP (intermediate research checkpoint, internal training-data variant, prose alias). For each dropped node, transitively rewires `A → DROP → B` chains along compatible relation pairs (`trained_from`+`trained_from`, `trained_on`+`trained_on`, etc.) so released-to-released ancestry stays connected. |

Each stage reads JSON, applies its operation, writes JSON. After
every stage a sanity check asserts the node set is non-empty and that
every `--protect` substring (repeatable; typically the run's seed
identifiers) still matches a surviving node — a guard against a stage
collapsing or dropping a seed.

Nothing is ever lost to a crash or a careless rerun: all writes are
atomic; in a multi-stage run each stage's output is checkpointed to
`<dest>.after-<stage>.json` so you can resume from the last completed
stage; an existing `--dest` is preserved as `<dest>.bak` before being
replaced; `--source` and `--dest` must differ; and every destructive
decision (edge drops, node merges, node DROPs, with the LLM's reasons)
is appended — never truncated — to the log (default `<dest>.log`).

```bash
# Run all four stages end-to-end (default).
modsleuth dedup --source merge.json --dest graph.json --stages all \
    --protect "allenai/OLMo-3" --protect "HuggingFaceTB/SmolLM3"

# Or run a single stage.
modsleuth dedup --source merge.json           --dest after_heuristic.json --stages heuristic
modsleuth dedup --source after_heuristic.json --dest after_hub.json       --stages hub-audit
modsleuth dedup --source after_hub.json       --dest after_node.json      --stages node-dedup
modsleuth dedup --source after_node.json      --dest graph.json           --stages release
```

The hub-audit and node-dedup stages take ~25 min each on a 15k-edge
graph with 24 parallel `claude` workers; release-filter takes under 2 min.

## Graph quality checks (`modsleuth check`)

```bash
modsleuth check --source path/to/graph.json
```

Deterministic, read-only invariant checks over any merged graph, each
corresponding to a defect class observed while QA'ing real pipeline
output. Blocker-grade (P1) findings: edges with empty `anchor_list`,
`dependency_kind` labels contradicting their canonical relation,
self-loops, duplicate `(subject, relation, object)` triples, subjects
that resolve to neither a lattice item nor a virtual concept address,
and fully flattened lattices (every group a singleton). Review-grade
(P2) findings cover dataset-kind subjects on weight-lineage relations,
families without roots, duplicate lattice items, and unparseable
bracket-style names; P3 lines report distribution facts (relation
histogram, evaluation-edge share, off-lattice object share, evidence
provenance coverage).

Exit code is 1 when any P1 finding exists, 0 otherwise — usable as a
gate between pipeline stages or in CI. Nothing is mutated and nothing
is judged semantically: findings route edges to review, they never
auto-fix.

## Visualizer

```bash
modsleuth viz --source path/to/graph.json --port 8102
```

A self-contained HTTP server with a Cytoscape + dagre frontend. The UI
has two tabs — **Graph** (force / dagre layout, lattice subsumption
edges, family chips, dep/relation filters) and **Operations** (each
training event grouped with its participating edges, descriptions, and
anchor citations) — plus a slide-out detail panel on the right that
surfaces aliases, link badges (HF model / dataset / collection, paper,
GitHub, blog, vendor docs), per-edge anchor blockquotes (verbatim
excerpts and explanations grounded to a source path), and outgoing /
incoming edge lists for any selected node.

For graphs above a few thousand edges, the all-at-once view is rarely
insightful — pass `--seed` to pre-prune the payload to a focused
ego-expansion centered on a chosen node:

```bash
modsleuth viz --source path/to/graph.json \
    --seed "Olmo-3-1025-7B" --depth 2 --target-size 80
```

The seed pattern is a case-insensitive substring matched against each
node's `formal_name` and aliases; the highest-degree match wins. From
there, BFS expands up to `--depth` hops and admits the highest-scored
neighbors first until ~`--target-size` nodes are captured. Edge scoring
prefers lineage-bearing relations (`trained_from`, `trained_on`,
`generated_by`, `transformed_by`, `filtered_by`, `merged_from`,
`composed_from`) and discounts evaluation/citation clutter
(`used_for_evaluation`, `cited_as_baseline`, `used_for_ablation`); direct
dependencies and anchor-grounded edges get small bumps. The result is a
focused subgraph that shows the actual training/data lineage around the
seed instead of a dense mass of evaluation edges.

Tunables:

- `--depth N` — hops to expand from the seed (default 2). Use 1 for
  immediate neighbors only, 3 for wider context.
- `--target-size N` — approximate node budget for the seeded expansion
  (default 80). The BFS stops admitting once it hits this; smaller is
  more readable.
- `--top-k N` / `--min-degree N` — server-side prune by degree (used
  only when `--seed` is not given). Useful for a degree-filtered global
  view of a large graph without picking a center.

Combine multiple framings by re-running with different `--seed` values
or different `--depth` / `--target-size` budgets.

## Baseline task prompts

`baselines/` contains the single-prompt specification of the
tracing task that the single-pass baseline systems received in the
comparison reported in the paper. `baseline_prompt.md` is the shared
template — it spells out the node / edge / anchor JSON contract, the
relation vocabulary, and the scoping rules — and the four
`baseline_prompt_<subject>.md` files are its per-target instantiations
(OLMo 3, Nemotron 3 Super, DR-Tulu, SmolLM3). Any web-capable agent
pointed at one of them produces a graph directly comparable to
ModSleuth's output for that target.

The launch scripts and evaluation harness behind the paper's
comparison are not part of this software release. The dependency
graphs recovered in the paper are browsable and downloadable from the
demo (https://modsleuth.cal-data-audit.org).

## Repository layout

```
.
├── modsleuth/                  # ModSleuth pipeline package
│   ├── cli.py                  # `modsleuth` CLI entry point
│   ├── check.py                # deterministic graph-quality checks
│   ├── config.py               # storage / model / env defaults
│   ├── pipeline.py             # all stage implementations
│   ├── prompts/                # stage-level markdown prompts
│   │   ├── discover.md         #   Gather (Stage 1)
│   │   ├── extract.md          #   Extract (Stage 2)
│   │   ├── organize.md         #   Resolve build (Stage 3)
│   │   ├── audit.md            #   Resolve revise (Stage 3)
│   │   ├── relate.md           #   Relate (Stage 4)
│   │   └── triage.md           #   Reconcile audit step (Stage 5)
│   ├── resolve.py              # identity-lattice resolver
│   ├── store.py                # SQLite-backed pipeline state
│   ├── subsets.py              # HF metadata pre-pass for the audit stage
│   ├── viz.py                  # interactive HTTP graph viewer
│   ├── recursive.py            # multi-hop expansion driver
│   └── dedup/                  # post-merge dedup pipeline (4 sub-stages)
│       ├── __main__.py         #   `python -m modsleuth.dedup`
│       └── lib.py              #   shared helpers (signatures, union-find, …)
├── baselines/                  # baseline task prompts (template + per-target)
│   ├── baseline_prompt.md
│   └── baseline_prompt_<subject>.md
├── pyproject.toml
├── requirements.txt
├── schema.sql
└── README.md
```

## Concepts

- **operation** — a structured group of edges that jointly describes one
  pipeline event (e.g., a DPO step). Edges within an operation share an
  anchor list and description, preserving the event structure that a flat
  pairwise edge list would erase. Merge artifacts carry a top-level
  `operations` index (`operation_id` → event description, event
  anchors, source batch) and every row in `relations` lists the
  `operation_ids` it belongs to, so event structure survives
  reconciliation and cross-run merging; each row additionally carries
  its own edge-level `description` and `anchor_list`.
- **anchor** — a source-side citation: file path, position, and verbatim
  excerpt grounding a claim to a specific spot in the source corpus.
- **identity lattice** — partial-order structure for artifact identity with
  vague-mention roots, partial-spec intermediate nodes, and pinned-link
  entity leaves. Each node is an open-vocabulary set of facets (`family`,
  `size`, `stage`, …); subset ordering on facet sets defines the hierarchy.
- **dependency-kind** — coarse type label (`direct` / `indirect`) on every
  edge, distinguishing artifacts that materially enter weights or training
  data from those that merely influence development decisions.
- **evidence provenance** — every source batch is stamped with the
  tracing target it was gathered for; edges inherit the stamp and
  accumulate `traced_targets` as runs corroborate them. A batch whose
  exact source content already exists keeps its *first* target
  (first-writer-wins at the batch level; union-at-the-edge level), and
  an edge whose subject's org never appears among its evidence targets
  is flagged `provenance_review: true` — routed to review, never
  dropped.
- **edge quarantine** — relate validation is strict per edge, not per
  batch: structurally invalid edges (no anchors, unresolved subject,
  self-loop, …) move to the artifact's `rejected_edges[]` with reasons
  while the valid remainder proceeds; string anchor entries are
  mechanically coerced to `{source: …}` objects first. A batch fails
  only when nothing valid survives.
