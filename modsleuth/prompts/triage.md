# Triage Upstream Nodes for Recursive Expansion

> **Goal: read the lattice + relations, classify every
> upstream entity-leaf as `auto_expand`, `decline`, or
> `manual`.** The output gates expansion two ways: the operator
> works the queue with `modsleuth run expand --node <formal_name>`,
> and the recursive driver (`modsleuth recursive`) expands
> `auto_expand` nodes automatically each round (depth and beam
> bounded by its flags). Your classification IS the gate.

Read `{{lattice_path}}` and `{{relations_path}}`. Write the
classification artifact to `{{artifact_path}}`.

## Filesystem scope

Read `{{lattice_path}}` (groups+items+links from
audit / organize) and `{{relations_path}}` (relations
artifact, possibly merged across batches). Write
`{{artifact_path}}`. Web search permitted for sanity-checking
borderline nodes; use sparingly.

## What "upstream" means

An entity-leaf is **upstream** of the original target if the
lattice contains an edge whose **subject** is the target (or a
descendant of the target) and whose **object** is that
entity-leaf. Concretely: walk the relations file. For every
edge whose `subject` is the run's target or transitively
trained-on/trained-from/generated-by/transformed-by/
filtered-by the target's lineage, the `object` (when it
resolves to a lattice formal_name) is upstream and a candidate
for triage.

Do not triage:
- Concept-level nodes (no resolvable link).
- The original target itself.
- Off-lattice mentions (free-text `object` strings that don't
  match any lattice formal_name) — they have no canonical
  release to expand.

## Classification

For each upstream entity-leaf, emit ONE of three
classifications with a one-line `rationale`:

### `auto_expand`

**Default to expansion when the node has documented lineage to
chase.** Recursion is what produces the lineage findings that
single-hop extract can't see — chains of distillation,
multi-stage data mixing, judges-of-judges, OCR / rewriter
models whose training corpora propagate through. Bias toward
expanding when the documentation bar below is met — but note
that in recursive runs this bucket is executed automatically,
so `auto_expand` is a commitment to spend an expansion run,
not a suggestion for a human to filter.

The single gate: the node must have **documented composition
the next pass can extract from.** Bar is high by default —
narrative description alone doesn't qualify; extract needs
structured sources to read.

**Primary forms** (either qualifies):

- A **detailed dataset card** with an explicit composition /
  sources / mix table that enumerates components. A
  one-paragraph card description does NOT qualify.
- A **standalone tech report or paper** on the node OR on
  its family root (papers are commonly attached at root
  level).

**Carve-out** — qualifies even without a card or paper:

- **Substantial documented composition in code** — a training
  repo whose configs / YAMLs / manifests genuinely enumerate
  mixture sources (e.g., `allenai/OLMo-core`'s
  `--dataset_mixer_list`, `allenai/dolma`'s source manifests).
  The bar is "extract has structured sources to read"; a
  passing README mention does not meet it.

**Explicitly does NOT qualify on its own:** release blogs and
vendor docs. Too narrative for extract to parse cleanly. If a
blog is the only documentation, decline.

If documentation exists at this bar, expand. Don't try to
predict whether recursion will be "interesting" — that
judgment belongs at extract time, not triage.

Rationale should name the relation that motivated expansion
AND the documentation anchor (e.g., `"trained_on at stage 1;
mixture documented in dolma3-mix card composition table"`).

### `decline`

Should NOT be expanded. Cases, in priority order:

1. **Closed-data model families.** Hardcoded skip even when a
   paper exists — the paper covers architecture / evals but
   the training data isn't published, so recursion yields no
   lineage edges. Match by org / family prefix:

   - Qwen — `Qwen/`, `Qwen2/`, `Qwen3/`, ...
   - Llama — `meta-llama/`, formal names starting `Llama-`
   - DeepSeek — `deepseek-ai/`
   - Kimi — `moonshotai/`, formal names starting `Kimi-`
   - Gemma — `google/Gemma*`, `google/gemma-*`
   - Mistral / Mixtral — `mistralai/`
   - GPT / o-series — `openai/`
   - Claude — `anthropic/`
   - Gemini — `google/Gemini*`, `google/gemini-*`
   - Phi — `microsoft/Phi*`, `microsoft/phi-*`
   - Yi — `01-ai/Yi-*`
   - Falcon — `tiiuae/`
   - Command-R — `CohereForAI/`, `cohere/Command-R*`

   Bias is skip-by-default; the operator can override via
   `manual` if a specific release is known to be open-data.
   Extend the list as new closed-data families surface.

2. **No documentation to recurse into.** No paper, no tech
   report, no composition table, no code-repo configs —
   re-extracting a one-paragraph card finds nothing new.

3. **Evaluation / ablation only.** Relation is
   `used_for_evaluation` / `cited_as_baseline` /
   `used_for_ablation` and nothing in the training pipeline.
   Recursion conflates eval with lineage.

4. **Deprecated / inaccessible.** HF dataset removed,
   irrecoverably gated, link rot.

Rationale should name which criterion fires (e.g.,
`"closed-data family (Qwen); training data not published"`,
`"used_for_evaluation only"`, `"no paper / tech report;
composition undocumented"`).

### `manual`

Operator should decide. Use sparingly — only when the
documentation status is genuinely ambiguous:

- Documentation exists but its scope is unclear (paper
  covers architecture only? mixture is partially specified?).
- Closed-data status is borderline — e.g., a fine-tune of a
  closed base where the fine-tune data IS published. The
  fine-tune's own lineage is partially extractable.
- A personal-namespace HF dataset with a composition hint but
  no anchoring paper, where operator judgment on budget is
  needed.

Rationale should describe what the operator needs to weigh.

## Output schema

```json
{
  "auto_expand": [
    {
      "formal_name": "allenai/dolma3-mix",
      "kind": "dataset",
      "primary_link": "https://huggingface.co/datasets/allenai/dolma3-mix",
      "rationale": "trained_on at stage 1; HF card composition table enumerates sub-mix sources",
      "motivating_relations": ["trained_on"]
    },
    {
      "formal_name": "allenai/olmOCR-7B-0225",
      "kind": "model",
      "primary_link": "https://huggingface.co/allenai/olmOCR-7B-0225",
      "rationale": "rewriter model whose training defines absorbed content; tech report documents its own training corpus",
      "motivating_relations": ["transformed_by"]
    }
  ],
  "decline": [
    {
      "formal_name": "Qwen/Qwen3-32B",
      "kind": "model",
      "primary_link": "https://huggingface.co/Qwen/Qwen3-32B",
      "rationale": "closed-data family (Qwen); used as DPO judge but training data not published — recursion yields no lineage edges",
      "motivating_relations": ["filtered_by"]
    },
    {
      "formal_name": "Qwen/Qwen3-7B-Instruct",
      "kind": "model",
      "primary_link": "https://huggingface.co/Qwen/Qwen3-7B-Instruct",
      "rationale": "used_for_evaluation only; expanding would conflate eval and training lineage",
      "motivating_relations": ["used_for_evaluation"]
    }
  ],
  "manual": [
    {
      "formal_name": "hamishivi/rlvr_general_mix",
      "kind": "dataset",
      "primary_link": null,
      "rationale": "personal-namespace mix; HF card has partial composition hint but no anchoring paper — operator should decide if expansion budget is justified",
      "motivating_relations": ["trained_on"]
    }
  ]
}
```

Every entity-leaf upstream of the target must appear in
exactly one bucket. If you skipped a node (concept, target,
off-lattice mention only), include it in a top-level
`skipped[]` array with a one-line reason — that lets the
operator audit coverage.

## Subagent dispatch

The Task tool is available — subagents run as
`{{subagent_model}}`. For a small lattice (< 50 upstream
nodes) classify inline. For a larger one, bucket by family
and dispatch subagents. Each subagent classifies its slice
and returns the same shape; aggregate before writing.

When dispatching, transcribe the three classification
definitions and rationale-style guidance verbatim — subagents
have none of your context.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
