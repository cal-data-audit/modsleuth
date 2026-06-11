# Extract Operations and Edges from this Batch's Sources

> **Goal: read this batch's source files, identify every
> training / data-construction / evaluation EVENT, and append
> ONE JSON line per event to `{{artifact_path}}`. Each event
> wraps its participating edges. Edge subjects are ALWAYS models
> resolved to lattice formal_names — never datasets, never
> off-lattice strings. Edge objects are models, datasets, or
> free-text descriptors. Only `model` and `dataset` are valid
> node types.**

Read the lattice at `{{lattice_path}}` and the source files
under `{{batch_dir}}`. Append events as JSON-Lines records to
`{{artifact_path}}`.

## Append-as-you-go (load-bearing)

You are NOT producing a single giant JSON object at the end.
Instead, **after you finish identifying each event, append one
JSON object as a single line** to `{{artifact_path}}`. The file
is JSONL — one record per line, each a self-contained event.
Use:

```
cat >> {{artifact_path}} <<'EOF'
{"description": "...", "anchor_list": [...], "edges": [...]}
EOF
```

(or use the Edit tool to append; either works). Don't keep
prior events in your working memory — they're persisted on
disk. After your turn, the pipeline reads this JSONL,
validates it, and assembles the per-batch artifact.

This means: NO operation IDs to maintain across edges. Edges
live INSIDE their operation as a nested array. If you forget
op-ids you've assigned, that's fine — there are no op-ids.

## Filesystem scope

Read `{{lattice_path}}` (the post-audit lattice — every dataset
node has `subsets[]` populated) and every file under
`{{batch_dir}}` recursively (PDFs, markdown, code, configs).
Skip `__pycache__`, `node_modules`, `.git`, `venv`. Do not
read anything outside `{{batch_dir}}` except the lattice.

Web search is **off** for this stage — we want grounded claims
from the source files, not synthesized knowledge.

## What is an event (operation)

A real training or data-construction event involves multiple
participants playing different roles. The OLMo-3 7B Think DPO
event, for example, involves at least four participants:

- the resulting model (`allenai/Olmo-3-7B-Think-DPO`)
- the base checkpoint (`allenai/Olmo-3-7B-Think-SFT`)
- the chosen-completion generator (`Qwen/Qwen3-32B`, thinking)
- the rejected-completion generator (`Qwen/Qwen3-0.6B`, thinking)

Capture this as ONE event whose `description` names what
happened, with N edges nested inside (one per participant
role).

Pair-only facts (e.g., "OLMo-3 inspired_by SwallowMath
recipe") wrap as a singleton-edge event — same shape, just
one edge in the array. Uniform schema.

## Lattice anchoring — preserve source specificity

The lattice is a partial order: every family has a **family
root** (concept node, identity `{family: X}` only) and zero-or-
more **entity leaves** (full identity, with HF / GitHub / vendor
docs link), plus interior **concept nodes** (partial identity,
no item-unique URL). Each item also carries a `subsets[]` field
listing slugs of sub-corpora the dataset contains.

**Use the most specific entity / concept the source pins.** Vague
mentions land on the family root or on intermediate concept
nodes; precise mentions land on leaves. Don't upgrade vague
mentions to specific leaves, and don't downgrade specific
mentions to roots.

### Resolving a mention string to a lattice address

For each mention you'd put in `subject` or `object`, run:

```
python -m modsleuth.resolve "<mention>" --top 5 --json
```

The output is a ranked list of candidates. Each candidate has:

- `formal_name`, `family`, `identity`, `kind`, `score`
- `address_form` — `"leaf"` | `"concept"` | `"root"` | `"subset"`
- `match_reasons` — why this candidate scored where it did
- `subset_of` (only when `address_form == "subset"`) —
  `{parent_formal_name, slug}`

Pick the candidate whose identity facets are closest to what the
**surrounding source context** implies. Three rules:

1. **Score < 50 with no `subset` candidate** — treat as
   off-lattice, emit the literal source mention as a free-text
   string in `subject` or `object`.

2. **`address_form == "subset"`** — the mention is a sub-corpus
   of a parent dataset (e.g., `MegaMath-Web` is a config of
   `LLM360/MegaMath`). **Use the parent's `formal_name` as the
   address**, and note the subset in the edge's `description`
   field (e.g., `"... trained on MegaMath-Web subset of
   LLM360/MegaMath ..."`). Do NOT emit a separate sub-corpus
   address; the lattice stops at the HF-dataset / parent level.

3. **Concept top-1, entity at rank 2, source pins specifics** —
   if surrounding ±2 sentences provide a date / stage / size that
   matches the entity but not the concept, prefer the entity.
   Otherwise the concept address is the right call (vague
   mention).

If `python -m modsleuth.resolve` finds no candidate with a family
pivot (score floor of 0 returned), fall back to free text in
`subject` or `object`. The query layer distinguishes
"address resolves to lattice item" vs "free text" at read time;
no flag needed on the edge.

### Subject vs object specificity

- **Subjects MUST be Models.** No dataset-subject edges, ever.
  Subject is usually a leaf checkpoint (pinned via
  `from_pretrained()` in configs). May also be a model concept
  (family-root or interior concept) when source describes an
  event at the family / stage level (e.g., "for the OLMo 3
  Think family we use Qwen3 32B as judge" → subject is the
  `{family: OLMo 3, stage: Think}` concept node, fan out per-leaf
  if the global-policy-edges rule applies).
- **Objects** can be Models, Datasets, family roots, concept
  nodes, or free-text. Cannot be tokenizers / frameworks /
  software libraries / codebases — those aren't node types.

If a source describes an action whose natural English subject
is a dataset (e.g., "olmOCR transformed PDFs into Dolma"),
restate with the consuming model as subject and the producer
as object: `<model-that-trained-on-Dolma> --transformed_by-->
allenai/olmOCR-7B-0225`. The dataset's role in this still
appears via the model's `trained_on` edge to the dataset.

If an artifact is mentioned only as a side-comment with no
relation to anything in the lattice, drop it; don't invent
relations.

## Subject is always a Model (load-bearing schema rule)

Every edge's `subject` MUST resolve to a lattice item with
`kind == "model"`. No dataset-subject edges. No free-text
subjects. If the source describes an action whose natural
subject is a dataset (e.g., "olmOCR transformed PDFs into the
academic corpus"), restate the edge with the **consuming model**
as subject:

```
WRONG:    dolma3_pool --transformed_by--> allenai/olmOCR-7B-0225
RIGHT:    allenai/Olmo-3-1025-7B --transformed_by--> allenai/olmOCR-7B-0225
          (with description noting the OCR'd content lives in dolma3_pool)
```

If multiple consumer models trained on the same dataset, fan
out: emit ONE `(consumer, action, producer)` edge per consumer.
Same fact at multiple subjects is exactly what the schema
expects — graph-level joinability comes from the consumer-as-
subject framing.

## Subject locality — sources are official only for the tracing target

The tracing target for this batch is: **{{traced_target}}**.

This batch holds the tracing target's own official materials. They
are authoritative for the target's release family (its sizes, stages,
and sibling models) and for other artifacts released by the same
organization — and for nothing else. A claim about some OTHER
organization's model found here is third-party evidence for that
model, and an edge built on it cannot be verified against the
subject's own documentation.

- Emit edges whose subject is the tracing target, a member of its
  release family, or another model from the same organization whose
  pipeline this batch's sources document first-hand.
- If a source here claims that a foreign organization's model used or
  trained on an artifact (e.g., a dataset card noting "X also trains
  on us" where X is unrelated to the target), do NOT emit that edge
  from this run. The claim belongs to X's own tracing run, where X's
  official materials can back it first-hand.
- The global-policy fan-out rule below stays within the tracing
  target's release family; it never licenses foreign-org subjects.

## Aggregator + leaf rule (load-bearing)

When `subject --trained_on--> aggregator` and the aggregator
has populated `subsets[]` listing named leaf sub-corpora
(e.g., `cranecode`, `cranemath`, `web`, `web-pro`), emit edges
at BOTH granularities:

- **Aggregator-level**: `subject --trained_on--> <aggregator>`
- **Leaf-level**: `subject --trained_on--> <leaf>` for EACH slug
  in the aggregator's `subsets[]`

The leaf address is the lattice item whose `formal_name` matches
the slug semantically. Common conventions:

- HF-prefixed leaf: `allenai/cranemath` (preferred when the leaf
  has its own HF release)
- Slug-only when no separate item exists: `cranemath` or
  `<aggregator-formal-name>/cranemath`

Pick the form that matches the lattice item if one exists; else
synthesize `<aggregator-formal-name>/<slug>` so the verifier
sees the explicit composition.

**Leaf edges look redundant with the aggregator edge but they
capture the dependency at the granularity reference graphs use.
Do NOT drop them.**

If the source says only some sub-corpora were used (e.g.,
"Olmo-3-7B-Base used CraneMath and FineMath4+ but not the
OMR-Rewrite subset"), emit leaf edges only for the named ones.

Sub-corpus mentions in source that match a slug pattern (e.g.,
`MegaMath-Web` matching `web` in LLM360/MegaMath's `subsets[]`)
resolve via `python -m modsleuth.resolve` with `address_form:
"subset"`. Emit the leaf edge using the resolve output's
`subset_of.parent_formal_name` + `subset_of.slug`. The
description should also name the sub-corpus verbatim for
query-time grep.

## Global-policy edges

When the source describes a model as the **global** judge,
generator, filter, or transformer for a **class** of training
stages, emit ONE edge per training-stage subject in the
lattice that participates in that class.

Example: the OLMo-3 paper says "Unless otherwise stated, for
an LM judge we host **Qwen3 32B** ... thinking mode turned
off". This is a global default for ALL OLMo-3 RLVR runs.
Emit `filtered_by Qwen/Qwen3-32B` for **every** OLMo-3
RLVR-trained model: `Olmo-3-7B-Think`, `Olmo-3-32B-Think`,
`Olmo-3.1-32B-Think`, `Olmo-3-7B-Instruct`, the 5+ RL-Zero
variants. Each lands in its own event (no shared op).

Same pattern for global generators: "we distill thinking
traces from GPT-4.1 and o4-mini" → emit `generated_by` edges
to both objects (off-lattice via free-text `object` if the
lattice doesn't have them) for every model whose midtraining
or post-training mixture incorporated those traces.

A global statement that you DO NOT expand to per-stage edges
will manifest downstream as missing license-inheritance and
provenance edges. Err on the side of expanding.

## Relation taxonomy (canonical labels — coining allowed when none fits)

`relation` is an open string, but the canonical labels below
cover the vast majority of LLM-pipeline events. The 5 DIRECT
and 3 INDIRECT buckets are designed to be orthogonal — every
direct dependency that fits one of them lands in exactly one.

**Use a canonical label when one fits.** Only coin a new
snake_case label when the source describes an event that none
of the canonical values capture, AND the object is a valid
node (model or dataset). Examples of legitimate coining:
`merged_from` (model souping — object is a Model),
`embedded_by` (object is an embedding Model). Five recurrent
coined labels have fixed kinds — `merged_from`,
`quantized_from`, `embedded_by`, `decontaminated_against`, and
`composed_from` are all `direct`. Do NOT coin
relation labels whose object would be a tokenizer / dedup tool /
decontamination tool / framework — those targets aren't valid
nodes.

**Do NOT coin** when the canonical label fits. The planner
that emits `cited_as_baseline` instead of skipping the edge
entirely (baseline comparisons are out-of-scope, see below)
is bad reasoning, not bad coining.

When you coin, the edge `description` MUST explain the role
clearly enough that a downstream reader understands what the
new label means without re-reading the source.

### DIRECT relations (`dependency_kind: "direct"` — object enters training)

| relation | typical direction | when to use |
|---|---|---|
| `trained_on` | M → D | the dataset is pretraining or post-training data for the model |
| `trained_from` | M → M | weight initialization (subject's parameters copied / continued from object) |
| `generated_by` | M → M | object model generated content used as subject's training data (synthetic data, distillation traces, RL rollouts) |
| `transformed_by` | M → M / M → D | object rewrote / OCR'd / reformatted content used in training (content originated elsewhere; object only modified) |
| `filtered_by` | M → M | object decided inclusion (judge in pairwise preference, quality classifier, RM-as-filter) — content unchanged |

### INDIRECT relations (`dependency_kind: "indirect"` — object shaped subject without entering training)

| relation | typical direction | when to use |
|---|---|---|
| `inspired_by` | M → M / M → D | methodology / data-recipe borrowed from a SPECIFIC published model or dataset ("based on SwallowMath data recipe — `tokyotech-llm/swallow-math`"; "follows DCLM filtering — `mlfoundations/dclm`"). Object MUST be a model or dataset; NOT a codebase, NOT a software library, NOT a paper standalone. If the recipe lives in a paper that has no released model / dataset artifact, do not emit. No weight or data inheritance. |
| `used_for_ablation` | M → M / M → D | object was a design-space variant in the subject team's OWN ablation studies (not a baseline they compared against) |
| `used_for_evaluation` | M → M / M → D | benchmark / eval set OR LLM judge model — used to evaluate the release. **Enumerate every named benchmark individually** when the source lists them in eval tables (MMLU, GSM8K, BBH, HumanEval, DeepMind Math, LBPP, AGIEval, ...) rather than collapsing into an umbrella name like "OlmoBaseEval" or "Held-out Suite". |

### Out-of-scope — do NOT emit

- **Baseline comparisons** ("we report scores against Llama-3
  in Table 7", "our model outperforms GPT-4 on GSM8K"). These
  are lateral comparisons, not provenance. Emit nothing — no
  node, no edge. The test: was this artifact part of the
  team's OWN development pipeline (ablation, eval, methodology
  borrowing), or just a published number to compare against?
  Only the former is in scope.
- **Generic architecture / algorithm primitives** (Transformer,
  RoPE, RMSNorm, AdamW, MoE, GQA, SwiGLU). These are math/ML
  techniques, not artifacts. Never edges.
- **Vague inspiration** ("inspired by the broader RL
  literature", "following common practice"). Never edges.

- **Tokenizers, frameworks, training software, inference
  infrastructure** (PyTorch, vLLM, Transformers, datatrove,
  tiktoken, ray, OLMo-core the codebase, Resiliparse the
  library, Duplodocus the dedup tool, `allenai/dolma3-tokenizer`
  the BPE vocab). NOT NODES. They are
  software / libraries / vocab files, not models or datasets.
  Skip entirely. Don't emit an edge to them, don't emit them as
  nodes. Only `model` and `dataset` are valid node types.
  *Distinction:* a neural OCR MODEL (`allenai/olmOCR-7B-0225-
  preview`, 7B params) IS a model and IS a valid edge target;
  the tokenizer it ships with is not. A neural quality
  classifier (Gemma-3, FastText) IS a model; the framework it
  runs in is not.
- **Numeric facts about a node** (size, training_tokens,
  release_date, parameter_count, context_length). These are
  NOT edges. They live in node descriptions. The lattice
  validator will reject them as relations.

### No STRUCTURAL bucket

Subset / contains / supersedes / released_with relationships
are NOT emitted as edges. Their information lives in:
- The dataset node's `subsets[]` field (composition).
- The aggregator+leaf rule (subject emits leaf-level edges).
- Node descriptions (release ordering, supersession,
  bundling).

This avoids dataset-as-subject edges and matches the way
reference graphs encode lineage.

## Why no overlap among the 5 DIRECT buckets

Two orthogonal axes separate them:

- **Content origin** — where did the training content come from?
  - external data → `trained_on`, `transformed_by`, `filtered_by`
  - external weights → `trained_from`
  - the object model itself → `generated_by`

- **What the object did to the content**:
  - the object IS the content (weights or generated text) →
    `trained_from`, `generated_by`, `trained_on`
  - the object **rewrote** existing content → `transformed_by`
  - the object **only decided inclusion** → `filtered_by`

```
                  | content from object | content from elsewhere, unchanged | content from elsewhere, modified
weights           | trained_from        | —                                  | —
content (data)    | generated_by        | trained_on (data) /                | transformed_by
                  |                     | filtered_by (decision-only model)  | (rewriter model)
```

Every M-to-M-or-M-to-D direct dependency lands in exactly one
cell. If you can't decide between two cells, the edge probably
needs to be split into two edges along different axes — emit
both.

### Worked examples

- `allenai/Olmo-3-7B-Instruct` is initialized from
  `allenai/Olmo-3-7B-Base` → `trained_from`.
- `allenai/Olmo-3-7B-Base` was pretrained on
  `allenai/dolma3_mix-6T-1025-7B` → `trained_on` (aggregator);
  also emit per-leaf `trained_on` edges for every subset slug
  in that aggregator's `subsets[]`.
- A paragraph in Dolma-3 was OCR'd from a PDF by
  `allenai/olmOCR-7B-0225`. Subject is the OLMo-3 model that
  trained on the OCR'd content; object is the OCR model. →
  `transformed_by`. Content came from the PDF, not from
  olmOCR.
- An RLHF DPO run for `allenai/Olmo-3-7B-Instruct-DPO` used
  `Qwen/Qwen3-32B` as the judge that picked preferred
  responses. → `filtered_by` (judge decided inclusion; it
  didn't generate the responses, it ranked them).
- `CraneMath` was generated by `Qwen/Qwen3-32B` rewriting
  math problems. From the model that trained on the parent
  Dolmino mix: `generated_by` Qwen3-32B (the content was
  produced by Qwen3-32B). The CraneMath dataset itself shows
  up as a leaf-level `trained_on` edge per the
  aggregator+leaf rule.

## Edge schema (nested inside an event)

```json
{
  "subject":         "<lattice formal_name, MUST be kind=model>",
  "relation":        "trained_on",
  "dependency_kind": "direct",
  "object":          "<lattice formal_name OR free-text descriptor>",
  "description":     "<lossless 1-3 sentences, ≤ ~500 chars>",
  "anchor_list":     [
    {
      "source":      "<URL or local path>",
      "position":    "<locator within source: section, page, table, line range, YAML field>",
      "excerpt":     "<verbatim quote from the cited source, ≤ ~200 chars>",
      "explanation": "<one sentence on how the excerpt supports the (subject, relation, object) claim>"
    }
  ]
}
```

- `subject`: MUST be one of:
  - a leaf `formal_name` whose lattice item has `kind == "model"`,
  - a family-root or interior concept `formal_name` whose item
    has `kind == "model"`,
  - a virtual concept address `<family> [<k>=<v>, ...]` notation
    when the source's specificity falls between root and leaf,
    AND the family is a model family.

  **Never emit a dataset as subject.** If the source describes
  an action whose natural subject is a dataset, restate with
  the consuming model as subject (see "Subject is always a
  Model" section above).
- `relation`: canonical from the table above when one fits;
  otherwise a coined snake_case label whose object is also a
  valid model/dataset node.
- `dependency_kind`: `"direct"` or `"indirect"`. Closed
  vocabulary; mismatch with the relation's bucket is a
  validation error.
- `object`: same shape as subject (leaf / root / concept) for
  model or dataset objects, OR a free-text descriptor when no
  family pivot exists. Cannot be a tokenizer / framework /
  software / codebase.
- `description`: lossless prose. MUST capture every
  structurally relevant fact that the relation, subject,
  object, and event description don't already express:
  training stage (sft/dpo/rl/midtraining/long_context),
  role sub-variants (Think-SFT vs Instruct-SFT, math vs
  code), quantities (prompt counts, token counts), specific
  subsets / filters, ordering / compositional context,
  caveats. ≤ ~500 chars.
- `anchor_list`: NON-EMPTY array. Each entry:
  - `source` (REQUIRED): URL or local path to the source.
  - `explanation` (REQUIRED): one sentence on HOW the cited
    source supports the specific `(subject, relation, object)`
    claim. Don't restate the source text — use `excerpt` for that
    and explanation for the connection.
  - `position` (RECOMMENDED): locator within the source —
    section, page, table, figure, line range, YAML field path.
  - `excerpt` (RECOMMENDED): a **verbatim quote** (≤ ~200 chars)
    from the cited source supporting the claim. For tables,
    figures, or non-quotable content, set `position` precisely
    and leave `excerpt` empty (acceptable in those cases).

  Verification reads the cited evidence and checks excerpts /
  locations when present; an edge without them can only be
  corroborated by independent search, which under-grounds the
  claim. Populate `excerpt` + `position` whenever the source
  allows.

## Event schema (one JSONL line)

```json
{
  "description": "<lossless prose describing the event itself: what happened, who participated in what role, in which training stage>",
  "anchor_list": [
    {"source": "...", "position": "...", "explanation": "..."}
  ],
  "edges": [
    { ... edge 1 ... },
    { ... edge 2 ... },
    { ... edge 3 ... }
  ]
}
```

- `description`: event-level prose. Distinct from per-edge
  descriptions: this captures the EVENT (the training run, the
  dataset construction, the eval pass), while edges describe
  individual participant roles.
- `anchor_list`: same shape as edge anchors; supports the
  event-level claim. Often the same primary source as
  per-edge anchors but can include event-wide citations
  (e.g., a recipe section that lists all participants).
- `edges`: NON-EMPTY array. Each edge follows the schema above.

## Worked example — full event

```json
{
  "description": "OLMo-3-7B Think DPO post-training event: continued from Olmo-3-7B-Think-SFT, with Qwen3-32B (thinking mode) generating chosen-completion candidates and Qwen3-0.6B (thinking mode) generating rejected candidates per the Delta-Learning recipe (Section 4.3.1).",
  "anchor_list": [
    {
      "source": "https://arxiv.org/abs/2512.13961",
      "position": "Section 4.3.1, paragraph beginning 'For Think-DPO'",
      "excerpt": "For Think-DPO, we use Qwen3 32B in thinking mode to generate chosen completions and Qwen3 0.6B in thinking mode to generate rejected completions, following the Delta-Learning recipe.",
      "explanation": "Paper paragraph names both generators and the recipe in one place."
    }
  ],
  "edges": [
    {
      "subject": "allenai/Olmo-3-7B-Think-DPO",
      "relation": "trained_from",
      "dependency_kind": "direct",
      "object": "allenai/Olmo-3-7B-Think-SFT",
      "description": "Olmo-3-7B-Think-DPO is initialized from the Think-SFT checkpoint and continues training with DPO preference optimization.",
      "anchor_list": [
        {
          "source": "https://arxiv.org/abs/2512.13961",
          "position": "Section 4.3.1",
          "excerpt": "Think-DPO continues from the Think-SFT checkpoint with preference optimization.",
          "explanation": "States the warm-start of Think-DPO from Think-SFT."
        }
      ]
    },
    {
      "subject": "allenai/Olmo-3-7B-Think-DPO",
      "relation": "generated_by",
      "dependency_kind": "direct",
      "object": "Qwen/Qwen3-32B",
      "description": "Qwen3-32B (thinking mode on) generated chosen-completion candidates for the Think-DPO preference pairs per the Delta-Learning recipe.",
      "anchor_list": [
        {
          "source": "https://arxiv.org/abs/2512.13961",
          "position": "Section 4.3.1, Delta-Learning paragraph",
          "excerpt": "Qwen3 32B in thinking mode generates the chosen completions for the preference pairs.",
          "explanation": "Documents Qwen3-32B as the chosen-completion generator."
        }
      ]
    },
    {
      "subject": "allenai/Olmo-3-7B-Think-DPO",
      "relation": "generated_by",
      "dependency_kind": "direct",
      "object": "Qwen/Qwen3-0.6B",
      "description": "Qwen3-0.6B (thinking mode on) generated rejected-completion candidates for the Think-DPO preference pairs per the Delta-Learning recipe.",
      "anchor_list": [
        {
          "source": "https://arxiv.org/abs/2512.13961",
          "position": "Section 4.3.1, Delta-Learning paragraph",
          "excerpt": "Qwen3 0.6B in thinking mode generates the rejected completions.",
          "explanation": "Documents Qwen3-0.6B as the rejected-completion generator."
        }
      ]
    }
  ]
}
```

That's ONE line in `{{artifact_path}}`. The next event you
identify is the next line. No cross-event references.

## Coverage expectation

There is no fixed target count. The number of events depends
on how richly the source describes the pipeline. Use these
qualitative checks:

- For every training stage the source names (pretraining,
  midtraining, long-context, SFT, DPO, RL, RL-Zero), there
  should be at least one event covering it.
- **Aggregator + leaf coverage:** for every aggregator mix
  the subject `trained_on`, emit ONE `subject --trained_on--> <leaf>`
  edge per slug in the parent's `subsets[]`, IN ADDITION to the
  aggregator-level edge. See "Aggregator + leaf rule" section.
- For every named generator / judge / rewriter / classifier
  the source mentions, emit a `generated_by` / `filtered_by`
  / `transformed_by` edge with the **consumer model as subject**
  (the model whose training data was generated / filtered /
  transformed by the producer model). E.g., `OLMo-3-1025-7B
  --transformed_by--> allenai/olmOCR-7B-0225` (OLMo-3 trained
  on data that olmOCR transformed), NOT `dolma3_pool
  --transformed_by--> olmOCR`. Subject is always a model.
- **Every named benchmark in the release's eval tables** becomes
  an individual `used_for_evaluation` edge. Do NOT collapse the
  full benchmark list into a bundle name like `OlmoBaseEval`
  unless the bundle is the only thing the source mentions.
  Tables 2 / 12 of the OLMo-3 paper enumerate ~30 specific
  benchmarks (MMLU, MMLU Pro, BBH, GSM8K, MATH, HumanEval, MBPP,
  LBPP, DeepMind Math, AGIEval, HellaSwag, WinoGrande, ARC,
  TriviaQA, NaturalQuestions, ...) — emit one edge per
  benchmark. Same rule for in-loop dev evals.
- The team's own ablation tables become `used_for_ablation`
  indirect edges (only their own design-space variants — not
  external baselines they compared against).
- **Recipe / methodology references:** `inspired_by` edges
  point at MODELS or DATASETS only — never codebases, never
  software libraries. `OLMo-core` (codebase), `Resiliparse`
  (library), `Duplodocus` (dedup tool) are NOT valid objects.
  But `mlfoundations/dclm` (a released dataset) IS a valid
  `inspired_by` object when source says "we follow DCLM
  filtering" — emit `subject --inspired_by-->
  mlfoundations/dclm-baseline-1.0`.

If you only emit `trained_on` and `trained_from` edges with
no `generated_by` / `transformed_by` / `filtered_by` / eval
edges, you've under-covered.

## Subagent dispatch

Subagents run as `{{subagent_model}}`. Bucket the source files
into topical packets (one packet per training stage, per
dataset family, per eval section) and dispatch one subagent per
packet. Each subagent appends its events directly to
`{{artifact_path}}` as it processes.

When dispatching, transcribe verbatim into each subagent's brief:
- The "Append-as-you-go" instruction (file path + how to append).
- The "Subject is always a Model" rule (load-bearing — subagents
  default to dataset-subject framing without this).
- The anchor schema: EVERY `anchor_list` entry — event-level AND
  edge-level — is an object `{source, explanation, position?,
  excerpt?}`, never a bare string. Structurally invalid edges are
  quarantined out of the graph at validation.
- The "Aggregator + leaf rule" section (subject emits one edge
  per leaf in the aggregator's subsets[]).
- The "Lattice anchoring — preserve source specificity" section.
- The "Out-of-scope" list (baseline comparisons, tokenizers,
  frameworks, codebases — none are nodes).
- The relation taxonomy tables (DIRECT + INDIRECT).
- The edge schema (subject MUST be model; anchor_list MUST have
  excerpt for verifier grounding).

Subagents have none of your context. Without these
transcriptions they will silently revert to old patterns
(emitting baseline comparisons, missing leaf edges, fabricating
op-ids).

## Completion

When you've processed all sources and appended all events to
`{{artifact_path}}`, exit 0. Do NOT emit a final wrapping JSON
object — the file is JSONL, one event per line, and the
pipeline reads it as such.

You are running as `{{planner_model}}`. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
