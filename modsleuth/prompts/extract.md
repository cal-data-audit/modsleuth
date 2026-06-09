# Extract Names

> **Goal: list every model and dataset this batch mentions, in
> the most specific form the source uses.** Output one line per
> mention, with `type` ('model' or 'dataset') and `name`.
> Nothing else.

Read `{{batch_dir}}` and write the artifact to
`{{artifact_path}}`.

## Inputs

- `{{input_path}}`: JSON with `batch_id` and `batch_dir`.
- `{{batch_dir}}/MANIFEST.txt`: tab-separated filename, source
  id, title — for orientation only; you do not cite it.

## Filesystem scope

Read `{{batch_dir}}` and `{{input_path}}`. Write
`{{artifact_path}}`. Do not read or write any other local path.
No web browsing. No HF API calls. No classification beyond
choosing `model` vs `dataset`. No fuzzy matching, normalization,
or deduplication beyond removing literal duplicates of the same
(type, name) pair within this batch.

## Output schema

```json
{
  "mentions": [
    {"type": "model",   "name": "Qwen/Qwen3-7B-Instruct"},
    {"type": "model",   "name": "Qwen3-7B-Instruct-FP8"},
    {"type": "dataset", "name": "HuggingFaceTB/finemath"},
    {"type": "dataset", "name": "finemath-3plus"},
    {"type": "model",   "name": "OLMo-3-1025-7B-Base"},
    {"type": "dataset", "name": "MMLU-Pro"}
  ]
}
```

The artifact has exactly one key, `mentions`, holding a list of
records. Each record has exactly two fields: `type` (`"model"`
or `"dataset"`) and `name` (the verbatim name string from the
source). Do NOT emit any other field — no kind aliases, no
identity, no atoms, no anchors, no links, no descriptions, no
file references, no excerpts, no aliases, no aux, no
descriptors, no concept_path. Adding extra fields will be
ignored, but bloats the artifact.

## What counts

A noun-phrase the source uses to refer to a specific model or
dataset. Examples (each line is one valid name):

```
Qwen2.5-7B-Instruct
Qwen/Qwen3-4B
Llama 3.1
OLMo-3-1025-7B-Base
allenai/dolma3-fasttext-quality-classifier
HuggingFaceTB/finemath
finemath-3plus
FineMath-3+
gpt-4o-mini-2024-07-18
MMLU-Pro
AIME 2024
DeepSeek-R1
Qwen3-7B-Instruct-FP8
```

### Neural artifacts framed as tools

Some sources name a neural model with tool-style phrasing
("using olmOCR (Poznanski et al., 2025a,b)", "we judge with
GPT-4"). The model is still a model. **Emit it as `model`** when
the surface names a *neural artifact that processes data*
(generates, transforms, filters, judges, embeds), even when the
prose frames it as a tool. Cues to look for:

- it appears alongside a paper citation (`(Author et al., 20XX)`)
- it is named after a model card slug (`allenai/olmOCR-7B-0225`,
  `Qwen/Qwen3-32B`)
- it is described as performing a learned function — OCR,
  classification, judging, rewriting, embedding, ranking

When in doubt, emit. The organize / audit stages drop noise via
web verification; missing a real model is irrecoverable.

### Looks like a dataset in context

Some sources name a data source with a non-dataset noun
("forum", "website", "archive", "repository", "dump",
"submissions", "competition", "subreddit"). When such a name is
the SOURCE of training content the target consumed — rewrites,
scrapes, filtered subsets — **emit it as `dataset`**, not as a
generic mention to skip. The literal noun is misleading; the
artifact role is "dataset that fed training". Organize will
resolve it to its canonical HF / paper release if one exists.

Cues:

- The name appears as the origin / source of content in a
  data-construction sentence ("sourced rewrites from X",
  "scraped from X", "filtered from X dumps", "experimented
  with rewrites of X").
- The name is well-known as a problem or document corpus
  with public releases (math problem forums, web archives,
  encyclopedic projects, scholarly venues, code repositories,
  question-answer communities).
- A paper citation follows the name, attributing the dataset
  paper rather than the forum / website itself.

This rule extends "when in doubt, emit" — but with a specific
disambiguation: if the name is the source of training data,
choose `dataset`, not skip-as-task / skip-as-org / skip-as-tool.

### Skip these (they aren't model/dataset names)

- license names (`Apache-2.0`, `MIT`, `CC-BY-4.0`)
- pure-software packages, frameworks, tokenizers as such
  (`transformers`, `vLLM`, `datatrove`, `tiktoken`,
  `pyarrow`) — non-neural libraries only. Neural artifacts
  framed as tools (above) are NOT skipped.
- author/organization names by themselves (`Anthropic`,
  `Meta`, `Allen Institute for AI`)
- paper titles
- task / capability names that are NOT a dataset
  (e.g., "math reasoning" the task vs. `MATH-500` the dataset)

### Where artifact names appear (code-file principle)

When reading code files, extract artifact names from places
where the source REFERENCES an artifact — not from every
identifier you see. The structural signals are:

- Quoted strings passed to model / dataset loaders:
  `from_pretrained("Qwen/Qwen3-32B")`,
  `load_dataset("HuggingFaceTB/finemath", "finemath-3plus")`
- CLI flags / launcher arguments pinning a model or dataset:
  `--model Qwen/Qwen3-32B`,
  `--dataset_mixer_list allenai/Dolci-Think-RL-7B 10000`
- Config-file fields naming an artifact:
  `base_model: meta-llama/Llama-3.1-8B`,
  `datasets: [tatsu-lab/alpaca]`
- Comments naming the artifact: `# distill from DeepSeek-R1`

Function definitions, class definitions, classmethod names,
factory constructors, and internal variable names are NOT
artifact references. They DEFINE or BUILD something; they are
not the thing itself. Skip them.

When in doubt, ask: is this surface string REFERRING TO an
existing released artifact, or DEFINING / CONSTRUCTING
something? References are in scope; definitions are not.

### Skip bibliography-only references

If a name appears ONLY in the paper's References / Bibliography
section and is not mentioned by name in body prose, tables,
captions, or code blocks, do NOT emit. A bibliography entry is
a scholarly citation, not an artifact mention. A name that ALSO
appears in body prose with a participation claim ("we used X as
the base", "X was the judge") IS in scope — the bibliography
rule applies only when bibliography is the sole occurrence.

**Important: a body-prose mention with a paper citation is NOT
bibliography-only.** Many primary sources cite the upstream
artifact's paper inline as `(Author et al., 20XX)` while
describing what the artifact did. That's body prose with a
participation claim — emit the artifact name even though a
citation accompanies it. The bibliography-only rule applies
only when the name is genuinely confined to the References /
Bibliography section with NO body-prose discussion.

The test: ignore the citation parenthetical. Does the rest of
the sentence say what the artifact did, where it came from, or
how it was used? If yes, it's body prose. Emit.

### Skip comparison-baseline-only mentions

If a name appears ONLY as a row in a leaderboard or evaluation
comparison table with no prose mention elsewhere in the source,
do NOT emit. A bare table row is just a number; the
participation claim that makes a comparison interesting ("we
benchmarked OUR model against X") lives in prose. A name that
appears in BOTH the comparison table AND prose with any role
context (judge, distillation source, methodology peer,
explicit comparison subject) IS in scope.

These two skips key on source position (where in the document a
name appears), not on entity type or relation type. Indirect
dependencies — judges, methodology references,
comparisons-with-context — survive because they appear in
prose. Pure noise (citation lists, bare leaderboard scores) is
filtered.

## "Most specific form" means

Match what the source actually wrote. Don't generalize, don't
expand.

- Source says `Qwen2.5-7B-Instruct` → emit `Qwen2.5-7B-Instruct`.
  Don't generalize to `Qwen2.5`.
- Source says `the Qwen3 family of models` → emit `Qwen3`. The
  source said `Qwen3`. Don't invent a size.
- Source says `from_pretrained("Qwen/Qwen3-4B")` → emit
  `Qwen/Qwen3-4B` exactly (with the org prefix, because the
  source has it).
- Source says `Tülu 3` (with umlaut) → emit `Tülu 3`. Keep the
  original characters; canonicalization happens later.
- A surface like `Qwen3-7B-Instruct-FP8` → emit it as is. Do
  not collapse to its canonical.
- Source mentions a dataset config: `load_dataset("HuggingFaceTB/finemath", "finemath-3plus")` →
  emit two records: `{"type": "dataset", "name": "HuggingFaceTB/finemath"}`
  AND `{"type": "dataset", "name": "finemath-3plus"}`.

## Within-batch dedup

If the source mentions the same `(type, name)` pair multiple
times, emit it once. Don't emit a (type, name) pair twice.
Do NOT collapse name variants — `Qwen3-7B`, `Qwen 3 7B`, and
`qwen3-7b` are three separate records here. The organize stage
handles fuzzy matching across name variants.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An empty
`mentions[]` list is legal only if the batch genuinely contains
no model or dataset names.

You are running as `{{planner_model}}`. Use subagents for
independent source packets within the batch. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
