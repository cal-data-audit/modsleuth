# Evaluation

This directory holds every reproducer for the paper's quantitative
results: Table 1 (pooled eval), Table 6 (full-graph audit), and the
graph-statistics tables (2 / 4 / 5).

| Script | Reproduces | Cost |
|---|---|---|
| `pooled_eval.py` | Table 1 (pooled LLM-as-judge across 6 systems × 4 targets) | Free if you reuse `outputs/verifications.jsonl`; ~$$ for fresh clusters |
| `build_modsleuth_inputs.py` | The two ModSleuth attribution outputs (`prov_*.json`, `prov_unbounded_*.json`) consumed by `pooled_eval.py`, computed from the merged graph using paper §B's depth-1 / unbounded rules | Free, deterministic |
| `full_graph_audit.py` | Table 6 (per-edge audit of the full 14,769-edge merged graph) | Free if you reuse `outputs/full_graph_verifications.jsonl`; ~$$$ for a fresh run |
| `compute_graph_stats.py` | Tables 2, 4, and 5 (audit-role split, per-target ancestry/depth, source-type distribution) | Free, deterministic |

The merged graph itself ships at `../data/merge_artifact.json` via
git-lfs; run `git lfs pull` after cloning.

## `pooled_eval.py` — Table 1

For each target model, every emitted edge across all systems is pooled
and clustered by canonicalized `(subject, object)` pair. Each cluster's
representative claim (longest description) is sent to a single
`claude-sonnet-4-6` verifier instance equipped with `web_search` (max 6
queries per call), which returns one of:

- `verified` — the relationship is real (cited evidence supports it,
  or the verifier independently corroborated it via web search)
- `refuted` — neither the cited evidence nor independent search
  supports the claim, or the verifier found direct contradiction
- `unclear` — genuinely ambiguous after honest search

A single verifier verdict cleanly attributes back to every system that
proposed an edge in that cluster, so per-system Verified / Refuted
counts mean: how many clusters did this system contribute to, broken
down by verdict.

The verifier prompt is at `verifier_prompt.md`.

### Reproducing the reported numbers

```bash
pip install -r ../requirements.txt
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py
```

`DEFAULT_SYSTEMS = ["gpt55pro", "gpt54pro", "cc", "o3dr", "prov", "prov_unbounded"]`
— all six rows in paper Table 1 — are pooled across the four targets.

The pre-computed verdicts in `outputs/verifications.jsonl` are the
exact ones used in the paper. With them in place, the run is just an
aggregation pass and finishes in seconds. To rerun from scratch,
delete `outputs/verifications.jsonl` first; expect hours and a
non-trivial Anthropic bill (the pool covers ~4,672 unique clusters).

The internal slugs map to paper labels via `SLUG_TO_LABEL` in
`pooled_eval.py`:

| Slug | Paper label |
|---|---|
| `gpt55pro` | GPT-5.5 Pro |
| `gpt54pro` | GPT-5.4 Pro |
| `cc` | CC-single |
| `o3dr` | ChatGPT Deep Research |
| `prov` | ModSleuth (depth-1) |
| `prov_unbounded` | ModSleuth (unbounded) |

### Evaluating a new submission

Drop your per-subject JSONs into `../baselines/outputs/` named
`<slug>_<subject>.json`, then:

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py \
    --systems gpt55pro,gpt54pro,cc,o3dr,prov,prov_unbounded,mysystem
```

Only clusters your system contributes that aren't already in
`outputs/verifications.jsonl` get fresh verifier calls; existing
verdicts attribute automatically once your slug shows up in the same
cluster.

## `build_modsleuth_inputs.py` — ModSleuth attribution outputs

The two ModSleuth rows in Table 1 are derived from the merged graph
under the depth-1 and unbounded attribution scopes defined in paper §B.
Re-derive `prov_<target>.json` + `prov_unbounded_<target>.json` from a
fresh merge:

```bash
python3 build_modsleuth_inputs.py \
    --merge-artifact ../data/merge_artifact.json \
    --out-dir ../baselines/outputs
```

The script is deterministic and reproduces exactly 4,563 raw edges →
4,236 unique `(target, canonical_subject, canonical_object)` clusters
(matching paper §B verbatim).

## `full_graph_audit.py` — Table 6

Same `claude-sonnet-4-6` + `web_search` verifier as `pooled_eval.py`,
but one verdict per edge over the full 14,769-edge graph instead of
per cluster:

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 full_graph_audit.py \
    --merge-artifact ../data/merge_artifact.json \
    --out outputs/full_graph_verifications.jsonl
```

Resumable, append-only. The committed
`outputs/full_graph_verifications.jsonl` (14,769 lines) and
`outputs/full_graph_verifications.score.json` are the exact records
behind Table 6 — 14,110 verified, 424 refuted, 235 unclear, precision
0.9708.

## `compute_graph_stats.py` — Tables 2 / 4 / 5

```bash
python3 compute_graph_stats.py --merge-artifact ../data/merge_artifact.json
```

Pure computation over the merged graph; every row of Tables 2, 4, and
5 reproduces exactly. See the script's docstring for the audit-role
mapping (Table 2), per-target seed list (Table 4), and source-type
classifier (Table 5).

## Outputs

| File | Produced by | Purpose |
|---|---|---|
| `outputs/verifications.jsonl` | `pooled_eval.py` | one record per verified cluster (`cluster_key`, `target`, canonical `subject`/`object`, contributing `systems`, `verdict`, `confidence`, `explanation`) |
| `outputs/score.json` | `pooled_eval.py` | aggregate verified/refuted/unclear/error counts per system + precision over decisive verdicts |
| `outputs/score_per_target.json` | `pooled_eval.py` | same breakdown, per target |
| `outputs/table.txt` / `outputs/table.latex` | `pooled_eval.py` | human-readable + LaTeX renderings of the aggregate table |
| `outputs/full_graph_verifications.jsonl` | `full_graph_audit.py` | one verdict per edge in the merged graph (14,769 lines) |
| `outputs/full_graph_verifications.score.json` | `full_graph_audit.py` | aggregate counts + precision (Table 6) |

## Schema

Each input edge needs at minimum:

```json
{
  "subject": "<canonical_id>",
  "object":  "<canonical_id>",
  "description": "<lossless prose>",
  "evidence": [
    {"source": "<URL>", "location": "<locator>",
     "excerpt": "<verbatim quote>", "explanation": "<how it supports>"}
  ]
}
```

`relation_type` is accepted but not graded — the verifier judges the
underlying relationship, not the bucket label. See
`../baselines/prompts/baseline_prompt.md` for the full schema and
admission criteria.
