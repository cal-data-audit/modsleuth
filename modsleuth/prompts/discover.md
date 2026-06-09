# Discover

> **Goal: fetch the target's own primary materials.** Names of
> upstream datasets, base models, eval benchmarks, classifiers,
> and baselines live INSIDE these materials and will surface in
> the extract stage. Do not chase cross-references; if you'd
> fetch a source because the paper *cites* it, that's extract's
> job.

## Inputs

- `{{input_path}}`: JSON with `target` and `workspace_dir`.
- Save fetched files into `{{workspace_dir}}` and record their
  relative paths in the artifact. Write the artifact to
  `{{artifact_path}}`. Nowhere else.

## Scope

**Fetch ONLY the target's primary materials, narrowly defined.**
Use HF and GitHub org APIs to enumerate the family-prefix matches:

- The target's tech report or paper (one PDF; HTML if no PDF).
- The target's official release blog post (one), if it exists.
- **The single primary code repo the release notes point at
  first** — e.g., for OLMo-3 that's `allenai/OLMo-core`. Use
  full `git clone`, NOT README-only. The recipes, configs,
  training launchers, and YAML files inside the repo carry
  `--dataset_mixer_list` flags, `DataMix.*` constants, and
  personal-namespace HF dataset references that never appear
  in the HF cards or paper. A `curl` of the README is a
  permanent lineage hole. Sub-repos (`allenai/open-instruct`,
  `allenai/dolma`, `allenai/olmo-cookbook`, etc.) are NOT in
  scope at discover — their names surface in extract from the
  primary repo / paper, and `expand` can recurse into them
  later if needed.
- **Same-generation HF model cards** in the family — match the
  target's exact generation prefix. For an `OLMo-3` target,
  match `^Olmo-3-` exactly. NOT `Olmo-3.1-*` (successor),
  NOT `Tulu-3-*` (predecessor product line),
  NOT `Olmo-2-*` (predecessor generation). Generation prefix
  match is the cut.
- **Same-generation HF dataset cards** under the target's org
  whose repo name shares the family prefix or is explicitly
  linked from the tech report.

Cross-generation, cross-family, and predecessor mentions are
the responsibility of `extract`, not `discover`. If a name
appears in the target's primary materials but the artifact
itself isn't a same-generation member of the family, extract
picks it up from prose; discover never proactively chases it.
This is what keeps the source set tight.

## How to fetch

> **Save raw response bytes.** Downstream extract reads the
> raw file; a summary IS a permanent lineage hole.

Use Bash with `curl` / `wget` / `git clone`. Do NOT use
WebFetch for files you save — it returns a summarized digest,
not the response body. WebFetch is acceptable only for
orientation (deciding whether a URL is in scope).

```
HF model card:        curl -sL https://huggingface.co/<repo>/raw/main/README.md
HF dataset card:      curl -sL https://huggingface.co/datasets/<repo>/raw/main/README.md
HF API (commit SHA):  curl -sL https://huggingface.co/api/models/<repo>/revision/main
HF org enumeration:   curl -sL https://huggingface.co/api/models?author=<org>
arXiv PDF (preferred): curl -sL https://arxiv.org/pdf/<id> -o paper.pdf
arXiv HTML (if no PDF): curl -sL https://arxiv.org/html/<id>v1 -o paper.html
GitHub README only:   curl -sL https://raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md
GitHub FULL repo:     git clone --depth 1 https://github.com/<owner>/<repo>
```

**HF auth:** if `HF_TOKEN` is set in the environment, add
`-H "Authorization: Bearer $HF_TOKEN"` to every
`huggingface.co` curl call (raw, API, everything). Without
auth, HuggingFace rate-limits unauthenticated traffic at
~30/min and will start returning rate-limit pages mid-fetch;
with auth the ceiling jumps ~30× and gated repos you have
access to start resolving. Skip the header for arXiv / GitHub
/ other non-HF hosts.

**For the target's own primary code repo, you MUST use
`git clone --depth 1`, not `curl` of the README.** A
README-only fetch is a permanent lineage hole: the actual
training scripts, mixture YAMLs, launcher flags, and config
constants live in the repo tree, not in the README. A
README-only fetch routinely loses 1000+ name mentions for an
OLMo-class target. The `curl` fallback is only acceptable for
peripheral / third-party repos cited in passing — never for
the target's primary release repo.

For papers / tech reports, fetch ONE format: PDF when
available, HTML when no PDF exists. Same content; don't fetch
both.

Pin commit_sha for HF and GitHub sources where possible. Use
lowercase, hyphenated filenames matching the artifact id
(`olmo-3-7b-base.md`, not `OLMo-3-7B-Base.md`).

## Verify what you fetched

Read the first ~50 lines of each captured file:

1. It's real content, not a JS shell, a 401/404 page, or a
   "Sign in to continue" stub.
2. The target's name appears multiple times and the source is
   *about* the target.
3. Markdown / HTML / blog files under 1 KB are almost
   certainly stripped — re-fetch with an explicit `-A` user-agent.
   (PDFs are typically >1 MB; this size threshold doesn't apply
   to them.)

## Batches

A batch is the unit of downstream extract work — a topically
coherent set a careful reader treats as one pass.

1. Base + SFT/DPO/Instruct for one size go in one batch (stage
   chains stay together).
2. Pretraining corpora, SFT mixes, DPO data are distinct
   families and split into separate batches.
3. The tech report is its own batch (the paper plus optional
   release blog).
4. Big sources go in exactly one batch. Small sources may be
   reused across batches when they belong with two families.

## Output

```json
{
  "batches": [
    {
      "label": "...",
      "summary": "why these sources belong together",
      "sources": [
        {"path": "relative/to/workspace.md", "url": "https://...", "commit_sha": null}
      ]
    }
  ]
}
```

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An empty
`batches[]` usually signals a misread or fetch failure — surface
the failure rather than emitting nothing.

You are running as `{{planner_model}}`. Use subagents for
independent fetch packets. Subagents run as `{{subagent_model}}`.

{{subagent_prompt}}
