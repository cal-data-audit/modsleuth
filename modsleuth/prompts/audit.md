# Audit and Revise

> **Goal: read the whole lattice, find what's weird, fix it.** You're
> a senior reviewer doing a second pass over organize's work. There
> is no fixed list of issues to look for — survey the structure,
> identify cases that don't make sense, and decide on appropriate
> edits. Same shape in, same shape out.

Read `{{organize_path}}` and write the revised artifact to
`{{artifact_path}}`.

## Filesystem scope

Read `{{organize_path}}` and `{{input_path}}` (same file). You
also have access to **the original source files** at
`{{batches_dir}}` — every batch's paper PDFs, model cards,
GitHub repos, configs, and other sources discover fetched are
materialized there as `{{batches_dir}}/<batch_id>/<filename>`.
Use them: re-reading the original sources is the only way to
catch over-specification (where the lattice's leaf is more
specific than what the source mention actually said) or
under-specification (where a leaf alias is too vague).

Write `{{artifact_path}}`. Web search, WebFetch, and
HF / GitHub URL lookups are permitted; you'll use them more
than organize did because the recheck pile demands fresh
investigation.

**HF auth:** if `HF_TOKEN` is set in the environment, add
`-H "Authorization: Bearer $HF_TOKEN"` to every
`huggingface.co` curl call (raw, API, everything). Raises the
unauthenticated rate limit (~30/min) by ~30× and unlocks
gated repos you have access to. Skip the header for non-HF
hosts.

## What runs before you (your input is pre-processed)

A pure-Python pass (`modsleuth.subsets.populate_then_flag`) runs BEFORE
you. It is **purely additive** — never moves items, never
restores drops, never renames anything. It does two things:

1. **Populates `subsets[]` on every dataset node** by fetching
   the HF README and parsing its YAML `configs:` field plus
   markdown composition / components / mix tables. Each subset
   slug is lowercase-kebab.

2. **Surfaces suspicious cases as a top-level `audit_hints[]`
   array.** Each hint flags a case that likely warrants your
   attention but doesn't prescribe an action. Hint kinds:

   - `item_matches_parent_subset` — an existing item's name slug
     appears in some other kept item's `subsets[]`. The
     `item_role` field labels what kind of item it is
     (`family-root` / `canonical` / `concept` / `unanchored`).
     Typical action: reshape sub-components under the parent
     when they have no own canonical anchor; keep
     foundational concepts standalone.
   - `dropped_matches_parent_subset` — a dropped name's slug
     appears in some kept item's `subsets[]`. Typical action:
     either leave dropped (the subset is captured in the parent's
     `subsets[]` field) or restore as a leaf if the dropped
     name has its own canonical release.
   - `sibling_identity_collision` — two items in the same family
     carry identical identity dicts. You MUST resolve.
   - `cross_org_family` — a family spans multiple `identity.org`
     / `identity.vendor` values. Your judgment: substring false
     positive (split) or legitimate product-line grouping (keep,
     possibly add a discriminating facet).
   - `formal_name_vs_canonical_url_mismatch` — the formal_name
     doesn't match the canonical path inside its primary HF URL.
     **Prescriptive: rename to the canonical HF path (lowercase
     `<owner>/<repo>` from the URL); move the old form to
     `aliases[]`.** Skip ONLY when the source explicitly uses the
     formal_name and not the canonical path.
   - `phantom_item` — empty aliases. The validator already
     rejects this; if the hint fires, fix or drop.
   - `missing_family_root` — a group has 2+ items but no family
     root (item with identity `{family: X}` only). You MUST
     synthesize one. Use the bare family name as the formal_name
     and as an alias. Add `paper` / `blog` / `hf_collection` link
     if the family has one; otherwise empty `links: []`. No
     production link.
   - `over_specified` — a leaf carries a bare family-name alias
     (e.g., `"olmOCR"`) but its formal_name pins specific facets
     (`allenai/olmOCR-7B-0225-preview`). **Re-read the source
     at `{{batches_dir}}` to verify.** If the source mention is
     genuinely bare ("we use olmOCR" with no version), move the
     bare alias to the family root and keep release-specific
     aliases on the leaf. Vague mentions should not silently bind
     to an arbitrary specific release.
   - `branch_variant_in_formal_name` — `<repo>@<branch>` HF
     git-revspec syntax. Collapse all branch variants into one
     leaf with the canonical repo formal_name; carry branch names
     in aliases.
   - `same_url_duplicate` — two items in the same family share
     the same primary URL. **Prescriptive: merge into one item.**
     Pick the keeper as: (a) the multi-facet entity over the
     concept root, (b) else the item with more aliases, (c) else
     the first lexicographically. Move all surface forms from
     the merged-out items to the keeper's `aliases[]`. Runtime-
     mode differences (`thinking` / `no-thinking` chat templates,
     sampling hyperparameters) belong on edges, NOT as facets.
   - `concept_subsumed_candidate` — within a family, item A's
     identity facets are a strict subset of sibling B's, and A
     has no item-unique anchor (no `hf_model` / `hf_dataset` /
     `vendor_docs`). A is likely a concept (a partial spec
     subsumed by B). Confirm `kind` matches the family's nature
     and that A's `links[]` hold only family-shared anchors
     (`paper`, `hf_collection`, `blog`) or are empty. **Don't
     drop A** — vague source mentions need somewhere to land.
   - `subset_with_anchor` — within a family, item A's facets
     are a strict subset of sibling B's, and BOTH have unique
     anchors. Dataset-config / subset-of relationship (e.g.,
     `HuggingFaceTB/finemath` ⊃ `infimm-webmath/infiwebmath-3+`).
     Both stay as entities; ensure both descriptions note the
     relationship so relate can emit a `subset_of` edge.
   - `same_url_cross_family` — same primary URL appears in items
     from DIFFERENT families. **Prescriptive: pick the family
     whose name matches the URL's namespace owner** (e.g.,
     `allenai/dolmino-mix-1124` → family `Dolmino`, not
     `OLMo 2`). Remove the duplicate items from other families
     entirely. Relate captures the cross-family usage as edges.
   - `concept_with_no_entity` — a family has multiple concept
     items but no entity. **Prescriptive: do an HF org enumeration
     before accepting the gap.** Run, in order:
     ```
     curl -sL "https://huggingface.co/api/models?author=<org>&search=<family>&limit=50"
     curl -sL "https://huggingface.co/api/datasets?author=<org>&search=<family>&limit=50"
     ```
     Likely creator orgs by family pattern: `allenai/` for OLMo /
     Dolma / olmOCR; `Qwen/` for Qwen / QwQ; `meta-llama/` for
     Llama; `mistralai/` for Mistral; `nvidia/` for Nemotron;
     `openai/` for GPT; `microsoft/` for Phi; `google/` for Gemma;
     check the Python configs in `{{batches_dir}}` for `from_pretrained`
     calls if no creator is obvious. If a result HEAD-checks 200
     and matches the family pattern, restore as an entity. Only
     accept the gap if the enumeration returns nothing relevant.
   - `family_root_invented_alias` — family root's aliases don't
     trace to any input-pile surface form. Vague relate mentions
     may fail to resolve. **Prescriptive when formal_name has
     parens-disambig form (`Phi (Microsoft)`, `GPT (OpenAI)`,
     `Falcon (TII)`):** if NO other family in the output has the
     same bare formal_name (`Phi`, `GPT`, `Falcon`), rename the
     root's `formal_name` to the bare form and move the parens
     form to `aliases[]`. **Also:** scan the input names pile
     for the family substring (case-insensitive) and add every
     match to the root's `aliases[]`.

   The hints are **suggestions, not commands** (except
   `missing_family_root`, which is mandatory — every family MUST
   have a root). A hint exists because Python found a pattern;
   you decide what's right given the broader context.

So when you read the input artifact: every dataset has populated
`subsets[]`, the lattice structure is exactly as organize left
it (no items moved or restored), and `audit_hints[]` lists what
Python found suspicious. **Your edits are the only changes that
happen to the lattice.**

## Synthesized concepts you'll see in the lattice

`modsleuth.subsets.expand_concept_lattice` ran in the pre-pass: every
interior concept implied by the leaves' facets has been
materialized as an item with `_generated: true` and aliases
auto-derived from the natural concept label (e.g., `OLMo 3 7B`
from facets `{family: OLMo 3, size: 7B}`). These nodes serve as
anchor points for relate when source specificity falls between
root and a leaf.

How to handle them:

- **If a `_generated` concept's natural alias matches a source
  mention** (e.g., the input pile contains `Apertus 8B` and a
  generated concept has `aliases: ["Apertus 8B"]`), find the
  leaf where the planner put that source alias, **move the
  alias from the leaf to the generated concept**, and **clear
  `_generated: true`**. The concept is now a source-mentioned
  node that the planner missed.
- **If a `_generated` concept seems redundant** (e.g., its
  facets are degenerate within this family), drop it.
- **Otherwise leave generated concepts alone.** They cost
  nothing and let relate land vague mentions.

A second `expand_concept_lattice` runs AFTER you complete to
catch concepts you may have left orphaned. Idempotent.

## How to think about this pass

You are a careful reviewer. The lattice in front of you was
produced by a planner working under time pressure across many
buckets in parallel. It will have inconsistencies. Your job is
to spot them and fix them. There is no exhaustive checklist
because the failure modes are open-ended; instead, hold these
questions in mind as you walk the lattice.

### The one universal rule: verify before acting

Before you rename or drop any item — including items whose names
look "weird," items whose URLs look unfamiliar, items that look
"invented" — **HEAD-check the URL the item points at.** If it
returns 200 and the page describes the named artifact, the item
is real, no matter how unusual the name or URL looks.

```
curl -sL -o /dev/null -w '%{http_code}' <url>
```

A real artifact's name and anchor do not have to look like what
you'd expect. Suffixes the planner thinks are "synthetic" are
sometimes the actual canonical names the artifact's authors
chose. Anchors that aren't HF or GitHub are sometimes the only
canonical home an artifact has (a foundational data project on
its own .org domain; a creator's HF org page or HF buckets
page; a benchmark released as a paper with no separate repo).
Pattern-matching on name shape or URL kind without checking
ground truth is the most expensive mistake you can make in this
pass — it deletes real items that already cost organize money
to find.

### Questions to walk the lattice with

- **Does each item correspond to exactly one real artifact?**
  An item is fine if (a) its primary URL HEAD-checks 200 AND
  (b) the page at that URL describes the named artifact. An
  item is broken when either fails: rename to its canonical
  identifier (carry the old name as alias), or drop with a
  recorded reason, or promote to `gated[]` if the URL is
  401/403, or split if the item absorbs two distinct artifacts.

  Valid anchors include all of these — none are second-class:
  - HF model / dataset / collection / org / buckets pages
  - GitHub org / repo
  - Paper (arXiv, ACL, venue page)
  - Vendor docs (OpenAI / Anthropic / Google API model pages)
  - Official project homepage on its own `.org` / `.com`
    domain (foundational data resources like web crawls,
    archives, encyclopedias, and forums often use this)
  - Release blog post

- **Are family groupings cohesive?** A family should hold items
  that share a real product line. Lineage cohesion matters
  more than substring matching — items related by release
  history (an extension / variant / converted form of a base
  artifact) belong together; items that only share a name
  fragment do not. The auditor's judgment here is what
  matters; document each split / merge in `notes`.

- **Within each family, do siblings actually differ?** If two
  items in the same family carry identical `identity` dicts,
  the lattice can't tell them apart. Either they're the same
  artifact (merge) or they need a discriminating facet
  (extend `identity_keys`).

- **Do `formal_name`s match the artifact's canonical anchor?**
  After verifying the URL is real, check that the formal_name
  matches the canonical identifier of the resolved page: HF
  artifacts use the lowercase HF `<owner>/<repo>` path that
  appears in the URL; closed-source models use
  `<vendor>/<slug>` from the vendor docs URL; family-concept
  roots use a clean family name (no `(collection)` /
  `(legacy family)` parentheticals — the cleaner form is just
  `Qwen3` / `Qwen3-Coder`). Rename when there's a clear
  mismatch (e.g., a fabricated slug whose URL doesn't 200).
  **Never rename based on suffix shape alone — only after
  proving the original 404s.**

- **Don't synthesize items to absorb scaling-config or
  factory-function aliases.** When you see dropped[] entries
  like `olmo2_14M`, `llama_like`, `gemma3_27B` — Python factory
  functions or scaling-experiment configs from a primary code
  repo — they are CODE, not artifacts. Don't create a synthetic
  family-concept item like `meta-llama/Llama-2-architecture` or
  `allenai/OLMo-2-architecture` to absorb them. The
  architecture relationship those configs reference (e.g.,
  "OLMo 2 uses a Llama-style architecture") is captured at
  relate stage as an `inspired_by` edge between two real model
  nodes, NOT as a separate node here. Let those drops stay
  dropped.

- **Does the lattice over-decompose?** Surface variants of one
  artifact should be one item with multiple aliases.
  Eval-harness reformulations of one benchmark should fold
  into the canonical with the harness as a facet. Date-snapshot
  variants of an API model should fold into the family
  canonical with the date as a facet.

- **Does the dropped pile contain false negatives?** Some
  dropped items will look like real artifacts the organize
  planner missed. Independently verify before accepting any
  drop reason that names an unrelated project, different
  community, or library namespace — those reasons are common
  cover for the planner not having found the right anchor.

- **Is every family root materialized?** Every group MUST have
  exactly one item with identity `{family: X}` only — the
  lattice top. Vague mentions like "Qwen 3" or "OLMo 3 Base"
  land here. If a `missing_family_root` hint fires, synthesize
  the root with the bare family name as alias and a paper /
  blog / hf_collection link if any (no production link).

- **Is anything over-specified?** When an `over_specified` hint
  fires, the planner glued a bare family-name alias (`"olmOCR"`)
  onto a specific HF leaf (`allenai/olmOCR-7B-0225-preview`).
  **Re-read the source files at `{{batches_dir}}` to verify.**
  Search the source for the alias string. If the source mentions
  the artifact ONLY at the bare family level (no version pinned),
  move the alias to the family root and keep only release-
  specific aliases on the leaf. Vague mentions should never
  silently bind to an arbitrary specific release; that's the
  whole point of the family root concept.

- **Are there items with `description: null` that have a
  fetchable source?** Organize sometimes leaves a description
  empty when its first attempt failed (e.g., raw README returned
  401/403 because the model is gated, planner ran out of budget
  for that bucket, or transient network error). Try again — the
  source ladder is the same as organize's:
  1. `curl` the raw README at `<page-url>/raw/main/README.md`;
  2. **WebFetch the page URL** when raw returns 401/403
     (rendered page exposes the card text even for gated repos
     like the Meta Llama family);
  3. HF API (`huggingface.co/api/models/<owner>/<repo>`) for
     `pipeline_tag` / `library_name` / `tags` / `cardData`;
  4. GitHub README first paragraph (if no HF card);
  5. arXiv abstract first sentence (if paper-anchored);
  6. WebSearch on `"<formal_name>"` for a release blog,
     vendor docs, or HF blog post.

  Same writing rules: target-independent, third-person, ≤3
  sentences. Leave `description: null` only after the full
  ladder fails.

These questions overlap. Walk the artifact looking through all
of them; many edits will address several at once. The universal
rule (verify before acting) trumps any pattern-match.

## Edits available to you

Anything that produces a cleaner lattice in the same schema:

- **Rename** an item's `formal_name` to its canonical form when
  the current name is fabricated, mistyped, or doesn't follow
  the convention for its kind. Carry the old form into
  `aliases`.
- **Split** an item that absorbs two distinct artifacts
  (different release events, different weights, different
  papers).
- **Merge** two items that are really the same artifact under
  different surface forms.
- **Move** an item from one family to another when it was
  bucketed by substring rather than by lineage.
- **Split a family** that spans multiple orgs / artifact kinds /
  product lines. Document each split briefly in `notes`.
- **Merge two families** that turn out to refer to the same
  product line.
- **Add a discriminating facet** when siblings carry identical
  identity. Extend the family's `identity_keys` if needed.
- **Drop an item** that shouldn't exist (invented placeholder,
  unverifiable name, item the lattice has no canonical anchor
  for).
- **Restore a dropped name** as a new item if you find a real
  anchor matching the 3-form criterion (open HF/GitHub release;
  vendor docs page; paper anchor).
- **Promote a dropped name to `gated[]`** if you confirm the
  artifact exists at a 401/403 HF URL.
- **Augment a dropped reason** with `[audit-confirmed]` plus
  the evidence you collected, when recheck confirms no anchor
  exists.
- **Fill a missing description** when an item has
  `description: null` AND a fetchable source exists. Apply the
  source ladder above (raw README → page WebFetch → HF API →
  GitHub README → arXiv abstract → WebSearch). Keep the same
  rules organize uses: target-independent, third-person, ≤3
  sentences, no fabrication. Skip silently when the ladder
  exhausts (description stays null).

When you make any non-trivial edit, briefly note it in the
top-level `notes` field — what changed and why. The notes are
what an operator scans to understand what the audit pass did.

## Sweeps to run before writing the artifact

Two deterministic sweeps to apply at the end of your pass — they
catch regressions audit's narrative attention may have missed.

### 1. Umbrella-with-subset-facet sweep

For each item where `links[0]` is `hf_dataset` or `hf_model`:
- If the URL path is `huggingface.co/datasets/<owner>/<repo>` or
  `huggingface.co/<owner>/<repo>` with NO `/viewer/<config>` and
  NO `?config=` parameter (i.e., the umbrella page), AND
- The item's `identity` carries a `subset` facet,
- **Drop the `subset` facet.** The umbrella isn't the slice; the
  parent dataset's `subsets[]` field captures the configs.
  Example: `LLM360/MegaMath` should have `identity={family:
  MegaMath}`, not `identity={family: MegaMath, subset: 'web'}`.

### 2. Description completion sweep

Walk every item with `description: null` (excluding items
flagged `_generated: true` — those are Python-derived concepts
and may legitimately be null). For each, apply the source ladder:

1. `curl <page-url>/raw/main/README.md` (HF raw README)
2. WebFetch the page URL (renders gated cards)
3. HF API `huggingface.co/api/<models|datasets>/<owner>/<repo>`
4. GitHub README first paragraph
5. arXiv abstract first sentence
6. WebSearch on `"<formal_name>"` for blog / vendor docs

Cap at 30 items per sweep (the highest-priority ones: family
roots first, then entity leaves with most aliases). Leave the
remainder null. Skip silently per item only after the ladder
fully exhausts on that specific item.

### 3. End-of-pass invariant check

Before writing the artifact, scan for these regressions:

- **No within-family same-URL duplicates.** For each family,
  for each unique-anchor URL (`hf_model` / `hf_dataset` /
  `vendor_docs`), at most one item carries it as `links[0]`.
- **Every input name still traceable.** Read the names pile at
  `{{input_path}}` (or scan the original organize input). For
  each name, verify it appears in some item's `aliases` /
  `formal_name`, in `dropped[]`, or in `gated[]`.
- **Every family has exactly one root** (identity == {family}).

If any invariant fails, fix it before writing. Note in `notes`.

## When uncertain

Dispatch a sonnet subagent to investigate one specific case
(e.g., "is `<name>` a real release? what's its canonical URL?
what role does it play?"). Subagents have none of your
context — transcribe the relevant rule fragments verbatim.
Cap total dispatched investigations at ~10 per pass — this is
a check-and-fix pass, not a full re-organize. For broader
sweeps (e.g., "rename every bare-benchmark name to its HF
canonical path"), apply the rule yourself in batch using
`Bash` rather than dispatching one subagent per item.

## Recheck investigations — what to actually do

For each `dropped[]` entry you decide to RECHECK:

1. If the entry has `attempted_canonicals[]`, do NOT re-try
   those exact paths — organize already verified they don't
   exist. Search adjacent variants instead.

2. Standard search patterns:
   ```
   curl -sL https://huggingface.co/api/models?author=<org>
   curl -sL https://huggingface.co/api/datasets?author=<org>
   curl -sL 'https://huggingface.co/api/models?search=<name>&limit=20'
   curl -sL 'https://huggingface.co/api/datasets?search=<name>&limit=20'
   ```

3. If the dropped name might be an internal codename
   referenced in a primary repo's README, grep the
   already-fetched primary repo for the slug.

4. For API-only models, check the vendor's docs page directly.

5. After HEAD-200 on any candidate, fetch the card and verify
   the content matches the input context — same
   "verify-before-adopting" discipline organize was supposed
   to apply.

## Output schema

Same shape as the organize artifact, plus an OPTIONAL top-level
`gated[]` array. Required fields unchanged from organize:
`groups[]` with `family`, `identity_keys`, `items[]`; each item
with `kind`, `formal_name`, `identity`, `aliases`, `links`,
`description`. The `subsets[]` field on dataset items is carried
through from the Python pre-pass — leave it alone unless you
have a specific correction.

Top-level optional fields:
- `notes` — brief summary of what you changed, with counts
  ("recheck: restored N, promoted M to gated, kept K dropped;
  split L families; renamed P formal_names; ...").
- `dropped` — the union of organize-dropped entries you didn't
  restore or promote, plus any new drops you make. Same schema
  as organize's `dropped[]`.
- `gated` — entries promoted from `dropped[]` for which you
  confirmed a 401/403 gated URL. Each entry: `{name, kind,
  reason, gated_url, attempted_canonicals?}`.

```json
{
  "groups": [
    {
      "family": "...",
      "identity_keys": [...],
      "items": [
        {"kind": "...", "formal_name": "...", "identity": {...},
         "aliases": [...], "links": [...], "description": "..."}
      ]
    }
  ],
  "gated": [
    {"name": "...", "kind": "...", "reason": "...",
     "gated_url": "...", "attempted_canonicals": []}
  ],
  "dropped": [...],
  "notes": "..."
}
```

The output is the WHOLE revised lattice, not a diff — every
family that should remain in the lattice must appear in the
output, even if unchanged.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
