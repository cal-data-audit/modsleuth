"""Pre-audit Python pass — purely additive, never mutates the lattice.

Two phases:

1. **populate_subsets** — fetch each dataset node's HF README and fill
   in the `subsets[]` field with parsed config / component slugs.

2. **flag_audit_issues** — scan the augmented lattice for cases that
   likely warrant audit attention. Surfaces them as a top-level
   `audit_hints[]` array. Doesn't move items, doesn't restore drops,
   doesn't rename anything. The LLM auditor reads hints and decides
   what to do.

Design principle: Python finds patterns deterministically; the LLM
auditor exercises judgment. Python never deletes or rewrites what
organize produced.
"""
from __future__ import annotations

import os
import re
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Subset parsing
# ---------------------------------------------------------------------------

_TABLE_ROW = re.compile(r"^\|\s*(?P<col1>[A-Za-z][\w\-+. /]+?)\s*\|")
_SUBSET_HEADING_KEYWORDS = (
    "component", "subset", "composition", "source", "mix",
    "ingredient", "data source", "constituent", "sub-corpus",
    "sub-corpora", "data mix", "contents",
)
_HEADER_CELLS = frozenset({
    "subset", "subsets", "name", "source", "component", "components",
    "ingredient", "mix", "tokens", "size", "count", "split", "split name",
    "config", "config name", "data source", "data sources", "dataset",
    "datasets", "language", "format", "stage", "domain", "topic", "type",
    "category", "n", "rows", "examples", "samples", "filter", "url",
    "license", "---", ":---:", ":---", "---:", "feature", "features",
    "field", "key", "constituent", "constituents", "section",
})

# Family-root identity: exactly the single `family` key. Anything
# else is an entity leaf or an intermediate concept. Used by
# `_classify_item_role` and the missing-root detection.
_FAMILY_ROOT_KEYS = frozenset({"family"})

# Link kinds that typically identify ONE artifact alone. Used by
# `_has_unique_anchor`, the subsumption check, and the cross-family
# same-URL detector. Paper / hf_collection / blog are family-shared.
# `github` is excluded by default — too ambiguous (the family's repo
# vs. an item-specific repo); audit can override per-item.
_UNIQUE_ANCHOR_KINDS = frozenset({"hf_model", "hf_dataset", "vendor_docs"})


def fetch_card(formal_name: str, *, timeout: float = 10.0) -> str | None:
    url = f"https://huggingface.co/datasets/{formal_name}/raw/main/README.md"
    headers = {"User-Agent": "modsleuth/1.0"}
    # Same auth the LLM stages use: unauthenticated HF traffic is
    # rate-limited (~30/min), which silently starves `subsets[]` on
    # lattices with hundreds of dataset nodes. Auth also resolves
    # gated cards the token has access to.
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
    except requests.RequestException:
        return None
    return r.text if r.status_code == 200 else None


def _split_frontmatter(readme: str) -> tuple[dict | None, str]:
    m = re.match(r"---\n(.*?)\n---\n", readme, re.DOTALL)
    if not m:
        return None, readme
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = None
    return (fm if isinstance(fm, dict) else None), readme[m.end():]


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-+.]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def parse_subsets(readme: str) -> list[str]:
    raw_names: list[str] = []
    fm, body = _split_frontmatter(readme)

    if fm:
        configs = fm.get("configs")
        if isinstance(configs, list):
            for c in configs:
                if isinstance(c, dict) and isinstance(c.get("config_name"), str):
                    raw_names.append(c["config_name"])
                elif isinstance(c, str):
                    raw_names.append(c)

    sections = re.split(r"\n(#{1,6}\s+.+?)\n", "\n" + body)
    for i in range(1, len(sections), 2):
        heading_lower = sections[i].lower()
        if not any(kw in heading_lower for kw in _SUBSET_HEADING_KEYWORDS):
            continue
        section_body = sections[i + 1] if (i + 1) < len(sections) else ""
        for line in section_body.splitlines():
            m2 = _TABLE_ROW.match(line)
            if not m2:
                continue
            cell = m2.group("col1").strip()
            if cell.lower() in _HEADER_CELLS:
                continue
            if re.fullmatch(r"-+|:?-+:?", cell):
                continue
            raw_names.append(cell)

    seen, out = set(), []
    for raw in raw_names:
        slug = _slugify(raw)
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


def populate_subsets(lattice: dict, *, dry_run: bool = False) -> dict:
    """Walk the lattice; for each dataset node with empty `subsets`,
    fetch its HF README and populate `subsets[]`. Idempotent.

    Returns: {populated, failed, skipped, total_subsets}."""
    populated = failed = skipped = 0
    total_subsets = 0
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            if it.get("kind") != "dataset":
                continue
            it.setdefault("subsets", [])
            if it["subsets"]:
                skipped += 1
                continue
            card = fetch_card(it.get("formal_name") or "")
            if card is None:
                failed += 1
                continue
            subs = parse_subsets(card)
            if not dry_run:
                it["subsets"] = subs
            populated += 1
            total_subsets += len(subs)
    return {"populated": populated, "failed": failed, "skipped": skipped,
            "total_subsets": total_subsets}


# ---------------------------------------------------------------------------
# Audit-hint generation — purely additive, surfaces flags only
# ---------------------------------------------------------------------------


def _alnum_form(s: object) -> str:
    """Lowercase, strip everything that isn't [a-z0-9]. Used as a
    canonical fuzzy-match form to bridge `finemath-4plus` ↔
    `finemath4+` ↔ `FineMath4Plus`."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace("+", "plus")
    return re.sub(r"[^a-z0-9]", "", s)


def _name_variants(name: str) -> set[str]:
    """Return slug variants of a name for fuzzy matching against
    populated `subsets[]`. Generates the lowercase-kebab base form,
    common substitutions, and an alphanumeric-only canonical form."""
    s = (name or "").strip()
    slug = _slugify(s)
    out = {
        slug,
        slug.replace("-", ""),
        slug.replace("-", "_"),
        slug.replace("+", "plus"),
        slug.replace("+", "-plus"),
        _alnum_form(s),
    }
    out.add(slug.replace("-", "").replace("+", "plus"))
    out.add(slug.replace("-", "_").replace("+", "plus"))
    out.discard("")
    return out


def _is_bare_domain_url(url: object) -> bool:
    from urllib.parse import urlparse
    if not isinstance(url, str):
        return False
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if not p.scheme or not p.netloc:
        return False
    return not p.path.strip("/")


def _build_subset_index(lattice: dict) -> dict[str, list[tuple[dict, dict]]]:
    """Index: subset_slug → [(group, parent_item)] across all kept items.
    Both the literal slug and its alphanumeric-only canonical form are
    indexed, so a lookup using either form finds the parent."""
    index: dict[str, list[tuple[dict, dict]]] = {}
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            for sub in it.get("subsets") or []:
                if not isinstance(sub, str):
                    continue
                index.setdefault(sub, []).append((grp, it))
                alnum = _alnum_form(sub)
                if alnum and alnum != sub:
                    index.setdefault(alnum, []).append((grp, it))
    return index


def _classify_item_role(it: dict) -> str:
    """Return a one-word descriptor of the item's role for hint context.
    'family-root' = identity is exactly {family: X}, top of the lattice;
    'concept' = identity has more than just family but no production link;
    'canonical' = entity leaf with HF / GitHub primary link;
    'soft-anchored' = entity leaf with paper / blog / vendor_docs link;
    'unanchored' = no links and not a family root."""
    links = it.get("links") or []
    primary_kind = (links[0].get("kind") if links and isinstance(links[0], dict) else None)
    identity = it.get("identity") or {}
    ident_keys = set(identity.keys()) if isinstance(identity, dict) else set()
    primary_url = links[0].get("url") if links and isinstance(links[0], dict) else None

    if ident_keys == _FAMILY_ROOT_KEYS:
        return "family-root"
    if not primary_kind:
        return "unanchored"
    if _is_bare_domain_url(primary_url):
        return "concept"
    if primary_kind in ("hf_model", "hf_dataset", "github"):
        return "canonical"
    if primary_kind in ("paper", "blog", "vendor_docs", "hf_collection"):
        return "concept"
    return "other"


def flag_audit_issues(lattice: dict, *, input_names_set: set | None = None) -> dict:
    """Scan the lattice and surface cases that likely warrant audit
    attention. Adds entries to a top-level `audit_hints[]` array.
    Mutates nothing else — items, families, dropped[], subsets[] all
    pass through untouched.

    `input_names_set` (optional) is the set of original surface forms
    from the input names pile. When provided, `family_root_invented_alias`
    fires for family roots whose aliases don't trace back to any input
    form. When omitted, that check is skipped.

    Hint kinds emitted:

    - `item_matches_parent_subset`: an existing item's name slug
      appears in some other item's `subsets[]`. Audit decides
      whether to reshape under parent (typical for soft-anchored
      sub-components) or keep standalone (typical for foundational
      concepts that are also referenced as subsets).

    - `dropped_matches_parent_subset`: a dropped name's slug
      appears in some kept item's `subsets[]`. Audit decides
      whether to restore as `<parent>/<slug>` child.

    - `sibling_identity_collision`: two items in the same family
      carry identical identity dicts. Audit must add a discriminating
      facet or merge.

    - `cross_org_family`: a family contains items spanning multiple
      `identity.org` / `identity.vendor` values. Audit decides whether
      this is a substring false positive (split) or a legitimate
      product-line grouping (keep, optionally add a facet).

    - `formal_name_vs_canonical_url_mismatch`: the formal_name
      doesn't match the canonical path inside the primary HF URL
      (e.g., formal_name='MMLU' but URL is huggingface.co/datasets/cais/mmlu).
      Audit decides whether to rename to canonical form.

    - `phantom_item`: an item with `aliases: []`. Validator already
      rejects, but the hint is surfaced for context if the lattice
      somehow contains one.

    - `missing_family_root`: a family group has 2+ items but no
      family root (item with identity exactly `{family: X}`). Audit
      MUST synthesize one — every family needs a top concept node so
      vague mentions like "OLMo 3" or "Qwen3" can land somewhere.

    - `over_specified`: an item has a bare family-name alias (e.g.,
      "olmOCR", "Qwen3") but its formal_name pins a specific HF
      release (e.g., "allenai/olmOCR-7B-0225-preview"). Audit checks
      the source: if the source mention is bare, the alias belongs on
      the family root, not the leaf. Typical action: split — move the
      bare alias to the family root and keep specific aliases on the
      leaf.

    - `branch_variant_in_formal_name`: HF git-revspec `<repo>@<branch>`
      in formal_name. Branches are revisions of one repo; collapse to
      canonical repo with branch names as aliases.

    - `concept_subsumed_candidate`: within a family, item A's identity
      facets are a strict subset of sibling B's, and A has no
      item-unique anchor. A is likely a concept (partial spec).

    - `subset_with_anchor`: within a family, item A's facets are a
      strict subset of sibling B's, and BOTH have item-unique anchors.
      Dataset-config / subset-of relationship; note in description,
      relate may emit a `subset_of` edge.

    - `same_url_duplicate`: two items in the same family share the
      same primary URL. Merge into one item; runtime-mode differences
      (thinking / no-thinking) belong on edges.

    - `same_url_cross_family`: same primary URL appears in items from
      different families. Indicates a missed cross-bucket merge — one
      family should own the artifact, others should drop or relocate.

    - `concept_with_no_entity`: family has multiple concept items but no
      entity (no item-unique HF / vendor URL). Either legit (foundational
      data resource) or extract / organize missed a release.

    - `family_root_invented_alias`: family root's aliases don't trace to
      any input-pile surface form (only fires when `input_names_set` is
      provided). Vague relate mentions may fail to resolve to this root.

    Returns counters by hint kind.
    """
    hints: list[dict] = []
    counts: dict[str, int] = {}

    def add(kind: str, **payload: Any) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        hints.append({"kind": kind, **payload})

    subset_index = _build_subset_index(lattice)

    # Hint 1 + 5 + 6: per-item scans
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            fn = it.get("formal_name") or ""
            family = grp.get("family")

            # Phantom items: empty aliases (now no exception — all items
            # must trace to input pile, including family roots whose alias
            # is the bare family name as the source wrote it).
            aliases = it.get("aliases") or []
            if not aliases:
                add("phantom_item",
                    item_formal_name=fn, item_family=family,
                    rationale="No aliases trace to input pile. Even a family root needs the bare family name as alias.")

            # formal_name vs canonical-URL mismatch
            links = it.get("links") or []
            if links and isinstance(links[0], dict):
                primary_kind = links[0].get("kind")
                primary_url = links[0].get("url") or ""
                if primary_kind in ("hf_model", "hf_dataset"):
                    m = re.search(r"huggingface\.co/(?:datasets/)?([^/]+/[^/?#]+)", primary_url)
                    if m:
                        canonical = m.group(1)
                        if canonical and canonical.lower() != fn.lower():
                            add("formal_name_vs_canonical_url_mismatch",
                                item_formal_name=fn, item_family=family,
                                canonical_from_url=canonical, primary_url=primary_url,
                                rationale="formal_name doesn't match the canonical HF path in primary URL.")

            # item slug matches some OTHER item's subsets[]
            slugs: set[str] = set()
            for n in [fn] + (aliases or []):
                slugs.update(_name_variants(n or ""))
            slugs.discard("")
            seen_parents: set[str] = set()
            for slug in slugs:
                for parent_grp, parent in subset_index.get(slug, []):
                    if parent is it:
                        continue
                    parent_fn = parent.get("formal_name") or ""
                    if parent_fn in seen_parents:
                        continue
                    seen_parents.add(parent_fn)
                    add("item_matches_parent_subset",
                        item_formal_name=fn,
                        item_family=family,
                        item_role=_classify_item_role(it),
                        matched_parent=parent_fn,
                        matched_subset_slug=slug,
                        rationale=(
                            "Item's name appears in parent's subsets[]. "
                            "If item is soft-anchored (paper/blog) and lacks "
                            "its own canonical release, consider reshaping under "
                            "parent as `<parent>/<slug>` child item. If item is "
                            "a foundational concept other things derive from "
                            "(role='concept'), keep standalone — it's referenced "
                            "as a subset incidentally."
                        ))

    # Hint 2: dropped name slug matches some kept item's subsets[]
    for d in lattice.get("dropped") or []:
        if not isinstance(d, dict):
            continue
        name = d.get("name") or ""
        if not name:
            continue
        slugs = _name_variants(name)
        seen_parents = set()
        for slug in slugs:
            for parent_grp, parent in subset_index.get(slug, []):
                parent_fn = parent.get("formal_name") or ""
                if parent_fn in seen_parents:
                    continue
                seen_parents.add(parent_fn)
                add("dropped_matches_parent_subset",
                    dropped_name=name,
                    dropped_kind=d.get("kind"),
                    matched_parent=parent_fn,
                    matched_subset_slug=slug,
                    rationale=(
                        "Dropped name appears in parent's subsets[]. "
                        "Consider restoring as `<parent>/<slug>` child item "
                        "with identity inheriting parent + `subset: <slug>`."
                    ))

    # Hint 3: sibling identity collisions
    from collections import defaultdict
    for grp in lattice.get("groups") or []:
        by_id: dict[str, list[str]] = defaultdict(list)
        for it in grp.get("items") or []:
            key = repr(sorted((it.get("identity") or {}).items()))
            by_id[key].append(it.get("formal_name") or "")
        for key, fns in by_id.items():
            if len(fns) > 1:
                add("sibling_identity_collision",
                    family=grp.get("family"),
                    identity_key=key,
                    items=fns,
                    rationale=(
                        "Siblings carry identical identity dicts; lattice "
                        "can't tell them apart. Add a discriminating facet "
                        "or merge the items."
                    ))

    # Hint 4: cross-org families
    for grp in lattice.get("groups") or []:
        orgs: set[str] = set()
        items = grp.get("items") or []
        if len(items) < 2:
            continue
        for it in items:
            ident = it.get("identity") or {}
            org = ident.get("org") or ident.get("vendor")
            if org:
                orgs.add(org)
        if len(orgs) > 1:
            add("cross_org_family",
                family=grp.get("family"),
                orgs=sorted(orgs),
                item_count=len(items),
                rationale=(
                    "Family contains items from multiple orgs / vendors. "
                    "Audit's call: split into per-org families if substring "
                    "false positive, or keep merged if items share a real "
                    "product line (e.g., a base + its strict-superset extension)."
                ))

    # Hint 5: items with `@branch` in formal_name (HF git-revspec syntax)
    # These name a specific revision within the same canonical repo —
    # different branches are different checkpoints in the training history,
    # but the canonical repo identifier is the part before `@`. Audit
    # should collapse all `@branch` variants into one item under the
    # canonical formal_name, with branch names carried in aliases.
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            fn = it.get("formal_name") or ""
            if "@" in fn:
                base, _, branch = fn.partition("@")
                add("branch_variant_in_formal_name",
                    item_formal_name=fn,
                    item_family=grp.get("family"),
                    canonical_repo=base,
                    branch=branch,
                    rationale=(
                        "`@branch` in formal_name is HF git-revspec syntax "
                        "for a revision within the canonical repo. Different "
                        "branches are different checkpoints, but the artifact "
                        "identifier is the repo (the part before `@`). Collapse "
                        "all `@branch` siblings of this base into one item with "
                        f"formal_name='{base}'; carry branch names in aliases."
                    ))

    # Hint 6: groups missing a family root.
    # Every family MUST have an item whose identity is exactly {family: X}
    # — the lattice's top concept. Vague mentions like "OLMo 3" or "Qwen3"
    # land here. If the group has 1+ items but no root, audit must
    # synthesize one (with the bare family name as alias and a paper /
    # blog link if available, no production link).
    for grp in lattice.get("groups") or []:
        items = grp.get("items") or []
        if not items:
            continue
        root_count = sum(
            1 for it in items
            if set((it.get("identity") or {}).keys()) == _FAMILY_ROOT_KEYS
        )
        if root_count == 0:
            family = grp.get("family") or ""
            family_from_items = next(
                (it.get("identity", {}).get("family") for it in items
                 if isinstance(it.get("identity"), dict)
                 and it.get("identity", {}).get("family")),
                None,
            )
            add("missing_family_root",
                family=family,
                family_from_items=family_from_items,
                item_count=len(items),
                rationale=(
                    f"Group has {len(items)} items but no item with identity "
                    "exactly {{family: <name>}}. Synthesize a family root "
                    "with the bare family name as alias; link to the family "
                    "paper / collection / blog if any; no production link."
                ))

    # Hint 7: over-specified items.
    # If an item has the bare family name (or close variants) as one of
    # its aliases AND its formal_name is a fully-pinnable HF release
    # (with version / size / stage facets), then a vague mention should
    # land on the family root, not on this leaf. Surface for audit to
    # split: move bare alias to root, keep specific aliases on leaf.
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            fn = it.get("formal_name") or ""
            identity = it.get("identity") or {}
            if not isinstance(identity, dict):
                continue
            family = identity.get("family")
            if not family:
                continue
            # Skip family roots themselves
            if set(identity.keys()) == _FAMILY_ROOT_KEYS:
                continue
            family_alnum = _alnum_form(family)
            if not family_alnum:
                continue
            for alias in it.get("aliases") or []:
                alias_alnum = _alnum_form(alias)
                if alias_alnum == family_alnum:
                    add("over_specified",
                        item_formal_name=fn,
                        item_family=family,
                        bare_alias=alias,
                        item_identity=identity,
                        rationale=(
                            f"Item formal_name pins specific facets "
                            f"({sorted(k for k in identity if k != 'family')}) "
                            f"but carries bare family-name alias {alias!r}. "
                            "Verify by re-reading the source: if the source "
                            "mention is bare, move this alias to the family "
                            "root; vague mentions should not silently bind to "
                            "an arbitrary specific release."
                        ))
                    break

    # Hint 8 + 9: facet-subsumption pairs and same-URL duplicates.
    # Within each family, find pairs (A, B) with A.facets ⊂ B.facets
    # (strict subset). Classify by whether A has an item-unique anchor.
    # Also flag pairs that share the same primary URL (same-URL =
    # same-entity rule violated).
    for grp in lattice.get("groups") or []:
        items = grp.get("items") or []
        # Build summaries: (item, facet_set, primary_url, has_unique_anchor)
        summaries: list[tuple[dict, frozenset, str, bool]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            ident = it.get("identity") or {}
            if not isinstance(ident, dict):
                continue
            facet_set = frozenset(
                (k, str(v)) for k, v in ident.items()
                if v is not None and str(v) != ""
            )
            primary_url = ""
            links = it.get("links") or []
            if links and isinstance(links[0], dict):
                primary_url = (links[0].get("url") or "").strip()
            has_unique = _has_unique_anchor(it)
            summaries.append((it, facet_set, primary_url, has_unique))

        n = len(summaries)
        for i in range(n):
            it_a, fs_a, url_a, anc_a = summaries[i]
            for j in range(n):
                if i == j:
                    continue
                it_b, fs_b, url_b, anc_b = summaries[j]

                # Same-URL duplicate (only emit once per pair, for the
                # lexicographically-earlier formal_name)
                if (url_a and url_a == url_b
                        and (it_a.get("formal_name") or "") <= (it_b.get("formal_name") or "")
                        and i < j):
                    add("same_url_duplicate",
                        family=grp.get("family"),
                        items=[it_a.get("formal_name"), it_b.get("formal_name")],
                        shared_url=url_a,
                        identity_a=dict(it_a.get("identity") or {}),
                        identity_b=dict(it_b.get("identity") or {}),
                        rationale=(
                            "Two items share the same primary URL — they "
                            "resolve to the same canonical artifact. Merge "
                            "into one item with both surface forms in "
                            "aliases[]; runtime-mode differences (thinking / "
                            "no-thinking, sampling params) belong on edges, "
                            "not facets."
                        ))

                # Subsumption: fs_a is a strict subset of fs_b.
                # Skip cases where A is the family root (single `family`
                # key) — the root is subsumed by every leaf by lattice
                # design, flagging is meaningless. Hints target INTERIOR
                # items only (multi-facet identity).
                if fs_a < fs_b and len(fs_a) > 1:
                    if not anc_a:
                        add("concept_subsumed_candidate",
                            family=grp.get("family"),
                            subsumed_formal_name=it_a.get("formal_name"),
                            subsumed_identity=dict(it_a.get("identity") or {}),
                            subsumes_formal_name=it_b.get("formal_name"),
                            subsumes_identity=dict(it_b.get("identity") or {}),
                            rationale=(
                                "Item's identity facets are a strict subset "
                                "of a sibling's, and item has no item-unique "
                                "anchor (hf_model / hf_dataset / vendor_docs). "
                                "Likely a concept (partial spec). Confirm "
                                "kind=concept and that links[] hold only "
                                "family-shared anchors."
                            ))
                    else:
                        add("subset_with_anchor",
                            family=grp.get("family"),
                            child_formal_name=it_a.get("formal_name"),
                            child_identity=dict(it_a.get("identity") or {}),
                            parent_formal_name=it_b.get("formal_name"),
                            parent_identity=dict(it_b.get("identity") or {}),
                            rationale=(
                                "Item's identity facets are a strict subset "
                                "of a sibling's, and BOTH items have unique "
                                "anchors. This is a dataset-config / "
                                "subset-of relationship; ensure both "
                                "descriptions note the relationship. Relate "
                                "may emit a `subset_of` edge later."
                            ))

    # Hint 10: same UNIQUE-ANCHOR URL across DIFFERENT families.
    # Restricted to hf_model / hf_dataset / vendor_docs — paper /
    # collection / blog URLs are family-shared by design and would
    # falsely fire (e.g., the family paper cited by every olmo-* family).
    from collections import defaultdict as _dd
    url_to_owners: dict[str, list[tuple[str, str]]] = _dd(list)
    for grp in lattice.get("groups") or []:
        fam = grp.get("family") or ""
        for it in grp.get("items") or []:
            if not isinstance(it, dict):
                continue
            links = it.get("links") or []
            if not links or not isinstance(links[0], dict):
                continue
            kind = links[0].get("kind")
            if kind not in _UNIQUE_ANCHOR_KINDS:
                continue
            url = (links[0].get("url") or "").strip()
            if url:
                url_to_owners[url].append((fam, it.get("formal_name") or ""))
    for url, owners in url_to_owners.items():
        fams = {f for f, _ in owners}
        if len(fams) > 1:
            add("same_url_cross_family",
                shared_url=url,
                families=sorted(fams),
                items=[{"family": f, "formal_name": fn} for f, fn in owners],
                rationale=(
                    "Same primary URL appears in items from different "
                    "families. The cross-bucket merge step missed this "
                    "duplicate. Decide which family owns the artifact "
                    "(typically the family the URL's namespace belongs to) "
                    "and remove or relocate the others."
                ))

    # Hint 11: family with only concepts, no entities.
    # The family-root concept exists but no item carries an item-unique
    # anchor. Either source genuinely never pinned a release (legit for
    # foundational data resources like Common Crawl), or extract / organize
    # missed the specific release names. Audit reads sources to decide.
    for grp in lattice.get("groups") or []:
        items = grp.get("items") or []
        if not items:
            continue
        has_entity = any(_has_unique_anchor(it) for it in items if isinstance(it, dict))
        if has_entity:
            continue
        # Skip if the only item is a single root with broad anchor (paper/
        # blog/collection/single-release-as-root) — that's a legitimate
        # single-artifact family
        if len(items) <= 1:
            continue
        add("concept_with_no_entity",
            family=grp.get("family"),
            item_count=len(items),
            item_formal_names=[
                it.get("formal_name") for it in items if isinstance(it, dict)
            ][:10],
            rationale=(
                "Family has multiple concept items but no entity (no "
                "item-unique HF / vendor URL). Either source genuinely "
                "describes the family without pinning a release "
                "(foundational data resource — keep as-is) OR a release "
                "exists but extract / organize missed it. Re-read sources "
                "at batches_dir to decide."
            ))

    # Hint 12: family-root with invented alias (no input-pile trace).
    # Fires when the family root's aliases are all planner-synthesized
    # forms that never appear in the input names pile. The lattice's
    # specificity-discipline rule expects the bare family name to be a
    # real input mention; if none of the aliases came from input, vague
    # source mentions in relate may fail to resolve to this root.
    if input_names_set is not None:
        for grp in lattice.get("groups") or []:
            for it in grp.get("items") or []:
                if not isinstance(it, dict):
                    continue
                ident = it.get("identity") or {}
                if not isinstance(ident, dict):
                    continue
                if list(ident.keys()) != ["family"]:
                    continue  # not a family root
                aliases = it.get("aliases") or []
                from_input = [a for a in aliases if a in input_names_set]
                if not from_input:
                    add("family_root_invented_alias",
                        family=grp.get("family"),
                        item_formal_name=it.get("formal_name"),
                        aliases=list(aliases),
                        rationale=(
                            "Family root has no alias from the input pile — "
                            "all aliases are planner-synthesized (e.g., "
                            "added parens disambiguator, abbreviated form, "
                            "or formal_name echo). Source mentions may use "
                            "a bare form that won't exact-match. Add the "
                            "actual input surface forms as aliases (find by "
                            "scanning input names for the family substring), "
                            "or accept the gap if the family genuinely had "
                            "no bare-name mention."
                        ))

    # Attach hints to lattice (additive only)
    if hints:
        lattice.setdefault("audit_hints", []).extend(hints)
    return counts


def _has_unique_anchor(item: dict) -> bool:
    """Return True if the item carries at least one link kind that
    typically identifies one artifact alone (`hf_model`, `hf_dataset`,
    `vendor_docs`). `paper`, `hf_collection`, `blog` are family-shared
    and do not count. `github` is excluded by default — too ambiguous
    (could be the family's repo). Audit can override per-item."""
    links = item.get("links") or []
    if not isinstance(links, list):
        return False
    for link in links:
        if not isinstance(link, dict):
            continue
        if (link.get("kind") in _UNIQUE_ANCHOR_KINDS
                and isinstance(link.get("url"), str)
                and link["url"].strip()):
            return True
    return False


def complete_lattice_structure(lattice: dict) -> dict:
    """Deterministic structural completion run AFTER organize, BEFORE
    validation. Two minimal phases:

    1. **formal_name → aliases echo** — every item's `formal_name` is
       added to its `aliases[]` if not already there. This guarantees
       no item has empty aliases (anti-phantom rule satisfied
       mechanically) and ensures the bare canonical label is always a
       searchable surface form.

    2. **Virtual family root synthesis** — for every group, if no item
       has identity exactly `{family: X}`, append a synthesized root
       with `formal_name = <family name>`, `identity = {family: X}`,
       `aliases = [X]`, empty links, and null description. Audit reads
       the augmented lattice; relate gets a guaranteed concept-level
       node to land vague mentions on.

    Idempotent: running twice is a no-op (the first run satisfies both
    invariants). Mutates `lattice` in place; returns counters.

    Returns: `{aliases_added, roots_synthesized}`.
    """
    aliases_added = 0
    roots_synthesized = 0
    if not isinstance(lattice, dict):
        return {"aliases_added": 0, "roots_synthesized": 0}
    groups = lattice.get("groups")
    if not isinstance(groups, list):
        return {"aliases_added": 0, "roots_synthesized": 0}

    for grp in groups:
        if not isinstance(grp, dict):
            continue
        items = grp.get("items")
        if not isinstance(items, list):
            continue

        # Phase 1: formal_name → aliases
        for it in items:
            if not isinstance(it, dict):
                continue
            fn = it.get("formal_name")
            if not isinstance(fn, str) or not fn:
                continue
            aliases = it.get("aliases")
            if not isinstance(aliases, list):
                aliases = []
                it["aliases"] = aliases
            if fn not in aliases:
                aliases.append(fn)
                aliases_added += 1

        # Phase 2: virtual family root if missing
        # Find the family value: prefer an item's identity.family, else
        # group's family field.
        family_val: str | None = None
        for it in items:
            if not isinstance(it, dict):
                continue
            ident = it.get("identity") or {}
            if isinstance(ident, dict):
                fv = ident.get("family")
                if isinstance(fv, str) and fv.strip():
                    family_val = fv
                    break
        if not family_val:
            family_val = grp.get("family")
        if not isinstance(family_val, str) or not family_val.strip():
            continue  # can't synthesize without a family name

        has_root = any(
            isinstance(it, dict)
            and list((it.get("identity") or {}).keys()) == ["family"]
            for it in items
        )
        if not has_root:
            # Inherit kind from the most common kind in the group
            kinds = [it.get("kind") for it in items
                     if isinstance(it, dict) and it.get("kind")]
            kind = kinds[0] if kinds else "model"
            root = {
                "kind": kind,
                "formal_name": family_val,
                "identity": {"family": family_val},
                "aliases": [family_val],
                "links": [],
                "description": None,
                "_synthesized": True,
            }
            items.insert(0, root)
            roots_synthesized += 1

    return {
        "aliases_added": aliases_added,
        "roots_synthesized": roots_synthesized,
    }


def _project_identity(identity: dict, keys: tuple[str, ...]) -> dict:
    """Return a new identity dict containing only `family` plus the
    given keys, in stable order (`family` first)."""
    out: dict = {}
    if "family" in identity:
        out["family"] = identity["family"]
    for k in keys:
        if k in identity and k != "family":
            out[k] = identity[k]
    return out


def _concept_label(identity: dict, key_order: list[str]) -> str:
    """Compose a human-readable label for a synthesized concept:
    `<family> <v1> <v2> ...` in `key_order`. `family` first."""
    parts = []
    if "family" in identity:
        parts.append(str(identity["family"]))
    for k in key_order:
        if k == "family":
            continue
        if k in identity:
            parts.append(str(identity[k]))
    return " ".join(parts).strip()


def _identity_signature(identity: dict) -> frozenset:
    """Canonical signature for identity-equality across items."""
    return frozenset(
        (k, str(v)) for k, v in (identity or {}).items()
        if v is not None and str(v) != ""
    )


def expand_concept_lattice(lattice: dict) -> dict:
    """Synthesize interior concept nodes by projecting leaves onto
    every non-empty proper subset of their non-family identity keys.

    Runs AFTER audit. The audit lattice contains only items the source
    mentioned (entity or concept). For each leaf with N>=2 facet keys,
    we generate 2^(N-1) - 1 projection points (subsets of size 1..N-1).
    Each projection that doesn't already match an existing item becomes
    a new concept node with `_generated: true`.

    The expansion is idempotent: running twice adds nothing the second
    time (every projection already exists).

    Returns: {concepts_synthesized}.
    """
    synthesized = 0
    if not isinstance(lattice, dict):
        return {"concepts_synthesized": 0}
    groups = lattice.get("groups")
    if not isinstance(groups, list):
        return {"concepts_synthesized": 0}

    from itertools import combinations

    for grp in groups:
        if not isinstance(grp, dict):
            continue
        items = grp.get("items")
        if not isinstance(items, list):
            continue

        existing_signatures: set[frozenset] = set()
        for it in items:
            if isinstance(it, dict):
                ident = it.get("identity")
                if isinstance(ident, dict):
                    existing_signatures.add(_identity_signature(ident))

        # Determine the family value and dominant kind
        family_val = None
        kind_default = "model"
        for it in items:
            if isinstance(it, dict):
                ident = it.get("identity") or {}
                if isinstance(ident, dict) and isinstance(ident.get("family"), str):
                    family_val = ident["family"]
                k = it.get("kind")
                if isinstance(k, str) and k in ("model", "dataset"):
                    kind_default = k
                    if family_val:
                        break
        if not family_val:
            continue

        # Collect every projection from leaves with >=2 non-family facets
        new_concepts: list[dict] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            ident = it.get("identity") or {}
            if not isinstance(ident, dict):
                continue
            non_family_keys = [k for k in ident if k != "family" and ident[k] not in (None, "")]
            if len(non_family_keys) < 2:
                continue  # nothing to project from

            kind = it.get("kind") if it.get("kind") in ("model", "dataset") else kind_default
            for r in range(1, len(non_family_keys)):
                for subset in combinations(non_family_keys, r):
                    proj = _project_identity(ident, subset)
                    sig = _identity_signature(proj)
                    if sig in existing_signatures:
                        continue
                    existing_signatures.add(sig)
                    label = _concept_label(proj, list(non_family_keys))
                    new_concepts.append({
                        "kind": kind,
                        "formal_name": label,
                        "identity": proj,
                        "aliases": [label] if label else [],
                        "links": [],
                        "description": None,
                        "_generated": True,
                    })

        if new_concepts:
            # Insert sorted by ascending facet-count, then alphabetically
            new_concepts.sort(key=lambda c: (
                len(c["identity"]),
                c["formal_name"],
            ))
            # Append after the family root if present, else at end
            insert_at = len(items)
            for idx, it in enumerate(items):
                if isinstance(it, dict):
                    ident = it.get("identity") or {}
                    if isinstance(ident, dict) and list(ident.keys()) == ["family"]:
                        insert_at = idx + 1
                        break
            items[insert_at:insert_at] = new_concepts
            synthesized += len(new_concepts)

    return {"concepts_synthesized": synthesized}


def populate_then_flag(
    lattice: dict, *,
    dry_run: bool = False,
    input_names_set: set | None = None,
) -> dict:
    """Two-phase Python pre-pass for audit:

    1. populate_subsets — fill in subsets[] on every dataset node.
    2. flag_audit_issues — surface suspicious cases as audit_hints[].

    BOTH PHASES ARE PURELY ADDITIVE. Items, families, dropped[]
    pass through untouched. The auditor reads the augmented lattice
    plus the hints and decides what to do.

    `input_names_set` (optional) is the set of original surface forms
    from the names pile; when provided, `family_root_invented_alias`
    hints fire for roots whose aliases don't trace to input.
    """
    pop = populate_subsets(lattice, dry_run=dry_run)
    if dry_run:
        return {"populate": pop, "flag": {}}
    flag = flag_audit_issues(lattice, input_names_set=input_names_set)
    return {"populate": pop, "flag": flag}
