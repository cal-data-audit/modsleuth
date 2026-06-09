"""Lattice resolution — given a mention string, return ranked candidates.

Used by relate to map a source mention onto the most-specific lattice
item (entity or concept). Also detects when the mention is a subset of
a parent dataset (matches the parent's `subsets[]` slug or a
sub-corpus suffix pattern) and surfaces that as a "subset_of" hint —
so relate emits ONE edge to the parent, with the subset noted in
description, instead of materializing leaf-level subset edges.

Scoring is hybrid:

- Exact alphanum-normalized match against any `formal_name` / alias
  scores +100 (always wins ties).
- Token overlap (Jaccard-like) between mention tokens and the item's
  full token bag (formal_name + aliases + identity values + family).
- Synthesized concepts (`_generated: true`) get a small penalty so
  planner-emitted entities are preferred when scores are otherwise
  equal.
- Subset detection: if mention's alphanum form matches a parent
  item's `subsets[]` slug exactly, OR mention has a known sub-corpus
  suffix (`-pool`, `-web`, `-edu`, ...) whose prefix matches a
  parent's surface form, the parent is surfaced with
  `subset_match_slug` and the candidate is flagged
  `address_form: "subset"`.

CLI: `python -m modsleuth.resolve "<mention>" [--top K] [--lattice PATH]`.

The function does NOT pick a winner — it ranks. Relate picks based on
surrounding source context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import click

from .store import all_rows, loads


_UNIQUE_ANCHOR_KINDS = frozenset({"hf_model", "hf_dataset", "vendor_docs"})

# Sub-corpus suffix patterns. When a mention ends with one of these and
# its prefix matches an existing item's surface form, treat as subset.
_SUBCORPUS_SUFFIXES = (
    "-pool", "-web", "-web-pro", "-web-pro-max", "-edu", "-baseline",
    "-refinedweb", "-code", "-math", "-qa", "-instruct-mix",
    "_pool", "_web", "_edu", "_code", "_math",
)


def _normalize_alnum(s: object) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(c.lower() for c in s if c.isalnum())


def _tokenize(s: object) -> list[str]:
    if not isinstance(s, str):
        return []
    return [t.lower() for t in re.split(r"[/_\-. ]+", s) if t]


def _build_index(lattice: dict) -> list[dict]:
    """Flatten the lattice into a list of indexed item dicts with
    precomputed token bags and alphanum surface sets."""
    out: list[dict] = []
    for grp in lattice.get("groups") or []:
        if not isinstance(grp, dict):
            continue
        fam = grp.get("family") or ""
        for it in grp.get("items") or []:
            if not isinstance(it, dict):
                continue
            fn = it.get("formal_name") or ""
            aliases = it.get("aliases") or []
            ident = it.get("identity") or {}
            subsets = it.get("subsets") or []
            surfaces = list({fn, *aliases})
            tokens = set(_tokenize(fam))
            for s in surfaces:
                tokens.update(_tokenize(s))
            if isinstance(ident, dict):
                for v in ident.values():
                    if isinstance(v, (str, int, float)):
                        tokens.update(_tokenize(str(v)))
            alnum_surfaces = {_normalize_alnum(s) for s in surfaces}
            alnum_surfaces.discard("")
            # Subset slug index — alphanum forms of every slug for fast lookup
            subset_alnum: dict[str, str] = {}
            if isinstance(subsets, list):
                for sub in subsets:
                    if isinstance(sub, str) and sub.strip():
                        subset_alnum[_normalize_alnum(sub)] = sub
            out.append({
                "family": fam,
                "formal_name": fn,
                "identity": ident if isinstance(ident, dict) else {},
                "aliases": aliases,
                "surfaces": surfaces,
                "tokens": tokens,
                "alnum_surfaces": alnum_surfaces,
                "subsets": subsets if isinstance(subsets, list) else [],
                "subset_alnum": subset_alnum,
                "is_synth": bool(it.get("_generated")),
                "kind": it.get("kind"),
                "has_unique_anchor": _has_unique_anchor(it),
            })
    return out


def _has_unique_anchor(it: dict) -> bool:
    links = it.get("links") or []
    if not isinstance(links, list):
        return False
    for l in links:
        if isinstance(l, dict) and l.get("kind") in _UNIQUE_ANCHOR_KINDS:
            url = l.get("url")
            if isinstance(url, str) and url.strip():
                return True
    return False


def _detect_subset(mention: str, mention_alnum: str, items: list[dict]) -> tuple[dict, str] | None:
    """Detect if mention is a subset of some lattice item.

    Two paths:
    1. mention's alphanum form exactly matches a slug in some item's `subsets[]`.
    2. mention ends in a known sub-corpus suffix and the prefix matches an item.

    Returns (parent_item, subset_slug) or None.
    """
    # Path 1: direct subset-slug hit
    for it in items:
        slug = it["subset_alnum"].get(mention_alnum)
        if slug:
            return (it, slug)

    # Path 2: suffix pattern. Lowercase original text for suffix check.
    m_lower = mention.lower().strip()
    for suf in _SUBCORPUS_SUFFIXES:
        if m_lower.endswith(suf) and len(m_lower) > len(suf) + 2:
            prefix = m_lower[: -len(suf)].rstrip("-_ ")
            prefix_alnum = _normalize_alnum(prefix)
            if not prefix_alnum:
                continue
            for it in items:
                if prefix_alnum in it["alnum_surfaces"]:
                    return (it, m_lower[len(prefix):].lstrip("-_ "))
    return None


def resolve(
    mention: str,
    lattice: dict,
    *,
    k: int = 3,
    score_floor: float = 0.0,
) -> list[dict]:
    """Return top-k candidates for `mention`.

    Each candidate dict has:

    - `formal_name`, `family`, `identity`, `kind`
    - `score` — total score
    - `match_reasons` — list of strings explaining the score
    - `address_form` — "leaf" | "concept" | "root" | "subset" | "off-lattice"
    - `subset_of` (only when address_form="subset") — `{parent_formal_name, slug}`

    Empty list if no mention tokens.
    """
    items = _build_index(lattice)
    m_alnum = _normalize_alnum(mention)
    m_tokens = set(_tokenize(mention))
    if not m_tokens:
        return []

    # Subset detection: surface a parent + slug if mention is a subset.
    # Score is intentionally below direct-exact-match (100) so that a
    # direct top-level alias hit wins. The subset candidate appears as
    # a secondary suggestion when the direct match is weak.
    sub_match = _detect_subset(mention, m_alnum, items)
    subset_candidate: dict | None = None
    if sub_match:
        parent, slug = sub_match
        subset_candidate = {
            "formal_name": parent["formal_name"],
            "family": parent["family"],
            "identity": parent["identity"],
            "kind": parent["kind"],
            "score": 70.0,
            "match_reasons": [
                f"mention '{mention}' matches subset slug '{slug}' "
                f"of parent '{parent['formal_name']}'",
            ],
            "address_form": "subset",
            "subset_of": {
                "parent_formal_name": parent["formal_name"],
                "slug": slug,
            },
        }

    scored: list[tuple[float, dict, list[str]]] = []
    for it in items:
        reasons: list[str] = []
        score = 0.0

        # Exact alphanum surface match
        if m_alnum and m_alnum in it["alnum_surfaces"]:
            score += 100.0
            reasons.append("exact-alphanum match on surface form")

        # Token overlap (covers fuzzy / multi-word cases)
        common = m_tokens & it["tokens"]
        if common:
            cov_m = len(common) / max(len(m_tokens), 1)
            cov_i = len(common) / max(len(it["tokens"]), 1)
            score += 10.0 * cov_m
            score += 5.0 * cov_i
            reasons.append(
                f"token overlap: {sorted(common)} "
                f"(mention coverage {cov_m:.2f}, item coverage {cov_i:.2f})"
            )

        # Synthesized concept penalty (slight) — prefers planner items on tie
        if it["is_synth"]:
            score -= 0.5

        if score < score_floor:
            continue
        scored.append((score, it, reasons))

    scored.sort(key=lambda x: -x[0])
    direct_candidates: list[dict] = []
    for score, it, reasons in scored[: max(k, 0)]:
        ident_keys = list(it["identity"].keys()) if isinstance(it["identity"], dict) else []
        if list(ident_keys) == ["family"]:
            af = "root"
        elif it["has_unique_anchor"]:
            af = "leaf"
        else:
            af = "concept"
        direct_candidates.append({
            "formal_name": it["formal_name"],
            "family": it["family"],
            "identity": it["identity"],
            "kind": it["kind"],
            "score": round(score, 2),
            "match_reasons": reasons,
            "address_form": af,
        })

    # Merge: direct candidates + subset candidate (if any), sorted by
    # score. Direct exact (>=100) wins over subset (70). Subset wins
    # when the best direct match is partial (token overlap only).
    all_candidates = direct_candidates[:]
    if subset_candidate is not None:
        all_candidates.append(subset_candidate)
    all_candidates.sort(key=lambda c: -c["score"])
    return all_candidates[:k]


def _latest_lattice_path() -> Path:
    """Find the most recent organize / audit artifact path on disk."""
    rows = all_rows(
        "SELECT id, stage, attrs FROM runs "
        "WHERE stage IN ('organize','audit') AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 1"
    )
    if not rows:
        raise click.ClickException(
            "no organize or audit run found; cannot resolve mentions"
        )
    attrs = loads(rows[0]["attrs"], default={}) or {}
    p = attrs.get("artifact_path")
    if not p or not Path(p).exists():
        raise click.ClickException(
            f"latest {rows[0]['stage']} artifact missing on disk"
        )
    return Path(p)


@click.command()
@click.argument("mention")
@click.option("--top", "top_k", type=int, default=3, help="number of candidates")
@click.option("--lattice", "lattice_path", type=str, default=None,
              help="path to lattice JSON; defaults to the latest organize/audit")
@click.option("--json", "as_json", is_flag=True,
              help="emit JSON output instead of human-readable text")
def main(mention: str, top_k: int, lattice_path: str | None, as_json: bool):
    """Resolve a mention against the lattice.

    \b
    Example: python -m modsleuth.resolve "OLMo 3 7B Base" --top 3
    """
    path = Path(lattice_path).resolve() if lattice_path else _latest_lattice_path()
    lattice = json.loads(path.read_text())
    cands = resolve(mention, lattice, k=top_k)
    if as_json:
        click.echo(json.dumps({"mention": mention, "candidates": cands}, indent=2))
        return
    if not cands:
        click.echo(f"(no candidates for {mention!r})")
        return
    click.echo(f"mention: {mention!r}")
    for i, c in enumerate(cands, 1):
        af = c["address_form"]
        marker = "[SUBSET]" if af == "subset" else f"[{af}]"
        click.echo(f"  {i}. {marker} {c['formal_name']}  (score {c['score']})")
        click.echo(f"     family: {c['family']}, identity: {c['identity']}")
        if af == "subset":
            click.echo(f"     subset_of: {c['subset_of']}")
        for r in c["match_reasons"]:
            click.echo(f"     - {r}")


if __name__ == "__main__":
    main()
