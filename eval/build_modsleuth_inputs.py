#!/usr/bin/env python3
"""Build per-target ModSleuth input files for `pooled_eval.py`.

Implements the depth-1 and unbounded attribution rules described in
paper §B ("Attribution scopes for ModSleuth"). Reads a merged graph
(``merge_artifact.json``, the 14,769-edge ModSleuth artifact) and emits

    prov_<target>.json            # depth-1 scope
    prov_unbounded_<target>.json  # depth-1 ∪ seed-tag ∪ uniquely-tied worker

into ``baselines/outputs/`` so that ``pooled_eval.py`` can pool ModSleuth
edges alongside the four single-pass baselines.

Usage:

    python build_modsleuth_inputs.py \
        --merge-artifact path/to/merge_artifact.json \
        --out-dir ../baselines/outputs

Run before ``pooled_eval.py`` to (re)generate ModSleuth's contribution.

Rules (verbatim from paper §B):

1. **depth-1**: a relationship is attributed to target T iff its
   subject's canonical form exactly matches T's canonical identifier,
   where the canonical form lowercases the string and collapses
   non-alphanumeric runs to hyphens (preserving any HF org prefix).

2. **unbounded**: depth-1 ∪ either of two storage-path signals on the
   relation's ``anchor_list``:

   * **seed-tagged anchor** — at least one anchor's source path contains
     ``/seeds/<seed_dir>/`` for T's seed directory.
   * **uniquely-tied worker** — at least one anchor's source path
     contains ``/workers/<worker>/``, and that worker — aggregated across
     the entire merge artifact — only co-occurs with seed-tagged anchors
     belonging to T's seed directory.

   Anchors that point only to public URLs (e.g., arxiv.org, hf.co)
   carry no per-investigation provenance and are excluded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Target → (canonical subject for depth-1, seed-directory name for unbounded).
# The canonical form here matches the canonicalize() rule used in pooled_eval.py.
TARGET_CONFIG: dict[str, dict[str, str]] = {
    "olmo3": {
        "canonical_subject": "allenai/olmo-3-1125-32b",
        "seed_dir":          "OLMo_3",
    },
    "nemotron3_super": {
        "canonical_subject": "nvidia/nvidia-nemotron-3-super-120b-a12b-bf16",
        "seed_dir":          "nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    },
    "dr_tulu": {
        "canonical_subject": "rl-research/dr-tulu-8b",
        "seed_dir":          "rl-research_DR-Tulu-8B",
    },
    "smollm3": {
        "canonical_subject": "huggingfacetb/smollm3-3b",
        "seed_dir":          "HuggingFaceTB_SmolLM3-3B",
    },
}


_SEP_RE = re.compile(r"[^a-z0-9]+")


def canonicalize(raw: str) -> str:
    """Same canonical form used by ``pooled_eval.py`` for cluster keys."""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    if "/" in s:
        org, rest = s.split("/", 1)
        rest = _SEP_RE.sub("-", rest).strip("-")
        return f"{org}/{rest}"
    return _SEP_RE.sub("-", s).strip("-")


_SEED_RE   = re.compile(r"/seeds/([^/]+)/")
_WORKER_RE = re.compile(r"/workers/([^/]+)/")


def _anchor_sources(relation: dict) -> list[str]:
    """Return all source-path strings on a relation's anchor_list."""
    out = []
    for a in (relation.get("anchor_list") or []):
        if not isinstance(a, dict):
            continue
        src = a.get("source") or a.get("source_id") or ""
        if isinstance(src, str) and src:
            out.append(src)
    return out


def _seed_tags(sources: list[str]) -> set[str]:
    return {m.group(1) for s in sources for m in [_SEED_RE.search(s)] if m}


def _worker_tags(sources: list[str]) -> set[str]:
    return {m.group(1) for s in sources for m in [_WORKER_RE.search(s)] if m}


def build_worker_to_seeds(relations: list[dict]) -> dict[str, set[str]]:
    """Aggregate, across the entire merge artifact, the set of seed
    directories each worker co-occurs with."""
    worker_to_seeds: dict[str, set[str]] = defaultdict(set)
    for r in relations:
        sources = _anchor_sources(r)
        if not sources:
            continue
        seeds = _seed_tags(sources)
        if not seeds:
            continue
        for w in _worker_tags(sources):
            worker_to_seeds[w] |= seeds
    return dict(worker_to_seeds)


def relation_to_edge(rel: dict) -> dict:
    """Convert a merge_artifact relation to the edge schema pooled_eval.py reads."""
    evidence = []
    for a in (rel.get("anchor_list") or []):
        if not isinstance(a, dict):
            continue
        excerpt = a.get("excerpt") or a.get("text") or ""
        evidence.append({
            "source":      a.get("source") or a.get("source_id") or "",
            "location":    a.get("location") or "",
            "excerpt":     excerpt if isinstance(excerpt, str) else str(excerpt),
            "explanation": a.get("explanation") or "",
        })
    return {
        "subject":          rel.get("subject", ""),
        "object":           rel.get("object", ""),
        "relation_type":    rel.get("relation", ""),
        "dependency_kind":  rel.get("dependency_kind", ""),
        "description":      rel.get("description", ""),
        "evidence":         evidence,
    }


def attribute_relations(
    relations: list[dict],
    targets: dict[str, dict[str, str]],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Return (depth1_per_target, unbounded_per_target) edge lists."""
    worker_to_seeds = build_worker_to_seeds(relations)
    seed_to_target = {cfg["seed_dir"]: t for t, cfg in targets.items()}

    depth1:    dict[str, list[dict]] = {t: [] for t in targets}
    unbounded: dict[str, list[dict]] = {t: [] for t in targets}

    for rel in relations:
        canon_subj = canonicalize(rel.get("subject", ""))
        sources    = _anchor_sources(rel)
        seed_tags  = _seed_tags(sources)
        worker_tags = _worker_tags(sources)

        # Workers uniquely-tied to one seed directory propagate that
        # seed's attribution to this relation.
        unique_worker_seeds: set[str] = set()
        for w in worker_tags:
            seeds = worker_to_seeds.get(w, set())
            if len(seeds) == 1:
                unique_worker_seeds |= seeds

        edge = relation_to_edge(rel)

        for target, cfg in targets.items():
            d1_match = canon_subj == cfg["canonical_subject"]
            ub_match = (
                d1_match
                or cfg["seed_dir"] in seed_tags
                or cfg["seed_dir"] in unique_worker_seeds
            )
            if d1_match:
                depth1[target].append(edge)
            if ub_match:
                unbounded[target].append(edge)
    return depth1, unbounded


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--merge-artifact", required=True, type=Path,
                   help="Path to ModSleuth's merged graph JSON (the 14,769-edge artifact).")
    p.add_argument("--out-dir", default=Path(__file__).resolve().parent.parent / "baselines" / "outputs",
                   type=Path,
                   help="Where to write prov_<target>.json + prov_unbounded_<target>.json.")
    args = p.parse_args()

    if not args.merge_artifact.exists():
        sys.exit(f"merge artifact not found: {args.merge_artifact}")
    G = json.loads(args.merge_artifact.read_text())
    relations = G.get("relations") or G.get("edges") or []
    if not relations:
        sys.exit("merge artifact has no `relations` field")
    print(f"Loaded {len(relations)} relations from {args.merge_artifact}", flush=True)

    depth1, unbounded = attribute_relations(relations, TARGET_CONFIG)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGET_CONFIG:
        d1_path = args.out_dir / f"prov_{target}.json"
        ub_path = args.out_dir / f"prov_unbounded_{target}.json"
        d1_path.write_text(json.dumps(
            {"subject": target, "edges": depth1[target]}, indent=1))
        ub_path.write_text(json.dumps(
            {"subject": target, "edges": unbounded[target]}, indent=1))
        print(f"  {target}: depth-1 {len(depth1[target]):>5} edges  →  {d1_path.name}", flush=True)
        print(f"  {target}: unbound {len(unbounded[target]):>5} edges  →  {ub_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
