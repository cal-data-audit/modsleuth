#!/usr/bin/env python3
"""Reproduce paper Tables 2, 4, and 5 from the merged ModSleuth graph.

This is the exact script used to compute the numbers in the paper. It
reads ``data/merge_artifact.json`` (the 14,769-edge ModSleuth merged
graph) and prints:

  * **Table 2** — Recovered dependency edges grouped by audit role and
    split by edge-level dependency kind. Counted directly from each
    relation's ``relation`` and ``dependency_kind`` fields.
  * **Table 4** — Per-target ancestor counts and max depths via BFS
    over the upstream-edge subset (every relation type *except*
    ``used_for_evaluation``). Each target uses one canonical seed node
    (`TABLE4_TARGETS` below).
  * **Table 5** — Source-type distribution of recovered operations.
    Each relation's ``anchor_list`` source paths are classified into
    {hf_card, code, pdf, blog, doc_other}; relations whose anchors fall
    in exactly one bucket count under "Only <bucket>", everything else
    counts as "Multiple source types".

Usage:

    python compute_graph_stats.py --merge-artifact ../data/merge_artifact.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


# ─── Table 2 grouping ─────────────────────────────────────────────────

AUDIT_ROLE_GROUPS: list[tuple[str, list[str]]] = [
    ("Training / evaluation inputs",     ["trained_on", "used_for_evaluation", "used_for_ablation"]),
    ("Upstream model operations on data", ["generated_by", "filtered_by", "transformed_by", "embedded_by"]),
    ("Weight-level model lineage",        ["trained_from", "merged_from", "quantized_from"]),
    ("Methodology / audit influence",     ["inspired_by", "decontaminated_against"]),
]


def table2(relations: list[dict]) -> str:
    total = len(relations)
    by_pair = Counter(
        (r["relation"], r.get("dependency_kind", "?")) for r in relations
    )

    lines = [
        f"Table 2 — Recovered dependency edges by audit role  (over {total} edges)",
        "",
        f"{'Audit role':<38} {'Direct':>7} {'Indirect':>9} {'Count':>6} {'Share':>7}",
        "-" * 72,
    ]
    grand_d = grand_i = 0
    for role, rel_types in AUDIT_ROLE_GROUPS:
        d = sum(by_pair.get((rt, "direct"),   0) for rt in rel_types)
        i = sum(by_pair.get((rt, "indirect"), 0) for rt in rel_types)
        n = d + i
        share = n / total if total else 0.0
        grand_d += d; grand_i += i
        lines.append(f"{role:<38} {d:>7} {i:>9} {n:>6} {share*100:>6.1f}%")
    grand_total = grand_d + grand_i
    lines.append("-" * 72)
    lines.append(f"{'Total':<38} {grand_d:>7} {grand_i:>9} {grand_total:>6} {'100.0%':>7}")
    return "\n".join(lines)


# ─── Table 4 BFS configuration ────────────────────────────────────────

# Per-target canonical seed node. These are the exact seeds used to
# compute Table 4 in the paper.
TABLE4_TARGETS: list[tuple[str, str]] = [
    ("Olmo-3-Instruct",      "allenai/Olmo-3-7B-Instruct"),
    ("Olmo-3-Think",         "allenai/Olmo-3-7B-Think"),
    ("Olmo-3-Base",          "allenai/Olmo-3-1025-7B"),
    ("Nemotron-3-Super",     "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"),
    ("Nemotron-3-Nano-Base", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"),
    ("DR-Tulu",              "rl-research/dr-tulu-8b"),
    ("SmolLM3-Base",         "HuggingFaceTB/SmolLM3-3B-Base"),
]

# Upstream relations: subject is downstream, object is upstream/ancestor.
# We exclude ``used_for_evaluation`` because evaluating on a benchmark
# doesn't make the benchmark an ancestor.
UPSTREAM_RELS = {
    "trained_from", "trained_on", "generated_by", "filtered_by",
    "transformed_by", "inspired_by", "used_for_ablation",
    "merged_from", "decontaminated_against", "embedded_by", "quantized_from",
}


def table4(relations: list[dict]) -> str:
    adj: dict[str, list[str]] = defaultdict(list)
    for r in relations:
        if r.get("relation") in UPSTREAM_RELS:
            adj[r["subject"]].append(r["object"])
    # Distinct subjects+objects across all relations (paper's "node" count).
    all_nodes = {r["subject"] for r in relations} | {r["object"] for r in relations}

    def bfs(seed: str) -> tuple[int | None, int | None]:
        if seed not in all_nodes:
            return None, None
        seen = {seed: 0}
        q = deque([seed])
        while q:
            n = q.popleft()
            d = seen[n]
            for nx in adj.get(n, []):
                if nx not in seen:
                    seen[nx] = d + 1
                    q.append(nx)
        if len(seen) <= 1:
            return 0, 0
        return len(seen) - 1, max(seen.values())

    lines = [
        "Table 4 — Per-target ancestor counts and max depths",
        "",
        f"{'Target':<25} {'Ancestors':>10} {'Max depth':>10}",
        "-" * 50,
    ]
    for name, seed in TABLE4_TARGETS:
        anc, md = bfs(seed)
        if anc is None:
            lines.append(f"{name:<25} {'(not found)':>10}")
        else:
            lines.append(f"{name:<25} {anc:>10} {md:>10}")
    lines.append("-" * 50)
    n_nodes   = len(all_nodes)
    n_edges   = len(relations)
    n_anchors = sum(len(r.get("anchor_list") or []) for r in relations)
    lines.append(f"Aggregate graph: {n_nodes} nodes, {n_edges} edges, {n_anchors} anchors")
    return "\n".join(lines)


# ─── Table 5 source-type classification ───────────────────────────────


def classify_source(s) -> str | None:
    """Match the hero-run classifier exactly.

    Returns one of: hf_card, pdf, code, blog, doc_other (or None for
    empty/null sources).
    """
    if not s:
        return None
    sl = str(s).lower()
    if "huggingface.co" in sl:
        return "hf_card"
    if "arxiv.org" in sl or sl.endswith(".pdf"):
        return "pdf"
    if (
        "github.com" in sl
        or any(sl.endswith(ext) for ext in (".py", ".yaml", ".yml", ".json", ".sh", ".md"))
        or "/blob/" in sl
    ):
        if "readme" in sl:
            return "doc_other"
        if any(p in sl for p in ("/blog", "blog/", "/posts", "/news/", "release-notes", "announcement")):
            return "blog"
        return "code"
    if any(p in sl for p in ("/blog", "blog/", "/posts", "/news/", "/announcement", "release-notes")):
        return "blog"
    return "doc_other"


_TABLE5_LABELS = [
    ("hf_card",   "Only Hugging Face cards"),
    ("code",      "Only training code / scripts"),
    ("pdf",       "Only PDFs / technical reports"),
    ("blog",      "Only release blogs / docs"),
    ("doc_other", "Only other docs"),
    ("multi",     "Multiple source types"),
]


def table5(relations: list[dict]) -> str:
    counts: Counter = Counter()
    for r in relations:
        classes: set[str] = set()
        for a in (r.get("anchor_list") or []):
            if not isinstance(a, dict):
                continue
            src = a.get("source") or a.get("url") or a.get("path")
            c = classify_source(src)
            if c:
                classes.add(c)
        if not classes:
            continue
        if len(classes) == 1:
            counts[next(iter(classes))] += 1
        else:
            counts["multi"] += 1
    total = sum(counts.values())

    lines = [
        f"Table 5 — Source-type distribution of recovered operations  (over {total} operations with anchors)",
        "",
        f"{'Source support':<35} {'Count':>6} {'Share':>7}",
        "-" * 55,
    ]
    for key, label in _TABLE5_LABELS:
        n = counts.get(key, 0)
        share = n / total if total else 0.0
        lines.append(f"{label:<35} {n:>6} {share*100:>6.1f}%")
    lines.append("-" * 55)
    lines.append(f"{'Total operations':<35} {total:>6} {'100.0%':>7}")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--merge-artifact", required=True, type=Path,
                   help="Path to ModSleuth's merged graph JSON.")
    p.add_argument("--tables", default="2,4,5",
                   help="Comma-separated subset of {2,4,5} to print.")
    args = p.parse_args()

    if not args.merge_artifact.exists():
        sys.exit(f"merge artifact not found: {args.merge_artifact}")
    G = json.loads(args.merge_artifact.read_text())
    relations = G.get("relations") or []
    if not relations:
        sys.exit("merge artifact has no `relations` field")

    wanted = {t.strip() for t in args.tables.split(",") if t.strip()}
    if "2" in wanted:
        print(table2(relations));  print()
    if "4" in wanted:
        print(table4(relations));  print()
    if "5" in wanted:
        print(table5(relations));  print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
