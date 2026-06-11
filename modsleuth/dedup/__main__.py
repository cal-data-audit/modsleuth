#!/usr/bin/env python3
"""Dedup pipeline for LLM dependency graphs.

Reads a merged JSON graph (output of ``modsleuth run merge``), runs four
stages of dedup + filtering, and writes a cleaned JSON graph. Each stage
can be run in isolation:

    python -m modsleuth.dedup --source merge.json --dest clean.json --stages all
    python -m modsleuth.dedup --source merge.json --dest v1.json    --stages heuristic
    python -m modsleuth.dedup --source v1.json    --dest v2.json    --stages hub-audit
    python -m modsleuth.dedup --source v2.json    --dest v3.json    --stages node-dedup
    python -m modsleuth.dedup --source v3.json    --dest final.json --stages release

The four stages are conceptually independent:

    1. heuristic   — signature clustering + fuzzy surface-form merge (no LLM).
    2. hub-audit   — per-hub LLM audit; drops dup/hallucinated/vacuous edges.
    3. node-dedup  — whole-graph LLM-verified node merges; conflict-guarded union.
    4. release     — LLM classifies KEEP / DROP per node; rewires edges through
                     dropped intermediates along compatible relations.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock

from .lib import (
    DEFAULT_WORKERS,
    StageLogger,
    assert_invariants,
    call_opus,
    can_merge,
    collect_node_names,
    degree_map,
    is_categorical_drop,
    lex_collapse,
    load_graph,
    rebuild_lattice,
    rewrite_edges,
    run_parallel_llm,
    sample_anchors,
    save_graph,
    signature,
    split_org,
    tokenize,
    ConflictGuardedUnionFind,
)

# ============================================================
# Stage 1: Heuristic dedup (no LLM)
# ============================================================
def heuristic_dedup(graph: dict, log: StageLogger) -> dict:
    """Signature-based clustering + fuzzy surface-form merge. Pure heuristic."""
    log.log("=" * 70)
    log.log("STAGE 1: heuristic dedup")
    log.log("=" * 70)

    edges = graph.get("relations", [])
    groups = graph.get("lattice", {}).get("groups", [])
    all_nodes = collect_node_names(graph)
    deg = degree_map(graph)
    log.log(f"Input: {len(all_nodes):,} nodes, {len(edges):,} edges")

    # Categorical drops: internal paths and free-text descriptions.
    dropped: list[str] = [n for n in all_nodes if is_categorical_drop(n)]
    keep = {n for n in all_nodes if n not in set(dropped)}
    log.log(f"Categorical drops (internal paths / prose): {len(dropped):,}")

    # Bucket by full signature.
    sigs: dict[str, tuple] = {n: signature(n) for n in keep}
    clusters: dict[tuple, list[str]] = defaultdict(list)
    for n, sig in sigs.items():
        clusters[sig].append(n)

    # Fold no-org bare names into prefixed clusters with the same bare_collapsed,
    # picking the most-popular target if multiple prefixed candidates qualify.
    bare_to_prefixed: dict[str, list[tuple]] = defaultdict(list)
    for sig in clusters:
        if sig[0]:  # has org
            bare_to_prefixed[sig[2]].append(sig)  # key on bare_collapsed for fuzzy match

    final_clusters: dict[tuple, list[str]] = {}
    folded = 0
    for sig, names in clusters.items():
        if sig[0] is None:
            candidates = [c for c in bare_to_prefixed.get(sig[2], []) if can_merge(sig, c)]
            if candidates:
                # Most-popular target tiebreak: pick the prefixed cluster with highest aggregate degree.
                target = max(
                    candidates,
                    key=lambda c: sum(deg.get(n, 0) for n in clusters[c]),
                )
                final_clusters.setdefault(target, []).extend(names)
                folded += len(names)
                continue
        final_clusters.setdefault(sig, []).extend(names)
    log.log(f"No-org names folded into prefixed clusters: {folded:,}")
    log.log(f"Distinct clusters: {len(final_clusters):,}")

    # Pick canonical: prefer org/name HF form, then highest degree, then longest.
    canon_map: dict[str, str | None] = {}
    for sig, names in final_clusters.items():
        prefixed = [n for n in names if "/" in n and not n.startswith("/")]
        pool = prefixed if prefixed else names
        canonical = max(pool, key=lambda m: (deg.get(m, 0), len(m)))
        for n in names:
            canon_map[n] = canonical
    for n in dropped:
        canon_map[n] = None

    # Edge rewrite + anchor merge.
    new_edges, dropped_endpt, dropped_self = rewrite_edges(edges, canon_map)
    log.log(
        f"Edges: {len(edges):,} → {len(new_edges):,}  "
        f"(dropped: {dropped_endpt:,} endpoint, {dropped_self:,} self-loop)"
    )

    # Low-signal filter: short bare concept names with degree < 3.
    incoming: Counter = Counter()
    outgoing: Counter = Counter()
    for e in new_edges:
        outgoing[e["subject"]] += 1
        incoming[e["object"]] += 1

    def looks_concept(name: str) -> bool:
        if "/" in name or "[" in name or "(" in name:
            return False
        return len(name) < 30 and bool(re.match(r"^[A-Za-z][A-Za-z0-9\-\s_.]*$", name))

    low_signal = {n for n in (set(incoming) | set(outgoing)) if looks_concept(n) and (incoming[n] + outgoing[n]) < 3}
    new_edges = [e for e in new_edges if e["subject"] not in low_signal and e["object"] not in low_signal]
    log.log(f"Low-signal concept drops (degree < 3): {len(low_signal):,}")

    final_node_set = {n for e in new_edges for n in (e["subject"], e["object"])}
    out_groups = rebuild_lattice(groups, canon_map, final_node_set)

    out = {
        **graph,
        "lattice": {"groups": out_groups},
        "relations": new_edges,
    }
    log.log(f"Output: {len(final_node_set):,} nodes, {len(new_edges):,} edges\n")
    return out


# ============================================================
# Stage 2: LLM hub audit
# ============================================================
HUB_PROMPT = """You are a careful graph quality reviewer. Below is one HUB node and a batch of {direction} edges.

Think carefully. Identify edges that should be DROPPED for one of these reasons:

1. DUPLICATE — the same fact is asserted by another edge in this same batch via a different surface form.
2. HALLUCINATED — the relationship is implausible or impossible given the artifacts involved.
3. VACUOUS — implausibly generic objects with no information value ("data", "text corpus", bare concepts).
4. WRONG_RELATION — the relation type chosen is wrong (e.g. trained_on vs used_for_evaluation).

DO NOT drop edges merely because:
- they have only 1 anchor (1 anchor is normal)
- they describe ablations / distinct stages / subset variants
- the org/name has an unfamiliar prefix

OUTPUT FORMAT — strict, one decision per line, nothing else:
  DROP {{id}} :: {{TAG}} :: {{one-sentence reason}}
  KEEP_ALL  (only if every edge looks fine)

Where TAG is one of: DUPLICATE, HALLUCINATED, VACUOUS, WRONG_RELATION.

SUBGRAPH:
{subgraph}
"""


def _build_hub_text(hub: str, role: str, edges: list[dict], idx_map: dict[int, int]) -> str:
    """Format a hub's outgoing/incoming edges for an Opus call."""
    lines = [f"HUB ({role}-side): {hub}", "", f"Edges ({len(idx_map)}):"]
    for local_id, edge_id in idx_map.items():
        e = edges[edge_id]
        anchors = e.get("anchor_list") or []
        n_anch = len(anchors)
        first = (anchors[0].get("path") or anchors[0].get("source") or anchors[0].get("url") or "")[:120] if anchors else ""
        if role == "subject":
            line = f"  [{local_id:3d}] --[{e['relation']}]--> {e['object']}"
        else:
            line = f"  [{local_id:3d}] {e['subject']} --[{e['relation']}]-->"
        line += f"   anchors={n_anch}"
        if first:
            line += f"  src≈{first}"
        desc = (e.get("description") or "")[:160]
        if desc:
            line += f"\n           desc: {desc}"
        lines.append(line)
    return "\n".join(lines)


def _parse_hub_drops(text: str, idx_map: dict[int, int]) -> list[tuple[int, str, str]]:
    """Parse `DROP <id> :: TAG :: reason` lines. Returns (global_edge_id, tag, reason)."""
    out: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        m = re.match(r"^DROP\s+(\d+)\s*::\s*(\w+)\s*::\s*(.+)$", line.strip())
        if m:
            local_id = int(m.group(1))
            if local_id in idx_map:
                out.append((idx_map[local_id], m.group(2), m.group(3).strip()))
    return out


def llm_hub_audit(
    graph: dict,
    log: StageLogger,
    *,
    top_out: int = 75,
    top_in: int = 30,
    batch_size: int = 40,
    per_hub_cap: int = 200,
    workers: int = 12,
    effort: str = "max",
) -> dict:
    """Per-hub LLM audit of edges: drops duplicates, hallucinations, vacuous, wrong-relation."""
    log.log("=" * 70)
    log.log("STAGE 2: LLM hub audit")
    log.log("=" * 70)

    edges = graph["relations"]
    out_deg, in_deg = Counter(), Counter()
    edges_by_subject: dict[str, list[int]] = defaultdict(list)
    edges_by_object: dict[str, list[int]] = defaultdict(list)
    for ei, e in enumerate(edges):
        out_deg[e["subject"]] += 1
        in_deg[e["object"]] += 1
        edges_by_subject[e["subject"]].append(ei)
        edges_by_object[e["object"]].append(ei)

    out_hubs = [n for n, _ in out_deg.most_common(top_out)]
    in_hubs = [n for n, _ in in_deg.most_common(top_in)]
    log.log(f"Auditing top {top_out} out-hubs + top {top_in} in-hubs")

    def sort_by_anchors(idx_list: list[int]) -> list[int]:
        return sorted(idx_list, key=lambda i: -len(edges[i].get("anchor_list", []) or []))

    # Build batched jobs (hub, role, idx_list (capped, sorted), batch label).
    jobs: list[tuple[str, str, list[int], str]] = []
    for hub in out_hubs:
        idxs = sort_by_anchors(edges_by_subject[hub])[:per_hub_cap]
        for i in range(0, len(idxs), batch_size):
            jobs.append((hub, "subject", idxs[i:i + batch_size], f"OUT[{hub[:30]}].b{i // batch_size + 1}"))
    for hub in in_hubs:
        idxs = sort_by_anchors(edges_by_object[hub])[:per_hub_cap]
        for i in range(0, len(idxs), batch_size):
            jobs.append((hub, "object", idxs[i:i + batch_size], f"IN[{hub[:30]}].b{i // batch_size + 1}"))
    log.log(f"Total batches: {len(jobs)}\n")

    drop_lock = Lock()
    all_drops: dict[int, tuple[str, str]] = {}  # edge_id -> (tag, reason)

    def worker(job: tuple) -> tuple[int, str]:
        hub, role, idx_list, label = job
        # Local→global edge-id map (so the LLM sees small numbers).
        idx_map = {local: ei for local, ei in enumerate(idx_list)}
        sg = _build_hub_text(hub, role, edges, idx_map)
        prompt = HUB_PROMPT.format(direction=("outgoing" if role == "subject" else "incoming"), subgraph=sg)
        out, _ = call_opus(prompt, effort=effort)
        drops = _parse_hub_drops(out, idx_map)
        with drop_lock:
            for eid, tag, reason in drops:
                all_drops[eid] = (tag, reason)
        return len(drops), label

    run_parallel_llm(jobs, worker, max_workers=workers, label="hub")
    log.log(f"\nDropped: {len(all_drops):,} edges")
    tag_counts = Counter(t for t, _ in all_drops.values())
    for t, n in tag_counts.most_common():
        log.log(f"  {n:>4}  {t}")
    # Full decision trail (log file only): every dropped edge with its reason.
    for eid in sorted(all_drops):
        tag, reason = all_drops[eid]
        e = edges[eid]
        log.append(f"DROP-EDGE :: {tag} :: {e.get('subject')} "
                   f"--{e.get('relation')}--> {e.get('object')} :: {reason}")

    kept = [e for i, e in enumerate(edges) if i not in all_drops]
    final_node_set = {n for e in kept for n in (e["subject"], e["object"])}
    out_groups = [
        g
        for g in graph["lattice"]["groups"]
        if any(it.get("formal_name") in final_node_set for it in g.get("items", []))
    ]
    log.log(f"Output: {len(final_node_set):,} nodes, {len(kept):,} edges\n")
    return {**graph, "lattice": {"groups": out_groups}, "relations": kept}


# ============================================================
# Stage 3: LLM node dedup (whole-graph)
# ============================================================
NODE_PROMPT = """You are a strict graph dedup verifier. Below is a candidate cluster of node names that loose blocking flagged as possibly the same released artifact.

Decide ONE of (output exactly one line, no preamble):

ALL_SAME :: {{canonical_index}} :: {{brief reason}}
  → All N items refer to the same released artifact. canonical_index = 0-based index of canonical name.

PARTIAL :: {{merge_indices}} :: {{canonical_within}} :: {{brief reason}}
  → A subset is the same. merge_indices = comma-separated 0-based indices that should merge.
    canonical_within = index within those merge_indices (0..len-1).

ALL_DISTINCT :: {{brief reason}}
  → Every item is its own released artifact.

CRITICAL — be strict. NEVER merge across:
  - Different versions (3 vs 3.1, gpt-3.5 vs gpt-4)
  - Different sizes (7B vs 13B vs 32B)
  - Different stages (Base vs Instruct vs Chat — usually distinct released checkpoints)
  - Different release dates / API snapshots
  - Subsets vs parent (gpqa_diamond ≠ gpqa)
  - Distinct community re-releases (Open-Orca/FLAN ≠ SirNeural/flan_v2)

DO merge when:
  - Same artifact, different surface forms (casing, hyphenation): "MMLU" ↔ "cais/mmlu"
  - Same artifact across orgs that are actually the SAME: "OpenAI/GPT-2" ↔ "openai-community/gpt2"
  - Bare descriptive name ↔ HF leaf for same artifact

CANDIDATE CLUSTER ({n_items} items):
{cluster_text}
"""


def _build_node_clusters(
    nodes: set[str],
    deg: Counter,
    *,
    min_token_jaccard: float = 0.60,
    top_k_token: int = 3,
    max_cluster_size: int = 6,
) -> list[list[str]]:
    """Build candidate dedup clusters from multiple cheap signals.

    Signals (all high-precision, no transitive cluster growth):
      1. Lex-collapse blocks (alphanumeric-only key match)
      2. Token-Jaccard ≥ threshold (top-K candidates per node)
      3. Substring containment (bare name's lex contained in a prefixed name's lex)
      4. Cross-org bare-lex match (same bare-part lex across different orgs)
      5. Suffix-stripping pairs (X and X-{turbo, Instruct, Base, hf})
    """
    node_lex = {n: lex_collapse(n) for n in nodes}
    node_tokens = {n: tokenize(n) for n in nodes}
    node_tokens = {n: t for n, t in node_tokens.items() if t}

    clusters: list[list[str]] = []
    seen_keys: set[tuple] = set()

    def add_cluster(members: list[str]) -> None:
        if len(members) < 2:
            return
        members = sorted(set(members), key=lambda x: -deg.get(x, 0))[:max_cluster_size]
        if len(members) < 2:
            return
        key = tuple(sorted(members))
        if key not in seen_keys:
            seen_keys.add(key)
            clusters.append(members)

    # Signal 1: lex-collapse blocks.
    lex_groups: dict[str, list[str]] = defaultdict(list)
    for n, k in node_lex.items():
        if k:
            lex_groups[k].append(n)
    for members in lex_groups.values():
        if 2 <= len(members) <= max_cluster_size:
            add_cluster(members)
        elif len(members) > max_cluster_size:
            v_sorted = sorted(members, key=lambda x: -deg.get(x, 0))
            anchor, rest = v_sorted[0], v_sorted[1:]
            for i in range(0, len(rest), max_cluster_size - 1):
                add_cluster([anchor] + rest[i:i + max_cluster_size - 1])

    # Signal 2: token-Jaccard top-K per node.
    token_to_nodes: dict[str, list[str]] = defaultdict(list)
    for n, toks in node_tokens.items():
        for t in toks:
            token_to_nodes[t].append(n)

    def jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    for n, toks in node_tokens.items():
        cnt: Counter = Counter()
        for t in toks:
            post = token_to_nodes[t]
            if len(post) > 200:
                continue
            for n2 in post:
                if n2 != n:
                    cnt[n2] += 1
        cands = []
        for n2, shared in cnt.most_common(20):
            if shared < 2:
                break
            j = jaccard(toks, node_tokens.get(n2, frozenset()))
            if j >= min_token_jaccard:
                cands.append(n2)
        if cands:
            add_cluster([n] + cands[:top_k_token])

    # Signal 3: substring containment (bare → prefixed).
    prefixed = [n for n in nodes if "/" in n and not n.startswith("/")]
    prefixed_lex = {n: lex_collapse(n.split("/", 1)[1]) for n in prefixed}
    for short in nodes:
        if "/" in short:
            continue
        sl = node_lex.get(short, "")
        if len(sl) < 4:
            continue
        matches = [p for p, pl in prefixed_lex.items() if pl == sl or pl.startswith(sl) or pl.endswith(sl)]
        if matches:
            matches.sort(key=lambda x: -deg.get(x, 0))
            add_cluster([short] + matches[:max_cluster_size - 1])

    # Signal 4: cross-org bare-lex match.
    by_bare_lex: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for n in nodes:
        org, bare = split_org(n)
        bl = lex_collapse(bare)
        if bl:
            by_bare_lex[bl].append((org, n))
    for members in by_bare_lex.values():
        orgs = {org for org, _ in members if org}
        bare_only = [n for org, n in members if org is None]
        if len(orgs) >= 2 or (len(orgs) >= 1 and bare_only):
            names = sorted({n for _, n in members}, key=lambda x: -deg.get(x, 0))
            if 2 <= len(names) <= max_cluster_size:
                add_cluster(names)

    # Signal 5: suffix-stripping pairs.
    suffixes = ("-turbo", "-Turbo", "-Instruct", "-instruct", "-Base", "-base",
                "-Chat", "-chat", "-it", "-IT", "-hf", "-HF", "-preview", "-Preview")
    for n in nodes:
        for suf in suffixes:
            if n.endswith(suf):
                stripped = n[:-len(suf)]
                if stripped in nodes:
                    add_cluster([stripped, n])
                break

    return clusters


def _parse_node_verdict(text: str, n_items: int) -> tuple[str, dict]:
    line = text.splitlines()[0].strip() if text else ""
    if line.startswith("ALL_SAME"):
        m = re.match(r"ALL_SAME\s*::\s*(\d+)\s*::\s*(.*)", line)
        if m and 0 <= int(m.group(1)) < n_items:
            return ("ALL_SAME", {"canonical_idx": int(m.group(1)), "reason": m.group(2)})
    elif line.startswith("PARTIAL"):
        m = re.match(r"PARTIAL\s*::\s*([\d,\s]+)\s*::\s*(\d+)\s*::\s*(.*)", line)
        if m:
            try:
                idxs = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
                idxs = [i for i in idxs if 0 <= i < n_items]
                ci = int(m.group(2))
                if len(idxs) >= 2 and 0 <= ci < len(idxs):
                    return ("PARTIAL", {"merge_indices": idxs, "canonical_within": ci, "reason": m.group(3)})
            except Exception:
                pass
    elif line.startswith("ALL_DISTINCT"):
        m = re.match(r"ALL_DISTINCT\s*::\s*(.*)", line)
        return ("ALL_DISTINCT", {"reason": m.group(1) if m else ""})
    return ("UNPARSED", {"raw": text[:300]})


def llm_node_dedup(
    graph: dict,
    log: StageLogger,
    *,
    workers: int = DEFAULT_WORKERS,
    effort: str = "max",
) -> dict:
    """Whole-graph node dedup via cheap candidate generation + LLM verification + conflict-guarded union."""
    log.log("=" * 70)
    log.log("STAGE 3: LLM node dedup")
    log.log("=" * 70)

    edges = graph["relations"]
    nodes = collect_node_names(graph)
    deg = degree_map(graph)
    log.log(f"Input: {len(nodes):,} nodes, {len(edges):,} edges")

    # Build candidate clusters from multiple signals.
    clusters = _build_node_clusters(nodes, deg)
    size_dist = Counter(len(c) for c in clusters)
    log.log(f"Candidate clusters: {len(clusters):,}")
    for sz in sorted(size_dist):
        log.log(f"  size={sz}: {size_dist[sz]:,}")

    anchors = sample_anchors(graph)

    # Build prompts; verify each cluster with Opus.
    def build_text(members: list[str]) -> str:
        lines = []
        for i, n in enumerate(members):
            d = deg.get(n, 0)
            a = anchors.get(n, "(none)")
            lines.append(f"  [{i}] {n}    degree={d}    sample_anchor={a}")
        return "\n".join(lines)

    def worker(cluster: list[str]) -> tuple[str, dict, list[str]]:
        prompt = NODE_PROMPT.format(n_items=len(cluster), cluster_text=build_text(cluster))
        out, _ = call_opus(prompt, effort=effort)
        tag, payload = _parse_node_verdict(out, len(cluster))
        return tag, payload, cluster

    log.log(f"\nLaunching {len(clusters)} verifications with {workers} workers...\n")
    results = run_parallel_llm(clusters, worker, max_workers=workers, label="node-dedup")

    tag_counts = Counter(r[0] for r in results.values())
    log.log(f"Verdicts: {dict(tag_counts)}")

    # Apply with conflict-guarded union-find.
    cguf = ConflictGuardedUnionFind(nodes)
    for tag, payload, cluster in results.values():
        if tag == "ALL_SAME":
            canonical = cluster[payload["canonical_idx"]]
            for m in cluster:
                cguf.try_unite(m, canonical)
        elif tag == "PARTIAL":
            merge_subset = [cluster[i] for i in payload["merge_indices"]]
            canonical = merge_subset[payload["canonical_within"]]
            for m in merge_subset:
                cguf.try_unite(m, canonical)
    log.log(f"Conflict-guarded unions skipped: {len(cguf.skipped)}")

    # Build canon_map: each node -> highest-degree (prefixed-form-preferred) member of its component.
    canon_map: dict[str, str | None] = {}
    nodes_merged = 0
    for root, members in cguf.components().items():
        if len(members) <= 1:
            canon_map[members[0]] = members[0]
            continue
        prefixed = [m for m in members if "/" in m and not m.startswith("/")]
        pool = prefixed if prefixed else members
        canonical = max(pool, key=lambda m: (deg.get(m, 0), len(m)))
        log.append(f"MERGE-NODES :: {canonical} <= "
                   f"{sorted(m for m in members if m != canonical)}")
        for m in members:
            canon_map[m] = canonical
            if m != canonical:
                nodes_merged += 1
    log.log(f"Nodes merged: {nodes_merged}")

    new_edges, dropped_endpt, dropped_self = rewrite_edges(edges, canon_map)
    log.log(f"Edges: {len(edges):,} → {len(new_edges):,}  (dropped {dropped_self} self-loops)")

    final_node_set = {n for e in new_edges for n in (e["subject"], e["object"])}
    out_groups = rebuild_lattice(graph["lattice"]["groups"], canon_map, final_node_set)
    log.log(f"Output: {len(final_node_set):,} nodes, {len(new_edges):,} edges\n")

    return {**graph, "lattice": {"groups": out_groups}, "relations": new_edges}


# ============================================================
# Stage 4: Release-only filter
# ============================================================
RELEASE_PROMPT = """You are filtering a graph of LLM dependencies down to OFFICIALLY RELEASED artifacts only. For each node listed below, classify it.

KEEP — the node is one of:
  - An officially released model checkpoint (HF org/name with public weights)
  - An officially released dataset (e.g., cais/mmlu, openai/gsm8k, Common Crawl)
  - A standard benchmark / evaluation suite (MMLU, GSM8K, AIME, HumanEval, BBH)
  - A well-known third-party API model (openai/gpt-4o, anthropic/claude-sonnet-4)

DROP — the node is one of:
  - Training-stage checkpoints ("Stage 2", "Ingredient 1", "Soup", "midtraining run 2")
  - Specific-step pretraining checkpoints ("step 10000")
  - Internal research data variants / preference-mix deltas
  - Bracket-tagged research metadata when a released sibling exists
  - Generic concept aliases ("Safety", "GPT", "olmo 3", "Llama" without size/version)
  - Experimental / preview / distill-student variants when not actually released
  - Off-lattice prose descriptions

For each item, output exactly one line:
  KEEP <id> :: <one-phrase reason>
  DROP <id> :: <one-phrase reason>

NODES TO CLASSIFY ({n_items}):
{node_list}
"""


# Compatible relation pairs for transitive edge rewiring through dropped nodes.
# Every relation this table can emit (trained_from / trained_on / filtered_by)
# is a *direct* dependency; rewired edges carry that kind.
COMPATIBLE_REWIRE = {
    ("trained_from", "trained_from"): "trained_from",
    ("trained_from", "merged_from"): "trained_from",
    ("merged_from", "trained_from"): "trained_from",
    ("trained_on", "trained_on"): "trained_on",
    ("filtered_by", "filtered_by"): "filtered_by",
}


def release_filter(
    graph: dict,
    log: StageLogger,
    *,
    batch_size: int = 20,
    workers: int = DEFAULT_WORKERS,
    effort: str = "high",
) -> dict:
    """LLM classifies KEEP/DROP per node. Drops intermediates, rewires edges through them."""
    log.log("=" * 70)
    log.log("STAGE 4: release-only filter")
    log.log("=" * 70)

    edges = graph["relations"]
    nodes = sorted(collect_node_names(graph))
    deg = degree_map(graph)
    anchors = sample_anchors(graph)
    log.log(f"Input: {len(nodes):,} nodes, {len(edges):,} edges")

    # Build batches.
    batches = [(i, nodes[i:i + batch_size]) for i in range(0, len(nodes), batch_size)]
    log.log(f"Classification batches: {len(batches)} ({batch_size} nodes each)")

    def build_node_list(start: int, members: list[str]) -> str:
        out = []
        for i, n in enumerate(members):
            out.append(f"  [{start + i}] {n}    (degree={deg.get(n, 0)}, sample_anchor={anchors.get(n, '(none)')})")
        return "\n".join(out)

    classifications: dict[str, tuple[str, str]] = {}
    cls_lock = Lock()

    def worker(job: tuple[int, list[str]]) -> int:
        start, members = job
        prompt = RELEASE_PROMPT.format(n_items=len(members), node_list=build_node_list(start, members))
        out, _ = call_opus(prompt, effort=effort)
        decisions: dict[str, tuple[str, str]] = {}
        for line in out.splitlines():
            m = re.match(r"^(KEEP|DROP)\s+(\d+)\s*::\s*(.*)$", line.strip())
            if not m:
                continue
            verdict, gid, reason = m.group(1), int(m.group(2)), m.group(3).strip()
            local = gid - start
            if 0 <= local < len(members):
                decisions[members[local]] = (verdict, reason)
        # Default unparsed nodes to KEEP for safety.
        for n in members:
            if n not in decisions:
                decisions[n] = ("KEEP", "(no verdict — default keep for safety)")
        with cls_lock:
            classifications.update(decisions)
        return len(decisions)

    run_parallel_llm(batches, worker, max_workers=workers, label="release")
    counts = Counter(v for v, _ in classifications.values())
    log.log(f"Verdicts: {dict(counts)}")

    keep_set = {n for n, (v, _) in classifications.items() if v == "KEEP"}
    drop_set = set(nodes) - keep_set
    log.log(f"KEEP: {len(keep_set):,}  ·  DROP: {len(drop_set):,}")
    # Full decision trail (log file only): every dropped node with its reason.
    for n in sorted(drop_set):
        _, reason = classifications.get(n, ("DROP", "(unclassified)"))
        log.append(f"DROP-NODE :: {n} :: {reason}")

    # Rewire compatible chains through dropped nodes.
    edges_in: dict[str, list[tuple[str, int]]] = defaultdict(list)
    edges_out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for ei, e in enumerate(edges):
        s, o = e.get("subject"), e.get("object")
        if s and o:
            edges_in[o].append((s, ei))
            edges_out[s].append((o, ei))

    rewired: list[dict] = []
    seen: set[tuple] = set()
    for x in drop_set:
        for s, ei_in in edges_in.get(x, []):
            if s in drop_set:
                continue
            r_in = edges[ei_in].get("relation")
            for o, ei_out in edges_out.get(x, []):
                if o in drop_set:
                    continue
                r_out = edges[ei_out].get("relation")
                new_rel = COMPATIBLE_REWIRE.get((r_in, r_out))
                if not new_rel:
                    continue
                key = (s, new_rel, o)
                if key in seen:
                    continue
                seen.add(key)
                anch_in = edges[ei_in].get("anchor_list") or []
                anch_out = edges[ei_out].get("anchor_list") or []
                rewired.append({
                    "subject": s,
                    "relation": new_rel,
                    "object": o,
                    "dependency_kind": "direct",
                    "description": f"(rewired through {x})",
                    "anchor_list": (anch_in + anch_out)[:6],
                    "description_variants": [],
                })
    log.log(f"Rewired edges (transitive through DROPs): {len(rewired):,}")

    # Drop edges with dropped endpoints; add rewired edges.
    new_edges_d: dict[tuple, dict] = {}
    for e in edges:
        s, o, rel = e.get("subject"), e.get("object"), e.get("relation", "")
        if s in keep_set and o in keep_set and rel:
            key = (s, rel, o)
            if key not in new_edges_d:
                new_edges_d[key] = {**e,
                                    "anchor_list": list(e.get("anchor_list") or []),
                                    "description_variants": list(e.get("description_variants") or [])}
            else:
                new_edges_d[key]["anchor_list"].extend(e.get("anchor_list") or [])
    for re_edge in rewired:
        key = (re_edge["subject"], re_edge["relation"], re_edge["object"])
        if key not in new_edges_d:
            new_edges_d[key] = re_edge

    new_edges = list(new_edges_d.values())
    final_node_set = {n for e in new_edges for n in (e["subject"], e["object"])}
    out_groups = [g for g in graph["lattice"]["groups"]
                  if any(it.get("formal_name") in final_node_set for it in g.get("items", []))]
    log.log(f"Output: {len(final_node_set):,} nodes, {len(new_edges):,} edges\n")

    return {**graph, "lattice": {"groups": out_groups}, "relations": new_edges}


# ============================================================
# Pipeline orchestration
# ============================================================
STAGES: dict[str, callable] = {
    "heuristic": heuristic_dedup,
    "hub-audit": llm_hub_audit,
    "node-dedup": llm_node_dedup,
    "release": release_filter,
}


def run_dedup(source: str | Path, dest: str | Path,
              stages: str = "all", log_path: str | Path | None = None,
              protect: tuple[str, ...] = (),
              model: str | None = None) -> int:
    """Run the dedup pipeline. Returns process exit code (0 on success, 2 on bad stage).

    ``protect`` lists substrings (typically the run's seed identifiers)
    that must still be present in the node set after every stage —
    a guard against a stage collapsing or dropping a seed node.
    ``model`` overrides the LLM used by the hub-audit / node-dedup /
    release stages (default: Opus; also settable via
    MODSLEUTH_DEDUP_MODEL).
    """
    if model:
        os.environ["MODSLEUTH_DEDUP_MODEL"] = model
    stage_names = list(STAGES) if stages == "all" else [s.strip() for s in stages.split(",")]
    for s in stage_names:
        if s not in STAGES:
            print(f"Unknown stage: {s!r}; valid stages: {', '.join(STAGES)}", file=sys.stderr)
            return 2

    src, dst = Path(source), Path(dest)
    if src.resolve() == dst.resolve():
        print("--dest must differ from --source (refusing in-place overwrite)",
              file=sys.stderr)
        return 2
    if log_path is None:
        log_path = f"{dst}.log"
    log = StageLogger(Path(log_path))
    log.log(f"Source: {src}")
    log.log(f"Dest:   {dst}")
    log.log(f"Stages: {' → '.join(stage_names)}\n")

    t_start = time.time()
    graph = load_graph(src)
    for i, s in enumerate(stage_names):
        graph = STAGES[s](graph, log)
        # Protected nodes must survive *every* stage, and each stage's
        # output is checkpointed so a crash later never loses its work.
        assert_invariants(collect_node_names(graph), tuple(protect))
        if i < len(stage_names) - 1:
            ckpt = dst.with_name(f"{dst.stem}.after-{s}{dst.suffix or '.json'}")
            save_graph(graph, ckpt)
            log.log(f"[checkpoint] {ckpt} — resume with "
                    f"--source {ckpt} --stages <remaining>")

    if dst.exists():
        bak = dst.with_name(dst.name + ".bak")
        os.replace(dst, bak)
        log.log(f"[backup] previous dest preserved at {bak}")
    save_graph(graph, dst)
    log.log(f"\n✓ Wrote {dst} ({dst.stat().st_size:,} bytes) in {(time.time() - t_start) / 60:.1f} min")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Dedup pipeline for LLM dependency graphs.")
    p.add_argument("--source", required=True, help="Path to input JSON (the merged graph).")
    p.add_argument("--dest", required=True, help="Path to output JSON.")
    p.add_argument(
        "--stages",
        default="all",
        help=f"Comma-separated stages, or 'all'. Available: {', '.join(STAGES)}",
    )
    p.add_argument("--log", default=None,
                   help="Log file path (default: <dest>.log; appended, never truncated).")
    p.add_argument(
        "--protect", action="append", default=[],
        help="Substring that must survive every stage (repeatable; "
             "typically the run's seed identifiers).",
    )
    p.add_argument(
        "--model", default=None,
        help="LLM for the hub-audit / node-dedup / release stages "
             "(default: Opus; also via MODSLEUTH_DEDUP_MODEL).",
    )
    args = p.parse_args()
    return run_dedup(args.source, args.dest, args.stages, args.log,
                     protect=tuple(args.protect), model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
