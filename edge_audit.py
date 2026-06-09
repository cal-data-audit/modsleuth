#!/usr/bin/env python3
"""Audit edge-count patterns — find residual noise after dedup.

Checks performed:
  1. Edge anchor distribution (zero-anchor edges, single-anchor edges, etc.)
  2. Top hubs by outgoing / incoming edges
  3. Per-relation anchor and degree breakdown
  4. Near-duplicate object names that survived dedup
  5. Near-duplicate subject names that survived dedup
  6. Suspect concept-only edges (both endpoints are short bare names)
  7. Weak-evidence claims (≤1 anchor in a high-frequency relation)
  8. (subject, object) pairs with multiple distinct relations

Run:  python edge_audit.py --source path/to/merge_artifact.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
p.add_argument("--source", required=True, type=Path, help="Path to a merged graph JSON.")
args = p.parse_args()

if not args.source.exists():
    print(f"ERROR: source not found: {args.source}", file=sys.stderr)
    sys.exit(1)

GRAPH = json.loads(args.source.read_text())
edges = GRAPH["relations"]
groups = GRAPH["lattice"]["groups"]

print(f"Source: {args.source}")
print(f"Total edges: {len(edges):,}")
print(f"Total lattice groups: {len(groups):,}\n")

# ============================================================
# 1. Edge anchor distribution
# ============================================================
print("="*60)
print("1. ANCHOR COVERAGE PER EDGE")
print("="*60)
anchor_counts = Counter()
zero_anchor_edges = []
for e in edges:
    n = len(e.get("anchor_list", []) or [])
    anchor_counts[n] += 1
    if n == 0: zero_anchor_edges.append(e)

print(f"Total anchors across all edges: {sum(n*c for n,c in anchor_counts.items()):,}")
print(f"Edges with 0 anchors: {anchor_counts[0]:,}")
print(f"Edges with 1 anchor:  {anchor_counts[1]:,}")
print(f"Edges with 2-3 anchors: {sum(c for n,c in anchor_counts.items() if 2 <= n <= 3):,}")
print(f"Edges with 4+ anchors: {sum(c for n,c in anchor_counts.items() if n >= 4):,}")

if zero_anchor_edges:
    print("\nSample zero-anchor edges (potential hallucinations):")
    for e in zero_anchor_edges[:10]:
        print(f"  {e['subject'][:40]} --[{e['relation']}]--> {e['object'][:40]}")

# ============================================================
# 2. Hub analysis — top out-degree and in-degree nodes
# ============================================================
print("\n" + "="*60)
print("2. HUBS (high degree)")
print("="*60)
out_deg = Counter()
in_deg = Counter()
for e in edges:
    out_deg[e["subject"]] += 1
    in_deg[e["object"]] += 1

print("\nTOP 25 OUT-degree (most outgoing edges):")
for n, d in out_deg.most_common(25):
    print(f"  {d:>5}  {n[:80]}")

print("\nTOP 25 IN-degree (most incoming edges):")
for n, d in in_deg.most_common(25):
    print(f"  {d:>5}  {n[:80]}")

# ============================================================
# 3. Per-relation breakdown
# ============================================================
print("\n" + "="*60)
print("3. PER-RELATION ANCHOR / DEGREE BREAKDOWN")
print("="*60)
rel_data = defaultdict(lambda: {"count": 0, "zero_anchor": 0, "total_anchors": 0, "uniq_subjects": set(), "uniq_objects": set()})
for e in edges:
    r = e["relation"]
    n = len(e.get("anchor_list", []) or [])
    rel_data[r]["count"] += 1
    if n == 0: rel_data[r]["zero_anchor"] += 1
    rel_data[r]["total_anchors"] += n
    rel_data[r]["uniq_subjects"].add(e["subject"])
    rel_data[r]["uniq_objects"].add(e["object"])

print(f"\n{'RELATION':<25} {'COUNT':>6} {'0-ANCHOR':>9} {'AVG_ANCH':>9} {'#SUBJ':>6} {'#OBJ':>6}")
for r in sorted(rel_data, key=lambda k: -rel_data[k]["count"]):
    d = rel_data[r]
    avg = d["total_anchors"] / max(d["count"],1)
    print(f"  {r:<23} {d['count']:>6,} {d['zero_anchor']:>9,} {avg:>8.2f}   {len(d['uniq_subjects']):>5,} {len(d['uniq_objects']):>5,}")

# ============================================================
# 4. Near-duplicate detection on objects (un-merged dupes)
# ============================================================
print("\n" + "="*60)
print("4. NEAR-DUPLICATE OBJECTS (un-merged dataset/benchmark dupes)")
print("="*60)

def lex_norm(s):
    """Aggressive normalization for lexical near-dupe detection."""
    s = re.sub(r"\([^)]*\)", "", s)  # strip parens
    s = re.sub(r"\[[^\]]*\]", "", s)  # strip brackets
    s = re.sub(r"[^a-z0-9]+", "", s.lower())  # collapse to alphanum lowercase
    return s

# Group object names by lex-normalized form
obj_lex = defaultdict(set)
for e in edges:
    obj = e["object"]
    obj_lex[lex_norm(obj)].add(obj)

# Surface dupe objects — same lex-norm but different surface
print("\nObjects that share lex-normalized form but have multiple surface names:")
print("(These are candidates for further merging that the dedup missed.)\n")
shown = 0
for lex, names in sorted(obj_lex.items(), key=lambda x: -len(x[1])):
    if len(names) <= 1: continue
    if shown >= 30: break
    # Compute total degree
    total = sum(in_deg[n] for n in names)
    print(f"  [{total} edges total] '{lex}'")
    for n in sorted(names, key=lambda x: -in_deg[x])[:6]:
        print(f"     [{in_deg[n]:>4}] {n[:80]}")
    shown += 1

# Subject-side
print("\n" + "="*60)
print("5. NEAR-DUPLICATE SUBJECTS (un-merged model dupes)")
print("="*60)
subj_lex = defaultdict(set)
for e in edges:
    subj_lex[lex_norm(e["subject"])].add(e["subject"])

shown = 0
for lex, names in sorted(subj_lex.items(), key=lambda x: -len(x[1])):
    if len(names) <= 1: continue
    if shown >= 30: break
    total = sum(out_deg[n] for n in names)
    print(f"  [{total} edges total] '{lex}'")
    for n in sorted(names, key=lambda x: -out_deg[x])[:6]:
        print(f"     [{out_deg[n]:>4}] {n[:80]}")
    shown += 1

# ============================================================
# 6. Edges where subject and object are both "concept-like"
# ============================================================
print("\n" + "="*60)
print("6. SUSPECT: edges where both endpoints are bare concept names")
print("="*60)
def is_concept(n):
    if not n: return False
    if "/" in n: return False
    if "[" in n: return False
    if "(" in n: return False
    if any(c.isdigit() and any(s in n for s in ["B","b"]) for c in n): return False
    return len(n) < 30

bare_edges = [e for e in edges if is_concept(e["subject"]) and is_concept(e["object"])]
print(f"\nEdges where both subject AND object are short bare names: {len(bare_edges):,}")
for e in bare_edges[:15]:
    n = len(e.get('anchor_list',[]) or [])
    print(f"  [a={n}] {e['subject']:<30} --[{e['relation']}]--> {e['object']}")

# ============================================================
# 7. Single-anchor low-frequency relations (potential weak claims)
# ============================================================
print("\n" + "="*60)
print("7. WEAK CLAIMS (1 anchor, rare relation)")
print("="*60)
rel_counts = Counter(e["relation"] for e in edges)
weak = []
for e in edges:
    a = len(e.get("anchor_list",[]) or [])
    if a <= 1 and rel_counts[e["relation"]] >= 100:
        weak.append(e)
print(f"Edges with ≤1 anchor in a common relation: {len(weak):,}")
print("Sample:")
for e in weak[:10]:
    a = len(e.get('anchor_list',[]) or [])
    print(f"  [a={a}] {e['subject'][:40]:<40} --[{e['relation']}]--> {e['object'][:40]}")

# ============================================================
# 8. Anchors per (subject, object) pair (regardless of relation)
# ============================================================
print("\n" + "="*60)
print("8. (S, O) PAIR FREQUENCY (multiple relations between same pair?)")
print("="*60)
so_pair = defaultdict(list)
for e in edges:
    so_pair[(e["subject"], e["object"])].append(e["relation"])

multi_rel = [(p, rs) for p, rs in so_pair.items() if len(set(rs)) > 1]
print(f"\n(s,o) pairs with multiple distinct relations: {len(multi_rel):,}")
print("Sample:")
for (s, o), rs in sorted(multi_rel, key=lambda x: -len(set(x[1])))[:15]:
    print(f"  {s[:40]:<40} ── {sorted(set(rs))} ──> {o[:40]}")
