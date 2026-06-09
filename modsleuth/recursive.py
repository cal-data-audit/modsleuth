#!/usr/bin/env python3
"""Reference recursive-expansion driver (paper §3.2 / §A).

This is the "outer loop" that turns the single-target base pipeline
(``modsleuth run discover|extract|organize|audit|relate|reconcile|
triage|merge``) into a multi-hop, evidence-grounded dependency graph.

For each seed we allocate a separate ``MODSLEUTH_STORAGE`` directory,
run the full base pipeline once, then iteratively expand the
top-K newly-discovered upstream artifacts up to a chosen depth
(BFS-style, ranked by *parent count* in the per-seed merged graph).
After each expansion round we re-run merge so the per-seed graph
stays current.

The exact expansion strategy used in the paper is target-specific
(seed list, per-seed K, optional shared-bridge pre-seeding); this
driver provides one concrete reference configuration that
reproduces the recursive behavior described in the paper.

Run as a module:

    python -m modsleuth.recursive \
        --seed allenai/OLMo-3-1125-32B \
        --seed nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --depth 3 --top-k 5

or via the CLI:

    modsleuth recursive \
        --seed allenai/OLMo-3-1125-32B \
        --depth 3 --top-k 5

Environment-managed state: a fresh ``storage/<seed_slug>/`` directory
is created for each seed (override with ``--storage-root``). Each
seed's final per-seed merged graph is written to
``storage/<seed_slug>/runs/<run>/merge_artifact.json``.

To abort cleanly, send ``SIGINT`` (Ctrl-C) to *this* Python process —
the recursive driver. Killing only the inner ``claude`` subprocess
will trigger the pipeline's automatic retry path, since the parent
process stays alive.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _slug(target: str) -> str:
    """Turn a HF-style ``org/Model-Name`` into a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", target).strip("_")[:80] or "seed"


def _run(env: dict[str, str], *args: str) -> None:
    """Run a ``modsleuth`` CLI invocation with the given env, fail loudly on error."""
    full = ["python3", "-m", "modsleuth.cli", *args]
    print(f"[run] {' '.join(full)}", flush=True)
    proc = subprocess.run(full, env=env, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}): {' '.join(full)}")


def _latest_merge_path(storage: Path) -> Path:
    """Return the most recent ``merge_artifact.json`` inside a per-seed storage dir."""
    candidates = sorted(storage.glob("runs/*/merge_artifact.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no merge_artifact.json under {storage}/runs/")
    return candidates[0]


def _new_upstreams(graph: dict, already: set[str]) -> Counter:
    """Score upstream nodes by the number of edges pointing at them.

    Returns a Counter mapping ``object`` (formal_name) → parent count,
    restricted to nodes we haven't expanded yet.
    """
    counts: Counter = Counter()
    for edge in graph.get("relations", []):
        obj = edge.get("object")
        if obj and obj not in already:
            counts[obj] += 1
    return counts


def _pick_bfs(scored: Counter, expanded: set[str], top_k: int) -> list[str]:
    """Breadth-first: take the top-K by parent count at this depth."""
    return [n for n, _ in scored.most_common(top_k)]


def _pick_dfs(scored: Counter, expanded: set[str], top_k: int) -> list[str]:
    """Depth-first: pick the single highest-scoring un-expanded node and
    follow that chain. ``top_k`` is ignored — DFS expands one node per
    round, deepening one chain at a time."""
    if not scored:
        return []
    return [scored.most_common(1)[0][0]]


def _connectivity_to_expanded(graph: dict, expanded: set[str],
                              already: set[str]) -> Counter:
    """Score un-expanded objects by connectivity to the expanded
    subgraph: the number of edges whose ``subject`` is already an
    expanded node. Matches paper §A's beam scoring rule.
    """
    counts: Counter = Counter()
    for edge in graph.get("relations", []):
        if edge.get("subject") not in expanded:
            continue
        obj = edge.get("object")
        if obj and obj not in already:
            counts[obj] += 1
    return counts


def _pick_beam(graph: dict, expanded: set[str], top_k: int,
               beam_history: dict[str, int]) -> list[str]:
    """Beam search (paper §A): keep the global top-K structurally
    central ancestors across depths, scored by cumulative connectivity
    to previously-expanded nodes. ``top_k`` is the beam width.
    """
    conn = _connectivity_to_expanded(graph, expanded, already=expanded)
    for n, s in conn.items():
        beam_history[n] = beam_history.get(n, 0) + s
    ranked = sorted(((n, s) for n, s in beam_history.items() if n not in expanded),
                    key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in ranked[:top_k]]


def expand_seed(seed: str, depth: int, top_k: int,
                storage_root: Path, strategy: str = "bfs") -> Path:
    """Run the base pipeline for ``seed``, then expand top-K parents up
    to ``depth`` hops using ``strategy`` ∈ {bfs, dfs, beam}. Returns the
    path to the final merged graph.
    """
    seed_storage = (storage_root / _slug(seed)).resolve()
    seed_storage.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MODSLEUTH_STORAGE"] = str(seed_storage)
    env["MODSLEUTH_PATH"] = str(seed_storage / "graph.db")

    print(f"\n=== seed={seed}  storage={seed_storage}  strategy={strategy} ===", flush=True)

    # Depth-1: full base pipeline against the seed.
    _run(env, "init")
    _run(env, "run", "discover", "--target", seed)
    for stage in ("extract", "organize", "audit", "relate",
                  "reconcile", "triage", "merge"):
        _run(env, "run", stage)

    expanded: set[str] = {seed}
    beam_history: dict[str, int] = {}  # only used by beam

    # Depth d ∈ [2..depth]: pick parents per the chosen strategy,
    # call run_expand on each, then re-merge.
    for d in range(2, depth + 1):
        merge_path = _latest_merge_path(seed_storage)
        graph = json.loads(merge_path.read_text())
        scored = _new_upstreams(graph, expanded)
        if not scored:
            print(f"[depth {d}] no new upstreams to expand; stopping early", flush=True)
            break
        if strategy == "bfs":
            candidates = _pick_bfs(scored, expanded, top_k)
        elif strategy == "dfs":
            candidates = _pick_dfs(scored, expanded, top_k)
        elif strategy == "beam":
            candidates = _pick_beam(graph, expanded, top_k, beam_history)
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")
        if not candidates:
            print(f"[depth {d}] no candidates returned by {strategy}; stopping", flush=True)
            break
        print(f"[depth {d}] {strategy}: expanding {len(candidates)} parent(s): {candidates}", flush=True)
        for node in candidates:
            _run(env, "run", "expand", "--node", node)
            expanded.add(node)
        # Re-merge so the per-seed graph reflects newly added relate artifacts.
        _run(env, "run", "merge")

    final = _latest_merge_path(seed_storage)
    print(f"\n[done] seed={seed}  merged graph: {final}", flush=True)
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--seed", action="append", required=True,
                   help="Target model identifier. Pass multiple times for multiple seeds.")
    p.add_argument("--depth", type=int, default=3,
                   help="Maximum recursion depth (depth 1 = base pipeline only).")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K parents per round (beam width for --strategy beam; "
                        "branching factor for --strategy bfs; ignored for --strategy dfs).")
    p.add_argument("--strategy", choices=("bfs", "dfs", "beam"), default="bfs",
                   help="Per-round expansion policy. bfs = level-by-level top-K; "
                        "dfs = follow the single highest-scoring chain; "
                        "beam = global top-K across depths by cumulative score.")
    p.add_argument("--storage-root", type=Path,
                   default=REPO_ROOT / "storage",
                   help="Root directory for per-seed MODSLEUTH_STORAGE dirs.")
    args = p.parse_args(argv)

    args.storage_root.mkdir(parents=True, exist_ok=True)
    finals: list[Path] = []
    for seed in args.seed:
        finals.append(expand_seed(seed, args.depth, args.top_k,
                                  args.storage_root, args.strategy))

    print("\n=== finished ===")
    for path in finals:
        print(f"  {path}")
    print(
        "\nTo merge across seeds into a single graph:\n"
        f"  modsleuth run merge "
        + " ".join(f"--source {p}" for p in finals)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
