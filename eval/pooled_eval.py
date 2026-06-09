#!/usr/bin/env python3
"""
pooled_eval.py — N-way pooled evaluation across systems.

For each target model, pool every emitted edge across all systems and cluster
by canonicalized `(subject, object)` pair. For each cluster, send the
representative claim (the longest description in the cluster) to a single
Claude verifier instance equipped with `web_search`. The verifier returns
`verified` / `refuted` / `unclear`. Per-system Verified/Refuted counts are
the number of clusters that system contributed an edge to, broken down by
verdict.

This script is the canonical entry point for evaluating any new submission
that follows the same per-target-file convention as the baselines (a single
JSON file per (system, target) pair, located in `--graphs-dir`):

    <system_slug>_<target_slug>.json

Each file is a `{nodes: [...], edges: [...]}` graph. Each edge needs at
minimum: `subject`, `object`, `description`, `evidence` (per the schema in
`baselines/prompts/baseline_prompt.md`).

Usage:
    cd eval
    ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py \\
        --graphs-dir ../baselines/outputs \\
        --systems gpt55pro,gpt54pro,cc,o3dr \\
        --concurrency 12

To add a new submission, drop its per-target files into `--graphs-dir`
(e.g., `mysystem_olmo3.json`, `mysystem_nemotron3_super.json`, ...) and
re-run with `--systems gpt55pro,gpt54pro,cc,o3dr,mysystem`. The script
appends new verifications to `outputs/verifications.jsonl` incrementally,
so kills don't lose work and resume picks up where it left off.

Outputs (in `outputs/`):
    verifications.jsonl       per-cluster verdict records
    score.json                aggregate verified/refuted/precision per system
    score_per_target.json     same, broken down per target
    table.txt / table.latex   plain-text and LaTeX renderings of the table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from anthropic import AsyncAnthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")


REPO_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_GRAPHS_DIR = REPO_ROOT / "baselines" / "outputs"
DEFAULT_OUT_DIR    = Path(__file__).resolve().parent / "outputs"

DEFAULT_TARGETS = ["olmo3", "nemotron3_super", "dr_tulu", "smollm3"]
DEFAULT_SYSTEMS = ["gpt55pro", "gpt54pro", "cc", "o3dr", "prov", "prov_unbounded"]
DEFAULT_MODEL   = "claude-sonnet-4-6"

# Display labels used in paper Table 1. Internal slug -> paper label.
SLUG_TO_LABEL = {
    "gpt55pro":       "GPT-5.5 Pro",
    "gpt54pro":       "GPT-5.4 Pro",
    "cc":             "CC-single",
    "o3dr":           "ChatGPT Deep Research",
    "prov":           "ModSleuth (depth-1)",
    "prov_unbounded": "ModSleuth (unbounded)",
}


# ─── Canonical-id normalization (cheap, conservative) ──────────────────

_SEP_RE = re.compile(r"[^a-z0-9]+")


def canonicalize(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip().lower()
    if "/" in s:
        org, rest = s.split("/", 1)
        rest = _SEP_RE.sub("-", rest).strip("-")
        return f"{org}/{rest}"
    return _SEP_RE.sub("-", s).strip("-")


# ─── Pool + cluster ────────────────────────────────────────────────────

@dataclass
class Cluster:
    target: str
    canon_subj: str
    canon_obj: str
    members: list[tuple[str, dict]] = field(default_factory=list)  # (system, edge)

    def representative(self) -> dict:
        return max(
            (e for _, e in self.members),
            key=lambda e: len(e.get("description") or ""),
        )

    def systems(self) -> set[str]:
        return {s for s, _ in self.members}

    def cluster_key(self) -> str:
        return f"{self.target}::{self.canon_subj}::{self.canon_obj}"


def pool_target(graphs_dir: Path, target: str, systems: list[str]) -> list[Cluster]:
    """Pool every edge for one target, cluster by (canon_subj, canon_obj)."""
    clusters: dict[tuple[str, str], Cluster] = {}
    for system in systems:
        path = graphs_dir / f"{system}_{target}.json"
        if not path.exists():
            print(f"  WARN: missing {path}", flush=True)
            continue
        graph = json.load(open(path))
        for edge in graph.get("edges", []):
            cs = canonicalize(edge.get("subject", ""))
            co = canonicalize(edge.get("object", ""))
            if not cs or not co:
                continue
            key = (cs, co)
            if key not in clusters:
                clusters[key] = Cluster(target=target, canon_subj=cs, canon_obj=co)
            clusters[key].members.append((system, edge))
    return list(clusters.values())


# ─── Verifier (Anthropic API + web_search) ─────────────────────────────

VERIFIER_PROMPT_PATH = Path(__file__).resolve().parent / "verifier_prompt.md"
VERIFIER_PROMPT = VERIFIER_PROMPT_PATH.read_text()


def _format_edge_for_verifier(target: str, edge: dict, contributors: list[str]) -> str:
    subject = edge.get("subject", "")
    obj = edge.get("object", "")
    rt = edge.get("relation_type", "")
    desc = edge.get("description", "")
    evidence = edge.get("evidence") or []
    ev_str = "\n".join(
        f"  - source: {ev.get('source','')}\n"
        f"    location: {ev.get('location','')}\n"
        f"    excerpt: {ev.get('excerpt','')}\n"
        f"    explanation: {ev.get('explanation','')}"
        for ev in evidence[:5]
    )
    return f"""# Candidate dependency relationship

Target model under investigation: {target}
Candidate edge submitted by: {', '.join(sorted(contributors))}

subject: {subject}
object:  {obj}
relation_type (hint, not judged): {rt}

description: {desc}

evidence:
{ev_str}
"""


async def verify_one(
    client: AsyncAnthropic,
    cluster: Cluster,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict:
    rep = cluster.representative()
    user_msg = _format_edge_for_verifier(cluster.target, rep, sorted(cluster.systems()))

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=1500,
                    system=VERIFIER_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 6,
                    }],
                )
                text = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
                m = re.search(r"\{[\s\S]*\}", text)
                if not m:
                    raise ValueError(f"No JSON in response: {text[:200]}")
                parsed = json.loads(m.group(0))
                return {
                    "cluster_key": cluster.cluster_key(),
                    "target": cluster.target,
                    "subject": cluster.canon_subj,
                    "object": cluster.canon_obj,
                    "systems": sorted(cluster.systems()),
                    "n_edges_in_cluster": len(cluster.members),
                    "verdict": parsed.get("verdict", "unclear"),
                    "confidence": parsed.get("confidence", 0.5),
                    "explanation": parsed.get("explanation", ""),
                }
            except Exception as e:
                if attempt == 2:
                    return {
                        "cluster_key": cluster.cluster_key(),
                        "target": cluster.target,
                        "subject": cluster.canon_subj,
                        "object": cluster.canon_obj,
                        "systems": sorted(cluster.systems()),
                        "n_edges_in_cluster": len(cluster.members),
                        "verdict": "error",
                        "confidence": 0.0,
                        "explanation": f"{type(e).__name__}: {e}",
                    }
                await asyncio.sleep(2 ** attempt)


# ─── Aggregation ────────────────────────────────────────────────────────

def aggregate(verifications: list[dict], systems: list[str]) -> dict:
    counts = {s: {"verified": 0, "refuted": 0, "unclear": 0, "error": 0} for s in systems}
    for v in verifications:
        verdict = v["verdict"]
        for s in v["systems"]:
            if s in counts:
                counts[s][verdict] = counts[s].get(verdict, 0) + 1

    table = []
    for s in systems:
        c = counts[s]
        decisive = c["verified"] + c["refuted"]
        precision = (c["verified"] / decisive) if decisive else None
        table.append({
            "system": s,
            "verified": c["verified"],
            "refuted": c["refuted"],
            "unclear": c["unclear"],
            "error": c["error"],
            "precision": precision,
        })
    return {"per_system": table, "raw_counts": counts}


def render_table(agg: dict) -> str:
    lines = []
    lines.append(f"{'System':<25}  {'Verified':>9}  {'Refuted':>8}  {'Unclear':>8}  {'Precision':>10}")
    lines.append("-" * 70)
    for row in agg["per_system"]:
        prec = "—" if row["precision"] is None else f"{row['precision']:.3f}"
        label = SLUG_TO_LABEL.get(row["system"], row["system"])
        lines.append(
            f"{label:<25}  {row['verified']:>9}  {row['refuted']:>8}  "
            f"{row['unclear']:>8}  {prec:>10}"
        )
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default=str(DEFAULT_GRAPHS_DIR),
                    help="directory containing per-system per-target JSON graphs")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="comma-separated target slugs")
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                    help="comma-separated system slugs")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="where to write verifications.jsonl, score.json, etc.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Anthropic verifier model")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap clusters per target (0 = no cap; useful for smoke tests)")
    args = ap.parse_args()

    # We don't require ANTHROPIC_API_KEY up-front: if every cluster has
    # already been verified, this run is just an aggregation pass and
    # makes no API calls. The check is deferred to right before we
    # actually instantiate the Anthropic client.

    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    verifications_path = out_dir / "verifications.jsonl"

    # Pool clusters across all targets.
    all_clusters: list[Cluster] = []
    for t in targets:
        ts = pool_target(graphs_dir, t, systems)
        if args.limit:
            ts = ts[: args.limit]
        all_clusters.extend(ts)
        print(f"  pool[{t}]: {len(ts)} clusters from {sum(len(c.members) for c in ts)} edges", flush=True)

    print(f"\nTotal clusters to verify: {len(all_clusters)}", flush=True)

    # Resume: skip clusters already verified.
    done_keys = set()
    existing: list[dict] = []
    if verifications_path.exists():
        for line in verifications_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done_keys.add(rec["cluster_key"])
                existing.append(rec)
        print(f"  resuming: {len(done_keys)} verifications already done", flush=True)

    pending = [c for c in all_clusters if c.cluster_key() not in done_keys]
    print(f"  to do:    {len(pending)} new verifications", flush=True)

    if pending:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set")
        client = AsyncAnthropic()
        sem = asyncio.Semaphore(args.concurrency)
        out_f = open(verifications_path, "a")
        completed = 0
        t0 = time.time()

        async def run_and_save(cluster):
            nonlocal completed
            r = await verify_one(client, cluster, sem, args.model)
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                rate = completed / max(time.time() - t0, 1e-3)
                eta = (len(pending) - completed) / max(rate, 1e-9)
                print(
                    f"  [{completed}/{len(pending)}]  "
                    f"verdict={r['verdict']:<10}  "
                    f"{rate:.2f}/s  ETA {eta/60:.1f}min  "
                    f"{r['target']}/{r['subject'][:30]}→{r['object'][:30]}",
                    flush=True,
                )
            return r

        results = await asyncio.gather(*[run_and_save(c) for c in pending], return_exceptions=False)
        out_f.close()
        existing.extend(results)

    # Aggregate (across all targets and per-target).
    agg = aggregate(existing, systems)
    (out_dir / "score.json").write_text(json.dumps(agg, indent=2))

    per_target = {}
    for t in targets:
        t_recs = [r for r in existing if r["target"] == t]
        per_target[t] = aggregate(t_recs, systems)
    (out_dir / "score_per_target.json").write_text(json.dumps(per_target, indent=2))

    table = render_table(agg)
    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    print(table)
    (out_dir / "table.txt").write_text(table + "\n")

    latex_lines = []
    for row in agg["per_system"]:
        prec = "—" if row["precision"] is None else f"{row['precision']:.3f}"
        latex_lines.append(f"{row['system']} & {row['verified']} & {row['refuted']} & {prec} \\\\")
    (out_dir / "table.latex").write_text("\n".join(latex_lines) + "\n")

    print(f"\nResults in {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
