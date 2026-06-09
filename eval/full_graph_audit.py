#!/usr/bin/env python3
"""Full-graph verifier for Table 6 (paper §D.2).

Audits every relation in a merged graph individually with a Claude
Sonnet 4.6 verifier (web_search), producing a per-edge JSONL of verdicts.
The pooled cluster-level evaluator (`pooled_eval.py`) measures comparative
recall across systems; this script measures *full-graph precision* on the
14,769-edge ModSleuth graph reported in Table 6.

Usage:

    ANTHROPIC_API_KEY=sk-ant-... python full_graph_audit.py \
        --merge-artifact path/to/merge_artifact.json \
        --out outputs/full_graph_verifications.jsonl

The script appends one verdict per line and resumes on kill — re-running
processes only edges not already in the output file. Aggregates print at
the end:

    Total: 14769
    Decisive: 14534  (Verified: 14110, Refuted: 424)
    Unclear:  235
    Precision: 0.9708

The verifier prompt is shared with `pooled_eval.py` (`verifier_prompt.md`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

try:
    from anthropic import AsyncAnthropic
except ImportError:
    sys.exit("anthropic SDK not installed; pip install -e . from the repo root")


HERE = Path(__file__).resolve().parent
VERIFIER_PROMPT = (HERE / "verifier_prompt.md").read_text()
DEFAULT_MODEL = "claude-sonnet-4-6"


def edge_id(rel: dict) -> str:
    """Stable identity for a relation = (subject, relation, object)."""
    return f"{rel.get('subject','')}::{rel.get('relation','')}::{rel.get('object','')}"


def format_edge(rel: dict) -> str:
    subject = rel.get("subject", "")
    obj     = rel.get("object", "")
    rt      = rel.get("relation", "")
    desc    = rel.get("description", "")
    anchors = rel.get("anchor_list") or []
    ev_lines = []
    for a in anchors[:5]:
        if not isinstance(a, dict):
            continue
        ex = a.get("excerpt") or a.get("text") or ""
        if not isinstance(ex, str):
            ex = str(ex)
        ev_lines.append(
            f"  - source: {a.get('source','')}\n"
            f"    location: {a.get('location','')}\n"
            f"    excerpt: {ex[:600]}\n"
            f"    explanation: {a.get('explanation','')}"
        )
    ev_str = "\n".join(ev_lines) if ev_lines else "  (no anchors)"
    return (
        f"# Candidate dependency relationship\n\n"
        f"Candidate edge from the ModSleuth merged graph (full-graph audit).\n\n"
        f"subject: {subject}\n"
        f"object:  {obj}\n"
        f"relation_type (hint, not judged): {rt}\n\n"
        f"description: {desc}\n\n"
        f"evidence:\n{ev_str}\n"
    )


async def verify_one(client: AsyncAnthropic, rel: dict, sem: asyncio.Semaphore,
                     model: str) -> dict:
    user_msg = format_edge(rel)
    async with sem:
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
                text = "".join(b.text for b in resp.content if getattr(b, "text", None))
                # Extract the JSON object the verifier returns.
                start = text.find("{")
                end   = text.rfind("}")
                payload = json.loads(text[start:end + 1]) if start != -1 and end != -1 else {}
                verdict = payload.get("verdict") or "unclear"
                if verdict not in ("verified", "refuted", "unclear"):
                    verdict = "unclear"
                return {
                    "edge_id":     edge_id(rel),
                    "subject":     rel.get("subject", ""),
                    "object":      rel.get("object", ""),
                    "relation":    rel.get("relation", ""),
                    "verdict":     verdict,
                    "confidence":  payload.get("confidence"),
                    "explanation": payload.get("explanation", ""),
                }
            except Exception as e:
                if attempt == 2:
                    return {"edge_id": edge_id(rel), "verdict": "error",
                            "error": str(e)[:300]}
                await asyncio.sleep(2 ** attempt)


def aggregate(verdicts: list[dict]) -> dict:
    counts = {"verified": 0, "refuted": 0, "unclear": 0, "error": 0}
    for v in verdicts:
        counts[v.get("verdict", "error")] = counts.get(v.get("verdict", "error"), 0) + 1
    total    = sum(counts.values())
    decisive = counts["verified"] + counts["refuted"]
    precision = (counts["verified"] / decisive) if decisive else None
    return {
        "total":     total,
        "decisive":  decisive,
        "verified":  counts["verified"],
        "refuted":   counts["refuted"],
        "unclear":   counts["unclear"],
        "error":     counts["error"],
        "precision": precision,
    }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--merge-artifact", required=True, type=Path,
                   help="Path to ModSleuth's merged graph JSON (the 14,769-edge artifact).")
    p.add_argument("--out", default=HERE / "outputs" / "full_graph_verifications.jsonl",
                   type=Path,
                   help="JSONL of per-edge verdicts (appended; safe to resume).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap edges audited (0 = no cap; useful for smoke tests).")
    args = p.parse_args()

    if not args.merge_artifact.exists():
        sys.exit(f"merge artifact not found: {args.merge_artifact}")
    # ANTHROPIC_API_KEY check is deferred until we actually need to call
    # the API (see below), so a re-aggregation against a complete
    # outputs/full_graph_verifications.jsonl runs without a key.

    G = json.loads(args.merge_artifact.read_text())
    relations = G.get("relations") or G.get("edges") or []
    if args.limit:
        relations = relations[: args.limit]
    print(f"loaded {len(relations)} edges from {args.merge_artifact}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    existing: list[dict] = []
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done_ids.add(rec["edge_id"])
                existing.append(rec)
        print(f"resuming: {len(done_ids)} edges already verified", flush=True)

    pending = [r for r in relations if edge_id(r) not in done_ids]
    print(f"to do:    {len(pending)} new verifications", flush=True)

    if pending:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set")
        client = AsyncAnthropic()
        sem = asyncio.Semaphore(args.concurrency)
        out_f = open(args.out, "a")
        completed = 0
        t0 = time.time()

        async def run_and_save(rel):
            nonlocal completed
            r = await verify_one(client, rel, sem, args.model)
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                rate = completed / max(time.time() - t0, 1e-3)
                eta  = (len(pending) - completed) / max(rate, 1e-9)
                print(f"  [{completed}/{len(pending)}]  verdict={r.get('verdict','?'):<10}  "
                      f"{rate:.2f}/s  ETA {eta/60:.1f}min", flush=True)
            return r

        results = await asyncio.gather(*[run_and_save(r) for r in pending],
                                       return_exceptions=False)
        out_f.close()
        existing.extend(results)

    agg = aggregate(existing)
    print()
    print("Full-graph audit (Table 6):")
    print(f"  Total:     {agg['total']}")
    print(f"  Decisive:  {agg['decisive']}  (Verified: {agg['verified']}, Refuted: {agg['refuted']})")
    print(f"  Unclear:   {agg['unclear']}")
    if agg["precision"] is not None:
        print(f"  Precision: {agg['precision']:.4f}")
    score_path = args.out.with_name(args.out.stem + ".score.json")
    score_path.write_text(json.dumps(agg, indent=2))
    print(f"  → {score_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
