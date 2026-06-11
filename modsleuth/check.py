"""Deterministic graph-quality checks for ModSleuth merge artifacts.

Every check corresponds to a defect class observed while QA'ing real
pipeline output: zero-anchor edges, dependency_kind labels that
contradict their relation, self-loops, duplicate triples, subjects
that don't resolve to lattice models, bracket-style endpoint names,
flattened (singleton-group) lattices, and evaluation-edge inflation.

Pure Python and read-only: checks count and sample, they never mutate
the artifact and never make semantic judgments — anything they flag is
routed to review, not auto-fixed.

Usage:

    modsleuth check --source path/to/merge_artifact.json

Exit code 1 when any P1 (blocker-grade) finding exists, else 0 —
suitable as a gate between pipeline stages or in CI.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .pipeline import (
    DEPENDENCY_KIND_VALUES,
    RELATION_DEPENDENCY_KIND,
    parse_virtual_address,
)

# Priorities: P1 = blocker (violates a hard invariant), P2 = review
# (suspicious, needs eyes), P3 = informational (distribution facts).
P1, P2, P3 = "P1", "P2", "P3"

# Weight lineage can only connect models — a dataset subject here is
# either a wrong subject or a wrong item kind, under every convention.
_WEIGHT_RELATIONS = {"trained_from", "merged_from", "quantized_from"}


class Report:
    def __init__(self, max_samples: int = 3):
        self.max_samples = max_samples
        self.findings: list[tuple[str, str, int, list[str]]] = []

    def add(self, priority: str, title: str, count: int,
            samples: list[str] | None = None) -> None:
        if count:
            self.findings.append(
                (priority, title, count, (samples or [])[: self.max_samples])
            )

    def info(self, title: str, detail: str) -> None:
        self.findings.append((P3, f"{title}: {detail}", 0, []))

    def render(self) -> tuple[str, int]:
        lines: list[str] = []
        p1_total = 0
        for priority in (P1, P2, P3):
            section = [f for f in self.findings if f[0] == priority]
            if not section:
                continue
            lines.append(f"--- {priority} ---")
            for _, title, count, samples in sorted(section, key=lambda f: -f[2]):
                lines.append(f"  {title}" + (f": {count}" if count else ""))
                for s in samples:
                    lines.append(f"      e.g. {s}")
                if priority == P1:
                    p1_total += count
            lines.append("")
        verdict = ("FAIL — fix P1 findings before releasing this graph"
                   if p1_total else "PASS — no blocker-grade findings")
        lines.append(verdict)
        return "\n".join(lines), (1 if p1_total else 0)


def _edge_str(e: dict) -> str:
    return f"{e.get('subject')} --{e.get('relation')}--> {e.get('object')}"


def _load_items(G: dict) -> tuple[list[dict], list[dict]]:
    groups = ((G.get("lattice") or {}).get("groups")) or G.get("groups") or []
    items = [it for g in groups
             for it in (g.get("items") or []) if isinstance(it, dict)]
    return items, groups


def run_checks(source: Path, *, max_samples: int = 3) -> int:
    G = json.loads(Path(source).read_text())
    relations = G.get("relations") or G.get("edges") or []
    items, groups = _load_items(G)
    report = Report(max_samples=max_samples)

    # Lattice index: formal_name / alias → item, for endpoint resolution.
    by_name: dict[str, dict] = {}
    families: set[str] = set()
    for it in items:
        fn = it.get("formal_name")
        if isinstance(fn, str) and fn:
            by_name.setdefault(fn, it)
        for a in it.get("aliases") or []:
            if isinstance(a, str) and a:
                by_name.setdefault(a, it)
        fam = (it.get("identity") or {}).get("family")
        if fam:
            families.add(str(fam))

    def resolves(name: str) -> dict | None:
        return by_name.get(name)

    def is_virtual(name: str) -> bool:
        virt = parse_virtual_address(name)
        return virt is not None and (not families or virt[0] in families)

    # ── Edge-level invariants ─────────────────────────────────────────
    zero_anchor, kind_contradiction, bad_kind = [], [], []
    self_loops, dup_triples = [], []
    subj_unresolved, subj_dataset_weight = [], []
    dataset_subject_dataops = 0
    bracket_garbage = []
    off_lattice_objects = 0
    seen_triples: Counter = Counter()
    rel_hist: Counter = Counter()
    provenance_flagged = 0
    with_traced_target = 0

    for e in relations:
        if not isinstance(e, dict):
            continue
        subject = str(e.get("subject") or "")
        obj = str(e.get("object") or "")
        relation = str(e.get("relation") or "")
        rel_hist[relation] += 1

        if not (e.get("anchor_list") or []):
            zero_anchor.append(_edge_str(e))

        kind = e.get("dependency_kind")
        expected = RELATION_DEPENDENCY_KIND.get(relation)
        if expected is not None and kind != expected:
            kind_contradiction.append(f"{_edge_str(e)} [kind={kind!r}, expected {expected!r}]")
        elif kind not in DEPENDENCY_KIND_VALUES:
            bad_kind.append(f"{_edge_str(e)} [kind={kind!r}]")

        if subject and subject == obj:
            self_loops.append(_edge_str(e))

        seen_triples[(subject, relation, obj)] += 1

        if items:
            subj_item = resolves(subject)
            if subj_item is None and not is_virtual(subject):
                subj_unresolved.append(subject)
            elif subj_item is not None and subj_item.get("kind") == "dataset":
                if relation in _WEIGHT_RELATIONS:
                    subj_dataset_weight.append(f"{_edge_str(e)} (subject kind=dataset)")
                else:
                    # Dataset subjects on data-operation relations are a
                    # sanctioned convention in reviewed graphs (the named
                    # intermediate dataset carries the edge); the pipeline
                    # itself emits consumer-model subjects.
                    dataset_subject_dataops += 1
            if resolves(obj) is None and not is_virtual(obj):
                off_lattice_objects += 1
                if "[" in obj and parse_virtual_address(obj) is None:
                    bracket_garbage.append(obj)
            if "[" in subject and subj_item is None and parse_virtual_address(subject) is None:
                bracket_garbage.append(subject)

        if e.get("provenance_review"):
            provenance_flagged += 1
        if e.get("traced_targets") or e.get("traced_target"):
            with_traced_target += 1

    dup_triples = [f"{s} --{r}--> {o} (x{n})"
                   for (s, r, o), n in seen_triples.items() if n > 1]

    report.add(P1, "Edges with empty anchor_list", len(zero_anchor), zero_anchor)
    report.add(P1, "dependency_kind contradicts canonical relation",
               len(kind_contradiction), kind_contradiction)
    report.add(P1, "Self-loop edges (subject == object)", len(self_loops), self_loops)
    report.add(P1, "Duplicate (subject, relation, object) triples",
               len(dup_triples), dup_triples)
    report.add(P2, "dependency_kind missing or invalid", len(bad_kind), bad_kind)
    if items:
        uniq_unresolved = sorted(set(subj_unresolved))
        report.add(P1, "Subjects resolving to neither lattice item nor virtual address",
                   len(uniq_unresolved), uniq_unresolved)
        report.add(P2, "Dataset-kind subjects on weight-lineage relations",
                   len(subj_dataset_weight), subj_dataset_weight)
        report.add(P2, "Bracket-style endpoint names that parse as nothing",
                   len(sorted(set(bracket_garbage))), sorted(set(bracket_garbage)))
        if dataset_subject_dataops:
            report.info("Dataset-subject data-operation edges",
                        f"{dataset_subject_dataops} — sanctioned in reviewed "
                        "graphs; fresh pipeline output uses consumer-model subjects")

    # ── Lattice hygiene ───────────────────────────────────────────────
    if groups:
        n_groups = len(groups)
        singleton = sum(1 for g in groups if len(g.get("items") or []) == 1)
        report.info("Lattice", f"{len(items)} items in {n_groups} groups "
                               f"({singleton} singleton)")
        # A ~100% singleton rate on a graph of any size means the family
        # hierarchy was flattened somewhere (every item its own group).
        if n_groups >= 50 and singleton == n_groups:
            report.add(P1, "Lattice is fully flattened — every group is a singleton "
                           "(family hierarchy lost)", n_groups)
        dup_keys: Counter = Counter()
        for it in items:
            link = next((l.get("url") for l in (it.get("links") or [])
                         if isinstance(l, dict) and l.get("url")), None)
            dup_keys[(it.get("formal_name"), link)] += 1
        dups = [f"{k[0]} ({k[1]})" for k, n in dup_keys.items() if n > 1]
        report.add(P2, "Duplicate (formal_name, primary_link) lattice items",
                   len(dups), dups)
        multi = [g for g in groups if len(g.get("items") or []) > 1
                 and g.get("identity_keys") is not None]
        rootless = [
            str(g.get("family"))
            for g in multi
            if not any((it.get("identity") or {}) == {"family": g.get("family")}
                       for it in g.get("items") or [])
        ]
        report.add(P2, "Multi-item families without a family root",
                   len(rootless), rootless)
    else:
        report.info("Lattice", "absent from artifact — lattice checks skipped")

    # ── Distribution facts ────────────────────────────────────────────
    total = len(relations)
    if total:
        eval_n = rel_hist.get("used_for_evaluation", 0)
        report.info("Edges", f"{total} total; top relations: "
                    + ", ".join(f"{r} {n}" for r, n in rel_hist.most_common(5)))
        report.info("Evaluation-edge share",
                    f"{eval_n}/{total} ({eval_n / total * 100:.1f}%)"
                    + (" — high; consider an eval-collapse review"
                       if eval_n / total > 0.35 else ""))
        if items:
            report.info("Off-lattice objects",
                        f"{off_lattice_objects}/{total} "
                        f"({off_lattice_objects / total * 100:.1f}%) — free-text "
                        "objects are legal but should stay a small minority")
        report.info("Evidence provenance",
                    f"{with_traced_target}/{total} edges carry traced_target(s); "
                    f"{provenance_flagged} flagged provenance_review")
        ops_index = G.get("operations") or {}
        if ops_index:
            with_ops = sum(1 for e in relations
                           if isinstance(e, dict) and e.get("operation_ids"))
            report.info("Operations",
                        f"{len(ops_index)} in index; {with_ops}/{total} edges "
                        "carry operation_ids")

    text, code = report.render()
    print(f"Source: {source}")
    print(f"Edges: {total}   Lattice items: {len(items)}\n")
    print(text)
    return code


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source", required=True, type=Path,
                   help="Path to a merged graph JSON (merge artifact).")
    p.add_argument("--max-samples", type=int, default=3,
                   help="Examples shown per finding.")
    args = p.parse_args(argv)
    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2
    return run_checks(args.source, max_samples=args.max_samples)


if __name__ == "__main__":
    sys.exit(main())
