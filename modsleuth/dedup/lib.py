"""Shared helpers for the dedup pipeline.

The pipeline runs four stages (see ``modsleuth.dedup.__main__``) over a
merged JSON graph:
  1. heuristic    — signature clustering + fuzzy surface-form merge (no LLM)
  2. hub-audit    — per-hub LLM audit of edges (drops dupes/hallucinations)
  3. node-dedup   — whole-graph LLM-verified node dedup
  4. release      — LLM classifies KEEP/DROP, drops intermediates, rewires edges

This module factors out the helpers used in more than one stage:
  - signature() / can_merge() — multi-attribute identity for a node name
  - ConflictGuardedUnionFind — union-find that refuses to merge components
    with mutually conflicting specifiers (different versions/sizes/stages/dates)
  - call_opus() / run_parallel_llm() — Opus 4.7 CLI wrapper + worker pool
  - rewrite_edges() — apply a canon_map and merge anchor lists
  - assert_invariants() — sanity checks every stage runs before writing
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable

# ============================================================
# Defaults
# ============================================================
MODEL = "claude-opus-4-7"
DEFAULT_WORKERS = 24

# ============================================================
# Regex helpers (single source of truth)
# ============================================================
ORG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)/(.+)$")
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)[Bb]\b")
VERSION_RE = re.compile(r"\b(\d+(?:[\._]\d+)+|\d+)\b")
PAREN_ALIAS_RE = re.compile(r"\s*\([^)]*\)\s*")
BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*")
DATE_RE = re.compile(r"(?<!\d)(0[1-9]\d{2}|1[0-2]\d{2})(?!\d)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
TOKEN_SPLIT_RE = re.compile(r"[\s\-_/.\[\](),=:]+")

INTERNAL_PATH_RE = re.compile(r"^(?:/|gs://|weka://|s3://|/weka/|/scratch/|/fsx/)")
WIKI_BRACKET_RE = re.compile(r"^(.+?)\s*\[([^\]]+)\]\s*$")

STAGE_KEYWORDS = {
    "sft", "dpo", "instruct", "think", "base", "rl", "rl-zero", "rlzero",
    "rlhf", "rlvr", "reasoning", "content-safety", "gen-rm", "genrm",
    "preview", "fp8", "bf16", "nvfp4", "onnx", "fp16", "int4", "int8",
    "quantized", "chat", "completions", "completion", "reward", "rm",
    "policy", "retriever", "encoder", "embedding", "embed", "tokenizer",
    "checkpoint", "intermediate", "mid", "cpt", "awq", "gguf",
}

# ============================================================
# I/O
# ============================================================
def load_graph(path: Path) -> dict:
    return json.loads(path.read_text())


def save_graph(graph: dict, path: Path) -> None:
    path.write_text(json.dumps(graph))


def collect_node_names(graph: dict) -> set[str]:
    nodes: set[str] = set()
    for e in graph.get("relations", []):
        if e.get("subject"):
            nodes.add(e["subject"])
        if e.get("object"):
            nodes.add(e["object"])
    for g in graph.get("lattice", {}).get("groups", []):
        for it in g.get("items", []):
            fn = it.get("formal_name")
            if fn:
                nodes.add(fn)
    return {n for n in nodes if isinstance(n, str) and n}


def to_str(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("formal_name") or v.get("name") or ""
    if isinstance(v, str):
        return v
    return ""


def degree_map(graph: dict) -> Counter:
    deg: Counter = Counter()
    for e in graph.get("relations", []):
        deg[e.get("subject", "")] += 1
        deg[e.get("object", "")] += 1
    return deg


def sample_anchors(graph: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in graph.get("relations", []):
        for n in (e.get("subject"), e.get("object")):
            if n and n not in out:
                anchors = e.get("anchor_list") or []
                if anchors:
                    a = anchors[0]
                    out[n] = (a.get("source") or a.get("url") or a.get("path") or "")[:140]
    return out


# ============================================================
# Name parsing + signatures (single source of truth)
# ============================================================
def parse_wiki_brackets(name: str) -> str:
    """Convert 'OLMo 3 [size=32B, stage=Instruct-SFT, version=3.1]' → 'OLMo-3.1-32B-Instruct-SFT'."""
    m = WIKI_BRACKET_RE.match(name)
    if not m:
        return name
    family, attrs_str = m.groups()
    attrs: dict[str, str] = {}
    for attr in attrs_str.split(","):
        if "=" in attr:
            k, v = attr.split("=", 1)
            attrs[k.strip()] = v.strip()
    version = attrs.get("version")
    if version:
        family = re.sub(r"\s+\d+(?:\.\d+)?\s*$", f" {version}", family)
    parts = [family.strip()]
    for k in ("size", "variant", "stage", "track"):
        if attrs.get(k):
            parts.append(attrs[k])
    return "-".join(parts).replace(" ", "-")


def lex_collapse(name: str) -> str:
    return NON_ALNUM_RE.sub("", name.lower())


def tokenize(name: str) -> frozenset[str]:
    return frozenset(t for t in TOKEN_SPLIT_RE.split(name.lower()) if t and len(t) > 1)


def split_org(name: str) -> tuple[str | None, str]:
    m = ORG_RE.match(name)
    if m:
        return m.group(1).lower(), m.group(2)
    return None, name


def signature(name: str) -> tuple:
    """Hard-separator signature for a node name.

    Two nodes can only safely merge if their signatures are compatible per
    can_merge() — same bare lex, no conflicting org/version/size/stage/date,
    same parens-suffix, same non-standard bracket attrs.
    """
    if not isinstance(name, str):
        return (None, "", "", frozenset(), frozenset(), frozenset(), frozenset(), None, frozenset())

    # Parens suffix is a distinguishing specifier (e.g. "cais/mmlu (STEM)" ≠ bare).
    paren_match = re.search(r"\(([^)]+)\)\s*$", name.strip())
    paren_specifier: str | None = None
    if paren_match:
        ps = paren_match.group(1).strip().lower()
        # Skip parens that look like an org-prefixed alias (info-equivalent).
        if "/" in ps and not any(
            k in ps for k in ("split", "subset", "variant", "ablation", "config", "version", "diamond")
        ):
            paren_specifier = None
        else:
            paren_specifier = ps

    # Non-standard bracket attrs (e.g. ablation=, variant=) are also specifiers.
    bracket_specs: list[tuple[str, str]] = []
    bracket_match = re.search(r"\[([^\]]+)\]", name)
    if bracket_match:
        for kv in bracket_match.group(1).split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                k = k.strip().lower()
                v = v.strip().lower()
                if k in ("size", "version", "stage", "track"):
                    continue  # captured in dedicated buckets below
                bracket_specs.append((k, v))
    bracket_specifier = frozenset(bracket_specs)

    # Bare core: strip parens + brackets, parse wiki form.
    work = parse_wiki_brackets(name)
    work_core = BRACKET_RE.sub("", PAREN_ALIAS_RE.sub("", work)).strip()
    org, bare = split_org(work_core)

    bare_norm = re.sub(r"[\s_]+", "-", bare.lower().strip())
    bare_norm = re.sub(r"-+", "-", bare_norm).strip("-")
    bare_collapsed = bare_norm.replace("-", "").replace(".", "")

    sizes = frozenset(s.lower() for s in SIZE_RE.findall(name))
    raw_versions = VERSION_RE.findall(name)
    versions = frozenset(
        v.replace("_", ".")
        for v in raw_versions
        if not v.lower().endswith("b") and len(v) <= 8
    )
    versions = versions - {s.lower() for s in sizes}

    name_lower = name.lower()
    stages = frozenset(t for t in STAGE_KEYWORDS if t in name_lower)
    dates = frozenset(
        d for d in DATE_RE.findall(name)
        if 1000 < int(d) < 2030 and d not in versions and d not in {s.lower() for s in sizes}
    )

    return (org, bare_norm, bare_collapsed, versions, sizes, stages, dates, paren_specifier, bracket_specifier)


def can_merge(sig_a: tuple, sig_b: tuple) -> bool:
    """Return True iff two signatures can be merged without violating any hard separator."""
    if sig_a is None or sig_b is None:
        return False
    org_a, bare_a, coll_a, ver_a, size_a, stage_a, date_a, paren_a, spec_a = sig_a
    org_b, bare_b, coll_b, ver_b, size_b, stage_b, date_b, paren_b, spec_b = sig_b
    if coll_a != coll_b:
        return False
    if org_a and org_b and org_a != org_b:
        return False
    if ver_a and ver_b and ver_a != ver_b:
        return False
    if size_a and size_b and size_a != size_b:
        return False
    if stage_a and stage_b and stage_a != stage_b:
        return False
    if date_a and date_b and date_a != date_b:
        return False
    if paren_a != paren_b:
        return False
    if spec_a != spec_b:
        return False
    return True


# ============================================================
# Categorical drops
# ============================================================
def is_categorical_drop(name: str) -> bool:
    """Names we drop without LLM review: internal paths, free-text descriptions."""
    if not name or not isinstance(name, str):
        return True
    n = name.strip()
    if not n:
        return True
    if INTERNAL_PATH_RE.match(n):
        return True
    if "://" in n and not n.startswith("https://huggingface.co/"):
        return True
    paren = re.search(r"\(([^)]+)\)", n)
    if paren and len(paren.group(1)) > 50:
        return True
    if len(n) > 200:
        return True
    return False


# ============================================================
# Specifier extraction for conflict-guarded union (lighter weight than full signature)
# ============================================================
def specs_for_conflict(name: str) -> dict[str, frozenset]:
    """Specifier sets used to detect conflicts when uniting two components."""
    sizes = frozenset(s.lower() for s in SIZE_RE.findall(name))
    raw_versions = VERSION_RE.findall(name)
    versions = frozenset(
        v.replace("_", ".") for v in raw_versions if not v.lower().endswith("b") and "." in v
    ) - sizes
    tokens = re.split(r"[\s\-_/.\[\](),=:]+", name.lower())
    stages = frozenset(t for t in tokens if t in STAGE_KEYWORDS)
    dates = frozenset(d for d in DATE_RE.findall(name))
    return {"dates": dates, "sizes": sizes, "versions": versions, "stages": stages}


def specs_conflict(s_a: dict, s_b: dict) -> bool:
    for k in ("dates", "versions", "sizes", "stages"):
        a, b = s_a[k], s_b[k]
        if a and b and a != b:
            return True
    return False


# ============================================================
# Conflict-guarded union-find
# ============================================================
class ConflictGuardedUnionFind:
    """Union-find that refuses to merge components with mutually-conflicting specifiers.

    This is what prevents `OLMo 2 32B Instruct` (bare alias) from chaining
    `OLMo-2-0325-32B-Instruct` and `OLMo-2-1124-32B-Instruct` into one component
    when an LLM individually approves both `(bare, 0325)` and `(bare, 1124)`.
    """

    def __init__(self, names: Iterable[str]):
        self._parent = {n: n for n in names}
        self._specs = {n: dict(specs_for_conflict(n)) for n in names}
        self.skipped: list[tuple[str, str]] = []

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def try_unite(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if specs_conflict(self._specs[ra], self._specs[rb]):
            self.skipped.append((a, b))
            return False
        self._parent[ra] = rb
        merged = {k: self._specs[ra][k] | self._specs[rb][k] for k in self._specs[ra]}
        self._specs[rb] = merged
        return True

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for n in self._parent:
            out[self.find(n)].append(n)
        return out


# ============================================================
# Edge rewriting + anchor merging
# ============================================================
def rewrite_edges(
    edges: list[dict],
    canon_map: dict[str, str | None],
    cap_anchors: int | None = None,
) -> tuple[list[dict], int, int]:
    """Apply canon_map to edges. Drops edges with None endpoints or self-loops.
    Returns (new_edges, dropped_endpoint, dropped_self_loop)."""
    new_edges: dict[tuple, dict] = {}
    dropped_endpoint = 0
    dropped_self = 0
    for e in edges:
        s = canon_map.get(to_str(e.get("subject")), to_str(e.get("subject")))
        o = canon_map.get(to_str(e.get("object")), to_str(e.get("object")))
        rel = e.get("relation", "")
        if not s or not o:
            dropped_endpoint += 1
            continue
        if s == o or not rel:
            dropped_self += 1
            continue
        key = (s, rel, o)
        if key not in new_edges:
            new_edges[key] = {
                **e,
                "subject": s,
                "object": o,
                "anchor_list": list(e.get("anchor_list") or []),
                "description_variants": list(e.get("description_variants") or []),
            }
        else:
            anchors = new_edges[key]["anchor_list"]
            anchors.extend(e.get("anchor_list") or [])
            if cap_anchors:
                new_edges[key]["anchor_list"] = anchors[:cap_anchors]
            for v in e.get("description_variants") or []:
                if v not in new_edges[key]["description_variants"]:
                    new_edges[key]["description_variants"].append(v)
    return list(new_edges.values()), dropped_endpoint, dropped_self


def rebuild_lattice(
    groups: list[dict], canon_map: dict[str, str], final_node_set: set[str]
) -> list[dict]:
    """Rebuild lattice groups by collapsing items into their canonical nodes."""
    canon_to_items: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        for it in g.get("items", []):
            fn = it.get("formal_name", "")
            if not fn:
                continue
            canon = canon_map.get(fn, fn)
            if canon and canon in final_node_set:
                canon_to_items[canon].append(it)
    out: list[dict] = []
    for canon, items in canon_to_items.items():
        primary = next((i for i in items if i.get("formal_name") == canon), items[0])
        primary = dict(primary)
        primary["formal_name"] = canon
        primary["alias_count"] = len(items)
        out.append({"items": [primary], "id": canon})
    return out


# ============================================================
# Sanity invariants (run before every write)
# ============================================================
SEED_PREFIXES = (
    "allenai/Olmo-3",
    "rl-research/DR-Tulu",
    "rl-research/dr-tulu",
    "nvidia/NVIDIA-Nemotron-3",
    "HuggingFaceTB/SmolLM3",
)


def assert_invariants(node_set: set[str]) -> None:
    """Each invariant caught at least one regression during development."""
    olmo3 = [n for n in node_set if "olmo-3-" in n.lower().replace(" ", "-") and "3.1" not in n.lower()]
    olmo31 = [n for n in node_set if "olmo-3.1" in n.lower()]
    assert olmo3, "Olmo-3 nodes empty"
    assert olmo31, "Olmo-3.1 nodes empty (version distinction collapsed)"

    for prefix in SEED_PREFIXES[:1] + SEED_PREFIXES[3:]:  # only canonical seeds
        if not any(prefix in n for n in node_set):
            # DR-Tulu has multiple casing variants; check that one of them is present.
            if prefix == "allenai/Olmo-3":
                continue
        assert any(prefix in n for n in node_set), f"Seed prefix '{prefix}' missing"
    assert any("rl-research/" in n.lower() and "tulu" in n.lower() for n in node_set), "DR-Tulu missing"

    aime_2024 = any("aime" in n.lower() and "2024" in n for n in node_set)
    aime_2025 = any("aime" in n.lower() and "2025" in n for n in node_set)
    assert aime_2024, "AIME 2024 missing"
    assert aime_2025, "AIME 2025 missing"


# ============================================================
# Opus CLI wrapper + parallel worker pool
# ============================================================
def call_opus(
    prompt: str,
    *,
    effort: str = "max",
    timeout: int = 600,
    model: str = MODEL,
) -> tuple[str, int]:
    """Run a single Opus call via the `claude` CLI. Returns (stdout, returncode)."""
    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--model", model,
                "--effort", effort,
                "--bare",
                "--output-format", "text",
                "--permission-mode", "bypassPermissions",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"ERR: {e!r}", -1


def run_parallel_llm(
    jobs: list,
    worker: Callable,
    max_workers: int = DEFAULT_WORKERS,
    label: str = "",
    progress_every: int = 25,
) -> dict:
    """Run `worker(job)` over `jobs` in a thread pool. Returns a dict keyed by job index.

    `worker` must accept a single `job` argument and return any value; that value
    is stored under the job's index in the returned dict.
    """
    results: dict = {}
    t_start = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker, job): i for i, job in enumerate(jobs)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = ("ERROR", repr(e))
            done += 1
            if progress_every and (done % progress_every == 0 or done == len(jobs)):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done:4d}/{len(jobs)}] {label}  elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
    return results


# ============================================================
# Logging helper
# ============================================================
class StageLogger:
    """Light wrapper for stage logs and verdict files."""

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path
        self._lock = Lock()
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")

    def log(self, msg: str) -> None:
        if self.log_path:
            with self._lock:
                with open(self.log_path, "a") as f:
                    f.write(msg + "\n")
        print(msg)

    def append(self, msg: str) -> None:
        if self.log_path:
            with self._lock:
                with open(self.log_path, "a") as f:
                    f.write(msg + "\n")
