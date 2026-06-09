"""Interactive lattice + relation graph viewer.

Loads a self-contained graph JSON (e.g. modsleuth's `merge_artifact.json`
or `graph.json` from `modsleuth dedup`) and serves an interactive
Cytoscape/dagre frontend on ``http://<host>:<port>/``.

For graphs above a few thousand edges, pass ``--seed`` to pre-prune the
payload to a focused ego-expansion centered on a chosen node; BFS up to
``--depth`` hops admits the highest-relevance neighbors first until
``--target-size`` nodes are captured. Edge-relevance scoring prefers
lineage-bearing relations (``trained_from``, ``trained_on``,
``generated_by``, ``transformed_by``, ``filtered_by``, ``merged_from``,
``composed_from``) and discounts evaluation/citation clutter.

Usage:

    modsleuth viz --source path/to/graph.json --port 8102
    modsleuth viz --source path/to/graph.json --seed "Olmo-3-1025-7B" \\
        --depth 2 --target-size 80 --port 8102

or directly as a module:

    python -m modsleuth.viz --source path/to/graph.json --port 8102
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


# Canonical relations from pipeline (for relation-canonicality coloring)
_CANONICAL_RELATIONS = frozenset({
    "trained_on", "trained_from", "generated_by", "transformed_by",
    "filtered_by", "composed_from", "merged_from", "tokenized_by",
    "deduplicated_by", "decontaminated_by", "released_with",
    "inspired_by", "used_for_ablation", "used_for_evaluation",
})

# Link kinds in priority order — entity-defining anchors first.
# Used to pick the "primary" URL shown in node summaries.
_LINK_KIND_PRIORITY = (
    "hf_model", "hf_dataset", "vendor_docs",
    "paper", "github", "hf_collection", "blog",
)


def _pick_primary_link(links: list[dict]) -> tuple[str, str]:
    """Pick the most-canonical link from an item's links[].
    Prefers entity-anchor kinds (hf_model > hf_dataset > vendor_docs)
    over reference kinds (paper, github, blog), so the UI shows the
    URL that actually defines the artifact."""
    if not isinstance(links, list):
        return "", ""
    best_kind = ""
    best_url = ""
    best_priority = len(_LINK_KIND_PRIORITY)
    for ln in links:
        if not isinstance(ln, dict):
            continue
        url = ln.get("url")
        kind = ln.get("kind") or ""
        if not isinstance(url, str) or not url.strip():
            continue
        try:
            priority = _LINK_KIND_PRIORITY.index(kind)
        except ValueError:
            priority = len(_LINK_KIND_PRIORITY)
        if priority < best_priority:
            best_priority = priority
            best_url = url
            best_kind = kind
    return best_url, best_kind


def _build_edge_record(edge: dict, nodes: dict, event_id: str | None,
                        b_label: str) -> dict:
    """Construct the viewer-side edge record. Tracks both subject and
    object lattice membership and per-edge corroboration metadata when
    present (set by reconcile)."""
    subj = edge.get("subject") or ""
    obj = edge.get("object") or ""
    s_in_lattice = subj in nodes
    o_in_lattice = obj in nodes
    return {
        "subject": subj,
        "subject_id": subj if s_in_lattice else (f"text::{subj}" if subj else ""),
        "subject_in_lattice": s_in_lattice,
        "object": obj,
        "object_id": obj if o_in_lattice else (f"text::{obj}" if obj else ""),
        "object_in_lattice": o_in_lattice,
        "relation": edge.get("relation"),
        "is_canonical_relation": edge.get("relation") in _CANONICAL_RELATIONS,
        "dependency_kind": edge.get("dependency_kind"),
        "description": edge.get("description") or "",
        "description_variants": edge.get("description_variants") or [],
        "anchor_list": edge.get("anchor_list") or [],
        "corroboration_count": edge.get("corroboration_count", 1),
        "source_batch_ids": edge.get("source_batch_ids") or [],
        "subsumes": edge.get("subsumes") or [],
        "operation_id": event_id,
        "batch": b_label,
    }


def _make_off_lattice_node(node_id: str, surface: str) -> dict:
    return {
        "id": node_id,
        "kind": "off_lattice",
        "family": "(off-lattice)",
        "identity": {},
        "aliases": [surface] if surface else [],
        "links": [],
        "n_links": 0,
        "primary_url": "",
        "primary_link_kind": "",
        "description": surface,
        "subsets": [],
        "_generated": False,
        "_synthesized": False,
        "in_degree": 0,
        "out_degree": 0,
    }


def _summarize_mend_artifact(art: dict) -> dict:
    """Brief stats for the stats bar."""
    if not art:
        return {}
    return {
        "lattice_additions": len(art.get("lattice_additions") or []),
        "edge_rewrites": len(art.get("edge_rewrites") or []),
        "edge_drops": len(art.get("edge_drops") or []),
        "auto_resolved_conflicts": len(art.get("auto_resolved_conflicts") or []),
        "confirmed_off_lattice": len(art.get("confirmed_off_lattice") or []),
    }


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>modsleuth &middot; lattice viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ================================================================
   DESIGN TOKENS (Ai2 design language)
   ================================================================ */
:root {
  --bg:             #FAF2E9;
  --bg-surface:     #FDF9F4;
  --bg-hover:       #F3EADB;
  --text:           #032629;
  --text-secondary: #344F4F;
  --text-muted:     #7F8C89;
  --accent-mint:    #0FCB8C;
  --accent-mint-bg: #D6F5EA;
  --accent-pink:    #F0529C;
  --accent-pink-bg: #FDE4F0;
  --neutral-100:    #FDF9F4;
  --neutral-200:    #E8E0D2;
  --neutral-300:    #C9C9C3;
  --neutral-400:    #7F8C89;
  --neutral-500:    #344F4F;
  --neutral-600:    #1C3A3C;
  --neutral-700:    #032629;
  --border:         #C9C9C3;
  --border-light:   #E8E0D2;
  --shadow-sm:      0 1px 2px rgba(0,0,0,0.05);
  --shadow-md:      0 2px 6px rgba(0,0,0,0.08);
  --model-bg:       #DDEEF0;
  --model-text:     #1C3A3C;
  --dataset-bg:     #D6F5EA;
  --dataset-text:   #0A5C3C;
  --off-lattice-bg: #F5EBD9;
  --off-lattice-text:#7A4F00;
  --direct-bg:      #FDE8D8;
  --direct-text:    #B14A00;
  --direct-accent:  #E87040;
  --indirect-bg:    #EDE0F8;
  --indirect-text:  #6E3AAC;
  --indirect-accent:#9B6AD0;
  --font-sans:      'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono:      'Roboto Mono', ui-monospace, monospace;
  --radius-sm:      4px;
  --radius-md:      6px;
  --radius-lg:      8px;
}

/* ================================================================
   RESET & BASE
   ================================================================ */
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%;font-family:var(--font-sans);background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--accent-pink);text-decoration:none}
a:hover{text-decoration:underline}
button{font-family:var(--font-sans);cursor:pointer;border:none;background:none;color:inherit;padding:0;margin:0}
input{font-family:var(--font-sans)}

/* ================================================================
   APP LAYOUT: nav (48) | stats (36) | main
   ================================================================ */
#app{display:grid;grid-template-rows:48px 36px 1fr;height:100vh;overflow:hidden}

/* ---- NAV BAR ---- */
.nav-bar{display:flex;align-items:center;padding:0 16px;background:var(--neutral-700);color:var(--neutral-100);gap:16px;z-index:50}
.nav-brand{font-family:var(--font-mono);font-weight:800;font-size:17px;letter-spacing:.04em;color:var(--accent-mint)}
.nav-tabs{display:flex;gap:2px;margin-left:8px;align-self:stretch;align-items:flex-end}
.nav-tab{padding:7px 18px 8px;border-radius:var(--radius-sm) var(--radius-sm) 0 0;font-weight:600;font-size:13px;color:var(--neutral-300);background:transparent;transition:color .15s,background .15s}
.nav-tab:hover{color:var(--neutral-100)}
.nav-tab.active{color:var(--text);background:var(--bg)}
.nav-search{margin-left:auto;position:relative}
.nav-search input{width:280px;padding:6px 12px 6px 30px;border:1px solid var(--neutral-500);border-radius:var(--radius-md);background:var(--neutral-600);color:var(--neutral-100);font-size:12.5px}
.nav-search input::placeholder{color:var(--neutral-400)}
.nav-search input:focus{outline:none;border-color:var(--accent-mint)}
.nav-search__icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:12px;color:var(--neutral-400);pointer-events:none}
.search-dropdown{position:absolute;top:100%;left:0;right:0;margin-top:4px;background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-md);box-shadow:var(--shadow-md);max-height:400px;overflow-y:auto;z-index:100;display:none;color:var(--text)}
.search-dropdown.open{display:block}
.search-group-label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);font-weight:700;padding:6px 12px 2px}
.search-result{padding:6px 12px;cursor:pointer;font-size:12.5px;border-bottom:1px solid var(--border-light);display:flex;align-items:center;gap:6px;color:var(--text)}
.search-result:hover{background:var(--bg-hover)}
.search-result__name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}
.search-result__meta{font-size:11px;color:var(--text-muted);margin-left:auto;white-space:nowrap}

/* ---- STATS BAR ---- */
.stats-bar{display:flex;align-items:center;padding:0 16px;gap:20px;background:var(--bg-surface);border-bottom:1px solid var(--border-light);font-size:12px;color:var(--text-muted)}
.stat-item strong{color:var(--text);font-weight:700;margin-right:3px}

/* ---- MAIN: content + detail panel ---- */
#main{display:grid;grid-template-columns:1fr 0px;transition:grid-template-columns .25s ease;overflow:hidden}
#main.detail-open{grid-template-columns:1fr 440px}
#view-area{overflow:hidden;position:relative}
.view-pane{display:none;height:100%;overflow:hidden}
.view-pane.active{display:block}

/* ================================================================
   DETAIL PANEL (slide-out right)
   ================================================================ */
#detail-panel{overflow-y:auto;border-left:1px solid var(--border);background:var(--bg-surface);position:relative}
.dp{padding:16px}
.dp__close{position:sticky;top:0;float:right;width:28px;height:28px;border-radius:var(--radius-sm);background:var(--neutral-200);font-size:16px;line-height:28px;text-align:center;color:var(--text-muted);z-index:5;cursor:pointer}
.dp__close:hover{background:var(--neutral-300);color:var(--text)}
.dp__name{font-size:16px;font-weight:800;word-break:break-word;line-height:1.3;margin-bottom:6px}
.dp__badges{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px}
.dp-section{margin-top:16px}
.dp-section__title{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);font-weight:700;margin-bottom:6px}
.dp-desc{font-size:13px;line-height:1.55;color:var(--text-secondary)}
.dp-link{display:flex;align-items:center;gap:6px;padding:3px 0;font-size:12.5px;word-break:break-all}
.dp-link a{color:var(--accent-pink)}
.dp-edge-row{display:flex;align-items:center;gap:6px;padding:6px 0;border-bottom:1px solid var(--border-light);font-size:12px;cursor:pointer}
.dp-edge-row:hover{background:var(--bg-hover);margin:0 -16px;padding-left:16px;padding-right:16px}
.dp-edge-row__sub,.dp-edge-row__obj{font-weight:500;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dp-edge-row__arrow{color:var(--text-muted);font-size:11px;flex-shrink:0}
.dp-edge-expand{padding:8px 0 4px;border-bottom:1px solid var(--border-light)}
.dp-empty{font-size:12px;color:var(--text-muted);font-style:italic}

/* ================================================================
   BADGES, PILLS, CHIPS
   ================================================================ */
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.badge--model{background:var(--model-bg);color:var(--model-text)}
.badge--dataset{background:var(--dataset-bg);color:var(--dataset-text)}
.badge--off_lattice{background:var(--off-lattice-bg);color:var(--off-lattice-text)}
.badge--direct{background:var(--direct-bg);color:var(--direct-text)}
.badge--indirect{background:var(--indirect-bg);color:var(--indirect-text)}
.badge--family{background:var(--neutral-200);color:var(--text-secondary)}
.badge--generated{background:#E8E0D2;color:var(--text-muted);font-style:italic}
.badge--count{background:var(--neutral-200);color:var(--text-secondary);font-family:var(--font-mono);font-size:10px;padding:1px 6px}
.pill{display:inline-block;padding:2px 7px;border-radius:var(--radius-sm);font-size:11px;font-family:var(--font-mono);background:var(--neutral-200);color:var(--text-secondary)}
.chip{display:inline-block;padding:2px 8px;margin:2px;border-radius:var(--radius-sm);font-size:11.5px;background:var(--neutral-200);color:var(--text-secondary)}
.link-badge{display:inline-flex;align-items:center;gap:2px;padding:1px 6px;border-radius:var(--radius-sm);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.02em}
.link-badge--hf_model{background:var(--model-bg);color:var(--model-text)}
.link-badge--hf_dataset{background:var(--dataset-bg);color:var(--dataset-text)}
.link-badge--hf_collection{background:#FFF3CD;color:#856404}
.link-badge--paper{background:var(--indirect-bg);color:var(--indirect-text)}
.link-badge--github{background:#E8E8E8;color:#24292E}
.link-badge--blog{background:var(--direct-bg);color:var(--direct-text)}
.link-badge--vendor_docs{background:#D6EAF8;color:#1B4F72}

/* ================================================================
   ANCHOR BLOCKQUOTE
   ================================================================ */
.anchor-q{border-left:3px solid var(--accent-mint);background:var(--bg);padding:8px 12px;margin:6px 0;border-radius:0 var(--radius-sm) var(--radius-sm) 0;font-size:12px;line-height:1.5}
.anchor-q__src{font-family:var(--font-mono);font-size:11px;color:var(--accent-pink);word-break:break-all}
.anchor-q__src a{color:var(--accent-pink)}
.anchor-q__pos{font-family:var(--font-mono);font-size:10.5px;color:var(--text-muted);margin-top:1px}
.anchor-q__exc{font-style:italic;margin-top:4px;color:var(--text-secondary)}
.anchor-q__exp{margin-top:4px;color:var(--text)}

/* ================================================================
   GRAPH VIEW
   ================================================================ */
.graph-view{position:relative;height:100%}
.graph-canvas{width:100%;height:100%}
.graph-toolbar{position:absolute;top:12px;left:12px;display:flex;align-items:center;gap:4px;z-index:10}
.gbtn{padding:5px 12px;font-size:11.5px;font-weight:600;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg-surface);color:var(--text);box-shadow:var(--shadow-sm);transition:background .1s,border-color .1s}
.gbtn:hover{background:var(--bg-hover)}
.gbtn.active{background:var(--accent-mint);color:var(--neutral-700);border-color:var(--accent-mint)}
.spread-control{display:flex;align-items:center;gap:4px;margin-left:8px;padding:3px 10px;background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-sm);box-shadow:var(--shadow-sm)}
.spread-control label{font-size:10px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
.spread-control input[type=range]{width:90px;height:4px;accent-color:var(--accent-mint);cursor:pointer}
.graph-filters{position:absolute;top:12px;right:12px;z-index:10;background:var(--bg-surface);border:1px solid var(--border-light);border-radius:var(--radius-md);padding:10px 12px;box-shadow:var(--shadow-sm);max-width:260px;max-height:55vh;overflow-y:auto;font-size:12px}
.fg-label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);font-weight:700;margin:8px 0 4px}
.fg-label:first-child{margin-top:0}
.filter-chip{display:inline-block;padding:2px 8px;margin:2px;border:1px solid var(--border);border-radius:12px;font-size:11px;cursor:pointer;user-select:none;background:var(--bg-surface);transition:background .1s,border-color .1s,opacity .1s}
.filter-chip:hover{border-color:var(--accent-mint)}
.filter-chip.active{background:var(--accent-mint);color:var(--neutral-700);border-color:var(--accent-mint)}
.filter-chip.chip-empty{opacity:.45;cursor:not-allowed}
.filter-chip.chip-empty:hover{border-color:var(--border)}
.filter-chip .chip-count{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);margin-left:2px}
.filter-chip.active .chip-count{color:var(--neutral-700)}
.graph-filters select{width:100%;padding:4px 6px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:12px;background:var(--bg);margin-top:2px}
.graph-filters label{font-size:11.5px;display:flex;align-items:center;gap:4px;margin-top:6px;cursor:pointer}
.graph-status{position:absolute;bottom:12px;left:12px;background:var(--bg-surface);border:1px solid var(--border-light);border-radius:var(--radius-sm);padding:4px 10px;font-size:11px;color:var(--text-muted);box-shadow:var(--shadow-sm);z-index:10}

/* ================================================================
   OPERATIONS VIEW
   ================================================================ */
.ops-view{padding:16px 20px;overflow-y:auto;height:100%}
.ops-search{margin-bottom:14px}
.ops-search input{width:100%;max-width:500px;padding:7px 12px;border:1px solid var(--border);border-radius:var(--radius-md);font-size:13px;background:var(--bg-surface)}
.ops-search input:focus{outline:none;border-color:var(--accent-mint)}
.op-card{background:var(--bg-surface);border:1px solid var(--border-light);border-radius:var(--radius-lg);margin-bottom:8px;box-shadow:var(--shadow-sm);overflow:hidden}
.op-card__hd{padding:10px 14px;cursor:pointer;display:flex;align-items:flex-start;gap:8px;transition:background .1s}
.op-card__hd:hover{background:var(--bg-hover)}
.op-card__chev{font-size:10px;color:var(--text-muted);transition:transform .15s;flex-shrink:0;margin-top:3px;font-family:var(--font-mono)}
.op-card.expanded .op-card__chev{transform:rotate(90deg)}
.op-card__desc{font-size:12.5px;line-height:1.5;flex:1;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.op-card.expanded .op-card__desc{-webkit-line-clamp:unset}
.op-card__meta{display:flex;gap:6px;align-items:center;flex-shrink:0}
.op-card__body{display:none;padding:0 14px 12px;border-top:1px solid var(--border-light)}
.op-card.expanded .op-card__body{display:block}
.op-edge{padding:6px 0;border-bottom:1px solid var(--border-light);font-size:12px}
.op-edge__row{display:flex;align-items:center;gap:6px;cursor:pointer}
.op-edge__row:hover .op-edge__sub,.op-edge__row:hover .op-edge__obj{color:var(--accent-pink)}
.op-edge__sub,.op-edge__obj{font-weight:600;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.op-edge__arrow{color:var(--text-muted);font-size:10px}
.op-edge__detail{padding:6px 0 2px;font-size:12px;color:var(--text-secondary);line-height:1.5}

/* ================================================================
   SCROLLBAR STYLING
   ================================================================ */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--neutral-300);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--neutral-400)}
</style>
</head>
<body>
<div id="app">

  <!-- ======== NAV BAR ======== -->
  <nav class="nav-bar">
    <span class="nav-brand">modsleuth</span>
    <div class="nav-tabs">
      <button class="nav-tab active" data-view="graph">Graph</button>
      <button class="nav-tab" data-view="operations">Operations</button>
    </div>
    <div class="nav-search">
      <span class="nav-search__icon">&#x1F50D;</span>
      <input type="search" id="global-search" placeholder="Search nodes, edges, operations..." />
      <div class="search-dropdown" id="search-dropdown"></div>
    </div>
  </nav>

  <!-- ======== STATS BAR ======== -->
  <div class="stats-bar" id="stats-bar"></div>

  <!-- ======== MAIN AREA ======== -->
  <div id="main">
    <div id="view-area">
      <div id="view-graph" class="view-pane active"></div>
      <div id="view-operations" class="view-pane"></div>
    </div>
    <div id="detail-panel"></div>
  </div>

</div>

<!-- CDN: Cytoscape + dagre -->
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>

<script>
"use strict";

/* ================================================================
   STATE
   ================================================================ */
const S = {
  raw: null,
  view: 'lattice',
  detailOpen: false,
  graph: {
    cy: null, inited: false,
    // hideUnlinked, hideIsolated, hideOffLattice are always-on by
    // design — the graph view shows lattice items connected via
    // canonical edges. To inspect free-text endpoints, use the
    // Operations view or click a node to see all its edges.
    filters: { kinds: new Set(), depKinds: new Set(), rels: new Set(), family: '', showLattice: true },
    ego: null, spread: 7,
  },
  ops: { search: '', expanded: new Set() },
};

/* ================================================================
   HELPERS
   ================================================================ */
function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s ?? '');
  return d.innerHTML;
}
function truncate(s, n) { return (!s || s.length <= n) ? (s||'') : s.slice(0,n-1) + '…'; }
function isUrl(s) { return typeof s === 'string' && /^https?:\/\//.test(s); }
function shortSource(s) {
  if (!s) return '';
  if (isUrl(s)) { try { return new URL(s).hostname + new URL(s).pathname.slice(0,40); } catch(e) { return s.slice(0,60); } }
  const parts = s.split('/');
  return parts.slice(-2).join('/');
}

function badgeHTML(kind, text) {
  const cls = ({model:'model',dataset:'dataset',off_lattice:'off_lattice',direct:'direct',indirect:'indirect',family:'family',generated:'generated'})[kind] || 'family';
  return `<span class="badge badge--${cls}">${esc(text || kind)}</span>`;
}
function linkBadgeHTML(kind) {
  const k = (kind||'').replace(/\s+/g,'_');
  return `<span class="link-badge link-badge--${esc(k)}">${esc(kind)}</span>`;
}
function pillHTML(k, v) { return `<span class="pill">${esc(k)}=${esc(v)}</span>`; }
function chipHTML(t) { return `<span class="chip">${esc(t)}</span>`; }

function anchorHTML(a) {
  if (!a) return '';
  const src = a.source || '';
  const srcLink = isUrl(src) ? `<a href="${esc(src)}" target="_blank">${esc(shortSource(src))}</a>` : esc(shortSource(src));
  const pos = a.position ? `<div class="anchor-q__pos">${esc(a.position)}</div>` : '';
  const exc = a.excerpt ? `<div class="anchor-q__exc">"${esc(truncate(a.excerpt, 300))}"</div>` : '';
  const exp = a.explanation ? `<div class="anchor-q__exp">${esc(a.explanation)}</div>` : '';
  return `<div class="anchor-q"><div class="anchor-q__src">${srcLink}</div>${pos}${exc}${exp}</div>`;
}

/* ================================================================
   FETCH DATA
   ================================================================ */
async function fetchData() {
  const r = await fetch('/api/graph');
  S.raw = await r.json();
  S.view = 'graph';
  renderStats();
  initGraph();
  renderOps();
  setupSearch();
}

/* ================================================================
   VIEW SWITCHING
   ================================================================ */
function switchView(v) {
  S.view = v;
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.view === v));
  document.querySelectorAll('.view-pane').forEach(p => p.classList.toggle('active', p.id === 'view-' + v));
  if (v === 'graph' && !S.graph.inited) initGraph();
  if (v === 'graph' && S.graph.cy) S.graph.cy.resize();
}
document.querySelectorAll('.nav-tab').forEach(t => t.addEventListener('click', () => switchView(t.dataset.view)));

/* ================================================================
   STATS BAR
   ================================================================ */
function renderStats() {
  const s = S.raw.stats;
  // Live count is updated by refreshStatsBarLive() after every
  // rebuildGraph() so the displayed numbers always match the graph.
  document.getElementById('stats-bar').innerHTML =
    `<span class="stat-item"><strong>${s.lattice_node_count}</strong> items</span>` +
    `<span class="stat-item"><strong>${s.edge_count}</strong> edges</span>` +
    `<span class="stat-item" id="stat-live"><strong>0</strong> nodes &middot; <strong>0</strong> edges in graph</span>`;
}

/* ================================================================
   DETAIL PANEL
   ================================================================ */
function openDetail(html) {
  const dp = document.getElementById('detail-panel');
  dp.innerHTML = html;
  document.getElementById('main').classList.add('detail-open');
  S.detailOpen = true;
}
function closeDetail() {
  document.getElementById('main').classList.remove('detail-open');
  document.getElementById('detail-panel').innerHTML = '';
  S.detailOpen = false;
  if (S.graph.cy) S.graph.cy.resize();
}

function nodeDetailHTML(n) {
  const identity = n.identity || {};
  const facets = Object.entries(identity).map(([k,v]) => pillHTML(k,v)).join('');
  const aliases = (n.aliases||[]).map(a => chipHTML(a)).join('');
  const links = (n.links||[]).map(l => `<div class="dp-link">${linkBadgeHTML(l.kind)} <a href="${esc(l.url)}" target="_blank">${esc(l.url)}</a></div>`).join('');
  const subsets = (n.subsets||[]).length ? `<div class="dp-section"><div class="dp-section__title">Subsets (${n.subsets.length})</div><div>${n.subsets.map(s=>chipHTML(s)).join('')}</div></div>` : '';
  const desc = n.description ? `<div class="dp-section"><div class="dp-section__title">Description</div><p class="dp-desc">${esc(n.description)}</p></div>` : '';
  const gen = n._generated ? ' ' + badgeHTML('generated','generated') : '';
  // Lookup edges by canonical id (lattice formal_name OR text::<surface>)
  const outEdges = S.raw.edges.filter(e => e.subject_id === n.id);
  const inEdges = S.raw.edges.filter(e => e.object_id === n.id);

  function edgeRowsHTML(edges, dir) {
    if (!edges.length) return `<div class="dp-empty">None</div>`;
    return edges.slice(0,50).map((e,i) => {
      const sub = dir === 'out' ? n.id : (e.subject || '');
      const obj = dir === 'out' ? (e.object || '') : n.id;
      const id = `dp-edge-${dir}-${i}`;
      // Annotate corroboration count when > 1 (multiple sources)
      const corrBadge = (e.corroboration_count||1) > 1
        ? `<span class="badge badge--count" title="Corroborated by ${e.corroboration_count} independent sources">${e.corroboration_count}×</span>` : '';
      const relBadge = e.is_canonical_relation
        ? `<span class="pill">${esc(e.relation)}</span>`
        : `<span class="pill" style="background:#F4E8D0;color:#5C3D14" title="Coined relation (outside canonical 14)">${esc(e.relation)}</span>`;
      return `<div class="dp-edge-row" onclick="document.getElementById('${id}').style.display=document.getElementById('${id}').style.display==='none'?'block':'none'">
        <span class="dp-edge-row__sub" title="${esc(sub)}">${esc(truncate(sub,30))}</span>
        <span class="dp-edge-row__arrow">→</span>
        <span>${badgeHTML(e.dependency_kind, e.dependency_kind)} ${relBadge} ${corrBadge}</span>
        <span class="dp-edge-row__arrow">→</span>
        <span class="dp-edge-row__obj" title="${esc(obj)}">${esc(truncate(obj,30))}</span>
      </div>
      <div id="${id}" class="dp-edge-expand" style="display:none">
        <div class="dp-desc">${esc(e.description)}</div>
        ${(e.description_variants||[]).length ? `<div class="dp-section__title" style="margin-top:6px">Other prose framings (${e.description_variants.length})</div>${(e.description_variants||[]).slice(0,3).map(v=>`<div class="dp-desc" style="opacity:.8;font-size:11px">${esc(v)}</div>`).join('')}` : ''}
        ${(e.subsumes||[]).length ? `<div class="dp-section__title" style="margin-top:6px">Subsumes (vague edges that folded in: ${e.subsumes.length})</div>${(e.subsumes||[]).slice(0,5).map(s=>`<div class="dp-empty" style="font-size:11px">${esc(s.subject||'?')} --${esc(s.relation||'?')}--> ${esc(s.object||'?')}</div>`).join('')}` : ''}
        ${(e.anchor_list||[]).map(a => anchorHTML(a)).join('')}
      </div>`;
    }).join('') + (edges.length > 50 ? `<div class="dp-empty">… ${edges.length - 50} more</div>` : '');
  }

  // Off-lattice node: show the surface form prominently and skip
  // the lattice-only sections (no identity, no links, no subsets).
  if (n.kind === 'off_lattice') {
    return `<div class="dp">
      <button class="dp__close" onclick="closeDetail()">&times;</button>
      <div class="dp__name">${esc(n.description || n.id)}</div>
      <div class="dp__badges">${badgeHTML('off_lattice', 'free-text')}</div>
      <div class="dp-section"><div class="dp-empty">This endpoint isn't in the lattice. Either it's an internal/unreleased artifact, a codebase / methodology label out of model+dataset scope, or a name the lattice missed.</div></div>
      <div class="dp-section"><div class="dp-section__title">Outgoing (${outEdges.length})</div>${edgeRowsHTML(outEdges, 'out')}</div>
      <div class="dp-section"><div class="dp-section__title">Incoming (${inEdges.length})</div>${edgeRowsHTML(inEdges, 'in')}</div>
    </div>`;
  }

  return `<div class="dp">
    <button class="dp__close" onclick="closeDetail()">&times;</button>
    <div class="dp__name">${esc(n.id)}</div>
    <div class="dp__badges">${badgeHTML(n.kind, n.kind)} ${badgeHTML('family', n.family)}${gen}</div>
    <div class="dp-section"><div class="dp-section__title">Identity</div><div>${facets || '<span class="dp-empty">none</span>'}</div></div>
    ${aliases ? `<div class="dp-section"><div class="dp-section__title">Aliases (${(n.aliases||[]).length})</div><div>${aliases}</div></div>` : ''}
    ${links ? `<div class="dp-section"><div class="dp-section__title">Links (${(n.links||[]).length})</div>${links}</div>` : ''}
    ${desc}
    ${subsets}
    <div class="dp-section"><div class="dp-section__title">Outgoing (${outEdges.length})</div>${edgeRowsHTML(outEdges, 'out')}</div>
    <div class="dp-section"><div class="dp-section__title">Incoming (${inEdges.length})</div>${edgeRowsHTML(inEdges, 'in')}</div>
  </div>`;
}

function showNodeDetail(nodeId) {
  const n = S.raw.nodes.find(x => x.id === nodeId);
  if (!n) return;
  openDetail(nodeDetailHTML(n));
}

function findOperationForEdge(edge) {
  if (edge.operation_id) return S.raw.operations.find(o => o.id === edge.operation_id) || null;
  const subj = edge.subject, rel = edge.relation, obj = edge.object;
  for (const op of S.raw.operations) {
    const triples = op.edge_triples || [];
    if (triples.some(t => t.subject === subj && t.relation === rel && t.object === obj)) return op;
  }
  return null;
}

function handleGraphEdgeClick(cyEdge) {
  const cy = S.graph.cy;
  cy.edges().removeClass('op-highlight op-highlight-peer');
  const idx = cyEdge.data('_idx');
  const edge = S.raw.edges[idx];
  if (!edge) return;
  cyEdge.addClass('op-highlight');
  const op = findOperationForEdge(edge);
  if (op) {
    const peerTriples = new Set((op.edge_triples||[]).map(t => t.subject+'||'+t.relation+'||'+t.object));
    const selfKey = edge.subject+'||'+edge.relation+'||'+edge.object;
    peerTriples.delete(selfKey);
    if (peerTriples.size > 0) {
      cy.edges().forEach(e => {
        if (e.id() === cyEdge.id() || e.data('edgeType') !== 'relation') return;
        const pe = S.raw.edges[e.data('_idx')];
        if (pe && peerTriples.has(pe.subject+'||'+pe.relation+'||'+pe.object)) {
          e.addClass('op-highlight-peer');
        }
      });
    }
  }
  openDetail(edgeDetailHTML(edge, op));
}

function edgeDetailHTML(edge, op) {
  const opId = op ? op.id : null;
  const peerEdges = opId ? S.raw.edges.filter(e => {
    if (e === edge) return false;
    const eOp = findOperationForEdge(e);
    return eOp && eOp.id === opId;
  }) : [];
  return `<div class="dp">
    <button class="dp__close" onclick="closeDetail()">&times;</button>
    <div class="dp-section__title" style="margin-top:0">Edge</div>
    <div class="dp__name" style="font-size:14px">${esc(edge.subject)}</div>
    <div style="display:flex;align-items:center;gap:6px;margin:6px 0">
      <span class="dp-edge-row__arrow" style="font-size:14px">→</span>
      ${badgeHTML(edge.dependency_kind, edge.dependency_kind)}
      <span class="pill" style="font-size:12px">${esc(edge.relation)}</span>
      <span class="dp-edge-row__arrow" style="font-size:14px">→</span>
    </div>
    <div class="dp__name" style="font-size:14px">${esc(edge.object)}</div>
    <div class="dp-section"><div class="dp-section__title">Description</div><p class="dp-desc">${esc(edge.description)}</p></div>
    <div class="dp-section"><div class="dp-section__title">Edge Evidence (${(edge.anchor_list||[]).length})</div>
      ${(edge.anchor_list||[]).length ? (edge.anchor_list||[]).map(a => anchorHTML(a)).join('') : '<div class="dp-empty">No anchors</div>'}
    </div>
    ${op ? `<div class="dp-section"><div class="dp-section__title">Operation</div>
      <p class="dp-desc">${esc(op.description)}</p>
      ${(op.anchor_list||[]).length ? `<div style="margin-top:6px"><div class="dp-section__title">Operation Evidence (${op.anchor_list.length})</div>${op.anchor_list.map(a => anchorHTML(a)).join('')}</div>` : ''}
    </div>` : ''}
    ${peerEdges.length ? `<div class="dp-section"><div class="dp-section__title">Other edges in this operation (${peerEdges.length})</div>
      ${peerEdges.map(pe => `<div class="dp-edge-row" style="cursor:default">
        <span class="dp-edge-row__sub" title="${esc(pe.subject)}">${esc(truncate(pe.subject,25))}</span>
        <span class="dp-edge-row__arrow">→</span>
        ${badgeHTML(pe.dependency_kind, pe.dependency_kind)}
        <span class="pill">${esc(pe.relation)}</span>
        <span class="dp-edge-row__arrow">→</span>
        <span class="dp-edge-row__obj" title="${esc(pe.object)}">${esc(truncate(pe.object,25))}</span>
      </div>`).join('')}
    </div>` : ''}
  </div>`;
}

/* ================================================================
   GRAPH VIEW (Cytoscape + dagre)
   ================================================================ */
// Relation-class taxonomy for edge styling. Lineage relations are
// the structural backbone (training, derivation); context relations
// are weaker provenance hints; noise relations are evaluation/citation
// chatter that crowds the view. Mirrors the weight signs in
// Python's _EDGE_RELEVANCE table.
const RELATION_CLASS = {
  trained_from:'lineage', trained_on:'lineage', generated_by:'lineage',
  transformed_by:'lineage', filtered_by:'lineage', merged_from:'lineage',
  composed_from:'lineage', tokenized_by:'lineage', deduplicated_by:'lineage',
  decontaminated_by:'lineage',
  released_with:'context', inspired_by:'context',
  used_for_evaluation:'noise', used_for_ablation:'noise', cited_as_baseline:'noise',
};
function relationClass(r) { return RELATION_CLASS[r] || 'context'; }

const CY_STYLE = [
  { selector: 'node', style: {
    'label': 'data(label)', 'font-family': "'Manrope',sans-serif", 'font-size': '7px', 'font-weight': 500,
    'color': '#344F4F', 'text-valign': 'bottom', 'text-halign': 'center', 'text-margin-y': 3,
    'text-max-width': '90px', 'text-wrap': 'ellipsis', 'width': 'data(size)', 'height': 'data(size)',
    'background-color': 'data(color)', 'border-width': 1, 'border-color': '#C9C9C3', 'overlay-padding': 1,
    'min-zoomed-font-size': 10,
    'text-background-color': '#FAF2E9', 'text-background-opacity': 0.7, 'text-background-padding': '1px',
  }},
  // Models stay dark teal. Datasets switch to warm amber so kind is
  // readable at a glance even when zoomed out (the shape difference
  // alone disappears below ~1.5x zoom).
  { selector: 'node[kind="model"]', style: { 'shape': 'ellipse', 'background-color': '#1C3A3C', 'border-color': '#0FCB8C' }},
  { selector: 'node[kind="dataset"]', style: { 'shape': 'round-rectangle', 'background-color': '#C9803B', 'border-color': '#7A4810' }},
  { selector: 'node[kind="off_lattice"]', style: { 'shape': 'diamond', 'background-color': '#7A4F00', 'border-color': '#C9C9C3', 'opacity': 0.7 }},
  // Seed of the ego-expansion: thick magenta ring + always-on label so
  // it's obvious which node anchors the view. Higher z-index keeps it
  // on top of overlapping neighbors.
  { selector: 'node[?is_seed]', style: {
    'border-color': '#F0529C', 'border-width': 3.5, 'width': 24, 'height': 24,
    'font-size': '11px', 'font-weight': 800, 'color': '#032629',
    'text-background-opacity': 0.95, 'min-zoomed-font-size': 0, 'z-index': 25,
  }},
  // Bridge / context nodes pulled in by a kind filter to keep lineage
  // intact. Rendered translucent so the user's selected kind still
  // visually dominates.
  { selector: 'node[?dimmed]', style: { 'opacity': 0.35 }},
  { selector: 'node.hover', style: { 'border-color': '#F0529C', 'border-width': 1.5, 'z-index': 20,
    'font-size': '9px', 'font-weight': 700, 'color': '#032629', 'min-zoomed-font-size': 0,
    'text-background-opacity': 0.85 }},
  { selector: 'node:selected', style: { 'border-color': '#0FCB8C', 'border-width': 2, 'z-index': 30,
    'overlay-color': '#0FCB8C', 'overlay-opacity': 0.08,
    'font-size': '9px', 'font-weight': 700, 'color': '#032629', 'min-zoomed-font-size': 0,
    'text-background-opacity': 0.85 }},
  // Base edge: bumped arrow-scale + opacity so direction is visible
  // without selecting the edge. Lineage edges keep the depKind colors;
  // noise/context selectors fade non-structural relations.
  { selector: 'edge', style: {
    'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'target-arrow-color': 'data(color)',
    'line-color': 'data(color)', 'width': 0.9, 'opacity': 0.65, 'arrow-scale': 1.2,
  }},
  { selector: 'edge[depKind="direct"]', style: { 'line-color': '#E87040', 'target-arrow-color': '#E87040', 'line-style': 'solid' }},
  { selector: 'edge[depKind="indirect"]', style: { 'line-color': '#9B6AD0', 'target-arrow-color': '#9B6AD0', 'line-style': 'dashed', 'line-dash-pattern': [6,4] }},
  // Context relations (released_with, inspired_by) — muted grey-green.
  { selector: 'edge[relClass="context"]', style: {
    'line-color': '#8FA89F', 'target-arrow-color': '#8FA89F',
    'opacity': 0.4, 'width': 0.7,
  }},
  // Noise relations (evaluation / ablation / cited_as_baseline) —
  // very faded so they don't drown out lineage.
  { selector: 'edge[relClass="noise"]', style: {
    'line-color': '#C9C9C3', 'target-arrow-color': '#C9C9C3',
    'opacity': 0.18, 'width': 0.5, 'arrow-scale': 0.8,
  }},
  { selector: 'edge:selected', style: { 'width': 2, 'opacity': 1, 'label': 'data(relation)',
    'font-size': '6px', 'font-family': "'Roboto Mono',monospace", 'color': '#344F4F', 'text-rotation': 'autorotate', 'text-margin-y': -6, 'z-index': 20 }},
  // Lattice hierarchy edges (identity subsumption)
  { selector: 'edge[edgeType="lattice"]', style: {
    'line-color': '#C9C9C3', 'target-arrow-color': '#C9C9C3', 'line-style': 'solid',
    'width': 1, 'opacity': 0.4, 'target-arrow-shape': 'triangle', 'arrow-scale': 0.8,
  }},
  // Operation-highlight class
  { selector: 'edge.op-highlight', style: { 'width': 2.5, 'opacity': 1, 'line-color': '#F0529C', 'target-arrow-color': '#F0529C', 'z-index': 15 }},
  { selector: 'edge.op-highlight-peer', style: { 'width': 1.2, 'opacity': 0.55, 'line-color': '#F7A4CA', 'target-arrow-color': '#F7A4CA', 'z-index': 12,
    'line-style': 'dashed', 'line-dash-pattern': [5,3] }},
];

function initGraph() {
  if (S.graph.inited) return;
  S.graph.inited = true;
  const el = document.getElementById('view-graph');
  el.innerHTML = `<div class="graph-view">
    <div class="graph-toolbar" id="graph-toolbar">
      <button class="gbtn" data-layout="dagre">Dagre</button>
      <button class="gbtn active" data-layout="cose">Force</button>
      <button class="gbtn" id="graph-reset">Reset</button>
      <button class="gbtn" id="graph-ego">Ego</button>
      <div class="spread-control">
        <label for="graph-spread">Spread</label>
        <input type="range" id="graph-spread" min="1" max="10" value="7" step="1" />
      </div>
    </div>
    <div class="graph-filters" id="graph-filters"></div>
    <div class="graph-canvas" id="cy-container"></div>
    <div class="graph-status" id="graph-status"></div>
  </div>`;

  cytoscape.use(cytoscapeDagre);
  S.graph.cy = cytoscape({ container: document.getElementById('cy-container'), style: CY_STYLE, elements: [],
    minZoom: 0.08, maxZoom: 4, wheelSensitivity: 0.3 });
  S.graph.cy.on('tap', 'node', e => {
    S.graph.cy.edges().removeClass('op-highlight op-highlight-peer');
    const nid = e.target.id();
    if (document.getElementById('graph-ego').classList.contains('active')) {
      S.graph.ego = nid;
      rebuildGraph();
    }
    showNodeDetail(nid);
  });
  S.graph.cy.on('tap', 'edge', e => {
    if (e.target.data('edgeType') === 'lattice') return;
    handleGraphEdgeClick(e.target);
  });
  S.graph.cy.on('tap', e => { if (e.target === S.graph.cy) { S.graph.cy.edges().removeClass('op-highlight op-highlight-peer'); } });
  S.graph.cy.on('mouseover', 'node', e => e.target.addClass('hover'));
  S.graph.cy.on('mouseout', 'node', e => e.target.removeClass('hover'));

  setupGraphFilters();
  rebuildGraph();

  document.querySelectorAll('#graph-toolbar .gbtn[data-layout]').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('#graph-toolbar .gbtn[data-layout]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    runLayout(btn.dataset.layout);
  }));
  document.getElementById('graph-reset').addEventListener('click', () => { S.graph.ego = null; document.getElementById('graph-ego').classList.remove('active'); rebuildGraph(); });
  document.getElementById('graph-ego').addEventListener('click', () => {
    const btn = document.getElementById('graph-ego');
    if (S.graph.ego) { S.graph.ego = null; btn.classList.remove('active'); rebuildGraph(); }
    else { btn.classList.add('active'); }
  });
  document.getElementById('graph-spread').addEventListener('input', e => {
    S.graph.spread = parseInt(e.target.value, 10);
    const layout = document.querySelector('#graph-toolbar .gbtn.active[data-layout]');
    runLayout(layout ? layout.dataset.layout : 'cose');
  });
}

function setupGraphFilters() {
  const s = S.raw.stats;

  let html = `<div class="fg-label">Kind</div>`;
  ['model','dataset'].forEach(k => html += `<span class="filter-chip" data-ftype="kind" data-fval="${k}"><span class="chip-label">${k}</span> <span class="chip-count">(0)</span></span>`);
  html += `<div class="fg-label">Dependency</div>`;
  ['direct','indirect'].forEach(k => html += `<span class="filter-chip" data-ftype="dep" data-fval="${k}"><span class="chip-label">${k}</span> <span class="chip-count">(0)</span></span>`);
  html += `<div class="fg-label">Relation</div>`;
  // Build a chip per relation present in the data. Counts will be
  // populated by refreshFilterChipCounts() after every rebuildGraph().
  s.relations.forEach(([r, _total]) => {
    html += `<span class="filter-chip" data-ftype="rel" data-fval="${r}"><span class="chip-label">${r}</span> <span class="chip-count">(0)</span></span>`;
  });
  html += `<div class="fg-label">Family</div><select id="graph-family-select"><option value="">All</option>`;
  s.families.forEach(([f,n]) => html += `<option value="${esc(f)}">${esc(f)} (${n})</option>`);
  html += `</select>`;
  html += `<label><input type="checkbox" id="graph-show-lattice" checked /> Show lattice hierarchy</label>`;
  document.getElementById('graph-filters').innerHTML = html;

  document.querySelectorAll('.filter-chip').forEach(fc => fc.addEventListener('click', () => {
    // Empty chips are no-ops under the current filter mix; clicking
    // them would just dead-end the user. Skip the click.
    if (fc.classList.contains('chip-empty')) return;
    fc.classList.toggle('active');
    const {ftype, fval} = fc.dataset;
    const set = ftype==='kind' ? S.graph.filters.kinds : ftype==='dep' ? S.graph.filters.depKinds : S.graph.filters.rels;
    set.has(fval) ? set.delete(fval) : set.add(fval);
    rebuildGraph();
  }));
  document.getElementById('graph-family-select').addEventListener('change', e => { S.graph.filters.family = e.target.value; rebuildGraph(); });
  document.getElementById('graph-show-lattice').addEventListener('change', e => { S.graph.filters.showLattice = e.target.checked; rebuildGraph(); });
  // Initial paint of chip counts before any user interaction
  refreshFilterChipCounts();
}

function buildGraphElements() {
  const f = S.graph.filters;
  const nodeMap = {};
  S.raw.nodes.forEach(n => nodeMap[n.id] = n);

  // Step 1: filter nodes. By design (not configurable), the graph
  // view shows ONLY lattice items with at least one verified link.
  // Off-lattice (free-text) endpoints are always hidden — they're
  // visible in the Operations view and via node-detail panels but
  // not in the lattice-canonical graph. Unlinked items are also
  // hidden (a lattice item with no anchor URL is a placeholder, not
  // a verifiable artifact).
  // Step 1a: build the "universe" set under always-on filters only
  // (lattice + n_links > 0). User filters (kind / family / dep / rel)
  // are NOT applied here — connectivity is computed on the full
  // pipeline-visible graph so that, e.g., GPT models stay in the
  // main component even when the user filters to kind=model
  // (because the underlying connection through datasets still exists).
  const universe = new Set();
  for (const n of S.raw.nodes) {
    if (n.kind === 'off_lattice') continue;
    if (n.n_links === 0) continue;
    universe.add(n.id);
  }

  // Step 1b: compute connectivity on the universe using ALL canonical
  // edges + lattice subsumption pairs (no dep/rel filter applied here
  // either — the goal is to find nodes that are part of the lineage
  // backbone regardless of which slice the user is currently viewing).
  const universeEdges = S.raw.edges.filter(e =>
    universe.has(e.subject_id) && universe.has(e.object_id)
  );
  const universeLattice = []; // {a, b}
  for (const grp of S.raw.lattice_groups) {
    const items = (grp.items||[]).filter(it => universe.has(it.formal_name));
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        const a = items[i], b = items[j];
        if (identitySubsumes(a.identity, b.identity) && isDirectParent(a.identity, b.identity, items)) {
          universeLattice.push({a: a.formal_name, b: b.formal_name});
        }
      }
    }
  }
  const universeAdj = new Map();
  function uAdd(u, v) {
    if (!universeAdj.has(u)) universeAdj.set(u, new Set());
    if (!universeAdj.has(v)) universeAdj.set(v, new Set());
    universeAdj.get(u).add(v); universeAdj.get(v).add(u);
  }
  for (const id of universe) { if (!universeAdj.has(id)) universeAdj.set(id, new Set()); }
  for (const e of universeEdges) uAdd(e.subject_id, e.object_id);
  for (const le of universeLattice) uAdd(le.a, le.b);
  const seenU = new Set();
  const universeComponents = [];
  for (const start of universe) {
    if (seenU.has(start)) continue;
    const comp = []; const stack = [start];
    while (stack.length) {
      const v = stack.pop();
      if (seenU.has(v)) continue;
      seenU.add(v); comp.push(v);
      const nbs = universeAdj.get(v);
      if (nbs) for (const w of nbs) if (!seenU.has(w)) stack.push(w);
    }
    universeComponents.push(comp);
  }
  universeComponents.sort((a, b) => b.length - a.length);
  const mainComp = new Set(universeComponents[0] || []);

  // Step 2: apply USER filters (kind / family) within the main connected
  // component. Smaller fragments are dropped — they tend to be ego-
  // expansion crumbs (single isolated mentions, 2-node pairs) that
  // visually clutter the lineage backbone without adding signal.
  // Kind is a "primary + dimmed bridge" filter rather than a hard
  // exclude — strict kind-only would empty the graph because datasets
  // only connect to models (and vice versa) so Step 4's isolated-prune
  // would drop everything. Bridge nodes get rendered at reduced opacity
  // so the user's selected kind still pops.
  const vis = new Set();
  const dimmed = new Set();
  const primary = new Set();
  for (const id of mainComp) {
    const n = nodeMap[id];
    if (!n) continue;
    if (f.kinds.size && !f.kinds.has(n.kind)) continue;
    if (f.family && n.family !== f.family) continue;
    vis.add(id); primary.add(id);
  }
  if (f.kinds.size) {
    for (const e of S.raw.edges) {
      const sIn = primary.has(e.subject_id);
      const oIn = primary.has(e.object_id);
      if (sIn && !oIn && mainComp.has(e.object_id)) {
        const n2 = nodeMap[e.object_id];
        if (n2 && (!f.family || n2.family === f.family)) {
          vis.add(e.object_id); dimmed.add(e.object_id);
        }
      }
      if (oIn && !sIn && mainComp.has(e.subject_id)) {
        const n1 = nodeMap[e.subject_id];
        if (n1 && (!f.family || n1.family === f.family)) {
          vis.add(e.subject_id); dimmed.add(e.subject_id);
        }
      }
    }
  }

  // Step 3: filter dependency edges by dep/rel + visibility.
  let vedges = S.raw.edges.filter(e => {
    if (!vis.has(e.subject_id)) return false;
    if (!vis.has(e.object_id)) return false;
    if (f.depKinds.size && !f.depKinds.has(e.dependency_kind)) return false;
    if (f.rels.size && !f.rels.has(e.relation)) return false;
    return true;
  });

  // Step 4: hide isolated nodes — nodes in the main component that
  // ended up with no visible edge after the user-filter narrowing.
  // Lattice subsumption (between two visible items in the same family)
  // counts as connectivity for this check, so chained concept→entity
  // items don't get pruned just because the user filtered out their
  // dependency edges.
  const connected = new Set();
  vedges.forEach(e => { connected.add(e.subject_id); connected.add(e.object_id); });
  for (const grp of S.raw.lattice_groups) {
    const items = (grp.items||[]).filter(it => vis.has(it.formal_name));
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        if (identitySubsumes(items[i].identity, items[j].identity)) {
          connected.add(items[i].formal_name);
          connected.add(items[j].formal_name);
        }
      }
    }
  }
  for (const id of [...vis]) { if (!connected.has(id)) vis.delete(id); }

  // Step 4: ego filter
  if (S.graph.ego) {
    const ego = S.graph.ego;
    const nb = new Set([ego]);
    vedges = vedges.filter(e => {
      if (e.subject_id === ego || e.object_id === ego) { nb.add(e.subject_id); nb.add(e.object_id); return true; }
      return false;
    });
    vis.clear();
    nb.forEach(id => {
      if (!nodeMap[id]) return;
      if (nodeMap[id].kind === 'off_lattice') return;
      vis.add(id);
    });
  }

  // Step 5: collect remaining edges + lattice hierarchy edges for the
  // current visible set (used for rendering only — connectivity was
  // already determined on the universe in Step 1b).
  const depEdges = vedges.filter(e => vis.has(e.subject_id) && vis.has(e.object_id));
  const latticeEdges = []; // {a, b}
  for (const grp of S.raw.lattice_groups) {
    const items = (grp.items||[]).filter(it => vis.has(it.formal_name));
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        const a = items[i], b = items[j];
        if (identitySubsumes(a.identity, b.identity) && isDirectParent(a.identity, b.identity, items)) {
          latticeEdges.push({a: a.formal_name, b: b.formal_name});
        }
      }
    }
  }

  // Step 6: build Cytoscape nodes (everything in vis is already in the
  // main component of the full graph)
  const elems = [];

  // When lattice mode is on, assign per-family hue so family members
  // are visually grouped by color. Dagre + lattice edges naturally
  // cluster same-family nodes spatially; the color reinforces it.
  const familyHues = {};
  if (f.showLattice) {
    const visFamilies = [...new Set([...vis].map(id => nodeMap[id]?.family).filter(Boolean))];
    visFamilies.sort();
    const goldenAngle = 137.508;
    visFamilies.forEach((fam, i) => { familyHues[fam] = (i * goldenAngle) % 360; });
  }

  const seedId = (S.raw.stats || {}).seed || null;
  for (const id of vis) {
    const n = nodeMap[id]; if (!n) continue;
    const deg = (n.in_degree||0)+(n.out_degree||0);
    let color;
    if (f.showLattice && familyHues[n.family] !== undefined) {
      const h = familyHues[n.family];
      color = n.kind === 'dataset' ? `hsl(${h},55%,30%)` : `hsl(${h},50%,25%)`;
    } else {
      color = (n.kind === 'off_lattice') ? '#B8B8AE'
              : ({model:'#1C3A3C', dataset:'#C9803B'})[n.kind] || '#7F8C89';
    }
    const data = { id:n.id, label:truncate(n.id,35),
      kind:n.kind, family:n.family,
      size: Math.min(16, 5+Math.sqrt(deg)*1.5), color };
    if (seedId && n.id === seedId) data.is_seed = 1;
    if (dimmed.has(n.id)) data.dimmed = 1;
    elems.push({ group:'nodes', data });
  }

  // Step 8: build dependency edges. Edge color reflects relation
  // canonicality (canonical = teal/orange by dep_kind; coined = lighter
  // grey to signal it's outside the canonical taxonomy). The relClass
  // attribute drives the lineage / context / noise styling above.
  depEdges.forEach((e,i) => {
    if (!vis.has(e.subject_id) || !vis.has(e.object_id)) return;
    const baseColor = e.dependency_kind === 'direct' ? '#E87040' : '#9B6AD0';
    const color = e.is_canonical_relation ? baseColor : '#B0B0A8';
    elems.push({ group:'edges', data:{ id:'re'+i,
      source: e.subject_id, target: e.object_id,
      relation: e.relation,
      depKind: e.dependency_kind || 'unknown',
      relClass: relationClass(e.relation),
      edgeType: 'relation',
      color, _idx: S.raw.edges.indexOf(e) }});
  });

  // Step 9: lattice hierarchy edges — only displayed when toggle is
  // on, but they always counted toward connectivity above.
  if (f.showLattice) {
    let lid = 0;
    for (const le of latticeEdges) {
      if (!vis.has(le.a) || !vis.has(le.b)) continue;
      elems.push({ group:'edges', data:{ id:'lat'+lid++, source:le.a, target:le.b,
        relation:'lattice', depKind:'lattice', edgeType:'lattice', color:'#C9C9C3' }});
    }
  }

  return elems;
}

function identitySubsumes(parent, child) {
  // Strict identity subsumption: parent ⊊ child.
  // Mirrors modsleuth.pipeline._identity_subsumes — requires same family
  // and that every key:value in parent appears in child, with parent
  // having strictly fewer keys (so it's a proper ancestor, not equal).
  if (!parent || !child) return false;
  if (parent.family !== child.family) return false;
  const pk = Object.keys(parent), ck = Object.keys(child);
  if (pk.length >= ck.length) return false;
  for (const k of pk) { if (parent[k] !== child[k]) return false; }
  return true;
}

function isDirectParent(parent, child, allItems) {
  const pk = Object.keys(parent).length;
  const ck = Object.keys(child).length;
  for (const mid of allItems) {
    const mk = Object.keys(mid.identity||{}).length;
    if (mk <= pk || mk >= ck) continue;
    if (identitySubsumes(parent, mid.identity) && identitySubsumes(mid.identity, child)) return false;
  }
  return true;
}

function rebuildGraph() {
  if (!S.graph.cy) return;
  const elems = buildGraphElements();
  S.graph.cy.elements().remove();
  S.graph.cy.add(elems);
  const layout = document.querySelector('#graph-toolbar .gbtn.active[data-layout]');
  runLayout(layout ? layout.dataset.layout : 'cose');
  const nc = elems.filter(e => e.group==='nodes').length;
  const ec = elems.filter(e => e.group==='edges').length;
  const nFam = new Set(elems.filter(e=>e.group==='nodes').map(e=>e.data.family)).size;
  let status = `${nc} nodes · ${ec} edges`;
  if (nFam > 1) status += ` · ${nFam} families`;
  document.getElementById('graph-status').textContent = status;
  refreshFilterChipCounts();
  refreshStatsBarLive(nc, ec);
}

/* Run the full filter pipeline (node-level + edge-level + isolated +
   connected-component) under a hypothetical filter state. Returns
   the set of visible-and-connected node ids and the edge list that
   survives. Used by both rebuildGraph (live filter state) and
   refreshFilterChipCounts (per-chip hypotheticals). */
function applyFilterPipeline(filterOverrides) {
  const f = {
    kinds: filterOverrides.kinds !== undefined ? filterOverrides.kinds : S.graph.filters.kinds,
    depKinds: filterOverrides.depKinds !== undefined ? filterOverrides.depKinds : S.graph.filters.depKinds,
    rels: filterOverrides.rels !== undefined ? filterOverrides.rels : S.graph.filters.rels,
    family: filterOverrides.family !== undefined ? filterOverrides.family : S.graph.filters.family,
  };
  const nodeMap = {};
  S.raw.nodes.forEach(n => nodeMap[n.id] = n);

  // Step A: build the universe under always-on filters only (lattice +
  // n_links > 0). User filters are NOT applied for connectivity.
  const universe = new Set();
  for (const n of S.raw.nodes) {
    if (n.kind === 'off_lattice') continue;
    if (n.n_links === 0) continue;
    universe.add(n.id);
  }

  // Step B: connectivity over the universe using ALL canonical edges +
  // lattice subsumption pairs (no dep/rel filter). Largest component
  // = the lineage backbone.
  const universeEdges = S.raw.edges.filter(e =>
    universe.has(e.subject_id) && universe.has(e.object_id)
  );
  const universeLattice = [];
  for (const grp of S.raw.lattice_groups) {
    const items = (grp.items||[]).filter(it => universe.has(it.formal_name));
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        const a = items[i], b = items[j];
        if (identitySubsumes(a.identity, b.identity) && isDirectParent(a.identity, b.identity, items)) {
          universeLattice.push([a.formal_name, b.formal_name]);
        }
      }
    }
  }
  const adjU = new Map();
  for (const id of universe) adjU.set(id, new Set());
  for (const e of universeEdges) {
    adjU.get(e.subject_id).add(e.object_id);
    adjU.get(e.object_id).add(e.subject_id);
  }
  for (const [a, b] of universeLattice) {
    adjU.get(a).add(b); adjU.get(b).add(a);
  }
  const seenU = new Set();
  const universeComponents = [];
  for (const start of universe) {
    if (seenU.has(start)) continue;
    const comp = []; const stack = [start];
    while (stack.length) {
      const v = stack.pop();
      if (seenU.has(v)) continue;
      seenU.add(v); comp.push(v);
      for (const w of adjU.get(v) || []) if (!seenU.has(w)) stack.push(w);
    }
    universeComponents.push(comp);
  }
  universeComponents.sort((a, b) => b.length - a.length);
  const mainComp = new Set(universeComponents[0] || []);

  // Step C: apply user filters (kind / family) within the main component.
  // Mirrors buildGraphElements Step 2 — kind is "primary + bridges"
  // (cross-kind 1-hop neighbors stay so lineage isn't broken).
  const vis = new Set();
  const primary = new Set();
  for (const id of mainComp) {
    const n = nodeMap[id];
    if (!n) continue;
    if (f.kinds.size && !f.kinds.has(n.kind)) continue;
    if (f.family && n.family !== f.family) continue;
    vis.add(id); primary.add(id);
  }
  if (f.kinds.size) {
    for (const e of S.raw.edges) {
      const sIn = primary.has(e.subject_id);
      const oIn = primary.has(e.object_id);
      if (sIn && !oIn && mainComp.has(e.object_id)) {
        const n2 = nodeMap[e.object_id];
        if (n2 && (!f.family || n2.family === f.family)) vis.add(e.object_id);
      }
      if (oIn && !sIn && mainComp.has(e.subject_id)) {
        const n1 = nodeMap[e.subject_id];
        if (n1 && (!f.family || n1.family === f.family)) vis.add(e.subject_id);
      }
    }
  }

  // Step D: edge filters (dep/rel) + endpoint visibility.
  let edges = S.raw.edges.filter(e => {
    if (!vis.has(e.subject_id)) return false;
    if (!vis.has(e.object_id)) return false;
    if (f.depKinds.size && !f.depKinds.has(e.dependency_kind)) return false;
    if (f.rels.size && !f.rels.has(e.relation)) return false;
    return true;
  });

  // Step E: isolated-node pruning within the filtered view. Lattice
  // subsumption between two visible items in the same family also
  // counts as connectivity here.
  const latticePairs = [];
  for (const grp of S.raw.lattice_groups) {
    const items = (grp.items||[]).filter(it => vis.has(it.formal_name));
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        const a = items[i], b = items[j];
        if (identitySubsumes(a.identity, b.identity) && isDirectParent(a.identity, b.identity, items)) {
          latticePairs.push([a.formal_name, b.formal_name]);
        }
      }
    }
  }
  const connected = new Set();
  for (const e of edges) { connected.add(e.subject_id); connected.add(e.object_id); }
  for (const [a, b] of latticePairs) { connected.add(a); connected.add(b); }
  for (const id of [...vis]) { if (!connected.has(id)) vis.delete(id); }

  edges = edges.filter(e => vis.has(e.subject_id) && vis.has(e.object_id));
  return { visibleNodes: vis, visibleEdges: edges, nodeMap };
}

/* Compute per-chip counts.
   - Kind chips report a PASSIVE current-view count: "how many nodes of
     this kind are rendered in the current graph." This avoids the
     misleading "what if I selected only this kind" simulation, which
     gets zeroed out by Step E's isolated-prune whenever the kind only
     connects through the other kind (e.g. datasets only connecting
     through models).
   - Dep/relation chips still use the "if this chip alone" hypothetical
     because for edges the isolated-prune is less destructive (edges
     are filtered, not their endpoints) and the predictive count is
     genuinely useful for choosing which relation to drill into. */
function refreshFilterChipCounts() {
  if (!S.raw) return;
  // Compute the current view ONCE for kind chips (no overrides).
  const current = applyFilterPipeline({});
  document.querySelectorAll('.filter-chip').forEach(chip => {
    const ftype = chip.dataset.ftype;
    const fval = chip.dataset.fval;
    let count = 0;
    if (ftype === 'kind') {
      for (const id of current.visibleNodes) {
        if (current.nodeMap[id] && current.nodeMap[id].kind === fval) count++;
      }
    } else {
      const overrides = ftype === 'dep'
        ? { depKinds: new Set([fval]) }
        : { rels: new Set([fval]) };
      const { visibleEdges } = applyFilterPipeline(overrides);
      if (ftype === 'dep') {
        count = visibleEdges.filter(e => e.dependency_kind === fval).length;
      } else {
        count = visibleEdges.filter(e => e.relation === fval).length;
      }
    }
    const countEl = chip.querySelector('.chip-count');
    if (countEl) countEl.textContent = '(' + count + ')';
    chip.classList.toggle('chip-empty', count === 0);
  });
}

/* Update the stats-bar entry for "what's currently in the graph".
   Global totals (across the whole pipeline) stay rendered in their
   own stat-items; this entry is the single live number. */
function refreshStatsBarLive(visibleNodes, visibleEdges) {
  const live = document.getElementById('stat-live');
  if (!live) return;
  const s = S.raw.stats;
  live.innerHTML = '<strong>' + visibleNodes + '</strong> nodes &middot; <strong>'
                 + visibleEdges + '</strong> edges in graph';
  live.title = 'Out of ' + s.lattice_node_count + ' lattice items / '
             + s.edge_count + ' canonical edges (some hide off-lattice / fragmented).';
}

function runLayout(name) {
  if (!S.graph.cy || S.graph.cy.nodes().length === 0) return;
  const n = S.graph.cy.nodes().length;
  const sp = S.graph.spread; // 1..10, default 5

  if (name === 'dagre') {
    const baseNodeSep = n > 100 ? 12 : n > 30 ? 20 : 30;
    const baseRankSep = n > 100 ? 30 : n > 30 ? 45 : 60;
    const mult = 0.4 + sp * 0.2; // spread 1→0.6x, 5→1.4x, 10→2.4x
    S.graph.cy.layout({
      name:'dagre', rankDir:'TB',
      nodeSep: Math.round(baseNodeSep * mult),
      rankSep: Math.round(baseRankSep * mult),
      edgeSep: 5,
      animate: n < 150, animationDuration: 300,
      fit: true, padding: 20,
    }).run();
  } else {
    // cose: built-in physics. Spread slider scales repulsion + edge length.
    // At spread=1 nodes are compact; at spread=10 they are very sparse.
    const repBase = n > 100 ? 50000 : 30000;
    const repulsion = repBase * (0.5 + sp * 0.5);  // 1→1x, 5→3x, 10→5.5x
    const edgeLen = (n > 100 ? 50 : 70) * (0.6 + sp * 0.15); // 1→0.75x .. 10→2.1x
    const grav = Math.max(0.02, 0.5 - sp * 0.05);  // 1→0.45, 5→0.25, 10→0.02
    S.graph.cy.layout({
      name: 'cose',
      animate: false,
      randomize: true,
      nodeRepulsion: () => repulsion,
      idealEdgeLength: () => edgeLen,
      edgeElasticity: () => 100,
      gravity: grav,
      numIter: n > 200 ? 500 : 1000,
      nodeOverlap: 20 + sp * 5,
      fit: true,
      padding: 20,
    }).run();
  }
}

/* ================================================================
   OPERATIONS VIEW
   ================================================================ */
function renderOps() {
  const el = document.getElementById('view-operations');
  el.innerHTML = `<div class="ops-view">
    <div class="ops-search"><input type="search" id="ops-search" placeholder="Search operations…" /></div>
    <div id="ops-list"></div>
  </div>`;
  renderOpsList();
  document.getElementById('ops-search').addEventListener('input', e => { S.ops.search = e.target.value.toLowerCase(); renderOpsList(); });
}

function renderOpsList() {
  const q = S.ops.search;
  const ops = S.raw.operations.filter(op => !q || op.description.toLowerCase().includes(q) || op.batch.toLowerCase().includes(q));
  const batchEdges = {};
  S.raw.edges.forEach(e => { (batchEdges[e.operation_id] = batchEdges[e.operation_id] || []).push(e); });

  document.getElementById('ops-list').innerHTML = ops.map((op,i) => {
    const edges = batchEdges[op.id] || [];
    const expanded = S.ops.expanded.has(op.id);
    return `<div class="op-card${expanded?' expanded':''}" data-op="${esc(op.id)}">
      <div class="op-card__hd" onclick="toggleOp('${esc(op.id)}')">
        <span class="op-card__chev">▶</span>
        <div class="op-card__desc">${esc(op.description)}</div>
        <div class="op-card__meta">${badgeHTML('family', op.batch)} <span class="badge badge--count">${op.edge_count} edges</span></div>
      </div>
      <div class="op-card__body">
        ${(op.anchor_list||[]).map(a => anchorHTML(a)).join('')}
        <div style="margin-top:8px">
          ${edges.map((e,j) => {
            const eid = `opedge-${i}-${j}`;
            return `<div class="op-edge">
              <div class="op-edge__row" onclick="document.getElementById('${eid}').style.display=document.getElementById('${eid}').style.display==='none'?'block':'none'">
                <span class="op-edge__sub" title="${esc(e.subject)}">${esc(truncate(e.subject,30))}</span>
                <span class="op-edge__arrow">→</span>
                ${badgeHTML(e.dependency_kind, e.dependency_kind)}
                <span class="pill">${esc(e.relation)}</span>
                <span class="op-edge__arrow">→</span>
                <span class="op-edge__obj" title="${esc(e.object)}">${esc(truncate(e.object,30))}</span>
              </div>
              <div id="${eid}" class="op-edge__detail" style="display:none">
                <div class="dp-desc">${esc(e.description)}</div>
                ${(e.anchor_list||[]).map(a => anchorHTML(a)).join('')}
              </div>
            </div>`;
          }).join('')}
        </div>
      </div>
    </div>`;
  }).join('');
}

function toggleOp(id) {
  S.ops.expanded.has(id) ? S.ops.expanded.delete(id) : S.ops.expanded.add(id);
  const card = document.querySelector(`.op-card[data-op="${CSS.escape(id)}"]`);
  if (card) card.classList.toggle('expanded');
}

/* ================================================================
   GLOBAL SEARCH
   ================================================================ */
function setupSearch() {
  const input = document.getElementById('global-search');
  const dd = document.getElementById('search-dropdown');
  let debounce = null;
  input.addEventListener('input', () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      const q = input.value.trim().toLowerCase();
      if (!q || q.length < 2) { dd.classList.remove('open'); dd.innerHTML = ''; return; }
      const nodeHits = S.raw.nodes.filter(n => {
        if (n.id.toLowerCase().includes(q)) return true;
        if ((n.aliases||[]).some(a => (a||'').toLowerCase().includes(q))) return true;
        if ((n.description||'').toLowerCase().includes(q)) return true;
        if ((n.subsets||[]).some(s => (s||'').toLowerCase().includes(q))) return true;
        if (Object.values(n.identity||{}).some(v => String(v||'').toLowerCase().includes(q))) return true;
        return false;
      }).slice(0, 20);
      const opHits = S.raw.operations.filter(op => op.description.toLowerCase().includes(q)).slice(0, 10);
      let html = '';
      if (nodeHits.length) {
        html += `<div class="search-group-label">Items</div>`;
        html += nodeHits.map(n =>
          `<div class="search-result" data-type="node" data-id="${esc(n.id)}">
            ${badgeHTML(n.kind, n.kind)}
            <span class="search-result__name">${esc(n.id)}</span>
            <span class="search-result__meta">${esc(n.family)}</span>
          </div>`
        ).join('');
      }
      if (opHits.length) {
        html += `<div class="search-group-label">Operations</div>`;
        html += opHits.map(op =>
          `<div class="search-result" data-type="op" data-id="${esc(op.id)}">
            ${badgeHTML('family', op.batch)}
            <span class="search-result__name">${esc(truncate(op.description, 60))}</span>
            <span class="search-result__meta">${op.edge_count} edges</span>
          </div>`
        ).join('');
      }
      if (!html) html = `<div class="search-result"><span class="dp-empty">No results</span></div>`;
      dd.innerHTML = html;
      dd.classList.add('open');
      dd.querySelectorAll('.search-result').forEach(sr => sr.addEventListener('click', () => {
        dd.classList.remove('open');
        if (sr.dataset.type === 'node') {
          showNodeDetail(sr.dataset.id);
        } else if (sr.dataset.type === 'op') {
          switchView('operations');
          S.ops.expanded.add(sr.dataset.id);
          renderOpsList();
          setTimeout(() => {
            const card = document.querySelector(`.op-card[data-op="${CSS.escape(sr.dataset.id)}"]`);
            if (card) card.scrollIntoView({behavior:'smooth',block:'center'});
          }, 50);
        }
      }));
    }, 200);
  });
  input.addEventListener('blur', () => setTimeout(() => dd.classList.remove('open'), 200));
  input.addEventListener('focus', () => { if (dd.innerHTML) dd.classList.add('open'); });
}

/* ================================================================
   INIT
   ================================================================ */
window._modsleuth = { S, showNodeDetail, handleGraphEdgeClick, closeDetail, switchView, rebuildGraph };
fetchData();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# External-JSON loader (modsleuth merge_artifact / dedup output)
# ---------------------------------------------------------------------------


def _load_graph_data_from_json(source_path: Path) -> dict:
    """Load graph data from a self-contained JSON graph (e.g. modsleuth's
    `merge_artifact.json` or `graph.json` from `modsleuth dedup`),
    bypassing the SQLite pipeline-state pickers. Mirrors the dict shape
    `load_graph_data()` returns so the viewer needs no template changes.

    Fields the external JSON does NOT carry render as defaults:
    operations (empty), corroboration_count (1), subsumes/source_batch_ids
    (empty), per-item subsets/description, lattice dropped/gated."""
    raw = json.loads(source_path.read_text())

    lattice = raw.get("lattice") or {}
    lattice_groups: list[dict] = []
    nodes: dict[str, dict] = {}
    family_counts: dict[str, int] = {}
    for grp in lattice.get("groups") or []:
        # modsleuth groups key the family slug as `id`; gdb uses `family`.
        family = grp.get("family") or grp.get("id") or ""
        items_raw = grp.get("items") or []
        family_counts[family] = family_counts.get(family, 0) + len(items_raw)
        lattice_groups.append({
            "family": family,
            "identity_keys": grp.get("identity_keys") or [],
            "items": items_raw,
        })
        for item in items_raw:
            formal = item.get("formal_name")
            if not formal:
                continue
            links = item.get("links") or []
            primary_url, primary_kind = _pick_primary_link(links)
            nodes[formal] = {
                "id": formal,
                "kind": item.get("kind"),
                "family": family,
                "identity": item.get("identity") or {},
                "aliases": item.get("aliases") or [],
                "links": links,
                "n_links": len(links),
                "primary_url": primary_url,
                "primary_link_kind": primary_kind,
                "description": item.get("description") or "",
                "subsets": item.get("subsets") or [],
                "_generated": bool(item.get("_generated")),
                "_synthesized": bool(item.get("_synthesized")),
                "in_degree": 0,
                "out_degree": 0,
            }

    edges: list[dict] = [
        _build_edge_record(e, nodes, event_id=None, b_label="")
        for e in (raw.get("relations") or [])
    ]

    # Synthesize off-lattice node placeholders for free-text endpoints.
    off_lattice_nodes: dict[str, dict] = {}
    for e in edges:
        if not e["object_in_lattice"] and e["object"]:
            if e["object_id"] not in off_lattice_nodes and e["object_id"] not in nodes:
                off_lattice_nodes[e["object_id"]] = _make_off_lattice_node(e["object_id"], e["object"])
        if not e["subject_in_lattice"] and e["subject"]:
            if e["subject_id"] not in off_lattice_nodes and e["subject_id"] not in nodes:
                off_lattice_nodes[e["subject_id"]] = _make_off_lattice_node(e["subject_id"], e["subject"])
    nodes.update(off_lattice_nodes)

    for e in edges:
        if e["subject_id"] in nodes:
            nodes[e["subject_id"]]["out_degree"] += 1
        if e["object_id"] in nodes:
            nodes[e["object_id"]]["in_degree"] += 1

    rel_counts: dict[str, int] = {}
    dep_kind_counts: dict[str, int] = {}
    canonical_rel_count = 0
    coined_rel_count = 0
    for e in edges:
        rel = e["relation"]
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
        dk = e.get("dependency_kind") or "unknown"
        dep_kind_counts[dk] = dep_kind_counts.get(dk, 0) + 1
        if rel in _CANONICAL_RELATIONS:
            canonical_rel_count += 1
        else:
            coined_rel_count += 1

    # modsleuth's merge_artifact carries a post-merge conflicts list whose
    # schema differs from gdb's reconcile-stage conflicts and can be tens of
    # MB; drop it for external sources rather than ship a payload the
    # client can't render meaningfully.
    conflicts: list = []
    dropped = lattice.get("dropped") or []
    gated = lattice.get("gated") or []

    return {
        "lattice_path": str(source_path),
        "source_stage": "external_json",
        "lattice_groups": lattice_groups,
        "dropped": dropped,
        "gated": gated,
        "nodes": list(nodes.values()),
        "edges": edges,
        "operations": [],
        "conflicts": conflicts,
        "human_review": [],
        "mend_artifact_summary": {},
        "stats": {
            "source_stage": "external_json",
            "node_count": len(nodes),
            "lattice_node_count": sum(1 for n in nodes.values()
                                       if n.get("kind") in ("model", "dataset")),
            "off_lattice_node_count": sum(1 for n in nodes.values()
                                           if n.get("kind") == "off_lattice"),
            "edge_count": len(edges),
            "canonical_relation_edges": canonical_rel_count,
            "coined_relation_edges": coined_rel_count,
            "operation_count": 0,
            "family_count": len(family_counts),
            "dropped_count": len(dropped),
            "gated_count": len(gated),
            "conflict_count": len(conflicts),
            "human_review_count": 0,
            "families": sorted(family_counts.items(), key=lambda kv: -kv[1]),
            "relations": sorted(rel_counts.items(), key=lambda kv: -kv[1]),
            "dependency_kinds": sorted(dep_kind_counts.items(), key=lambda kv: -kv[1]),
        },
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def make_handler(graph_payload: dict):
    body_bytes = PAGE_HTML.encode("utf-8")
    graph_json = json.dumps(graph_payload, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self._send(200, "text/html; charset=utf-8", body_bytes)
            elif path == "/api/graph":
                self._send(200, "application/json; charset=utf-8", graph_json)
            elif path == "/healthz":
                self._send(200, "text/plain", b"ok")
            else:
                self._send(404, "text/plain", b"not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


# Edge-relevance weights used for seeded expansion. Higher = more
# lineage-bearing. Negative scores discount evaluation/citation clutter
# that drowns out the actual training graph.
_EDGE_RELEVANCE: dict[str, int] = {
    "trained_from": 8, "trained_on": 7, "generated_by": 6,
    "transformed_by": 5, "filtered_by": 5, "merged_from": 6,
    "composed_from": 5, "tokenized_by": 3, "deduplicated_by": 3,
    "decontaminated_by": 3, "released_with": 2, "inspired_by": 0,
    "used_for_ablation": -2, "used_for_evaluation": -4,
    "cited_as_baseline": -4,
}


def _score_edge(edge: dict) -> int:
    rel = edge.get("relation") or ""
    score = _EDGE_RELEVANCE.get(rel, 1)
    if (edge.get("dependency_kind") or "") == "direct":
        score += 2
    score += min(3, len(edge.get("anchor_list") or []))
    return score


def _resolve_seed(payload: dict, pattern: str) -> str | None:
    """Match `pattern` (case-insensitive substring) against formal_name and
    aliases; return the highest-degree match's formal_name, or None."""
    needle = pattern.lower()
    candidates: list[tuple[int, str]] = []
    for n in payload["nodes"]:
        hay = (n["id"] + " " + " ".join(n.get("aliases") or [])).lower()
        if needle in hay:
            deg = n.get("in_degree", 0) + n.get("out_degree", 0)
            candidates.append((deg, n["id"]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _seeded_expand(payload: dict, seed_id: str, *,
                   depth: int, target_size: int) -> dict:
    """BFS from `seed_id` up to `depth` hops, greedily admitting the
    highest-relevance-scored neighbors at each frontier until we hit
    `target_size` nodes. Edges between admitted nodes are kept; everything
    else is dropped."""
    nodes_by_id = {n["id"]: n for n in payload["nodes"]}
    if seed_id not in nodes_by_id:
        return payload

    incident: dict[str, list[dict]] = {}
    for e in payload["edges"]:
        for endpoint in (e.get("subject_id"), e.get("object_id")):
            if endpoint:
                incident.setdefault(endpoint, []).append(e)

    keep: set[str] = {seed_id}
    frontier: set[str] = {seed_id}
    for _ in range(max(1, depth)):
        if len(keep) >= target_size:
            break
        candidates: dict[str, int] = {}  # other_id → best score seen
        for src in frontier:
            for e in incident.get(src, []):
                other = (e["object_id"] if e["subject_id"] == src
                         else e["subject_id"])
                if not other or other in keep:
                    continue
                s = _score_edge(e)
                if s > candidates.get(other, -10**9):
                    candidates[other] = s
        if not candidates:
            break
        # Tiebreak by total degree of the candidate node (more central wins).
        ranked = sorted(
            candidates.items(),
            key=lambda kv: (-kv[1],
                            -(nodes_by_id[kv[0]]["in_degree"]
                              + nodes_by_id[kv[0]]["out_degree"])),
        )
        next_frontier: set[str] = set()
        for other_id, _score in ranked:
            if len(keep) >= target_size:
                break
            keep.add(other_id)
            next_frontier.add(other_id)
        frontier = next_frontier

    pruned_nodes = [n for n in payload["nodes"] if n["id"] in keep]
    pruned_edges = [e for e in payload["edges"]
                    if e.get("subject_id") in keep
                    and e.get("object_id") in keep]

    for n in pruned_nodes:
        n["in_degree"] = 0
        n["out_degree"] = 0
    by_id = {n["id"]: n for n in pruned_nodes}
    for e in pruned_edges:
        if e["subject_id"] in by_id:
            by_id[e["subject_id"]]["out_degree"] += 1
        if e["object_id"] in by_id:
            by_id[e["object_id"]]["in_degree"] += 1

    rel_counts: dict[str, int] = {}
    dep_kind_counts: dict[str, int] = {}
    canonical_rel_count = 0
    coined_rel_count = 0
    for e in pruned_edges:
        rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
        dk = e.get("dependency_kind") or "unknown"
        dep_kind_counts[dk] = dep_kind_counts.get(dk, 0) + 1
        if e["relation"] in _CANONICAL_RELATIONS:
            canonical_rel_count += 1
        else:
            coined_rel_count += 1
    family_counts: dict[str, int] = {}
    for n in pruned_nodes:
        fam = n.get("family") or ""
        family_counts[fam] = family_counts.get(fam, 0) + 1
    pruned_groups = []
    for grp in payload.get("lattice_groups") or []:
        items = [it for it in (grp.get("items") or [])
                 if it.get("formal_name") in keep]
        if items:
            pruned_groups.append({**grp, "items": items})

    new_stats = dict(payload["stats"])
    new_stats.update({
        "node_count": len(pruned_nodes),
        "lattice_node_count": sum(1 for n in pruned_nodes
                                   if n.get("kind") in ("model", "dataset")),
        "off_lattice_node_count": sum(1 for n in pruned_nodes
                                       if n.get("kind") == "off_lattice"),
        "edge_count": len(pruned_edges),
        "canonical_relation_edges": canonical_rel_count,
        "coined_relation_edges": coined_rel_count,
        "family_count": len(family_counts),
        "families": sorted(family_counts.items(), key=lambda kv: -kv[1]),
        "relations": sorted(rel_counts.items(), key=lambda kv: -kv[1]),
        "dependency_kinds": sorted(dep_kind_counts.items(), key=lambda kv: -kv[1]),
        "seed": seed_id,
    })
    return {
        **payload,
        "nodes": pruned_nodes,
        "edges": pruned_edges,
        "lattice_groups": pruned_groups,
        "stats": new_stats,
    }


def _prune_payload(payload: dict, *, top_k: int | None,
                   min_degree: int) -> dict:
    """Server-side prune of a loaded payload to the top-K nodes by total
    degree (and/or filtered by min_degree). Edges between removed nodes
    are dropped; stats are recomputed so the UI reflects the pruned set.
    The viewer's always-on filters (lattice + n_links > 0) still apply
    on top of this on the client; this prune mostly matters when the
    graph is too large for Cytoscape to lay out (≥ a few thousand
    edges)."""
    if not top_k and min_degree <= 0:
        return payload

    nodes = payload["nodes"]
    deg = {n["id"]: n.get("in_degree", 0) + n.get("out_degree", 0) for n in nodes}
    keep_ids = {nid for nid, d in deg.items() if d >= min_degree}
    if top_k:
        ranked = sorted(keep_ids, key=lambda i: -deg[i])[:top_k]
        keep_ids = set(ranked)

    pruned_nodes = [n for n in nodes if n["id"] in keep_ids]
    pruned_edges = [e for e in payload["edges"]
                    if e.get("subject_id") in keep_ids
                    and e.get("object_id") in keep_ids]

    # Reset and recompute degrees on the pruned set.
    for n in pruned_nodes:
        n["in_degree"] = 0
        n["out_degree"] = 0
    by_id = {n["id"]: n for n in pruned_nodes}
    for e in pruned_edges:
        if e["subject_id"] in by_id:
            by_id[e["subject_id"]]["out_degree"] += 1
        if e["object_id"] in by_id:
            by_id[e["object_id"]]["in_degree"] += 1

    rel_counts: dict[str, int] = {}
    dep_kind_counts: dict[str, int] = {}
    canonical_rel_count = 0
    coined_rel_count = 0
    for e in pruned_edges:
        rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
        dk = e.get("dependency_kind") or "unknown"
        dep_kind_counts[dk] = dep_kind_counts.get(dk, 0) + 1
        if e["relation"] in _CANONICAL_RELATIONS:
            canonical_rel_count += 1
        else:
            coined_rel_count += 1

    family_counts: dict[str, int] = {}
    for n in pruned_nodes:
        fam = n.get("family") or ""
        family_counts[fam] = family_counts.get(fam, 0) + 1

    # Restrict lattice_groups to surviving items so the family-by-family
    # subsumption pass on the client doesn't reference dropped nodes.
    pruned_groups = []
    for grp in payload.get("lattice_groups") or []:
        items = [it for it in (grp.get("items") or [])
                 if it.get("formal_name") in keep_ids]
        if items:
            pruned_groups.append({**grp, "items": items})

    new_stats = dict(payload["stats"])
    new_stats.update({
        "node_count": len(pruned_nodes),
        "lattice_node_count": sum(1 for n in pruned_nodes
                                   if n.get("kind") in ("model", "dataset")),
        "off_lattice_node_count": sum(1 for n in pruned_nodes
                                       if n.get("kind") == "off_lattice"),
        "edge_count": len(pruned_edges),
        "canonical_relation_edges": canonical_rel_count,
        "coined_relation_edges": coined_rel_count,
        "family_count": len(family_counts),
        "families": sorted(family_counts.items(), key=lambda kv: -kv[1]),
        "relations": sorted(rel_counts.items(), key=lambda kv: -kv[1]),
        "dependency_kinds": sorted(dep_kind_counts.items(), key=lambda kv: -kv[1]),
    })

    return {
        **payload,
        "nodes": pruned_nodes,
        "edges": pruned_edges,
        "lattice_groups": pruned_groups,
        "stats": new_stats,
    }


def serve(source: Path,
          port: int = 8102, host: str = "127.0.0.1",
          top_k: int | None = None,
          min_degree: int = 0,
          seed: str | None = None,
          depth: int = 2,
          target_size: int = 80) -> None:
    src_path = Path(source)
    if not src_path.exists():
        print(f"ERROR: source not found: {src_path}", file=sys.stderr)
        sys.exit(1)
    payload = _load_graph_data_from_json(src_path)
    if seed:
        seed_id = _resolve_seed(payload, seed)
        if not seed_id:
            print(f"ERROR: seed pattern {seed!r} matched no node.")
            return
        before = (payload["stats"]["node_count"], payload["stats"]["edge_count"])
        payload = _seeded_expand(payload, seed_id,
                                  depth=depth, target_size=target_size)
        after = (payload["stats"]["node_count"], payload["stats"]["edge_count"])
        print(f"seeded expansion from {seed_id!r}: "
              f"{before[0]}→{after[0]} nodes, "
              f"{before[1]}→{after[1]} edges "
              f"(depth={depth}, target_size={target_size})")
    elif top_k or min_degree > 0:
        before = (payload["stats"]["node_count"], payload["stats"]["edge_count"])
        payload = _prune_payload(payload, top_k=top_k, min_degree=min_degree)
        after = (payload["stats"]["node_count"], payload["stats"]["edge_count"])
        print(f"pruned: {before[0]}→{after[0]} nodes, "
              f"{before[1]}→{after[1]} edges "
              f"(top_k={top_k}, min_degree={min_degree})")
    print(f"loaded: {payload['stats']['node_count']} nodes, "
          f"{payload['stats']['edge_count']} edges, "
          f"{payload['stats']['operation_count']} operations from "
          f"{payload['lattice_path']}")
    handler = make_handler(payload)
    server = HTTPServer((host, port), handler)
    print(f"serving on http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a merged dependency-graph JSON.")
    p.add_argument("--source", required=True, type=Path,
                   help="Path to the merged graph JSON (e.g. merge_artifact.json or a deduped output).")
    p.add_argument("--seed", default=None,
                   help="Pattern to match (case-insensitive substring on formal_name "
                        "and aliases) for a seeded ego-expansion. Highest-degree match wins.")
    p.add_argument("--depth", type=int, default=2,
                   help="Hops to expand from --seed (default: 2).")
    p.add_argument("--target-size", type=int, default=80,
                   help="Approximate target node count for --seed expansion (default: 80).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Cap to top-K nodes by total degree. Used only when --seed is not given.")
    p.add_argument("--min-degree", type=int, default=0,
                   help="Drop nodes with total degree < this value before serving.")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    serve(source=args.source, host=args.host, port=args.port,
          seed=args.seed, depth=args.depth, target_size=args.target_size,
          top_k=args.top_k, min_degree=args.min_degree)


if __name__ == "__main__":
    main()
