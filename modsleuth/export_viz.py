"""Static export of the viz for GitHub Pages.

Bundles four per-target seeded subgraphs into one static site:
- ``index.html`` (a thin adaptation of ``viz.PAGE_HTML``)
- ``data/<slug>.json`` per target
- ``.nojekyll`` so GitHub Pages serves the directory verbatim

The exported page reuses the running viz UI verbatim. A target-tab
strip in the nav-bar lets viewers switch between the four paper
targets; clicking a tab navigates to ``?t=<slug>``, which the page
reads on load to fetch the matching JSON. Each tab is its own URL so
visitors can deep-link.

Usage:
    modsleuth viz-export --source data/merge_artifact.json --out docs/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .viz import (
    PAGE_HTML,
    _load_graph_data_from_json,
    _resolve_seed,
    _seeded_expand,
)


# Paper targets, exact same identifiers used in the running viz CLI.
# The seed_pattern strings were verified to resolve uniquely against
# data/merge_artifact.json (the trailing space on SmolLM3 disambiguates
# the bare release from the higher-degree -Base / -GSM8K-SFT / -ONNX
# variants that all contain "SmolLM3-3B" as a prefix).
TARGETS: list[dict[str, str]] = [
    {"slug": "olmo3",     "label": "OLMo 3",           "seed_pattern": "OLMo-3-1125-32B"},
    {"slug": "nemotron3", "label": "Nemotron 3 Super", "seed_pattern": "Nemotron-3-Super-120B-A12B-NVFP4"},
    {"slug": "drtulu",    "label": "DR-Tulu",          "seed_pattern": "DR-Tulu-8B"},
    {"slug": "smollm3",   "label": "SmolLM3",          "seed_pattern": "HuggingFaceTB/SmolLM3-3B "},
]


def export_static(source: Path, out_dir: Path, *,
                  depth: int = 3, target_size: int = 60,
                  targets: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Build a self-contained static viz at `out_dir` with one tab per target."""
    targets = targets or TARGETS
    out_dir = out_dir.resolve()
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_graph_data_from_json(source)

    resolved: list[dict[str, Any]] = []
    for t in targets:
        seed_id = _resolve_seed(payload, t["seed_pattern"])
        if not seed_id:
            print(f"WARNING: seed pattern {t['seed_pattern']!r} matched no node "
                  f"— skipping {t['slug']!r}")
            continue
        sub = _seeded_expand(payload, seed_id,
                             depth=depth, target_size=target_size)
        json_path = data_dir / f"{t['slug']}.json"
        json_path.write_text(
            json.dumps(sub, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        resolved.append({
            **t,
            "seed_id": seed_id,
            "node_count": len(sub["nodes"]),
            "edge_count": len(sub["edges"]),
            "json_kb": round(json_path.stat().st_size / 1024, 1),
        })
        print(f"  {t['slug']:10s}  seed={seed_id!r}")
        print(f"              -> {len(sub['nodes']):3d} nodes / "
              f"{len(sub['edges']):4d} edges  ({resolved[-1]['json_kb']} KB)")

    if not resolved:
        raise RuntimeError("No targets resolved — nothing to export")

    html = _build_static_html(resolved)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "targets": resolved,
        "depth": depth,
        "target_size": target_size,
    }


# ---------------------------------------------------------------------------
# HTML adaptation
# ---------------------------------------------------------------------------

# Marker strings we substitute in PAGE_HTML. Each is verified to appear
# exactly once at build time so that template drift in viz.py surfaces
# as a clear failure rather than a silent broken export.
_FETCH_LINE = "const r = await fetch('/api/graph');"
_BRAND_LINE = '<span class="nav-brand">modsleuth</span>'
_STYLE_CLOSE = "</style>"
_BODY_CLOSE = "</body>"


def _build_static_html(targets: list[dict[str, Any]]) -> str:
    default_slug = targets[0]["slug"]

    # 1. Tab-aware fetch. Reads ?t=<slug> from the URL; falls back to
    #    the first target when absent or unrecognized.
    known_slugs_js = json.dumps([t["slug"] for t in targets])
    new_fetch = (
        "const _params = new URLSearchParams(window.location.search);\n"
        f"  const _knownSlugs = {known_slugs_js};\n"
        f"  const _requested = _params.get('t');\n"
        f"  const activeSlug = _knownSlugs.includes(_requested) ? _requested : '{default_slug}';\n"
        "  const r = await fetch('./data/' + activeSlug + '.json');"
    )
    html = _replace_once(PAGE_HTML, _FETCH_LINE, new_fetch, "fetch line")

    # 2. Target-tab strip injected after the brand. Each tab is an
    #    anchor: clicking reloads the page with a fresh ?t=<slug>,
    #    which gives a clean Cytoscape state without manual teardown.
    tab_html_parts = ['<div class="target-tabs">']
    for t in targets:
        tab_html_parts.append(
            f'<a class="target-tab" href="?t={t["slug"]}" '
            f'data-slug="{t["slug"]}">{_html_escape(t["label"])}</a>'
        )
    tab_html_parts.append("</div>")
    tab_html = "".join(tab_html_parts)
    html = _replace_once(html, _BRAND_LINE,
                         _BRAND_LINE + "\n    " + tab_html,
                         "brand marker")

    # 3. CSS for the tab strip — matches the existing nav-tab look but
    #    sits to the left of the view tabs as a separate group.
    extra_css = (
        "\n.target-tabs{display:flex;gap:2px;align-self:stretch;align-items:flex-end;"
        "margin-left:6px;padding-left:12px;border-left:1px solid var(--neutral-600)}\n"
        ".target-tab{display:flex;align-items:center;padding:0 14px;color:var(--neutral-200);"
        "text-decoration:none;font-size:13px;font-weight:500;border-bottom:2px solid transparent;"
        "transition:background .12s,color .12s,border-color .12s}\n"
        ".target-tab:hover{background:var(--neutral-600);color:var(--neutral-50)}\n"
        ".target-tab.active{color:var(--accent-mint);border-bottom-color:var(--accent-mint);"
        "font-weight:700}\n"
    )
    html = _replace_once(html, _STYLE_CLOSE, extra_css + _STYLE_CLOSE, "style close")

    # 4. Tiny boot script that marks the active tab from ?t=. Runs
    #    before fetchData() so users don't see an unselected strip
    #    flash during load.
    boot_js = (
        "\n<script>\n"
        "(function(){\n"
        "  const p = new URLSearchParams(window.location.search);\n"
        f"  const known = {known_slugs_js};\n"
        f"  const req = p.get('t');\n"
        f"  const active = known.includes(req) ? req : {default_slug!r};\n"
        "  document.querySelectorAll('.target-tab').forEach(function(t){\n"
        "    if (t.dataset.slug === active) t.classList.add('active');\n"
        "  });\n"
        "})();\n"
        "</script>\n"
    )
    html = _replace_once(html, _BODY_CLOSE, boot_js + _BODY_CLOSE, "body close")

    return html


def _replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    if text.count(needle) != 1:
        raise RuntimeError(
            f"export_viz: expected exactly one occurrence of {label} in PAGE_HTML, "
            f"found {text.count(needle)}. The export template is out of sync with viz.py."
        )
    return text.replace(needle, replacement, 1)


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))
