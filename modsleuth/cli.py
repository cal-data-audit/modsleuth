from __future__ import annotations

from pathlib import Path

import click

from . import config
from .pipeline import (
    install_signal_handlers,
    names_packet,
    run_audit,
    run_discover,
    run_expand,
    run_extract,
    run_merge,
    run_organize,
    run_reconcile,
    run_relate,
    run_triage,
)
from .store import all_rows, db, emit_json, loads, read_json


@click.group()
def main():
    """ModSleuth: agentic recursive dependency tracing for LLM releases."""
    install_signal_handlers()


_USAGE_KEYS = ("input_tokens", "output_tokens",
               "cache_creation_input_tokens", "cache_read_input_tokens")


def _fmt_tokens(n: int) -> str:
    if not n:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_duration(s: float) -> str:
    if not s:
        return "—"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def _usage_totals(rows: list) -> dict:
    """Sum token/cost/elapsed usage over runs-table rows (attrs JSON)."""
    tot = {"runs": 0, "cost": 0.0, "elapsed": 0.0,
           **{k: 0 for k in _USAGE_KEYS}}
    for row in rows:
        attrs = loads(row["attrs"], default={}) or {}
        tot["runs"] += 1
        for k in _USAGE_KEYS:
            v = attrs.get(k)
            if isinstance(v, (int, float)):
                tot[k] += int(v)
        if isinstance(attrs.get("cost_usd"), (int, float)):
            tot["cost"] += float(attrs["cost_usd"])
        if isinstance(attrs.get("elapsed_s"), (int, float)):
            tot["elapsed"] += float(attrs["elapsed_s"])
    return tot


@main.command()
@click.option("--fresh", is_flag=True, help="Delete the SQLite DB first.")
@click.option("--yes", is_flag=True, help="Required with --fresh.")
@click.option("--I-mean-it", "i_mean_it", is_flag=True, help="Required with --fresh.")
def init(fresh: bool, yes: bool, i_mean_it: bool):
    """Create local storage and initialize SQLite."""
    if fresh:
        if not (yes and i_mean_it):
            raise click.ClickException("--fresh requires --yes and --I-mean-it")
        for path in (config.DB_PATH,
                     Path(str(config.DB_PATH) + "-wal"),
                     Path(str(config.DB_PATH) + "-shm")):
            path.unlink(missing_ok=True)
    config.STORAGE.mkdir(parents=True, exist_ok=True)
    with db():
        pass
    click.echo(f"storage: {config.STORAGE}")
    click.echo(f"db:      {config.DB_PATH}")


@main.command()
def status():
    """Show pipeline progress for the current storage.

    One line per stage: when it last completed and its headline
    numbers (batch progress, item/edge counts, conflicts, provenance
    flags), plus a suggested next command.
    """
    stage_order = ("discover", "extract", "organize", "audit",
                   "relate", "reconcile", "triage", "merge")
    last: dict = {}
    for row in all_rows(
        "SELECT stage, started_at, ended_at, attrs FROM runs "
        "WHERE ended_at IS NOT NULL ORDER BY started_at"
    ):
        if row["stage"] in stage_order:
            last[row["stage"]] = row

    n_batches = all_rows("SELECT COUNT(*) AS n FROM batches")[0]["n"]
    per_batch: dict[str, dict[str, int]] = {}
    unit_totals: dict[str, int] = {}
    for st in ("extract", "relate"):
        per_batch[st] = {}
        key = "edge_count" if st == "relate" else "names_committed"
        for row in all_rows(
            "SELECT status, attrs FROM batch_artifacts WHERE stage=?", (st,)
        ):
            per_batch[st][row["status"]] = per_batch[st].get(row["status"], 0) + 1
            attrs = loads(row["attrs"], default={}) or {}
            if row["status"] == "complete" and isinstance(attrs.get(key), int):
                unit_totals[st] = unit_totals.get(st, 0) + attrs[key]

    def detail(stage: str) -> str:
        run = last.get(stage)
        attrs = (loads(run["attrs"], default={}) or {}) if run else {}
        if stage in ("extract", "relate"):
            if not n_batches:
                return ""
            done = per_batch[stage].get("complete", 0)
            bad = per_batch[stage].get("failed", 0)
            out = f"{done}/{n_batches} batches complete"
            if bad:
                out += f", {bad} failed"
            if stage in unit_totals:
                unit = "edges" if stage == "relate" else "mentions"
                out += f" · {unit_totals[stage]:,} {unit}"
            return out
        picks = (
            ("batch_count", "batches"),
            ("item_count", "items"), ("group_count", "groups"),
            ("total_edge_count", "edges"), ("relation_count", "edges"),
            ("operation_count", "operations"),
            ("conflict_count", "conflicts"),
            ("provenance_review_count", "provenance flags"),
            ("auto_expand", "auto_expand"), ("decline", "decline"),
            ("manual", "manual"),
        )
        parts = [
            f"{attrs[k]:,} {label}"
            for k, label in picks
            if isinstance(attrs.get(k), int)
        ]
        return " · ".join(parts)

    def stage_done(stage: str) -> bool:
        # Per-batch stages are done only when EVERY batch is complete —
        # an ended run with failed/missing batches is partial, not done.
        if stage in ("extract", "relate"):
            return bool(n_batches) and per_batch[stage].get("complete", 0) >= n_batches
        return stage in last

    click.echo(f"storage: {config.STORAGE}")
    click.echo(f"db:      {config.DB_PATH}")
    click.echo("")
    next_stage = None
    for stage in stage_order:
        run = last.get(stage)
        when = str(run["ended_at"])[:19].replace("T", " ") if run else "—"
        if stage_done(stage):
            mark = "✓"
        elif run or per_batch.get(stage):
            mark = "◐"
        else:
            mark = "·"
        click.echo(f"  {mark} {stage:<9} {when:<20} {detail(stage)}".rstrip())
        if next_stage is None and not stage_done(stage):
            next_stage = stage
    tot = _usage_totals(all_rows("SELECT attrs FROM runs"))
    if tot["input_tokens"] or tot["output_tokens"] or tot["cost"]:
        cost = f" · ${tot['cost']:.2f} reported" if tot["cost"] else ""
        click.echo(
            f"\nusage: {_fmt_tokens(tot['input_tokens'])} in / "
            f"{_fmt_tokens(tot['output_tokens'])} out · "
            f"{_fmt_tokens(tot['cache_read_input_tokens'])} cache-read"
            f"{cost} — `modsleuth usage` for the breakdown"
        )
    failed_batches = sum(
        per_batch[st].get("failed", 0) for st in ("extract", "relate")
    )
    if failed_batches:
        click.echo(f"\n! {failed_batches} failed batch(es) — re-running that "
                   "stage retries just those (completed batches are skipped)")
    if next_stage:
        cmd = ("modsleuth run discover --target <model>"
               if next_stage == "discover" else f"modsleuth run {next_stage}")
        click.echo(f"\nnext: {cmd}")


@main.command()
@click.option("--runs", "show_runs", is_flag=True,
              help="List every run with its own usage.")
def usage(show_runs: bool):
    """Token usage and reported cost per stage for the current storage.

    Numbers come from each planner run's own stream accounting
    (subagent turns included). Aborted or watchdog-killed runs report
    the usage accumulated before they stopped. Pure-Python stages and
    the dedup pipeline's one-shot workers spend no planner tokens and
    show dashes.
    """
    rows = all_rows(
        "SELECT id, stage, seed, started_at, attrs FROM runs ORDER BY started_at"
    )
    if not rows:
        click.echo("no runs recorded in this storage")
        return
    stage_order = ("discover", "extract", "organize", "audit",
                   "relate", "reconcile", "triage", "merge", "expand")
    # Retry attempts are separate runs (stage='retry', seed=<original
    # run id>); attribute their spend to the stage they retried.
    stage_of = {row["id"]: row["stage"] for row in rows}
    by_stage: dict[str, list] = {}
    for row in rows:
        stage = row["stage"] or "?"
        if stage == "retry":
            stage = stage_of.get(row["seed"]) or "retry"
        by_stage.setdefault(stage, []).append(row)
    ordered = ([s for s in stage_order if s in by_stage]
               + sorted(set(by_stage) - set(stage_order)))

    click.echo(f"storage: {config.STORAGE}\n")
    header = (f"  {'stage':<10} {'runs':>4} {'input':>8} {'output':>8} "
              f"{'cache-wr':>9} {'cache-rd':>9} {'cost':>8} {'time':>6}")
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for stage in ordered:
        t = _usage_totals(by_stage[stage])
        cost = f"${t['cost']:.2f}" if t["cost"] else "—"
        click.echo(
            f"  {stage:<10} {t['runs']:>4} {_fmt_tokens(t['input_tokens']):>8} "
            f"{_fmt_tokens(t['output_tokens']):>8} "
            f"{_fmt_tokens(t['cache_creation_input_tokens']):>9} "
            f"{_fmt_tokens(t['cache_read_input_tokens']):>9} "
            f"{cost:>8} {_fmt_duration(t['elapsed']):>6}"
        )
    tot = _usage_totals(rows)
    cost = f"${tot['cost']:.2f}" if tot["cost"] else "—"
    click.echo("  " + "-" * (len(header) - 2))
    click.echo(
        f"  {'total':<10} {tot['runs']:>4} {_fmt_tokens(tot['input_tokens']):>8} "
        f"{_fmt_tokens(tot['output_tokens']):>8} "
        f"{_fmt_tokens(tot['cache_creation_input_tokens']):>9} "
        f"{_fmt_tokens(tot['cache_read_input_tokens']):>9} "
        f"{cost:>8} {_fmt_duration(tot['elapsed']):>6}"
    )
    if show_runs:
        click.echo("")
        for row in rows:
            attrs = loads(row["attrs"], default={}) or {}
            t = _usage_totals([row])
            when = str(row["started_at"])[:19].replace("T", " ")
            model = str(attrs.get("model") or "")[:28]
            stalled = "  [stalled]" if attrs.get("killed_for_stall") else ""
            cost = f"${t['cost']:.2f}" if t["cost"] else "—"
            stage = row["stage"] or "?"
            if stage == "retry":
                # `*` marks a retry attempt of the named stage.
                stage = (stage_of.get(row["seed"]) or "retry") + "*"
            click.echo(
                f"  {when}  {stage:<10} {model:<28} "
                f"{_fmt_tokens(t['input_tokens']):>8} in "
                f"{_fmt_tokens(t['output_tokens']):>8} out "
                f"{cost:>8} {_fmt_duration(t['elapsed']):>6}{stalled}"
            )


@main.group()
def run():
    """Run pipeline stages."""


@run.command("discover")
@click.option("--target", required=True)
@click.option("--artifact", "artifact_path",
              help="Ingest an existing discover artifact instead of launching an agent.")
@click.option("--workspace", "workspace_dir",
              help="Workspace holding paths referenced by --artifact.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
def discover_cmd(target: str, artifact_path: str | None, workspace_dir: str | None,
                 planner_model: str, subagent_model: str):
    emit_json(run_discover(
        target=target,
        artifact_path=artifact_path,
        workspace_dir=workspace_dir,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("extract")
@click.option("--batch-id", help="Limit to one batch. Required when --artifact is used.")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing extract artifact instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
@click.option("--max-workers", type=int,
              help="Override MODSLEUTH_MAX_PARALLEL_BATCHES for this process.")
@click.option("--force", is_flag=True,
              help="Redo all batches (by default, completed batches are skipped).")
def extract_cmd(batch_id: str | None, artifact_path: str | None,
                planner_model: str, subagent_model: str, max_workers: int | None,
                force: bool):
    if artifact_path and not batch_id:
        raise click.ClickException("--batch-id is required with --artifact")
    if max_workers:
        config.MAX_PARALLEL_BATCHES = max(1, max_workers)
    emit_json(run_extract(
        batch_id=batch_id,
        artifact_path=artifact_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
        force=force,
    ))


@run.command("organize")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing organize artifact instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
def organize_cmd(artifact_path: str | None, planner_model: str, subagent_model: str):
    emit_json(run_organize(
        artifact_path=artifact_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("audit")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing audit artifact instead of launching an agent.")
@click.option("--source", "source_path",
              help="Audit a specific lattice artifact (default: most recent organize or audit).")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
def audit_cmd(artifact_path: str | None, source_path: str | None,
              planner_model: str, subagent_model: str):
    """Read the latest lattice artifact, revise it, write the result."""
    emit_json(run_audit(
        artifact_path=artifact_path,
        source_path=source_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("relate")
@click.option("--batch-id", help="Limit to one batch. Required when --artifact is used.")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing relate artifact instead of launching an agent.")
@click.option("--lattice", "lattice_path",
              help="Lattice path (default: most recent organize / audit).")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
@click.option("--max-workers", type=int,
              help="Override MODSLEUTH_MAX_PARALLEL_BATCHES for this process.")
@click.option("--force", is_flag=True,
              help="Redo all batches (by default, completed batches are skipped).")
def relate_cmd(batch_id: str | None, artifact_path: str | None,
               lattice_path: str | None, planner_model: str,
               subagent_model: str, max_workers: int | None, force: bool):
    """Per-batch lattice-anchored relation extraction."""
    if artifact_path and not batch_id:
        raise click.ClickException("--batch-id is required with --artifact")
    if max_workers:
        config.MAX_PARALLEL_BATCHES = max(1, max_workers)
    emit_json(run_relate(
        batch_id=batch_id,
        artifact_path=artifact_path,
        lattice_path=lattice_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
        force=force,
    ))


@run.command("reconcile")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing reconcile artifact for validation.")
@click.option("--lattice", "lattice_path",
              help="Lattice path (default: most recent organize / audit).")
@click.option("--relations", "relations_path",
              help="Single relate artifact (default: aggregate every per-batch relate artifact).")
def reconcile_cmd(artifact_path: str | None, lattice_path: str | None,
                  relations_path: str | None):
    """Pure-Python lattice-aware reconciliation of relate edges.

    Performs subsumption (merging vague edges into specific ones along
    the identity lattice), corroboration (stacking anchors when
    independent sources describe the same dependency), and conflict
    detection (sibling-endpoint disagreements flagged for review).
    No LLM call.
    """
    emit_json(run_reconcile(
        artifact_path=artifact_path,
        lattice_path=lattice_path,
        relations_path=relations_path,
    ))


@run.command("triage")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing triage artifact instead of launching an agent.")
@click.option("--lattice", "lattice_path",
              help="Lattice path (default: most recent organize / audit).")
@click.option("--relations", "relations_path",
              help="Pre-aggregated relations file (default: aggregate completed relate artifacts).")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
def triage_cmd(artifact_path: str | None, lattice_path: str | None,
               relations_path: str | None,
               planner_model: str, subagent_model: str):
    """Classify upstream entity-leaves as auto_expand / decline / manual."""
    emit_json(run_triage(
        artifact_path=artifact_path,
        lattice_path=lattice_path,
        relations_path=relations_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("merge")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing merge artifact for shape validation.")
@click.option("--source", "sources", multiple=True,
              help="Lattice artifact path OR a prior merge_artifact.json "
                   "(cross-seed merge). Pass multiple times. Default: this "
                   "storage's latest organize/audit lattice.")
@click.option("--relations", "relations_sources", multiple=True,
              help="Relations artifact path. Pass multiple times. Default: "
                   "relations carried by --source merge artifacts, or (bare "
                   "merge) every completed relate artifact in this storage.")
def merge_cmd(artifact_path: str | None, sources: tuple[str, ...],
              relations_sources: tuple[str, ...]):
    """Pure-Python cross-run merge of lattices and relations."""
    emit_json(run_merge(
        artifact_path=artifact_path,
        sources=list(sources) if sources else None,
        relations_sources=list(relations_sources) if relations_sources else None,
    ))


@run.command("expand")
@click.option("--node", required=True,
              help="Lattice formal_name to expand into a fresh discover-through-relate run.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for the stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID.")
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True,
              help="Claude model for subagents — the planner passes it "
                   "on every Task call it makes.")
@click.option("--skip", multiple=True,
              type=click.Choice(["discover", "extract", "organize", "audit",
                                 "relate", "reconcile"]),
              help="Skip one or more stages. Pass multiple times to skip several.")
def expand_cmd(node: str, planner_model: str, subagent_model: str,
               skip: tuple[str, ...]):
    """Run discover → reconcile against an upstream node as a fresh
    target, within the current storage.

    Expansion deliberately stops at reconcile: run `modsleuth run
    merge` afterwards to fold the new edges into the graph (and
    `modsleuth run triage` to re-gate further expansion). The
    recursive driver batches several expands per round and does both
    once per round.
    """
    emit_json(run_expand(
        node=node,
        planner_model=planner_model,
        subagent_model=subagent_model,
        skip=tuple(skip),
    ))


@main.command("recursive")
@click.option("--seed", "seeds", multiple=True, required=True,
              help="Target model identifier. Pass multiple times for multiple seeds.")
@click.option("--depth", type=int, default=3, show_default=True,
              help="Maximum recursion depth (depth 1 = base pipeline only).")
@click.option("--top-k", type=int, default=5, show_default=True,
              help="Top-K parents per round (beam width for --strategy beam; "
                   "branching factor for --strategy bfs; ignored for --strategy dfs).")
@click.option("--strategy",
              type=click.Choice(["bfs", "dfs", "beam"], case_sensitive=False),
              default="bfs", show_default=True,
              help="Per-round expansion policy: bfs (level-by-level top-K), "
                   "dfs (single highest-scoring chain), or beam (global top-K "
                   "across depths by cumulative score).")
@click.option("--storage-root", "storage_root", default=None,
              help="Root directory for per-seed MODSLEUTH_STORAGE dirs. "
                   "Defaults to ./storage under the current directory.")
@click.option("--triage-gate/--no-triage-gate", default=True, show_default=True,
              help="Expand only nodes the triage queue marks auto_expand "
                   "(decline/manual nodes never consume expansion budget).")
@click.option("--planner-model", default=None,
              help="Claude model for every stage planner — an alias "
                   "('opus', 'sonnet', 'haiku') or a full model ID. "
                   "Default: each stage's own default.")
@click.option("--subagent-model", default=None,
              help="Claude model the planners pass on every Task call.")
def recursive_cmd(seeds: tuple[str, ...], depth: int, top_k: int,
                  strategy: str, storage_root: str | None, triage_gate: bool,
                  planner_model: str | None, subagent_model: str | None):
    """Reference recursive-expansion driver.

    Multi-hop driver. For each seed, runs the base pipeline, then
    iteratively expands newly-discovered upstream artifacts up to
    ``--depth`` hops using the selected ``--strategy`` (BFS / DFS /
    beam). Each round refreshes the triage queue and, by default,
    expands only ``auto_expand`` nodes. Each seed gets its own
    MODSLEUTH_STORAGE directory; merge across seeds afterwards by
    passing all per-seed merge_artifact.json files to
    ``modsleuth run merge``.
    """
    from .recursive import main as recursive_main
    argv = []
    for s in seeds:
        argv += ["--seed", s]
    argv += ["--depth", str(depth), "--top-k", str(top_k),
             "--strategy", strategy]
    if not triage_gate:
        argv += ["--no-triage-gate"]
    if storage_root:
        argv += ["--storage-root", storage_root]
    if planner_model:
        argv += ["--planner-model", planner_model]
    if subagent_model:
        argv += ["--subagent-model", subagent_model]
    raise SystemExit(recursive_main(argv))


@main.command("dedup")
@click.option("--source", required=True,
              help="Input merged graph JSON (output of `modsleuth run merge`).")
@click.option("--dest", required=True,
              help="Output cleaned graph JSON.")
@click.option("--stages", default="all", show_default=True,
              help="Comma-separated stages. Available: heuristic, hub-audit, node-dedup, release. "
                   "`all` runs them in order.")
@click.option("--log", "log_path", default=None,
              help="Log file path (default: <dest>.log; appended, never truncated).")
@click.option("--protect", multiple=True,
              help="Substring that must survive every stage (repeatable; "
                   "typically the run's seed identifiers).")
@click.option("--model", default=None,
              help="LLM for the hub-audit / node-dedup / release stages "
                   "(default: Opus; also via MODSLEUTH_DEDUP_MODEL).")
def dedup_cmd(source: str, dest: str, stages: str, log_path: str | None,
              protect: tuple[str, ...], model: str | None):
    """Post-merge dedup pipeline.

    Four stages over a merged JSON graph: heuristic clustering (no LLM),
    LLM hub-audit, LLM-verified node-dedup with conflict-guarded union-find,
    and a KEEP/DROP release filter that transitively rewires through dropped
    intermediate checkpoints. See modsleuth/dedup/__main__.py for details.
    """
    from .dedup.__main__ import run_dedup
    raise SystemExit(run_dedup(source, dest, stages, log_path,
                               protect=protect, model=model))


@main.command("check")
@click.option("--source", "source_path", required=True,
              help="Path to a merged graph JSON (merge artifact).")
@click.option("--max-samples", type=int, default=3, show_default=True,
              help="Examples shown per finding.")
def check_cmd(source_path: str, max_samples: int):
    """Deterministic graph-quality checks for a merged artifact.

    Pure-Python invariant checks (zero-anchor edges, dependency_kind
    contradictions, self-loops, duplicate triples, unresolved subjects,
    flattened lattices, evaluation-edge inflation). Exit code 1 when
    any blocker-grade finding exists — usable as a stage gate.
    """
    from .check import run_checks
    raise SystemExit(run_checks(Path(source_path), max_samples=max_samples))


@main.command("viz")
@click.option("--source", "source_path", required=True,
              help="Path to a merged or cleaned graph JSON to visualize.")
@click.option("--seed", default=None,
              help="Pattern to match (case-insensitive substring on formal_name "
                   "and aliases) for a seeded ego-expansion. Highest-degree match "
                   "wins. When given, the server pre-prunes the graph to a focused "
                   "subgraph centered on this node.")
@click.option("--depth", type=int, default=2, show_default=True,
              help="Hops to expand from --seed.")
@click.option("--target-size", type=int, default=80, show_default=True,
              help="Approximate target node count for --seed expansion. Highest-"
                   "relevance neighbors fill the budget first.")
@click.option("--top-k", type=int, default=None,
              help="Cap to top-K nodes by total degree (used only when --seed is "
                   "not given).")
@click.option("--min-degree", type=int, default=0, show_default=True,
              help="Drop nodes with total degree < this value before serving.")
@click.option("--port", type=int, default=8102, show_default=True,
              help="HTTP port to serve on.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address.")
def viz_cmd(source_path: str, seed: str | None, depth: int, target_size: int,
            top_k: int | None, min_degree: int, port: int, host: str):
    """Serve an interactive graph viewer for a merged JSON graph."""
    from .viz import serve
    serve(source=Path(source_path), host=host, port=port,
          seed=seed, depth=depth, target_size=target_size,
          top_k=top_k, min_degree=min_degree)


@main.group()
def debug():
    """Read-only inspection helpers."""


@debug.command("names")
@click.option("--limit", type=int)
@click.option("--kind", type=click.Choice(["model", "dataset"]))
def debug_names(limit: int | None, kind: str | None):
    """List collected names from extract."""
    sql = "SELECT * FROM names"
    params: tuple = ()
    if kind:
        sql += " WHERE kind = ?"
        params = (kind,)
    sql += " ORDER BY kind, name"
    if limit:
        sql += f" LIMIT {int(limit)}"
    emit_json({"names": all_rows(sql, params)})


@debug.command("conflicts")
@click.option("--limit", type=int, default=50, show_default=True,
              help="Max entries listed per category.")
def debug_conflicts(limit: int):
    """List flagged conflicts and provenance-review edges.

    Reads the latest reconcile and merge artifacts: every conflict the
    pipeline flagged for review (sibling-object disagreements,
    description / dependency_kind variants, identity-value clashes)
    plus every edge carrying `provenance_review: true` — the queue the
    flag-for-review policy expects someone to work through.
    """
    def latest_artifact(stage: str) -> dict | None:
        rows = all_rows(
            "SELECT attrs FROM runs WHERE stage=? AND ended_at IS NOT NULL "
            "ORDER BY started_at DESC LIMIT 5", (stage,))
        for row in rows:
            attrs = loads(row["attrs"], default={}) or {}
            path = attrs.get("artifact_path")
            if path and Path(path).exists():
                return read_json(path)
        return None

    out: dict = {}
    for stage in ("reconcile", "merge"):
        art = latest_artifact(stage)
        if art is None:
            continue
        conflicts = [c for c in (art.get("conflicts") or []) if isinstance(c, dict)]
        by_kind: dict[str, int] = {}
        for c in conflicts:
            kind = str(c.get("kind") or "sibling_object")
            by_kind[kind] = by_kind.get(kind, 0) + 1
        flagged = [
            {"subject": e.get("subject"), "relation": e.get("relation"),
             "object": e.get("object"),
             "traced_targets": e.get("traced_targets") or []}
            for e in (art.get("relations") or art.get("edges") or [])
            if isinstance(e, dict) and e.get("provenance_review")
        ]
        out[stage] = {
            "conflict_count": len(conflicts),
            "conflicts_by_kind": by_kind,
            "conflicts": conflicts[:limit],
            "provenance_review_count": len(flagged),
            "provenance_review": flagged[:limit],
        }
    if not out:
        raise click.ClickException(
            "no reconcile or merge artifacts found; run the pipeline first")
    emit_json(out)


@debug.command("names-packet")
def debug_names_packet():
    """Show the deduped (type, name) packet that organize will read.
    Useful for sanity-checking how many distinct names exist before
    spending an organize call."""
    emit_json(names_packet())


@debug.command("organize")
@click.option("--latest/--all", default=True,
              help="Show only the most recent organize run (default) or all of them.")
def debug_organize(latest: bool):
    """Show the groups+items artifact(s) the organize stage produced.

    The artifact lives on disk; the run row's `attrs.artifact_path`
    points at it. We read the file at display time so consumers get
    the current contents, not a stale DB snapshot.
    """
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='organize' AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC"
    )
    if latest:
        rows = rows[:1]
    out = []
    for row in rows:
        attrs = loads(row["attrs"], default={})
        path = attrs.get("artifact_path")
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "run_id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "artifact_path": path,
            "group_count": attrs.get("group_count"),
            "item_count": attrs.get("item_count"),
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"organize_runs": out})


@debug.command("audit")
@click.option("--latest/--all", default=True,
              help="Show only the most recent audit run (default) or all of them.")
def debug_audit(latest: bool):
    """Show the revised lattice produced by audit (same shape as organize)."""
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='audit' AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC"
    )
    if latest:
        rows = rows[:1]
    out = []
    for row in rows:
        attrs = loads(row["attrs"], default={})
        path = attrs.get("artifact_path")
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "run_id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "source_artifact_path": attrs.get("source_artifact_path"),
            "group_count": attrs.get("group_count"),
            "item_count": attrs.get("item_count"),
            "notes": attrs.get("notes"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"audit_runs": out})


@debug.command("lattice")
@click.option("--query", "-q", help="Substring to match against formal_name or aliases (case-insensitive).")
@click.option("--kind", type=click.Choice(["model", "dataset"]),
              help="Filter by kind.")
@click.option("--family", help="Substring match against family name.")
@click.option("--include-unlinked", is_flag=True, default=False,
              help="Also surface items with no resolved link (hidden by default).")
@click.option("--unlinked-only", is_flag=True, default=False,
              help="Show ONLY items with no resolved link.")
@click.option("--source", "source_path",
              help="Search a specific lattice artifact (default: most recent organize / audit).")
@click.option("--limit", type=int, default=50, show_default=True,
              help="Max results.")
@click.option("--full", is_flag=True,
              help="Dump full item JSON instead of compact one-line summary.")
def debug_lattice(query: str | None, kind: str | None, family: str | None,
                  include_unlinked: bool, unlinked_only: bool,
                  source_path: str | None,
                  limit: int, full: bool):
    """Search the latest lattice (organize / audit) by name, kind, family.

    By default, only items with at least one verified link are shown.
    Pass --include-unlinked to also surface unresolved items, or
    --unlinked-only to see ONLY the unresolved pile.

    The compact output shows: kind, formal_name, link count, family.
    Use --full for the complete item record.
    """
    from .pipeline import _latest_lattice_artifact_path

    if source_path:
        path = Path(source_path).resolve()
    else:
        path = _latest_lattice_artifact_path()
    artifact = read_json(str(path))

    if unlinked_only and include_unlinked:
        raise click.ClickException("--include-unlinked and --unlinked-only are mutually exclusive")

    needle = (query or "").casefold()
    fam_needle = (family or "").casefold()
    matches: list[dict] = []
    for grp in artifact.get("groups") or []:
        fam_name = grp.get("family") or ""
        if fam_needle and fam_needle not in fam_name.casefold():
            continue
        for item in grp.get("items") or []:
            if kind and item.get("kind") != kind:
                continue
            has_link = bool(item.get("links") or [])
            if unlinked_only and has_link:
                continue
            if not unlinked_only and not include_unlinked and not has_link:
                continue
            if needle:
                hay = [(item.get("formal_name") or "")] + list(item.get("aliases") or [])
                if not any(needle in (s or "").casefold() for s in hay):
                    continue
            matches.append({**item, "_family": fam_name})
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    if full:
        emit_json({"lattice_path": str(path), "match_count": len(matches),
                   "matches": matches})
        return
    # Compact one-line per match.
    out_rows: list[dict] = []
    for it in matches:
        first_link = ""
        links = it.get("links") or []
        if links and isinstance(links[0], dict):
            first_link = links[0].get("url") or ""
        out_rows.append({
            "kind": it.get("kind"),
            "formal_name": it.get("formal_name"),
            "family": it.get("_family"),
            "n_links": len(links),
            "first_link": first_link,
            "description": (it.get("description") or "")[:120],
        })
    emit_json({"lattice_path": str(path), "match_count": len(matches),
               "matches": out_rows})


@debug.command("relate")
@click.option("--batch-id", help="Limit to one batch's relate artifact.")
def debug_relate(batch_id: str | None):
    """Show the per-batch relate artifacts (typed lattice-anchored edges)."""
    sql = ("SELECT batch_id, artifact_path, status, attrs, updated_at "
           "FROM batch_artifacts WHERE stage='relate'")
    params: tuple = ()
    if batch_id:
        sql += " AND batch_id=?"
        params = (batch_id,)
    sql += " ORDER BY updated_at DESC"
    out = []
    for row in all_rows(sql, params):
        attrs = loads(row["attrs"], default={})
        path = row["artifact_path"]
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "batch_id": row["batch_id"],
            "status": row["status"],
            "updated_at": row["updated_at"],
            "relation_count": attrs.get("relation_count"),
            "off_lattice_object_count": attrs.get("off_lattice_object_count"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"relate_artifacts": out})


@debug.command("triage")
@click.option("--latest/--all", default=True,
              help="Show only the most recent triage run (default) or all.")
def debug_triage(latest: bool):
    """Show the upstream-node classification artifact."""
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='triage' AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC"
    )
    if latest:
        rows = rows[:1]
    out = []
    for row in rows:
        attrs = loads(row["attrs"], default={})
        path = attrs.get("artifact_path")
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "run_id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "auto_expand_count": attrs.get("auto_expand_count"),
            "decline_count": attrs.get("decline_count"),
            "manual_count": attrs.get("manual_count"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"triage_runs": out})


@debug.command("merge")
@click.option("--latest/--all", default=True,
              help="Show only the most recent merge run (default) or all.")
def debug_merge(latest: bool):
    """Show the cross-run merged lattice + relations."""
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='merge' AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC"
    )
    if latest:
        rows = rows[:1]
    out = []
    for row in rows:
        attrs = loads(row["attrs"], default={})
        path = attrs.get("artifact_path")
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "run_id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "sources": attrs.get("sources"),
            "relations_sources": attrs.get("relations_sources"),
            "group_count": attrs.get("group_count"),
            "item_count": attrs.get("item_count"),
            "relation_count": attrs.get("relation_count"),
            "conflict_count": attrs.get("conflict_count"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"merge_runs": out})


if __name__ == "__main__":
    main()
