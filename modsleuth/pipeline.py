from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click


def _progress(msg: str) -> None:
    """One-line progress note. Goes to stderr — stdout stays reserved
    for the JSON results the CLI emits."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

from . import config
from .store import (
    all_rows,
    compute_batch_fingerprint,
    db,
    dumps,
    json_text,
    loads,
    materialize_batch,
    new_id,
    now,
    read_json,
    scan_and_register,
    set_batch_artifact,
    upsert_batch_by_fingerprint,
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{new_id()}.{path.name}")
    try:
        tmp.write_text(json_text(payload))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def new_run(stage: str, *, seed: str | None = None, label: str | None = None,
            parent_run_id: str | None = None) -> str:
    run_id = new_id()
    with db() as conn:
        conn.execute(
            """INSERT INTO runs (id, stage, seed, parent_run_id, label, attrs, started_at)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, stage, seed, parent_run_id, label, "{}", now()),
        )
        conn.commit()
    return run_id


def close_run(run_id: str, attrs: dict) -> None:
    with db() as conn:
        row = conn.execute("SELECT attrs FROM runs WHERE id=?", (run_id,)).fetchone()
        existing = loads(row["attrs"], default={}) if row else {}
        existing.update(attrs)
        conn.execute("UPDATE runs SET attrs=?, ended_at=? WHERE id=?", (dumps(existing), now(), run_id))
        conn.commit()


def subagent_prompt_for(model: str) -> str:
    """The `{{subagent_prompt}}` block for a stage prompt — instructs
    the planner to pass `model` verbatim on every Task call."""
    return config.SUBAGENT_PROMPT_CLAUDE.format(model=model)


def render_prompt(stage: str, variables: dict[str, str]) -> str:
    prompt_path = config.PROMPTS_DIR / f"{stage}.md"
    if not prompt_path.exists():
        raise click.ClickException(f"prompt not found: {prompt_path}")
    text = prompt_path.read_text()
    if "subagent_model" in variables and "subagent_prompt" not in variables:
        variables = {**variables, "subagent_prompt": subagent_prompt_for(variables["subagent_model"])}
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", value)
    return text


def runtime_env(run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env[config.MODSLEUTH_STORAGE_ENV] = str(config.STORAGE)
    env[config.MODSLEUTH_PATH_ENV] = str(config.DB_PATH)
    env[config.MODSLEUTH_RUN_ID_ENV] = run_id
    return env


def child_pids(pid: int) -> list[int]:
    try:
        result = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def kill_descendants(pid: int, sig: signal.Signals) -> None:
    for child in child_pids(pid):
        kill_descendants(child, sig)
        try:
            os.kill(child, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate_pgrp(pid: int) -> None:
    kill_descendants(pid, signal.SIGTERM)
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + config.PROCESS_KILL_GRACE_S
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)
    kill_descendants(pid, signal.SIGKILL)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# Live `claude` planner pids. Planners run in their own sessions (so the
# watchdog can group-kill them without touching us), which means they do
# NOT die with this process — every abort path must reap them explicitly
# or they keep running (and billing) as orphans.
_LIVE_SPAWN_PIDS: set[int] = set()


def _reap_live_spawns() -> None:
    for pid in list(_LIVE_SPAWN_PIDS):
        _LIVE_SPAWN_PIDS.discard(pid)
        terminate_pgrp(pid)


atexit.register(_reap_live_spawns)


def install_signal_handlers() -> None:
    """Convert SIGTERM into SystemExit so `finally` blocks and atexit
    run — a bare `kill`/`pkill` otherwise skips Python cleanup and
    orphans any live planner. Called once from the CLI entry point;
    no-op outside the main thread."""
    def _on_term(signum, _frame):
        raise SystemExit(128 + signum)
    try:
        signal.signal(signal.SIGTERM, _on_term)
    except ValueError:
        pass


def _tail_usage(stream_path: Path, offset: int, carry: bytes) -> tuple[int, bytes, int]:
    """Incrementally accumulate output tokens from a growing stream:
    read bytes appended since `offset`, parse complete JSONL lines, and
    return (new_offset, carried_partial_line, output_token_delta).
    Cheap enough to run every poll — it never re-reads old bytes."""
    try:
        with stream_path.open("rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return offset, carry, 0
    if not data:
        return offset, carry, 0
    new_offset = offset + len(data)
    buf = carry + data
    lines = buf.split(b"\n")
    carry = lines.pop()  # last element is a partial line (or b"")
    delta = 0
    for line in lines:
        if b'"usage"' not in line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if rec.get("type") != "assistant":
            continue
        value = ((rec.get("message") or {}).get("usage") or {}).get("output_tokens")
        if isinstance(value, (int, float)):
            delta += int(value)
    return new_offset, carry, delta


def parse_stream_json(stream_path: Path) -> dict:
    out: dict[str, Any] = {
        "turns": 0, "cost_usd": 0.0,
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "tool_calls": [], "final_text": None,
    }
    if not stream_path.exists():
        return out
    for line in stream_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = rec.get("type")
        if kind == "assistant":
            out["turns"] += 1
            message = rec.get("message") or {}
            # Accumulate per-call usage as we go: aborted / watchdog-
            # killed runs never reach the terminal `result` event, and
            # their spend must still be visible. When a result event IS
            # present, its authoritative totals overwrite these sums.
            usage = message.get("usage") or {}
            for key in ("input_tokens", "output_tokens",
                        "cache_creation_input_tokens", "cache_read_input_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    out[key] += int(value)
            for content in message.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "tool_use":
                    out["tool_calls"].append(content.get("name") or "tool_use")
                elif content.get("type") == "text":
                    text = content.get("text")
                    if text:
                        out["final_text"] = text
        elif kind == "result":
            cost = rec.get("total_cost_usd")
            if cost is not None:
                out["cost_usd"] = float(cost)
            usage = rec.get("usage") or {}
            for key in ("input_tokens", "output_tokens",
                        "cache_creation_input_tokens", "cache_read_input_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    out[key] = int(value)
    return out


def spawn_claude(run_id: str, prompt: str, *, model: str = config.CLAUDE_MODEL,
                 label: str | None = None) -> dict:
    if not shutil.which("claude"):
        raise click.ClickException("claude CLI not found; pass --artifact to ingest an existing stage artifact")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / config.RUN_PROMPT_FILE).write_text(prompt)
    stream_path = run_root / config.RUN_STREAM_FILE
    err_path = run_root / config.RUN_STDERR_FILE
    cmd = [
        "claude", "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--disallowedTools", "ScheduleWakeup",
    ]
    started = time.monotonic()
    killed_for_stall = False
    silence_limit = config.STREAM_SILENCE_LIMIT_S
    POLL_INTERVAL_S = 30.0
    HEARTBEAT_EVERY_S = 120.0
    with stream_path.open("w") as stdout, err_path.open("w") as stderr:
        proc = subprocess.Popen(
            cmd, cwd=config.ROOT, env=runtime_env(run_id),
            stdout=stdout, stderr=stderr, text=True, start_new_session=True,
        )
        _LIVE_SPAWN_PIDS.add(proc.pid)
        try:
            last_size = 0
            last_activity = time.monotonic()
            last_heartbeat = time.monotonic()
            tail_offset = 0
            tail_carry = b""
            live_out_tokens = 0
            while True:
                try:
                    rc = proc.wait(timeout=POLL_INTERVAL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                try:
                    cur_size = stream_path.stat().st_size
                except OSError:
                    cur_size = last_size
                tail_offset, tail_carry, delta = _tail_usage(
                    stream_path, tail_offset, tail_carry)
                live_out_tokens += delta
                if time.monotonic() - last_heartbeat >= HEARTBEAT_EVERY_S:
                    last_heartbeat = time.monotonic()
                    _progress(
                        f"{label or run_id}: running · "
                        f"{(time.monotonic() - started) / 60:.1f}m elapsed · "
                        f"stream {cur_size / 1024:.0f}KB · "
                        f"{live_out_tokens / 1000:.1f}K tokens out"
                    )
                if cur_size != last_size:
                    last_size = cur_size
                    last_activity = time.monotonic()
                elif time.monotonic() - last_activity > silence_limit:
                    # A planner that is quietly waiting on a long
                    # subagent is silent too — the limit must stay far
                    # above legitimate wait times (tune with
                    # MODSLEUTH_STREAM_SILENCE_S).
                    terminate_pgrp(proc.pid)
                    try:
                        rc = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        rc = -9
                    killed_for_stall = True
                    try:
                        with err_path.open("a") as ef:
                            ef.write(
                                "\n[watchdog] killed subprocess after "
                                f"{int(silence_limit)}s of stream silence\n"
                            )
                    except OSError:
                        pass
                    break
        except BaseException:
            # Any abort path (Ctrl-C, SIGTERM-as-SystemExit, internal
            # error) must reap the planner — it lives in its own
            # session and never dies with us.
            terminate_pgrp(proc.pid)
            raise
        finally:
            _LIVE_SPAWN_PIDS.discard(proc.pid)
    elapsed = time.monotonic() - started
    stats = parse_stream_json(stream_path)
    tool_calls = stats.pop("tool_calls", [])
    final_text = stats.pop("final_text", None)
    attrs = {
        "runtime": "claude", "model": model, "exit_code": rc, "elapsed_s": elapsed,
        "tool_call_count": len(tool_calls),
        "tool_calls_by_name": {name: tool_calls.count(name) for name in set(tool_calls)},
        **stats,
    }
    if killed_for_stall:
        attrs["killed_for_stall"] = True
    close_run(run_id, attrs)
    if final_text:
        (run_root / "final.txt").write_text(final_text)
    return {
        "run_id": run_id, "exit_code": rc, "elapsed_s": elapsed,
        "log_dir": str(run_root), "killed_for_stall": killed_for_stall,
        "output_tokens": attrs.get("output_tokens", 0),
        "cost_usd": attrs.get("cost_usd", 0.0),
    }


def _stream_indicates_rate_limit(stream_path: Path) -> bool:
    """Return True iff the stream JSONL contains rate-limit / 429 /
    overloaded error markers. Used by `dispatch_spawn` to decide
    whether a non-zero exit is worth retrying."""
    if not stream_path.exists():
        return False
    needles = (
        "rate_limit", "rate-limit", "overloaded_error",
        "429", "too many requests", "RATE_LIMIT",
    )
    try:
        for line in stream_path.read_text(errors="replace").splitlines():
            if not line:
                continue
            low = line.lower()
            if any(n.lower() in low for n in needles):
                return True
    except OSError:
        return False
    return False


def dispatch_spawn(
    run_id: str,
    prompt: str,
    *,
    model: str,
    max_retries: int = 4,
    label: str | None = None,
) -> dict:
    """Dispatch one Claude planner spawn. On non-zero exit, retry up to
    `max_retries` times with exponential backoff (10s, 30s, 90s, 270s)
    when the failure looks rate-limit-related (stream JSONL contains
    `rate_limit` / `429` / `overloaded_error` markers) or the watchdog
    killed a stalled stream; fail immediately otherwise. Each retry
    creates a NEW run row so logs / streams don't clobber.
    """
    backoff_schedule = (10, 30, 90, 270)
    lab = label or run_id
    attempt_run_id = run_id
    last_result: dict = {}
    for attempt in range(max_retries + 1):
        _progress(
            f"{lab}: started (model={model}) → logs: "
            f"{config.STORAGE / config.RUNS_SUBDIR / attempt_run_id}"
        )
        result = spawn_claude(attempt_run_id, prompt, model=model, label=label)
        last_result = result
        rc = result.get("exit_code", 0)
        if rc == 0:
            out_tok = result.get("output_tokens") or 0
            cost = result.get("cost_usd") or 0.0
            extra = f" · {out_tok / 1000:.1f}K tokens out" if out_tok else ""
            extra += f" · ${cost:.2f}" if cost else ""
            _progress(f"{lab}: done in "
                      f"{result.get('elapsed_s', 0) / 60:.1f}m{extra}")
            return result
        if attempt >= max_retries:
            break

        # Decide: is this a rate-limit failure or watchdog stall worth retrying?
        killed_for_stall = result.get("killed_for_stall", False)
        run_root = config.STORAGE / config.RUNS_SUBDIR / attempt_run_id
        rate_limited = _stream_indicates_rate_limit(
            run_root / config.RUN_STREAM_FILE
        )
        if not rate_limited and not killed_for_stall:
            break

        sleep_s = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
        _progress(
            f"{lab}: {'stalled' if killed_for_stall else 'rate-limited'}; "
            f"retrying in {sleep_s}s (attempt {attempt + 2}/{max_retries + 1})"
        )
        time.sleep(sleep_s)
        # Mint a fresh run id so the next attempt's stream doesn't
        # overwrite the failed one.
        attempt_run_id = new_run(
            "retry", seed=run_id,
            label=f"retry:{run_id[:8]}:attempt{attempt + 2}",
        )
    _progress(
        f"{lab}: FAILED (exit {last_result.get('exit_code')}) — "
        f"logs: {last_result.get('log_dir', '?')}"
    )
    return last_result


# ---------------------------------------------------------------------------
# Stage 1 — discover
# ---------------------------------------------------------------------------


def ingest_discovery_artifact(artifact: dict, workspace_dir: Path,
                              target: str | None = None) -> dict:
    enriched, per_batch_maps = scan_and_register(workspace_dir, artifact)
    maps = {m["batch_idx"]: m["file_map"] for m in per_batch_maps}
    with db() as conn:
        cur = conn.cursor()
        for idx, batch in enumerate(enriched.get("batches") or []):
            source_ids = [s.get("source_id") for s in batch.get("sources") or [] if s.get("source_id")]
            if not source_ids:
                continue
            fingerprint = compute_batch_fingerprint(cur, source_ids)
            batch_id, created = upsert_batch_by_fingerprint(
                cur,
                fingerprint=fingerprint,
                source_ids=source_ids,
                label=batch.get("label"),
                summary=batch.get("summary"),
                file_map=maps.get(idx) or {},
                # Which tracing target this batch's sources are official
                # for — the provenance ground truth for every edge later
                # extracted from the batch.
                extra_attrs={"traced_target": target} if target else None,
            )
            batch["batch_id"] = batch_id
            batch["created"] = created
        conn.commit()
    return enriched


def run_discover(
    *,
    target: str,
    artifact_path: str | None = None,
    workspace_dir: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
) -> dict:
    run_id = new_run("discover", seed=target, label=f"discover:{target}")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    workspace = Path(workspace_dir).resolve() if workspace_dir else run_root / config.WORKSPACE_SUBDIR
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.DISCOVER_ARTIFACT_FILE
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / config.RUN_INPUT_FILE).write_text(json_text({"target": target, "workspace_dir": str(workspace)}))
    if artifact_path:
        artifact = read_json(artifact_path)
        used_artifact = Path(artifact_path)
    else:
        prompt = render_prompt("discover", {
            "run_id": run_id,
            "target": target,
            "workspace_dir": str(workspace),
            "worker_dir": str(run_root / config.WORKERS_SUBDIR),
            "artifact_path": str(artifact_out),
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model,
                               label=f"discover {target}")
        if spawn["exit_code"] != 0:
            raise click.ClickException(f"discover failed; logs at {spawn['log_dir']}")
        if not artifact_out.exists():
            raise click.ClickException(f"discover wrote no artifact at {artifact_out}")
        artifact = read_json(artifact_out)
        used_artifact = artifact_out
    enriched = ingest_discovery_artifact(artifact, workspace, target=target)
    close_run(run_id, {"artifact_path": str(used_artifact), "batch_count": len(enriched.get("batches") or [])})
    return {
        "run_id": run_id,
        "artifact_path": str(used_artifact),
        "batches": [
            {"batch_id": b.get("batch_id"), "created": b.get("created"),
             "source_count": len(b.get("sources") or [])}
            for b in enriched.get("batches") or []
        ],
    }


# ---------------------------------------------------------------------------
# Stage 2 — extract (per batch, name + kind only)
# ---------------------------------------------------------------------------


def commit_names(artifact: dict, *, batch_id: str | None = None,
                 run_id: str | None = None) -> dict:
    """Commit `{type, name}` records from an extract artifact.

    Schema accepted: `{"mentions": [{"type": "model"|"dataset", "name": "..."}, ...]}`.
    Skips entries that are missing either field, have an invalid kind,
    or are exact (kind, name) duplicates of another entry in this artifact.
    No anchors, atoms, identity, links, or descriptions live here.
    """
    if not isinstance(artifact, dict):
        return {"status": "failed", "errors": [{"code": "invalid_artifact"}],
                "names_committed": 0, "names_skipped": 0}
    raw = artifact.get("mentions")
    if not isinstance(raw, list):
        return {"status": "failed", "errors": [{"code": "invalid_artifact"}],
                "names_committed": 0, "names_skipped": 0}

    seen: set[tuple[str, str]] = set()
    skipped: list[dict] = []
    accepted: list[tuple[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            skipped.append({"index": idx, "reason": "not_a_dict"})
            continue
        kind = (item.get("type") or item.get("kind") or "").strip().casefold()
        name = (item.get("name") or "").strip()
        if not name:
            skipped.append({"index": idx, "reason": "empty_name"})
            continue
        if kind not in ("model", "dataset"):
            skipped.append({"index": idx, "reason": "invalid_kind", "name": name, "kind": kind})
            continue
        key = (kind, name)
        if key in seen:
            skipped.append({"index": idx, "reason": "duplicate", "name": name, "kind": kind})
            continue
        seen.add(key)
        accepted.append(key)

    committed = 0
    with db() as conn:
        cur = conn.cursor()
        if batch_id:
            cur.execute("DELETE FROM names WHERE batch_id=?", (batch_id,))
        for kind, name in accepted:
            cur.execute(
                """INSERT INTO names (id, batch_id, run_id, kind, name, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (new_id(), batch_id, run_id, kind, name, now()),
            )
            committed += 1
        if batch_id:
            set_batch_artifact(
                cur,
                batch_id=batch_id,
                stage="extract",
                artifact_path=str(Path(artifact.get("_artifact_path", "")).resolve())
                              if artifact.get("_artifact_path") else "",
                status="complete",
                attrs={"names_committed": committed, "names_skipped": len(skipped)},
            )
        conn.commit()
    return {
        "status": "complete",
        "names_committed": committed,
        "names_skipped": len(skipped),
        "skipped": skipped[:50],
    }


def run_extract(
    *,
    batch_id: str | None = None,
    artifact_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
    force: bool = False,
) -> dict:
    if artifact_path:
        artifact = read_json(artifact_path)
        artifact["_artifact_path"] = str(artifact_path)
        return commit_names(artifact, batch_id=batch_id)
    batch_ids = [batch_id] if batch_id else [
        row["id"] for row in all_rows("SELECT id FROM batches ORDER BY created_at")
    ]
    skipped_complete = 0
    if not batch_id and not force:
        done_ids = {
            row["batch_id"] for row in all_rows(
                "SELECT batch_id FROM batch_artifacts "
                "WHERE stage='extract' AND status='complete'"
            )
        }
        skipped_complete = sum(1 for b in batch_ids if b in done_ids)
        batch_ids = [b for b in batch_ids if b not in done_ids]
        if skipped_complete:
            _progress(
                f"extract: {skipped_complete} batch(es) already complete — "
                f"skipped; {len(batch_ids)} to do (--force redoes all)"
            )
    if not batch_ids:
        return {"results": [], "failed": 0, "parallel_workers": 0,
                "skipped_complete": skipped_complete}
    workers = max(1, min(config.MAX_PARALLEL_BATCHES, len(batch_ids) or 1))

    def extract_one(bid: str) -> dict:
        run_id = new_run("extract", label=f"extract:{bid[:8]}")
        run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
        batch_dir = materialize_batch(bid, run_root / config.BATCH_SUBDIR)
        artifact_out = run_root / config.EXTRACT_ARTIFACT_FILE
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / config.RUN_INPUT_FILE).write_text(
            json_text({"batch_id": bid, "batch_dir": str(batch_dir)})
        )
        prompt = render_prompt("extract", {
            "run_id": run_id,
            "batch_id": bid,
            "batch_dir": str(batch_dir),
            "worker_dir": str(run_root / config.WORKERS_SUBDIR),
            "artifact_path": str(artifact_out),
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model,
                               label=f"extract {bid[:8]}")
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            _mark_batch_failed(bid, "extract", run_id,
                               f"planner exit {spawn['exit_code']}"
                               if spawn["exit_code"] != 0 else "no artifact written")
            return {"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]}
        artifact = read_json(artifact_out)
        artifact["_artifact_path"] = str(artifact_out)
        result = commit_names(artifact, batch_id=bid, run_id=run_id)
        result["batch_id"] = bid
        result["run_id"] = run_id
        return result

    results: list[dict] = []
    total = len(batch_ids)

    def _note(result: dict) -> None:
        bid8 = str(result.get("batch_id") or "")[:8]
        if result.get("status") == "complete":
            _progress(f"extract: batch {bid8} done ({len(results)}/{total}) · "
                      f"{result.get('names_committed', '?')} mentions")
        else:
            _progress(f"extract: batch {bid8} FAILED ({len(results)}/{total}) — "
                      f"logs: {result.get('log_dir', '?')}")

    if workers == 1:
        for bid in batch_ids:
            results.append(extract_one(bid))
            _note(results[-1])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_one, bid): bid for bid in batch_ids}
            for future in as_completed(futures):
                results.append(future.result())
                _note(results[-1])
    results.sort(key=lambda r: str(r.get("batch_id") or ""))
    failed = [r for r in results if r.get("status") != "complete"]
    if failed:
        _progress(f"extract: {len(failed)}/{total} batch(es) failed — "
                  "re-running `modsleuth run extract` retries just those")
    return {"results": results, "failed": len(failed),
            "parallel_workers": workers, "skipped_complete": skipped_complete}


# ---------------------------------------------------------------------------
# Stage 3 — organize (one planner reads names file, emits lattice)
# ---------------------------------------------------------------------------


def names_packet() -> dict:
    """The deduped `{type, name}` list the organize planner reads.

    Counts are intentionally absent — they don't change how the planner
    decides whether two surfaces refer to the same entity.
    """
    rows = all_rows(
        "SELECT DISTINCT kind, name FROM names ORDER BY kind, name"
    )
    return {"names": [{"type": r["kind"], "name": r["name"]} for r in rows]}


# Production link kinds — pin a deployable artifact (HF release,
# GitHub repo, vendor API model, HF Space demo). Required on entity
# leaves.
_PRODUCTION_LINK_KINDS = frozenset({
    "hf_model", "hf_dataset", "hf_space", "github", "vendor_docs",
})
# Concept link kinds — describe a family / product line concept
# (paper, blog, family-level HF collection page). Allowed on family
# roots and intermediate concept items; never required.
_CONCEPT_LINK_KINDS = frozenset({
    "paper", "blog", "hf_collection", "hf_dataset_config",
})
_LINK_KINDS = _PRODUCTION_LINK_KINDS | _CONCEPT_LINK_KINDS


def _is_family_root(identity: dict) -> bool:
    """Return True iff identity carries exactly one key, `family` —
    i.e., this item is the family root concept. The lattice's top."""
    if not isinstance(identity, dict):
        return False
    return list(identity.keys()) == ["family"]


def _validate_organize_artifact(artifact: dict) -> tuple[int, int]:
    """Validate the organize / audit lattice shape and return
    (group_count, item_count).

    The validator is intentionally minimal — it checks structural
    invariants only. Quality concerns (root mandate, production-vs-
    concept link policy, missing descriptions, over-specification,
    sibling collisions, multi-root groups) are reported as audit hints
    by `modsleuth.subsets.flag_audit_issues` and resolved by the audit pass.

    Required:

    1. `groups[]` is a list; each entry is a dict with `items[]` list.
    2. Every item has `identity.family` — a non-empty string.
    3. All items in the same group share the same `identity.family`
       value (sanity check against accidental cross-family bundling).
    4. Every item has at least one alias (anti-phantom: phantom items
       invented by HF org enumeration the input never named are
       rejected).
    5. `links`, when present and non-empty, has `links[0].kind` in the
       closed vocabulary and `links[0].url` is an http(s) string.
    6. `description`, when present, is `null` or a string.
    7. `subsets`, when present, is a list of non-empty strings.
    8. `display_name` (if present) is a non-empty string. `formal_name`
       (legacy / always-present) is a non-empty string.

    Not validated here (these are audit's responsibility):
    - exactly one family root per group
    - production-vs-concept link kind policy
    - description content quality / completeness
    - URL HEAD-check status
    - over-specification (bare alias on specific leaf)
    - sibling identity collision
    - cross-org family judgment
    """
    groups = artifact.get("groups") if isinstance(artifact, dict) else None
    if not isinstance(groups, list):
        raise click.ClickException("organize artifact missing groups[]")
    item_count = 0
    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise click.ClickException(f"groups[{i}] is not a dict")
        items = group.get("items")
        if not isinstance(items, list):
            raise click.ClickException(f"groups[{i}].items is not a list")

        family_value: str | None = None

        for j, item in enumerate(items):
            where = f"groups[{i}].items[{j}]"
            if not isinstance(item, dict):
                raise click.ClickException(f"{where} is not a dict")

            identity = item.get("identity") or {}
            if not isinstance(identity, dict):
                raise click.ClickException(f"{where}.identity must be a dict")
            family = identity.get("family")
            if not isinstance(family, str) or not family.strip():
                raise click.ClickException(
                    f"{where} formal_name={item.get('formal_name')!r} "
                    "missing required `identity.family` "
                    "(must be a non-empty string identifying the product line)"
                )
            if family_value is None:
                family_value = family
            elif family_value != family:
                click.echo(
                    f"WARNING: {where} identity.family={family!r} differs "
                    f"from sibling family={family_value!r} in same group; "
                    "permitted (synthetic-data lineage permits mixed-family groups)",
                    err=True,
                )

            # Display + formal_name shape (display_name is the new
            # human-readable label; formal_name is kept for back-compat
            # and may equal display_name)
            for fld in ("formal_name", "display_name"):
                if fld in item and item[fld] is not None:
                    val = item[fld]
                    if not isinstance(val, str) or not val.strip():
                        raise click.ClickException(
                            f"{where}.{fld} must be a non-empty string"
                        )

            if "links" in item:
                links = item["links"]
                if not isinstance(links, list):
                    raise click.ClickException(f"{where}.links must be a list")
                if links:
                    head = links[0]
                    if not isinstance(head, dict):
                        raise click.ClickException(
                            f"{where}.links[0] is not a dict"
                        )
                    kind = head.get("kind")
                    if kind not in _LINK_KINDS:
                        raise click.ClickException(
                            f"{where}.links[0].kind={kind!r} "
                            f"not in {sorted(_LINK_KINDS)}"
                        )
                    url = head.get("url")
                    if not isinstance(url, str) or not url.startswith(
                        ("http://", "https://")
                    ):
                        raise click.ClickException(
                            f"{where}.links[0].url={url!r} "
                            "must be an http(s) URL string"
                        )

            description = item.get("description")
            if description is not None and not isinstance(description, str):
                raise click.ClickException(
                    f"{where}.description must be string or null"
                )

            if "subsets" in item:
                subsets = item["subsets"]
                if not isinstance(subsets, list):
                    raise click.ClickException(
                        f"{where}.subsets must be a list"
                    )
                for k, s in enumerate(subsets):
                    if not isinstance(s, str) or not s.strip():
                        raise click.ClickException(
                            f"{where}.subsets[{k}] must be a non-empty string"
                        )

            aliases = item.get("aliases") or []
            if not isinstance(aliases, list):
                raise click.ClickException(f"{where}.aliases must be a list")
            if not aliases:
                raise click.ClickException(
                    f"{where} formal_name={item.get('formal_name')!r} "
                    "has empty aliases — every item must fold ≥1 real input "
                    "surface form. Phantom items invented from HF org "
                    "enumeration are not allowed."
                )

        item_count += len(items)
    return len(groups), item_count


def _count_link_stats(artifact: dict) -> dict:
    """Tally link / description / kind / lattice-position stats across
    the artifact: total items, items with >=1 link, items with a
    description, model vs dataset counts, root vs leaf counts, total
    links, link-kind histogram."""
    total_items = 0
    items_with_links = 0
    items_with_description = 0
    n_models = 0
    n_datasets = 0
    n_family_roots = 0
    n_entity_leaves = 0
    total_links = 0
    by_kind: dict[str, int] = {}
    for group in artifact.get("groups") or []:
        for item in group.get("items") or []:
            total_items += 1
            links = item.get("links") or []
            if links:
                items_with_links += 1
            if isinstance(item.get("description"), str) and item.get("description").strip():
                items_with_description += 1
            kind = item.get("kind")
            if kind == "model":
                n_models += 1
            elif kind == "dataset":
                n_datasets += 1
            if _is_family_root(item.get("identity") or {}):
                n_family_roots += 1
            else:
                n_entity_leaves += 1
            for link in links:
                if not isinstance(link, dict):
                    continue
                total_links += 1
                k = link.get("kind") or "unknown"
                by_kind[k] = by_kind.get(k, 0) + 1
    return {
        "total_items": total_items,
        "items_with_links": items_with_links,
        "items_without_links": total_items - items_with_links,
        "items_with_description": items_with_description,
        "items_without_description": total_items - items_with_description,
        "n_models": n_models,
        "n_datasets": n_datasets,
        "n_family_roots": n_family_roots,
        "n_entity_leaves": n_entity_leaves,
        "total_links": total_links,
        "links_by_kind": by_kind,
    }


def run_organize(
    *,
    artifact_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """Single planner reads the consolidated names file, groups by
    family, collapses surface variants, picks a canonical formal_name
    and structured identity per item, and writes one record per real
    artifact.

    With `--artifact`, ingest an externally produced organize artifact
    instead of spawning a planner. The artifact lives on disk; we
    record its location in the run row and stop.
    """
    if artifact_path:
        run_id = new_run("organize", label="organize:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        # Structural completion (idempotent) before validation
        from .subsets import complete_lattice_structure
        completion_stats = complete_lattice_structure(artifact)
        atomic_write_json(used, artifact)
        group_count, item_count = _validate_organize_artifact(artifact)
        link_stats = _count_link_stats(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            "completion": completion_stats,
            **link_stats,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count,
                "completion": completion_stats,
                **link_stats}

    run_id = new_run("organize", label="organize")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    names_path = run_root / config.ORGANIZE_NAMES_FILE
    artifact_out = run_root / config.ORGANIZE_ARTIFACT_FILE
    atomic_write_json(names_path, names_packet())
    prompt = render_prompt("organize", {
        "run_id": run_id,
        "names_path": str(names_path),
        "input_path": str(names_path),
        "artifact_path": str(artifact_out),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model,
                           label="organize")
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"organize failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    # Structural completion before validation: ensure every item's
    # formal_name is in aliases[], synthesize a virtual family root
    # for any group missing one. The on-disk artifact is the
    # post-completion lattice.
    from .subsets import complete_lattice_structure
    completion_stats = complete_lattice_structure(artifact)
    atomic_write_json(artifact_out, artifact)
    group_count, item_count = _validate_organize_artifact(artifact)
    link_stats = _count_link_stats(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "group_count": group_count,
        "item_count": item_count,
        "completion": completion_stats,
        **link_stats,
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count,
            "completion": completion_stats,
            **link_stats}


# ---------------------------------------------------------------------------
# Stage 4 — audit (revise the lattice in place; same shape in, same out)
# ---------------------------------------------------------------------------


def _latest_lattice_artifact_path() -> Path:
    """Return the path of the most recent groups+items artifact.

    Searches both `organize` and `audit` runs since audit emits the
    same shape and is the authoritative successor when present.
    """
    rows = all_rows(
        "SELECT id, stage, attrs FROM runs "
        "WHERE stage IN ('organize','audit') AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 1"
    )
    if not rows:
        raise click.ClickException(
            "no organize or audit run found; run `modsleuth run organize` first"
        )
    attrs = loads(rows[0]["attrs"], default={}) or {}
    path = attrs.get("artifact_path")
    if not path or not Path(path).exists():
        raise click.ClickException(
            f"{rows[0]['stage']} artifact missing on disk for run {rows[0]['id']}"
        )
    return Path(path)


def run_audit(
    *,
    artifact_path: str | None = None,
    source_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """Read the latest lattice artifact, revise it, write the result.

    Audit's output schema matches organize's (`groups[].items[]`). The
    agent makes edits directly — splits, merges, formal_name fixes,
    identity_key adjustments — and emits the whole revised lattice.

    With `--artifact`, ingest an externally produced audit artifact
    instead of spawning a planner. With `--source`, audit a specific
    artifact (organize or prior audit) instead of the most recent one.
    """
    def _short_notes(art: dict) -> str | None:
        n = art.get("notes")
        return n[:500] if isinstance(n, str) else None

    if artifact_path:
        run_id = new_run("audit", label="audit:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        # Expand hidden concepts by facet projection (idempotent).
        from .subsets import expand_concept_lattice
        expansion_stats = expand_concept_lattice(artifact)
        atomic_write_json(used, artifact)
        group_count, item_count = _validate_organize_artifact(artifact)
        link_stats = _count_link_stats(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            "notes": _short_notes(artifact),
            "expansion": expansion_stats,
            **link_stats,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count,
                "expansion": expansion_stats,
                **link_stats}

    source_artifact_path = (
        Path(source_path).resolve() if source_path
        else _latest_lattice_artifact_path()
    )

    run_id = new_run("audit", label="audit", seed=str(source_artifact_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: Python pre-pass — populate subsets[] on every dataset
    # node, then cross-check dropped[] against populated subsets[] and
    # restore matches as child items. The pre-processed lattice is
    # what the LLM auditor sees.
    pre_processed = read_json(str(source_artifact_path))
    # Build the input-names set so family_root_invented_alias can fire
    # for roots whose aliases don't trace back to input.
    try:
        input_names_set = {n["name"] for n in names_packet().get("names", [])
                           if isinstance(n, dict) and isinstance(n.get("name"), str)}
    except Exception:
        input_names_set = None
    try:
        from .subsets import populate_then_flag, expand_concept_lattice
        subset_stats = populate_then_flag(pre_processed,
                                          input_names_set=input_names_set)
        # Expand hidden concepts BEFORE audit so the auditor sees the
        # full interior lattice (synthesized concept nodes are flagged
        # `_generated: true`). Audit can then merge source-mentioned
        # aliases onto matching synthesized concepts (clearing the flag)
        # or drop redundant ones.
        pre_expansion_stats = expand_concept_lattice(pre_processed)
    except Exception as exc:  # network / parse failures shouldn't block audit
        subset_stats = {"populate": {"error": str(exc)},
                        "restore": {"error": str(exc)}}
        pre_expansion_stats = {"concepts_synthesized": 0, "error": str(exc)}
    pre_processed_path = run_root / "audit_input_with_subsets.json"
    atomic_write_json(pre_processed_path, pre_processed)

    # Phase 2: materialize all batches into one directory so audit can
    # re-read the original sources (paper PDFs, model cards, code repos)
    # for over-specification checks and source-grounded edits. Each batch
    # gets its own subdir to preserve provenance.
    batches_dir = run_root / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_ids_for_audit = [
        row["id"] for row in all_rows("SELECT id FROM batches ORDER BY created_at")
    ]
    for bid in batch_ids_for_audit:
        try:
            materialize_batch(bid, batches_dir / bid)
        except Exception:
            pass

    artifact_out = run_root / config.AUDIT_ARTIFACT_FILE
    prompt = render_prompt("audit", {
        "run_id": run_id,
        "organize_path": str(pre_processed_path),
        "input_path": str(pre_processed_path),
        "artifact_path": str(artifact_out),
        "batches_dir": str(batches_dir),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model,
                           label="audit")
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"audit failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    # Post-audit: expand_concept_lattice runs again — idempotent. Pre-pass
    # already synthesized interior concepts; this catches any new gaps if
    # audit added entities the pre-pass didn't see.
    post_expansion_stats = expand_concept_lattice(artifact)
    atomic_write_json(artifact_out, artifact)
    group_count, item_count = _validate_organize_artifact(artifact)
    link_stats = _count_link_stats(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "source_artifact_path": str(source_artifact_path),
        "pre_processed_path": str(pre_processed_path),
        "subset_stats": subset_stats,
        "pre_expansion": pre_expansion_stats,
        "post_expansion": post_expansion_stats,
        "group_count": group_count,
        "item_count": item_count,
        "notes": _short_notes(artifact),
        **link_stats,
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count,
            "subset_stats": subset_stats,
            "pre_expansion": pre_expansion_stats,
            "post_expansion": post_expansion_stats,
            **link_stats}


# ---------------------------------------------------------------------------
# Stage 5 — relate (per batch, lattice-anchored typed edges)
# ---------------------------------------------------------------------------


_DIRECT_RELATIONS = (
    "trained_on", "trained_from", "generated_by",
    "transformed_by", "filtered_by",
)
_INDIRECT_RELATIONS = (
    "inspired_by", "used_for_ablation", "used_for_evaluation",
)
# Canonical labels — guidance for the relate prompt and tracking.
# `relation` is OPEN: the planner may coin a new snake_case label when
# none of these canonical values fits the source's described event.
CANONICAL_RELATION_VALUES = (
    *_DIRECT_RELATIONS, *_INDIRECT_RELATIONS,
)
# Recurrent coined labels whose dependency kind is fixed: the
# audit-role taxonomy reported in the paper groups all five under
# direct roles (data operations / weight-level model lineage).
# Coining stays open — these just get their kind enforced the same
# way canonical labels do.
_COINED_DIRECT_RELATIONS = (
    "embedded_by", "decontaminated_against", "composed_from",
    "merged_from", "quantized_from",
)
# Map of known relation → its `dependency_kind` bucket.
RELATION_DEPENDENCY_KIND = {
    **{r: "direct" for r in _DIRECT_RELATIONS},
    **{r: "indirect" for r in _INDIRECT_RELATIONS},
    **{r: "direct" for r in _COINED_DIRECT_RELATIONS},
}
# Closed vocabulary for `dependency_kind`.
DEPENDENCY_KIND_VALUES = ("direct", "indirect")


_VIRTUAL_ADDRESS_RE = __import__("re").compile(
    r"^(?P<family>[^\[]+?)\s*\[(?P<facets>[^\[\]]*)\]$"
)


def parse_virtual_address(s: str) -> tuple[str, dict[str, str]] | None:
    """Parse a virtual concept address `<family> [<k>=<v>, ...]` into
    (family, {facet: value}). Returns None if the string is not a
    virtual address.

    Examples:
        "OLMo 3 [stage=Base]"       → ("OLMo 3", {"stage": "Base"})
        "Qwen3 [size=4B, stage=Base]" → ("Qwen3", {"size": "4B", "stage": "Base"})
        "olmOCR [version=v1]"       → ("olmOCR", {"version": "v1"})
        "allenai/Olmo-3-1025-7B"    → None  (no brackets)
    """
    if not isinstance(s, str):
        return None
    m = _VIRTUAL_ADDRESS_RE.match(s.strip())
    if not m:
        return None
    family = m.group("family").strip()
    facets_raw = m.group("facets").strip()
    facets: dict[str, str] = {}
    if facets_raw:
        for piece in facets_raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece:
                return None
            k, _, v = piece.partition("=")
            k = k.strip()
            v = v.strip()
            if not k or not v:
                return None
            facets[k] = v
    if not family:
        return None
    return family, facets


def _lattice_family_names(lattice_artifact: dict) -> set[str]:
    """Return the set of `identity.family` values across all items."""
    out: set[str] = set()
    for group in lattice_artifact.get("groups") or []:
        for item in group.get("items") or []:
            ident = item.get("identity") or {}
            fam = ident.get("family") if isinstance(ident, dict) else None
            if isinstance(fam, str) and fam:
                out.add(fam)
    return out


def _is_snake_case_label(value: object) -> bool:
    """Lightweight shape check for coined labels. Avoids accepting empty
    strings, whitespace, or sentence fragments (e.g.
    'training data filter')."""
    if not isinstance(value, str) or not value.strip():
        return False
    s = value.strip()
    if not s.replace("_", "").replace("-", "").isalnum():
        return False
    if len(s) > 64:
        return False
    return True


def _coerce_anchor_list(anchors: object) -> tuple[list, int]:
    """Mechanically coerce string anchor entries into dicts —
    `'olmo3-paper.pdf'` becomes `{source: 'olmo3-paper.pdf',
    explanation: '(unstructured anchor)'}`. Pure shape routing, no
    judgment. Returns (coerced_list, coercion_count)."""
    if not isinstance(anchors, list):
        return [], 0
    out: list = []
    coerced = 0
    for anc in anchors:
        if isinstance(anc, str) and anc.strip():
            out.append({"source": anc.strip(),
                        "explanation": "(unstructured anchor)"})
            coerced += 1
        else:
            out.append(anc)
    return out, coerced


def _anchor_list_error(anchors: object) -> str | None:
    """Return the first schema problem in an anchor_list, or None if it
    is a non-empty list of dicts with `source` + `explanation` strings
    (optional `position` string)."""
    if not isinstance(anchors, list) or not anchors:
        return "anchor_list must be a non-empty list"
    for j, anc in enumerate(anchors):
        if not isinstance(anc, dict):
            return f"anchor_list[{j}] is not a dict"
        src = anc.get("source")
        if not isinstance(src, str) or not src.strip():
            return f"anchor_list[{j}].source must be a non-empty string"
        expl = anc.get("explanation")
        if not isinstance(expl, str) or not expl.strip():
            return f"anchor_list[{j}].explanation must be a non-empty string"
        pos = anc.get("position")
        if pos is not None and not isinstance(pos, str):
            return f"anchor_list[{j}].position must be a string when present"
    return None


def _validate_relate_artifact(artifact: dict, *,
                              lattice_formal_names: set[str] | None = None,
                              lattice_family_names: set[str] | None = None,
                              ) -> dict:
    """Validate the assembled relate artifact with per-edge QUARANTINE
    semantics. Structurally invalid edges are moved to the artifact's
    top-level `rejected_edges[]` with reasons instead of failing the
    batch — rejection is per edge (the paper's rule), and deterministic
    code routes rather than destroys. String anchor entries are
    mechanically coerced to dicts first; event-wrapper problems never
    cost edges (the wrapper is context — the per-edge anchors are the
    load-bearing evidence). The artifact is mutated in place.

    Raises only when nothing is salvageable: wrong artifact shape, or
    zero valid edges remain.

    Returns:

    {
      "operation_count":         int,   # operations with ≥1 valid edge
      "edge_count":              int,   # valid edges kept
      "singleton_event_count":   int,
      "off_lattice_object_count": int,
      "direct_count":            int,
      "indirect_count":          int,
      "kind_correction_count":   int,   # known-relation kinds corrected in place
      "coerced_anchor_count":    int,   # string anchors mechanically dict-ified
      "rejected_edge_count":     int,
      "dropped_operation_count": int,   # operations left with no valid edge
      "coined_relations":        {label: count},
    }

    Schema (post-fix):

    {
      "batch_id":       "...",
      "batch_label":    "...",
      "operations": [
        {
          "description": "...",
          "anchor_list": [{"source": "...", "position"?: "...", "explanation": "..."}],
          "edges": [
            {
              "subject":         "<lattice formal_name>",
              "relation":        "trained_on" | ... | "<coined>",
              "dependency_kind": "direct" | "indirect",
              "object":          "<formal_name OR free-text>",
              "description":     "...",
              "anchor_list":     [...]
            }
          ]
        }
      ],
      "rejected_edges": [
        {"op_index": 0, "edge_index": 2, "subject": "...",
         "relation": "...", "object": "...", "reason": "..."}
      ]
    }

    Enforced per edge (violations quarantine the edge): subject resolves
    to the lattice (formal_name or virtual concept address) when a
    lattice is provided; snake_case relation; dependency_kind ∈
    {direct, indirect} after deterministic relation→kind coercion;
    non-empty object / description; no self-loops; dict-shaped
    anchor_list carrying source + explanation.

    Open-vocab tracked: coined relations counted, never rejected.
    """
    if not isinstance(artifact, dict):
        raise click.ClickException("relate artifact is not a dict")
    operations = artifact.get("operations")
    if not isinstance(operations, list):
        raise click.ClickException("relate artifact missing operations[]")

    canonical_relations = set(CANONICAL_RELATION_VALUES)

    singleton_events = 0
    off_lattice = 0
    direct_count = 0
    indirect_count = 0
    kind_corrections = 0
    coerced_anchors = 0
    dropped_ops = 0
    coined_relations: dict[str, int] = {}
    rejected: list[dict] = []
    kept_ops: list[dict] = []

    def _edge_check(edge: dict) -> tuple[str | None, int, int]:
        """Mutates the edge (anchor coercion, kind correction); returns
        (rejection_reason_or_None, kind_corrections, anchor_coercions)."""
        corrected = 0
        anchors, coerced = _coerce_anchor_list(edge.get("anchor_list"))
        edge["anchor_list"] = anchors
        subject = edge.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            return "subject is missing", corrected, coerced
        if lattice_formal_names is not None and subject not in lattice_formal_names:
            # Subject may also be a virtual concept address
            # `<family> [<k>=<v>, ...]` whose family pivots to the lattice.
            virt = parse_virtual_address(subject)
            if virt is None:
                return (f"subject {subject!r} resolves to neither a lattice "
                        "formal_name nor a virtual concept address"), corrected, coerced
            fam_name, _facets = virt
            if (lattice_family_names is not None
                    and fam_name not in lattice_family_names):
                return (f"subject virtual address pivots to unknown family "
                        f"{fam_name!r}"), corrected, coerced
        relation = edge.get("relation")
        if not _is_snake_case_label(relation):
            return (f"relation {relation!r} is not a valid label "
                    "(non-empty snake_case ≤64 chars)"), corrected, coerced
        dep_kind = edge.get("dependency_kind")
        expected_kind = RELATION_DEPENDENCY_KIND.get(relation)
        if expected_kind is not None and dep_kind != expected_kind:
            # Known relations carry a fixed dependency_kind; the mapping
            # is deterministic, so correct the label in place rather
            # than trust the planner's value.
            edge["dependency_kind"] = expected_kind
            dep_kind = expected_kind
            corrected = 1
        if dep_kind not in DEPENDENCY_KIND_VALUES:
            return (f"dependency_kind {dep_kind!r} not in "
                    f"{DEPENDENCY_KIND_VALUES}"), corrected, coerced
        obj = edge.get("object")
        if not isinstance(obj, str) or not obj.strip():
            return "object must be a non-empty string", corrected, coerced
        if obj.strip() == subject.strip():
            # An artifact cannot depend on itself; every self-loop in
            # the QA'd release run was an extraction error.
            return (f"self-loop — subject and object are both "
                    f"{subject!r}"), corrected, coerced
        edge_desc = edge.get("description")
        if not isinstance(edge_desc, str) or not edge_desc.strip():
            return "description is missing or empty", corrected, coerced
        anc_err = _anchor_list_error(anchors)
        if anc_err:
            return anc_err, corrected, coerced
        return None, corrected, coerced

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            dropped_ops += 1
            rejected.append({"op_index": i,
                             "reason": "operation is not a dict"})
            continue
        desc = op.get("description")
        if not isinstance(desc, str) or not desc.strip():
            # Event wrappers are context, not evidence — default rather
            # than lose the edges inside.
            op["description"] = "(undescribed event)"
        op_anchors, n = _coerce_anchor_list(op.get("anchor_list"))
        coerced_anchors += n
        # Event-level anchors are contextual; keep only well-formed
        # entries and never reject edges over the wrapper.
        op["anchor_list"] = [
            a for a in op_anchors
            if isinstance(a, dict)
            and isinstance(a.get("source"), str) and a["source"].strip()
        ]
        edges = op.get("edges")
        if not isinstance(edges, list) or not edges:
            dropped_ops += 1
            rejected.append({"op_index": i,
                             "description": op.get("description"),
                             "reason": "operation has no edges[]"})
            continue
        kept_edges: list[dict] = []
        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                rejected.append({"op_index": i, "edge_index": j,
                                 "reason": "edge is not a dict"})
                continue
            reason, corrected, coerced = _edge_check(edge)
            kind_corrections += corrected
            coerced_anchors += coerced
            if reason:
                rejected.append({
                    "op_index": i, "edge_index": j,
                    "subject": edge.get("subject"),
                    "relation": edge.get("relation"),
                    "object": edge.get("object"),
                    "reason": reason,
                })
                continue
            kept_edges.append(edge)
            relation = edge["relation"]
            if relation not in canonical_relations:
                coined_relations[relation] = coined_relations.get(relation, 0) + 1
            if edge["dependency_kind"] == "direct":
                direct_count += 1
            else:
                indirect_count += 1
            obj = edge["object"]
            if lattice_formal_names is not None and obj not in lattice_formal_names:
                # A virtual concept address is still on-lattice when its
                # family pivot resolves.
                virt = parse_virtual_address(obj)
                fam_resolves = (
                    virt is not None
                    and (lattice_family_names is None
                         or virt[0] in lattice_family_names)
                )
                if not fam_resolves:
                    off_lattice += 1
        if not kept_edges:
            dropped_ops += 1
            continue
        op["edges"] = kept_edges
        if len(kept_edges) == 1:
            singleton_events += 1
        kept_ops.append(op)

    artifact["operations"] = kept_ops
    if rejected:
        artifact["rejected_edges"] = rejected
    else:
        artifact.pop("rejected_edges", None)

    edge_total = sum(len(op["edges"]) for op in kept_ops)
    if edge_total == 0:
        raise click.ClickException(
            "relate artifact has no valid edges "
            f"({len(rejected)} rejected — see rejected_edges[])"
        )
    return {
        "operation_count": len(kept_ops),
        "edge_count": edge_total,
        "singleton_event_count": singleton_events,
        "off_lattice_object_count": off_lattice,
        "direct_count": direct_count,
        "indirect_count": indirect_count,
        "kind_correction_count": kind_corrections,
        "coerced_anchor_count": coerced_anchors,
        "rejected_edge_count": len(rejected),
        "dropped_operation_count": dropped_ops,
        "coined_relations": coined_relations,
    }


def assemble_relate_artifact_from_jsonl(
    events_path: Path, *,
    batch_id: str | None = None,
    batch_label: str | None = None,
) -> dict:
    """Read JSONL events from `events_path`, one event per line, and
    assemble into a single relate artifact dict:

    {batch_id, batch_label, operations: [<event>, ...]}.

    Each event is the parsed JSON object on its line. The pipeline calls
    this after the planner exits — the planner appends events as it
    works, so the JSONL file is the durable record."""
    operations: list[dict] = []
    if not events_path.exists():
        return {
            "batch_id": batch_id,
            "batch_label": batch_label,
            "operations": operations,
        }
    text = events_path.read_text()
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            op = loads(line, default=None)
        except Exception:
            op = None
        if op is None:
            try:
                op = __import__("json").loads(line)
            except Exception as e:
                raise click.ClickException(
                    f"{events_path} line {n}: not valid JSON: {e!r}"
                )
        if not isinstance(op, dict):
            raise click.ClickException(
                f"{events_path} line {n}: top-level must be a JSON object"
            )
        operations.append(op)
    return {
        "batch_id": batch_id,
        "batch_label": batch_label,
        "operations": operations,
    }


def _lattice_formal_names(lattice_artifact: dict) -> set[str]:
    names: set[str] = set()
    for group in lattice_artifact.get("groups") or []:
        for item in group.get("items") or []:
            formal = item.get("formal_name")
            if isinstance(formal, str) and formal:
                names.add(formal)
    return names


def _mark_batch_failed(batch_id: str, stage: str, run_id: str | None,
                       error: str) -> None:
    """Record a failed batch in `batch_artifacts` so `status` and resume
    logic can see it. Never clobbers an existing `complete` row."""
    with db() as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT status FROM batch_artifacts WHERE batch_id=? AND stage=?",
            (batch_id, stage),
        ).fetchone()
        if row and row["status"] == "complete":
            return
        set_batch_artifact(
            cur, batch_id=batch_id, stage=stage, artifact_path="",
            status="failed", run_id=run_id, attrs={"error": error[:500]},
        )
        conn.commit()


def _batch_traced_target(batch_id: str | None) -> str | None:
    """The tracing target whose discover run created this batch — i.e.
    the artifact the batch's sources are *official for*. Recorded in
    batch attrs by `ingest_discovery_artifact`."""
    if not batch_id:
        return None
    rows = all_rows("SELECT attrs FROM batches WHERE id=?", (batch_id,))
    if not rows:
        return None
    attrs = loads(rows[0]["attrs"], default={}) or {}
    traced = attrs.get("traced_target")
    return str(traced) if traced else None


def commit_relations_artifact(
    artifact: dict, *,
    batch_id: str | None = None,
    run_id: str | None = None,
    artifact_path: Path | None = None,
    lattice_formal_names: set[str] | None = None,
    lattice_family_names: set[str] | None = None,
) -> dict:
    """Validate a relate artifact and record it as a per-batch
    artifact. No DB rows for individual operations or relations —
    the JSON file on disk is the data, the run + batch_artifact
    rows index it.

    Returns a dict including coined-vocabulary tallies so operators
    can see what new relation / provenance labels the planner
    introduced this batch.
    """
    stats = _validate_relate_artifact(
        artifact,
        lattice_formal_names=lattice_formal_names,
        lattice_family_names=lattice_family_names,
    )
    # Stamp a stable operation_id on every event so operation structure
    # survives reconciliation and cross-run merges, and stamp every
    # edge with the batch's tracing target so evidence provenance does.
    traced = _batch_traced_target(batch_id)
    stamped = 0
    op_prefix = str(batch_id or "nobatch")[:8]
    for idx, op in enumerate(artifact.get("operations") or []):
        if isinstance(op, dict) and not op.get("operation_id"):
            op["operation_id"] = f"op:{op_prefix}:{idx}"
            stamped += 1
    if traced:
        if artifact.get("traced_target") != traced:
            artifact["traced_target"] = traced
            stamped += 1
        for op in artifact.get("operations") or []:
            for edge in (op.get("edges") or []) if isinstance(op, dict) else []:
                if isinstance(edge, dict) and edge.get("traced_target") != traced:
                    edge["traced_target"] = traced
                    stamped += 1
        stats["traced_target"] = traced
    # Validation may have corrected kinds, coerced anchors, or
    # quarantined edges in place, and stamping adds provenance; the
    # file on disk is the data, so persist any mutation.
    if artifact_path and (stamped
                          or stats.get("kind_correction_count")
                          or stats.get("coerced_anchor_count")
                          or stats.get("rejected_edge_count")
                          or stats.get("dropped_operation_count")):
        atomic_write_json(artifact_path.resolve(), artifact)
    if batch_id and artifact_path:
        with db() as conn:
            cur = conn.cursor()
            set_batch_artifact(
                cur,
                batch_id=batch_id,
                stage="relate",
                artifact_path=str(artifact_path.resolve()),
                status="complete",
                run_id=run_id,
                attrs=stats,
            )
            conn.commit()
    return {"status": "complete", **stats}


def run_relate(
    *,
    batch_id: str | None = None,
    artifact_path: str | None = None,
    lattice_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
    force: bool = False,
) -> dict:
    """Per-batch parallel: spawn one Claude planner per batch to
    extract typed lattice-anchored edges. Subjects must be lattice
    `formal_name`s; the closed 8-bucket relation taxonomy is enforced
    on ingest by `_validate_relate_artifact`.

    With `--artifact`, ingest an externally produced relate artifact
    instead of spawning a planner.
    """
    if artifact_path:
        if not batch_id:
            raise click.ClickException("--batch-id is required with --artifact")
        artifact = read_json(artifact_path)
        # When ingesting standalone, we don't have the lattice in hand,
        # so subject formal-name validation is shape-only.
        result = commit_relations_artifact(
            artifact,
            batch_id=batch_id,
            artifact_path=Path(artifact_path),
        )
        return result

    source_lattice_path = (
        Path(lattice_path).resolve() if lattice_path
        else _latest_lattice_artifact_path()
    )
    lattice_artifact = read_json(str(source_lattice_path))
    # Pass the full lattice to the relate planner. Items without a
    # verified link can still be valid edge endpoints (e.g., gated HF
    # repos, API-only judges, internal AI2 names referenced in source).
    # The relate prompt's free-text `object` field handles off-lattice
    # mentions; the planner should not be deprived of extracted-but-
    # unlinkable entities.
    formal_names = _lattice_formal_names(lattice_artifact)
    family_names = _lattice_family_names(lattice_artifact)
    n_total = sum(len(g.get("items") or []) for g in lattice_artifact.get("groups") or [])
    n_linked = sum(
        1 for g in lattice_artifact.get("groups") or []
        for it in g.get("items") or [] if (it.get("links") or [])
    )

    batch_ids = [batch_id] if batch_id else [
        row["id"] for row in all_rows("SELECT id FROM batches ORDER BY created_at")
    ]
    skipped_complete = 0
    if not batch_id and not force:
        done_ids = {
            row["batch_id"] for row in all_rows(
                "SELECT batch_id FROM batch_artifacts "
                "WHERE stage='relate' AND status='complete'"
            )
        }
        skipped_complete = sum(1 for b in batch_ids if b in done_ids)
        batch_ids = [b for b in batch_ids if b not in done_ids]
        if skipped_complete:
            _progress(
                f"relate: {skipped_complete} batch(es) already complete — "
                f"skipped; {len(batch_ids)} to do (--force redoes all)"
            )
    if not batch_ids:
        return {"results": [], "failed": 0, "parallel_workers": 0,
                "skipped_complete": skipped_complete,
                "lattice_path": str(source_lattice_path)}
    workers = max(1, min(config.MAX_PARALLEL_BATCHES, len(batch_ids) or 1))

    def _batch_label(bid: str) -> str | None:
        rows = all_rows("SELECT label FROM batches WHERE id=?", (bid,))
        if rows:
            return rows[0]["label"]
        return None

    def relate_one(bid: str) -> dict:
        run_id = new_run("relate", label=f"relate:{bid[:8]}",
                         seed=str(source_lattice_path))
        run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
        batch_dir = materialize_batch(bid, run_root / config.BATCH_SUBDIR)
        # The planner appends events as JSONL into events_path during
        # its turn. After it exits we assemble that JSONL into the
        # canonical relate_artifact.json.
        events_path = run_root / config.RELATE_EVENTS_FILE
        artifact_out = run_root / config.RELATE_ARTIFACT_FILE
        run_root.mkdir(parents=True, exist_ok=True)
        events_path.touch()  # ensure the planner can append from line 1
        (run_root / config.RUN_INPUT_FILE).write_text(
            json_text({"batch_id": bid, "batch_dir": str(batch_dir),
                       "lattice_path": str(source_lattice_path)})
        )
        prompt = render_prompt("relate", {
            "run_id": run_id,
            "batch_id": bid,
            "batch_dir": str(batch_dir),
            "traced_target": _batch_traced_target(bid)
                or "(not recorded — use the batch's own release context)",
            "lattice_path": str(source_lattice_path),
            "worker_dir": str(run_root / config.WORKERS_SUBDIR),
            "artifact_path": str(events_path),  # JSONL append target
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model,
                               label=f"relate {bid[:8]}")
        if spawn["exit_code"] != 0:
            _mark_batch_failed(bid, "relate", run_id,
                               f"planner exit {spawn['exit_code']}")
            return {"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]}
        # Assemble JSONL → JSON
        try:
            artifact = assemble_relate_artifact_from_jsonl(
                events_path, batch_id=bid, batch_label=_batch_label(bid),
            )
        except click.ClickException as exc:
            _mark_batch_failed(bid, "relate", run_id, str(exc))
            return {"batch_id": bid, "status": "failed",
                    "log_dir": spawn["log_dir"], "error": str(exc)}
        atomic_write_json(artifact_out, artifact)
        try:
            result = commit_relations_artifact(
                artifact,
                batch_id=bid,
                run_id=run_id,
                artifact_path=artifact_out,
                lattice_formal_names=formal_names,
                lattice_family_names=family_names,
            )
        except click.ClickException as exc:
            _mark_batch_failed(bid, "relate", run_id, str(exc))
            return {"batch_id": bid, "status": "failed",
                    "log_dir": spawn["log_dir"], "error": str(exc)}
        result["batch_id"] = bid
        result["run_id"] = run_id
        result["artifact_path"] = str(artifact_out)
        result["events_path"] = str(events_path)
        return result

    results: list[dict] = []
    total = len(batch_ids)

    def _note(result: dict) -> None:
        bid8 = str(result.get("batch_id") or "")[:8]
        if result.get("status") == "complete":
            _progress(f"relate: batch {bid8} done ({len(results)}/{total}) · "
                      f"{result.get('edge_count', '?')} edges / "
                      f"{result.get('operation_count', '?')} operations")
        else:
            _progress(f"relate: batch {bid8} FAILED ({len(results)}/{total}) — "
                      f"logs: {result.get('log_dir', '?')}")

    if workers == 1:
        for bid in batch_ids:
            results.append(relate_one(bid))
            _note(results[-1])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(relate_one, bid): bid for bid in batch_ids}
            for future in as_completed(futures):
                results.append(future.result())
                _note(results[-1])
    results.sort(key=lambda r: str(r.get("batch_id") or ""))
    failed = [r for r in results if r.get("status") != "complete"]
    if failed:
        _progress(f"relate: {len(failed)}/{total} batch(es) failed — "
                  "re-running `modsleuth run relate` retries just those")
    return {"results": results, "failed": len(failed),
            "lattice_path": str(source_lattice_path),
            "lattice_total_items": n_total,
            "lattice_linked_items": n_linked,
            "parallel_workers": workers,
            "skipped_complete": skipped_complete}


# ---------------------------------------------------------------------------
# Stage 6 — reconcile (pure-Python lattice-aware merge of relate edges)
# ---------------------------------------------------------------------------


def _identity_for_address(address: str, lattice: dict) -> dict[str, str] | None:
    """Resolve an edge endpoint string to its identity dict.

    - Lattice formal_name → return that item's identity dict.
    - Virtual concept address `<family> [<k>=<v>, ...]` → return
      `{family: <name>, **<facets>}`.
    - Free-text → return None (off-lattice).
    """
    if not isinstance(address, str) or not address:
        return None
    # Try formal_name lookup first
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            if it.get("formal_name") == address:
                ident = it.get("identity") or {}
                return dict(ident) if isinstance(ident, dict) else None
    # Virtual concept address
    virt = parse_virtual_address(address)
    if virt is None:
        return None
    fam, facets = virt
    out: dict[str, str] = {"family": fam}
    out.update(facets)
    return out


def _identity_subsumes(parent: dict, child: dict) -> bool:
    """Return True iff `parent` ⊑ `child` (parent identity is a subset
    of child identity — every key-value pair in parent appears in child).
    Equivalent to: child is more specific than parent (or equal).

    Required for both endpoints to share `family` (otherwise there's no
    lineage relationship — they're in different families).
    """
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return False
    if parent.get("family") != child.get("family"):
        return False
    for k, v in parent.items():
        if child.get(k) != v:
            return False
    return True


def _edge_subsumes(parent_edge: dict, child_edge: dict, lattice: dict) -> bool:
    """An edge subsumes another iff:
    - Same `relation` and `dependency_kind`.
    - Subject identity of parent ⊑ subject identity of child.
    - Object identity of parent ⊑ object identity of child.
    """
    if parent_edge.get("relation") != child_edge.get("relation"):
        return False
    if parent_edge.get("dependency_kind") != child_edge.get("dependency_kind"):
        return False
    s1 = _identity_for_address(parent_edge.get("subject", ""), lattice)
    s2 = _identity_for_address(child_edge.get("subject", ""), lattice)
    if s1 is None or s2 is None or not _identity_subsumes(s1, s2):
        return False
    o1 = _identity_for_address(parent_edge.get("object", ""), lattice)
    o2 = _identity_for_address(child_edge.get("object", ""), lattice)
    if o1 is None or o2 is None or not _identity_subsumes(o1, o2):
        return False
    # Equal endpoints aren't subsumption — they're corroboration.
    if s1 == s2 and o1 == o2:
        return False
    return True


def _edge_siblings_conflict(e1: dict, e2: dict, lattice: dict) -> bool:
    """Two edges are sibling conflicts if:
    - Same relation, dependency_kind.
    - Same subject identity (verbatim).
    - Object identities share family but differ on at least one
      facet AND neither object subsumes the other (siblings, not
      ancestor/descendant).
    """
    if e1.get("relation") != e2.get("relation"):
        return False
    if e1.get("dependency_kind") != e2.get("dependency_kind"):
        return False
    s1 = _identity_for_address(e1.get("subject", ""), lattice)
    s2 = _identity_for_address(e2.get("subject", ""), lattice)
    if s1 is None or s2 is None or s1 != s2:
        return False
    o1 = _identity_for_address(e1.get("object", ""), lattice)
    o2 = _identity_for_address(e2.get("object", ""), lattice)
    if o1 is None or o2 is None:
        return False
    if o1.get("family") != o2.get("family"):
        return False
    # Same identity → not a conflict, that's corroboration.
    if o1 == o2:
        return False
    # If one subsumes the other → that's subsumption, not conflict.
    if _identity_subsumes(o1, o2) or _identity_subsumes(o2, o1):
        return False
    return True


def _org_prefix(name: str) -> str | None:
    """HF-style org prefix of a formal name (``org/Repo`` → ``org``),
    lowercased. Names without an org prefix (family roots, concepts,
    free-text objects) return None and are never provenance-flagged."""
    if "/" in name:
        org = name.split("/", 1)[0].strip().lower()
        return org or None
    return None


def _all_relate_edges(
    relate_artifacts: list[dict],
) -> tuple[list[dict], dict[str, dict]]:
    """Flatten edges across all per-batch relate artifacts. Each edge
    is annotated with its source batch_id, operation id, and event
    description. Also returns the operations index
    `{operation_id: {description, anchor_list, batch_id}}` so event
    structure survives the flattening."""
    out: list[dict] = []
    operations: dict[str, dict] = {}
    for art in relate_artifacts:
        bid = art.get("batch_id")
        for i, op in enumerate(art.get("operations") or []):
            if not isinstance(op, dict):
                continue
            op_id = str(op.get("operation_id")
                        or f"op:{str(bid or 'nobatch')[:8]}:{i}")
            event_desc = op.get("description")
            event_anchors = op.get("anchor_list") or []
            operations.setdefault(op_id, {
                "description": event_desc,
                "anchor_list": list(event_anchors),
                "batch_id": bid,
            })
            for edge in op.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                out.append({
                    **edge,
                    "_batch_id": bid,
                    "_operation_id": op_id,
                    "_event_description": event_desc,
                    "_event_anchor_list": list(event_anchors),
                })
    return out, operations


def _reconcile_edges(edges: list[dict], lattice: dict) -> dict:
    """Pure-Python reconciliation:

    1. **Corroboration** — group edges by (subject, relation, object).
       Within each group, accumulate `anchor_list[]` from all sources;
       if descriptions differ, keep both as `description_variants[]`.
    2. **Subsumption** — for each pair of corroboration-merged edges
       sharing relation+dep_kind, if one's endpoints lattice-subsume
       the other's, mark the vague edge as subsumed by the specific.
       The specific edge keeps both anchor sets; the vague edge's
       `subsumed_by` field points at the specific.
    3. **Conflict** — for sibling-endpoint pairs (same subject + relation,
       different but not subsumption-related objects in same family),
       record in `conflicts[]` for human review.
    """
    # Phase 1: corroboration. Bucket by (subject, relation, dep_kind, object).
    bucket: dict[tuple[str, str, str, str], dict] = {}
    for edge in edges:
        key = (
            edge.get("subject") or "",
            edge.get("relation") or "",
            edge.get("dependency_kind") or "",
            edge.get("object") or "",
        )
        if key not in bucket:
            bucket[key] = {
                "subject": edge.get("subject"),
                "relation": edge.get("relation"),
                "dependency_kind": edge.get("dependency_kind"),
                "object": edge.get("object"),
                "description": edge.get("description"),
                "description_variants": [],
                "anchor_list": list(edge.get("anchor_list") or []),
                "source_batch_ids": [],
                "traced_targets": [],
                "operation_ids": [],
                "corroboration_count": 0,
                "subsumed_by": None,
                "subsumes": [],
            }
            if edge.get("_batch_id"):
                bucket[key]["source_batch_ids"].append(edge["_batch_id"])
            if edge.get("traced_target"):
                bucket[key]["traced_targets"].append(edge["traced_target"])
            if edge.get("_operation_id"):
                bucket[key]["operation_ids"].append(edge["_operation_id"])
            bucket[key]["corroboration_count"] = 1
            continue
        target = bucket[key]
        target["corroboration_count"] += 1
        bid = edge.get("_batch_id")
        if bid and bid not in target["source_batch_ids"]:
            target["source_batch_ids"].append(bid)
        traced = edge.get("traced_target")
        if traced and traced not in target["traced_targets"]:
            target["traced_targets"].append(traced)
        op_id = edge.get("_operation_id")
        if op_id and op_id not in target["operation_ids"]:
            target["operation_ids"].append(op_id)
        for anc in edge.get("anchor_list") or []:
            target["anchor_list"].append(anc)
        this_desc = edge.get("description")
        if (this_desc and target["description"]
                and this_desc != target["description"]
                and this_desc not in target["description_variants"]):
            target["description_variants"].append(this_desc)

    merged_edges = list(bucket.values())

    # Phase 2: subsumption. For each pair of edges, if one subsumes the
    # other, mark the vague one as subsumed.
    for i, e_i in enumerate(merged_edges):
        for j, e_j in enumerate(merged_edges):
            if i == j:
                continue
            # If e_i subsumes (= is more general than) e_j, then e_i is
            # subsumed BY e_j (the more specific one).
            if _edge_subsumes(e_i, e_j, lattice):
                e_i["subsumed_by"] = (
                    e_j.get("subject"),
                    e_j.get("relation"),
                    e_j.get("object"),
                )
                e_j["subsumes"].append((
                    e_i.get("subject"),
                    e_i.get("relation"),
                    e_i.get("object"),
                ))
                # Push e_i's anchors onto e_j too — the specific edge
                # inherits the vague edge's evidence (and its evidence
                # provenance and operation membership).
                for anc in e_i["anchor_list"]:
                    e_j["anchor_list"].append(anc)
                for traced in e_i.get("traced_targets") or []:
                    if traced not in e_j["traced_targets"]:
                        e_j["traced_targets"].append(traced)
                for op_id in e_i.get("operation_ids") or []:
                    if op_id not in e_j["operation_ids"]:
                        e_j["operation_ids"].append(op_id)

    # Phase 3: conflicts. Sibling endpoints (same subject + relation,
    # different objects in same family, neither subsumes the other).
    conflicts: list[dict] = []
    seen_conflict_keys: set[frozenset] = set()
    for i, e_i in enumerate(merged_edges):
        for j, e_j in enumerate(merged_edges):
            if i >= j:
                continue
            if _edge_siblings_conflict(e_i, e_j, lattice):
                key = frozenset({
                    (e_i.get("subject"), e_i.get("relation"), e_i.get("object")),
                    (e_j.get("subject"), e_j.get("relation"), e_j.get("object")),
                })
                if key in seen_conflict_keys:
                    continue
                seen_conflict_keys.add(key)
                conflicts.append({
                    "subject": e_i.get("subject"),
                    "relation": e_i.get("relation"),
                    "object_a": e_i.get("object"),
                    "object_b": e_j.get("object"),
                    "anchors_a": list(e_i.get("anchor_list") or []),
                    "anchors_b": list(e_j.get("anchor_list") or []),
                })

    # Provenance routing (deterministic, flag-only). An edge whose
    # subject belongs to a different org than every tracing target
    # that contributed its evidence is a cross-target claim — the
    # subject's own official sources never backed it. Such edges are
    # flagged for review, never dropped: org mismatch routes the edge
    # to a reviewer, it does not judge it.
    provenance_review_count = 0
    for e in merged_edges:
        subj_org = _org_prefix(e.get("subject") or "")
        target_orgs = {
            org for org in (
                _org_prefix(t) for t in (e.get("traced_targets") or [])
            ) if org
        }
        if subj_org and target_orgs and subj_org not in target_orgs:
            e["provenance_review"] = True
            provenance_review_count += 1

    # Drop tuples to plain dicts for JSON-friendliness
    def _tup_to_dict(t):
        if t is None:
            return None
        return {"subject": t[0], "relation": t[1], "object": t[2]}

    for e in merged_edges:
        e["subsumed_by"] = _tup_to_dict(e["subsumed_by"])
        e["subsumes"] = [_tup_to_dict(t) for t in e["subsumes"]]

    # Counters
    canonical_edges = [e for e in merged_edges if e["subsumed_by"] is None]
    subsumed_edges = [e for e in merged_edges if e["subsumed_by"] is not None]
    return {
        "edges": merged_edges,
        "canonical_edge_count": len(canonical_edges),
        "subsumed_edge_count": len(subsumed_edges),
        "total_edge_count": len(merged_edges),
        "corroboration_count": sum(
            1 for e in merged_edges if e["corroboration_count"] > 1
        ),
        "conflict_count": len(conflicts),
        "provenance_review_count": provenance_review_count,
        "conflicts": conflicts,
    }


def run_reconcile(
    *,
    artifact_path: str | None = None,
    lattice_path: str | None = None,
    relations_path: str | None = None,
) -> dict:
    """Pure-Python reconciliation pass after relate.

    Inputs (resolved automatically when not passed):
    - lattice: most recent organize/audit artifact
    - relate artifacts: every per-batch relate artifact registered in
      batch_artifacts (stage='relate', status='complete')

    Outputs the reconciled artifact at
    `<run_root>/reconcile_artifact.json` with shape:
        {
          "edges": [<merged edge with subsumed_by/subsumes/corroboration_count>, ...],
          "conflicts": [...],
          "canonical_edge_count": int,
          "subsumed_edge_count": int,
          "corroboration_count": int,
          "conflict_count": int
        }

    With `--artifact`, ingest a pre-computed reconcile artifact for
    shape validation only.
    """
    if artifact_path:
        run_id = new_run("reconcile", label="reconcile:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        if not isinstance(artifact, dict) or "edges" not in artifact:
            raise click.ClickException("reconcile artifact missing 'edges'")
        close_run(run_id, {
            "artifact_path": str(used),
            "total_edge_count": len(artifact.get("edges") or []),
            "conflict_count": len(artifact.get("conflicts") or []),
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "total_edge_count": len(artifact.get("edges") or []),
                "conflict_count": len(artifact.get("conflicts") or [])}

    source_lattice_path = (
        Path(lattice_path).resolve() if lattice_path
        else _latest_lattice_artifact_path()
    )
    lattice = read_json(str(source_lattice_path))

    if relations_path:
        relate_artifacts = [read_json(relations_path)]
    else:
        rows = all_rows(
            "SELECT batch_id, artifact_path FROM batch_artifacts "
            "WHERE stage='relate' AND status='complete'"
        )
        if not rows:
            raise click.ClickException(
                "no relate artifacts found; run `modsleuth run relate` first"
            )
        relate_artifacts = []
        for row in rows:
            path = Path(row["artifact_path"])
            if path.exists():
                relate_artifacts.append(read_json(str(path)))

    edges, operations_index = _all_relate_edges(relate_artifacts)
    result = _reconcile_edges(edges, lattice)
    result["operations"] = operations_index
    result["operation_count"] = len(operations_index)

    run_id = new_run("reconcile", label="reconcile",
                     seed=str(source_lattice_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.RECONCILE_ARTIFACT_FILE
    atomic_write_json(artifact_out, result)

    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "lattice_path": str(source_lattice_path),
        "input_edge_count": len(edges),
        "total_edge_count": result["total_edge_count"],
        "canonical_edge_count": result["canonical_edge_count"],
        "subsumed_edge_count": result["subsumed_edge_count"],
        "corroboration_count": result["corroboration_count"],
        "conflict_count": result["conflict_count"],
        "provenance_review_count": result["provenance_review_count"],
        "operation_count": result["operation_count"],
    })
    return {
        "run_id": run_id,
        "artifact_path": str(artifact_out),
        "input_edge_count": len(edges),
        "total_edge_count": result["total_edge_count"],
        "canonical_edge_count": result["canonical_edge_count"],
        "subsumed_edge_count": result["subsumed_edge_count"],
        "corroboration_count": result["corroboration_count"],
        "conflict_count": result["conflict_count"],
        "provenance_review_count": result["provenance_review_count"],
        "operation_count": result["operation_count"],
    }


# ---------------------------------------------------------------------------
# Stage 7 — triage (one planner classifies upstream nodes)
# ---------------------------------------------------------------------------


_TRIAGE_BUCKETS = ("auto_expand", "decline", "manual")


def _validate_triage_artifact(artifact: dict) -> dict[str, int]:
    """Validate triage artifact shape; return per-bucket counts."""
    if not isinstance(artifact, dict):
        raise click.ClickException("triage artifact is not a dict")
    counts: dict[str, int] = {}
    for bucket in _TRIAGE_BUCKETS:
        items = artifact.get(bucket)
        if not isinstance(items, list):
            raise click.ClickException(
                f"triage artifact missing {bucket!r} list"
            )
        for i, entry in enumerate(items):
            if not isinstance(entry, dict):
                raise click.ClickException(
                    f"triage.{bucket}[{i}] is not a dict"
                )
            for required in ("formal_name", "rationale"):
                if not entry.get(required):
                    raise click.ClickException(
                        f"triage.{bucket}[{i}] missing {required!r}"
                    )
        counts[bucket] = len(items)
    return counts


def _aggregate_relations_artifact(out_path: Path) -> Path:
    """Concatenate every batch's relate artifact into one file the
    triage planner reads. Each batch artifact has the
    `operations[].edges[]` shape; we flatten edges across all
    operations and tag each with the batch / event it came from.
    """
    rows = all_rows(
        "SELECT batch_id, artifact_path FROM batch_artifacts "
        "WHERE stage='relate' AND status='complete'"
    )
    if not rows:
        raise click.ClickException(
            "no relate artifacts found; run `modsleuth run relate` first"
        )
    merged: list[dict] = []
    batch_ids: list[str] = []
    for row in rows:
        path = Path(row["artifact_path"])
        if not path.exists():
            continue
        artifact = read_json(str(path))
        operations = artifact.get("operations") or []
        for op_idx, op in enumerate(operations):
            if not isinstance(op, dict):
                continue
            event_desc = op.get("description")
            event_anchors = op.get("anchor_list") or []
            for edge in op.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                merged.append({
                    **edge,
                    "_batch_id": row["batch_id"],
                    "_event_index": op_idx,
                    "_event_description": event_desc,
                    "_event_anchor_list": event_anchors,
                })
        batch_ids.append(row["batch_id"])
    atomic_write_json(out_path, {
        "batch_ids": batch_ids,
        "edges": merged,
    })
    return out_path


def run_triage(
    *,
    artifact_path: str | None = None,
    lattice_path: str | None = None,
    relations_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """One planner reads the merged lattice + relations and classifies
    every upstream entity-leaf as auto_expand / decline / manual.

    With `--artifact`, ingest an externally produced triage artifact
    instead of spawning a planner.
    """
    if artifact_path:
        run_id = new_run("triage", label="triage:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        counts = _validate_triage_artifact(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            **{f"{bucket}_count": counts[bucket] for bucket in _TRIAGE_BUCKETS},
        })
        return {"run_id": run_id, "artifact_path": str(used), **counts}

    source_lattice_path = (
        Path(lattice_path).resolve() if lattice_path
        else _latest_lattice_artifact_path()
    )

    run_id = new_run("triage", label="triage", seed=str(source_lattice_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if relations_path:
        relations_file = Path(relations_path).resolve()
    else:
        relations_file = run_root / config.TRIAGE_RELATIONS_FILE
        _aggregate_relations_artifact(relations_file)

    artifact_out = run_root / config.TRIAGE_ARTIFACT_FILE
    prompt = render_prompt("triage", {
        "run_id": run_id,
        "lattice_path": str(source_lattice_path),
        "relations_path": str(relations_file),
        "input_path": str(relations_file),
        "artifact_path": str(artifact_out),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model,
                           label="triage")
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"triage failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    counts = _validate_triage_artifact(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "lattice_path": str(source_lattice_path),
        "relations_path": str(relations_file),
        **{f"{bucket}_count": counts[bucket] for bucket in _TRIAGE_BUCKETS},
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out), **counts}


# ---------------------------------------------------------------------------
# Stage 7 — merge (pure-Python cross-run lattice + relations merge)
# ---------------------------------------------------------------------------


def _merge_lattices(artifacts: list[dict]) -> tuple[dict, list[dict]]:
    """Pure-Python merge of N lattice artifacts. Items unify by
    (formal_name, primary_link_url). Aliases and identity dicts merge;
    conflicts surface in the returned conflicts list.
    """
    by_family: dict[str, dict] = {}
    conflicts: list[dict] = []

    def primary_link(item: dict) -> str | None:
        for link in item.get("links") or []:
            if isinstance(link, dict) and link.get("url"):
                return str(link.get("url"))
        return None

    items_by_key: dict[tuple[str, str | None], dict] = {}
    for art in artifacts:
        groups = art.get("groups")
        if groups is None:
            # Prior merge artifacts nest the lattice one level down —
            # accepted so cross-seed merges can consume merge outputs.
            groups = (art.get("lattice") or {}).get("groups")
        for grp in groups or []:
            family = grp.get("family") or ""
            family_entry = by_family.setdefault(family, {
                "family": family,
                "identity_keys": list(grp.get("identity_keys") or []),
                "items": [],
            })
            existing_keys = list(family_entry["identity_keys"])
            for key in grp.get("identity_keys") or []:
                if key not in existing_keys:
                    existing_keys.append(key)
            family_entry["identity_keys"] = existing_keys

            for item in grp.get("items") or []:
                formal = item.get("formal_name") or ""
                key = (formal, primary_link(item))
                if key in items_by_key:
                    target = items_by_key[key]
                    aliases = list(target.get("aliases") or [])
                    for alias in item.get("aliases") or []:
                        if alias not in aliases:
                            aliases.append(alias)
                    target["aliases"] = aliases
                    target_links = {
                        (l.get("kind"), l.get("url")): l
                        for l in (target.get("links") or [])
                        if isinstance(l, dict)
                    }
                    for link in item.get("links") or []:
                        if not isinstance(link, dict):
                            continue
                        target_links.setdefault(
                            (link.get("kind"), link.get("url")), link
                        )
                    target["links"] = list(target_links.values())
                    target_identity = dict(target.get("identity") or {})
                    new_identity = item.get("identity") or {}
                    for ikey, ival in new_identity.items():
                        if ikey not in target_identity:
                            target_identity[ikey] = ival
                        elif target_identity[ikey] != ival:
                            conflicts.append({
                                "kind": "identity_value",
                                "formal_name": formal,
                                "identity_key": ikey,
                                "values": sorted({
                                    str(target_identity[ikey]),
                                    str(ival),
                                }),
                            })
                    target["identity"] = target_identity
                else:
                    new_item = {
                        "kind": item.get("kind"),
                        "formal_name": formal,
                        "identity": dict(item.get("identity") or {}),
                        "aliases": list(item.get("aliases") or []),
                        "links": [
                            dict(l) for l in (item.get("links") or [])
                            if isinstance(l, dict)
                        ],
                    }
                    items_by_key[key] = new_item
                    family_entry["items"].append(new_item)

    return ({"groups": list(by_family.values())}, conflicts)


def _endpoint_str(v: object) -> str:
    """Edge endpoints are normally strings; tolerate dict-shaped
    endpoints from older artifacts by pivoting to their formal_name."""
    if isinstance(v, dict):
        return v.get("formal_name") or v.get("name") or ""
    return v or ""


def _merge_relations(
    artifacts: list[dict],
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Pure-Python merge of N relate artifacts. Edges unify by
    (subject, relation, object). The accumulated `anchor_list` of
    each merged edge carries every source from every contributing
    artifact. Differing per-edge descriptions surface in conflicts.

    Each artifact is either relate-shaped — `{operations:
    [{description, anchor_list, edges: [...]}]}` — or a prior merge
    artifact: flat `relations[]` plus an `operations` index dict
    (accepted so cross-seed merges can consume per-seed merge
    outputs). Edges are flattened and unified by triple; every merged
    edge keeps its `operation_ids` and the returned operations index
    `{operation_id: {description, anchor_list, batch_id}}` preserves
    the event structure.
    """
    by_key: dict[tuple[str, str, str], dict] = {}
    operations: dict[str, dict] = {}
    conflicts: list[dict] = []
    for art in artifacts:
        bid = art.get("batch_id")
        ops = art.get("operations")
        edge_iter: list[tuple[str | None, dict]] = []
        if isinstance(ops, dict):
            # Flat-edge input: a prior merge artifact (`relations[]`) or
            # a reconcile artifact (`edges[]`). Union its operations
            # index and surfaced conflicts; its edges already carry
            # operation_ids and provenance.
            for op_id, op in ops.items():
                if isinstance(op, dict):
                    operations.setdefault(str(op_id), op)
            conflicts.extend(c for c in art.get("conflicts") or []
                             if isinstance(c, dict))
            flat = art.get("relations")
            if not isinstance(flat, list):
                flat = art.get("edges")
            edge_iter = [(None, e) for e in flat or []
                         if isinstance(e, dict)]
        else:
            for i, op in enumerate(ops or []):
                if not isinstance(op, dict):
                    continue
                op_id = str(op.get("operation_id")
                            or f"op:{str(bid or 'nobatch')[:8]}:{i}")
                operations.setdefault(op_id, {
                    "description": op.get("description"),
                    "anchor_list": list(op.get("anchor_list") or []),
                    "batch_id": bid,
                })
                for edge in op.get("edges") or []:
                    if isinstance(edge, dict):
                        edge_iter.append((op_id, edge))
        for op_id, edge in edge_iter:
            key = (
                _endpoint_str(edge.get("subject")),
                _endpoint_str(edge.get("relation")),
                _endpoint_str(edge.get("object")),
            )
            anchors = list(edge.get("anchor_list") or [])
            traced_targets = [
                t for t in (edge.get("traced_targets")
                            or ([edge["traced_target"]]
                                if edge.get("traced_target") else []))
            ]
            edge_op_ids = list(edge.get("operation_ids") or [])
            if op_id and op_id not in edge_op_ids:
                edge_op_ids.append(op_id)
            if key in by_key:
                target = by_key[key]
                target.setdefault("anchor_list", []).extend(anchors)
                for t in traced_targets:
                    if t not in target.setdefault("traced_targets", []):
                        target["traced_targets"].append(t)
                for oid in edge_op_ids:
                    if oid not in target.setdefault("operation_ids", []):
                        target["operation_ids"].append(oid)
                if edge.get("provenance_review"):
                    target["provenance_review"] = True
                this_kind = edge.get("dependency_kind")
                if (this_kind and target.get("dependency_kind")
                        and this_kind != target["dependency_kind"]):
                    # Runs disagree on the kind of the same triple:
                    # flag for review instead of silently keeping
                    # the first writer's label.
                    conflicts.append({
                        "kind": "dependency_kind_variant",
                        "subject": edge.get("subject"),
                        "relation": edge.get("relation"),
                        "object": edge.get("object"),
                        "values": sorted({this_kind, target["dependency_kind"]}),
                    })
                target_desc = target.get("description")
                this_desc = edge.get("description")
                if (this_desc and target_desc and this_desc != target_desc
                        and this_desc not in (target.get("description_variants") or [])):
                    variants = list(target.get("description_variants") or [])
                    if target_desc not in variants:
                        variants.append(target_desc)
                    variants.append(this_desc)
                    target["description_variants"] = variants
                    conflicts.append({
                        "kind": "description_variant",
                        "subject": edge.get("subject"),
                        "relation": edge.get("relation"),
                        "object": edge.get("object"),
                        "variants": variants,
                    })
            else:
                # Keep every non-private field the source edge carries —
                # reconcile inputs add subsumed_by / subsumes /
                # corroboration_count / description_variants, and those
                # must survive into the merged artifact. Union fields
                # get fresh copies.
                merged = {k: v for k, v in edge.items()
                          if not k.startswith("_") and k != "traced_target"}
                merged["anchor_list"] = anchors
                merged["traced_targets"] = traced_targets
                merged["operation_ids"] = edge_op_ids
                by_key[key] = merged
    return (list(by_key.values()), operations, conflicts)


def run_merge(
    *,
    sources: list[str] | None = None,
    relations_sources: list[str] | None = None,
    artifact_path: str | None = None,
) -> dict:
    """Pure-Python cross-run merge. Reads N lattice JSONs and N
    relations JSONs (counts may differ — relations merge is optional)
    and writes one merged artifact.

    With `--artifact`, ingest an externally produced merge artifact
    for shape validation only.
    """
    if artifact_path:
        run_id = new_run("merge", label="merge:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        if not isinstance(artifact, dict) or "lattice" not in artifact:
            raise click.ClickException("merge artifact missing 'lattice' field")
        group_count = len(artifact.get("lattice", {}).get("groups") or [])
        item_count = sum(
            len(g.get("items") or [])
            for g in artifact.get("lattice", {}).get("groups") or []
        )
        relation_count = len(artifact.get("relations") or [])
        conflict_count = len(artifact.get("conflicts") or [])
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            "relation_count": relation_count,
            "conflict_count": conflict_count,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count,
                "relation_count": relation_count,
                "conflict_count": conflict_count}

    if not sources:
        # Bare `modsleuth run merge`: default to the current storage's
        # latest lattice, and for relations prefer the latest reconcile
        # artifact — its subsumption marks and sibling conflicts must
        # survive into the final graph. Fall back to the completed
        # per-batch relate artifacts when reconcile hasn't run.
        sources = [str(_latest_lattice_artifact_path())]
        if relations_sources is None:
            recon = sorted(
                (config.STORAGE / config.RUNS_SUBDIR).glob(
                    f"*/{config.RECONCILE_ARTIFACT_FILE}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if recon:
                relations_sources = [str(recon[0])]
            else:
                rows = all_rows(
                    "SELECT artifact_path FROM batch_artifacts "
                    "WHERE stage='relate' AND status='complete'"
                )
                relations_sources = [
                    row["artifact_path"] for row in rows
                    if row["artifact_path"] and Path(row["artifact_path"]).exists()
                ]
    lattice_artifacts = [read_json(s) for s in sources]
    merged_lattice, lattice_conflicts = _merge_lattices(lattice_artifacts)

    merged_relations: list[dict] = []
    merged_operations: dict[str, dict] = {}
    relation_conflicts: list[dict] = []
    if relations_sources:
        rel_artifacts = [read_json(s) for s in relations_sources]
    else:
        # Cross-seed merge: sources that are prior merge artifacts
        # carry their relations too — merge those by default.
        rel_artifacts = [a for a in lattice_artifacts
                         if isinstance(a.get("relations"), list)]
    if rel_artifacts:
        merged_relations, merged_operations, relation_conflicts = (
            _merge_relations(rel_artifacts)
        )

    # Provenance routing over the merged graph (same flag-only rule as
    # reconcile). Recomputed here because the merge may have united an
    # edge's evidence across runs: gaining the subject's own run clears
    # the flag, evidence from foreign runs only sets it.
    provenance_review_count = 0
    for e in merged_relations:
        subj_org = _org_prefix(str(e.get("subject") or ""))
        target_orgs = {
            org for org in (
                _org_prefix(str(t)) for t in (e.get("traced_targets") or [])
            ) if org
        }
        if subj_org and target_orgs:
            if subj_org not in target_orgs:
                e["provenance_review"] = True
            else:
                e.pop("provenance_review", None)
        if e.get("provenance_review"):
            provenance_review_count += 1

    run_id = new_run("merge", label="merge")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.MERGE_ARTIFACT_FILE

    payload = {
        "sources": list(sources),
        "relations_sources": list(relations_sources or []),
        "lattice": merged_lattice,
        "relations": merged_relations,
        "operations": merged_operations,
        "conflicts": lattice_conflicts + relation_conflicts,
    }
    atomic_write_json(artifact_out, payload)

    group_count = len(merged_lattice.get("groups") or [])
    item_count = sum(
        len(g.get("items") or []) for g in merged_lattice.get("groups") or []
    )
    relation_count = len(merged_relations)
    conflict_count = len(payload["conflicts"])
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "sources": list(sources),
        "relations_sources": list(relations_sources or []),
        "group_count": group_count,
        "item_count": item_count,
        "relation_count": relation_count,
        "operation_count": len(merged_operations),
        "conflict_count": conflict_count,
        "provenance_review_count": provenance_review_count,
    })
    return {
        "run_id": run_id,
        "artifact_path": str(artifact_out),
        "group_count": group_count,
        "item_count": item_count,
        "relation_count": relation_count,
        "operation_count": len(merged_operations),
        "conflict_count": conflict_count,
    }


# ---------------------------------------------------------------------------
# expand — operator-driven recursion. CLI wrapper that runs the full
# pipeline against an upstream node (queued by triage).
# ---------------------------------------------------------------------------


def run_expand(
    *,
    node: str,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
    skip: tuple[str, ...] = (),
) -> dict:
    """Run discover → reconcile against `node` as a fresh target inside
    the current storage. Deliberately stops before triage/merge — the
    caller (operator or recursive driver) merges once after its batch
    of expansions. Pass `skip=(...)` to skip stages. Each stage's
    result is captured in the returned dict.
    """
    out: dict[str, Any] = {"node": node, "stages": {}}
    if "discover" not in skip:
        out["stages"]["discover"] = run_discover(
            target=node,
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "extract" not in skip:
        out["stages"]["extract"] = run_extract(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "organize" not in skip:
        out["stages"]["organize"] = run_organize(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "audit" not in skip:
        out["stages"]["audit"] = run_audit(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "relate" not in skip:
        out["stages"]["relate"] = run_relate(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "reconcile" not in skip:
        try:
            out["stages"]["reconcile"] = run_reconcile()
        except click.ClickException as exc:
            out["stages"]["reconcile"] = {"status": "skipped", "reason": str(exc)}
    return out
