from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import click

from . import config


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def loads(raw: str | None, default: Any = None) -> Any:
    if raw in (None, ""):
        return {} if default is None else default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {} if default is None else default


def read_json(path: str | Path | None) -> Any:
    raw = Path(path).read_text() if path else sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"bad JSON: {exc}") from exc


def emit_json(value: Any) -> None:
    click.echo(json_text(value))


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def key(value: Any) -> str:
    return normalize_space(value).casefold()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def truncate(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    suffix = " ... [truncated]"
    keep = max(limit - len(suffix), 0)
    return value[:keep].rstrip() + suffix[:limit]


_MIGRATED: set[str] = set()


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(config.SCHEMA_PATH.read_text())
    conn.commit()


@contextmanager
def db():
    config.STORAGE.mkdir(parents=True, exist_ok=True)
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    existed = config.DB_PATH.exists()
    conn = sqlite3.connect(config.DB_PATH, timeout=config.SQLITE_BUSY_TIMEOUT_S)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {int(config.SQLITE_BUSY_TIMEOUT_S * 1000)}")
        db_key = str(config.DB_PATH)
        if not existed or db_key not in _MIGRATED:
            migrate(conn)
            _MIGRATED.add(db_key)
        yield conn
    finally:
        conn.close()


def all_rows(sql: str, args: tuple[Any, ...] = ()) -> list[dict]:
    with db() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(config.HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def purge_store_noise(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        for dirname in list(dirnames):
            target = Path(dirpath) / dirname
            if dirname in config.SKIP_DIRS or target.is_symlink():
                shutil.rmtree(target, ignore_errors=True) if target.is_dir() and not target.is_symlink() else target.unlink(missing_ok=True)
                dirnames.remove(dirname)
        for filename in filenames:
            target = Path(dirpath) / filename
            if target.is_symlink():
                target.unlink(missing_ok=True)


def fingerprint_dir(path: Path) -> str:
    h = hashlib.sha256()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in config.SKIP_DIRS and not (Path(dirpath) / d).is_symlink()
        ]
        for filename in filenames:
            item = Path(dirpath) / filename
            if item.is_symlink():
                continue
            rel = item.relative_to(path)
            if any(part in config.SKIP_DIRS for part in rel.parts):
                continue
            files.append(item)
    if not files:
        return hashlib.sha256(b"gdb:empty-dir:v1").hexdigest()
    for item in sorted(files, key=lambda p: p.relative_to(path).as_posix()):
        h.update(item.relative_to(path).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(item).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def content_hash(path: Path) -> str:
    return fingerprint_dir(path) if path.is_dir() else sha256_file(path)


def content_store_dir(content_hash_value: str) -> Path:
    return config.STORAGE / config.SOURCES_SUBDIR / content_hash_value[:2] / content_hash_value


def storage_ignore(src_dir: str, names: list[str]) -> set[str]:
    out: set[str] = set()
    for name in names:
        target = Path(src_dir) / name
        if name in config.SKIP_DIRS or target.is_symlink():
            out.add(name)
    return out


def store_source(src: Path, content_hash_value: str) -> Path:
    dst_dir = content_store_dir(content_hash_value)
    if src.is_dir():
        if dst_dir.exists():
            purge_store_noise(dst_dir)
            return dst_dir
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_dir.parent / f".tmp.{uuid.uuid4().hex}.{dst_dir.name}"
        try:
            shutil.copytree(src, tmp, ignore=storage_ignore)
            os.replace(tmp, dst_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return dst_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"content{src.suffix or '.bin'}"
    if dst.exists():
        return dst
    tmp = dst.with_name(f".tmp.{uuid.uuid4().hex}.{dst.name}")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def git_commit_sha(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    try:
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--ignored"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0:
            return None
        for line in status.stdout.splitlines():
            if len(line) < 4:
                continue
            rel = line[3:].split(" -> ")[-1]
            if not any(part in config.SKIP_DIRS for part in rel.split("/")):
                return None
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode == 0:
            return head.stdout.strip() or None
    except FileNotFoundError:
        return None
    return None


def guess_content_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    return {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".py": "text/x-python",
    }.get(path.suffix.lower(), "application/octet-stream")


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str) -> str:
    return _SLUG_RE.sub("-", text or "").strip("-_.") or "source"


def batch_dir_filename(title: str | None, storage_ref: str | None = None, fallback_ext: str = "") -> str:
    ext = Path(storage_ref).suffix if storage_ref else fallback_ext
    name = slug(title or "source")
    if ext and not name.lower().endswith(ext.lower()):
        return f"{name}{ext}"
    return name


def record_source_url(cur: sqlite3.Cursor, source_id: str, url: str | None) -> None:
    if not url:
        return
    timestamp = now()
    cur.execute(
        """INSERT INTO source_urls (source_id, url, first_seen_at, last_seen_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(source_id, url) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
        (source_id, url, timestamp, timestamp),
    )


def record_source_commit(cur: sqlite3.Cursor, source_id: str, commit_sha: str | None) -> None:
    if not commit_sha:
        return
    timestamp = now()
    cur.execute(
        """INSERT INTO source_commits (source_id, commit_sha, first_seen_at, last_seen_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(source_id, commit_sha) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
        (source_id, commit_sha, timestamp, timestamp),
    )


def scan_and_register(workspace_dir: Path, artifact: dict) -> tuple[dict, list[dict]]:
    enriched = json.loads(json.dumps(artifact))
    workspace_root = workspace_dir.resolve()
    per_batch_maps: list[dict] = []
    with db() as conn:
        cur = conn.cursor()
        for batch_idx, batch in enumerate(enriched.get("batches") or []):
            file_map: dict[str, str] = {}
            for source in batch.get("sources") or []:
                rel_path = source.get("path") or ""
                if not rel_path:
                    raise click.ClickException(f"batch[{batch_idx}] source missing path")
                abs_path = (workspace_dir / rel_path).resolve()
                if not abs_path.is_relative_to(workspace_root):
                    raise click.ClickException(f"source path escapes workspace: {abs_path}")
                if not abs_path.exists():
                    raise click.ClickException(f"source path does not exist: {abs_path}")
                url = normalize_space(source.get("url") or "") or None
                title = source.get("title") or abs_path.name
                raw_commit = normalize_space(source.get("commit_sha") or "") or None
                commit_sha = raw_commit
                if abs_path.is_dir() and (abs_path / ".git").exists():
                    verified = git_commit_sha(abs_path)
                    commit_sha = verified if not raw_commit or raw_commit == verified else None
                chash = content_hash(abs_path)
                existing = cur.execute("SELECT * FROM sources WHERE content_hash=?", (chash,)).fetchone()
                if existing:
                    source_id = existing["id"]
                    record_source_url(cur, source_id, url)
                    record_source_commit(cur, source_id, commit_sha)
                else:
                    stored = store_source(abs_path, chash)
                    source_id = new_id()
                    timestamp = now()
                    cur.execute(
                        """INSERT INTO sources
                           (id, content_hash, content_type, storage_ref, title,
                            canonical_url, remote_url, commit_sha, attrs, fetched_at, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            source_id,
                            chash,
                            guess_content_type(abs_path),
                            str(stored),
                            title,
                            url,
                            url if commit_sha else None,
                            commit_sha,
                            "{}",
                            timestamp,
                            timestamp,
                        ),
                    )
                    record_source_url(cur, source_id, url)
                    record_source_commit(cur, source_id, commit_sha)
                source["source_id"] = source_id
                source["content_hash"] = chash
                candidate = batch_dir_filename(title, fallback_ext=abs_path.suffix if abs_path.is_file() else "")
                used_lower = {name.lower(): sid for name, sid in file_map.items()}
                used_lower.setdefault(config.BATCH_MANIFEST_FILE.lower(), "__reserved__")
                name = candidate
                stem, dot, suffix = candidate.rpartition(".")
                if not stem:
                    stem, dot, suffix = candidate, "", ""
                counter = 2
                while name.lower() in used_lower and used_lower[name.lower()] != source_id:
                    name = f"{stem}-{counter}{dot}{suffix}" if dot else f"{stem}-{counter}"
                    counter += 1
                file_map[name] = source_id
            per_batch_maps.append({"batch_idx": batch_idx, "file_map": file_map})
        conn.commit()
    return enriched, per_batch_maps


def compute_batch_fingerprint(cur: sqlite3.Cursor, source_ids: Iterable[str]) -> str:
    pairs: list[tuple[str, str]] = []
    for source_id in sorted(set(source_ids)):
        row = cur.execute("SELECT content_hash FROM sources WHERE id=?", (source_id,)).fetchone()
        if row:
            pairs.append((source_id, row["content_hash"] or ""))
    return hash_text("|".join(f"{sid}:{chash}" for sid, chash in pairs))


def upsert_batch_by_fingerprint(
    cur: sqlite3.Cursor,
    *,
    fingerprint: str,
    source_ids: list[str],
    label: str | None,
    summary: str | None,
    file_map: dict[str, str],
) -> tuple[str, bool]:
    existing = cur.execute(
        "SELECT * FROM batches WHERE content_fingerprint=?",
        (fingerprint,),
    ).fetchone()
    timestamp = now()
    if existing:
        attrs = loads(existing["attrs"], default={}) or {}
        merged = dict(attrs.get("file_map") or {})
        existing_sids = set(merged.values())
        for filename, sid in file_map.items():
            if sid not in existing_sids:
                merged[filename] = sid
                existing_sids.add(sid)
        attrs["file_map"] = merged
        cur.execute(
            """UPDATE batches
                  SET label=COALESCE(label, ?), summary=COALESCE(summary, ?),
                      attrs=?, updated_at=?
                WHERE id=?""",
            (label, summary, dumps(attrs), timestamp, existing["id"]),
        )
        for ordinal, sid in enumerate(source_ids):
            cur.execute(
                "INSERT OR IGNORE INTO batch_sources (batch_id, source_id, ordinal) VALUES (?,?,?)",
                (existing["id"], sid, ordinal),
            )
        return existing["id"], False
    batch_id = new_id()
    cur.execute(
        """INSERT INTO batches
           (id, label, summary, content_fingerprint, attrs, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (batch_id, label, summary, fingerprint, dumps({"file_map": file_map}), timestamp, timestamp),
    )
    for ordinal, sid in enumerate(source_ids):
        cur.execute(
            "INSERT OR IGNORE INTO batch_sources (batch_id, source_id, ordinal) VALUES (?,?,?)",
            (batch_id, sid, ordinal),
        )
    return batch_id, True


def batch_file_map(cur: sqlite3.Cursor, batch_id: str) -> dict[str, str]:
    row = cur.execute("SELECT attrs FROM batches WHERE id=?", (batch_id,)).fetchone()
    if not row:
        return {}
    return dict((loads(row["attrs"], default={}) or {}).get("file_map") or {})


def materialize_batch(batch_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = ["# filename\tsource_id\ttitle"]
    with db() as conn:
        cur = conn.cursor()
        file_map = batch_file_map(cur, batch_id)
        inverse = {sid: filename for filename, sid in file_map.items()}
        rows = cur.execute(
            """SELECT s.id, s.title, s.storage_ref
                 FROM batch_sources bs JOIN sources s ON s.id=bs.source_id
                WHERE bs.batch_id=? ORDER BY bs.ordinal, s.title, s.id""",
            (batch_id,),
        ).fetchall()
    used = {config.BATCH_MANIFEST_FILE.lower()}
    for row in rows:
        stored = Path(row["storage_ref"] or "")
        filename = inverse.get(row["id"]) or batch_dir_filename(row["title"], row["storage_ref"])
        base = filename
        counter = 2
        while filename.lower() in used:
            stem, dot, suffix = base.rpartition(".")
            if not stem:
                stem, dot, suffix = base, "", ""
            filename = f"{stem}-{counter}{dot}{suffix}" if dot else f"{stem}-{counter}"
            counter += 1
        used.add(filename.lower())
        target = dest / filename
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        if stored.exists():
            os.symlink(stored, target)
        manifest.append(f"{filename}\t{row['id']}\t{row['title'] or ''}")
    (dest / config.BATCH_MANIFEST_FILE).write_text("\n".join(manifest) + "\n")
    return dest


def set_batch_artifact(
    cur: sqlite3.Cursor,
    *,
    batch_id: str,
    stage: str,
    artifact_path: str,
    status: str = "complete",
    run_id: str | None = None,
    attrs: dict | None = None,
) -> None:
    timestamp = now()
    existing = cur.execute(
        "SELECT 1 FROM batch_artifacts WHERE batch_id=? AND stage=?",
        (batch_id, stage),
    ).fetchone()
    payload = dumps(attrs or {})
    if existing:
        cur.execute(
            """UPDATE batch_artifacts
                  SET artifact_path=?, status=?, run_id=?, attrs=?, updated_at=?
                WHERE batch_id=? AND stage=?""",
            (artifact_path, status, run_id, payload, timestamp, batch_id, stage),
        )
    else:
        cur.execute(
            """INSERT INTO batch_artifacts
               (batch_id, stage, artifact_path, status, run_id, attrs, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (batch_id, stage, artifact_path, status, run_id, payload, timestamp, timestamp),
        )
