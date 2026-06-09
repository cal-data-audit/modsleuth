PRAGMA foreign_keys = ON;

-- One row per planner / pipeline-stage invocation.
CREATE TABLE IF NOT EXISTS runs (
  id             TEXT PRIMARY KEY,
  stage          TEXT NOT NULL,
  seed           TEXT,
  parent_run_id  TEXT REFERENCES runs(id) ON DELETE SET NULL,
  label          TEXT,
  attrs          TEXT NOT NULL DEFAULT '{}',
  started_at     TEXT NOT NULL,
  ended_at       TEXT
);
CREATE INDEX IF NOT EXISTS runs_stage_idx ON runs(stage);
CREATE INDEX IF NOT EXISTS runs_seed_idx  ON runs(seed);

-- A content-addressed source artifact registered by discover.
CREATE TABLE IF NOT EXISTS sources (
  id             TEXT PRIMARY KEY,
  content_hash   TEXT NOT NULL UNIQUE,
  content_type   TEXT,
  storage_ref    TEXT,
  title          TEXT,
  canonical_url  TEXT,
  remote_url     TEXT,
  commit_sha     TEXT,
  attrs          TEXT NOT NULL DEFAULT '{}',
  fetched_at     TEXT,
  created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sources_remote_sha_idx ON sources(remote_url, commit_sha);

CREATE TABLE IF NOT EXISTS source_urls (
  source_id      TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  url            TEXT NOT NULL,
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL,
  PRIMARY KEY(source_id, url)
);
CREATE INDEX IF NOT EXISTS source_urls_url_idx ON source_urls(url);

CREATE TABLE IF NOT EXISTS source_commits (
  source_id      TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  commit_sha     TEXT NOT NULL,
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL,
  PRIMARY KEY(source_id, commit_sha)
);
CREATE INDEX IF NOT EXISTS source_commits_sha_idx ON source_commits(commit_sha);

-- A batch is a set of sources discover groups together as one extract unit.
CREATE TABLE IF NOT EXISTS batches (
  id                   TEXT PRIMARY KEY,
  label                TEXT,
  summary              TEXT,
  content_fingerprint  TEXT NOT NULL UNIQUE,
  attrs                TEXT NOT NULL DEFAULT '{}',
  created_at           TEXT NOT NULL,
  updated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_sources (
  batch_id  TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  ordinal   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (batch_id, source_id)
);
CREATE INDEX IF NOT EXISTS batch_sources_source_idx ON batch_sources(source_id);

CREATE TABLE IF NOT EXISTS batch_artifacts (
  batch_id       TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  stage          TEXT NOT NULL,
  artifact_path  TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','complete','failed','superseded')),
  run_id         TEXT REFERENCES runs(id) ON DELETE SET NULL,
  attrs          TEXT NOT NULL DEFAULT '{}',
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  PRIMARY KEY (batch_id, stage)
);
CREATE INDEX IF NOT EXISTS batch_artifacts_stage_idx ON batch_artifacts(stage, status);

-- One row per name the extract planner found in a batch. The only fields
-- the planner emits are kind ('model'|'dataset') and the name string. No
-- anchors, atoms, identity, links, or descriptions live here.
CREATE TABLE IF NOT EXISTS names (
  id          TEXT PRIMARY KEY,
  batch_id    TEXT REFERENCES batches(id) ON DELETE SET NULL,
  run_id      TEXT REFERENCES runs(id) ON DELETE SET NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('model','dataset')),
  name        TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS names_kind_name_idx ON names(kind, name);
CREATE INDEX IF NOT EXISTS names_batch_idx     ON names(batch_id);

-- The organize stage writes its lattice artifact to disk; the path
-- lives in the run's `attrs.artifact_path`. No dedicated table — the
-- runs table is the index, the file on disk is the data.
