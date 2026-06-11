from __future__ import annotations

import os
import sysconfig
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


def _resolve_runtime_path(name: str) -> Path:
    primary = ROOT / name
    if primary.exists():
        return primary
    data = Path(sysconfig.get_path("data")) / "share" / "modsleuth" / name
    return data


SCHEMA_PATH = _resolve_runtime_path("schema.sql")
PROMPTS_DIR = ROOT / "modsleuth" / "prompts"
if not PROMPTS_DIR.exists():
    PROMPTS_DIR = Path(sysconfig.get_path("data")) / "share" / "modsleuth" / "prompts"

load_dotenv(Path.cwd() / ".env")

# Environment variable names
MODSLEUTH_STORAGE_ENV = "MODSLEUTH_STORAGE"
MODSLEUTH_PATH_ENV = "MODSLEUTH_PATH"
MODSLEUTH_RUN_ID_ENV = "MODSLEUTH_RUN_ID"

# Storage defaults to ./storage in the invoking directory (like `git init`);
# spawned children always receive the resolved path via MODSLEUTH_STORAGE.
STORAGE = Path(os.environ.get(MODSLEUTH_STORAGE_ENV) or Path.cwd() / "storage").resolve()
DB_PATH = Path(os.environ.get(MODSLEUTH_PATH_ENV) or STORAGE / "graph.db").resolve()

# Storage layout — directory names under STORAGE and run_root
RUNS_SUBDIR = "runs"
SOURCES_SUBDIR = "sources"
WORKSPACE_SUBDIR = "workspace"
WORKERS_SUBDIR = "workers"
BATCH_SUBDIR = "batch"

# Per-run files (under STORAGE/runs/<run_id>/)
RUN_PROMPT_FILE = "prompt.md"
RUN_STREAM_FILE = "stream.jsonl"    # claude (--output-format stream-json)
RUN_STDERR_FILE = "stderr.txt"
RUN_INPUT_FILE = "input.json"
BATCH_MANIFEST_FILE = "MANIFEST.txt"

# Per-stage artifact filenames written under each run_root
DISCOVER_ARTIFACT_FILE = "discover_artifact.json"
EXTRACT_ARTIFACT_FILE = "extract_artifact.json"
ORGANIZE_NAMES_FILE = "names.json"
ORGANIZE_ARTIFACT_FILE = "organize_artifact.json"
AUDIT_ARTIFACT_FILE = "audit_artifact.json"
RELATE_EVENTS_FILE = "relate_events.jsonl"
RELATE_ARTIFACT_FILE = "relate_artifact.json"
RECONCILE_ARTIFACT_FILE = "reconcile_artifact.json"
TRIAGE_ARTIFACT_FILE = "triage_artifact.json"
TRIAGE_RELATIONS_FILE = "relations.json"
MERGE_ARTIFACT_FILE = "merge_artifact.json"

# Directory walk filter (skipped during scan, fingerprint, copytree)
SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", ".git"}

# Timeouts and limits
SQLITE_BUSY_TIMEOUT_S = 30.0
PROCESS_KILL_GRACE_S = 5.0
MAX_PARALLEL_BATCHES = int(os.environ.get("MODSLEUTH_MAX_PARALLEL_BATCHES", "32"))
HASH_CHUNK_BYTES = 1 << 20   # streaming chunk size for sha256_file

# Models. Names are free-form: any alias (`opus`, `sonnet`, `haiku`)
# or full model ID the `claude` CLI accepts — for the stage planner
# and for the subagents the planner dispatches via the Task tool.
CLAUDE_MODEL = os.environ.get("MODSLEUTH_CLAUDE_MODEL", "opus")

# `{{subagent_prompt}}` template rendered by
# `pipeline.subagent_prompt_for(model)`. The planner reads this to
# learn how to dispatch sub-work this run.

SUBAGENT_PROMPT_CLAUDE = (
    "## Subagent dispatch (Task tool)\n"
    "\n"
    "The Task tool is available. Pass `model: \"{model}\"` on every "
    "Task call so each subagent runs as `{model}`. "
    "Use them when the work has parallel structure: a directory "
    "of sources, a list of family buckets, anything where one "
    "unit can be analyzed without reading the others. Each Task "
    "call's reading + reasoning happens in the subagent's own "
    "context, not yours, so dispatching keeps your main context "
    "free for synthesis.\n"
    "\n"
    "**You decide whether to dispatch — it's not mandatory.** "
    "Run inline when the work is small. Dispatch when there's "
    "real fan-out.\n"
    "\n"
    "When you dispatch, brief the subagent like a stranger — it "
    "has none of your context. Transcribe the relevant rules "
    "from this prompt verbatim; rule erosion at dispatch is the "
    "main cause of subagent output drifting from the rules you "
    "were given."
)
