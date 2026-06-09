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

load_dotenv(ROOT / ".env")

# Environment variable names
MODSLEUTH_STORAGE_ENV = "MODSLEUTH_STORAGE"
MODSLEUTH_PATH_ENV = "MODSLEUTH_PATH"
MODSLEUTH_RUN_ID_ENV = "MODSLEUTH_RUN_ID"

STORAGE = Path(os.environ.get(MODSLEUTH_STORAGE_ENV) or ROOT / "storage").resolve()
DB_PATH = Path(os.environ.get(MODSLEUTH_PATH_ENV) or STORAGE / "graph.db").resolve()

# Storage layout — directory names under STORAGE and run_root
RUNS_SUBDIR = "runs"
SOURCES_SUBDIR = "sources"
WORKSPACE_SUBDIR = "workspace"
WORKERS_SUBDIR = "workers"
BATCH_SUBDIR = "batch"

# Per-run files (under STORAGE/runs/<run_id>/)
RUN_PROMPT_FILE = "prompt.md"
RUN_STDOUT_FILE = "stdout.txt"      # codex (plain text)
RUN_STREAM_FILE = "stream.jsonl"    # claude (--output-format stream-json)
RUN_STDERR_FILE = "stderr.txt"
RUN_INPUT_FILE = "input.json"
BATCH_MANIFEST_FILE = "MANIFEST.txt"

# Pipeline stages (execution order)
STAGE_NAMES = ("discover", "extract", "organize", "audit",
               "relate", "reconcile", "triage", "merge")

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

# Models
CLAUDE_MODEL = os.environ.get("MODSLEUTH_CLAUDE_MODEL", "opus")
CODEX_MODEL = os.environ.get("MODSLEUTH_CODEX_MODEL", "gpt-5.5")
CODEX_EFFORT_CHOICES = ("low", "medium", "high", "xhigh")

# CLI-restricted model choices. The planner is always Claude (only
# Claude can drive the Task tool / artifact-writing flow); the
# subagent may be Claude or Codex.
PLANNER_CHOICES = ("opus", "sonnet")
SUBAGENT_CHOICES = (
    "opus", "sonnet",
    "codex-low", "codex-medium", "codex-high", "codex-xhigh",
)

# `{{subagent_prompt}}` templates rendered by
# `pipeline.subagent_prompt_for(model)`. The planner reads one of
# these to learn how to dispatch sub-work this run.

SUBAGENT_PROMPT_CLAUDE = (
    "## Subagent dispatch (Task tool)\n"
    "\n"
    "The Task tool is available — subagents run as `{model}`. "
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

SUBAGENT_PROMPT_CODEX = (
    "You are running as the **orchestrator**. Plan and synthesize "
    "the final artifact yourself, but delegate every unit of "
    "reading-and-analysis to a codex subagent. **Do NOT call the "
    "`Agent` or `Task` tool** — they are off-policy for this run. "
    "Subagents are dispatched ONLY via the codex CLI.\n"
    "\n"
    "Each codex subagent runs in a **fresh process with no shared "
    "context** — it cannot see your prompt or your running state. "
    "Brief it like a stranger: include filesystem scope, schema, "
    "rules, and the completion contract in its prompt.\n"
    "\n"
    "## CLI invocation\n"
    "\n"
    "```bash\n"
    "codex exec -m {codex_model} \\\n"
    "  -c model_reasoning_effort={effort} \\\n"
    "  --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \\\n"
    "  \"<self-contained prompt: scope + schema + rules + completion>\"\n"
    "```\n"
    "\n"
    "Dispatch and wait for codex subagents in **one** Bash "
    "invocation, not across multiple. Pass `timeout: 600000` "
    "(10 minutes) on this Bash call. The Bash tool's default "
    "2-minute timeout will kill a long `wait` and leave codex "
    "grandchildren orphaned in the process tree."
)
