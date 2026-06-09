#!/usr/bin/env python3
"""
launch_baselines.py — fire all 16 baseline investigator runs in parallel.

  Subjects (4): olmo3, nemotron3_super, dr_tulu, smollm3
  Systems  (4): o3-deep-research, gpt-5.5-pro, gpt-5.4-pro (OpenAI background mode)
                claude-code (local `claude` CLI in print mode)

Each system gets the same per-subject baseline prompt (see ./prompts/).
Outputs land at outputs/<system>_<subject>.json (raw response text). The
baseline prompt instructs each model to emit pure JSON, so the file is
parseable JSON in the success case; if it isn't (a few Claude Code runs
have prefaced JSON with one line of preamble), strip the preamble manually.

Requires:
  - OPENAI_API_KEY env var (for the OpenAI runs)
  - `claude` CLI installed and logged in (for the Claude Code runs)
  - pip install openai

Run:
    cd baselines
    OPENAI_API_KEY=sk-... python3 launch_baselines.py

Re-running skips outputs that already exist, so failures can be retried by
deleting individual <system>_<subject>.json files and relaunching.
"""

import asyncio
import os
import sys
import time
import tempfile
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai")

REPO_ROOT  = Path(__file__).resolve().parent.parent
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

SUBJECTS = ["olmo3", "nemotron3_super", "dr_tulu", "smollm3"]

# Filename slug -> OpenAI Responses-API model id. Adjust the `model` strings
# if your account exposes them under different IDs.
OPENAI_SYSTEMS = {
    "o3dr":     {"model": "o3-deep-research", "background": True},
    "gpt55pro": {"model": "gpt-5.5-pro",      "background": True},
    "gpt54pro": {"model": "gpt-5.4-pro",      "background": True},
}

CLAUDE_CODE_SLUG = "cc"

POLL_S    = 15           # poll interval for OpenAI background jobs
MAX_RUN_S = 60 * 90      # 90 min cap per run


def out_path(slug: str, subject: str) -> Path:
    return OUTPUT_DIR / f"{slug}_{subject}.json"


def load_prompt(subject: str) -> str:
    p = PROMPT_DIR / f"baseline_prompt_{subject}.md"
    if not p.exists():
        raise FileNotFoundError(f"Prompt not found: {p}")
    return p.read_text()


async def run_openai(client: AsyncOpenAI, slug: str, cfg: dict, subject: str) -> None:
    target = out_path(slug, subject)
    tag = f"[{slug}/{subject}]"
    if target.exists():
        print(f"{tag} skip (output already exists)", flush=True)
        return

    prompt = load_prompt(subject)
    t0 = time.time()
    print(f"{tag} START  model={cfg['model']} bg={cfg['background']}", flush=True)

    try:
        if cfg["background"]:
            r = await client.responses.create(
                model=cfg["model"],
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                background=True,
            )
            rid = r.id
            elapsed = 0
            while elapsed < MAX_RUN_S:
                await asyncio.sleep(POLL_S)
                elapsed += POLL_S
                r = await client.responses.retrieve(rid)
                if r.status in ("completed", "failed", "cancelled", "expired"):
                    break
            if r.status != "completed":
                print(f"{tag} FAIL   status={r.status} elapsed={elapsed}s", flush=True)
                return
            text = r.output_text
        else:
            r = await client.responses.create(
                model=cfg["model"],
                input=prompt,
                tools=[{"type": "web_search_preview"}],
            )
            text = r.output_text

        target.write_text(text)
        print(f"{tag} DONE   {int(time.time() - t0)}s -> {target.name}", flush=True)

    except Exception as e:
        print(f"{tag} EXCEPTION: {type(e).__name__}: {e}", flush=True)


async def run_claude_code(subject: str) -> None:
    slug = CLAUDE_CODE_SLUG
    target = out_path(slug, subject)
    tag = f"[{slug}/{subject}]"
    if target.exists():
        print(f"{tag} skip (output already exists)", flush=True)
        return

    prompt = load_prompt(subject)
    t0 = time.time()
    print(f"{tag} START  claude -p (headless)", flush=True)

    # Run inside a temp dir so any local CLAUDE.md / project state doesn't
    # leak into the investigator's context.
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "--model", "claude-opus-4-7[1m]",   # Opus 4.7, 1M context (paper §C)
            "--dangerously-skip-permissions",   # required for headless runs
            cwd=tmpdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()),
                timeout=MAX_RUN_S,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            print(f"{tag} TIMEOUT after {MAX_RUN_S}s", flush=True)
            return

    if proc.returncode != 0:
        print(
            f"{tag} FAIL   rc={proc.returncode}  "
            f"stderr={stderr.decode(errors='replace')[:400]}",
            flush=True,
        )
        return

    target.write_bytes(stdout)
    print(f"{tag} DONE   {int(time.time() - t0)}s -> {target.name}", flush=True)


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; export it before running.")

    for s in SUBJECTS:
        load_prompt(s)  # fail fast if any prompt is missing

    client = AsyncOpenAI()
    tasks: list = []

    for slug, cfg in OPENAI_SYSTEMS.items():
        for subject in SUBJECTS:
            tasks.append(run_openai(client, slug, cfg, subject))
    for subject in SUBJECTS:
        tasks.append(run_claude_code(subject))

    print(f"Launching {len(tasks)} runs in parallel ...\n", flush=True)
    await asyncio.gather(*tasks, return_exceptions=False)
    print("\nAll runs finished. Check logs above for any FAIL/EXCEPTION/TIMEOUT.")


if __name__ == "__main__":
    asyncio.run(main())
