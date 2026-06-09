# Baselines

`launch_baselines.py` fires the **16 single-pass baseline runs**
(4 systems × 4 subjects) reported in paper §3.3 / §C, in parallel:

- 12 OpenAI runs go through the Responses API in **background mode**
  with the `web_search_preview` tool, polled every 15 s up to a 90 min
  cap.
- 4 Claude Code runs spawn a `claude -p --model claude-opus-4-7[1m] --dangerously-skip-permissions`
  subprocess each, in a fresh `tempfile.TemporaryDirectory()` so any
  local `CLAUDE.md` / project state doesn't leak in.

Each (system, subject) pair produces one file at
`outputs/<slug>_<subject>.json`. Re-running skips outputs that already
exist, so a partial failure is recoverable by deleting just the
failing `<slug>_<subject>.json` files and re-launching.

## Run

```bash
pip install -r ../requirements.txt
OPENAI_API_KEY=sk-... python3 launch_baselines.py 2>&1 | tee /tmp/baselines.log
```

Wall time: roughly 30–90 min total. The slowest path is
`o3-deep-research`, which can take 15–30 min per subject. Watch
`/tmp/baselines.log` to track progress, or `ls -la outputs/` to see
files appearing as they complete.

## Layout of `outputs/`

The `outputs/` directory holds **24 files total**:

- **16 baseline outputs** produced by `launch_baselines.py`:
  `<slug>_<subject>.json` for every combination of
  `slug ∈ {gpt55pro, gpt54pro, cc, o3dr}` and
  `subject ∈ {olmo3, nemotron3_super, dr_tulu, smollm3}`.
- **8 ModSleuth attribution outputs** produced by
  `../eval/build_modsleuth_inputs.py` from the merged graph:
  `prov_<subject>.json` and `prov_unbounded_<subject>.json` for each
  of the four subjects. These are the inputs that let `pooled_eval.py`
  pool ModSleuth's depth-1 and unbounded scopes alongside the four
  baselines (paper §B).

## Layout of `prompts/`

`prompts/baseline_prompt.md` is the master template described in paper
§C. The four `prompts/baseline_prompt_<subject>.md` files are
target-specific instantiations — same template, with the `SUBJECT`
block populated with the target model's identifier, display name,
authoritative URLs, and recursion depth. `launch_baselines.py` reads
the per-subject file and pipes it into the model's stdin verbatim.

## Notes

- **Claude Code model** — pinned to `claude-opus-4-7[1m]` (Opus 4.7,
  1M context) at default effort, matching the paper. Edit the
  `--model` flag in `launch_baselines.py` if you need a different
  Anthropic model.
- **OpenAI model IDs** — `gpt-5.5-pro` and `gpt-5.4-pro` are the
  API-side identifiers exposed to our account. If your account exposes
  them under different IDs, edit `OPENAI_SYSTEMS` in
  `launch_baselines.py`.
- A few Claude Code runs may preface the JSON with one line of model
  preamble (e.g., "I have all the information I need. Now I'll
  compose..."). Strip before parsing if your downstream tool can't
  tolerate it. The pooled evaluator handles either form.
