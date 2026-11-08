---
name: verify-lanegame
description: Drive lanegame (this repo's League of Legends laning game) the way its user does and capture proof that a change works. Covers the `python -m lanegame` CLI (exact solver, scripted env demo, PPO train and eval) and the web demo (real headless Edge over DevTools), with transcripts, screenshots, written-file hashes and a repo-untouched guard. Use after changing mechanics, presets, the env, the CLI, RL code or the demo, when passing tests are not enough to show the user-visible behaviour.
---

# verify-lanegame

lanegame is a research codebase: a two-player zero-sum laning game, an exact solver for the early rulesets, an RL
environment and PPO self-play for the champion rulesets, and a web demo of the final agents. Its user touches two
surfaces:

- **The CLI** (primary): `python -m lanegame <command>`. The commands are listed in the docstring of
  `lanegame/__main__.py`. Everything is CPU and seconds-scale at the sizes this skill uses.
- **The web demo**: `python -m lanegame.demo` serves the API and the built frontend (`web/dist`). `$VL demo start`
  runs it on a free port in 8700-8799, `browser open` waits for the page's ready flag, and `browser js` reads the
  app's state hook. The feature file is `features/web-demo.md`; the scripted journey with screenshots is
  `python -m experiments.m6b browser`.

Not driven by this skill:
- The study drivers (`python -m experiments.<study> <stage>`). They write into the repo's `artifacts/<study>/` and
  `runs/`, many stages are multi-hour campaigns, and expensive ones are gated by `--confirm-<study>-<stage>` flags
  that mean the user approved the compute. **Never pass a `--confirm-*` flag and never run a campaign stage to verify
  something.** If a change needs one of those stages, ask the user.
- The library API and the pytest suite. Run `python -m pytest` as well, but tests are not this skill's proof.

The Streamlit lane debugger this skill once drove was retired on 2026-11-08; it is at commit `f613f0a0`.

All commands below are Git Bash, run from the repo root. The helper is
`.claude/skills/verify-lanegame/scripts/vl.py`, run with the repo's venv:

```bash
VL=".venv/Scripts/python.exe .claude/skills/verify-lanegame/scripts/vl.py"
$VL --help
```

PowerShell form: `& .venv\Scripts\python.exe .claude\skills\verify-lanegame\scripts\vl.py <command> ...`. Windows
PowerShell 5.1 can mangle non-ASCII arguments such as `"Next ▶"`, so there use the ASCII part with `--contains`
(`--text Next --contains`).

## Launch

Every verification starts a **run**: a folder outside the repo (`%TEMP%\lanegame-verify\<id>`, or
`$LANEGAME_VERIFY_ROOT/<id>`) with `evidence/` (kept), `work/` (scratch: the CLI's working directory and the browser
profile) and `state/` (the processes this run started).

```bash
RUN=$($VL new-run | sed -n 's/^RUN=//p'); echo "$RUN"
```

`new-run` also records the repo's state (git status with content hashes, plus the git-ignored `runs/checkpoints/`
folder) so later commands can prove the run did not write into the repo.

**CLI.** No server. The only setup is the venv with lanegame installed editable; if doctor says it is missing, run
`.venv/Scripts/python.exe -m pip install -e ".[test,rl,demo]"`. Each drive is one `$VL cli` call. It runs
`python -m lanegame ...` with `RUN/work/cli` as the working directory, so the CLI's relative default paths (`runs/...`)
land in the run folder, never in the repo. Files written by one call are visible to the next one (a training run's
checkpoint, then its evaluation).

**Web demo.** Build the frontend once (`cd web && npm install && npm run build`), then:

```bash
$VL demo start    --run "$RUN"   # python -m lanegame.demo on a free port in 8700-8799; ready when /api/status answers
$VL browser start --run "$RUN"   # headless Edge with a throwaway profile in RUN/work, DevTools on a random port
$VL browser open  --run "$RUN"   # navigates to the demo and waits for document.body.dataset.ready === '1'
```

The demo runs with the repo root as its working directory, because it reads the shipped networks in
`artifacts/checkpoints/` and the research recordings in `artifacts/m5/traces/`. It only reads files. Its log goes to
`RUN/evidence/demo.log`.

**Isolation.** Runs are independent: each has its own folder, demo port and browser profile, so two agents can verify
side by side. Never drive a demo this run did not start.

**Teardown** is `$VL cleanup --run "$RUN"` (see Cleanup).

## Doctor

Run it first, and again whenever anything looks off. It is read-only.

```bash
$VL doctor --run "$RUN"
```

Every line is PASS, WARN or FAIL, and the command exits 1 on any FAIL. It checks:
- the venv's Python and that `import lanegame` resolves to this checkout from outside the repo (the editable install);
- the git HEAD and the number of dirty entries (WARN only);
- torch, fastapi, psutil and websockets;
- that a Chromium browser exists.

With `--run` it also checks:
- that the demo process is ours and alive and that its status endpoint answers;
- that the browser process is alive and its DevTools endpoint answers;
- **repo untouched since new-run**. A FAIL there names the changed paths. The guard cannot tell who made a change: if
  you or the user edited a file on purpose during the run, say so in the report. A changed path under `artifacts/` is
  a real failure; stop and report it.

## Drive

### CLI

```bash
$VL cli --run "$RUN" --label <label> [--timeout 900] [--tail 15] -- <arguments of python -m lanegame>
# e.g.
$VL cli --run "$RUN" --label solve-siege-K8 -- solve --horizon 8 --out solve_K8.json --no-q
```

The call prints the command, its exit code and duration, the last lines of stdout, a stderr tail and every file the
command wrote, with its SHA-256. It exits with the command's own exit code, or 3 if the repo changed. Labels are unique
within a run; evidence is never overwritten. Pass `--out` (or `--trace`) with a relative path so outputs land in
`RUN/work/cli`. Keep budgets tiny: `--horizon 8` for the exact game (siege K = 32 takes minutes and gigabytes) and
`--steps 4096 --n-envs 4` for training (about 5 seconds).

### Web demo (headless Edge over the DevTools protocol)

| What | Command |
| --- | --- |
| Press a button | `$VL browser click --run "$RUN" --text "Next ▶"` (exact visible text; `--contains` for a substring; `--nth N` among several) |
| Wait for text | `$VL browser wait --run "$RUN" --text "Turn 4" [--timeout 30]` |
| Read the page | `$VL browser text --run "$RUN" [--label L]` |
| Read the app's state | `$VL browser js --run "$RUN" --expr "window.__demo.state()" [--label L]` (the hooks are listed in `features/web-demo.md`) |
| Capture | `$VL browser screenshot --run "$RUN" --label L [--full]` writes L.png and L.txt (page text) |

While Watch is playing the page keeps changing, so pass `--no-settle` to `click` and `wait` until playback stops.

### Feature map

`features/README.md` indexes the user-facing features, and each file there is a journey with exact commands and the
observable end state that proves it. A proof of a feature covers every entry point its file lists, not just the most
convenient one.

## Evidence

Everything goes to `RUN/evidence/` and survives cleanup:

- `actions.log`: every helper command with a timestamp and its result, in order. This is the record of what was done.
- `<label>.cmd.txt`, `.stdout.txt`, `.stderr.txt` and `.json` per CLI call. The `.json` holds the exit code, the
  duration, each written file with size and SHA-256, and the repo-guard result. Written files up to 2 MB are copied to
  `<label>.files/`.
- `<label>.png` and `.txt` per screenshot; `<label>.txt` for `browser text --label`; `<label>.json` for `browser js
  --label`.
- `demo.log`, `browser.log`, `run.json` (git HEAD at the start) and `repo_guard.json` (the final check, written by
  cleanup).

Proof standards:
- **Real user path only.** Drive `python -m lanegame` and the demo's page. Calling lanegame functions from Python is
  not proof. Reading an artifact the CLI wrote, such as a trace JSON or a checkpoint's weights digest, or the demo's
  state hook, is observing a side effect, and that is allowed.
- **Action and resulting state.** Pair each meaningful action with a capture of what changed: `browser text` or
  `browser js` after the click, and a screenshot at the end state. For the CLI, the transcript is the pair.
- **Side effects too.** Check the files a command wrote (SHA-256 in `<label>.json`) and that the repo guard stayed
  clean. A command that printed "wrote X" but left no X fails.
- **Determinism is a contract here.** The same seed and the same arguments must reproduce: identical trace SHA-256 for
  `env-demo --trace`, identical weights digest for `rl-shared`, identical evaluation numbers. A difference is a finding,
  not noise. Raw `.pt` bytes and the timing fields of `history.jsonl` legitimately differ.
- **No mocks.** The game has no external systems. Nothing here has a dry-run mode, and nothing needs one.
- **Report skips honestly.** A journey that could not run is reported with the command tried and the unmet
  precondition, never as verified through a different path.

## Cleanup

```bash
$VL cleanup --run "$RUN"
```

Cleanup:
- kills the browser and the demo **by the PIDs this run recorded**, including their child processes. It never kills
  by process name, and never touches processes this run did not start.
- writes `evidence/repo_guard.json` and deletes `RUN/work` and `RUN/state`;
- lists the evidence it kept, and exits 3 if the repo changed during the run.

Run it after every attempt, including failed ones, so no demo or browser is left holding a port. To stop just one
process, use `$VL demo stop --run "$RUN"` or `$VL browser stop --run "$RUN"`. Old evidence folders accumulate under
`%TEMP%\lanegame-verify\`. Delete them only when the user asks.

## Helpers

- `scripts/vl.py` is the only helper (standard library plus psutil and websockets, both in the venv). Every
  subcommand is shown above, and `$VL <command> --help` lists the options. It finds the repo from its own location, and
  `LANEGAME_VERIFY_BROWSER` points it at a different Chromium executable.
