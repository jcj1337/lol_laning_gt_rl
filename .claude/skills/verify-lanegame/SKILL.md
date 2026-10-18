---
name: verify-lanegame
description: Drive lanegame (this repo's League of Legends laning game) the way its user does and capture proof that a change works. Covers the `python -m lanegame` CLI (exact solver, scripted env demo, PPO train and eval) and the Streamlit lane debugger (real headless Edge over DevTools), with transcripts, screenshots, written-file hashes and a repo-untouched guard. Use after changing mechanics, presets, the env, the CLI, RL code or the viewer, when passing tests are not enough to show the user-visible behaviour.
---

# verify-lanegame

lanegame is a research codebase: a two-player zero-sum laning game, an exact solver for the early rulesets, an RL
environment and PPO self-play for the champion rulesets. Its user touches two surfaces:

- **The CLI** (primary): `python -m lanegame <command>`. The commands are listed in the docstring of
  `lanegame/__main__.py`. Everything is CPU and seconds-scale at the sizes this skill uses.
- **The lane debugger** (secondary): a Streamlit app, `lanegame/viewer/app.py`, that replays games from (setup,
  actions) and lets a person play side A by hand.

Not driven by this skill:
- The milestone experiment drivers (`python -m lanegame.experiments[.rl].<milestone> <stage>`). They write into the
  repo's `results/<milestone>/`, many stages are multi-hour campaigns, and expensive ones are gated by
  `--confirm-<milestone>-<stage>` flags that mean the user approved the compute. **Never pass a `--confirm-*` flag and
  never run a campaign stage to verify something.** If a change needs one of those stages, ask the user.
- The library API and the pytest suite. Run `python -m pytest` as well, but tests are not this skill's proof.

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

`new-run` also records the repo's state (git status with content hashes, plus the git-ignored
`results/rl/checkpoints/` folder) so later commands can prove the run did not write into the repo.

**CLI.** No server. The only setup is the venv with lanegame installed editable; if doctor says it is missing, run
`.venv/Scripts/python.exe -m pip install -e ".[test,rl,viewer]"`. Each drive is one `$VL cli` call. It runs
`python -m lanegame ...` with `RUN/work/cli` as the working directory, so the CLI's relative default paths
(`results/...`) land in the run folder, never in the repo. Files written by one call are visible to the next one
(a training run's checkpoint, then its evaluation).

**Viewer.** Three steps. Ready means the page has rendered, not only that the server answers:

```bash
$VL viewer start  --run "$RUN"   # streamlit on a free port in 8600-8699 (never 8501); waits for /_stcore/health = ok
$VL browser start --run "$RUN"   # headless Edge with a throwaway profile in RUN/work, DevTools on a random port
$VL browser open  --run "$RUN"   # navigates, waits for "Lane debugger" and for Streamlit to finish its first run
```

`viewer start` takes about 6 seconds and `browser open` takes 2. The viewer runs with the repo root as its working
directory, because Recorded trajectory mode reads `results/rl/m43b/trajectories/`. It only reads files. Its log goes to
`RUN/evidence/viewer.log`.

**Isolation.** Runs are independent: each has its own folder, viewer port, browser profile and Streamlit session, so two
agents can verify side by side. Never drive a viewer this run did not start. A user's own `streamlit run` usually sits
on 8501; `browser open` refuses a viewer port that is not owned by this run's process.

**Teardown** is `$VL cleanup --run "$RUN"` (see Cleanup).

## Doctor

Run it first, and again whenever anything looks off. It is read-only.

```bash
$VL doctor --run "$RUN"
```

Every line is PASS, WARN or FAIL, and the command exits 1 on any FAIL. It checks:
- the venv's Python and that `import lanegame` resolves to this checkout from outside the repo (the editable install);
- the git HEAD and the number of dirty entries (WARN only);
- torch, streamlit, plotly, psutil and websockets;
- that a Chromium browser exists.

With `--run` it also checks:
- that the viewer process is ours and alive, that our process tree owns the port, and that the health endpoint answers;
- that the browser process is alive and its DevTools endpoint answers;
- **repo untouched since new-run**. A FAIL there names the changed paths. The guard cannot tell who made a change: if
  you or the user edited a file on purpose during the run, say so in the report. A changed path under `results/` is a
  real failure; stop and report it.

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

### Viewer (headless Edge over the DevTools protocol)

Handles come from the app's own widget keys and visible labels. Streamlit wraps every widget created with `key=K` in a
container with class `st-key-K`.

| What | Command |
| --- | --- |
| Press a button, a tab, a radio option | `$VL browser click --run "$RUN" --text "Next ▶"` (exact visible text; `--contains` for a substring; `--key K` or `--scope sidebar\|main` to narrow; `--nth N` among several) |
| Choose in a selectbox | `$VL browser select --run "$RUN" --key matchup --option "4A-6. Crash, then Recall"` (types to filter, clicks the exact option, then checks the widget shows it) |
| Type into a number or text input | `$VL browser fill --run "$RUN" --key seed --value 7` (select all, type, Enter) |
| Wait for text | `$VL browser wait --run "$RUN" --text "Turn 4 of 40" [--timeout 30]` |
| Read the page | `$VL browser text --run "$RUN" --scope main [--label L]` |
| Read every widget's value | `$VL browser widgets --run "$RUN" [--label L]`. Selectbox and number values are **not** in the page text, so read them here. |
| Capture | `$VL browser screenshot --run "$RUN" --label L [--full]` writes L.png, L.txt (page text) and L.widgets.json. It scrolls the main pane to the top first; `--full` captures the whole page, `--here` keeps the current scroll. |

After every click, select or fill the helper waits until Streamlit settles: no running indicator, no stale elements,
and the page text unchanged across two polls. If the app raised an exception, the command fails and prints it. While
the viewer is **playing** (after "▶ Play") the page never settles, so pass `--no-settle` to `click` and `wait` until you
have pressed "⏸ Pause".

Widget keys: `mode` (radio: Strategy matchup / Manual play / Recorded trajectory), `matchup`, `config`, `scenario`,
`gap`, `seed`, `pa`, `pb`, `turns`, `turn` (slider), `speed`, `frame` (radio: start / tick 1 / tick 2 / end),
`animate_ticks`, `rec_folder`, `rec_file`, and in Manual play `m_wave`, `m_beh`, `m_obj`, `m_auto`, `m_auto_n`.
Buttons: "⏮ Start", "◀ Back", "▶ Play" / "⏸ Pause", "Next ▶", "End ⏭", "Take over A manually from turn <i>",
"Play: <action>", "Auto-play", "Undo last", "Branch here: …". Tabs: Champions, Wave management, Timelines, Event log,
Turn details, How to read this.

The line that identifies the state is the second line of `browser text --scope main`, for example
`Turn 3 of 40 · start · k = 4 · equal power · seed 0`. The first line is the matchup title, `Manual play: you are A
(blue)`, or `Recorded: <label>`.

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
- `<label>.png`, `.txt` and `.widgets.json` per screenshot; `<label>.txt` for `browser text --label`.
- `viewer.log`, `browser.log`, `run.json` (git HEAD at the start) and `repo_guard.json` (the final check, written by
  cleanup).

Proof standards:
- **Real user path only.** Drive `python -m lanegame` and the viewer's widgets. Calling lanegame functions from Python,
  using Streamlit's AppTest, or setting session state is not proof. Reading an artifact the CLI wrote, such as a trace
  JSON or a checkpoint's weights digest, is observing a side effect, and that is allowed.
- **Action and resulting state.** Pair each meaningful action with a capture of what changed: `browser text` or
  `widgets` after the click, and a screenshot at the end state. For the CLI, the transcript is the pair.
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
- kills the browser and viewer **by the PIDs this run recorded**, including their child processes. It never kills by
  process name, and never touches processes this run did not start.
- writes `evidence/repo_guard.json` and deletes `RUN/work` and `RUN/state`;
- lists the evidence it kept, and exits 3 if the repo changed during the run.

Run it after every attempt, including failed ones, so no viewer or browser is left holding a port. To stop just one
process, use `$VL viewer stop --run "$RUN"` or `$VL browser stop --run "$RUN"`. Old evidence folders accumulate under
`%TEMP%\lanegame-verify\`. Delete them only when the user asks.

## Helpers

- `scripts/vl.py` is the only helper (standard library plus psutil and websockets, both in the venv). Every
  subcommand is shown above, and `$VL <command> --help` lists the options. It finds the repo from its own location, and
  `LANEGAME_VERIFY_BROWSER` points it at a different Chromium executable.
