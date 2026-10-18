# Training and evaluation

Training lets a user run shared-policy PPO self-play (one network playing both seats against a current/historical
mixture) and get a checkpoint with its history. The checkpoint carries a SHA-256 digest of its weights. Evaluation
plays a champion-game checkpoint head to head against scripted opponents in both seats and prints payoff, win rate
and behaviour statistics. Training on a fixed seed is bit-reproducible.

## Sub-features

- `train-champions` trains on the finite champion game and writes `final.pt`, `history.jsonl` and `run.json`.
- `train-m4` trains on an M4 preset under continuing time.
- `train-determinism` reproduces: the same seed and arguments give the same weights digest.
- `eval-head-to-head` evaluates a champion checkpoint against named scripted opponents.
- `eval-wrong-game` rejects a checkpoint whose action space does not match the champion game.

## How to get to it (user POV)

- `python -m lanegame rl-shared --model <champions|m4|m4_lasthit|m4_four_actions_econ|...> --steps N --n-envs E --seed S --out DIR`
- `python -m lanegame rl-eval CHECKPOINT --opponents a,b,... --episodes N [--seed S] [--greedy] [--skill x,y]`

## Driving it with vl.py

Preconditions:

- A run exists (`RUN`), `$VL doctor --run "$RUN"` exits 0, and its `module torch` line is PASS.

- **Train on the champion game.** Run `$VL cli --run "$RUN" --label train-a --tail 3 -- rl-shared --model champions --steps 4096 --n-envs 4 --seed 0 --out a/run`.
  Exit 0 in about 5 s. The last line is `shared self-play: 4096 agent transitions, 2048 environment turns in …s; pool []`.
  The transcript lists `a/run/final.pt`, `a/run/history.jsonl` and `a/run/run.json`. `run.json` holds
  `"method": "shared_selfplay"` and `"seed": 0`.
- **Retrain with the same seed.** Run `$VL cli --run "$RUN" --label train-b --tail 3 -- rl-shared --model champions --steps 4096 --n-envs 4 --seed 0 --out b/run`.
  Keep the same basename `run`, because the run name is stored in the checkpoint. Then read both weights digests:
  `.venv/Scripts/python.exe -c "import sys; from lanegame.rl import checkpoint_info as ci; [print(p, ci(p)['digest']) for p in sys.argv[1:]]" "$RUN/evidence/train-a.files/a/run/final.pt" "$RUN/evidence/train-b.files/b/run/final.pt" | tee "$RUN/evidence/train-digests.txt"`.
  The two digests are equal (`625db359…` on the reference run).
- **Evaluate.** Run `$VL cli --run "$RUN" --label eval-a -- rl-eval a/run/final.pt --opponents hold_farmer,aggressive_trader --episodes 40`.
  Exit 0. The header reads `run vs each opponent, both seats, 40 episodes per seat, seed 10000`. Then one row per
  opponent. The reference run gave `vs hold_farmer -4.26 +- 0.37 win 0.01 seat A -4.39 B -4.14 …` and
  `vs aggressive_trader -1.84 +- 0.42 win 0.17 …`.
- **Evaluate the retrained copy.** Run `$VL cli --run "$RUN" --label eval-b -- rl-eval b/run/final.pt --opponents hold_farmer,aggressive_trader --episodes 40`.
  The rows are identical to `eval-a`'s.
- **Train on an M4 preset.** Run `$VL cli --run "$RUN" --label train-m4 --tail 3 -- rl-shared --model m4_four_actions_econ --steps 4096 --n-envs 4 --seed 0 --out m4/run`.
  Exit 0. The update lines read `[run] update 4 transitions 4096 turns 2048 …`, and the checkpoint files are written.
- **Wrong game.** Run `$VL cli --run "$RUN" --label eval-m4 -- rl-eval m4/run/final.pt --opponents hold_farmer --episodes 10`.
  Exit 1. Stderr ends with `ValueError: run: the network has 61 action outputs but this environment's action space has 46 codes …`.
- **Proof.** The `train-*` and `eval-*` transcripts, `train-digests.txt` with equal digests, and a clean repo guard.

## Gotchas

- The `--steps` default is 307,200 transitions, which takes minutes. Always pass `--steps 4096 --n-envs 4`.
- The default `--out` is `results/rl/checkpoints/rl_shared`. Under `vl.py` it resolves inside `RUN/work/cli`, but pass
  `--out` so the evidence names the run.
- Raw `final.pt` bytes differ between identical runs, because the checkpoint carries provenance. `history.jsonl` also
  differs, but only in the timing fields `collect_s`, `update_s` and `elapsed`. Compare weights digests and evaluation
  rows instead.
- `rl-eval` and `rl-br` load the champion game only (`GameConfig.champions`). An M4 checkpoint fails with the
  action-space error above; that is the expected behaviour, not a regression.
- `--model` accepts only the presets in `RL_MODELS` of `lanegame/__main__.py`. The research preset `m45_calibrated`
  is trained through the experiment drivers, which this skill does not run.
- Reference numbers were observed on 2026-10-18 at commit 82568827 (CPU torch 2.14.1). Any change to the observation,
  the network or PPO moves them. A digest mismatch between two same-seed runs of one checkout is a bug.
