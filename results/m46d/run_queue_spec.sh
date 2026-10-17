#!/bin/bash
# M4.6d queue (workers 3): waits for the C1 verify stage, then the decomposition and the critic diagnostic.
until grep -q "== specialists done" results/m46d/spec_verify.log 2>/dev/null; do sleep 20; done
python -m lanegame.experiments.rl.m46d specialists --part decomp --workers 3 > results/m46d/spec_decomp.log 2>&1
python -m lanegame.experiments.rl.m46d critic --part calib --workers 3 > results/m46d/critic_calib.log 2>&1
