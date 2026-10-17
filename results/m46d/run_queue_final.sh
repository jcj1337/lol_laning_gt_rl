#!/bin/bash
# M4.6d final queue: the confirmation sample once N is done (3 workers, alongside Ab), then the critic ranking once the
# calibration is done (5 workers).
until grep -q "matrix N\] 144 jobs" results/m46d/matrix_BNAb.log 2>/dev/null; do sleep 15; done
python -m lanegame.experiments.rl.m46d confirm --workers 3 > results/m46d/confirm.log 2>&1
until grep -q "== critic done" results/m46d/critic_calib.log 2>/dev/null; do sleep 15; done
python -m lanegame.experiments.rl.m46d critic --part rank --workers 5 > results/m46d/critic_rank.log 2>&1
