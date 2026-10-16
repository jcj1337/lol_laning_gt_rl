#!/usr/bin/env bash
# M4.6c pre-registered extension (amendment of 2026-10-15): 12 runs x 8,192,000, then the amended evaluation families.
set -e
P="python -m lanegame.experiments.rl.m46c"
W=${W:-7}
$P train --phase ext --workers 6
$P train_check --phase ext
$P eval_premium_ext eval_h2h_ext eval_roleswap eval_roleswap_ext --workers $W
$P eval_natural_ext eval_battery_ext eval_natural_cf_ext --workers $W
$P analyse report
