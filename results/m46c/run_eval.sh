#!/usr/bin/env bash
# M4.6c evaluation chain (after the 12 training runs): every stage is resumable (per-job records on disk).
set -e
P="python -m lanegame.experiments.rl.m46c"
W=${W:-7}
$P train_check
$P eval_premium eval_premium_mid --workers $W
$P analyse
$P eval_h2h --workers $W
$P eval_battery --workers $W
$P eval_natural eval_records --workers $W
$P eval_natural_cf --workers $W
$P eval_freeze --workers $W
$P eval_specialists --workers $W
$P eval_premium150 --workers $W
$P analyse report
