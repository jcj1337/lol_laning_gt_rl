#!/usr/bin/env bash
# EXPLORATORY powered 150-turn premium (amendment of 2026-10-15) and the representative traces, after the extension chain.
set -e
P="python -m lanegame.experiments.rl.m46c"
$P eval_premium150p eval_premium150p_ext --workers 7
$P traces
$P analyse report
