#!/usr/bin/env bash
# M4.6f evaluation queue of the 4.096M phase, as run (from the repository root, after `train --phase main`).
set -e
PY=".venv/Scripts/python -m lanegame.experiments.rl.m46f"
$PY train_check --phase main
$PY value --phase main --part calib --workers 8
$PY value --phase mid --part calib --workers 8
$PY value --phase main --part natural,probes --workers 8
$PY behaviour --phase main --workers 8
$PY ext_rule analyse report
