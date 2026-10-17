#!/usr/bin/env bash
# M4.6f: the pre-registered extension (fired by ext_rule on the 4.096M results) and its evaluation, as run.
set -e
PY=".venv/Scripts/python -m lanegame.experiments.rl.m46f"
$PY train --phase ext --workers 6
$PY train_check --phase ext
$PY value --phase ext --workers 8
$PY behaviour --phase ext --workers 8
$PY analyse report
