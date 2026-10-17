#!/bin/sh
# M4.6e queue after the label stage (run from the repository root)
PY=.venv/Scripts/python.exe
$PY -m lanegame.experiments.rl.m46e fit --workers 6 > results/m46e/fit.log 2>&1
$PY -m lanegame.experiments.rl.m46e ranking --workers 6 > results/m46e/ranking.log 2>&1
$PY -m lanegame.experiments.rl.m46e exposure --workers 6 > results/m46e/exposure.log 2>&1
$PY -m lanegame.experiments.rl.m46e observation --workers 6 > results/m46e/observation.log 2>&1
echo "queue done"
