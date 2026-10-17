#!/bin/bash
# M4.6d queue (workers 6): waits for matrix A + AL, then B, N and the baseline reference Ab.
until grep -q "== matrix done" results/m46d/matrix_A.log 2>/dev/null; do sleep 20; done
python -m lanegame.experiments.rl.m46d matrix --part A2,B,N,Ab --workers 6 > results/m46d/matrix_BNAb.log 2>&1
