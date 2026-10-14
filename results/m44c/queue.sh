#!/bin/sh
# M4.4c stage queue: every stage is resumable (finished jobs are kept in results/m44c/partial/ until its results file
# is written), so re-running this script after an interruption continues where it stopped.
cd "$(dirname "$0")/../.."
run() { log=$1; shift; python -m lanegame.experiments.rl.m44c "$@" --workers 11 >> "results/m44c/$log.log" 2>&1; }
[ -f results/m44c/recall.json ] || [ -f results/m44c/recall.json.gz ] || run recall recall
[ -f results/m44c/recall_rep.json ] || [ -f results/m44c/recall_rep.json.gz ] || run recall_rep recall_rep
[ -f results/m44c/freeze.json ] || [ -f results/m44c/freeze.json.gz ] || run freeze freeze
[ -f results/m44c/freeze_init.json ] || [ -f results/m44c/freeze_init.json.gz ] || run freeze_init freeze_init --envs econ,aggro
[ -f results/m44c/freeze_rep.json ] || [ -f results/m44c/freeze_rep.json.gz ] || run freeze_rep freeze_rep
[ -f results/m44c/zoning.json ] || [ -f results/m44c/zoning.json.gz ] || run zoning zoning
[ -f results/m44c/zoning_nat.json ] || [ -f results/m44c/zoning_nat.json.gz ] || run zoning_nat zoning_nat
[ -f results/m44c/zoning_mech.json ] || [ -f results/m44c/zoning_mech.json.gz ] || run zoning_mech zoning_mech
[ -f results/m44c/vision.json ] || [ -f results/m44c/vision.json.gz ] || run vision vision
[ -f results/m44c/waves_mech.json ] || [ -f results/m44c/waves_mech.json.gz ] || run waves_mech waves_mech
[ -f results/m44c/waves.json ] || [ -f results/m44c/waves.json.gz ] || run waves waves
echo done > results/m44c/queue.done
