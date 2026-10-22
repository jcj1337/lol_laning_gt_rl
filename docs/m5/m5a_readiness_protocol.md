# M5a readiness protocol

*2026-10-17. Written before any readiness check was run; the readiness record is `results/m5/readiness.json`, and the
report is [m5a_readiness_report.md](m5a_readiness_report.md). The checks follow the M5a brief and the closeout
report's §8.4. Engineering probes made while writing the code (whether widened outputs are exact, and three
exact-reproduction spot checks) are re-run here as recorded checks.*

**Scope.** M5a readiness verifies that the M5 campaign of [m5_design.md](m5_design.md) can run as specified and that its
instruments measure what they claim. It does not train the campaign, run the evaluation battery or train attackers.

**Compute ceiling.** About 1–2 h of experimental compute in all, plus the regression suite. This is a ceiling, not a
target. If a check cannot finish within it, experimental compute stops, partial results are kept and the remaining work
is reported with a measured estimate.

**Stand-ins.** No M5 policy exists yet. Instruments are checked on the historical checkpoints they were published with,
and the campaign's two input formats are checked on the closest existing policies:
- **om format** (116 inputs): the OM2 treatment finals, trained by the same collector;
- **sp format** (90 inputs): the M4.7 control finals, pure self-play continuations of `ext/baseline:s` at γ 0.99, and
  `ext/baseline:s` itself.

Results on stand-ins test the pipeline and the costs. They carry no information about M5's outcomes.

## Checks

| id | check | method | pass rule |
|---|---|---|---|
| R0 | artifacts and lineage | read-only: every dependency present, digests verified on load; the metadata of `ext/baseline:s` and of its start | all present; the lineage reads as M4.5c fresh 16.384M (seed s, s0 starts) → M4.6c 8.192M (seed 98,000 + s, archive starts, LR re-warmed); 24.576M prior transitions |
| R1 | seed block | every M5 root in 70,000–70,999; no other study's root constant in that block | both hold |
| R2 | initialization | for every seed: the widened om network against `ext/baseline:s` on ≥ 800 states (natural and archive), with the prior, random belief features and varied previous actions | identical outputs in batched evaluation; single-row differences ≤ 10⁻⁵ (float32 rounding), reported |
| R3 | defaults unchanged | the historical digest pins and OM2's default-trainer pin (targeted pytest); exact reproduction of stored records through the hooked loops: an M4.5c battery job, an M4.6c weaker-side job, an M4.6f freeze job (C1), OM2 evaluation games | pins pass; every reproduction exact |
| R4 | smoke and re-run | each arm, seed 0, about 10 updates of the real 32.768M schedule (`stop_at`), and a second identical run | the M4.6c / OM2 smoke conditions (finite statistics, KL, no illegal action, archive starts, initial digest, input width); both re-runs bit-identical (weights and deterministic history fields) |
| R5 | common random numbers | the first 200 game starts of both arms' collectors, drawn directly | the same start kind, archive state, mirroring, game length and chance seed in both arms |
| R6 | inputs | om: belief features vary, are posteriors, reset at every game start; the previous-action field shows game starts and actions. sp: exactly the 90-entry observation | all hold in the smoke rollouts and in a direct collector probe |
| R7 | opponents and shares | the matchup sampler simulated for a campaign's number of games; the smoke and benchmark histories | all 15 fixed opponents drawn with near-uniform frequency; the fixed-opponent game share and learner-transition share measured and reported (no target value for the transition share) |
| R8 | boundaries and information | unit checks (also in `tests/test_m5.py`): the truncation bootstrap reads the post-step belief; the current turn's opponent code never reaches the current input; an `OMPolicy` and OM2's `ArmPlayer` play identically; a widened network in the om path equals the 90-input network in a battery job | all hold (the last within 10⁻⁵ on values, identical actions) |
| R9 | concurrency | short runs of the campaign mix at several worker counts (at least 4, 5, 6 and one above 6), steady-state throughput per run and in aggregate, memory; calibrated against the historical full-length runs of the sp format | a recommended worker count and a measured wall-clock estimate |
| R10 | instruments at reduced samples | each M5b–M5d instrument end to end at a few states or games, on stand-ins: s0 families and head to head, the battery, the M4.6 cells, the lane-exit readout from recorded om states, attacker training and evaluation | runs end to end; cost per unit measured; reduced-sample agreement with published values reported as weak evidence only |
| R11 | regression suite | `python -m pytest` | passes; skips and their reasons listed |

**On failure.** A failed check is investigated with targeted probes. An implementation defect is fixed within scope and
the check re-run; both runs are reported. A design limitation is reported with a recommendation and left for review. No
threshold is changed after its result is seen.
