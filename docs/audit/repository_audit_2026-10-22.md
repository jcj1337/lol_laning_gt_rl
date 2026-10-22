# Repository audit, 2026-10-22

*A read-only pass over the whole repository at commit 87e9abd0 plus the uncommitted M6c step 1 in the working tree.
Nothing was changed during the audit except four lines in the README and the web README that the audit itself showed
to be wrong (the Node version the toolchain needs, the viewer's status, the test duration) and one packaging line
(`guides/*.json`), all part of the M6c work under review. The historical reports were not edited; corrections they
need are listed for the living summaries and the reference's §12.*

## 1. How the audit was done

Seven read-only passes, one per slice, each reading its files in full and running probes, recomputations or test
subsets, followed by two deep dives on the findings that could change a reported research verdict or misrepresent
the simulator. The slices were the wave engine and the exact game, the champion game and the environment, the RL
stack and the belief, the M5 campaign's statistics and provenance, the demo backend, the committed frontend, and the
documentation, packaging and test suite. Every finding below carries the evidence the pass produced; anything that
could not be supported by a quote, a probe or a recomputation is marked as a suspicion.

Tools used, all without writing to the repository: random-walk invariant probes on the wave engine (48,000 turns),
an RNG-counting wrapper around the environment (9,000 steps), legality cross-checks over 18,000 seat-turns, a
hand-checked GAE trajectory, a batched-versus-sequential belief probe, recomputation of every M5 endpoint, p-value
and attacker bound from the stored JSON, a TestClient against the demo with 59 malformed submissions and 30 static
path probes, a headless-browser probe of the keyboard behind a modal dialog, a scan of the three verified replays
for the narration cases the frontend pass named, and the full test suite (1,595 passed, 4 skipped, 2 expected
failures, 1 failed, in 9.4 minutes). This is not an exhaustive verification; §6 lists what was not covered.

## 2. The headline

No defect was found that changes a stored research result. The registered M5 endpoints, their t intervals and
classes, the 15 one-sided p-values and their Holm rejections, the attacker lower bounds and the S8 verdicts recompute
from the stored JSON to the reported values. The environment draws exactly ten chance uniforms per turn whatever the
actions; four implementations of legality agree; the observation is 90 entries in [0, 1] with no information leak;
the reward is the gold difference and nothing else; the reference's numbers match the code; the exact solver's
values, best responses and reachable layers verify against brute force at K = 16; the stored pins are literal
digests, not recomputed ones.

What the audit did find, in order of practical weight:

1. **The demo states two simulator facts the engine did not say.** Every damage line in the event texts quotes the
   damage dealt before the victim's durability, not the HP lost, so after an investment the text contradicts the HP
   bar (56 of the 62 damage turns in the seed-0 head to head). The frontend's tick narration places a wave fight "at
   your base" whenever the two fronts are in adjacent cells (150 of the 900 ticks in the three verified replays) and
   says a wave "walks" when it was cleared and the next wave took its place (79 of 597 such sentences). Both
   contradict the documented rule that the browser computes no game rule.
2. **Provenance.** All 212 result files that record their tree state record a dirty one, and the M5 code digest
   matches neither candidate commit. The numbers themselves reproduce from the records at HEAD, and every network
   file hashes to its inventory entry, but no commit can be named that produced them.
3. **One registered verdict rests on a different reading in each arm.** "Concession under threat does not replicate
   in either arm" is correct under the registered rule, but for pure self-play it comes from the two seeds whose
   family was re-read at 150 turns; at the pre-registered 50-turn statistic that arm replicates in six of six. The
   closeout report and the M6 wording table omit the qualifier.
4. **A claim in the closeout and the M6 wording table misattributes a number.** "+6.4 against held-out scripted
   variants" is the mean over the whole held-out family; the four scripted variants are +3.7 to +14.4 (mean +11.4),
   and the three conditional styles −0.2 to −0.7 pull the family mean down.
5. **The demo is stallable once the host is widened** (a chunked body bypasses the 16 KB limit, an unlimited history
   call holds the single simulation lock, the session cap evicts active games, and one slow Watch request disables
   Watch for the rest of the process), and one API test fails by chance about one run in three.

## 3. Findings

Severity: **high** changes what a reader or visitor is told about the simulator or the research, or weakens the
chain of evidence; **medium** breaks behaviour under a reachable condition or misleads in a narrower way; **low** is
cosmetic, dormant or already documented. Confidence names how the finding was established. Effort is S (under an
hour), M (a session) or L.

### 3.1 Implementation defects

| id | sev | finding | location | evidence | impact | conf. | fix (effort) |
|---|---|---|---|---|---|---|---|
| D1 | high | Outcome texts quote pre-durability damage as HP lost | `lanegame/demo/frames.py:163-178`; `champions_m4.py:728-729` | The resolver sums trade, tower, gank and minion damage and divides the sum by the victim's M(S); the texts print the pre-division amounts. In the h2h replay 56 of 62 damage turns differ from `delta.hp_taken`: turn 35 says "tower shot A for 141 HP", A lost 108; turn 50 says "A hit B in a trade for 420 HP", B lost 303 | After any investment the history and Details overstate HP removed by up to 2.5×. The first guided replay is unaffected: its notes quote HP lost and tower damage only (checked for both versions of it), but a guide author who copies `data.damage` after an investment states a wrong fact | high (probe) | Word the texts as damage and add the HP lost from `delta.hp_taken`, or print both; add a test over a replay with investments (S) |
| D2 | high | Tick narration puts adjacent-cell fights "at a base" | `web/src/game/words.ts:118-121, 136-138`; `viewer/session.py:444-449` | `contact_of` returns the midpoint of the two fronts, a half-integer for adjacent cells; `cellName` maps any non-integer to "{seat}'s base". Fronts at −1 and 0 → "fight at your base". 150 of 900 ticks in the three replays have a half-integer contact | One stepped tick in six tells the visitor the waves fight at a base, where waves never fight; the lane drawing places the marker correctly, so the picture and the words disagree | high (code read, counted on the replays) | Handle half-integers in `cellName` ("between the lane and your side") and pin the narration to the tick table in a test (S) |
| D3 | medium | "X's wave walks to …" when the front was cleared | `web/src/game/words.ts:125, 140-143`; `mechanics.py:741-757` | With no contact the client compares the previous and current front's cell and says "walks" unless a crash happened; a front popped by champion damage makes the next wave the front. 79 of 597 "walks" sentences in the three replays come with a smaller packet count | A cleared wave is narrated as a retreat; Details' tick table shows the truth | medium (code read; the count is an upper bound of the trigger) | Say "walks" only when the packet count is unchanged, else that the front wave is gone (S) |
| D4 | medium | The 16 KB body limit checks only the Content-Length header | `lanegame/demo/api.py:192-198` | A 655,360-byte chunked body without the header was accepted (200, session created) | On 127.0.0.1 nothing; with `--host 0.0.0.0` a client can make the process read arbitrary bodies into memory. The plan §8.2 and the M6b report state a limit that chunked requests bypass | high (probe) | Count bytes in the ASGI receive and abort over the limit (S) |
| D5 | medium | The simulation pool latches "unavailable" on any exception and never recovers | `lanegame/demo/api.py:115-129` | With a 0.5 s timeout one run gave 503 "the simulation worker failed: " (an empty `TimeoutError` message); later runs with a 120 s timeout stayed 503; the timed-out future is not cancelled | One slow Watch request disables Watch for the process's life, and status shows no reason | high (probe) | Latch only on `BrokenProcessPool`; on a timeout cancel the future, keep "ready", return 503 with the exception's name (S) |
| D6 | medium | A failed stop at the turn cap is swallowed | `web/src/game/store.ts:323, 372-379` (HEAD) | `finish()` has no catch; a timeout or dropped request at the cap leaves `conclude` unrun, no recording, no retry and, in hidden mode, a belief that never appears until a reload | At the one moment the record and belief become public, a transient failure leaves the page stuck | high (code path) | Wrap the call in try/catch and route it through `failed(e, retry)` (S) |
| D7 | medium | One API test depends on a chance seed | `tests/test_demo_api.py:171-179`; `session.py:168` | The session is created without a seed; the test asserts a Recall happened by turn 3. Failed 2 of 6 isolated runs here, 2 of 3 in another pass, and once in the full suite | `python -m pytest` is red about one run in three for a reason unrelated to the code, so a real regression in the demo would be indistinguishable from noise | high (reproduced) | Pass a fixed `seed` in the POST, or Recall at the first turn where code 12 is legal (S) |
| D8 | low | The guide files were not package data | `pyproject.toml:27` | `package-data` listed `manifest.json` only; a non-editable install would ship no `guides/` and the first `/api/guides` would raise | Fixed in this pass (uncommitted); M6e's clean-install check should use a non-editable install | medium (read, not installed) | Done; add the non-editable install to M6e (S) |
| D9 | low | `GameConfig.validate` has range gaps and is not run at construction | `lanegame/config.py:974-1019`; called only in `game.py:163` and `env/core.py:123` | `wave_hp=0`, `horizon=0`, negative damage and loss all validate; `step_turn` ran on a combination `validate` rejects. All 37 other inconsistent combinations tried were refused; every preset validates, is frozen and matches its `*_FIELDS` tuple | Direct users of `mechanics`, `lane_enum` and `scenarios` can run an inconsistent config silently; no preset is affected | high (probe) | Validate in `__post_init__` and add the range asserts (S) |
| D10 | low | The LP's gap certificate is recorded but never enforced | `lanegame/matrix_game.py:124-127`; `solver.py:113` | The column-LP fallback returns `(lo + up) / 2` whatever the gap; over 3,000 random games the worst relative gap was 3.5e-9 (absolute gaps up to 3.5e-6 at scale 1e3) with no exception; `tol` is absolute | None on stored results (recorded max gaps 4.4e-15 at K = 32); a future payoff scale could silently return an uncertified value | high (probe) | Raise when `gap > tol` and make `tol` relative to the matrix scale (S) |
| D11 | low | `target_kl` early stop reads one minibatch | `lanegame/rl/ppo.py:202-203` | `stats["approx_kl"][-1]` is the last minibatch, not the epoch mean | Dormant: `target_kl` is None in every campaign | high (code) | Average the epoch's KLs (S) |
| D12 | low | `freeze_maintained` reports Thin on the four-action presets | `champions_m4.py:642-643`; `env/trace.py:140-141`; `env/agents_m4.py:191` | Slot 2 is Thin from `m4_four_actions()` on, so the flag is true for a Thin near the own tower; three Thin turns from s0 give `freeze_maintained == [3, 0]` | A reader of a version-2 trace or of `AdaptiveBreaker` (used only by the `m4()`-era experiments) reads "freeze" for a Thin; no M4.5+ result changes | high (probe) | Gate the flag on `cfg.freeze_maintenance` or rename the counters (S) |
| D13 | low | Three small frontend gaps | `web/src/game/words.ts:91-110`; `ui/Surface.tsx:23-25`; `api/client.ts:32-38` (HEAD) | The history row drops the major "return" outcome; the turn label says "Stopped after turn 150" while the capped turn is still being stepped; a 200 with an unparsable body becomes `null` and throws outside the handled path | Minor inconsistencies between the three event views; a rare client one turn behind until the next reconciliation | high (code) | Add the `return` case; skip the label override while stepping; throw `ApiError(0, 'bad_body')` (S) |

Not reproduced: the frontend pass suspected that Enter or an arrow key pressed while a modal dialog is open reaches
the game behind it (`App.tsx:60-65`). In headless Chromium a click on the dialog's text makes the dialog itself the
active element, the handler's guard holds, and neither Enter nor an arrow changed the game. Low; worth one check in
Firefox and Safari.

### 3.2 Research-validity risks

| id | sev | finding | location | evidence | impact | conf. | fix (effort) |
|---|---|---|---|---|---|---|---|
| R1 | high | Every stored result comes from a dirty tree, and the M5 code digest matches no commit | `results/m5/inventory.json` (`code`), `reproducibility.md:6`; `experiments/rl/m5/common.py:228-247` | 212 result files record `working_tree_dirty: true`, none `false`. The recorded digest 7ebc49ca… over 390 files at 312d118d (dirty) differs from the digest recomputed over `git archive` of 312d118d (388 files) and of a07e3fdf (390 files), with LF and CRLF normalization both tried. `code_digest` hashes raw bytes and the checkout uses `autocrlf`, so the digest is also checkout-dependent. Mitigation verified: at HEAD `analysis.analyse(save=False)` reproduces `analysis_main.json` with zero numeric differences and `strategic.assemble` reproduces `strategic_main.json`; all 429 inventory networks and the 7 demo checkpoints hash to their recorded SHA-256; the decision log records code edits after `reproduce` ran | No commit reproduces the published numbers' code state bit for bit. The README's findings rest on these files. The M6d row of the roadmap already plans a `reproduce` re-run on the committed tree | high (recomputed) | Re-run `reproduce` on a clean checkout and commit it; make `code_digest` normalize line endings; add `.gitattributes`; have `reproduce` refuse or warn loudly on a dirty tree (M) |
| R2 | medium | The S8 verdict "concession under threat does not replicate" rests on a different reading in each arm | `experiments/rl/m5/strategic.py:134-136`; `docs/m5/m5e_closeout_report.md:144-146`; `m5c_strategic_report.md:138`; `docs/m6/m6_plan.md` §9 | Recomputed from the raw battery records: sp seeds 4 and 5 hold at the 50-turn statistic (+0.058, +0.045) and fail at 150 turns (+0.212, +0.329); both failed horizon validation, so the registered exception 3 reads the 150-turn value; the other four sp seeds hold at 50 turns. The om failures (seeds 0, 3, 4) are at 50 turns with validation passed. The 150-versus-300 diagnostic passes for both sp seeds | The statement is correct under the registered rule, but for pure self-play it is a horizon finding, not a 50-turn one; the M5c report says "read at 150 turns after a re-run" without saying the 50-turn reading holds, and the closeout and the M6 wording table omit the qualifier. The deep dive on all 51 re-run pairs is recorded in §4 | high (recomputed) | One sentence in the living summaries and the M6 wording: "pure self-play fails only in the two seeds whose zoning family was re-read at 150 turns; at 50 turns it holds in all six"; report both readings in `strategic_main.json` (S) |
| R3 | medium | The pre-registration's amendment chain has no anchor outside its own file | `experiments/rl/m5/common.py:309-323, 336-359, 362-376` | On a scratch copy, editing "δ = 0.25" to "0.50" is refused, but deleting `preregistration.json` and re-running `write_prereg` on the edited document yields a version-0 record that `require_prereg` accepts. `verify_chain` checks internal consistency only. The only anchor is git: the document's SHA has been 21e35e07… at every commit since 384eea13 | `prereg_check` proves the document matches a chain, not that the chain is the one frozen before training; a silent re-freeze would show only in git history | high on the mechanism, low on practical risk | Pin the original record's SHA in a test, refuse `write_prereg` when archives exist, print the introducing commit (M) |
| R4 | low | The om arm's evaluation reconstructs its training inputs within 1e-5, not bit for bit | `om2/collector.py:77-87` (batched), `m5/players.py:172-176` (sequential), `readiness.py:29` | Over 40 turns the batched and sequential log-likelihoods differ by at most 3.9e-6 (a single row differs from a batched row by 6.7e-7); the posterior difference after 38 updates was 9e-26. The default hooks are bit-identical (pin reproduced). A CRN inverse-CDF draw within 1e-6 of a boundary could flip an action | The instruments present the evaluation games as the trained behaviour; the reconstruction is within the documented tolerance, and no flip was seen in 120 turns tested. The docstring's "identical to training" is slightly strong | high on the tolerance; the flip rate is untested (low) | State the tolerance in `players.py`; optionally count flips over the M5 evaluation games (S) |
| R5 | low | The resolver accepts illegal codes; about 50 instrument call sites bypass the environment's check | `champions_m4.py:470-474, 493, 738, 783-795`; `mechanics.py:841`; callers in `lane_exit.py:202`, `m5/games.py:132`, `battery.py:62`, `attackers.py:156`, `m45c/horizon.py:180` | Ward from the own tower cell resolves with a ward placed; Absent while present sets `absent = −1`; Pressure without a siege raises only a bare `assert`. All 60,080 legal pairs on 240 reachable states resolved cleanly; the env rejects every illegal code (9,000 random steps) | A masking bug in a policy would corrupt an instrument rollout silently instead of raising; no evidence any did | high (probe) | An opt-in `check_legal` in `resolve_turn_m4`, turned on in the four rollout loops (S) |
| R6 | low | Research-replay verification needs bit-identical float32 network outputs across machines | `lanegame/demo/session.py:535-540`; `service.py:214`; `api.py:151-154`; `tests/test_demo_api.py:258` (`== 0.0`) vs `test_demo_session.py:158` (`<= 1e-5`) | Sampled actions come from `searchsorted(cumsum(p), u)` over float32 forward passes; a mismatch is a startup failure. On this machine everything reproduces with difference 0.0 | On another CPU or torch build a single action within float noise of a boundary would stop the server from starting; the two tests disagree on the tolerance they call exact | low (cannot be probed here) | Skip a non-reproducing replay and report it in `/api/status` instead of aborting; align the tolerances (S) |
| R7 | low | The Phase 1 Monte Carlo consumes random draws action-dependently on v1 and v2 | `lanegame/simulate.py:104-107` | A choice menu draws a uniform only when it has more than one option, so later gank draws shift with the actions; the siege default has single-option menus | `python -m lanegame simulate` is seed-reproducible, as its docstring claims, but not comparable across policies on common random numbers; the CLAUDE.md contract concerns the env, which draws a fixed count | high (code) | Draw the choice uniforms unconditionally, or state the limit (S) |

### 3.3 Documentation drift

| id | sev | finding | location | evidence | impact | conf. | fix (effort) |
|---|---|---|---|---|---|---|---|
| W1 | high | "+6.4 against held-out scripted variants" is the whole held-out family's mean | `docs/m5/m5e_closeout_report.md:110-112`; `docs/m6/m6_plan.md` §9 table | `per_opponent.S_out` in `analysis_main.json`, om − sp at 114: the four scripted variants +14.33, +14.42, +13.12, +3.70 (mean +11.39); the three conditional styles −0.18, −0.67, −0.20; the seven-opponent mean +6.36. The M5b report has it right (+13.1 to +14.4, g_slow_guard +3.7) | A reader concludes the gain roughly halves on unseen scripted styles; three of four are exploited as hard as the library archetypes. The M6 wording table would carry it into the demo's text | high (recomputed) | Reword in the living summaries and the plan: "+6.4 against the held-out family (four scripted variants +3.7 to +14.4, three conditional styles −0.2 to −0.7)"; the historical report gets a §12 correction (S) |
| W2 | medium | "The browser computes no game rule" is stated as absolute | `CLAUDE.md`, `docs/m6/m6_plan.md` §4.2, `web/README.md`, `m6b_report.md` §1 | The client derives the tick narration from frame differences (D2, D3) and the "even" label from a 0.005 threshold | The guarantee a reader relies on is narrower than stated; the narration is where the demo states simulator facts the server did not say | high | After D2 and D3, reword to "computes no rule and narrates only frame differences, pinned by tests" (S) |
| W3 | medium | The About dialog softens and under-specifies the registered numbers | `web/src/ui/Dialogs.tsx:107-110` | "slightly worse: about 0.4 to 0.6 gold" for P2a −0.63 [−0.98, −0.27] and P2b −0.44 [−0.72, −0.16], registered as a material loss in every seed; no seed count or interval; "can beat both" without "lower bound"; no mention that pure self-play had not converged. "+9 gold" and "half of its training games" match | A visitor takes the competence cost as marginal; the plan's §9 rules ask for the interval, the seed count and the limitations | medium (wording against §9) | "worse in every seed, by about 0.4 to 0.6 gold (six seed pairs; intervals in the report)", add "lower bounds" and the not-converged caveat (S) |
| W4 | medium | "Node 20 or later" is wrong for the pinned toolchain | `README.md`, `web/README.md`, `web/package.json` (no `engines`) | vite 7.3.7 and the React plugin need `^20.19.0 || >=22.12.0`; vitest 5.0.5 needs `^22.12.0 || ^24.0.0 || >=26.0.0` | `npm run build` fails on Node 20.0 to 20.18, `npm test` below 22.12 | high (engine fields read) | Corrected to "Node 22.12 or later" in both READMEs in this pass; add `engines` to `package.json` (S) |
| W5 | low | The viewer's status is told differently in different places | `README.md` (this pass), `CLAUDE.md:44, 122`, roadmap M6 plan table, `verify-lanegame/SKILL.md:13` | The README called the viewer legacy while the plan says the archive is M6d work, not done | A reader is told the move happened | high | The README now says "due to be archived"; the move itself is M6d (S) |
| W6 | low | Stale numbers and names in code docstrings and reports | `env/discounted.py:17-20` (Ω = 68.3, bounds at W = 460 and 1000); `rl/ppo.py:84-85` ("provisional"); `champions_m4.py:17, 29`; `env/core.py:34`; `mechanics.py:4, 110-111`; `env/observation_m4.py:27`; `champions.py:118`; `rl/agent.py:11-13` (the digest covers the weights, not the file; the dtype is not hashed); `m5/players.py:185`, `m45c/horizon.py:135` (`/ 0.1` hard-coded); `docs/phase1/phase1_implementation_report.md:500` (`max_states`); `docs/m6/m6a_technical_report.md:141-148` (request fields); `m5/analysis.py:4-6` (S6 is unpaired; the SE is right); `m5e_closeout_report.md:87-88, 146-147` and `m5c_strategic_report.md:144` ("within 0.03": one seed is 0.034 away; "+0.25 includes 0": C_cross does not); `README.md:38` ("three times": 3.6×); `CLAUDE.md:36-37` (four drivers' docstrings list no stages: m45b, rl/m43b, rl/m46a, rl/m46c); `CLAUDE.md:204` (`legacy/` also holds the M6a prototype); roadmap line 6 ("L–BA"); `requirements.txt` (no torch, no pins, last changed 2026-10-10); README typos ("youre", "humanm", "advantges", "researcg train sagainst") | Each is a wrong detail a reader of the code or the page takes as current; the reference's §12 already lists six of the docstrings | high | One docstring pass; dated corrections in the living summaries and §12; typo fixes in the README (S) |

### 3.4 Intentional design limitations (documented or accepted; listed so they are not mistaken for defects)

| id | finding | location | note |
|---|---|---|---|
| L1 | The plain checkpoint loader unpickles before any check | `rl/agent.py:117, 129`; CLI `--opponents`, `--init`, `--checkpoint` | Accepted for a local research tool; the demo refuses any file whose SHA-256 is not in its manifest before unpickling. The docstring should say the digest covers the weights (W6) |
| L2 | The demo's limits are sized for localhost | `api.py:59, 208, 249-252`; `service.py:163-164` | `/history` is unlimited and re-simulates under the single lock (124 ms per call on a 150-turn session); the session cap evicts active games and one address can fill it in about eight minutes; all clients behind a proxy share one bucket; `--max-sessions 0` pops the session it just created. M6e's scope |
| L3 | CRN policies cannot decide past their pre-drawn horizon | `m5/players.py:122, 165` | Every caller passes the loop length (the demo passes its turn cap); a longer loop gets a bare `IndexError` |
| L4 | Tick frames show the start-of-turn HP and status | `web/src/game/frames.ts:52`; plan §5.2 | By design (turn-level events settle at the end); the viewer is not told that the mid-turn HP is the pre-turn value |
| L5 | `window.__demo` exposes the store in production | `web/src/App.tsx:73-109` | The verification hooks; the client holds no secret and the server validates every code |
| L6 | Tests that prove little, and tests that skip in a clone | 13 assert-free soak tests; `test_demo_session.py:161` (asserts a hash against the same function); about 40 checkpoint-keyed skips; 178 of 420 modules imported by no test (mostly report generators) | In a fresh clone the demo and M5 instruments are not exercised and the suite still passes; the soaks rely on the env's internal invariant checks |
| L7 | Repository hygiene | 2,501 tracked files; `results/` 345 MB over 1,854 files including 170 logs and a 68 MB JSON; working tree 1,083 CRLF, 485 LF, 19 mixed; no `.gitattributes`; 19 temporary objects in `.git` | Noisy diffs, a checkout-dependent code digest (R1), slow clones |
| L8 | The farm-spot approximation | reference §5.2 (a 28 % per-tick front mismatch against the authoritative step) | A documented limitation of the positioning model; its effect on the states PPO visits was not measured (§6) |

## 4. Deep dives

**The 50-turn versus 150-turn readings (R2).** For each of the 51 (final, family) pairs that failed horizon
validation and were re-run at 150 turns, the registered S8 statistic was recomputed from the raw battery records at
both readings, exactly as `strategic.family_gap` computes it (the critic-bootstrapped statistic for natural
families, the pure rollout for constructed ones). The stored 150-turn gaps reproduce for all 51, so the registered
reading was applied as written. Five pairs change their per-seed status between the readings: pure self-play seed 3
on early wave management (fails at 50, holds at 150) and on early trading (holds at 50, fails at 150), pure
self-play seeds 4 and 5 on constructed zoning (hold at 50: +0.058 and +0.045; fail at 150: +0.212 and +0.329), and
the modelling arm's seed 5 on constructed trading (holds at 50, fails at 150). The other 46 keep their status, four
of them failing under both. The 150-versus-300 diagnostic passes for all five differing pairs and for all 51, so
the registered reading is the better-supported one in every case. Across the registered verdicts only one
changes: pure self-play's concession under threat replicates in six of six seeds at 50 turns and in four of six
under the registered reading, because of seeds 4 and 5 alone; the modelling arm's concession verdict fails under
both readings. No family outside the registered verdicts changes an arm-level status that a verdict depends on.
Conclusion: the shared statement "concession under threat does not replicate in either arm" is a legitimate
registered outcome, and the write-ups should say that the two arms fail on different evidence. No stored number
needs to change. A side observation: for the four constructed families the 300-turn validation records already
held the turn-150 values the re-run recomputed (same rollout seeds), so that part of the 5.6-hour re-run duplicated
data on disk.

**Tick narration (D2, D3).** A scan of the three verified replays (900 ticks) counted 150 half-integer contacts,
which the client narrates as a fight at a base, and 79 "walks" sentences where the packet count fell, which means
the front was cleared rather than moved. The counts are upper bounds on the visible mistakes, since a visitor sees
only the ticks they step through.

**The keyboard behind a modal (not reproduced).** With an action selected in a live game, opening About, clicking its
text and pressing Enter and an arrow changed nothing: the dialog became the active element and the handler's guard
held. The finding stays low and browser-specific.

## 5. Ordered remediation plan (for review)

1. **Make the demo say only what the engine said.** Fix the damage texts (D1) and the two narration rules (D2, D3),
   with tests that pin the words to the tick table and to `delta.hp_taken`; then narrow the "computes no game rule"
   claim (W2). Small, and it removes the only places the demo misstates the simulator.
2. **Correct the research wording in the living documents.** The +6.4 attribution (W1), the concession qualifier (R2)
   with both readings in `strategic_main.json`, the About dialog (W3), and the rounding-level drifts (W6), as dated
   corrections in the roadmap summary, the M6 plan §9 and the reference's §12. The historical reports stay as they
   are.
3. **Close the provenance gap.** Re-run `reproduce` on a clean checkout with a line-ending-normalized code digest,
   add `.gitattributes`, make `reproduce` warn on a dirty tree, and pin the pre-registration root's hash (R1, R3).
   This is M6d's planned work; it should happen before any bundle is published.
4. **Stabilize the suite.** Seed the chance-dependent API test (D7), give the soak tests an assertion (L6), and
   rerun `tests/test_demo_*` several times once the M6c review settles.
5. **Harden the demo before the host is widened** (M6e). The chunked-body limit (D4), pool recovery (D5), the
   failed-stop retry (D6), a history rate limit and a per-address session cap (L2), and a non-aborting replay check
   (R6).
6. **One pass of small code fixes.** Validation at construction (D9), the LP gap (D10), the KL mean (D11), the freeze
   flag (D12), the three frontend gaps (D13), the opt-in legality check in the instruments (R5), the CRN horizon
   error (L3), the `engines` field and the docstrings (W4 to W6).
7. **Coverage gaps worth one pass each** (§6): the minion-targeting branch, the A_max enumeration, the farm-spot
   mismatch on PPO-visited states, a non-editable install and a Node 20 build in M6e.

## 6. Coverage gaps

The audit did not inspect, or inspected only through existing pins:

- the M4.5a minion-targeting branch of `_tick_natural` (`_designation`) against `positional.py`'s coordinate
  conventions, covered only by the m45a and m45b digest pins;
- the enumeration that produced `A_max = 408` for the tail bound (the arithmetic of Ω was checked, not the bound's
  input);
- the farm-spot approximation's end-cell mismatch on the states PPO visits (the reference reports 28 % per tick on
  scripted play);
- the internals of `m5/cells.py`, `bench.py`, `instruments.py`, `readiness.py`, the battery's constructed-state
  builders, `m45c`'s and `m46a`'s engine ledgers, `inventory_files.json.gz`, and the `om1` package;
- the working-tree frontend beyond the guide feature (the frontend was audited at HEAD because it was being edited);
- cross-machine float reproducibility of the research replays (R6) and the flip rate of the om arm's reconstruction
  (R4);
- a non-editable `pip install .`, a Node 20 build, the "Train an agent" README snippet and `solve --horizon 32`;
- the Streamlit viewer's UI (`viewer/app.py`, `render.py`), 178 experiment and report modules that no test imports,
  and h11-level handling of a wrong Content-Length (the ASGI test client ignores it);
- the demo under real concurrency beyond eight simultaneous submissions, and any browser other than headless Edge.
