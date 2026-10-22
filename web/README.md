# The laning simulator's web frontend

The visitor-facing application of the M6 demo: **Play** against a trained M5 agent and **Watch** two agents play, in
the glass design approved in M6a ([decision record](../docs/m6/m6a_decision_record.md) §7). React with TypeScript,
built by Vite, served by the Python API (`python -m lanegame.demo`). The browser computes no game rule: every number
it shows comes from the server's frames.

## Run it

From the repository root, with the venv installed with the `demo` extra (`pip install -e ".[demo]"`) and the 7
checkpoint files of `lanegame/demo/manifest.json` on disk:

```bash
cd web && npm install && npm run build && cd ..     # once, and after a frontend change (Node 20 or later)
python -m lanegame.demo                              # http://127.0.0.1:8765/
```

The server loads the engine (about 3 s), verifies the three seed-0 research replays, starts the simulation worker
process and serves the built frontend from `web/dist/`. Without the checkpoints the page still loads and says that the
engine is unavailable.

## Develop

```bash
python -m lanegame.demo            # the API on 8765 (keep it running)
cd web && npm run dev              # Vite on http://127.0.0.1:5173/, proxying /api to 8765, with hot reload
npm run typecheck                  # tsc --noEmit
npm test                           # vitest: the history rules and the stale-response guard (src/game/*.test.ts)
npm run build                      # type-check, then dist/
```

**The API types are generated.** `web/openapi.json` is the FastAPI app's schema and `src/api/schema.d.ts` the
TypeScript types made from it. After a change to `lanegame/demo/schemas.py` or the routes:

```bash
python -m lanegame.demo --openapi web/openapi.json   # from the repository root
cd web && npm run types
```

`tests/test_demo_api.py` fails when the committed schema no longer matches the app.

## Layout

| path | content |
|---|---|
| `src/api/client.ts`, `src/api/schema.d.ts` | the typed fetch client and the generated types |
| `src/game/types.ts`, `frames.ts`, `words.ts` | type aliases, the sub-frames of a turn, and every word on screen |
| `src/game/controller.ts` | the history controller: progress and the viewed position kept apart, playback, stepped ticks |
| `src/game/store.ts` | the application state and actions: sessions, submissions with reconciliation, reload recovery, the stale-response guard, layout and theme |
| `src/scene/lane.ts` | the top-down lane (imperative SVG), fitted around the glass surface |
| `src/ui/*.tsx` | the top bar, the surface and its zones, the controls, the graph, Details, the dialogs |
| `src/styles.css` | the glass design's stylesheet |

The browser verification (`python -m lanegame.experiments.m6b browser`) drives the built app through real clicks
and the app's `window.__demo` hooks; its evidence is in `results/m6/m6b_browser.json` and `results/m6/m6b/`.
