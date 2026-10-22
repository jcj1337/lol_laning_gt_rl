// The application: the scene, the one glass surface with the top bar, the Details drawer and the dialogs. The store
// owns the state (src/game/store.ts); this component wires the window (keys, resize, the system theme) and exposes
// the hooks the browser verification drives (window.__demo).
import { Component, useEffect, useState } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Details } from './ui/Details';
import { Dialogs } from './ui/Dialogs';
import { LaneView } from './ui/LaneView';
import { Surface, turnLabel } from './ui/Surface';
import { TopBar } from './ui/TopBar';
import { StoreContext, useStore } from './ui/useStore';
import { AppStore } from './game/store';
import type { Selection } from './game/store';
import type { ModelId } from './game/types';

declare global {
  interface Window { __demo?: Record<string, unknown>; __errors?: string[] }
}

/** A rendering error never leaves a blank page: the message and a way to reload (the server still holds the game). */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { window.__errors?.push(`render: ${error.message} ${info.componentStack ?? ''}`); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="problem err" role="alert" style={{ position: 'fixed', inset: '20% 20% auto 20%', zIndex: 20 }}>
        <p><b>The page hit an error.</b> {this.state.error.message}</p>
        <p className="note-text">The game is still on the server. Reload the page to continue it.</p>
        <div className="row-end"><button className="pillbtn gold" onClick={() => location.reload()}>Reload</button></div>
      </div>
    );
  }
}

function Root() {
  const S = useStore();
  return (
    <>
      <LaneView turnLabel={turnLabel(S)} />
      <div className="hud" id="hud">
        {S.arr !== 'rail' && <TopBar />}
        <Surface />
      </div>
      <Details />
      <Dialogs />
    </>
  );
}

export function App() {
  const [store] = useState(() => new AppStore());
  useEffect(() => {
    window.__errors = window.__errors ?? [];
    const onErr = (e: ErrorEvent) => window.__errors!.push(String(e.message || e));
    const onRej = (e: PromiseRejectionEvent) => window.__errors!.push(String((e.reason && e.reason.message) || e.reason));
    addEventListener('error', onErr);
    addEventListener('unhandledrejection', onRej);
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t?.closest('input, select, textarea, dialog[open]') || e.altKey || e.ctrlKey || e.metaKey) return;
      if ((e.key === ' ' || e.key === 'Enter') && t?.closest('button, a, summary, [role="button"]')) return;   // the control itself
      if (store.key(e)) e.preventDefault();
    };
    addEventListener('keydown', onKey);
    let rz: ReturnType<typeof setTimeout>;
    const onResize = () => { clearTimeout(rz); rz = setTimeout(() => { store.arrange(); store.refitLane(); }, 120); };
    addEventListener('resize', onResize);
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const onScheme = () => { if (store.theme === 'auto') store.applyTheme('auto'); };
    mq.addEventListener('change', onScheme);
    window.__demo = {
      store,
      newGame: (o: ModelId = 'sp:0', b: 'hidden' | 'shown' = 'hidden', seed: number | null = null) => store.newGame(o, b, seed),
      submit: (code: number) => store.submit(code),
      quick: async (code: number) => {                        // play a turn and finish showing it at once
        const p = store.submit(code);
        await new Promise((r) => setTimeout(r, 30));
        const c = store.ctrl;
        if (c?.step) { c.skip(); c.advance(); }
        await p;
      },
      step: () => store.ctrl?.advance(), skip: () => store.ctrl?.skip(),
      stepTurn: (t: number) => store.ctrl?.stepTurn(t), ticks: (m: string) => store.setTicksAuto(m === 'auto'),
      select: (sel: Partial<Selection>) => store.setSelection(sel), continueAway: () => store.continueAway(), stop: () => store.stopGame(),
      goto: (p: number) => store.ctrl?.goto(p), watch: (n: string, v = 0) => store.watch(n, v), simulate: (a: ModelId, b: ModelId) => store.simulate(a, b),
      hover: (p: number | null) => store.setHover(p), details: (s: string | null) => store.openDetails(s),
      history: (open: boolean) => store.setHistory(open), graph: (big: boolean) => store.setGraphBig(big),
      layout: (name: string) => store.setLayout(name), theme: (name: 'auto' | 'light' | 'dark') => store.applyTheme(name),
      download: () => store.downloadRecord(),
      boxes: () => (store.lane && store.ctrl ? store.lane.objectBoxes(store.ctrl.frame) : {}),
      state: () => {
        const c = store.ctrl;
        return {
          layout: store.layout, arrangement: store.arr, phone: store.phone, layoutChoice: store.layoutChoice, theme: store.theme, dark: store.isDark(),
          lane: store.lane ? { scale: +store.lane.a.toFixed(3), angle: Math.round((store.lane.t * 180) / Math.PI) } : null,
          mode: store.mode, progress: c?.progress ?? null, view: c?.view ?? null, stopped: c?.stopped ?? null, live: !!store.live, sid: store.live?.sid ?? null,
          turns: store.game?.turns.map((t) => t.codes) ?? [], menu: c?.menu() ?? null, error: store.error, problem: store.problem ? { kind: store.problem.kind, message: store.problem.message } : null,
          busy: store.busy, loading: store.loading, canSubmit: c?.canSubmit ?? false, ticksAuto: store.ticksAuto, playing: c?.playing ?? false,
          recovered: store.recovered, engine: store.engine ? { ready: store.engine.ready, simulations: store.engine.simulations } : null,
          replays: store.replays.map((r) => r.name), kind: store.game?.meta.provenance.kind ?? null, name: store.game?.name ?? null,
          seeds: store.game?.meta.seeds ?? null, marks: store.game?.meta.marks ?? null, beliefs: !!store.game?.beliefs, speed: c?.speed ?? null,
          histOpen: store.histOpen, graphBig: store.graphBig, detailsOpen: store.detailsOpen, dialog: store.dialog,
          step: c?.step ? { t: c.step.t, i: c.step.i, n: c.step.frames.length, sub: c.frame.sub, auto: c.step.auto,
            head: document.querySelector('.st-head b')?.textContent, text: document.querySelector('.st-desc')?.textContent } : null,
        };
      },
    };
    void store.start();
    return () => {
      removeEventListener('error', onErr); removeEventListener('unhandledrejection', onRej); removeEventListener('keydown', onKey);
      removeEventListener('resize', onResize); mq.removeEventListener('change', onScheme);
    };
  }, [store]);
  return <StoreContext.Provider value={store}><ErrorBoundary><Root /></ErrorBoundary></StoreContext.Provider>;
}
