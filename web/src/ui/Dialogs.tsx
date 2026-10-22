// The dialogs: New game (the opponent, the belief's visibility, an optional chance seed), New game between agents
// (Watch), and About (what a game shows, the players, what the research found, layout, theme and ticks).
import { useEffect, useRef } from 'react';
import { useStore } from './useStore';
import { LAYOUTS, THEMES } from '../game/store';
import type { ModelId } from '../game/types';
import { AGENTS, BOTS, DOCS_URL, gameTitle } from '../game/words';

function useDialog(open: boolean, onClose: () => void) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    else if (!open && d.open) d.close();
  }, [open]);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    const h = () => onClose();
    d.addEventListener('close', h);
    return () => d.removeEventListener('close', h);
  }, [onClose]);
  return ref;
}

export function NewGameDialog() {
  const S = useStore();
  const ref = useDialog(S.dialog === 'newGame', () => { if (S.dialog === 'newGame') S.closeDialog(); });
  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    const f = new FormData(e.currentTarget);
    if ((e.nativeEvent as SubmitEvent).submitter?.getAttribute('value') !== 'ok') return;
    const opp = f.get('opp') === 'om:0' ? 'om:0' : 'sp:0';
    const seedRaw = String(f.get('seed') ?? '').trim();
    const seed = seedRaw === '' ? null : Math.max(0, Math.floor(Number(seedRaw)));
    void S.newGame(opp, opp === 'om:0' && f.get('belief') === 'shown' ? 'shown' : 'hidden', seed !== null && Number.isFinite(seed) ? seed : null);
  };
  const engine = !!S.engine?.ready;
  return (
    <dialog id="newGame" aria-labelledby="ngTitle" ref={ref}>
      <form method="dialog" id="ngForm" onSubmit={submit}>
        <h2 id="ngTitle">New game</h2>
        {S.recovered === 'gone' && <p className="recovered">Your previous game is no longer on the server (it expired or the server restarted).</p>}
        {!engine && <p className="recovered">The engine is unavailable on this server, so a game cannot start.</p>}
        <fieldset><legend>Play against</legend>
          <label><input type="radio" name="opp" value="sp:0" defaultChecked /> Self-play agent <span className="muted">learned only by playing against itself</span></label>
          <label><input type="radio" name="opp" value="om:0" /> Adaptive agent <span className="muted">guesses your style and adjusts to it</span></label>
        </fieldset>
        <fieldset><legend>The adaptive agent's guess about your style</legend>
          <label><input type="radio" name="belief" value="hidden" defaultChecked /> Hidden until the game ends</label>
          <label><input type="radio" name="belief" value="shown" /> Shown during the game</label>
        </fieldset>
        <details>
          <summary>More options</summary>
          <label className="field">Chance seed <input type="number" name="seed" min={0} step={1} placeholder="drawn by the server" /></label>
          <p className="note-text">A chosen seed fixes the trade, gank and tower rolls, so the same seed gives the same chances against either agent. The record marks it as chosen. Leave it empty to let the server draw one, kept secret until the game stops.</p>
        </details>
        <div className="row-end"><button className="pillbtn" value="cancel">Cancel</button><button className="pillbtn gold" value="ok" disabled={!engine}>Start</button></div>
      </form>
    </dialog>
  );
}

export function NewSimDialog() {
  const S = useStore();
  const ref = useDialog(S.dialog === 'newSim', () => { if (S.dialog === 'newSim') S.closeDialog(); });
  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    if ((e.nativeEvent as SubmitEvent).submitter?.getAttribute('value') !== 'ok') return;
    const f = new FormData(e.currentTarget);
    void S.simulate(f.get('a') as ModelId, f.get('b') as ModelId);
  };
  const ready = S.engine?.simulations === 'ready' || S.engine?.simulations === 'in_process';
  return (
    <dialog id="newSim" aria-labelledby="nsTitle" ref={ref}>
      <form method="dialog" id="nsForm" onSubmit={submit}>
        <h2 id="nsTitle">New game between two agents</h2>
        <p className="muted">The engine plays 150 turns, then you watch them. One game shows how the agents play, not which is better.</p>
        {!ready && <p className="recovered">{S.engine?.simulations === 'starting' ? 'The simulation worker is still starting; try again in a moment.' : 'Simulations are unavailable on this server.'}</p>}
        <label className="sel">Left (A) <select name="a" defaultValue="om:0"><option value="om:0">Adaptive agent</option><option value="sp:0">Self-play agent</option></select></label>
        <label className="sel">Right (B) <select name="b" defaultValue="sp:0"><option value="sp:0">Self-play agent</option><option value="om:0">Adaptive agent</option></select></label>
        <div className="row-end"><button className="pillbtn" value="cancel">Cancel</button><button className="pillbtn gold" value="ok" disabled={!ready}>Simulate</button></div>
      </form>
    </dialog>
  );
}

export function AboutDialog() {
  const S = useStore();
  const ref = useDialog(S.dialog === 'about', () => { if (S.dialog === 'about') S.closeDialog(); });
  const G = S.game;
  const kind = G?.meta.provenance.kind;
  const intro = ({ research_record: 'A game recorded in the research and checked by replaying it exactly.',
    new_simulation: 'A game the engine just played between two trained agents.',
    human_game: 'Your game against a trained agent. Every action you choose is the one the engine plays.' } as Record<string, string>)[kind ?? ''] ?? '';
  const title = G ? gameTitle(G) : '';
  return (
    <dialog id="about" aria-labelledby="aboutTitle" ref={ref}>
      <div id="aboutBody">
        <h2 id="aboutTitle">About this game</h2>
        {G && G.name !== 'idle' && <p>{title && <b>{title}. </b>}{intro} A single game shows how the agents play. It is an example, not a measurement.</p>}
        <h3>The players</h3>
        <ul><li><b>Self-play agent</b>: {AGENTS['sp:0'].about}.</li>
          <li><b>Adaptive agent</b>: {AGENTS['om:0'].about}. Half of its training games were against fixed play styles.</li>
          <li><b>Always-push bot</b>: {BOTS.g_shove.about}. It is not trained.</li></ul>
        <p className="muted">Both agents are the final models of the research's last training run, and both pick their actions with some randomness, as they did in the research.</p>
        <h3>What the research found</h3>
        <ul><li>Against simple bots like the ones it trained against, the adaptive agent ends the lane far ahead: about 9 gold more than the self-play agent manages.</li>
          <li>Against strong opponents, including the self-play agent, it does slightly worse: about 0.4 to 0.6 gold.</li>
          <li>Opponents trained specially to beat them can beat both.</li></ul>
        <p>Neither agent is better overall, and neither was ever tested against people. The full numbers are in the research report (<a href={DOCS_URL} target="_blank" rel="noreferrer">the research documentation</a>).</p>
        <h3>Layout</h3>
        <div className="lay-pick"><div className="seg" role="group" aria-label="Layout">{(['auto', ...LAYOUTS] as const).map((l) =>
          <button key={l} data-lay={l} aria-pressed={l === S.layoutChoice} onClick={() => { S.closeDialog(); S.setLayout(l); }}>{l[0].toUpperCase() + l.slice(1)}</button>)}</div>
          <span className="muted">Auto uses the rail on a phone held sideways and the dock elsewhere. The game carries on when you switch.</span></div>
        <h3>Theme</h3>
        <div className="lay-pick"><div className="seg" role="group" aria-label="Theme">{THEMES.map((t) =>
          <button key={t} data-theme-pick={t} aria-pressed={t === S.theme} onClick={() => { S.closeDialog(); S.applyTheme(t, true); }}>{t[0].toUpperCase() + t.slice(1)}</button>)}</div>
          <span className="muted">Auto follows your system setting.</span></div>
        <h3>Ticks</h3>
        <div className="lay-pick"><label className="tick-auto"><input type="checkbox" id="aboutTicks" checked={S.ticksAuto} onChange={(e) => S.setTicksAuto(e.target.checked, true)} /> Play ticks automatically</label>
          <span className="muted">Off: after each turn you step through its two ticks yourself (Next tick). Watch's Play and away turns always play them on their own.</span></div>
        <h3>Keys</h3>
        <p className="muted">← → step through the turns, Home and End jump, Enter plays the chosen action, Space steps a tick or plays and pauses, Esc returns to the current turn or closes a panel.</p>
        <form method="dialog"><button className="btn" autoFocus>Close</button></form>
      </div>
    </dialog>
  );
}

export function Dialogs() {
  return <><NewGameDialog /><NewSimDialog /><AboutDialog /></>;
}
