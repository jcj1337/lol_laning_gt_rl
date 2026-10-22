// The primary interaction: the action rows in Play (a wave action, a behaviour, Pressure, or Recall or Ward, and the
// submit button that names the choice), the one control for forced away turns, the step panel of a turn shown frame
// by frame, and playback in Watch and after a stop, with the sequence's notes above it when one is open.
// Unavailable choices stay visible and explain why, with the engine's own reason. Browsing history disables
// submission and offers "Return to current turn".
import { useState } from 'react';
import { GuidePanel } from './Guide';
import { Glyph } from './Surface';
import { useStore } from './useStore';
import { Live } from '../game/controller';
import { SPEEDS } from '../game/frames';
import { SEATS } from '../game/types';
import type { MenuChoice, MenuSingle } from '../game/types';
import { actionWords, BEH, BEH_ORDER, fill, frameWords, sgn, WAVE, WAVE_ORDER } from '../game/words';

function Problem() {
  const S = useStore();
  const p = S.problem!;
  const title = { lost: 'The game is no longer on the server', engine: 'The engine is unavailable', unanswered: 'No answer from the server', error: 'Something went wrong' }[p.kind];
  return (
    <div className={`problem${p.kind === 'error' ? ' err' : ''}`} role="alert">
      <p><b>{title}.</b> {p.message}</p>
      <div className="row-end">
        {p.kind === 'lost' ? <button className="pillbtn small gold" onClick={() => { S.problem = null; S.openDialog('newGame'); }}>New game</button>
          : p.retry ? <button className="pillbtn small gold" onClick={p.retry}>{p.kind === 'unanswered' ? 'Reconnect and retry' : 'Try again'}</button> : null}
        <button className="pillbtn small" onClick={() => { S.problem = null; S.emit(); }}>Dismiss</button>
      </div>
    </div>
  );
}

export function PlaybackRow({ primary = true }: { primary?: boolean }) {
  const S = useStore();
  const c = S.ctrl!;
  return (
    <div className="pb">
      <button className={`btn${primary ? ' primary' : ''}`} id="btnPlay" onClick={() => S.togglePlay()}>{c.playing ? 'Pause' : c.view >= c.progress ? 'Play from the start' : 'Play'}</button>
      <div className="seg speed" role="group" aria-label="Playback speed">
        {SPEEDS.map((x) => <button key={x} aria-pressed={x === c.speed} data-s={x} onClick={() => S.setSpeed(x)}>{x}×</button>)}
      </div>
      <span className="muted pb-pos">{c.view === 0 ? 'Start' : 'Turn ' + c.view} of {c.progress}</span>
    </div>
  );
}

function TurnCard() {
  const S = useStore();
  const c = S.ctrl!, v = c.view, f = c.frame;
  if (v === 0) return <div className="turncard" id="turnCard" aria-live="polite"><p className="muted">Start of the game. Both champions begin at their own tower.</p></div>;
  const tr = S.game!.turns[v - 1];
  const acts = SEATS.map((n) => <span key={n}><Glyph n={n} /><b>{S.short[n]}</b> {actionWords(tr.parts[n === 'A' ? 0 : 1])}</span>);
  let list: React.ReactNode;
  if (f.sub === 'start' || f.sub === 'tick') list = <li className="muted">Resolving: the waves and champions move; outcomes follow at the end of the turn.</li>;
  else {
    const major = tr.outcomes.filter((e) => e.major), minor = tr.outcomes.length - major.length;
    list = <>
      {major.length ? major.slice(0, 4).map((e, i) => <li key={i}>{fill(e.text, S.short)}</li>) : <li className="muted">No trades, crashes, Recalls or deaths.</li>}
      {(major.length > 4 || minor > 0) && <li><button className="linkbtn" id="tcMore" onClick={() => S.openDetails('events')}>{major.length > 4 ? `+${major.length - 4} more` : `${minor} smaller event${minor === 1 ? '' : 's'}`}</button></li>}
    </>;
  }
  return (
    <div className="turncard" id="turnCard" aria-live="polite">
      <div className="tc-acts"><b>Turn {v}</b>{acts}<button className="linkbtn" id="tcTicks" onClick={() => c.stepTurn(v)}>Show the ticks</button></div>
      <ul className="tc-list">{list}</ul>
    </div>
  );
}

function StopSummary() {
  const S = useStore();
  const c = S.ctrl as Live;
  const pos = S.game!.positions[c.progress];
  const mk = S.game!.meta.marks || {};
  const marks = [mk.lane_exit && `Gold lead at the lane exit (turn 94): ${sgn(mk.lane_exit.panel.D0)}.`,
    mk.lead_114 && `After 114 turns: ${sgn(mk.lead_114.lead_A)} (this game only; M5 averaged it over many games).`].filter(Boolean);
  return (
    <div className="stopsum" id="stopSummary">
      <b>{c.progress >= c.n ? `The game reached the ${c.n}-turn limit.` : `You stopped the game after turn ${c.progress}.`}</b>{' '}
      Stopping declares no winner; the lane shows the state at the stop. Gold lead {sgn(pos.lead_A)}. {marks.join(' ')}{' '}
      <button className="linkbtn" id="btnDownload" onClick={() => void S.downloadRecord()}>Download the record</button>
      <button className="linkbtn" id="btnNew2" onClick={() => S.openDialog('newGame')}>New game</button>
    </div>
  );
}

function Playback() {
  const S = useStore();
  const live = S.mode === 'play' && !!S.live && !!S.ctrl?.stopped;
  return (
    <>
      {S.guide && <GuidePanel />}
      <PlaybackRow />
      {live && <StopSummary />}
      {S.mode === 'watch' && <TurnCard />}
    </>
  );
}

/** The frame of the turn being stepped: what it shows, Next tick (Space or →), Skip to the end, the automatic mode. */
export function StepPanel({ foot = false }: { foot?: boolean }) {
  const S = useStore();
  const c = S.ctrl!, s = c.step!, last = s.frames.length - 1, watching = S.mode === 'watch' || c.stopped;
  const W = frameWords(S.game!, s.t, s.frames, s.i, S.short, watching ? 2 : 3);
  const label = s.i < last - 1 ? 'Next tick' : s.i === last - 1 ? 'Show the result' : watching ? 'Continue' : `Continue to turn ${s.t + 1}`;
  const text = (
    <div className="steptext" aria-live="polite">
      <div className="st-head"><b>{W.head}</b><span className="st-dots" aria-hidden="true">{s.frames.map((_, j) => <i key={j} className={j < s.i ? 'done' : j === s.i ? 'now' : ''} />)}</span></div>
      <p className="st-desc">{W.text}</p>
      {W.outcomes && <ul className="st-out">{W.outcomes.map((o, i) => <li key={i}>{o}</li>)}{W.more && <li><button className="linkbtn" id="stMore" onClick={() => S.openDetails('events')}>{W.more}</button></li>}</ul>}
    </div>
  );
  const tick = <button className="btn primary submit" id="btnTick" title={`${label} (Space or right arrow)`} onClick={() => c.advance()}>{label}</button>;
  const skip = s.i < last ? <button className="linkbtn" id="btnSkip" onClick={() => c.skip()}>Skip to the end of the turn</button> : <span />;
  if (foot) return <>{text}{tick}{skip}</>;
  const inFoot = S.phone && S.arr === 'rail';
  return (
    <>
      {watching && <PlaybackRow primary={false} />}
      <div className="act stepper">
        {!inFoot && text}
        <div className="side">
          <label className="tick-auto"><input type="checkbox" id="tickAuto" checked={S.ticksAuto} onChange={(e) => S.setTicksAuto(e.target.checked, true)} /> Play ticks automatically</label>
          {!inFoot && tick}
          <div className="foot">{!inFoot && skip}</div>
        </div>
      </div>
    </>
  );
}

const dis = (label: string, cls = 'opt') => <button key={label} className={cls} aria-disabled="true" tabIndex={-1}>{label}</button>;

/** The choices greyed out while the champion is away, and one button that plays the away turns one engine turn at
 * a time. */
function Away({ foot = false }: { foot?: boolean }) {
  const S = useStore();
  const c = S.ctrl as Live, review = c.historical;
  const main = review
    ? <button className="btn primary submit" id="btnBack" onClick={() => c.returnToCurrent()}>Return to current turn</button>
    : <button className="btn primary submit" id="btnAway" disabled={S.busy} aria-label="Continue: your champion is away, so each away turn is played for you" onClick={() => void S.continueAway()}>Continue</button>;
  const stop = <button className="linkbtn" id="btnStop" onClick={() => void S.stopGame()}>Stop game</button>;
  if (foot) return <>{main}{stop}</>;
  const inFoot = S.phone && S.arr === 'rail';
  return (
    <>
      <div className="act">
        <div className="rows">
          <div className="r"><span className="lbl">Wave</span><div className="opts">{WAVE_ORDER.map((w) => dis(WAVE[w]))}</div></div>
          <div className="r"><span className="lbl">Behaviour</span><div className="opts">{BEH_ORDER.map((b) => dis(BEH[b]))}</div></div>
          <div className="r"><span className="lbl">Objective</span><div className="opts one">{dis('Pressure')}</div></div>
        </div>
        <div className="side">
          <div className="singles">{dis('Recall', 'single')}{dis('Ward', 'single')}</div>
          {!inFoot && main}
          <div className="foot"><span />{!inFoot && stop}</div>
        </div>
      </div>
      <div className="help">{S.error && <span className="err" role="alert">{S.error}</span>}</div>
    </>
  );
}

function Actions({ foot = false }: { foot?: boolean }) {
  const S = useStore();
  const c = S.ctrl as Live, m = c.menu()!, t = c.progress + 1, review = c.historical, s = S.sel;
  const [helpKey, setHelpKey] = useState<string | null>(null);
  const R: Record<string, string> = {};
  const pr = m.objectives.find((o) => o.id === 1)!;
  if (s.pressure && !pr.legal) s.pressure = false;
  const opt = (key: string, label: string, on: boolean, reason: string | null | undefined, help: string | null | undefined) => {
    R[key] = reason ? `${label} is unavailable. ${reason}` : `${label}. ${help ?? ''}`;
    const off = !!reason || review;
    return (
      <button key={key} className={`opt${on ? ' on' : ''}`} data-k={key} aria-pressed={on} aria-disabled={off || undefined} aria-label={R[key]}
        onMouseEnter={() => setHelpKey(key)} onFocus={() => setHelpKey(key)} onMouseLeave={() => setHelpKey(null)} onBlur={() => setHelpKey(null)}
        onClick={() => { if (review) return; if (off) { setHelpKey(key); return; } S.choose(key); }}>{label}</button>
    );
  };
  const waves = m.waves.map((w: MenuChoice) => opt(`w${w.id}`, WAVE[w.id], s.kind === 'lane' && s.wave === w.id, null, w.help));
  const behs = m.behaviours.map((b: MenuChoice) => opt(`b${b.id}`, BEH[b.id], s.kind === 'lane' && s.beh === b.id,
    s.pressure && ![0, 2].includes(b.id) ? 'Pressure combines with Hold or Trade only.' : null, b.help));
  const press = opt('p', 'Pressure', s.kind === 'lane' && s.pressure, pr.legal ? null : pr.reason, pr.help);
  const [rc, wd] = m.singles;
  const single = (x: MenuSingle, kind: 'recall' | 'ward') => {
    R[kind] = x.legal ? `${x.name}. ${x.help}` : `${x.name} is unavailable. ${x.reason}`;
    const off = !x.legal || review;
    return (
      <button key={kind} className={`single${s.kind === kind ? ' on' : ''}`} data-k={kind} aria-pressed={s.kind === kind} aria-disabled={off || undefined} aria-label={R[kind]}
        onMouseEnter={() => setHelpKey(kind)} onFocus={() => setHelpKey(kind)} onMouseLeave={() => setHelpKey(null)} onBlur={() => setHelpKey(null)}
        onClick={() => { if (review) return; if (off) { setHelpKey(kind); return; } S.choose(kind); }}>
        <span className="nm">{x.name}</span><span className="why">{x.legal ? (kind === 'recall' ? 'Back to base' : 'In the bush') : x.reason}</span>
      </button>
    );
  };
  const code = S.selectionCode();
  const ok = code !== null && m.legal.includes(code) && !c.historical && !S.busy;
  const what = s.kind === 'recall' ? 'Recall' : s.kind === 'ward' ? 'Ward'
    : s.wave !== null && s.beh !== null ? `${WAVE[s.wave]}, ${BEH[s.beh]}${s.pressure ? ', Pressure' : ''}` : null;
  const off = [rc, wd].filter((x) => !x.legal).map((x) => x.reason).join(' ');
  const inBush = !!S.game!.positions[c.progress]?.champions?.A?.at_ward;
  const bush = 'You are in the bush. Any wave action with a behaviour walks you back to the lane; choose Safe to step back to your side, then you can Recall next turn.';
  const base = review ? `You are looking at an earlier turn. Return to turn ${t} to choose.`
    : S.error || (inBush ? bush : off) || (what ? 'Point at any choice to see what it does.' : 'Choose a wave action and a behaviour, or Recall or Ward.');
  const main = review
    ? <button className="btn primary submit" id="btnBack" onClick={() => c.returnToCurrent()}>Return to current turn</button>
    : <button className="btn primary submit" id="btnSubmit" disabled={!ok} onClick={() => void S.submit(S.selectionCode())}>{what ? `Play turn ${t}: ${what}` : `Play turn ${t}`}</button>;
  const stop = <button className="linkbtn" id="btnStop" onClick={() => void S.stopGame()}>Stop game</button>;
  if (foot) return <>{main}{stop}</>;
  const inFoot = S.phone && S.arr === 'rail';
  return (
    <>
      <div className="act">
        <div className="rows">
          <div className="r"><span className="lbl">Wave</span><div className="opts">{waves}</div></div>
          <div className="r"><span className="lbl">Behaviour</span><div className="opts">{behs}</div></div>
          <div className="r"><span className="lbl">Objective</span><div className="opts one">{press}</div><span className="why">{pr.legal ? 'Combines with Hold or Trade.' : pr.reason}</span></div>
        </div>
        <div className="side">
          <div className="singles">{single(rc, 'recall')}{single(wd, 'ward')}</div>
          {!inFoot && main}
          <div className="foot"><span className="muted">The opponent has already chosen.</span>{!inFoot && stop}</div>
        </div>
      </div>
      <div className="help" id="help">{helpKey && R[helpKey] ? R[helpKey] : S.error ? <span className="err" role="alert">{S.error}</span> : base}</div>
    </>
  );
}

export function Controls() {
  const S = useStore();
  const c = S.ctrl;
  if (S.loading) return <p className="loading" role="status">{S.loading}</p>;
  if (S.problem) return <Problem />;
  if (S.mode === 'play' && !S.live) {
    return <p className="note">{S.engine?.ready
      ? <>No game on screen. <button className="linkbtn" onClick={() => S.openDialog('newGame')}>Start a game</button>.</>
      : <>Play needs the engine, which this server could not load: {S.engine?.error ?? 'the server did not answer'}. Watch works without it only for research games the server has verified.</>}</p>;
  }
  if (!c) return <p className="note">Loading…</p>;
  if (c.stepping) return <StepPanel />;
  if (S.mode === 'watch' || c.stopped) return <Playback />;
  if (!(c instanceof Live) || !c.menu()) return <p className="note">Waiting for the engine…</p>;
  return c.menu()!.forced !== null ? <Away /> : <Actions />;
}

/** A phone's rail keeps the main button (or the playback row), Stop and a stepped turn's words in a footer that
 * stays in view while the rail scrolls. */
export function ControlsFoot() {
  const S = useStore();
  const c = S.ctrl;
  if (S.loading || S.problem || !c || (S.mode === 'play' && !S.live)) return null;
  if (c.stepping) return <StepPanel foot />;
  if (S.mode === 'watch' || c.stopped) return <PlaybackRow />;
  if (!(c instanceof Live) || !c.menu()) return null;
  return c.menu()!.forced !== null ? <Away foot /> : <Actions foot />;
}
