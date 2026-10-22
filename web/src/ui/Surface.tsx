// The one glass surface and its zones: the status strip (rail and phone), the turn line, the turn history with its
// navigation, the controls, and the gold-lead graph (compact, and large in a sheet). The CSS arranges the same zones
// as a dock, a rail or a console; here only the top bar and the turn buttons change their place.
import { useEffect, useRef } from 'react';
import { Controls, ControlsFoot } from './Controls';
import { GraphBig, GraphPanel } from './Graph';
import { TopBar } from './TopBar';
import { useStore } from './useStore';
import type { Seat } from '../game/types';
import { actionWords, fill, phrase } from '../game/words';

export function Glyph({ n }: { n: Seat }) {
  return n === 'A'
    ? <svg className="glyph" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="var(--a)" /><text x="8" y="11.5" textAnchor="middle">A</text></svg>
    : <svg className="glyph" viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="3" width="10" height="10" rx="1" transform="rotate(45 8 8)" fill="var(--b)" /><text x="8" y="11.5" textAnchor="middle">B</text></svg>;
}

/** The turn label: where the game is, or the position shown. */
export function turnLabel(S: ReturnType<typeof useStore>): string {
  const c = S.ctrl;
  if (!c) return 'No game';
  const f = c.frame, v = c.view, live = S.mode === 'play' && !!S.live;
  let lab = v === 0 ? 'Start' : f.sub === 'start' || f.sub === 'tick' ? `Turn ${f.turn}` : `After turn ${v}`;
  if (live && !c.stopped && !c.historical && !c.stepping) lab = `Turn ${c.progress + 1} of ${c.n} · your move`;
  else if (live && c.stopped && v === c.progress) lab = `Stopped after turn ${c.progress}`;
  return lab;
}

function StatusSeat({ n }: { n: Seat }) {
  const S = useStore();
  const st = S.ctrl?.frame.status[n];
  if (!st) return <div className="st" />;
  const pct = Math.max(0, Math.min(100, (100 * st.hp) / st.hp_max));
  return (
    <div className="st" id={`st${n}`} aria-label={`Seat ${n}`}>
      <div className="st-row"><Glyph n={n} /><span className="nm" title={S.names[n]}>{S.names[n]}</span><span className="lvl">Level {st.level}</span></div>
      <div className="st-row2">
        {st.status === 'present'
          ? <><span className="bar" role="img" aria-label={`HP ${st.hp} of ${st.hp_max}`}><i style={{ width: `${pct}%`, background: `var(--${n.toLowerCase()})` }} /></span><span className="num">{st.hp} / {st.hp_max} HP</span></>
          : <span className="away">{st.status === 'dead' ? 'Dead' : 'Recalling'}, back in {st.returns_in} turn{st.returns_in === 1 ? '' : 's'}</span>}
      </div>
    </div>
  );
}

function TurnLine() {
  const S = useStore();
  const c = S.ctrl;
  const live = S.mode === 'play' && !!S.live;
  const hist = !!c && live && !c.stopped && c.historical;
  useEffect(() => { document.body.classList.toggle('historical', hist); }, [hist]);
  if (!c) return <div className="z-line" id="zLine"><span className="turnlabel" id="turnPill"><span id="turnLabel">No game</span></span></div>;
  const v = c.view, f = c.frame;
  let summary: React.ReactNode = null;
  if (S.mode === 'play') {
    if (v === 0) summary = <span className="muted">Start of the game. Both champions begin at their own tower.</span>;
    else {
      const tr = S.game!.turns[v - 1];
      if (f.sub === 'start' || f.sub === 'tick') {
        summary = <><span><Glyph n="A" />{actionWords(tr.parts[0])}</span><span><Glyph n="B" />{actionWords(tr.parts[1])}</span></>;
      } else {
        const major = tr.outcomes.filter((e) => e.major), minor = tr.outcomes.length - major.length;
        const text = major.length ? major.slice(0, 3).map((e) => fill(e.text, S.short)).join(' · ') : 'No trades, crashes, Recalls or deaths.';
        const more = tr.outcomes.length - Math.min(3, major.length);
        summary = <><b>Turn {v}</b><span className="sum-text">{text}</span>
          {(more > 0 || minor > 0) && <button className="linkbtn" id="sumMore" onClick={() => S.openDetails('events')}>{more > 0 ? `+${more} more` : 'All events'}</button>}
          {!c.stepping && <button className="linkbtn" id="sumTicks" onClick={() => c.stepTurn(v)}>Show the ticks</button>}</>;
      }
    }
  }
  return (
    <div className="z-line" id="zLine">
      <span className="turnlabel" id="turnPill" hidden={hist}><span id="turnLabel">{turnLabel(S)}</span></span>
      <span className="banner" id="banner" hidden={!hist} role="status">
        <span id="bannerText">{hist ? `Viewing ${c.view === 0 ? 'the start' : 'turn ' + c.view} · the game is at turn ${c.progress + 1}` : ''}</span>
        <button className="pillbtn small" id="btnReturn" onClick={() => c.returnToCurrent()}>Return to current turn</button>
      </span>
      {S.mode === 'play' && <div className="summary" id="summary" aria-live="polite">{summary}</div>}
    </div>
  );
}

function NavButtons() {
  const S = useStore();
  const c = S.ctrl;
  const atStart = !c || c.view === 0, atEnd = !c || c.view >= c.progress;
  return (
    <div className="nav" id="nav" role="group" aria-label="Move through the turns">
      <button id="btnFirst" title="First (Home)" aria-label="First position" disabled={atStart} onClick={() => c?.first()}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3.5v9M11.5 3.5 7 8l4.5 4.5" /></svg></button>
      <button id="btnPrev" title="Previous (Left arrow)" aria-label="Previous turn" disabled={atStart} onClick={() => c?.prev()}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3.5 5.5 8l4.5 4.5" /></svg></button>
      <button id="btnNext" title="Next (Right arrow)" aria-label="Next turn" disabled={atEnd} onClick={() => c?.next()}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3.5 4.5 4.5L6 12.5" /></svg></button>
      <button id="btnLast" title="Latest (End)" aria-label="Latest position" disabled={atEnd} onClick={() => c?.latest()}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11.5 3.5v9M4.5 3.5 9 8l-4.5 4.5" /></svg></button>
    </div>
  );
}

function HistoryZone() {
  const S = useStore();
  const c = S.ctrl;
  const listRef = useRef<HTMLOListElement>(null);
  const view = c?.view ?? 0;
  const rs = c?.resolving() ?? null;
  const live = S.mode === 'play' && !!S.live;
  useEffect(() => {
    const list = listRef.current;
    const row = list?.querySelector<HTMLElement>(`[data-p="${view}"]`);
    if (list && row) {
      const lr = list.getBoundingClientRect(), rr = row.getBoundingClientRect();
      if (rr.top < lr.top || rr.bottom > lr.bottom) list.scrollTop += rr.top - lr.top - lr.height / 2;
    }
  }, [view, c?.progress]);
  const rows: React.ReactNode[] = [];
  if (c) {
    for (let t = 0; t <= c.progress; t++) {
      const cls = ['row', t === 0 ? 'start' : '', t === view ? 'viewed' : '', t === S.hover ? 'hover' : '', rs && rs.t === t ? 'resolving' : ''].filter(Boolean).join(' ');
      const go = () => c.goto(t);
      if (t === 0) {
        rows.push(<li key={0} className={cls} data-p={0} role="button" tabIndex={0} aria-current={t === view || undefined} onClick={go} onKeyDown={(e) => { if (e.key === 'Enter') go(); }}><span className="n">0</span><span className="a">Start</span></li>);
        continue;
      }
      const tr = S.game!.turns[t - 1];
      const ph = tr.outcomes.filter((e) => e.major).map((e) => phrase(e, S.short)).filter((x): x is string => !!x);
      const a = actionWords(tr.parts[0]), b = actionWords(tr.parts[1]);
      rows.push(
        <li key={t} className={cls} data-p={t} role="button" tabIndex={0} aria-current={t === view || undefined} onClick={go} onKeyDown={(e) => { if (e.key === 'Enter') go(); }}
          aria-label={`Turn ${t}: ${S.names.A} ${a}; ${S.names.B} ${b}. ${ph.join('. ')}`}>
          <span className="n">{t}</span><span className="a" title={a}>{a}</span><span className="b" title={b}>{b}</span>
          {ph.length > 0 && <span className="o">{ph.slice(0, 2).join(' · ')}{ph.length > 2 ? ` +${ph.length - 2}` : ''}</span>}
        </li>,
      );
    }
    if (live && !c.stopped) {
      rows.push(<li key="pending" className="row pending"><span className="n">{c.progress + 1}</span><span className="a">{c.forced() ? 'Away' : 'Your move'}</span><span className="b" title="Chosen before your move; revealed with it">Chosen</span></li>);
    }
  }
  const toggleTitle = S.arr === 'console' ? 'Hide the turns' : S.histOpen ? 'Show fewer turns' : 'Show all turns';
  return (
    <>
      <div className="h-top">
        <span className="h-title">Turns</span>
        <button className="iconbtn" id="hToggle" aria-expanded={S.histOpen} aria-controls="zHist" title={toggleTitle} aria-label={toggleTitle} onClick={() => S.setHistory(!S.histOpen)}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 10l4-4 4 4" /></svg></button>
      </div>
      <div className="h-head"><span>#</span><span id="hA"><Glyph n="A" /><span title={S.names.A}>{S.head.A}</span></span><span id="hB"><Glyph n="B" /><span title={S.names.B}>{S.head.B}</span></span></div>
      <ol className="h-list" id="hList" ref={listRef}>{rows}</ol>
    </>
  );
}

export function Surface() {
  const S = useStore();
  const stepping = !!S.ctrl?.stepping;
  const mainRef = useRef<HTMLDivElement>(null);
  const mainH = useRef(0);
  // the step panel takes the actions' place at their height, so nothing moves when a turn starts or ends
  useEffect(() => {
    const zm = mainRef.current;
    if (!zm) return;
    if (stepping && !mainH.current) mainH.current = zm.offsetHeight;
    if (!stepping) mainH.current = 0;
    zm.style.minHeight = stepping && mainH.current ? `${mainH.current}px` : '';
  }, [stepping]);
  const watch = S.mode === 'watch' || !!S.ctrl?.stopped;
  return (
    <section className="glass surface" id="surface" aria-label="Game controls">
      <div className="z-scroll" id="zScroll">
        {S.arr === 'rail' && <TopBar />}
        <div className="z-status" id="zStatus"><StatusSeat n="A" /><StatusSeat n="B" /></div>
        <TurnLine />
        <div className={`z-hist${S.arr === 'dock' && S.histOpen ? ' pop' : ''}`} id="zHist" aria-label="Turn history">
          <HistoryZone />
          {S.arr !== 'console' && <NavButtons />}
        </div>
        <div className="z-main" aria-label="Actions" ref={mainRef}><div className={`controls${watch ? ' watch' : ''}`} id="controls"><Controls /></div></div>
        <GraphPanel />
        <GraphBig />
        <div className="z-bar" id="zBar">
          {S.arr === 'console' && <NavButtons />}
          <button className="pillbtn small" id="cHist" aria-expanded={S.histOpen} aria-controls="zHist" onClick={() => S.setHistory(!S.histOpen)}>Turns</button>
        </div>
      </div>
      <div className="z-foot" id="zFoot">{S.phone && S.arr === 'rail' && <ControlsFoot />}</div>
    </section>
  );
}

