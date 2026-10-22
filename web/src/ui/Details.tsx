// The Details drawer: the shown turn's full event log, the waves and ticks, the champions' numbers, the belief where
// the game may show it, and the game record (the players, the research ids, the seeds once revealed, the marks).
import { useEffect, useState } from 'react';
import { Glyph } from './Surface';
import { useStore } from './useStore';
import { turnSubframes } from '../game/frames';
import type { Seat, SeatPublic } from '../game/types';
import { other } from '../game/types';
import { BELIEF_NOTE, fill, GUESS, recordType, sgn, turnsWord, who } from '../game/words';

export function Details() {
  const S = useStore();
  const [open, setOpen] = useState<Record<string, boolean>>({ events: true });
  const section = S.detailsSection;
  useEffect(() => { if (section) setOpen((o) => ({ ...o, [section]: true })); }, [section, S.detailsOpen]);
  if (!S.detailsOpen) return <aside className="details" id="details" hidden aria-label="Details" />;
  const c = S.ctrl, G = S.game;
  if (!c || !G) return <aside className="details" id="details" aria-label="Details"><div className="d-head"><h2>Details</h2><button className="pillbtn" onClick={() => S.closeDetails()}>Close</button></div></aside>;
  const v = c.shownView(), pos = G.positions[v], tr = v > 0 ? G.turns[v - 1] : null;
  const sec = (key: string, title: string, body: React.ReactNode) => (
    <details data-s={key} open={!!open[key]} onToggle={(e) => { const isOpen = (e.currentTarget as HTMLDetailsElement).open; setOpen((o) => (o[key] === isOpen ? o : { ...o, [key]: isOpen })); }}>
      <summary>{title}</summary>{body}
    </details>
  );
  const events = tr
    ? <><ol className="ev">{tr.outcomes.length ? tr.outcomes.map((e, i) => <li key={i} className={e.major ? '' : 'minor'}>{fill(e.text, S.short)}</li>) : <li>No events.</li>}</ol>
      <p className="muted">In the resolver's order. Waves and fights move tick by tick; trades, tower shots, ganks, deaths, plates and Recalls are known for the whole turn.</p></>
    : <p className="muted">The start: no turn played yet.</p>;
  const side = (n: Seat) => {
    const st = pos.champions[n], ln = pos.lane, front = ln.packets[n].find((q) => q[2]);
    return <>
      <tr><th colSpan={2}><Glyph n={n} /> {S.names[n]}</th></tr>
      <tr><td>Front wave</td><td>{front ? `${front[1]} HP at unit ${front[0] > 0 ? '+' : ''}${front[0]}` : 'none'}</td></tr>
      <tr><td>Waves behind</td><td>{ln.packets[n].filter((q) => !q[2]).map((q) => `${q[1]} HP`).join(', ') || 'none'}</td></tr>
      <tr><td>Crashed under the enemy tower</td><td>{ln.siege[n][1]} HP</td></tr>
      <tr><td>Wave damage per turn</td><td>{st.capacity ?? '–'} HP</td></tr>
      <tr><td>Budgets for the next turn</td><td>{Object.entries(st.wave_budgets || {}).map(([k, x]) => `${k} ${x}`).join(', ')}</td></tr>
      {tr && <tr><td>Enemy HP removed, tick 1 + 2</td><td>{tr.ticks.map((k) => k.wave_removed[n]).join(' + ')}</td></tr>}
    </>;
  };
  const pk = (l: [number, number, boolean][]) => l.map((q) => `${q[1]}@${q[0] > 0 ? '+' : ''}${q[0]}`).join(' ') || '–';
  const ticks = tr && (
    <>
      <p className="muted">Waves are listed as HP@unit (units counted from the lane: A's tower at −2, B's at +2). Units are where each champion stands after the frame.</p>
      <table className="dt"><thead><tr><th>Frame</th><th>A waves</th><th>B waves</th><th>Units A / B</th><th>Minion hits A / B</th></tr></thead><tbody>
        {turnSubframes(G, v).map((f, i) => <tr key={i}><td>{f.sub === 'tick' ? 'tick ' + f.tick!.tick : f.sub}</td><td>{pk(f.lane.packets.A)}</td><td>{pk(f.lane.packets.B)}</td>
          <td>{f.cells.A ?? (f.at_ward.A ? 'W' : '–')} / {f.cells.B ?? (f.at_ward.B ? 'W' : '–')}</td>
          <td>{f.tick ? `${f.tick.minion_fire.A.units} / ${f.tick.minion_fire.B.units}` : ''}</td></tr>)}
      </tbody></table>
    </>
  );
  const champ = (n: Seat) => {
    const st = pos.champions[n];
    return <>
      <tr><th colSpan={2}><Glyph n={n} /> {S.names[n]}</th></tr>
      <tr><td>Level, XP</td><td>{st.level}, {st.xp}{st.xp_next ? ` (next level at ${st.xp_next})` : ''}</td></tr>
      <tr><td>Gold earned</td><td>{st.gold_earned.toFixed(2)}</td></tr>
      <tr><td>Unspent, invested</td><td>{st.gold_unspent.toFixed(2)}, {st.invested.toFixed(2)}</td></tr>
      <tr><td>Attack power</td><td>{st.attack_power.toFixed(3)}</td></tr>
      <tr><td>Ward vision</td><td>{st.vision ? turnsWord(st.vision) : 'none'}</td></tr>
    </>;
  };
  const bl = c.belief(v);
  const omSeat = G.meta.seats.find((x) => x.kind === 'model' && x.arm === 'om');
  let belief: React.ReactNode;
  if (!omSeat) belief = <p className="muted">Only the adaptive agent keeps a guess about its opponent, and it is not playing here.</p>;
  else if (!bl) belief = <><p className="muted">Hidden in this game, as chosen when it started. It appears when the game ends.</p><p className="muted">{BELIEF_NOTE}</p></>;
  else belief = <>
    {Object.entries(bl.bySeat).map(([s, post]) => post && (
      <div key={s}>
        <p>The adaptive agent's guess about {S.names[other(s as Seat)] === 'You' ? 'your style' : S.names[other(s as Seat)].toLowerCase()} {v === 0 ? 'at the start' : 'after turn ' + v}:</p>
        <div className="belief">{bl.types.map((t, i) => [t, post[i]] as const).sort((x, y) => y[1] - x[1]).map(([t, p]) => (
          <span key={t} style={{ display: 'contents' }}><span>{GUESS[t] || t}</span><span className="bb"><i style={{ width: `${(100 * p).toFixed(1)}%` }} /></span><span className="num">{(100 * p).toFixed(1)}%</span></span>))}</div>
      </div>))}
    <p className="muted">{BELIEF_NOTE}</p>
  </>;
  const m = G.meta, sd = m.seeds, mk = m.marks || {};
  const seat = (x: SeatPublic) => (
    <tr key={x.seat}><td>{x.seat}</td><td>{x.kind === 'human' ? 'You' : x.kind === 'model'
      ? <>{who(x).name}<br /><span className="muted">research id {x.id}: M5's {x.arm === 'om' ? 'opponent-modelling' : 'pure self-play'} agent, seed {x.seed}</span><br /><span className="mono">file SHA-256 {(x.file_sha256 ?? '').slice(0, 16)}…</span></>
      : <>{who(x).name}<br /><span className="muted">rule ({x.id}): {x.rule}</span></>}</td></tr>
  );
  const stoppedLive = !!S.live && c.stopped;
  const record = (
    <>
      <table className="dt"><tbody>
        {m.seats.map(seat)}
        <tr><td>Record type</td><td>{recordType(m.provenance.kind)}</td></tr>
        <tr><td>Chance seed</td><td className="mono">{sd.chance ?? 'revealed when the game stops'}{sd.chance_chosen ? ' (chosen)' : ''}</td></tr>
        <tr><td>Policy seed</td><td className="mono">{sd.policy ?? 'revealed when the game stops'}</td></tr>
        {mk.lane_exit && <tr><td>Gold lead at the lane exit (turn 94)</td><td>{sgn(mk.lane_exit.panel.D0)}</td></tr>}
        {mk.lead_114 && <tr><td>Gold lead after 114 turns</td><td>{sgn(mk.lead_114.lead_A)}</td></tr>}
      </tbody></table>
      {stoppedLive && <p><button className="linkbtn" onClick={() => void S.downloadRecord()}>Download the record</button> <span className="muted">(a JSON file that replays bit for bit with the Python package)</span></p>}
      {S.live && !c.stopped && <p className="muted">The record can be downloaded once the game has stopped; its seeds stay secret until then.</p>}
    </>
  );
  return (
    <aside className="details" id="details" aria-label="Details">
      <div className="d-head"><h2 id="dTitle">{v === 0 ? 'Details: start' : `Details: turn ${v}`}</h2><button className="pillbtn" id="dClose" onClick={() => S.closeDetails()}>Close</button></div>
      <div className="d-body" id="dBody">
        {sec('events', 'Turn events', events)}
        {sec('waves', 'Waves and ticks', <><table className="dt"><tbody>{side('A')}{side('B')}</tbody></table>{ticks}</>)}
        {sec('champs', 'Champions', <table className="dt"><tbody>{champ('A')}{champ('B')}</tbody></table>)}
        {sec('belief', 'Belief', belief)}
        {sec('record', 'Game record', record)}
      </div>
    </aside>
  );
}
