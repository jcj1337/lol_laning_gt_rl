// The gold-lead graph: blue above zero while the left player has earned more gold, red below, a grey tie, unsigned
// axis numbers, the selected turn marked and the hovered one read out. The y-axis is symmetric and at least ±2 gold,
// so small changes stay small. In a running game the x-axis spans only the turns played so far (at least 20). A plot
// lower than 90 px is a sparkline. Clicking a turn shows that position (navigation only; never a submission).
import { useEffect, useRef, useState } from 'react';
import { useStore } from './useStore';
import type { AppStore } from '../game/store';
import { GRAPH_NOTE } from '../game/words';

function niceStep(m: number, target = 3): number {
  for (const s of [0.5, 1, 2, 5, 10, 20, 25, 50, 100]) if (m / s <= target) return s;
  return 200;
}

function shownProgress(S: AppStore): number {
  const c = S.ctrl!, s = c.resolving();
  return s && S.live && s.t === c.progress ? s.t - 1 : c.progress;
}

export function graphLabel(S: AppStore, p: number): string {
  const G = S.game!;
  const ahead = { A: S.live ? 'You' : S.short.A, B: S.live ? 'Opponent' : S.short.B };
  const v = G.positions[p].lead_A;
  return `${p === 0 ? 'Start' : 'Turn ' + p}: ${Math.abs(v) < 0.005 ? 'even' : `${v > 0 ? ahead.A : ahead.B} +${Math.abs(v).toFixed(2)}`}`;
}

function useSize(ref: React.RefObject<HTMLDivElement | null>): { w: number; h: number } {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => { const r = el.getBoundingClientRect(); setSize((s) => (s.w === r.width && s.h === r.height ? s : { w: r.width, h: r.height })); };
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

export function Plot({ id, onMini }: { id: string; onMini?: (mini: boolean) => void }) {
  const S = useStore();
  const ref = useRef<HTMLDivElement>(null);
  const size = useSize(ref);
  const geo = useRef({ l: 0, r: 0, n: 1 });
  const c = S.ctrl, G = S.game;
  const H = Math.max(20, size.h || 140), mini = H < 90;
  useEffect(() => { onMini?.(mini); }, [mini, onMini]);
  if (!c || !G) return <div ref={ref} id={id} className="g-plot" />;
  const W = Math.max(120, size.w || 300);
  const M = mini ? { l: 4, r: 4, t: 4, b: 4 } : { l: 30, r: 8, t: 21, b: 17 };
  const running = !!S.live && !c.stopped;
  const sv = c.shownView(), prog = shownProgress(S);
  const n = running ? Math.max(20, Math.ceil((prog + 1) / 10) * 10) : Math.max(prog, 1);
  const X = (p: number) => M.l + ((W - M.l - M.r) * p) / n;
  let mx = 0;
  for (let p = 0; p <= prog; p++) mx = Math.max(mx, Math.abs(G.positions[p].lead_A));
  const span = Math.max(mx, 2), step = niceStep(span, 2.5), top = step * Math.ceil(span / step);
  const Y = (v: number) => M.t + ((H - M.t - M.b) * (top - v)) / (2 * top);
  const y0 = Y(0);
  const pts: string[] = [];
  for (let p = 0; p <= prog; p++) pts.push(`${X(p).toFixed(1)} ${Y(G.positions[p].lead_A).toFixed(1)}`);
  const line = `M${pts.join(' L')}`;
  const area = `${line} L${X(prog).toFixed(1)} ${y0} L${X(0)} ${y0} Z`;
  geo.current = { l: M.l, r: M.r, n };
  const grid: React.ReactNode[] = [];
  for (let v = -top; v <= top + 1e-9; v += step) {
    const y = Y(v), zero = Math.abs(v) < 1e-9;
    if (mini && !zero) continue;
    grid.push(<line key={`g${v}`} x1={M.l} x2={W - M.r} y1={y} y2={y} className={zero ? 'g-zero' : 'g-grid'} />);
    if (!mini) grid.push(<text key={`t${v}`} x={M.l - 6} y={y + 4} textAnchor="end" className="g-axis">{zero ? '0' : Math.abs(v).toFixed(step < 1 ? 1 : 0)}</text>);
  }
  const xs = niceStep(n, 6);
  const xaxis: React.ReactNode[] = [];
  if (!mini) for (let p = 0; p <= n; p += xs) xaxis.push(<text key={p} x={X(p)} y={H - 4} textAnchor="middle" className="g-axis">{p}</text>);
  const events: React.ReactNode[] = [];
  if (S.events && !mini) {
    for (const [p, lab] of [[94, 'Lane exit (94)'], [114, 'Turn 114']] as const) {
      if (p > n) continue;
      events.push(<g key={`m${p}`}><line x1={X(p)} x2={X(p)} y1={M.t} y2={H - M.b} className="g-mark" /><text x={X(p) + 4} y={M.t + 24} className="g-axis g-marklab">{lab}</text></g>);
    }
    for (let t = 1; t <= prog; t++) {
      G.turns[t - 1].outcomes.forEach((e, i) => {
        if (!e.seat) return;
        const x = X(t), y = e.seat === 'A' ? M.t + 6 : H - M.b - 6, col = `var(--${e.seat.toLowerCase()})`;
        const key = `e${t}-${i}`;
        if (e.kind === 'death') events.push(<path key={key} d={`M${x - 3.5} ${y - 3.5} l7 7 m0 -7 l-7 7`} stroke={col} strokeWidth={1.8} />);
        else if (e.kind === 'recall') events.push(<path key={key} d={`M${x} ${y - 4} l4 7 h-8 z`} fill="none" stroke={col} strokeWidth={1.2} />);
        else if (e.kind === 'crash') events.push(<rect key={key} x={x - 2.3} y={y - 2.3} width={4.6} height={4.6} fill={col} />);
        else if (e.kind === 'plate') events.push(<rect key={key} x={x - 3} y={y - 3} width={6} height={6} transform={`rotate(45 ${x} ${y})`} className="g-plate" stroke={col} strokeWidth={1.3} />);
      });
    }
  }
  const mark = (p: number, cls: string) => {
    const v = G.positions[p].lead_A, x = X(p), text = graphLabel(S, p);
    const dot = Math.abs(v) < 0.005 ? 'var(--muted)' : v > 0 ? 'var(--a)' : 'var(--b)';
    if (mini) return <g key={cls} className={cls}><line x1={x} x2={x} y1={M.t - 2} y2={H - M.b + 2} /><circle cx={x} cy={Y(v)} r={3} style={{ stroke: dot }} /></g>;
    const w = text.length * 6.4 + 14, lx = Math.max(M.l, Math.min(W - M.r - w, x - w / 2));
    return <g key={cls} className={cls}><line x1={x} x2={x} y1={M.t} y2={H - M.b} /><circle cx={x} cy={Y(v)} r={4} style={{ stroke: dot }} />
      <rect x={lx} y={1} width={w} height={17} rx={3} /><text x={lx + w / 2} y={13} textAnchor="middle">{text}</text></g>;
  };
  const ahead = { A: S.live ? 'You' : S.short.A, B: S.live ? 'Opponent' : S.short.B };
  const turnAt = (ev: React.PointerEvent | React.MouseEvent): number => {
    const r = (ev.currentTarget as HTMLElement).getBoundingClientRect(), g = geo.current;
    const p = Math.round(((ev.clientX - r.left - g.l) / (r.width - g.l - g.r)) * g.n);
    return Math.max(0, Math.min(shownProgress(S), p));
  };
  return (
    <div ref={ref} id={id} className="g-plot" onPointerMove={(e) => S.setHover(turnAt(e))} onPointerLeave={() => S.setHover(null)} onClick={(e) => c.goto(turnAt(e))}>
      <svg className={mini ? 'g-mini' : ''} viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Gold lead by turn. ${GRAPH_NOTE}`}>
        <defs>
          <clipPath id={`${id}Up`}><rect x={0} y={0} width={W} height={Math.max(0, y0 - 1)} /></clipPath>
          <clipPath id={`${id}Dn`}><rect x={0} y={y0 + 1} width={W} height={H} /></clipPath>
        </defs>
        {grid}{xaxis}
        <path d={area} fill="var(--a)" fillOpacity="var(--g-fill)" clipPath={`url(#${id}Up)`} />
        <path d={area} fill="var(--b)" fillOpacity="var(--g-fill)" clipPath={`url(#${id}Dn)`} />
        <path d={line} className="g-line" stroke="var(--g-neutral)" />
        <path d={line} className="g-line" stroke="var(--a)" clipPath={`url(#${id}Up)`} />
        <path d={line} className="g-line" stroke="var(--b)" clipPath={`url(#${id}Dn)`} />
        {events}
        {mark(sv, `g-sel${c.historical && running ? ' past' : ''}`)}
        {S.hover !== null && S.hover !== sv && S.hover <= prog && mark(S.hover, 'g-hover')}
        {!mini && <><text x={M.l + 6} y={M.t + 12} className="g-side" fill="var(--a)">{ahead.A} ahead</text>
          <text x={M.l + 6} y={H - M.b - 6} className="g-side" fill="var(--b)">{ahead.B} ahead</text></>}
      </svg>
    </div>
  );
}

export function GraphPanel() {
  const S = useStore();
  const [mini, setMini] = useState(false);
  const c = S.ctrl, G = S.game;
  let val = '';
  if (mini && c && G) {
    const prog = shownProgress(S);
    val = graphLabel(S, S.hover !== null && S.hover <= prog ? S.hover : c.shownView());
  }
  return (
    <div className="z-graph" id="graphPanel" aria-labelledby="gTitle">
      <div className="g-head">
        <h2 id="gTitle">Gold lead</h2>
        <span className="g-val" id="gVal">{val}</span>
        <button className="linkbtn" id="gExpand" aria-expanded={S.graphBig} aria-controls="graphBig" onClick={() => S.setGraphBig(!S.graphBig)}>{S.graphBig ? 'Collapse' : 'Expand'}</button>
      </div>
      {c && G ? <Plot id="graph" onMini={setMini} /> : <div id="graph" className="g-plot" />}
    </div>
  );
}

export function GraphBig() {
  const S = useStore();
  return (
    <div className={`z-big${S.arr !== 'console' ? ' pop' : ''}`} id="graphBig" hidden={!S.graphBig} aria-labelledby="gbTitle">
      <div className="g-head">
        <h2 id="gbTitle">Gold lead</h2>
        <span className="g-note" id="gNote">{GRAPH_NOTE}</span>
        <label className="g-toggle"><input type="checkbox" id="gEvents" checked={S.events} onChange={(e) => S.setEvents(e.target.checked)} /> Show events</label>
        <button className="linkbtn" id="gCollapse" onClick={() => S.setGraphBig(false)}>Collapse</button>
      </div>
      {S.graphBig && S.ctrl && S.game ? <Plot id="graphB" /> : <div id="graphB" className="g-plot" />}
    </div>
  );
}
