// The scene mounted in React: one <svg> that the imperative Lane class draws into, plus the text equivalent of the
// frame for screen readers. The lane is fitted on a resize or a layout change (store.refit), never as turns advance.
import { useEffect, useLayoutEffect, useRef } from 'react';
import type { Frame } from '../game/frames';
import { arrowsFor } from '../game/moves';
import { Lane } from '../scene/lane';
import { SEATS } from '../game/types';
import type { Names, Turn } from '../game/types';
import { arrowWords } from '../game/words';
import { useStore } from './useStore';

export function laneText(f: Frame, names: Names, turnLabel: string, turns: Turn[]): string {
  const out = [turnLabel + '.'];
  for (const n of SEATS) {
    const st = f.status[n], front = f.lane.packets[n].find((q) => q[2]);
    const at = f.at_ward[n] ? 'in the bush at the ward spot (the bush hides no one here)' : `unit ${f.cells[n]}`;
    out.push(`${names[n]}: level ${st.level}, ${st.status === 'present' ? `${st.hp} of ${st.hp_max} HP, ${at}` : st.status}; wave ${front ? `${front[1]} HP at unit ${front[0]}` : 'none'}.`);
  }
  for (const a of arrowsFor(f, turns)) out.push(`${arrowWords(a, names).text}.`);
  return out.join(' ');
}

export function LaneView({ turnLabel }: { turnLabel: string }) {
  const S = useStore();
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const lane = new Lane(ref.current!, () => ({ names: S.names, head: S.head, live: S.mode === 'play' && !!S.live, turns: S.game?.turns ?? [] }));
    lane.labels = S.arr !== 'rail' && !S.phone;
    S.lane = lane;
    S.refitLane();
    return () => { if (S.lane === lane) S.lane = null; };
  }, [S]);
  const frame = S.ctrl?.frame ?? null, highlight = S.guideHighlight();
  useLayoutEffect(() => {
    const lane = S.lane;
    if (!lane || !frame) return;
    if (!lane.W) lane.layout(true);
    lane.highlight = highlight;
    lane.update(frame);
  }, [S, frame, S.names, highlight]);
  return (
    <>
      <svg id="lane" className="scene" ref={ref} role="img" aria-labelledby="laneText" />
      <p id="laneText" className="sr-only" aria-live="polite">{frame ? laneText(frame, S.names, turnLabel, S.game?.turns ?? []) : 'No game on screen.'}</p>
    </>
  );
}
