// Drawable frames. A position frame is the state after p turns. A resolved turn is shown as a sequence of sub-frames:
// the start (both actions revealed), one frame per recorded tick, and the end. HP, gold and XP change at the end of a
// turn (turn-level outcomes), so tick frames carry the pre-turn status. Nothing here is computed: every value is a
// field of the server's frames (lanegame/demo/frames.py).
import type { ChampionStatus, Game, LaneView, Seat, Tick, Tower } from './types';

export const SPEEDS = [0.5, 1, 2, 4] as const;
export type Speed = (typeof SPEEDS)[number];
export const BASE_FRAME_MS = 420;            // one sub-frame at 1x, when frames are not stepped
export const TICK_MS = 1100;                 // one stepped frame at 1x when ticks play on their own

export type Sub = 'position' | 'start' | 'tick' | 'end';

export interface Frame {
  p: number;
  sub: Sub;
  label: string;
  turn?: number;
  lane: LaneView;
  cells: Record<Seat, number | null>;
  at_ward: Record<Seat, boolean>;
  status: Record<Seat, ChampionStatus>;
  towers: Record<Seat, Tower>;
  tick: Tick | null;
}

const seats: readonly Seat[] = ['A', 'B'];

export function positionFrame(game: Game, p: number): Frame {
  const pos = game.positions[p];
  const cells = { A: null, B: null } as Record<Seat, number | null>;
  const ward = { A: false, B: false } as Record<Seat, boolean>;
  for (const n of seats) {
    cells[n] = pos.champions[n].cell;
    ward[n] = pos.champions[n].at_ward;
  }
  return {
    p, sub: 'position', label: p === 0 ? 'initial position' : `after turn ${p}`, lane: pos.lane, cells, at_ward: ward,
    status: pos.champions as Record<Seat, ChampionStatus>, towers: pos.towers as Record<Seat, Tower>, tick: null,
  };
}

export function turnSubframes(game: Game, t: number): Frame[] {
  const tr = game.turns[t - 1];
  const pre = positionFrame(game, t - 1);
  const out: Frame[] = [{ ...pre, sub: 'start', label: `turn ${t}: start`, turn: t }];
  tr.ticks.forEach((tk, j) => {
    out.push({
      p: t - 1, sub: 'tick', label: `turn ${t}: tick ${j + 1}`, turn: t,
      lane: { packets: tk.packets, siege: tk.siege, contact: tk.contact },
      cells: tk.cells as Record<Seat, number | null>, at_ward: tk.at_ward as Record<Seat, boolean>,
      status: pre.status, towers: pre.towers, tick: tk,
    });
  });
  out.push({ ...positionFrame(game, t), sub: 'end', label: `turn ${t}: end`, turn: t });
  return out;
}

export const prefersReducedMotion = (): boolean =>
  typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
