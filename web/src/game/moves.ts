// The lane's arrows: how each champion and each side's leading wave moved over the turn that produced the position on
// screen, from the server's per-turn moves (lanegame/demo/frames.moves). A turn's start and tick frames animate the
// motion themselves, so they carry no arrows, and neither does the initial position.
import type { Frame } from './frames';
import type { ChampionMove, Seat, Turn, WaveMove } from './types';
import { SEATS } from './types';

export type Arrow = { seat: Seat } & (({ what: 'champion' } & ChampionMove) | ({ what: 'wave' } & WaveMove));

/** The arrows of frame `f`, in the order champion A, wave A, champion B, wave B; none when its turn is not known. */
export function arrowsFor(f: Frame, turns: Turn[]): Arrow[] {
  const t = f.sub === 'end' ? f.turn : f.sub === 'position' && f.p >= 1 ? f.p : undefined;
  const tr = t === undefined ? undefined : turns[t - 1];
  if (!tr) return [];
  const out: Arrow[] = [];
  for (const seat of SEATS) {
    const m = tr.moves[seat];
    if (m?.champion) out.push({ seat, what: 'champion', ...m.champion });
    if (m?.wave) out.push({ seat, what: 'wave', ...m.wave });
  }
  return out;
}
