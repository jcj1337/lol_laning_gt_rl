// Aliases of the generated API types (src/api/schema.d.ts) and the client-side game shape. The browser computes no
// game rule: every number it shows comes from these shapes, which the Python server produced.
import type { components } from '../api/schema';

type S = components['schemas'];

export type Status = S['Status'];
export type ModelCard = S['ModelCard'];
export type View = S['View'];
export type Position = S['Position'];
export type Turn = S['Turn'];
export type Tick = S['Tick'];
export type Outcome = S['Outcome'];
export type CodeParts = S['CodeParts'];
export type Menu = S['Menu'];
export type MenuChoice = S['MenuChoice'];
export type MenuSingle = S['MenuSingle'];
export type SeatPublic = S['SeatPublic'];
export type Seeds = S['Seeds'];
export type Marks = S['Marks'];
export type ChampionStatus = S['ChampionStatus'];
export type Tower = S['Tower'];
export type LaneView = S['LaneView'];
export type BeliefNow = S['BeliefNow'];
export type RecordingBeliefs = S['RecordingBeliefs'];
export type Recording = S['Recording'];
export type History = S['History'];
export type TurnResult = S['TurnResult'];
export type CreateResponse = S['CreateResponse'];
export type StopResponse = S['StopResponse'];
export type GameBundle = S['GameBundle'];
export type BundleMeta = S['BundleMeta'];
export type ReplayEntry = S['ReplayEntry'];
export type ModelId = S['CreateRequest']['opponent'];
export type BeliefMode = S['CreateRequest']['belief'];
export type Kind = S['Provenance']['kind'];

export type Seat = 'A' | 'B';
export const SEATS: readonly Seat[] = ['A', 'B'];
export const other = (n: Seat): Seat => (n === 'A' ? 'B' : 'A');

/** What the client knows about a game: a bot game arrives whole (a GameBundle), a human game grows one resolved
 * turn at a time from the server's turn results, and never holds an unresolved turn. */
export interface GameMeta {
  provenance: S['Provenance'];
  seats: SeatPublic[];
  seeds: Seeds;
  marks: Marks;
  belief_mode: BeliefMode;
  turn_cap: number;
  status?: S['RecordStatus'];
  lead_A?: number;
  config?: Record<string, string>;
  policy_rule?: string;
}

export interface Game {
  name: string;
  label: string;
  meta: GameMeta;
  positions: Position[];
  turns: Turn[];
  beliefs: RecordingBeliefs | null;
  recording?: Recording;
  lane_exit_position: number;
  read_position: number;
}

export function gameFromBundle(b: GameBundle): Game {
  return {
    name: b.name, label: b.label, meta: b.meta, positions: b.positions, turns: b.turns, beliefs: b.beliefs ?? null,
    recording: b.recording, lane_exit_position: b.lane_exit_position, read_position: b.read_position,
  };
}

export type Names = Record<Seat, string>;
