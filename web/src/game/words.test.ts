// The words of a tick frame name only what the server's tick states: where the two fronts fight (a unit, or between
// two adjacent units when the contact is a half-integer), a front wave cleared when the tick's packets_lost says a
// packet died, and a walk only when the seat lost no packet, its front changed cell and it did not crash.
import { describe, expect, it } from 'vitest';
import type { Frame } from './frames';
import type { Names, Tick } from './types';
import { tickWords } from './words';

type Packet = [number, number, boolean];
const NAMES: Names = { A: 'Self-play', B: 'Push bot' };
const LIVE: Names = { A: 'You', B: 'the opponent' };

function frame(A: Packet[], B: Packet[], contact: number | null = null, tick: Partial<Tick> = {}): Frame {
  const tk: Tick = {
    tick: 1, packets: { A, B }, siege: { A: [-2, 0], B: [2, 0] }, contact, cells: { A: -2, B: 2 }, at_ward: { A: false, B: false },
    wave_removed: { A: 0, B: 0 }, crashed: { A: 0, B: 0 }, packets_lost: { A: 0, B: 0 }, minion_fire: { A: { cause: '', units: 0 }, B: { cause: '', units: 0 } }, ...tick,
  };
  return { p: 0, sub: 'tick', label: '', lane: { packets: tk.packets, siege: tk.siege, contact: tk.contact }, cells: { A: -2, B: 2 },
    at_ward: { A: false, B: false }, status: {} as Frame['status'], towers: {} as Frame['towers'], tick: tk };
}

describe('tickWords', () => {
  it('places a fight on a unit, or between the two units of a half-integer contact, never at a base', () => {
    const before = frame([[-2, 8, true]], [[2, 8, true]]);
    expect(tickWords(before, frame([[0, 8, true]], [[0, 8, true]], 0), NAMES)).toEqual(["Self-play's wave and Push bot's wave fight at the lane."]);
    expect(tickWords(before, frame([[0, 8, true]], [[1, 8, true]], 0.5), NAMES)).toEqual(["Self-play's wave and Push bot's wave fight between the lane and Push bot's side of the lane."]);
    expect(tickWords(before, frame([[-1, 8, true]], [[0, 8, true]], -0.5), NAMES)).toEqual(["Self-play's wave and Push bot's wave fight between Self-play's side of the lane and the lane."]);
    expect(tickWords(before, frame([[1, 8, true]], [[2, 8, true]], 1.5), LIVE)).toEqual(["Your wave and the opponent's wave fight between the opponent's side of the lane and the opponent's tower."]);
    const fighting = frame([[0, 8, true]], [[1, 8, true]], 0.5);
    expect(tickWords(fighting, frame([[0, 6, true]], [[1, 5, true]], 0.5), NAMES)).toEqual(["Self-play's wave and Push bot's wave keep fighting between the lane and Push bot's side of the lane."]);
  });

  it('says a front wave is cleared when the tick lost a packet, with or without a fight, and never that it walked', () => {
    const before = frame([[-1, 8, true], [-3, 8, false]], [[2, 8, true]]);
    const after = frame([[-2, 8, true]], [[1, 8, true]], null, { packets_lost: { A: 1, B: 0 } });
    expect(tickWords(before, after, NAMES)).toEqual(["Self-play's front wave is cleared.", "Push bot's wave walks to Push bot's side of the lane."]);
    expect(tickWords(before, after, LIVE)).toEqual(['Your front wave is cleared.', "The opponent's wave walks to the opponent's side of the lane."]);
    const fight = frame([[0, 8, true]], [[1, 8, true]], 0.5, { packets_lost: { A: 0, B: 1 } });
    expect(tickWords(frame([[0, 8, true]], [[0, 2, true], [1, 8, false]], 0), fight, LIVE))
      .toEqual(["Your wave and the opponent's wave fight between the lane and the opponent's side of the lane.", "The opponent's front wave is cleared."]);
  });

  it('says a wave walks only when its front changed cell without a lost packet or a crash', () => {
    const before = frame([[-1, 8, true]], [[2, 8, true]]);
    expect(tickWords(before, frame([[0, 8, true]], [[2, 8, true]]), NAMES)).toEqual(["Self-play's wave walks to the lane."]);
    expect(tickWords(before, frame([[-1, 8, true]], [[2, 8, true]]), NAMES)).toEqual(['Nothing changes in this tick.']);
    expect(tickWords(frame([[1, 8, true], [-1, 8, false]], [[3, 8, true]]), frame([[0, 8, true]], [[3, 8, true]], null, { crashed: { A: 8, B: 0 } }), NAMES))
      .toEqual(["Self-play's wave crashes into Push bot's tower (8 HP)."]);
  });
});
