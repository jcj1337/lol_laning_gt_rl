// A movement arrow is a short trail of a fixed length that ends 3 px behind the trailing edge of what moved, on the
// line through its centre, and points the way it moved.
import { describe, expect, it } from 'vitest';
import type { Pt } from './trail';
import { heading, trail } from './trail';

const dist = (p: Pt, q: Pt) => Math.hypot(q.x - p.x, q.y - p.y);
const unit = (x: number, y: number): Pt => heading({ x: 0, y: 0 }, { x, y })!;
/** Where p lies against an axis-aligned box: 1 on its edge, below 1 inside, above 1 outside. */
const edge = (p: Pt, c: Pt, w: number, h: number) => Math.max(Math.abs(p.x - c.x) / (w / 2), Math.abs(p.y - c.y) / (h / 2));

describe('heading', () => {
  it('is the unit vector from the start to the end, and null when they are the same point', () => {
    expect(heading({ x: 1, y: 2 }, { x: 4, y: -2 })).toEqual({ x: 0.6, y: -0.8 });
    expect(heading({ x: 5, y: 5 }, { x: 5, y: 5 })).toBeNull();
  });
});

describe('trail', () => {
  const L = 36;

  it('ends 3 px outside a circle, straight behind its centre, for any direction', () => {
    const c = { x: 200, y: 120 }, r = 14;
    for (const u of [unit(1, 0), unit(0, -1), unit(-1, 1), unit(0.906, -0.423)]) {
      const { tip, tail } = trail(u, { c, r }, L);
      expect(dist(tip, c)).toBeCloseTo(r + 3, 9);
      expect(tip.x - c.x).toBeCloseTo(-u.x * (r + 3), 9);
      expect(tip.y - c.y).toBeCloseTo(-u.y * (r + 3), 9);
      expect(dist(tail, tip)).toBeCloseTo(L, 9);
      expect(tip.x - tail.x).toBeCloseTo(u.x * L, 9);
      expect(tip.y - tail.y).toBeCloseTo(u.y * L, 9);
    }
  });

  it('ends outside a box for a diagonal direction, 3 px back along the direction from the edge it leaves through', () => {
    const c = { x: 100, y: 50 }, w = 60, h = 30;
    for (const u of [unit(3, -4), unit(-0.906, 0.423), unit(1, 1)]) {
      const { tip, tail } = trail(u, { c, w, h }, L);
      expect(edge(tip, c, w, h)).toBeGreaterThan(1);
      expect(edge({ x: tip.x + 3 * u.x, y: tip.y + 3 * u.y }, c, w, h)).toBeCloseTo(1, 9);
      expect(Math.sign(tip.x - c.x)).toBe(-Math.sign(u.x));
      expect(Math.sign(tip.y - c.y)).toBe(-Math.sign(u.y));
      expect(dist(tail, tip)).toBeCloseTo(L, 9);
    }
  });

  it('ends 3 px beyond the middle of the trailing side for a direction along an axis of the box', () => {
    const c = { x: 100, y: 50 }, w = 60, h = 30;
    expect(trail(unit(1, 0), { c, w, h }, L)).toEqual({ tip: { x: 67, y: 50 }, tail: { x: 31, y: 50 } });
    expect(trail(unit(0, -1), { c, w, h }, L)).toEqual({ tip: { x: 100, y: 68 }, tail: { x: 100, y: 104 } });
  });
});
