// The geometry of the lane's movement arrows: each one is a short trail that ends just behind what moved and points
// the way it went.

export type Pt = { x: number; y: number };
/** What moved, as drawn: a token or a tower ring (a circle), or a cluster of balls (an axis-aligned box). */
export type Body = { c: Pt; r: number } | { c: Pt; w: number; h: number };

/** The space between a trail's tip and the trailing edge of what moved. */
export const GAP = 3;

/** The unit vector from p to q; null when the two are the same point. */
export function heading(p: Pt, q: Pt): Pt | null {
  const d = Math.hypot(q.x - p.x, q.y - p.y);
  return d < 1e-6 ? null : { x: (q.x - p.x) / d, y: (q.y - p.y) / d };
}

/** How far the body reaches from its centre along the unit vector u. */
function reach(b: Body, u: Pt): number {
  if ('r' in b) return b.r;
  const along = (half: number, k: number) => (Math.abs(k) > 1e-9 ? half / Math.abs(k) : Infinity);
  return Math.min(along(b.w / 2, u.x), along(b.h / 2, u.y));
}

/** A trail of length L pointing along u, its tip GAP px behind the body's trailing edge. */
export function trail(u: Pt, b: Body, L: number): { tip: Pt; tail: Pt } {
  const d = reach(b, u) + GAP, tip = { x: b.c.x - u.x * d, y: b.c.y - u.y * d };
  return { tip, tail: { x: tip.x - u.x * L, y: tip.y - u.y * L } };
}
