"""Markov state of the laning game and its canonical encoding.

Each side's data is stored in that side's OWN-perspective coordinates
(own base = -base, own tower = -tower, enemy tower = +tower, enemy base = +base).
For player A own coordinates equal global coordinates; for player B,
own x = -(global x). With this convention the mirror symmetry
(swap players and reflect the lane) is just "swap the two side records".

Per side:
  fx, fh  leading non-sieging packet ("front"): own position and HP (fh == 0: no front)
  rears   own positions of packets behind the front, most advanced first.
          Rear packets never fight, so their HP is always one fresh wave (W).
  siege   this side's crashed minions under the ENEMY tower: HP mass ("mass"
          tower mode) or packet count ("packet" mode, original model).
  v       remaining turns of vision protection (0 .. vision_turns), counting
          the turn about to be played
  plates  plate opportunities this side can still claim on the ENEMY tower by
          applying pressure: pending opportunities of the current siege ("crash"
          plate mode, lapse when the siege ends) or the tower's remaining budget
          ("tower" mode); always 0 in the "choice" siege mode (no plates)

The decision index k is not stored in the state; values and policies are
indexed by (k, state). The spawn phase is a function of k.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple


class State(NamedTuple):
    ax: int
    ah: int
    ar: Tuple[int, ...]
    asg: int
    bx: int
    bh: int
    br: Tuple[int, ...]
    bsg: int
    va: int
    vb: int
    ap: int = 0
    bp: int = 0

    def side(self, s: int):
        """(fx, fh, rears, siege, v, plates) for side s (0 = A, 1 = B), own coordinates."""
        if s == 0:
            return self.ax, self.ah, self.ar, self.asg, self.va, self.ap
        return self.bx, self.bh, self.br, self.bsg, self.vb, self.bp

    def has_front(self, s: int) -> bool:
        return (self.ah if s == 0 else self.bh) > 0

    def siege_of(self, s: int) -> int:
        return self.asg if s == 0 else self.bsg

    def vision_of(self, s: int) -> int:
        return self.va if s == 0 else self.vb

    def plates_of(self, s: int) -> int:
        return self.ap if s == 0 else self.bp

    def mirror(self) -> "State":
        return State(self.bx, self.bh, self.br, self.bsg,
                     self.ax, self.ah, self.ar, self.asg,
                     self.vb, self.va, self.bp, self.ap)

    def front_gap(self) -> int | None:
        """Global distance between the two fronts (B position minus A position)."""
        if self.ah <= 0 or self.bh <= 0:
            return None
        return -self.bx - self.ax

    def describe(self) -> str:
        def side(name, fx, fh, rears, sg, v, pl):
            front = f"front@{fx:+d} hp{fh}" if fh > 0 else "no front"
            r = ",".join(f"{x:+d}" for x in rears) or "-"
            return f"{name}[{front}; rears {r}; siege {sg}; vision {v}; plates {pl}]"
        return (side("A", self.ax, self.ah, self.ar, self.asg, self.va, self.ap) + " "
                + side("B", self.bx, self.bh, self.br, self.bsg, self.vb, self.bp)
                + "  (own coordinates)")

    def to_list(self) -> list:
        return [self.ax, self.ah, list(self.ar), self.asg,
                self.bx, self.bh, list(self.br), self.bsg, self.va, self.vb, self.ap, self.bp]

    @classmethod
    def from_list(cls, v) -> "State":
        extra = tuple(v[10:12]) if len(v) >= 12 else (0, 0)
        return cls(v[0], v[1], tuple(v[2]), v[3], v[4], v[5], tuple(v[6]), v[7], v[8], v[9], *extra)


def make_side(packets, siege: int):
    """Canonical side record from a list of [own_x, hp] packets (any order)."""
    packets = sorted(packets, key=lambda p: -p[0])
    if not packets:
        return 0, 0, (), siege
    fx, fh = packets[0]
    return fx, fh, tuple(p[0] for p in packets[1:]), siege
