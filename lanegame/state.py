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
  siege   number of this side's packets queued under the ENEMY tower (crashed,
          awaiting consumption). Their HP is irrelevant to every future
          transition, so only the count is stored.
  v       remaining turns of vision protection (0 .. vision_turns)

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

    def side(self, s: int):
        """(fx, fh, rears, siege, v) for side s (0 = A, 1 = B), own coordinates."""
        if s == 0:
            return self.ax, self.ah, self.ar, self.asg, self.va
        return self.bx, self.bh, self.br, self.bsg, self.vb

    def has_front(self, s: int) -> bool:
        return (self.ah if s == 0 else self.bh) > 0

    def mirror(self) -> "State":
        return State(self.bx, self.bh, self.br, self.bsg,
                     self.ax, self.ah, self.ar, self.asg,
                     self.vb, self.va)

    def front_gap(self) -> int | None:
        """Global distance between the two fronts (B position minus A position)."""
        if self.ah <= 0 or self.bh <= 0:
            return None
        return -self.bx - self.ax

    def describe(self) -> str:
        def side(name, fx, fh, rears, sg, v):
            front = f"front@{fx:+d} hp{fh}" if fh > 0 else "no front"
            r = ",".join(f"{x:+d}" for x in rears) or "-"
            return f"{name}[{front}; rears {r}; siege {sg}; vision {v}]"
        return (side("A", self.ax, self.ah, self.ar, self.asg, self.va) + " "
                + side("B", self.bx, self.bh, self.br, self.bsg, self.vb)
                + "  (own coordinates)")

    def to_list(self) -> list:
        return [self.ax, self.ah, list(self.ar), self.asg,
                self.bx, self.bh, list(self.br), self.bsg, self.va, self.vb]

    @classmethod
    def from_list(cls, v) -> "State":
        return cls(v[0], v[1], tuple(v[2]), v[3], v[4], v[5], tuple(v[6]), v[7], v[8], v[9])


def make_side(packets, siege: int):
    """Canonical side record from a list of [own_x, hp] packets (any order)."""
    packets = sorted(packets, key=lambda p: -p[0])
    if not packets:
        return 0, 0, (), siege
    fx, fh = packets[0]
    return fx, fh, tuple(p[0] for p in packets[1:]), siege
