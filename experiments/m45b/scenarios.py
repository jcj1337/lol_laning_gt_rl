"""M4.5b start states registered with the viewer (`viewer.session.SCENARIOS`), so that every M4.5b recording replays
from its Setup (configuration, scenario, gap "equal", seed). Each start already contains its power gap and levels
(`states`); the viewer's own gap must stay "equal" for them.

    m45b_freeze_<L>_<gap>_h<held>   states.freeze_state(level L, gap, held HP, A full HP)
    m45b_trade_L<L>_w<wave>         states.trade_bench(level L, enemy wave HP, own cover, equal power)
    m45b_vision_<situation>_L4      states.vision_state(situation, level 4, equal power)
    m45b_zoning_neutral_L6_large    states.zoning_state("neutral", level 6, A x1.76)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from . import states as S

STARTS: Dict[str, Tuple[str, Callable]] = {}


def _reg(name: str, desc: str, fn: Callable) -> None:
    STARTS[name] = (desc, fn)


for _L, _g, _h in ((10, "large", 12), (10, "large", 5), (4, "large", 12), (4, "equal", 12), (10, "equal", 12)):
    _reg(f"m45b_freeze_{_L}_{_g}_h{_h}",
         f"M4.5b freeze: A ({_g}{' x1.76' if _g == 'large' else ''}, level {_L}) holds B's {_h}-HP wave at A's -1, "
         f"A's 8-HP front in its tower cell, B walked up to its +1.",
         (lambda L, g, h: (lambda cfg: S.freeze_state(cfg, L, g, h, 8, (1.0, 1.0))))(_L, _g, _h))
for _L, _w in ((2, 24), (2, 8), (18, 24)):
    _reg(f"m45b_trade_L{_L}_w{_w}",
         f"M4.5b trade bench: level {_L}, equal power; A with its 8-HP front at A's -1 against B standing in its "
         f"{_w}-HP wave at the lane cell.",
         (lambda L, w: (lambda cfg: S.trade_bench(cfg, L, w, "cover", "equal")))(_L, _w))
for _s in ("neutral", "forward", "presiege", "frozen_on"):
    _reg(f"m45b_vision_{_s}_L4", f"M4.5b Vision situation '{_s}' at level 4, equal power: "
                                 f"{S.VISION_SITUATIONS[_s][3]}.",
         (lambda s: (lambda cfg: S.vision_state(cfg, s, 4, "equal")))(_s))
_reg("m45b_zoning_neutral_L6_large", "M4.5b zoning: neutral lane at level 6, A x1.76 attack power.",
     lambda cfg: S.zoning_state(cfg, "neutral", 6, "large"))

__all__ = ["STARTS"]
