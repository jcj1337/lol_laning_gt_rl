import random
from typing import Callable, Tuple

from .env import SP, SH, F, state_ss

Policy = Callable[[state_ss], int]


def always_shove(_: state_ss) -> int:
    return SH


def freeze_if_possible(obs: state_ss) -> int:
    w, m_self, m_enemy, v_self, v_enemy, g = obs
    return F if w <= -1 else SH


def stack_then_crash(obs: state_ss) -> int:
    w, m_self, m_enemy, v_self, v_enemy, g = obs
    if m_self < 2 and w < 2:
        return SP
    return SH


def random_mixed(p_sp=0.4, p_sh=0.4, p_f=0.2) -> Policy:
    # returns a policy function with fixed probabilities
    def pi(_: state_ss) -> int:
        r = random.random()
        if r < p_sp:
            return SP
        if r < p_sp + p_sh:
            return SH
        return F
    return pi