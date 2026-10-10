from typing import Tuple
from .utils import bernoulli

# Actions
SP, SH, F = 0, 1, 2
ACTIONS = [SP, SH, F]

# Observation/state tuple initializatoin, 
# (w_self, m_self, m_opp, v_self, v_opp, g_self)
state_ss = Tuple[int, int, int, int, int, int]


def I(cond: bool) -> int:
    """Indicator: 1 if cond else 0."""
    return 1 if cond else 0


class LaneParams:
    """Parameters for LaneEnv."""
    def __init__(
        self,
        T=40,
        G=3,
        eps=0.3,
        p_v=0.6,
        b_plates=0.6,
        b_deny=0.4,
        b_deny_bonus=0.2,
        b_crash=0.8,
        q0=0.05,
        q1=0.20,
        q2=0.15,
        L=5.0,
    ):
        self.T = T
        self.G = G
        self.eps = eps
        self.p_v = p_v

        self.b_plates = b_plates
        self.b_deny = b_deny
        self.b_deny_bonus = b_deny_bonus
        self.b_crash = b_crash

        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.L = L


class LaneEnv:
    """
    2-player Markov game environment.

    Global state (from your perspective):
      w   in {-2,-1,0,1,2}
      m_self, m_opp in {0,1,2}
      v_self, v_opp in {0,1}
      g in [-G, G]     (gd = self - opp)
      t timestep
    """

    def __init__(self, params: LaneParams):
        self.p = params
        self.reset()

    # ---- observations ----
    def ss_you(self) -> state_ss:
        return (self.w, self.m_self, self.m_opp, self.v_self, self.v_opp, self.g)

    def ss_opp(self) -> state_ss:
        # mirrored for enemy
        return (-self.w, self.m_opp, self.m_self, self.v_opp, self.v_self, -self.g)

    def delta_w(self, a: int) -> int:
        if a == SP:
            return +1
        if a == SH:
            return +2
        if a == F:
            return -1
        raise ValueError("Invalid action (must be SP, SH, or F).")

    def payoff_matrix(self, a: int, b: int) -> float:
        """
        Like rock paper scissors, e.g. 
          SH beats SP
          F beats SH
          SP beats F
        """
        e = self.p.eps
        if a == b:
            return 0.0

        wins = {(SH, SP), (F, SH), (SP, F)}
        return +e if (a, b) in wins else -e

    def p_gank(self, w: int, v: int, a: int) -> float:
        base = self.p.q0 + self.p.q1 * I(w == 2) + self.p.q2 * I(a == SH)
        base = max(0.0, min(1.0, base))
        return (1 - v) * base

    def update_stack(self, m: int, a: int, w_next: int) -> int:
        """
        Stack update (in player's own perspective):
          SP: build until 2 (if not already crashed)
          SH: if crash at w_next==2, reset to 0
          F : trim by 1
        """
        if a == SP and w_next < 2:
            return min(2, m + 1)
        if a == SH and w_next == 2:
            return 0
        if a == F:
            return max(0, m - 1)
        return m

    # ---- reset ----
    def reset(self):
        self.w = 0
        self.m_self = 0
        self.m_opp = 0
        self.v_self = bernoulli(self.p.p_v)
        self.v_opp = bernoulli(self.p.p_v)
        self.g = 0
        self.t = 0
        return self.ss_you(), self.ss_opp()

    # ---- reward ----
    def reward(self, w: int, m: int, v: int, a_self: int, a_opp: int, w_next: int) -> float:
        """
        r = M[a,b]
            + b_plates * 1[a=SH]*1[w>=1]
            + b_deny   * 1[a=F ]*1[w<=-1]
            + b_deny_bonus * 1[a=F]*1[w<=-1]*1[opp in {SP,SH}]
            + b_crash  * 1[w_next=2]*1[m=2]
            - L * gank,  gank ~ Bernoulli(p_gank(w,v,a))
        """
        r = 0.0
        r += self.payoff_matrix(a_self, a_opp)

        r += self.p.b_plates * I(a_self == SH) * I(w >= 1)

        r += self.p.b_deny * I(a_self == F) * I(w <= -1)
        r += self.p.b_deny_bonus * I(a_self == F) * I(w <= -1) * I(a_opp in (SP, SH))

        r += self.p.b_crash * I(w_next == 2) * I(m == 2)

        gank = bernoulli(self.p_gank(w, v, a_self))
        r -= self.p.L * gank

        return r

    # ---- step ----
    def step(self, a_self: int, a_opp: int):
        # wave transition (global, from your perspective)
        raw_w_next = self.w + self.delta_w(a_self) - self.delta_w(a_opp)
        w_next = max(-2, min(2, raw_w_next))

        # set perspectives 
        w_self = self.w
        w_opp = -self.w
        w_next_self = w_next
        w_next_opp = -w_next

        # set rewards
        r_self = self.reward(w_self, self.m_self, self.v_self, a_self, a_opp, w_next_self)
        r_opp = self.reward(w_opp, self.m_opp, self.v_opp, a_opp, a_self, w_next_opp)

        # set gold
        raw_g_next = self.g + (r_self - r_opp)
        g_next = max(-self.p.G, min(self.p.G, raw_g_next))

        # set wave stacks
        self.m_self = self.update_stack(self.m_self, a_self, w_next_self)
        self.m_opp = self.update_stack(self.m_opp, a_opp, w_next_opp)

        # apply updates to wave, gold, vision
        self.w = w_next
        self.g = g_next
        self.v_self = bernoulli(self.p.p_v)
        self.v_opp = bernoulli(self.p.p_v)

        self.t += 1
        # done is our flag for episode termination
        done = (self.t >= self.p.T)

        return (self.ss_you(), self.ss_opp()), (r_self, r_opp), done
