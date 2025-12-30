from typing import Tuple
from .utils import clip, bernoulli

# Actions 
SP, SH, F = 0, 1, 2 
ACTIONS = [SP, SH, F] 

# State screenshot init, e.g. w, m_y, m_o, v_y, v_o, g
state_ss = Tuple[int, int, int, int, int, int]

# Indicator function to be used in rewards
def I(cond: bool) -> int:
    return 1 if cond else 0


# parameters for lane
class LaneParams: 
    def __init__(
        self, 
        T = 40,
        G = 3, #max of 3 for gold parameter
        eps = 0.3,
        p_v=0.6, 
        b_plates = 0.6,
        b_deny = 0.4, 
        b_deny_bonus = 0.2, 
        b_crash = 0.8, 
        q0 = 0.05,
        q1 = 0.2,
        q2 = 0.15,
        L = 5
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

# lane environment
class LaneEnv: 
    """
    A 2-player Markov game environment for laning.

    Internal state (global / from YOU perspective):
      w   in {-2,-1,0,1,2} : wave position 
      m_self in {0,1,2}       : your minion stack
      m_opp in {0,1,2}       : opponent minion stack
      v_self, v_opp in {0,1}    : each player's vision (Bernoulli per turn)
      g in [-G,G]          : advantage (you - opp), default G is 3 
      t                    : # waves as time step
    """

    # init
    def __init__(self, params: LaneParams): 
        self.p = params
        self.reset()

    # State screenshots for each player
    def ss_you(self) -> state_ss: 
        return (self.w, self.m_self, self.m_opp, self.v_self, self.v_opp, self.g)
    def ss_opp(self) -> state_ss: 
        return (-self.w, self.m_opp, self.m_self, self.v_opp, self.v_self, -self.g)
    
    #--------------------- Markov model ------------------------

    # delta w
    def delta_w(self, a: int) -> int: 
        if a == SP:
            return 1
        elif a == SH:
            return 2    
        elif a == F: 
            return -1
        raise ValueError("INVALID ACTION, NOT SP, SH, F")

    # M[a,b] interaction matrix
    def payoff_matrix(self, a: int, b: int) -> float: 
        e = self.p.eps

        # if we get the "counter matchup"
        interactions = ( 
            (SP, SH), 
            (SH, F), 
            (F, SP)
        )

        # if on diagonal 
        if a == b :
            return 0 
        
        # since the table is symmetric we can take a shortcut like this instead of checking every condition in the matrix
        # if we get the "counter matchup" "+"" otherwise "-""
        return + e if (a,b) in interactions else - e 

    # gank penalty, gank_t 
    def gank_penalty(self, w: int, v: int, a: int) -> float: 
        base = self.p.q0 + self.p.q1 * (w == 2) + self.p.q2 * (a == SH)
        base = max(0.0, min(1.0, base))   
        return (1 - v) * base
    
    # wave update depending on our action
    def wave_update(self, m: int, a: int, w_next: int) -> int: 
        if a == SP and w_next < 2:
            return min(2, m + 1)
        if a == SH and w_next ==2 :
            return 0
        if a == F :
            return max(0, m-1)
        else :
            return m

    # reset environment 
    def reset(self) -> None : 
        self.w = 0
        self.m_self = 0
        self.m_opp = 0
        self.v_self = 0
        self.v_opp = 0
        self.g = 0
        self.t = 0

    # reward function
    def reward(self, w: int, m: int, v: int, a_self: int, a_opp: int, w_next: int) -> float : 
        # positive: init > Matrix payoff > B_plates > B_deny (2 terms) > B_crash 
        r = 0.0 
        r += self.payoff_matrix(a_self, a_opp) 
        r += self.p.b_plates * I(a_self == SH) * I(w >= 1)
        r += self.p.b_deny * I(a_self == F) * I(w <= -1) + self.p.b_deny_bonus * I(a_self == F) * I(w <= -1) * I(a_opp in (SP, SH))
        r += self.p.b_crash * I(w_next == 2) * I(m == 2)

        # negative: gank penalty
        p = self.gank_penalty(self, w, v, a_self)
        gank = bernoulli(p) 
        r -= self.p.L * gank

        return r 

    # game step
    def step(self, a_self: int, a_opp: int) : 
        """
        markov game step:

        Given current state s_t and actions (a_y, a_o):
          1) compute w_{t+1}
          2) compute rewards r_self, r_opp using each player's perspective
          3) update gd, g_{t+1} 
          4) update stacks, m_{t+1}
          5) sample vision, v_{t+1} 
        """
        # each wave for both players
        w_next = self.w + self.delta_w(a_self) - self.delta_w(a_opp)
        w_next = max(-2, min(2, w_next)) 

        w_self, w_opp = self.w, -self.w 
        w_next, w_next_opp = w_next, -w_next 

        # each reward for both players
        r_self = self.reward(w_self, self.m_self, self.v_self, a_self, a_opp, w_next)
        r_opp = self.reward(w_opp, self.m_opp, self.v_opp, a_opp, a_self, w_next_opp)

        # gold update
        g_next = self.g + (r_self - r_opp)
        g_next = max(-self.p.G, min(self.p.G, g_next))

        # wave update
        self.m_self = self.update_stack(self.m_self, a_self, w_next)
        self.m_opp = self.update_stack(self.m_opp, a_opp, w_next_opp)

        # vision and updates
        self.w = w_next 
        self.g = g_next 
        self.v_self = bernoulli(self.p.p_v)
        self.v_opp = bernoulli(self.p.p_v)
        self.t += 1 

        # flag to check if we're done our episodes 
        flag = self.t >= self.p.T

        # return relevant screenshots, rewards, flag
        return (self.ss_you(), self.ss_opp()), (r_self, r_opp), flag
 

