"""
Tabular instantiation of POLA (Zhao et al., NeurIPS 2022, "Proximal
Learning With Opponent-Learning Awareness"), the proximal successor to
LOLA (Foerster et al., 2018), for stateless repeated matrix games. Matches
the ARAMARL Agent interface, mirroring loqa_agent.py.

POLA's core mechanism: the agent optimizes its own return while anticipating
that the opponent takes a *proximal* naive-learning step, and differentiates
its return through that anticipated step (the opponent-shaping term). In a
stateless 2x2 game each agent's policy is a single cooperation probability,
so the proximal inner problem (the opponent's regularized best response) has
a closed form -- a clipped gradient step of size eta -- and POLA's outer
proximal update reduces to the exact-LOLA lookahead gradient, evaluated at
the opponent's anticipated policy. The two therefore coincide at the same
fixed point in this setting; we implement that shared update.

As in loqa_agent.py, both value functions are estimated from observed play
(running-mean payoff estimates) rather than assumed known, and the opponent
is *modeled* as a naive gradient learner -- a modeling assumption that is
deliberately misspecified against the Q-learning DMs it faces, exactly as
LOQA models its opponent as a softmax Q-learner.
"""

import numpy as np
from numpy.random import choice

from agent import Agent


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class POLAAgent(Agent):
    """
    POLA / exact-LOLA shaping adversary for a stateless repeated matrix game.

    Actions: 0 = Cooperate, 1 = Defect. Own policy is parameterized by a
    single logit theta with P(cooperate) = sigmoid(theta). Payoff estimates
    are indexed RA[a, b] / RB[a, b] with a the opponent's (DM's) action and
    b this agent's own action -- the same convention as loqa_agent.py.

    Conventions (matching ARAMARL): in update(obs, actions, rewards, new_obs),
    actions[0]/rewards[0] are this agent's own, actions[1]/rewards[1] are the
    opponent's (the DM's).
    """

    def __init__(self, action_space, enemy_action_space, learning_rate=0.5,
                 opp_lr=1.0, epsilon=0.05):
        Agent.__init__(self, action_space)
        self.enemy_action_space = enemy_action_space
        self.lr = learning_rate      # own (outer) proximal step, on the logit
        self.eta = opp_lr            # assumed opponent naive-learning step
        self.epsilon = epsilon

        assert len(action_space) == 2 and len(enemy_action_space) == 2, \
            'POLAAgent is defined for 2x2 games'

        # Own policy logit; P(cooperate) = sigmoid(theta)
        self.theta = 0.0

        # Running-mean payoff estimates RA[a, b], RB[a, b]
        self.RA = np.zeros((2, 2))
        self.RB = np.zeros((2, 2))
        self.counts = np.zeros((2, 2))

        # Running estimate of the opponent's (DM's) cooperation probability
        self.pA = 0.5
        self.beta = 0.02             # smoothing rate for pA

    # -- internal model ----------------------------------------------------
    def _opp_lookahead(self, q):
        """
        Opponent's anticipated cooperation prob after one proximal naive step.
        For a stateless game V_A is linear in p_A, so the proximal best
        response is a clipped gradient step; we also return its derivative
        w.r.t. q (0 when the step is clipped).
        """
        # dV_A/dp_A = q (RA[0,0]-RA[1,0]) + (1-q) (RA[0,1]-RA[1,1])
        c1 = self.RA[0, 0] - self.RA[1, 0]
        c2 = self.RA[0, 1] - self.RA[1, 1]
        dVA_dp = q * c1 + (1.0 - q) * c2
        p_raw = self.pA + self.eta * dVA_dp
        if 0.0 < p_raw < 1.0:
            return p_raw, self.eta * (c1 - c2)          # interior
        return min(1.0, max(0.0, p_raw)), 0.0           # clipped

    def _lookahead_value(self, theta):
        """V_B evaluated at the opponent's anticipated policy (the POLA/LOLA
        lookahead objective J(theta))."""
        q = sigmoid(theta)
        p_prime, _ = self._opp_lookahead(q)
        VB = (p_prime * (q * self.RB[0, 0] + (1 - q) * self.RB[0, 1])
              + (1 - p_prime) * (q * self.RB[1, 0] + (1 - q) * self.RB[1, 1]))
        return VB

    def value_of(self, theta):
        return self._lookahead_value(theta)

    def grad(self):
        """Exact gradient of the lookahead objective J w.r.t. theta."""
        q = sigmoid(self.theta)
        p_prime, dp_dq = self._opp_lookahead(q)

        # partials of V_B(p_prime, q) at the anticipated opponent policy
        dVB_dq = (p_prime * (self.RB[0, 0] - self.RB[0, 1])
                  + (1 - p_prime) * (self.RB[1, 0] - self.RB[1, 1]))
        dVB_dp = (q * (self.RB[0, 0] - self.RB[1, 0])
                  + (1 - q) * (self.RB[0, 1] - self.RB[1, 1]))

        dJ_dq = dVB_dq + dVB_dp * dp_dq
        return dJ_dq * q * (1.0 - q)      # chain rule through the logit

    # -- Agent interface ----------------------------------------------------
    def act(self, obs=None):
        pC = (1 - self.epsilon) * sigmoid(self.theta) + self.epsilon / 2.0
        return choice(self.action_space, p=[pC, 1.0 - pC])

    def update(self, obs, actions, rewards, new_obs):
        b, a = actions          # own action, opponent (DM) action
        rB, rA = rewards        # own reward, opponent reward

        # Update payoff estimates (running means)
        self.counts[a, b] += 1
        c = self.counts[a, b]
        self.RA[a, b] += (rA - self.RA[a, b]) / c
        self.RB[a, b] += (rB - self.RB[a, b]) / c

        # Update estimate of the opponent's cooperation probability
        self.pA += self.beta * ((1.0 if a == 0 else 0.0) - self.pA)

        # One proximal POLA / exact-LOLA ascent step on the own logit
        self.theta = self.theta + self.lr * self.grad()


if __name__ == '__main__':
    # (1) Finite-difference check of the exact lookahead gradient.
    rng = np.random.RandomState(1)
    ag = POLAAgent([0, 1], [0, 1], learning_rate=0.5, opp_lr=0.3)
    ag.RA = rng.randn(2, 2)
    ag.RB = rng.randn(2, 2)
    ag.pA = 0.4
    ag.theta = 0.3   # interior, non-clipped point

    g = ag.grad()
    eps = 1e-6
    g_fd = (ag.value_of(ag.theta + eps) - ag.value_of(ag.theta - eps)) / (2 * eps)
    print('analytic grad:', g)
    print('finite-diff  :', g_fd)
    assert abs(g - g_fd) < 1e-6, 'gradient check failed'
    print('gradient check passed')

    # (2) Sanity: against the true Chicken payoffs, the shaping term must
    #     drive the agent to DEFECT (bully). We hand the agent the exact
    #     matrices and check the sign of the update at the uniform point.
    #     RA[a,b], a=DM action, b=own action; 0=C, 1=D.
    RA = np.array([[0.0, -2.0], [1.0, -4.0]])   # DM's reward
    RB = np.array([[0.0, 1.0], [-2.0, -4.0]])   # own reward (symmetric role)
    ag2 = POLAAgent([0, 1], [0, 1], learning_rate=0.5, opp_lr=0.3)
    ag2.RA, ag2.RB, ag2.pA, ag2.theta = RA, RB, 0.5, 0.0
    print('grad at uniform (should be < 0 => drive P(coop) down => defect):',
          ag2.grad())
    assert ag2.grad() < 0, 'expected shaping to drive defection'
    print('bully-direction check passed')
