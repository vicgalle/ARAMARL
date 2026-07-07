"""
Tabular instantiation of LOQA (Aghajohari et al., ICLR 2024,
"LOQA: Learning with Opponent Q-Learning Awareness") for stateless
repeated matrix games, matching the ARAMARL Agent interface.

LOQA's core mechanism: the agent models its opponent as acting softmax over
the opponent's action-values, estimates those action-values from the
opponent's observed rewards, and performs policy-gradient ascent on its own
return *differentiating through the opponent's softmax policy* (the
opponent-shaping term), where the dependence of the opponent's Q-values on
the agent's policy is estimated via REINFORCE.

In a stateless repeated 2x2 game both expectations can be computed exactly:
 * For a stateless Q-learner at its fixed point,
   Q_A(a) - Q_A(a') = E_b[r_A(a,b)] - E_b[r_A(a',b)], so modeling the
   opponent as softmax over expected immediate rewards under pi_B is exact
   up to the action-independent constant that softmax ignores.
 * The REINFORCE estimator grad_theta Q_A(a) = E_b[r_A(a,b) grad log pi_B(b)]
   equals the exact gradient of sum_b pi_B(b) rhat_A(a,b).

Hence this implementation replaces LOQA's Monte-Carlo estimators by their
exact tabular counterparts; everything else (objective, both gradient paths,
softmax opponent model with temperature tau) follows the original method.
"""

import numpy as np
from numpy.random import choice

from agent import Agent


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


class LOQAAgent(Agent):
    """
    LOQA adversary for a stateless repeated matrix game.

    Conventions (matching ARAMARL): in update(obs, actions, rewards, new_obs),
    actions[0]/rewards[0] are the agent's own, actions[1]/rewards[1] are the
    opponent's. Payoff estimates are indexed [opponent_action, own_action].
    """

    def __init__(self, action_space, enemy_action_space, learning_rate=0.05,
                 epsilon=0.05, tau=1.0):
        Agent.__init__(self, action_space)
        self.enemy_action_space = enemy_action_space
        self.lr = learning_rate
        self.epsilon = epsilon
        self.tau = tau

        nA = len(enemy_action_space)   # opponent actions (agent A, the DM)
        nB = len(action_space)         # own actions (agent B)

        # Policy logits (the actor)
        self.theta = np.zeros(nB)

        # Running-mean payoff estimates rhat_A[a, b], rhat_B[a, b]
        self.RA = np.zeros((nA, nB))
        self.RB = np.zeros((nA, nB))
        self.counts = np.zeros((nA, nB))

    # -- internal model ----------------------------------------------------
    def _pi_B(self):
        return softmax(self.theta)

    def _model(self, theta):
        """Return (pi_B, QA_hat, pi_A_hat, V_B) for given logits."""
        pi_B = softmax(theta)
        QA = self.RA @ pi_B                      # QA[a] = sum_b pi_B(b) RA[a,b]
        pi_A = softmax(QA / self.tau)            # opponent modeled as softmax-over-Q
        V = pi_A @ (self.RB @ pi_B)              # modeled expected own reward
        return pi_B, QA, pi_A, V

    def value_of(self, theta):
        return self._model(theta)[3]

    def grad(self):
        """Exact gradient of V_B w.r.t. theta, through both paths."""
        pi_B, QA, pi_A, _ = self._model(self.theta)
        nB = len(self.theta)

        # Jacobian of pi_B: J[b, k] = pi_B[b] (delta_bk - pi_B[k])
        J = np.diag(pi_B) - np.outer(pi_B, pi_B)

        # dQA[a]/dtheta[k] = sum_b RA[a, b] J[b, k]
        dQA = self.RA @ J

        # dpi_A[a]/dtheta[k] (softmax with temperature tau)
        dpiA = (pi_A[:, None] / self.tau) * (dQA - (pi_A @ dQA)[None, :])

        # Path 1 (opponent shaping): d pi_A . (RB pi_B)
        u = self.RB @ pi_B                       # u[a] = sum_b pi_B(b) RB[a,b]
        g_shape = dpiA.T @ u

        # Path 2 (direct policy gradient): pi_A . RB dpi_B
        g_direct = (pi_A @ self.RB) @ J

        return g_shape + g_direct

    # -- Agent interface ----------------------------------------------------
    def act(self, obs=None):
        pi = (1 - self.epsilon) * self._pi_B() \
            + self.epsilon / len(self.action_space)
        return choice(self.action_space, p=pi)

    def update(self, obs, actions, rewards, new_obs):
        b, a = actions          # own action, opponent action
        rB, rA = rewards        # own reward, opponent reward

        # Update payoff estimates (running means)
        self.counts[a, b] += 1
        c = self.counts[a, b]
        self.RA[a, b] += (rA - self.RA[a, b]) / c
        self.RB[a, b] += (rB - self.RB[a, b]) / c

        # One step of exact policy gradient ascent (LOQA update)
        self.theta = self.theta + self.lr * self.grad()


if __name__ == '__main__':
    # Finite-difference check of the exact gradient.
    rng = np.random.RandomState(0)
    ag = LOQAAgent([0, 1], [0, 1], learning_rate=0.1, tau=0.7)
    ag.RA = rng.randn(2, 2)
    ag.RB = rng.randn(2, 2)
    ag.theta = rng.randn(2)

    g = ag.grad()
    eps = 1e-6
    g_fd = np.zeros_like(g)
    for k in range(len(ag.theta)):
        tp = ag.theta.copy(); tp[k] += eps
        tm = ag.theta.copy(); tm[k] -= eps
        g_fd[k] = (ag.value_of(tp) - ag.value_of(tm)) / (2 * eps)

    print('analytic:', g)
    print('finite-diff:', g_fd)
    assert np.allclose(g, g_fd, atol=1e-6), 'gradient check failed'
    print('gradient check passed')
