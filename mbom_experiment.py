"""
Alternative-DM comparison mirroring MBOM's own evaluation protocol
(Yu et al., NeurIPS 2022, Sec. 5): a decision-maker faces three opponent
types -- fixed policy, naive learner, reasoning learner -- in a competitive
(zero-sum) game, here iterated Matching Pennies.

The DM (matcher) receives +1 when its action equals the opponent's, -1
otherwise; the opponent (mismatcher) receives the negation. The equilibrium
value for the DM is 0, so a positive mean reward measures how much the DM
exploits the opponent.

We compare three DMs in the same seat:
  * IndQ   -- opponent-unaware tabular Q-learner (naive baseline);
  * L3Dir  -- our type-based DM with Bayesian opponent averaging
              (Level3QAgentMixDir), the direct analog of MBOM;
  * MBOM   -- the recent model-based recursive-reasoning method.

Matching Pennies is competitive, so MBOM's recursive imagination is
non-trivial (unlike in Chicken, where iterated best response collapses).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent import IndQLearningAgent, FPLearningAgent
from mbom_agent import MBOMAgent
from typebased_dm import TypeBasedDM

N_EXP = 20
T = 30000          # repeated single-shot steps per seed
GAMMA = 0.96
LR = 0.1
EPS = 0.05
P_FIX = 0.75       # fixed opponent plays action 0 with this prob (exploitable)
ACTIONS = [0, 1]


class FixedAgent:
    """Fixed stochastic policy: plays action 0 with probability p."""
    def __init__(self, action_space, p):
        self.action_space = action_space
        self.p = p

    def act(self, obs=None):
        return 0 if np.random.rand() < self.p else 1

    def update(self, *args, **kwargs):
        pass


def make_dm(kind):
    if kind == 'IndQ':
        return IndQLearningAgent(ACTIONS, n_states=1, learning_rate=LR,
                                 epsilon=EPS, gamma=GAMMA)
    if kind == 'TypeBased':
        return TypeBasedDM(ACTIONS, ACTIONS, n_states=1,
                           learning_rate=LR, epsilon=EPS, gamma=GAMMA)
    if kind == 'MBOM':
        return MBOMAgent(ACTIONS, ACTIONS, n_states=1, learning_rate=LR,
                         epsilon=EPS, gamma=GAMMA)
    raise ValueError(kind)


def make_opponent(kind):
    # Opponent is the "mismatcher"; it maximizes its own reward from its seat.
    if kind == 'fixed':
        return FixedAgent(ACTIONS, P_FIX)
    if kind == 'naive':
        return IndQLearningAgent(ACTIONS, n_states=1, learning_rate=LR,
                                 epsilon=EPS, gamma=GAMMA)
    if kind == 'reasoning':
        return FPLearningAgent(ACTIONS, ACTIONS, n_states=1, learning_rate=LR,
                               epsilon=EPS, gamma=GAMMA)
    raise ValueError(kind)


def run(dm_kind, opp_kind):
    dm_rewards = []
    for seed in range(N_EXP):
        np.random.seed(seed)
        dm = make_dm(dm_kind)
        opp = make_opponent(opp_kind)
        s = 0
        rs = []
        for _ in range(T):
            a = dm.act(s)        # DM action (matcher)
            b = opp.act(s)       # opponent action (mismatcher)
            r_dm = 1.0 if a == b else -1.0
            r_opp = -r_dm
            dm.update(s, (a, b), (r_dm, r_opp), s)
            opp.update(s, (b, a), (r_opp, r_dm), s)
            rs.append(r_dm)
        dm_rewards.append(rs)
    return np.asarray(dm_rewards)


def moving_average(a, n=200):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    import os
    os.makedirs('img', exist_ok=True)
    opponents = ['fixed', 'naive', 'reasoning']
    dms = ['IndQ', 'TypeBased', 'MBOM']
    colors = {'IndQ': 'tab:gray', 'TypeBased': 'tab:blue', 'MBOM': 'tab:red'}
    labels = {'IndQ': 'IndQ (baseline)',
              'TypeBased': 'Type-based DM (ours)', 'MBOM': 'MBOM'}

    results = {}
    for opp in opponents:
        for dm in dms:
            results[(dm, opp)] = run(dm, opp)

    tail = 10000
    print('Mean DM reward over final %d steps (equilibrium value = 0; '
          'higher = more exploitation):\n' % tail)
    header = 'DM \\ opponent   ' + ''.join(f'{o:>14}' for o in opponents)
    print(header)
    for dm in dms:
        row = f'{labels[dm]:<16}'
        for opp in opponents:
            m = results[(dm, opp)][:, -tail:].mean()
            sd = results[(dm, opp)][:, -tail:].mean(axis=1).std()
            row += f'  {m:+.3f}±{sd:.2f}'
        print(row)

    # 3-panel trajectory figure, one panel per opponent type
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    titles = {'fixed': 'vs.\\ fixed policy', 'naive': 'vs.\\ naive learner',
              'reasoning': 'vs.\\ reasoning learner'}
    for ax, opp in zip(axes, opponents):
        for dm in dms:
            mean_curve = results[(dm, opp)].mean(axis=0)
            ax.plot(moving_average(mean_curve), color=colors[dm],
                    label=labels[dm], lw=1.5)
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_title(titles[opp].replace('\\', ''))
        ax.set_xlabel('t')
    axes[0].set_ylabel('DM reward')
    axes[0].legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('img/mbom_matching_pennies.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('\nfigure saved: img/mbom_matching_pennies.png')
