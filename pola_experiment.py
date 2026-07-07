"""
Head-to-head experiments: level-k Q-learners (ARAMARL) vs a POLA adversary
(proximal LOLA; Zhao et al., NeurIPS 2022) on the iterated Chicken game.
Same protocol as loqa_experiment.py / Experiment2-C.ipynb: stateless
repeated game, 20 steps/episode x 1000 episodes, 10 seeds, moving-average-100.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from engine import RMG
from agent import (IndQLearningAgent, FPLearningAgent, Level2QAgent,
                   Level3QAgentMixDir)
from pola_agent import POLAAgent

N_EXP = 10
MAX_STEPS = 20
N_ITER = 1000
GAMMA = 0.96

# Chicken payoffs, same convention as loqa_experiment.py:
# r0 = payout_mat[a1][a0], i.e. M[opponent][own]
CHICKEN = np.array([[0., 1.], [-2., -4.]])

ACTIONS = [0, 1]


def make_dm(kind):
    if kind == 'L0':
        return IndQLearningAgent(ACTIONS, n_states=1, learning_rate=0.1,
                                 epsilon=0.05, gamma=GAMMA)
    if kind == 'L1':
        return FPLearningAgent(ACTIONS, ACTIONS, n_states=1,
                               learning_rate=0.1, epsilon=0.05, gamma=GAMMA)
    if kind == 'L2':
        return Level2QAgent(ACTIONS, ACTIONS, n_states=1,
                            learning_rate=0.1, epsilon=0.05, gamma=GAMMA)
    if kind == 'L3mix':
        return Level3QAgentMixDir(ACTIONS, ACTIONS, n_states=1,
                                  learning_rate=0.1, epsilon=0.05, gamma=GAMMA)
    raise ValueError(kind)


def run_matchup(dm_kind, lr=0.5, opp_lr=1.0, eps=0.05):
    r0ss, r1ss = [], []
    for n in range(N_EXP):
        np.random.seed(n)
        env = RMG(max_steps=MAX_STEPS, payouts=CHICKEN, batch_size=1)
        env.reset()

        dm = make_dm(dm_kind)
        adversary = POLAAgent(ACTIONS, ACTIONS, learning_rate=lr,
                              opp_lr=opp_lr, epsilon=eps)
        s = 0
        r0s, r1s = [], []
        for _ in range(N_ITER):
            done = False
            while not done:
                a0 = dm.act(s)
                a1 = adversary.act(s)
                _, (r0, r1), done, _ = env.step(([a0], [a1]))
                r0, r1 = float(r0[0]), float(r1[0])
                dm.update(s, (a0, a1), (r0, r1), s)
                adversary.update(s, (a1, a0), (r1, r0), s)
                r0s.append(r0)
                r1s.append(r1)
            env.reset()
        r0ss.append(r0s)
        r1ss.append(r1s)
    return np.asarray(r0ss), np.asarray(r1ss)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_matchup(r0ss, r1ss, fname):
    plt.style.use('ggplot')
    plt.figure(figsize=(6.4, 4.8))
    plt.axis([0, MAX_STEPS * N_ITER, -4.5, 1.5])
    for i in range(N_EXP):
        plt.plot(moving_average(r0ss[i], 100), 'b', alpha=0.05)
        plt.plot(moving_average(r1ss[i], 100), 'r', alpha=0.05)
    plt.plot(moving_average(r0ss.mean(axis=0), 100), 'b', alpha=0.5)
    plt.plot(moving_average(r1ss.mean(axis=0), 100), 'r', alpha=0.5)
    plt.xlabel('t')
    plt.ylabel('R')
    custom_lines = [Line2D([0], [0], color='b'), Line2D([0], [0], color='r')]
    plt.legend(custom_lines, ['Agent A', 'Agent B'])
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('img', exist_ok=True)
    summary = []
    for kind in ['L0', 'L1', 'L2', 'L3mix']:
        r0ss, r1ss = run_matchup(kind)
        plot_matchup(r0ss, r1ss, f'img/{kind}vsPOLA_C.png')
        tail = 5000
        mA = r0ss[:, -tail:].mean()
        mB = r1ss[:, -tail:].mean()
        summary.append((kind, mA, mB))
        print(f'{kind:6s} vs POLA | mean last-{tail} reward: '
              f'DM (A) = {mA:+.3f}   POLA (B) = {mB:+.3f}')

    print('\nSummary (Chicken reference points: concede/cave = -2, '
          'bully = +1, crash = -4, coordinate = 0):')
    for kind, mA, mB in summary:
        print(f'  {kind}: A {mA:+.2f}, B {mB:+.2f}')
