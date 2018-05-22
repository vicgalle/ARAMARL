import numpy as np

from engine import RMG
from agent import RandomAgent



batch_size = 1
max_steps = 20

# Reward matrix for the Iterated Prisoner's Dilemma
ipd_rewards = np.array([[-1., 0.], [-3., -2.]])

env = RMG(max_steps=max_steps, payouts=ipd_rewards, batch_size=batch_size)
env.reset()

possible_actions = [0, 1]  # Cooperate or Defect
cooperator, defector = RandomAgent(possible_actions, p=0.9), RandomAgent(possible_actions, p=0.1)

# Stateless interactions (agents do not have memory)
s = None

n_iter = 1000
for i in range(n_iter):

    # A full episode:
    done = False

    while not done:

        # Agents decide
        a0 = cooperator.act()
        a1 = defector.act()

        # World changes
        new_s, (r0, r1), done, _ = env.step(([a0], [a1]))

        # Agents learn
        cooperator.update(s, (a0, a1), (r0, r1), new_s )
        defector.update(s, (a1, a0), (r1, r0), new_s )

        s = new_s
        print(r0, r1)

    env.reset()
