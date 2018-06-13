"""
Some tests and usage examples for the engine module
"""

import numpy as np

from engine import RMG, AdvRw

# Now we perform some tests

batch_size = 1
max_steps = 100

# Reward matrix for the Iterated Prisoner's Dilemma
ipd_rewards = np.array([[-1., 0.], [-3., -2.]])

env = RMG(max_steps=max_steps, payouts=ipd_rewards, batch_size=batch_size)
env.reset()

# Both agents defect
a = ([1], [1])
s, r, d, _ = env.step(a)
print(s,r,d)

# Player 1 cooperates but her opponent defects
a = ([0], [1])
s, r, d, _ = env.step(a)
print(s,r,d)


batch_size = 2
max_steps = 100

env = RMG(max_steps=max_steps, payouts=ipd_rewards, batch_size=batch_size)
env.reset()

# Both agents defect in one simulation but in the other both cooperate
a = ([1,0], [1,0])
s, r, d, _ = env.step(a)
print(s,r,d)



# AdvRW tests
print('AdvRW tests')

print('friend env')
env = AdvRw()

env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(1)
env.step(1)
env.step(1)
env.step(1)


print('adversary env')
env = AdvRw('adversary')

env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(1)
env.step(1)
env.step(1)
env.step(1)



