import numpy as np


class IPD():
    """
    A two-agent environment for the Iterated Prisoner's Dilemma.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros((self.batch_size, self.NUM_STATES))
        init_state[:, -1] = 1
        observations = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observations, info

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = []

        # The state is a OHE vector indicating [CC, CD, DC, DD, initial], (iff NUM_STATES = 5)
        state0 = np.zeros((self.batch_size, self.NUM_STATES))
        state1 = np.zeros((self.batch_size, self.NUM_STATES))
        for i, (a0, a1) in enumerate(zip(ac0, ac1)):  # iterates over batch dimension
            rewards.append([self.payout_mat[a1][a0], self.payout_mat[a0][a1]])
            state0[i, a0 * 2 + a1] = 1
            state1[i, a1 * 2 + a0] = 1
        rewards = list(map(np.asarray, zip(*rewards)))
        observations = [state0, state1]

        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, rewards, done, info


# Now we perform some tests

batch_size = 1
max_steps = 100

env = IPD(max_steps=max_steps, batch_size=batch_size)
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

env = IPD(max_steps=max_steps, batch_size=batch_size)
env.reset()

# Both agents defect in one simulation but in the other both cooperate
a = ([1,0], [1,0])
s, r, d, _ = env.step(a)
print(s,r,d)

