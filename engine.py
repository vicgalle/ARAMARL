"""
This module implements several environments, i.e., the simulators in which agents will interact and learn.
Any environment is characterized by the following two methods:
 * step : receives the actions taken by the agents, and returns the new state of the simulator and the rewards
 perceived by each agent, amongst other things.
 * reset : sets the simulator at the initial state.
"""

import numpy as np


class RMG():
    """
    A two-agent environment for a repeated matrix (symmetric) game.
    Possible actions for each agent are (C)ooperate (0) and (D)efect (1).
    The state is s_t = (a_{t-1}, b_{t-1}) with a_{t-1} and b_{t-1} the actions of the two players in the last turn,
    plus an initial state s_0.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_AGENTS*NUM_ACTIONS + 1   # we add the initial state.

    def __init__(self, max_steps, payouts, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = payouts
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


class AdvRw():
    """
    A two-action stateless environment in which an adversary controls the reward
    """

    def __init__(self, mode='friend', p=0.5):
        self._mode = mode
        # adversary estimation of our action
        self._policy = np.asarray([0.5, 0.5])
        self._learning_rate = 0.25
        self._p = p  # probability for the neutral environment

    def reset(self):
        # self._policy = np.asarray([0.5, 0.5])
        return

    def step(self, action):

        if self._mode == 'friend':
            if np.argmax(self._policy) == action:
                reward = +50
            else:
                reward = -50
        elif self._mode == 'adversary':
            if np.argmax(self._policy) == action:
                reward = -50
            else:
                reward = +50
        elif self._mode == 'neutral':
            box = np.random.rand() < self._p
            if int(box) == action:
                reward = +50
            else:
                reward = -50

        self._policy = (self._learning_rate * np.array([1.0-action, action])
                        + (1.0-self._learning_rate) * self._policy)
        self._policy /= np.sum(self._policy)

        #print('---')
        #print('r', reward)
        #print('p', self._policy)
        #print('---')

        return None, (reward, -reward), True, None

class AdvRw2():
    """
    Friend or Foe modified to model adversary separately..
    """

    def __init__(self, max_steps, payout=50, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout = payout
        self.available_actions = np.array([0,1])
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        dm_reward = self.payout if ac0 == ac1 else -self.payout

        rewards = [dm_reward, -dm_reward] #Assuming zero-sum...
        observations = None

        done = (self.step_count == self.max_steps)

        return observations, rewards, done
#

class AdvRwGridworld():
    """
    Friend or Foe modified to model adversary separately, with gridworld
    """

    def __init__(self, max_steps, batch_size=1):
        self.H = 4
        self.W = 3
        self.world = np.array([self.H, self.W])  # The gridworld

        self.targets = np.array([[0,0], [0,2]])  # Position of the targets
        self.DM = np.array([3,1])  # Initial position of the DM


        self.max_steps = max_steps
        self.batch_size = batch_size
        self.available_actions_DM = np.array([0,1,2,3])  # Up, right, down,left
        self.available_actions_Adv = np.array([0,1])  # Select target 1 or 2.
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.DM = np.array([3,1])
        return

    def _coord2int(self, pos):
        return pos[0] + self.H*pos[1]

    def step(self, action):
        ac_DM, ac_Adv = action

        self.step_count += 1

        if ac_DM == 0: # Up
            self.DM[0] = np.maximum(0, self.DM[0] - 1)
        elif ac_DM == 1: # Right
            self.DM[1] = np.minimum(self.W - 1, self.DM[1] + 1)
        elif ac_DM == 2: # Down
            self.DM[0] = np.minimum(self.H - 1, self.DM[0] + 1)
        elif ac_DM == 3: # Left
            self.DM[1] = np.maximum(0, self.DM[1] - 1)

        done = False
        dm_reward = -1 # One step more
        adv_reward = 0

        # Check if DM is @ targets, then finish

        if np.all(self.DM == self.targets[0,:]):
            if ac_Adv == 0:
                dm_reward += 50
                adv_reward -= 50
            else:
                dm_reward -= 50
                adv_reward += 50
            done = True

        if np.all(self.DM == self.targets[1,:]):
            if ac_Adv == 1:
                dm_reward += 50
                adv_reward -= 50
            else:
                dm_reward -= 50
                adv_reward += 50
            done = True


        # Check if step limit, then finish

        if self.step_count == self.max_steps:
            done = True

        

        #dm_reward = self.payout if ac0 == ac1 else -self.payout

        #rewards = [dm_reward, -dm_reward] #Assuming zero-sum...
        #observations = None

        #done = (self.step_count == self.max_steps)

        return self._coord2int(self.DM), (dm_reward, adv_reward), done
