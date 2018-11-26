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

class KUHN():
    """
    A two-agent environment for a kuhn poker game.
    The "state" is s_t = (c_{d}, c_{a}) with c_{d}, c_{a} the cards of player 1
    and player 2, respectively.
    Player 1 (P1) actions are represented as vectors of 3 components.
    The firs is the action taken in the second round given that his opponent has
    passed. The second one is the action taken in the second round given that
    his opponent has bet. The third one is the action taken in the first round.
    Player 2 (P2) can either pass (0) or bet (1).
    """
    # Possible actions
    NUM_AGENTS = 2
    #NUM_ACTIONS_P1 = 8
    #NUM_ACTIONS_P2 = 2


    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        #self.payout_mat = payouts
        #self.available_actions = [
        #    np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
        #    for _ in range(self.NUM_AGENTS)
        #]

        self.step_count = None


    def reset(self):
        self.step_count = 0
        #init_state = np.zeros((self.batch_size, self.NUM_STATES))
        #init_state[:, -1] = 1
        #observations = [init_state, init_state]
        #info = [{'available_actions': aa} for aa in self.available_actions]
        #return observations, info


    def step(self, action, state):
        ac0, ac1 = action
        ac0 = np.array([ ac0[2], ac0[ac1] ])
        win = -2.0*np.argmax(state) + 1.0 #+1 is P1 has the biggest card, -1 otherwise

        self.step_count += 1

        # Zero-Sum so we just need one componet where we store reward for P1

        if ac0[0] == ac1:
            reward = win + win*ac1
        else:
            if ac0[0] == 1:
                reward = 1
            else:
                reward = (2*win +1)*ac0[1] - ac1

        rewards = [reward, -reward]
        observations = [state[0], state[1]]



        """
        # The state is a OHE vector indicating [CC, CD, DC, DD, initial], (iff NUM_STATES = 5)
        state0 = np.zeros((self.batch_size, self.NUM_STATES))
        state1 = np.zeros((self.batch_size, self.NUM_STATES))
        for i, (a0, a1) in enumerate(zip(ac0, ac1)):  # iterates over batch dimension
            rewards.append([self.payout_mat[a1][a0], self.payout_mat[a0][a1]])
            state0[i, a0 * 2 + a1] = 1
            state1[i, a1 * 2 + a0] = 1
        """


        done = (self.step_count == self.max_steps)
        #info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, rewards, done
