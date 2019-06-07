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

        # print('---')
        #print('r', reward)
        #print('p', self._policy)
        # print('---')

        return None, (reward, -reward), True, None


class AdvRw2():
    """
    Friend or Foe modified to model adversary separately..
    """

    def __init__(self, max_steps, payout=50, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout = payout
        self.available_actions = np.array([0, 1])
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        dm_reward = self.payout if ac0 == ac1 else -self.payout

        rewards = [dm_reward, -dm_reward]  # Assuming zero-sum...
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

        self.targets = np.array([[0, 0], [0, 2]])  # Position of the targets
        self.DM = np.array([3, 1])  # Initial position of the DM

        self.max_steps = max_steps
        self.batch_size = batch_size
        self.available_actions_DM = np.array(
            [0, 1, 2, 3])  # Up, right, down,left
        self.available_actions_Adv = np.array([0, 1])  # Select target 1 or 2.
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.DM = np.array([3, 1])
        return

    def _coord2int(self, pos):
        return pos[0] + self.H*pos[1]

    def step(self, action):
        ac_DM, ac_Adv = action

        self.step_count += 1

        if ac_DM == 0:  # Up
            self.DM[0] = np.maximum(0, self.DM[0] - 1)
        elif ac_DM == 1:  # Right
            self.DM[1] = np.minimum(self.W - 1, self.DM[1] + 1)
        elif ac_DM == 2:  # Down
            self.DM[0] = np.minimum(self.H - 1, self.DM[0] + 1)
        elif ac_DM == 3:  # Left
            self.DM[1] = np.maximum(0, self.DM[1] - 1)

        done = False
        dm_reward = -1  # One step more
        adv_reward = 0

        # Check if DM is @ targets, then finish

        if np.all(self.DM == self.targets[0, :]):
            if ac_Adv == 0:
                dm_reward += 50
                adv_reward -= 50
            else:
                dm_reward -= 50
                adv_reward += 50
            done = True

        if np.all(self.DM == self.targets[1, :]):
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

        # rewards = [dm_reward, -dm_reward] #Assuming zero-sum...
        #observations = None

        #done = (self.step_count == self.max_steps)

        return self._coord2int(self.DM), (dm_reward, adv_reward), done


class Blotto():
    """
    Blotto game with multiple adversaries
    """

    def __init__(self, max_steps, payout=50, batch_size=1, deterministic=True):
        self.max_steps = max_steps
        self.batch_size = batch_size
        #self.payout = payout
        self.available_actions = np.array([0, 1])
        self.step_count = 0
        self.deterministic = deterministic

    def reset(self):
        self.step_count = 0
        return

    def step(self, actions):
        """ action[0] is that of the defender """
        self.step_count += 1

        num_attackers = len(actions) - 1

        actions = np.asarray(actions)

        att_rew = np.sum(actions[1:, ], axis=0)
        tmp = actions[0, ] - att_rew

        draw_pos = tmp == 0
        if self.deterministic != True:
            tmp[tmp == 0] = np.random.choice(
                [-1, 1], size=len(tmp[tmp == 0]))*(actions[0, draw_pos] > 0)
        else:
            tmp[tmp == 0] = 0

        ind = np.sum(actions, axis=0) > 0

        tmp = tmp*ind

        tmp[tmp < 0] = -1
        tmp[tmp > 0] = 1

        # print('tmp', tmp)

        reward_dm = tmp.sum()

        tmp2 = actions[1:, ] - actions[0, ]
        tmp2[tmp2 > 0] = 1
        tmp2[tmp2 < 0] = -1

        # print('tmp2', tmp2)

        s = np.sum(actions[1:, draw_pos], axis=0)
        z = draw_pos & actions[1:, ]

        z_new = z/z.sum(axis=0)
        z_new = np.nan_to_num(z_new)
        z_new = z_new*ind

        # print('z_new', z_new)

        #z_new = np.zeros_like(z_new)
        z_new[:, draw_pos] = z_new[:, draw_pos]*np.sign(-tmp[draw_pos])

        tmp2[z == 1.] = 0

        # print('tmp2', tmp2)

        z_new = tmp2 + z_new

        # print('z-new', z_new)
        # print('tmp2', tmp2)

        rewards_atts = np.sum(z_new*(actions[1:, ] > 0), axis=1)

        rewards = [reward_dm]

        for r in rewards_atts:
            rewards.append(r)

        observations = None

        done = (self.step_count == self.max_steps)

        return observations, rewards, done


class Urban():
    """
    A two-agent environment for a urban resource allocation problem.
    """

    def __init__(self):
        # The state is designated by s = (s_0, s_1, s_2, s_3)
        # s_0 represents wheter we are in the initial state or not
        # s_i, i>0 represent whether the attack was successful on the site i.
        self.state = np.array([1, 0, 0, 0])
        self.step_count = 0
        self.max_steps = 2  # as in the ARA for Urban alloc. paper
        self.payoffs = np.array([1., 0.75, 2.])  # v_i from the paper

        # Transition dynamics

        # p(s_1_i = 1 | d1_i, a_i)  for site i
        self.p_s1_d1_a = np.array([[0, 0.85, 0.95],
                                   [0, 0.6, 0.75],
                                   [0, 0.3, 0.5],
                                   [0, 0.05, 0.1],
                                   [0, 0,  0.05]])

        # p(s_2_i = 1 | s_1_i, d2_i) for site i
        self.p_s2_s1_d2 = np.array([[0, 0, 0, 0],
                                    [0.95, 0.8, 0.6, 0.4]])

        self.n_sites = 3
        self.k = 0.005
        self.rho = 0.1
        self.c_A = 10
        self.c_D = 10

    def state2idx(self, state):
        """
        In [19]: state = np.array([1, 0, 0, 1])
        In [20]: state2idx(state)
        Out[20]: 9
        """
        pows = np.array([1 << i for i in range(len(state))[::-1]])
        return np.dot(pows, state)

    def idx2state(self, idx):
        """
        In [28]: idx = 9
        In [30]: idx2state(idx)
        Out[30]: array([1, 0, 0, 1])
        """
        return (idx & (1 << np.arange(len(self.state))) > 0).astype(int)

    def actionDM2idx(self, a):
        """ Now we have 3 sites, in which we can defend with up to 5 units. """
        pows = np.array([5**i for i in range(self.n_sites)[::-1]])
        return np.dot(pows, a)

    def idx2actionDM(self, idx):
        return list(map(int, (list(np.base_repr(idx, 5, padding=3))[-self.n_sites:])))

    def valid_actionDM(self, state, action, prev_action):
        if state[0] == 1: #initial state
            return np.sum(action) == 4
        else:  # second move
            c1 = np.sum(action) == 4
            c2 = action[0] <= prev_action[0] + prev_action[1]
            c3 = action[1] <= prev_action[0] + prev_action[1] + prev_action[2]
            c4 = action[2] <= prev_action[1] + prev_action[2]
            return c1 & c2 & c3 & c4

    def reset(self):
        self.step_count = 0
        self.state = np.array([1, 0, 0, 0])
        return

    def step(self, action):

        # first action is that from the DM
        ac0, ac1 = action

        self.step_count += 1

        if self.step_count == 1:

            self.state = np.array([0, 0, 0, 0])
            for i in range(self.n_sites):
                p = self.p_s1_d1_a[ac0[i], ac1[i]]
                u = np.random.rand()
                if u <= p:
                    self.state[i + 1] = 1  # success

            rewards = [0., 0.]   # no rewards until end of episode
            observations = self.state

            done = False

            return observations, rewards, done

        elif self.step_count == 2:  # end of episode

            for i in range(self.n_sites):
                p = self.p_s2_s1_d2[self.state[i+1], ac0[i]]
                u = np.random.rand()
                if u <= p:
                    self.state[i + 1] = 1  # success

            done = True
            observations = self.state
            rewards = [- np.exp(self.c_D * self.rho * np.sum(self.payoffs * self.state[1:])),
                       np.exp(self.c_A * np.sum(self.payoffs * self.state[1:] - ac1 * self.k))]  # what to do with the Adversary!?

            return observations, rewards, done
