"""
This module implements several agents. An agent is characterized by two methods:
 * act : implements the policy, i.e., it returns agent's decisions to interact in a MDP or Markov Game.
 * update : the learning mechanism of the agent.
"""

import numpy as np
from numpy.random import choice

from engine import RMG


class Agent():
    """
    Parent abstract Agent.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        """
        This implements the policy, \pi : S -> A.
        obs is the observed state s
        """
        raise NotImplementedError()

    def update(self, obs, actions, rewards, new_obs):
        """
        This is after an interaction has ocurred, ie all agents have done their respective actions, observed their rewards and arrived at
        a new observation (state).
        For example, this is were a Q-learning agent would update her Q-function
        """
        pass


class DummyAgent(Agent):
    """
    A dummy and stubborn agent that always takes the first action, no matter what happens.
    """

    def act(self, obs=None):
        # obs is the state (in this case)

        return self.action_space[0]

    " This agent is so simple it doesn't even need to implement the update method! "


class RandomAgent(Agent):
    """
    An agent that chooses actions at random
    """

    def __init__(self, action_space, policy):
        Agent.__init__(self, action_space)
        self.policy = policy

    def act(self, obs=None):

        action_idx = choice(range(len(self.action_space)), p = self.policy)
        return self.action_space[action_idx]

    " This agent is so simple it doesn't even need to implement the update method! "

class ExpSmoother(Agent):
    """
    An agent predicting probability of DM putting resource in each position
    using an exponential smoother.
    """

    def __init__(self, action_space, n_pos, learning_rate):
        Agent.__init__(self, action_space)

        self.alpha = learning_rate
        self.n_pos = n_pos
        # Initial forecast
        self.prob = np.ones( self.n_pos )
        self.prob = self.prob/np.sum(self.prob)

    def act(self, obs=None):
        """Just chooses the less probable place"""
        action = np.zeros(self.n_pos, dtype="int")
        action[np.argmin(self.prob)] = 1
        return action


    def update(self, obs, actions, rewards, new_obs):
        """Update the exp smoother"""
        a0 = actions[1]

        self.prob = self.alpha*self.prob + (1-self.alpha)*a0 # Update beliefs about DM
        self.prob = self.prob/np.sum(self.prob)


class IndQLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return self.action_space[choice(range(len(self.action_space)))]
        else:
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0 = actions[0]
        r0 = rewards[0]
        idx = int(np.where(np.all(self.action_space == a0, axis=1))[0])

        self.Q[obs, idx] = (1 - self.alpha)*self.Q[obs, idx] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))


class FPLearningAgent(Agent):
    """
    A Q-learning agent that treats the other players as level 0 agents.
    She learns from other's actions in a bayesian way.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy1_action_space, enemy2_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy1_action_space = enemy1_action_space
        self.enemy2_action_space = enemy2_action_space
        # This is the Q-function Q(s, a, b, c)
        self.Q = np.zeros([self.n_states, len(self.action_space), len(self.enemy1_action_space), len(self.enemy2_action_space)])
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir1 = np.ones( len(self.enemy1_action_space) )
        self.Dir2 = np.ones( len(self.enemy2_action_space) )

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return self.action_space[choice(range(len(self.action_space)))]
        else:
            collapse1 = np.dot( self.Q[obs], self.Dir2/np.sum(self.Dir2) )
            collapse2 = np.dot( collapse1, self.Dir1/np.sum(self.Dir1) )
            return self.action_space[ np.argmax( collapse2 ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1, a2 = actions
        r0, r1, r2 = rewards

        idx = int(np.where(np.all(self.action_space == a0, axis=1))[0])
        idx1 = int(np.where(np.all(self.enemy1_action_space == a1, axis=1))[0])
        idx2 = int(np.where(np.all(self.enemy2_action_space == a2, axis=1))[0])

        self.Dir1[idx1] += 1 # Update beliefs about adversariy 1
        self.Dir2[idx2] += 1 # Update beliefs about adversary 2

        collapse1 = np.dot( self.Q[new_obs], self.Dir2/np.sum(self.Dir2) )
        collapse2 = np.dot( collapse1, self.Dir1/np.sum(self.Dir1) )
        aux = np.max( collapse2 )
        self.Q[obs, idx, idx1, idx2] = (1 - self.alpha)*self.Q[obs, idx, idx1, idx2] + self.alpha*(r0 + self.gamma*aux)


##
class Level2QAgent(Agent):
    """
    A Q-learning agent that treats the other players as a level 1 agents.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy1_action_space, enemy2_action_space,
        n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alphaA = learning_rate
        self.alphaB = learning_rate
        self.alphaC = learning_rate
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.epsilonC = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        self.gammaC = self.gammaA
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy1_action_space = enemy1_action_space
        self.enemy2_action_space = enemy2_action_space

        ## Other agents
        # Enemy 1
        self.enemy1 = FPLearningAgent(self.enemy1_action_space, self.action_space,
            self.enemy2_action_space, n_states=1,
                      learning_rate=self.alphaB, epsilon=self.epsilonB,
                      gamma=self.gammaB)

        # Enemy 2
        self.enemy2 = FPLearningAgent(self.enemy2_action_space, self.action_space,
            self.enemy1_action_space, n_states=1,
                      learning_rate=self.alphaC, epsilon=self.epsilonC,
                      gamma=self.gammaC)

        # This is the Q-function Q_A(s, a, b1, b2) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy1_action_space), len(self.enemy2_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return self.action_space[choice(range(len(self.action_space)))]
        else:
            b = self.enemy1.act()
            c = self.enemy2.act()
            ##
            idxb = int(np.where(np.all(self.enemy1_action_space == b, axis=1))[0])
            idxc = int(np.where(np.all(self.enemy2_action_space == c, axis=1))[0])
            #
            return self.action_space[ np.argmax( self.QA[obs, :, idxb, idxc ] ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b, c = actions
        rA, rB, rC = rewards

        idxa = int(np.where(np.all(self.action_space == a, axis=1))[0])
        idxb = int(np.where(np.all(self.enemy1_action_space == b, axis=1))[0])
        idxc = int(np.where(np.all(self.enemy2_action_space == c, axis=1))[0])

        # A0.update(0, [a0,a1,a2], rewards, 0)
        self.enemy1.update(obs, [b,a,c], [rB, rA, rC], new_obs )
        self.enemy2.update(obs, [c,a,b], [rC, rA, rB], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy1.act()
        cc = self.enemy2.act()

        idxbb = int(np.where(np.all(self.enemy1_action_space == bb, axis=1))[0])
        idxcc = int(np.where(np.all(self.enemy2_action_space == cc, axis=1))[0])



        # Finally we update the supported agent's Q-function
        self.QA[obs, idxa, idxb, idxc] = (1 - self.alphaA)*self.QA[obs, idxa, idxb, idxc] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, idxbb, idxcc]))
