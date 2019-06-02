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

    def __init__(self, action_space):
        Agent.__init__(self, action_space)

    def act(self, obs=None):

        return self.action_space[choice(range(len(self.action_space)))]

    " This agent is so simple it doesn't even need to implement the update method! "


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
