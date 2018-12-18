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


############################ KUHN POKER ########################################

class kuhnAgent2(Agent):
    """
    Stationary second agent in kuhn poker game. His action is parametrized using
    two parameters and depends on his card and the previous movement of his opponent.
    Cards: J=0, Q=1, K=2.
    Actions: pass=0, bet=1.
    """

    def __init__(self, action_space, zeta, eta):
        Agent.__init__(self, action_space)

        self.zeta = zeta
        self.eta = eta

    def act(self, card, enemy_action):
        if card == 2:
            return 1
        if card == 0:
            if enemy_action == 0:
                return 1 if np.random.rand() < self.zeta else  0
            else:
                return 0
        else:
            if enemy_action == 0:
                return 0
            else:
                return 1 if np.random.rand() < self.eta else 0

class FPLearningAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        # This is the Q-function Q(s, a, b)
        self.Q = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        # Parameters of the Dirichlet distribution used to model the other agent, conditioned
        # Initialized using a uniform prior
        self.Dir = np.ones( [self.n_states, len(self.enemy_action_space)] )

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return self.action_space[choice(np.arange(len(self.action_space)))]
        else:
            return self.action_space[ np.argmax( np.dot( self.Q[obs[0],:,:],
                    self.Dir[obs[0],:]/np.sum(self.Dir[obs[0],:]) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        self.Dir[obs[0],a1] += 1 # Update beliefs about adversary
        aux = np.max( np.dot( self.Q[new_obs[0],:,:],
            self.Dir[new_obs[0],:]/np.sum(self.Dir[new_obs[0],:]) ) )

        self.Q[obs[0], a0, a1] = ( (1 - self.alpha)*self.Q[obs[0], a0, a1] +
            self.alpha*(r0 + self.gamma*aux) )
