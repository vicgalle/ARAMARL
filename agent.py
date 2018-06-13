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
    An agent that with probability p chooses the first action
    """

    def __init__(self, action_space, p):
        Agent.__init__(self, action_space)
        self.p = p

    def act(self, obs=None):

        assert len(self.action_space) == 2
        return choice(self.action_space, p=[self.p, 1-self.p])

    " This agent is so simple it doesn't even need to implement the update method! "


class IndQLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma):
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
            return choice(self.action_space)
        else:
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

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
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones( len(self.enemy_action_space) )

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[ np.argmax( np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        self.Dir[a1] += 1 # Update beliefs about adversary

        aux = np.max( np.dot( self.Q[new_obs], self.Dir/np.sum(self.Dir) ) )
        self.Q[obs, a0, a1] = (1 - self.alpha)*self.Q[obs, a0, a1] + self.alpha*(r0 + self.gamma*aux)

class FPQwForgetAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way, plus a discount to ignore distant observations.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, forget=0.95):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        # This is the Q-function Q(s, a, b)
        self.Q = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones( len(self.enemy_action_space) )
        self.forget = forget

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[ np.argmax( np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        self.Dir *= self.forget
        self.Dir[a1] += 1 # Update beliefs about adversary

        aux = np.max( np.dot( self.Q[new_obs], self.Dir/np.sum(self.Dir) ) )
        self.Q[obs, a0, a1] = (1 - self.alpha)*self.Q[obs, a0, a1] + self.alpha*(r0 + self.gamma*aux)

class TFT(Agent):
    """
    An agent playing TFT
    """

    def __init__(self, action_space):
        Agent.__init__(self, action_space)

    def act(self, obs):

        if obs[0] == None: #MAAAL esto lo interpreta como vacío si (0,0)!!!
            return(self.action_space[0]) # First move is cooperate
        else:
            return(obs[1]) # Copy opponent's previous action



    " This agent is so simple it doesn't even need to implement the update method! "

class Mem1FPLearningAgent(Agent):
    """
    Extension of the FPLearningAgent to the case of having memory 1
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        # This is the Q-function Q(s, a, b)
        self.Q = np.zeros( [len(self.action_space),len(self.enemy_action_space),
            len(self.action_space), len(self.enemy_action_space)] )
        # Parameters of the Dirichlet distribution used to model the other agent, conditioned to the previous action
        # Initialized using a uniform prior
        self.Dir = np.ones( [len(self.action_space),
            len(self.enemy_action_space),len(self.enemy_action_space)] )

    def act(self, obs):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            if obs[0] == None:
                unif = np.ones(len(self.action_space))
                return self.action_space[ np.argmax( np.dot( self.Q[obs[0], obs[1],:,:],
                    unif/np.sum(unif) ) ) ]
            else:
                return self.action_space[ np.argmax( np.dot( self.Q[obs[0], obs[1],:,:],
                    self.Dir[obs[0], obs[1],:]/np.sum(self.Dir[obs[0], obs[1],:]) ) ) ]



    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        if obs[0] == None:
            unif = np.ones(len(self.action_space))
            aux = np.max( np.dot( self.Q[new_obs[0],new_obs[1],:,:], unif/np.sum(unif) ) )
        else:
            self.Dir[obs[0],obs[1],a1] += 1 # Update beliefs about adversary
            aux = np.max( np.dot( self.Q[new_obs[0],new_obs[1],:,:],
                self.Dir[new_obs[0],new_obs[1],:]/np.sum(self.Dir[new_obs[0],new_obs[1],:]) ) )

        self.Q[obs[0], obs[1], a0, a1] = ( (1 - self.alpha)*self.Q[obs[0], obs[1], a0, a1] +
            self.alpha*(r0 + self.gamma*aux) )
