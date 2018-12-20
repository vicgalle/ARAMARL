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
            return choice(self.action_space)
        else:
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

class Exp3QLearningAgent(Agent):
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
        self.S = np.zeros([self.n_states, len(self.action_space)])
        self.p = np.ones([self.n_states, len(self.action_space)])/len(self.action_space)


    def act(self, obs=None):
        """An epsilon-greedy policy"""
        return choice(self.action_space, p=self.p[obs,:])

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))
        self.S[obs, a0] = self.S[obs, a0] + self.Q[obs, a0]/self.p[obs, a0]

        K = len(self.action_space)

        for i in self.action_space:
            self.p[obs, i] = (1-self.epsilon)/( np.exp((self.S[obs, :] - self.S[obs, i])*self.epsilon/K).sum() ) + self.epsilon/K



class PHCLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, delta, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])
        self.pi = 1/len(self.action_space)*np.ones([self.n_states, len(self.action_space)])

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        #print(self.pi[obs,:])
        #print(self.n_states)
        return choice(self.action_space, p=self.pi[obs,:])


    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

        a = self.action_space[np.argmax(self.Q[obs, :])]
        self.pi[obs, :] -= self.delta*self.alpha / len(self.action_space)
        self.pi[obs, a] += ( self.delta*self.alpha + self.delta*self.alpha / len(self.action_space))
        self.pi[obs, :] = np.maximum(self.pi[obs, :], 0)
        self.pi[obs, :] /= self.pi[obs,:].sum()


class WoLFPHCLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, delta_w, delta_l, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta_w = delta_w
        self.delta_l = delta_l
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])
        self.pi = 1/len(self.action_space)*np.ones([self.n_states, len(self.action_space)])
        self.pi_ = 1/len(self.action_space)*np.ones([self.n_states, len(self.action_space)])
        self.C = np.zeros(self.n_states)

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        #print(self.pi[obs,:])
        #print(self.n_states)
        return choice(self.action_space, p=self.pi[obs,:])


    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

        self.C[obs] += 1
        self.pi_[obs, :] += (self.pi[obs,:]-self.pi_[obs,:])/self.C[obs]
        a = self.action_space[np.argmax(self.Q[obs, :])]

        if np.dot(self.pi[obs, :], self.Q[obs,:]) > np.dot(self.pi_[obs, :], self.Q[obs,:]):
            delta = self.delta_w
        else:
            delta = self.delta_l

        self.pi[obs, :] -= delta*self.alpha / len(self.action_space)
        self.pi[obs, a] += ( delta*self.alpha + delta*self.alpha / len(self.action_space))
        self.pi[obs, :] = np.maximum(self.pi[obs, :], 0)
        self.pi[obs, :] /= self.pi[obs,:].sum()

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

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, forget=0.8):
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

###############################################
## Agents for the friend or foe environment
###############################################
class ExpSmoother(Agent):
    """
    An agent predicting its opponent actions using an exponential smoother.
    """

    def __init__(self, action_space, enemy_action_space, learning_rate):
        Agent.__init__(self, action_space)

        self.alpha = learning_rate
        self.action_space = action_space
        self.enemy_action_space = enemy_action_space
        # Initial forecast
        self.prob = np.ones( len(self.enemy_action_space) )*0.5

    def act(self, obs=None):
        """Just chooses the less probable place"""
        return self.action_space[np.argmin(self.prob)]


    def update(self, obs, actions, rewards, new_obs):
        """Update the exp smoother"""
        a0, a1 = actions
        OHE = np.array([[1,0],[0,1]])

        self.prob = self.alpha*self.prob + (1-self.alpha)*OHE[a1] # Update beliefs about DM

##
class Level2QAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alphaA = learning_rate
        self.alphaB = learning_rate
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = FPLearningAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            b = self.enemy.act()
            # Add epsilon-greedyness
            return self.action_space[ np.argmax( self.QA[obs, :, b ] ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy.act()

        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, bb]))


##
class Level3QAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 2 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alphaA = learning_rate
        self.alphaB = learning_rate
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = Level2QAgent(self.enemy_action_space, self.action_space,
         self.n_states, self.alphaB, self.epsilonB, self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            b = self.enemy.act()
            # Add epsilon-greedyness
            return self.action_space[ np.argmax( self.QA[obs, :, b ] ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy.act()

        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, bb]))


class Level3QAgentMix(Agent):
    """
    A Q-learning agent that treats the other player as a level 2 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alphaA = learning_rate
        self.alphaB = learning_rate
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = pLevel2QAgent(self.enemy_action_space, self.action_space,
         self.n_states, self.alphaB, self.epsilonB, self.gammaB)

        self.enemy2 = FPLearningAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA1 = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        self.QA2 = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            b = self.enemy.act()
            b2 = self.enemy2.act()
            # Add epsilon-greedyness
        res1 = self.action_space[ np.argmax( self.QA1[obs, :, b ] ) ]
        res2 = self.action_space[ np.argmax( self.QA2[obs, :, b2 ] ) ]
        return choice(np.array([res1, res2]))

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update( obs, [b,a], [rB, rA], new_obs )
        self.enemy2.update( obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy.act()
        bb2 = self.enemy2.act()

        # Finally we update the supported agent's Q-function
        self.QA1[obs, a, b] = (1 - self.alphaA)*self.QA1[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA1[new_obs, :, bb]))
        self.QA2[obs, a, b] = (1 - self.alphaA)*self.QA2[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA2[new_obs, :, bb2]))
