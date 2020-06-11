"""
This module implements several agents in which the Q function is approximated
"""

from agent import IndQLearningAgentSoftmax, Level2QAgent

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from numpy.random import choice
from scipy.special import softmax
from scipy.signal import convolve


def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

class RegressionIndQLearningSoftmax(IndQLearningAgentSoftmax):


    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None):
        IndQLearningAgentSoftmax.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space)
        # Regression weights
        self.n_a = len(action_space)
        self.w = 0.001*np.random.randn(9*4, len(action_space))

        
        #self.grad_fn = jax.jit(jax.grad(Q_val))
        
    def act(self, obs=None):
        obs_flat = obs.flatten()
        Q = np.dot(obs_flat, self.w)
        p = stable_softmax(Q)
        #print(Q)
        #print(p)
        #return np.argmax(np.dot(obs_flat, self.w))
        return choice(self.action_space, p=p)

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        Qp = np.dot(new_obs.flatten(), self.w)
        Q = np.dot(obs.flatten(), self.w)
        #w_jax = jnp.array(self.w)
        #grad = self.grad_fn(w_jax, obs, a0)[:, a0]
        #print(grad)
        grad = obs.flatten()
        #print(grad)
        #grad = np.clip(grad, -2, 2)
        #print(grad.shape)
        #print((r0 + self.gamma*jnp.max(Qp) - Q[a0]).shape)
        
        self.w[:, a0] = self.w[:, a0] + self.alpha*(r0 + self.gamma*np.max(Qp) - Q[a0])*grad
        #self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

class DQN(IndQLearningAgentSoftmax):


    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None):
        IndQLearningAgentSoftmax.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space)
        # Regression weights
        self.n_a = len(action_space)
        self.w = 0.001*np.random.randn(9*4, len(action_space))

        self.W1 = np.random.normal(0, 2 / np.sqrt(4*2 * 2), size=(4, 2, 2))

    def relu(self, x):
        return np.where(x>0,x,0)
    
    def relu_prime(self, x):
        return np.where(x>0,1,0)


    def forward(self, W1, W2, obs, y):
        l0 = np.einsum('ijk->kji', obs)
        #l0 = obs[0, :, :]
        l0_conv = convolve(l0, W1[::-1, ::-1], 'same', 'direct')
        l1 = self.relu(l0_conv)
        
        

        
    def act(self, obs=None):
        self.forward(self.W1, self.W1, obs, None)
        obs_flat = obs.flatten()
        Q = np.dot(obs_flat, self.w)
        p = stable_softmax(Q)
        #print(Q)
        #print(p)
        #return np.argmax(np.dot(obs_flat, self.w))
        return choice(self.action_space, p=p)

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        Qp = np.dot(new_obs.flatten(), self.w)
        Q = np.dot(obs.flatten(), self.w)
        #w_jax = jnp.array(self.w)
        #grad = self.grad_fn(w_jax, obs, a0)[:, a0]
        #print(grad)
        grad = obs.flatten()
        #print(grad)
        #grad = np.clip(grad, -2, 2)
        #print(grad.shape)
        #print((r0 + self.gamma*jnp.max(Qp) - Q[a0]).shape)
        
        self.w[:, a0] = self.w[:, a0] + self.alpha*(r0 + self.gamma*np.max(Qp) - Q[a0])*grad
        #self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))
    
class RegressionLevel2QAgentSoftmax(Level2QAgent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Level2QAgent.__init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma)

        self.enemy = RegressionIndQLearningSoftmax(self.enemy_action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB)

        self.w = 0.001*np.random.randn(9*4, len(action_space), len(enemy_action_space))
        
    def act(self, obs=None):
        b = self.enemy.act(obs)
        obs_flat = obs.flatten()
        Q = np.einsum('i,ijk->jk', obs_flat, self.w)[:, b]
        #Q = np.dot(obs_flat, self.w)
        p = stable_softmax(Q)
        return choice(self.action_space, p=p)

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy.act(obs)

        #Qp = np.dot(new_obs.flatten(), self.w)
        Qp = np.einsum('i,ijk->jk', new_obs.flatten(), self.w)
        #Q = np.dot(obs.flatten(), self.w)
        Q = np.einsum('i,ijk->jk', obs.flatten(), self.w)
        grad = obs.flatten()

        # Finally we update the supported agent's Q-function
        self.w[:, a, b] = self.w[:, a, b] + self.alphaA*(rA + self.gammaA*np.max(Qp[:, bb]) - Q[a, b])*grad
        #self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, bb]))