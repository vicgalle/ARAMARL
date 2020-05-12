"""
This module implements several agents in which the Q function is approximated
"""

from agent import IndQLearningAgentSoftmax

import numpy as np
import jax
import jax.numpy as jnp
from numpy.random import choice
from scipy.special import softmax

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

        def Q_val(params, obs, a0):
            return jnp.dot(obs.flatten(), params)[a0]
        
        self.grad_fn = jax.jit(jax.grad(Q_val))
        
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
    
    