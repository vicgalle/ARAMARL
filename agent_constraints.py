"""
This module implements several agents which have constraints over their action space
"""

from agent import IndQLearningAgent

import numpy as np
from numpy.random import choice

class RestrictedIndQLearningAgent(IndQLearningAgent):


    """we only need to override this method"""
    def act(self, obs=None, valid_action=None, previous_action=None):

        # mask the Q-function here
        # TODO
        if valid_action is not None:
            mask = np.array([ valid_action(obs, a, previous_action) for a in self.action_space ])
            Q = self.Q[obs, :]
            #print(Q)
            Q[~mask] = -np.infty
            #print(Q)

            if np.random.rand() < self.epsilon:
                mask = mask.astype(float)
                mask /= mask.sum()
                a = choice(self.action_space, p=mask)
                if a == 0:
                    print('fuck')
                return choice(self.action_space, p=mask)
            else:
                return self.action_space[np.argmax(Q)]

        return super(RestrictedIndQLearningAgent, self).act(obs)
    
    