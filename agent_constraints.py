"""
This module implements several agents which have constraints over their action space
"""

from agent import IndQLearningAgent

class RestrictedIndQLearningAgent(IndQLearningAgent):


    """we only need to override this method"""
    def act(self, obs=None):

        # mask the Q-function here
        # TODO

        return super(RestrictedIndQLearningAgent, self).act(obs)
    
    