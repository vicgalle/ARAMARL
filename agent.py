import numpy as np
from numpy.random import choice

from engine import RMG


class Agent():
    """Parent abstract Agent."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.
        Args:
          reward: The single reward scalar to this agent.
        """
        pass


class DummyAgent(Agent):
    """A dummy and stubborn agent that always takes the first action, no matter what happens."""

    def act(self, obs=None):
        # obs is the state (in this case)
        # This implementes the policy, \pi(s) -> action.

        return self.action_space[0]


class RandomAgent(Agent):
    """An agent that with probability p chooses the first action"""

    def __init__(self, action_space, p):
        Agent.__init__(self, action_space)
        self.p = p

    def act(self, obs=None):

        assert len(self.action_space) == 2
        return choice(self.action_space, p=[self.p, 1-self.p])





