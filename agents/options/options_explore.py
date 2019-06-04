import numpy as np
from abc import ABCMeta, abstractmethod


class OptionExploreAbstract(metaclass=ABCMeta):

    def __init__(self, action_space):
        self.initial_state = None
        self.action_space = action_space

    def __repr__(self):
        return "".join(["Option_explore(", str(self.initial_state), ")"])

    def __str__(self):
        return "explore option from " + str(self.initial_state)

    def reset(self, initial_state):
        """
        Only the initial_state has to be reset since the option acts randomly
        :param initial_state:
        :return: void
        """
        self.initial_state = initial_state

    def check_end_option(self, new_state):
        """
        Checks if the option has terminated or not.
        :param new_state:
        :return: True if the new_state is different from the initial state
        """
        return new_state != self.initial_state

    @abstractmethod
    def update_option(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def act(self):
        raise NotImplementedError()


class OptionRandomExplore(OptionExploreAbstract):
    """
    This is a special option to explore
    """

    def update_option(self, *args):
        """
        Nothing to update here
        :return:
        """
        pass

    def act(self):
        """
        :return: a random action from the action space
        """
        return np.random.choice(self.action_space)
