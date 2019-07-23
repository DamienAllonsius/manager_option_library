import numpy as np
from abc import ABCMeta, abstractmethod
from ao.utils.miscellaneous import obs_equal


class OptionExploreAbstract(metaclass=ABCMeta):

    def __init__(self, action_space):
        self.initial_state = None
        self.action_space = action_space
        self.index = None
        self.score = 0  # random option can find rewards too

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
        self.score = 0

    def check_end_option(self, new_state):
        """
        Checks if the option has terminated or not.
        :param new_state:
        :return: True if the new_state is different from the initial state
        """
        return not obs_equal(self.initial_state, new_state)

    @abstractmethod
    def update_option(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def act(self, *args, **kwargs):
        raise NotImplementedError()


class OptionRandomExplore(OptionExploreAbstract):
    """
    This is a special option to explore
    """

    def update_option(self, o_r_d_i, intra_reward, action, end_option, train_episode=None):
        """
        Nothing to update here
        :return:
        """
        if o_r_d_i[1] > 0:
            self.score = o_r_d_i[1]
        else:
            self.score = 0

    def act(self, train_episode):
        """
        :return: a random action from the action space
        """
        return np.random.choice(self.action_space)
