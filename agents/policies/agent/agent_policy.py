"""
Policies that can be applied only on agents
"""
from abc import ABCMeta, abstractmethod


class PolicyAbstractAgent(object):
    """
    Given a state, a policy returns an action
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        reset the specified parameters (for example the current state)
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def find_best_action(self, state):
        """
        Find the best action for the parameter state. It returns two elements: how to go and where to go.
        - how to go: the index of the option to activate at this state
        - where to go: the terminal target state of the option
        :param state:
        :return: best_option_index, terminal_state
        if both are None then this means that the exploration has to be performed
        """
        raise NotImplementedError()

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        """
        updates the value of a state specified in the arguments
        :param args:
        :param kwargs:
        :return: void
        """
        raise NotImplementedError()

    @abstractmethod
    def max_number_successors(self):
        """
        Get the maximal number of successor states over all states.
        :return: an integer between 0 and the number of states
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_state(self):
        """
        the current state is *not* stored in the Agent class to avoid to
        double track this variable
        :return: the current state
        """
        raise NotImplementedError()
