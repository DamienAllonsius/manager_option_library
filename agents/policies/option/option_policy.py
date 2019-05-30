"""
Policies that can be applied only on options

parameters:
probability_random_action_agent
learning_rate
"""
import numpy as np
from abc import ABCMeta, abstractmethod


class PolicyAbstractOption(object):
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
        :return: True (indicates to the option that the policy was reset)
        """
        raise NotImplementedError()

    @abstractmethod
    def find_best_action(self, state):
        """
        Find the best action for the parameter state.
        :param state:
        :return: best_action: an element from the action space
        """
        raise NotImplementedError()

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        """
        updates the policy
        :param args:
        :param kwargs:
        :return: void
        """
        raise NotImplementedError()

    @abstractmethod
    def get_random_action(self, *args, **kwargs):
        raise NotImplementedError()
