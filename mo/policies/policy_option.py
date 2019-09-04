"""
Policies that can be applied only on options

parameters:
probability_random_action_agent
learning_rate
"""
from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractPolicyOption(metaclass=ABCMeta):
    """
    Given a state, a policy returns an action
    """

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


class PolicyOptionQArray(AbstractPolicyOption):
    """
    Example of a possible option
    _ This class is used when the number of states is unknown but the number of actions is known
    _ state_list = [state_1, state_2, ...] # a list of all states
    _ Unlike QTree, this class does not allow an efficient exploration.
    _ But here the Q function cannot be represented with a Tree
      because there exist sets of states which are not connected.
    _Action should be an integer between 0 and number_actions - 1
    """
    def __init__(self, action_space, parameters):
        """
        :param action_space: typically, an array of integers (Montezuma's actions)
        :param parameters:
        """
        self.action_space = action_space

        self.number_actions = len(action_space)
        self.parameters = parameters

        self.values = list()
        self.state_list = list()
        self.len_state_list = 0
        self.current_state_index = None

    def __len__(self):
        """
        :return: number of states
        """
        return len(self.state_list)

    def __str__(self):
        message = ""
        for idx in range(len(self.state_list)):
            message += str(self.state_list[idx]) + \
                       " values: " + str(self.values[idx]) + \
                       "\n"

        return message

    def reset(self, current_state):
        self._update_states(current_state)

    def _update_states(self, next_state):
        try:
            i = self.state_list.index(next_state)

        except ValueError:
            self.state_list.append(next_state)
            self.values.append(np.zeros(self.number_actions, dtype=np.float64))
            self.len_state_list += 1
            i = self.len_state_list - 1

        self.current_state_index = i

    def _activate_random(self, train_episode):
        # either the random can be triggered by flipping a coin
        rdm = np.random.rand() < self.parameters["probability_random_action_option"] * \
            np.exp(-train_episode * self.parameters["random_decay"])

        # or if all values are equal
        val = np.all(self.values[self.current_state_index][0] == self.values[self.current_state_index])

        return rdm or val

    def find_best_action(self, train_episode=None):
        """
        :param train_episode:
        :return: an action at the lower level
        """
        if (train_episode is not None) and self._activate_random(train_episode):
            return np.random.randint(self.number_actions)

        else:
            return np.argmax(self.values[self.current_state_index])

    def update_policy(self, new_state, reward, action, end_option, train_episode):
        """
        updates the values of the policy and the state list (with an update on the current state).
        Only updates the current state in the simulation phase which corresponds to train_episode == None.
        :param new_state:
        :param reward:
        :param action:
        :param end_option:
        :param train_episode:
        type == None -> simulation,
        type == integer -> training.
        :return: void
        """
        if train_episode is not None:  # training phase: update value
            if end_option:
                best_value = 0
                # todo, take the max with other options
            else:
                try:
                    new_state_idx = self.state_list.index(new_state)
                    best_value = np.max(self.values[new_state_idx])

                except ValueError:
                    best_value = 0

            self.values[self.current_state_index][action] *= (1 - self.parameters["learning_rate"])
            self.values[self.current_state_index][action] += self.parameters["learning_rate"] * (reward + best_value)

        self._update_states(new_state)