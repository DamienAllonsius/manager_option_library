import numpy as np
from abc import ABCMeta, abstractmethod
from agents.policies import QArray


class OptionAbstract(object):
    """
    Abstract option class that barely only needs update, reset and act functions.
    """
    __metaclass__ = ABCMeta

    def __init__(self, action_space, parameters):
        """
        initial_state: where the option starts;
        terminal_state: where the option has to go;
        current_state: where the option actually is;
        """
        self.action_space = action_space
        self.parameters = parameters
        self.initial_state = None
        self.current_state = None
        self.terminal_state = None

    def __repr__(self):
        return "".join(["Option(", str(self.initial_state), ",", str(self.terminal_state), ")"])

    def __str__(self):
        return "option from " + str(self.initial_state) + " to " + str(self.terminal_state)

    @abstractmethod
    def update_option(self,  *args, **kwargs):
        """
        Updates the option's characteristics
        :return a boolean which is True iff the option is done.
        This function can return, for instance, the output of function check_end_option(self, new_state)
        """
        raise NotImplementedError()

    def check_end_option(self, new_state):
        """
        Checks if the option has terminated or not.
        The new_state must be *of the same form* as the initial_state (transformed or not).
        :param new_state:
        :return: True if the new_state is different from the initial state
        """
        return new_state != self.initial_state

    @abstractmethod
    def act(self):
        """
        Performs an action
        :return: an integer in range(self.number_actions)
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError()

    def reset_states(self, initial_state, current_state, terminal_state):
        """
        reset the initial, current and terminal states
        :param initial_state: where the option starts;
        :param current_state: where the option has to go;
        :param terminal_state: where the option actually is;
        :return: void
        """
        self.initial_state = initial_state
        self.current_state = current_state
        self.terminal_state = terminal_state

    @abstractmethod
    def compute_total_reward(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return: the total reward for a given step to update the policy
        """
        raise NotImplementedError()


class OptionExplore(OptionAbstract):

    def update_option(self, new_state):
        """
        Nothing to update, the policy is random
        :return: true iif the option is done
        """
        return self.check_end_option(new_state)

    def act(self):
        """
        :return: a random action from the action space
        """
        return np.random.choice(self.action_space)

    def reset(self, initial_state):
        """
        Only the initial_state has to be reset since the option acts randomly
        :param initial_state:
        :return: void
        """
        self.initial_state = initial_state

    def compute_total_reward(self):
        """
        Nothing to compute here
        :return: void
        """
        pass

    def __repr__(self):
        return "".join(["Option_explore(",str(self.terminal_state), ")"])

    def __str__(self):
        return "explore option from " + str(self.initial_state)


class OptionQLearning(OptionAbstract):
    """
    Another Abstract class where its policy is upgraded with Q learning algorithm.
    The policy is stored and computed through variable *q* which is a list of arrays
    (number of actions known, number of states unknown)
    """
    __metaclass__ = ABCMeta

    def __init__(self, action_space, parameters):
        """
        attribute self.q represents the q function.
        """
        super().__init__(action_space, parameters)
        self.q = None

    @abstractmethod
    def update_option(self, *args, **kwargs):
        """
        updates the parameters of the option, in particular self.q.
        :param args:
        :param kwargs:
        :return: void
        """
        raise NotImplementedError()

    def update_q(self, total_reward, new_state, action):
        """
        Performs a general q learning update on self.q
        :param total_reward:
        :param new_state:
        :param action:
        :return: True iif the option ended
        """
        end_option = self.check_end_option(new_state)

        # Update the states/actions of Q function and compute the corresponding value
        self.q.add_state(new_state)
        self.q.update_q_value(self.current_state, action, total_reward, new_state, end_option,
                              self.parameters["learning_rate"])

        return end_option

    def act(self):
        """
        An epsilon-greedy policy based on the values of self.q (Q function)
        :return: an action at the lower level
        """
        if np.random.rand() < self.parameters["probability_explore_in_option"]:
            best_action = self.q.get_random_action(self.current_state)

        else:
            best_action = self.q.find_best_action(self.current_state)

        return best_action

    def reset(self, initial_state, current_state, terminal_state):
        """
        Reset the initial, current and terminal states
        :param initial_state: where the option starts;
        :param current_state: where the option has to go;
        :param terminal_state: where the option actually is;
        :return: void
        """
        super().reset_states(initial_state, current_state, terminal_state)

        if self.q is None:
            self.q = QArray(current_state, self.action_space)

        self.q.add_state(current_state)

    @abstractmethod
    def compute_total_reward(self, *args, **kwargs):
        """
        This function depends on the environment in which the agent evolves.
        The child class will compute the total reward here, depending on the specific
        features of the environment.
        :param args:
        :param kwargs:
        :return: The total reward for a given step. This is needed to update the policy.
        """
        raise NotImplementedError()
