"""
parameters:
probability_explore_in_option
reward_end_option
penalty_end_option
penalty_option_action
"""
from abc import ABCMeta, abstractmethod
from ao.policies.option.option_policy import PolicyAbstractOption


class OptionAbstract(metaclass=ABCMeta):
    """
    Abstract option class that barely only needs update, reset and act functions.
    """

    def __init__(self, action_space, parameters, index):
        """
        initial_state: where the option starts;
        terminal_state: where the option has to go;
        type_policy: the name of the class used for the policy
        """
        self.action_space = action_space
        self.parameters = parameters
        self.score = 0
        self.initial_state = None
        self.terminal_state = None
        self.activated = False  # safeguard that indicated if this option has been activated correctly
        self.index = index

    def __repr__(self):
        return "".join(["Option(", str(self.initial_state), ",", str(self.terminal_state), ")"])

    def __str__(self):
        return "option from " + str(self.initial_state) + " to " + str(self.terminal_state)

    @abstractmethod
    def update_option(self,  *args, **kwargs):
        """
        Updates the option's characteristics
        :return void
        """
        raise NotImplementedError()

    def check_end_option(self, new_state):
        """
        Checks if the option has terminated or not.
        The new_state must be *of the same form* as the initial_state (transformed or not).
        :param new_state:
        :return: True if the new_state is different from the initial state
        """
        end_option = (new_state != self.initial_state)
        if end_option:
            self.activated = False

        return end_option

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        Performs an action
        :return: an integer in range(self.number_actions)
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        reset parameters and turn activation to True: self.activated = True
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def reset_states(self, initial_state, terminal_state):
        """
        reset the initial and terminal states
        :param initial_state: where the option starts;
        :param terminal_state: where the option has to go;
        :return: void
        """
        self.initial_state = initial_state
        self.terminal_state = terminal_state

    @abstractmethod
    def compute_total_reward(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return: the total reward earned by the last action
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_total_score(self, *args, **kwargs):
        """
        update self.score: an overall score accumulated so far
        :param args:
        :param kwargs:
        :return: void
        """
        raise NotImplementedError()


class OptionQLearning(OptionAbstract):
    """
    Another Abstract class where its policy is updated with Q learning algorithm.
    The policy is stored and computed through variable *policy* which inherits from PolicyAbstract class
    """

    def __init__(self, action_space, parameters, index):
        """
        attribute self.policy represents the q function
        """
        super().__init__(action_space, parameters, index)
        self.policy = self.get_policy()

    def update_option(self, o_r_d_i, action, train_episode=None):
        """
        updates the parameters of the option, in particular self.policy.
        Train mode and simulate mode are distinguished by the value of train_episode.
        self.policy.update_policy :
        - In simulation mode : updates only the current state.
        - In learning mode : updates also the values of Q function.
        :param o_r_d_i:  Observation, Reward, Done, Info
        :param action: the last action performed
        :param train_episode: the number of the current training episode
        :return: void
        """
        # check if the option is done
        end_option = self.check_end_option(o_r_d_i[0]["agent"])

        # compute the rewards
        total_reward = self.compute_total_reward(o_r_d_i, action, end_option)

        # update the q function
        self.policy.update_policy(o_r_d_i[0]["option"], total_reward, action, end_option, train_episode)

        # compute the total score
        self.score = self.compute_total_score(o_r_d_i, action, end_option, train_episode)

    def act(self, train_episode=None):
        """
        An epsilon-greedy policy based on the values of self.policy (Q function)
        :param train_episode: if not None -> training phase. If None -> test phase
        :return: an action at the lower level
        """
        assert self.activated
        ba = self.policy.find_best_action(train_episode)
        return ba

    def reset(self, initial_state, current_state, terminal_state):
        """
        Reset the initial and terminal states
        :param initial_state: where the option starts;
        :param current_state: where the option is;
        :param terminal_state: where the option has to go;
        :return: void
        """
        # reset variables self.initial_state and self.terminal_state
        super().reset_states(initial_state, terminal_state)

        # add the new state if needed and update the current state of self.policy
        self.policy.reset(current_state)

        # activate the option
        self.activated = True

    @abstractmethod
    def compute_total_reward(self, o_r_d_i, action, end_option):
        """
        Computes the total reward following actions to update the policy
        This function depends on the environment in which the agent evolves.
        The child class will compute the total reward here, depending on the specific
        features of the environment.
        :param o_r_d_i: Observation, Reward, Done, Info
        :param action: The last action performed
        :param end_option: True iff the option ended
        :return: The total reward for a given step. This is needed to update the policy.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_total_score(self, *args, **kwargs):
        """
        updates self.score: an overall score accumulated so far. This score is used by the agent
        :param args:
        :param kwargs:
        :return: void
        """
        raise NotImplementedError()

    @abstractmethod
    def get_policy(self) -> PolicyAbstractOption:
        """
        Defines the class policy that this option will use
        :return:
        """
        raise NotImplementedError()
