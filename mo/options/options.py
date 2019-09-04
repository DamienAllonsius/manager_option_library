from abc import ABCMeta, abstractmethod
from mo.policies.policy_option import PolicyOptionQArray


class AbstractOption(metaclass=ABCMeta):
    """
    Abstract option class.
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
        self.index = index

    def __str__(self):
        return "option " + str(self.index)

    def update_option(self, o_r_d_i, action, correct_termination, train_episode=None):
        """
        updates the parameters of the option
        Train mode and simulate mode are distinguished by the value of train_episode.
        :param o_r_d_i:  Observation, Reward, Done, Info
        :param action: the last action performed
        :param correct_termination: None -> option is not done. True, False -> terminated correctly or not.
        :param train_episode: the number of the current training episode
        :return: void
        """
        # compute the rewards
        total_reward = self.compute_total_reward(o_r_d_i, correct_termination)

        # update the parameters
        end_option = correct_termination is not None
        self.update_option_policy(o_r_d_i[0]["option"], total_reward, action, end_option, train_episode)

        # compute the total score
        self.score = self.compute_total_score(o_r_d_i, action, correct_termination)

    def compute_total_score(self, o_r_d_i, action, correct_termination):
        return self.score + o_r_d_i[1]

    def compute_total_reward(self, o_r_d_i, correct_termination):
        total_reward = o_r_d_i[1]

        if correct_termination is None:
            return total_reward

        if correct_termination:
            total_reward += self.parameters["reward_end_option"]

        else:
            total_reward += self.parameters["penalty_end_option"]

        return total_reward

    # methods that have to be implemented by the sub classes.

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        Performs an action
        :return: an integer in range(self.number_actions)
        """
        raise NotImplementedError()

    @abstractmethod
    def update_option_policy(self, state, total_reward, action, correct_termination, train_episode):
        """
        update the options's policy
        :return: an integer in range(self.number_actions)
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        reset parameters
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()


class OptionQLearning(AbstractOption):
    """
    Example of option using a policy.
    Policy is updated with Q learning algorithm.
    The policy is stored and computed through variable *policy* which inherits from PolicyAbstract class
    """
    def __init(self, action_space, parameters, index):
        super().__init__(action_space, parameters, index)
        self.policy = PolicyOptionQArray(action_space, parameters)

    def act(self, train_episode=None):
        """
        An epsilon-greedy policy based on the values of self.policy (Q function)
        :param train_episode: if not None -> training phase. If None -> test phase
        :return: an action at the lower level
        """
        return self.policy.find_best_action(train_episode)

    def update_option_policy(self, state, total_reward, action, end_option, train_episode):
        self.policy.update_policy(state, total_reward, action, end_option, train_episode)

    def reset(self, initial_state, current_state, terminal_state):
        """
        Reset the initial and terminal states
        :param initial_state: where the option starts;
        :param current_state: where the option is;
        :param terminal_state: where the option has to go;
        :return: void
        """
        # add the new state if needed and update the current state of self.policy
        self.policy.reset(current_state)
