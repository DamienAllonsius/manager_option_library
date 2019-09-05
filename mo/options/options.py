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

    def compute_goal_reward(self, correct_termination):
        """
        A function that computes the reward or the penalty gotten when terminating.
        :param correct_termination:
        :return:
        """
        goal_reward = 0
        if correct_termination is None:
            return goal_reward

        if correct_termination:
            goal_reward += self.parameters["reward_end_option"]

        else:
            goal_reward += self.parameters["penalty_end_option"]

        return goal_reward

    # methods that have to be implemented by the sub classes.
    @abstractmethod
    def act(self, train_episode):
        """
        Performs an action
        :return: an integer in range(self.number_actions)
        """
        raise NotImplementedError()

    @abstractmethod
    def update_option(self, o_r_d_i, action, correct_termination, train_episode=None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, state):
        """
        reset parameters
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

    def reset(self, state):
        """
        Reset the current state
        :return: void
        """
        self.policy.reset(state)

    def update_option(self, o_r_d_i, action, correct_termination, train_episode=None):
        # compute the rewards
        total_reward = o_r_d_i[1] + self.compute_goal_reward(correct_termination)

        # update the parameters
        end_option = correct_termination is not None
        self.policy.update_policy(o_r_d_i[0]["option"], total_reward, action, end_option, train_episode)

        # compute the total score
        self.score += o_r_d_i[1]
