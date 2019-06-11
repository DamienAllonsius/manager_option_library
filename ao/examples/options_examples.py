from ao.options.options import OptionQLearning
from ao.examples.policy_examples_option import QArray


class OptionQArray(OptionQLearning):
    """
    Test ok
    Example of an OptionQLearning using an Array policy.
    """

    def compute_total_reward(self, o_r_d_i, action, end_option):
        """
        test ok
        :param o_r_d_i:
        :param action:
        :param end_option:
        :return:
        """
        total_reward = o_r_d_i[1]
        if end_option:
            total_reward += (self.terminal_state == o_r_d_i[0]["agent"]) * self.parameters["reward_end_option"]
            total_reward += (self.terminal_state != o_r_d_i[0]["agent"]) * self.parameters["penalty_end_option"]

        return total_reward

    @staticmethod
    def compute_total_score(o_r_d_i, action, end_option, train_episode):
        """
        test prepared
        :param o_r_d_i:
        :param action:
        :param end_option:
        :param train_episode:
        :return:
        """
        if train_episode is None:
            return o_r_d_i[1]

        else:  # Todo
            return o_r_d_i[1]

    def get_policy(self):
        return QArray(action_space=self.action_space, parameters=self.parameters)
