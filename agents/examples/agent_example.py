from agents.agent.agent import AbstractAgent, AbstractAgentOption
from agents.examples.options_examples import OptionQArray
from agents.examples.policy_examples_agent import QGraph
from agents.utils.utils import ShowRender


class AgentOptionMontezuma(AbstractAgentOption):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
        self.show_render = None

    def get_option_states(self, terminal_state):
        """
        :param terminal_state:
        :return: the initial and terminal states
        """
        return self.policy.get_current_state(), terminal_state

    def compute_total_score(self, o_r_d_i, option_index, train_episode):
        return o_r_d_i[1]

    def compute_total_reward(self, o_r_d_i, option_index, train_episode):
        """
        todo get a better parametrization
        KISS for the moment
        :param o_r_d_i:
        :param option_index:
        :param train_episode:
        :return:
        """
        return o_r_d_i

    def append_new_option(self):
        self.option_list.append(OptionQArray(self.action_space, self.parameters))

    def get_policy(self):
        return QGraph(self.parameters)

    def check_end_agent(self, o_r_d_i, option_index, train_episode):
        """
        :param o_r_d_i:
        :param option_index:
        :param train_episode:
        :return: True iff the character lost all his lives.
        """
        return o_r_d_i[2]

    def display_state(self, environment, train_episode):
        if self.show_render is None:
            self.show_render = ShowRender(environment)

        self.show_render.display()

    def get_current_state(self):
        return self.policy.get_current_state()


class AgentQLearning(AbstractAgent):

    def reset(self, initial_state):
        pass

    def act(self, *args, **kwargs):
        pass

    def update_agent(self, *args, **kwargs):
        pass

    def train_agent(self, *args, **kwargs):
        pass

    def simulate_agent(self, *args, **kwargs):
        pass
