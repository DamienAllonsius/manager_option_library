from ao.agent.agent import AbstractAgent, AbstractAgentOption
from ao.examples.options_examples import OptionQArray
from ao.examples.policy_examples_agent import QGraph
from ao.options.options import OptionAbstract
from ao.utils.utils import ShowRender
from ao.options.options_explore import OptionRandomExplore


class AgentOptionMontezuma(AbstractAgentOption):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
        self.show_render = None

    def get_option_states(self, o_r_d_i, terminal_state):
        """
        :param terminal_state:
        :param o_r_d_i:
        :return: the initial, current and terminal states
        """
        return o_r_d_i[0]["agent"], o_r_d_i[0]["option"], terminal_state

    def compute_total_score(self, o_r_d_i, current_option, train_episode):
        return o_r_d_i[1]

    def compute_total_reward(self, o_r_d_i, current_option, train_episode):
        """
        todo get a better parametrization
        KISS for the moment
        :param o_r_d_i:
        :param current_option:
        :param train_episode:
        :return:
        """
        return o_r_d_i[1]

    def get_policy(self):
        return QGraph(self.parameters)

    def check_end_agent(self, o_r_d_i, current_option, train_episode):
        """
        :param o_r_d_i:
        :param current_option:
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

    def get_explore_option(self):
        """
        can be overwritten if needed
        :return:
        """
        return OptionRandomExplore(self.action_space)

    def get_option(self) -> OptionAbstract:
        return OptionQArray(self.action_space, self.parameters, len(self))


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
