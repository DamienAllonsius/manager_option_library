from ao.agent.agent import AbstractAgentOption
from ao.examples.options_examples import OptionQArray
from ao.examples.policy_examples_agent import QGraph
from ao.options.options import OptionAbstract
from ao.options.options_explore import OptionRandomExplore
from ao.agent.agent import AbstractAgent
from tqdm import tqdm
import numpy as np
from ao.examples.policy_examples_option import QArray
from ao.utils.save_results import SaveResults
from ao.utils.show_render import ShowRender
from abc import abstractmethod


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

    def compute_total_reward(self, o_r_d_i, option_index, train_episode):
        """
        :param o_r_d_i:
        :param option_index:
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


class PlainQLearning(AbstractAgent):
    """
    Plan Q Learning implementation.
    *NOT TESTED*
    """

    def __init__(self, action_space, parameters):
        self.parameters = parameters
        self.current_state = None
        self.policy = QArray(action_space, parameters)
        self.score = 0
        self.show_render = None

    def _train_simulate_agent(self, env, train_episode=None):
        # reset the parameters
        obs = env.reset()
        self.reset(obs)
        done = False

        # render the image
        if self.parameters["display_environment"]:
            self.show_render.display()

        while not done:
            # choose an action
            action = self.act(train_episode)

            # get the output
            o_r_d_i = env.step(action)

            # update the agent
            self.update_agent(o_r_d_i, action, train_episode)

            # display the observation if needed
            if self.parameters["display_environment"]:
                self.show_render.display()

            # update variable done
            done = self.check_end_agent(o_r_d_i)

    def train_agent(self, environment, seed=0):
        """
        Method used to train the RL agent. It calls function _train_simulate_agent with the current training episode
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare to display the states
        if self.parameters["display_environment"]:
            self.show_render = ShowRender()

        for t in tqdm(range(1, self.parameters["number_episodes"] + 1)):
            self._train_simulate_agent(environment, t)

    def simulate_agent(self, environment, seed=0):
        """
        Method used to train the RL agent.
        It calls _train_simulate_agent method with parameter "train_episode" set to None
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare the file for the results
        save_results = SaveResults(self.parameters)
        save_results.write_setting()
        save_results.set_file_results_name(seed)

        # simulate
        self._train_simulate_agent(environment)

        # write the results and write that the experiment went well
        save_results.write_reward(self.parameters["number_episodes"], self.score)
        save_results.write_message("Experiment complete.")

    def reset(self, initial_state):
        self.policy.reset(initial_state)

    @staticmethod
    def compute_total_score(o_r_d_i):
        return o_r_d_i[1]

    def act(self, train_episode):
        if (train_episode is not None) and (np.random.rand() < self.parameters["probability_random_action_agent"]):
            return self.policy.get_random_action()

        else:
            return self.policy.find_best_action(train_episode)

    def update_agent(self, o_r_d_i, action, train_episode):
        # update the policy
        total_reward = self.compute_total_reward(o_r_d_i, train_episode)
        self.policy.update_policy(o_r_d_i[0], total_reward, action, False, train_episode)

        # update the score
        self.score += self.compute_total_score(o_r_d_i)

    @abstractmethod
    def compute_total_reward(self, o_r_d_i, train_episode):
        raise NotImplementedError()

    @abstractmethod
    def check_end_agent(self, *args, **kwargs) -> bool:
        raise NotImplementedError()
