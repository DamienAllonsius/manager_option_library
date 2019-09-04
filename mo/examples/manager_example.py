from mo.manager.manager import AbstractManager
from mo.examples.options_examples import OptionQArray
from mo.examples.policy_examples_manager import QGraph
from mo.options.options import AbstractOption
from mo.options.options_explore import OptionExplore
from tqdm import tqdm
import numpy as np
from mo.examples.policy_examples_option import QArray
from mo.utils.save_results import SaveResults
from mo.utils.show_render import ShowRender
from abc import abstractmethod


class ManagerMontezuma(AbstractManager):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
        self.show_render = None

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

    def new_policy(self):
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

    def new_explore_option(self):
        """
        can be overwritten if needed
        :return:
        """
        return OptionExplore(self.action_space)

    def new_option(self) -> AbstractOption:
        return OptionQArray(self.action_space, self.parameters, len(self.option_list))


class PlainQLearning():
    """
    Plan Q Learning implementation.
    *NOT TESTED*
    """

    def __init__(self, action_space, parameters):
        self.parameters = parameters
        self.current_state = None
        self.policy = QArray(action_space, parameters)
        self.score = 0
        self.show_render = None  # useful to resize the observation

    def _train_simulate_agent(self, env, train_episode=None):
        # reset the parameters
        obs = env.reset()
        self.reset(obs)
        done = False

        # render the image
        if self.parameters["display_environment"]:
            self.show_render.render(obs)

        while not done:
            # choose an action
            action = self.act(train_episode)

            # get the output
            o_r_d_i = env.step(action)

            # update the manager
            self.update_agent(o_r_d_i, action, train_episode)

            # display the observation if needed
            if self.parameters["display_environment"]:
                self.show_render.render(o_r_d_i[0])

            # update variable done
            done = self.check_end_agent(o_r_d_i)

    def train_agent(self, environment, seed=0):
        """
        Method used to train the RL manager. It calls function _train_simulate_agent with the current training episode
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
        Method used to train the RL manager.
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
