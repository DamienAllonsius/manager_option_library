"""
This library can be connected to a gym environment or any kind of environment as long as it has the following methods:
- env.reset
- env.step
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from mo.policies.policy_manager import AbstractPolicyManager
from mo.options.options import AbstractOption
from mo.options.options_explore import AbstractOptionExplore
from mo.utils.save_results import SaveResults
from mo.utils.show_render import ShowRender
from mo.utils.miscellaneous import obs_equal, check_type
from collections import deque
import matplotlib.pyplot as plt


class AbstractManager(metaclass=ABCMeta):
    """
    Abstract Manager class.
    A Manager manages options (see Sutton's framework)
    """

    def __init__(self, action_space, parameters):
        """
        initialize the manager's parameters.
        :param action_space: the actions to be selected by the options.
        :param parameters: a dictionary containing the parameters of the experiment (for the agent and the environment).
        """
        self.action_space = action_space
        self.parameters = parameters
        self.option_list = []
        # the score is only used in the simulation mode
        self.score = 0

        self.policy = self.new_policy()
        self.explore_option = self.new_explore_option()
        self.show_render = None

        self.save_results = SaveResults(self.parameters)
        self.successful_transition = deque(maxlen=100)  # A list of 0 and 1 of size <=100.

        # checks that policy and options have the right type.
        check_type(self.policy, AbstractPolicyManager)
        check_type(self.explore_option, AbstractOptionExplore)
        check_type(self.new_option(), AbstractOption)

    def reset(self, initial_state):
        self.score = 0
        self.policy.reset(initial_state)

    def _train_simulate_manager(self, env, train_episode=None):
        """
        Method used to train or simulate the manager (the main loop)

        a) choose an option
        b) option acts and updates
        c) if a new state is found then update manager

        :param env: the environment.
        :param train_episode: the episode of training.
        - if not None: training
        - if None: simulating
        :return: void
        """
        # The initial observation
        o_r_d_i = [env.reset()] + [None]*3  # o_r_d_i means "Observation_Reward_Done_Info"
        # Reset all the manager parameters
        self.reset(o_r_d_i[0]["manager"])
        done = False
        current_option = None
        # Render the current state
        if self.parameters["display_environment"]:
            self.show_render.render(o_r_d_i[0])

        while not done:
            # If no option is activated then choose one
            if current_option is None:
                current_option = self.select_option(o_r_d_i, train_episode)

            # choose an action
            action = current_option.act(train_episode)

            # make an action and display the state space
            o_r_d_i = env.step(action)
            if self.parameters["display_environment"]:
                self.show_render.render(o_r_d_i[0])

            # check if the option ended correctly
            correct_termination = self.check_end_option(current_option, o_r_d_i[0]["manager"])

            # update the option
            intra_reward = self.get_intra_reward(correct_termination, o_r_d_i[0]["option"], current_option,
                                                 train_episode)
            current_option.update_option(o_r_d_i, intra_reward, action, correct_termination, train_episode)

            # If the option is done, update the manager
            if correct_termination is not None:
                if check_type(type(current_option), AbstractOption):
                    # record the correct transition when the option is a regular option (i.e. not an explore option)
                    self.write_success_rate_transitions(correct_termination)

                # the agent does not need to know if the correct_termination is 0 or 1.
                self.update_agent(o_r_d_i, current_option, train_episode)
                current_option = None

            done = self.check_end_agent(o_r_d_i, current_option, train_episode)

    def select_option(self, o_r_d_i, train_episode=None):
        """
        The manager will
        1) choose an option in option_list
        2) reset the parameters of this option
        3) return the option
        :return: an option
        """
        best_option_index = self.policy.find_best_action(train_episode)
        if best_option_index is None:
            # in this case : explore
            return self.explore_option

        else:
            # set the option at the right position and return it
            self.option_list[best_option_index].reset(o_r_d_i[0]["option"])
            return self.option_list[best_option_index]

    def update_agent(self, o_r_d_i, option, train_episode=None):
        """
        updates the manager parameters.
        In simulation mode, updates
        - score
        In training mode, updates
        - policy
        - option
        :param o_r_d_i : Observation, Reward, Done, Info given by function step
        :param option: the index of the option that did the last action
        :param train_episode: the number of the current training episode
        :return : void
        """
        if train_episode is None:  # in simulation mode
            self.score = self.compute_total_score(o_r_d_i, option, train_episode)

        else:  # in training mode
            self._update_policy(o_r_d_i, option)

            # add a new option if necessary
            missing_option = self.policy.get_max_number_successors() - len(self.option_list)
            assert missing_option == 1 or missing_option == 0, "number of options is wrong"
            if missing_option:
                self.option_list.append(self.new_option())

    def _update_policy(self, o_r_d_i, option):
        self.policy.update_policy(o_r_d_i[0]["manager"], option.score)

    def train_agent(self, environment, seed=0):
        """
        Method used to train the RL manager. It calls function _train_simulate_agent with the current training episode
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare the file for the results
        self.save_results.write_setting()

        # prepare to display the states
        if self.parameters["display_environment"]:
            self.show_render = AbstractManager.get_show_render_train()

        for t in tqdm(range(1, self.parameters["number_episodes"] + 1)):
            self._train_simulate_manager(environment, t)

            if not t % 200:
                self.plot_success_rate_transitions()

        self.show_render.close()

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
        self.save_results.set_file_results_name(seed)

        # prepare to display the states if needed
        if self.parameters["display_environment"]:
            self.show_render = AbstractManager.get_show_render_simulate()

        # simulate
        self._train_simulate_manager(environment)

        # write the results
        self.save_results.write_reward(self.parameters["number_episodes"], self.score)
        self.save_results.write_message("Experiment complete.")

        self.show_render.close()

    def write_success_rate_transitions(self, correct_termination):
        """
        Write in a file the sum of the last 100 transitions.
        A transition is 0 or 1.
        1 if the option terminates at the right abstract state and 0 otherwise.
        :param correct_termination: 1 if the transition is correct. 0 otherwise.
        :return: void
        """
        self.successful_transition.append(correct_termination)
        self.save_results.write_message_in_a_file("success_rate_transition",
                                                  str(sum(self.successful_transition)) + "\n")

    def print_success_rate_transitions(self, correct_termination):
        """
        Print the sum of the last 100 transitions.
        A transition is 0 or 1.
        1 if the option terminates at the right abstract state and 0 otherwise.
        :param correct_termination: 1 if the transition is correct. 0 otherwise.
        :return: void
        """
        self.successful_transition.append(correct_termination)
        print(str(sum(self.successful_transition)) + "%")

    def plot_success_rate_transitions(self):
        f = open(str(self.save_results.dir_path) + "/success_rate_transition")
        lines = f.readlines()
        f.close()
        x = [float(line.split()[0]) for line in lines]
        plt.plot(x)
        plt.title("success rate of options' transitions")
        plt.draw()
        plt.pause(0.01)
        plt.savefig(str(self.save_results.dir_path) + "/success_rate_transition" + "success_rate_transition")
        plt.savefig("success_rate_transition")

    def get_intra_reward(self, correct_termination, next_state, current_option, train_episode):
        """
        returns a reward based on the maximum value of the next_state over all options
        (maybe one should select some options instead of using all options).
        """
        return 0

    def check_end_option(self, option, o_r_d_i):
        """
        check if the option ended and if the termination is correct.
        :param option: explore option or regular option
        :param o_r_d_i:
        :return:
        - None if the option is not done.
        Otherwise:
        if option is an explore option:
        - True
        if option is a regular option:
        - True if ended in the correct new abstract state, False if the new abstract state is wrong.
        """
        if self.get_current_state() == o_r_d_i[0]["manager"]:
            # option is not done
            return None

        else:
            # option is done
            if check_type(type(option), AbstractOptionExplore):
                return True

            elif check_type(type(option), AbstractOption):
                return obs_equal(o_r_d_i[0]["manager"], self.policy.get_next_state(option.index))

            else:
                raise Exception("Option type not supported")

    @staticmethod
    def get_show_render_train():
        return ShowRender()

    @staticmethod
    def get_show_render_simulate():
        return ShowRender()

    # Method to be implemented by the sub classes

    @abstractmethod
    def compute_total_score(self, o_r_d_i, option, train_episode):
        """
        :return: a float corresponding of the score of the manager accumulated so far
        """
        raise NotImplementedError()

    @abstractmethod
    def check_end_agent(self, o_r_d_i, option, train_episode):
        """
        Check if the current episode is over or not.
        The output of this function will update the variable "done" in method self._train_simulate_manager
        :param o_r_d_i:
        :param option:
        :param train_episode:
        :return: True iff the manager is done.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_state(self):
        raise NotImplementedError()

    @abstractmethod
    def new_option(self) -> AbstractOption:
        """
        Make a new option to update the list: self.option_list
        This method depends on the kind of option we want to use.
        :return: a class which inherits from AbstractOption
        """
        raise NotImplementedError()

    @abstractmethod
    def new_explore_option(self) -> AbstractOptionExplore:
        """
        Make a new option explore
        :return: a class which inherits from AbstractOptionExplore
        """
        raise NotImplementedError()

    @abstractmethod
    def new_policy(self) -> AbstractPolicyManager:
        """
        make a new policy for the manager
        :return: a class which inherits from AbstractPolicyManager
        """
        raise NotImplementedError()
