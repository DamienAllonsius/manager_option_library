"""
This library can be connected to a gym environment or any kind of environment as long as it has the following methods:
- env.reset
- env.step
"""
import numpy as np

from agents.options.options_explore import OptionRandomExplore
from agents.utils.utils import SaveResults
from abc import ABCMeta, abstractmethod
from tqdm import tqdm


class AbstractAgent(object):
    """
    Very general abstract skeleton for Agent class for any kind of purpose
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def reset(self, initial_state):
        """
        resets the state space parameter.
        :param initial_state: an element of the state space
        :return: void
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        makes an action
        :param args:
        :param kwargs:
        :return: an element of the action space
        """
        raise NotImplementedError()

    @abstractmethod
    def update_agent(self, *args, **kwargs):
        """
        updates the agent parameters (the policy for instance)
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def train_agent(self, *args, **kwargs):
        """
        Performs the training phase
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def simulate_agent(self, *args, **kwargs):
        """
        Performs the simulation phase
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()


class AbstractAgentOption(AbstractAgent):
    """
    Abstract Agent class with the Options framework
    """

    __metaclass__ = ABCMeta

    def __init__(self, action_space, parameters):
        """
        initialize the agent's parameters.
        :param action_space:
        :param parameters:
        """

        self.action_space = action_space
        self.parameters = parameters
        self.policy = self.get_policy()
        self.explore_option = self.get_explore_option()
        self.option_list = []
        self.score = 0

    def __len__(self):
        return len(self.option_list)

    def reset(self, initial_state):
        self.score = 0
        self.policy.reset(initial_state)

    def _get_option(self, index=None):
        """
        return the option corresponding to the given index
        :param index: if None -> exploration option, else -> self.option_list[index]
        :return: an option
        """
        if index is not None:
            return self.option_list[index]

        else:
            return self.explore_option

    def _train_simulate_agent(self, environment, train_episode=None):
        """
        Method used to train or simulate the agent

        a) choose an option
        b) option acts and updates
        c) if a new state is found then update agent

        :param environment:
        :param train_episode: the episode of training.
        :return:
        """

        # The initial observation
        obs = environment.reset()

        # Reset all the parameters
        self.reset(obs["agent"])
        done = False
        option_index = None

        # Render the current state
        self.display_state(environment, train_episode)

        while not done:
            # If no option is activated then choose one
            if option_index is None:
                option_index = self.act(obs["agent"], train_episode)

                # get the corresponding option
                option_chosen = self._get_option(option_index)

            # choose an action
            action = option_chosen.act()

            # make an action and display the state space
            #  todo record the learning curve
            o_r_d_i = environment.step(action)

            self.display_state(environment, train_episode)

            # update the option
            option_chosen.update_option(o_r_d_i, action, train_episode)

            # check if the option ended
            end_option = option_chosen.check_end_option(o_r_d_i[0]["agent"])

            # If the option is done, update the agent
            if end_option:
                self.update_agent(o_r_d_i, option_index, train_episode)
                done = self.check_end_agent(o_r_d_i, option_index, train_episode)
                option_index = None

    def train_agent(self, environment, seed=0):
        """
        Method used to train the RL agent. It calls function _train_simulate_agent with the current training episode
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        for t in tqdm(range(1, self.parameters["number_episodes"])):
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

    def act(self, state, train_episode):
        """
        the action for an Agent in the Option framework corresponds to
        1) choose an option in option_list
        2) reset the parameters of this option
        3) return the integer corresponding to this option

        :return: an integer. It corresponds to an option in option_list.
        """
        best_option_index, terminal_state = self.policy.find_best_action(train_episode)

        if terminal_state is None:

            # in this case : explore
            self.explore_option.reset(state)

            # None is the signal that the exploration option has to be chosen by the agent
            return None

        else:  # in this case, activate an option from the list self.option_set

            # get the right initial, current and terminal states for the option
            option_states = self.get_option_states(terminal_state)

            # set the parameters  of the option with that states
            self.option_list[best_option_index].reset(option_states)

            return best_option_index

    def update_agent(self, o_r_d_i, option_index, train_episode=None):
        """
        update the agent parameters:
        - score (only in testing mode)
        - policy and option_list (only in learning mode)
        - update the policy's state list
        :param o_r_d_i : Observation, Reward, Done, Info given by function step
        :param option_index: the index of the option that did the last action
        :param train_episode: the number of the current training episode
        :return : void
        """

        if train_episode is None:  # in simulate mode

            # compute total score
            self.score = self.compute_total_score(o_r_d_i, option_index, train_episode)

        else:  # in training mode

            self._update_policy(o_r_d_i, option_index, train_episode)

            # add a new option if necessary
            if self.policy.max_number_successors > len(self):
                self.append_new_option()

        self.policy.update_states(o_r_d_i)

    def _update_policy(self, o_r_d_i, option_index, train_episode):

        # first, compute the total reward to update the policy
        total_reward = self.compute_total_reward(o_r_d_i, option_index, train_episode)

        # update the q value only if the option is not the explore_option
        if option_index is not None:
            self.policy.update_policy(o_r_d_i, self.option_list[option_index], total_reward)

    def get_explore_option(self):
        """
        can be overwritten if needed
        :return:
        """
        return OptionRandomExplore(self.action_space)

    @staticmethod
    @abstractmethod
    def display_state(environment, train_episode):
        raise NotImplementedError()

    @abstractmethod
    def get_option_states(self, terminal_state):
        """
        This function highly depends on the environment.
        It get the initial and terminal states for the option. Used to reset the option just before to activate it.
        :param terminal_state: the source of terminal_state for the option
        :return: initial_state, current_state, terminal_state
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_total_score(self, o_r_d_i, option_index, train_episode):
        """
        This function highly depends on the environment.
        :return: a float corresponding of the score of the agent accumulated so far
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_total_reward(self, o_r_d_i, option_index, train_episode):
        """
        This function highly depends on the environment.
        :return: a float corresponding of the current reward of the agent
        """
        raise NotImplementedError()

    @abstractmethod
    def append_new_option(self):
        """
        Updates the list: self.option_list
        This method depends on the kind of option we want to use.
        :return: void
        """
        raise NotImplementedError()

    @abstractmethod
    def get_policy(self):
        """
        :return: An instance of a class which inherits from PolicyAbstract
        """
        raise NotImplementedError()

    @abstractmethod
    def check_end_agent(self, o_r_d_i, option_index, train_episode):
        """
        Check if the current episode is over or not. The output of this function will update the variable "done" in
        method self._train_simulate_agent
        :param o_r_d_i:
        :param option_index:
        :param train_episode:
        :return: True iff the agent is done.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_state(self):

        raise NotImplementedError()
