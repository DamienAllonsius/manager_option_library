import numpy as np
from ao.policies.agent.agent_policy import PolicyAbstractAgent
from ao.utils.miscellaneous import obs_equal


class QGraph(PolicyAbstractAgent):
    """
    TESTED.
    A policy for options.
    Number of states unknown.
    Number of actions known
    """

    def __init__(self, parameters):
        self.parameters = parameters

        self.states = []  # list of states (those can be of any type)
        self.values = []  # values[i][j] : value of action j at state i
        self.state_graph = []
        self.current_state_index = None  # index of the current state

        self.len_state_list = 0
        self.max_states = 0  # maximum of: number of edges, over: every nodes

    def __str__(self):
        """
        test ok
        :return: a string representation of the graph
        """
        s = ""
        for idx, elements in zip(range(self.len_state_list), self.state_graph):
            s += str(idx) + ": "
            for e in elements:
                s += str(e) + ", "

            s += "\n"

        return s

    def reset(self, state):
        """
        The agent has to reset the option's Q function BEFORE to ask him to make an action
        in order to update self.current_state first.
        :param state:
        :return: void
        """
        if self.get_current_state() is None or not obs_equal(state, self.get_current_state()):
            self._update_states(state)

    def _update_states(self, state):
        """
        add state in the graph at the current_state
        :param state:
        :return: void
        """
        if self.states == list():
            self.states.append(state)
            self.values.append([])
            self.state_graph.append([])
            self.current_state_index = 0

            self.len_state_list = 1
            self.max_states = 1

        else:
            # Add the new state in the list of states and in the graph.
            # Get the index i of this state
            try:
                # Try to reach the index leading to this state
                i = self.states.index(state)

                # In the graph: add i in the child nodes of the current state
                if i not in self.state_graph[self.current_state_index]:
                    self.state_graph[self.current_state_index].append(i)
                    self.values[self.current_state_index].append(0)

            except ValueError:
                self.states.append(state)
                self.len_state_list += 1
                i = self.len_state_list - 1

                # add the state index i as a new node since it did not already exist
                self.state_graph.append([])
                self.values.append([])

                # add a new node and connect it to the current_state_index
                self.state_graph[self.current_state_index].append(i)
                self.values[self.current_state_index].append(0)

            # update the maximum number of states
            if len(self.state_graph[self.current_state_index]) > self.max_states:
                self.max_states += 1

            self.current_state_index = i

    def find_best_action(self, train_episode=None):
        """
        :param train_episode: if not None -> training mode, potentially activate the explore option
        :return: best_option_index, terminal_state
        - in test mode : the best_option_index and the corresponding terminal state
        - in training mode : the best value possible OR (None,None) -> signal to activate the explore_option.
        """
        # todo change this condition
        if not self.state_graph[self.current_state_index]:  # no alternative: explore
            return None, None

        if (train_episode is not None) and (np.random.rand() < self.parameters["probability_random_action_agent"]):
            return None, None

        else:
            # Should implement the case where all actions have the same value.
            option_index = np.argmax(self.values[self.current_state_index])
            state_index = self.state_graph[self.current_state_index][option_index]
            return option_index, self.states[state_index]

    def update_policy(self, new_state, reward, action, train_episode):
        if train_episode is None:  # simulating phase
            self._update_states(new_state)

        else:  # training phase
            if action is not None:  # update only if there is a previous action
                self.update_value(new_state, reward, action)

            self._update_states(new_state)

    def update_value(self, new_state, reward, action):
        """
        updates self.values
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_t(new_position, action)]
        :param new_state: the new observation
        :param reward: the reward gotten
        :param action: the last action chosen
        :return: void
        """
        try:
            new_index = self.states.index(new_state)
            best_value = np.max(self.values[new_index])

        except ValueError:
            best_value = 0

        self.values[self.current_state_index][action] *= (1 - self.parameters["learning_rate"])
        self.values[self.current_state_index][action] += self.parameters["learning_rate"] * (reward + best_value)

    def get_random_action(self):
        """
        test ok
        :return: a random integer corresponding to a state of self.states
        """
        return np.random.choice(self.state_graph[self.current_state_index])

    def get_max_number_successors(self):
        return self.max_states

    def get_current_state(self):
        """
        :return: the current state (not the index, rather the node.data)
        """
        if not self.states:  # no states !
            return None
        else:
            return self.states[self.current_state_index]
