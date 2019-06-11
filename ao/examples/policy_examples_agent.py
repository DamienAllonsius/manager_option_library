import numpy as np
from ao.policies.agent.agent_policy import PolicyAbstractAgent


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
            assert (state != self.get_current_state())
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

    def update_policy(self, new_state, reward, action):
        if action is None:
            self._update_states(new_state)

        else:
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

    def max_number_successors(self):
        return self.max_states

    def get_current_state(self):
        """
        :return: the current state (not the index, rather the node.data)
        """
        return self.states[self.current_state_index]

# class QTree(PolicyAbstractAgent):
#     """
#     This class is used when the number of actions and the number of states are unknown.
#     """
#
#     def __init__(self, parameters, action_space):
#         self.parameters = parameters
#         self.action_space = action_space
#         self.tree = None
#
#     def __len__(self):
#         return len(self.tree)
#
#     def __str__(self):
#         return str(self.tree)
#
#     def reset(self, initial_state):
#         if self.tree is None:
#             self.tree = Tree(initial_state)
#
#         try:
#             self.tree.set_current_node(self.tree.get_node_from_state(initial_state))
#
#         except ValueError:
#             self.tree.add_node(Node(initial_state))
#
#     def find_best_action(self, state):
#         """
#         returns the best value and the best action. If best action is None, then explore.
#         :return: best_option_index, terminal_state
#         """
#         values = self.tree.get_values_children()
#         if not values:
#             return 0, None
#
#         if all(val == values[0] for val in values):  # all the values are the same
#
#             # choose a random action
#             best_option_index = self.tree.get_random_child_index()
#
#         else:  # there exists a proper best action
#             best_reward = max(values)
#             best_option_index = values.index(best_reward)
#
#         return best_option_index, self.tree.get_current_node().children[best_option_index].data
#
#     def update(self, action, reward, new_state):
#         """
#         Performs the Q learning update :
#         Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
#                                          += learning_rate * [reward + max_{actions} Q_(new_position, action)]
#
#         """
#         node which value attribute is Q_t(current_position, action)
#         node_activated = self.tree.get_child_node(action)
#
#         try:
#             new_node = self._get_node_from_state(new_state)  # maybe different than node_activated
#             if new_node.children:  # there are children, take the maximum value
#                 best_value = max(new_node.get_values())
#
#             else:  # there are no children -> best_value is 0
#                 best_value = 0
#
#         except ValueError:  # this new_state does not exist for the moment
#             best_value = 0
#
#         node_activated.value *= (1 - self.parameters["learning_rate"])
#         node_activated.value += self.parameters["learning_rate"] * (reward + best_value)
#
#         self._update_states(new_state)
#
#     def max_number_successors(self):
#         return self.tree.get_max_width()
#
#     def get_current_state(self):
#         return self.tree.get_current_node().data
#
#    def _no_return_update(self, new_state):
#        """
#      (no return option)
#            does not add anything if
#            for action in q[option.terminal_state]:
#            action.terminal_state = option.initial_state
#        """
#        try:
#            new_node = self._get_node_from_state(new_state)
#            for node in new_node.children:
#                if node.data == self.current_node.data:
#                    return False
#            return True
#        except ValueError:
#            return True

