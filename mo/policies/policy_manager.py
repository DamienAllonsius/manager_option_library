"""
Policies that can be applied only on agents
"""
from abc import ABCMeta, abstractmethod
from mo.utils.miscellaneous import red, white, obs_equal, find_element_in_list
from mo.structures.tree import Tree
import numpy as np


class AbstractPolicyManager(metaclass=ABCMeta):
    """
    Given a state, a policy returns an action
    """

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        reset the specified parameters (for example the current state)
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def find_best_action(self, train_episode=None):
        """
        Find the best action for the parameter state. It returns two elements: how to go and where to go.
        - how to go: the index of the option to activate at this state
        - where to go: the terminal target state of the option
        :param train_episode:
        :return: best_option_index, terminal_state
        if both are None then this means that the exploration has to be performed
        """
        raise NotImplementedError()

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        """
        updates the value of a state specified in the arguments
        :param args:
        :param kwargs:
        :return: void
        """
        raise NotImplementedError()

    @abstractmethod
    def get_max_number_successors(self):
        """
        Get the maximal number of successor states over all states.
        :return: an integer between 0 and the number of states
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_state(self):
        """
        the current state is *not* stored in the Agent class to avoid to
        track twice this variable
        :return: the current state
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_state(self, index_next_state):
        """
        from index_next_state and the current state return the next state.
        :return: the next state
        """
        raise NotImplementedError()


class GraphPlanningPolicyManager(AbstractPolicyManager):
    """
    A policy for the manager based on a graph.
    Number of states unknown.
    Number of actions unknown.
    """
    def __init__(self, parameters):
        self.parameters = parameters

        self.states = []  # Vertices. List of states (those can be of any type, pixels for instance).
        self.transitions = []  # Edges. Contains couples: (state indices i.e. integers, values of transitions).

        self.max_degree = 0  # maximum number of degrees (degree = number of edges that leave the node).
        self.current_state_index = None  # index of the current state
        self.number_explorations = []  # number of explorations done for each state
        self.max_explore = 10  # each state will be explored at most max_explore times todo : put it in parameters
        self.edge_cost = -0.01  # todo: put it in parameters

        # the path that the agent is currently following
        self.current_path = []

    def __str__(self):
        """
        :return: a string containing a representation of the graph
        """
        s = str()
        for idx in range(len(self.states)):
            if idx == self.current_state_index:
                s += red + str(idx) + white

            else:
                s += str(idx)

            s += ": " + str(self.transitions[idx]) + "\n"
        return s

    def reset(self, new_state):
        if not self.states:  # first state added
            self.states = [new_state]
            self.transitions = [[]]
            self.number_explorations = [0]
            self.current_state_index = 0

        else:
            self._update_graph(new_state, 0)

    def update_policy(self, state, option_score):
        self._update_graph(state, option_score)

    def _update_graph(self, new_state, value):
        edge_value = value + self.edge_cost
        new_state_index = find_element_in_list(new_state, self.states)  # bottleneck. Maybe hash states.

        if new_state_index is None:

            self.states.append(new_state)
            self.number_explorations.append(0)

            new_state_index = len(self.states) - 1
            self.transitions.append([])  # new edge without vertex
            self.transitions[self.current_state_index].append((new_state_index, edge_value))  # new vertex with a value

        else:
            if new_state_index not in [t[0] for t in self.transitions[self.current_state_index]]:
                self.transitions[self.current_state_index].append((new_state_index, edge_value))

        self.update_max_degree()

        self.current_state_index = new_state_index

    def update_max_degree(self):
        degree = len(self.transitions[self.current_state_index])
        if degree > self.max_degree:
            assert degree == self.max_degree + 1, "max_degree is not updated correctly"
            self.max_degree += 1

    def get_max_number_successors(self):
        return self.max_degree

    def get_current_state(self):
        """
        :return: the current data of the current node
        """
        if self.current_state_index is None:
            return None

        else:
            return self.states[self.current_state_index]

    def get_next_state(self, option_index):
        (next_state_index, _) = self.transitions[self.current_state_index][option_index]
        return self.states[next_state_index]

    def find_best_action(self, train_episode=None):
        if self.current_state_index is None or not self.transitions[self.current_state_index]:
            return None  # explore

        if not self.current_path:  # if the current path is empty, make a new path.

            # compute the global best path from the current state index
            max_valued_vertex, predecessors = self.find_best_path(self.current_state_index)
            state_to_explore = self.get_state_to_explore(predecessors, self.current_state_index)

            if state_to_explore is not None and train_episode is not None and np.random.rand() < 0.1:  # todo: put this in the parameters
                # make a new path for exploring
                self.current_path = self.make_sub_path(predecessors, self.current_state_index, state_to_explore)+[None]
                # print(self)
                # print("target state to explore " + str(state_to_explore))
                # print("new_path " + str(self.current_path))

            else:
                # make a new path for exploiting
                self.current_path = self.make_sub_path(predecessors, self.current_state_index, max_valued_vertex)+[None]
                # print(self)
                # print("target = " + str(max_valued_vertex))
                # print("new_path " + str(self.current_path))

            self.current_path.pop(0)
            next_state = self.current_path[0]
            # return index of the next option
            if next_state is None:
                return None
            else:
                return [t[0] for t in self.transitions[self.current_state_index]].index(next_state)

        else:  # follow the existing path
            a = self.current_path.pop(0)
            if self.current_state_index == a:  # follow the path
                next_state = self.current_path[0]  # return the next state of the path
                # print(self)
                # print("next target " + str(next_state))
                # print("current path " + str(self.current_path))
                if next_state is None:
                    return None
                else:
                    return [t[0] for t in self.transitions[self.current_state_index]].index(next_state)

            else:  # we are out of the path, make a new one and start again
                self.current_path = []
                return self.find_best_action(train_episode)

    def find_best_path(self, root):
        """
        Bellman-Ford algorithm to get the longest path (highest value)
        :param root: the origin of the path
        :return: the longest path from the root.
        """
        number_vertices = len(self.states)
        distances = [-float("inf")] * number_vertices
        distances[root] = 0
        predecessors = [None] * number_vertices

        for _ in range(number_vertices - 1):
            for origin in range(number_vertices):
                for (target, value) in self.transitions[origin]:
                    if distances[target] < distances[origin] + value:
                        distances[target] = distances[origin] + value
                        predecessors[target] = origin

        # compute the vertices with the highest value, excluding the root
        distances[root] = -float("inf")
        most_valued_vertices = np.nonzero(distances == np.max(distances))[0]
        # choose at *random* among the most valuable vertices
        most_valued_vertex = np.random.choice(most_valued_vertices)
        return most_valued_vertex, predecessors

    def make_sub_path(self, predecessors, origin, target):
        if origin == target:
            return [target]
        else:
            return self.make_sub_path(predecessors, origin, predecessors[target]) + [target]

    def _get_path(self, predecessors, current_state, already_visited=None):
        if already_visited is None:
            already_visited = []

        successors = np.nonzero(np.array(predecessors) == current_state)[0]
        path_from_current_state = [current_state]

        if len(successors) > 0:
            for s in successors:
                if s not in already_visited:
                    already_visited.append(s)
                    path_from_current_state += self._get_path(predecessors, s, already_visited)

        return path_from_current_state

    def _get_unexplored_states(self, path):
        unexplored_states = []
        for state in path:
            if self.number_explorations[state] < self.max_explore:
                unexplored_states.append(state)

        return unexplored_states

    def get_state_to_explore(self, predecessors, current_state):
        """
        return None if no state to explore available
        return the state index to explore which is a state accessible from current_state and is the least explored
        :param predecessors:
        :param current_state:
        :return:
        """
        path = self._get_path(predecessors, current_state)
        unexplored_states = self._get_unexplored_states(path)
        if unexplored_states:
            number_explorations = [self.number_explorations[explorable_state] for explorable_state in unexplored_states]
            return unexplored_states[int(np.argmin(number_explorations))]

        else:
            return None

    # def manual_select_option(self):
    #     while True:
    #         option_chosen = input("which option should I choose ? (-1 = explore)")
    #
    #         try:
    #             option_chosen = int(option_chosen)
    #             number_possible_options = len(self.transitions[self.current_state_index])
    #             if option_chosen >= number_possible_options:
    #                 print("wrong option chosen.")
    #                 print("enter an integer less than " + str(number_possible_options))
    #             else:
    #                 return option_chosen
    #         except ValueError:
    #             print("type an integer")
    #
    # def manual_find_best_action(self, train_episode=None):
    #     if self.current_state_index is None or self.transitions[self.current_state_index] == []:
    #         return None
    #
    #     if self.current_path and self.current_state_index == self.current_path[0]:
    #         self.current_path.pop(0)
    #         print("following the path " + str(self.current_path))
    #         (next_index_path, _) = self.transitions[self.current_state_index].index(self.current_path[0])
    #         return next_index_path
    #
    #     else:
    #         print(self)
    #         option_chosen = self.select_option()
    #
    #         if option_chosen == -1:
    #             return None
    #
    #         else:
    #             return option_chosen


class TreeQPolicyManager(AbstractPolicyManager):
    """
    todo refactor this
    This class is used when the number of actions and the number of states are unknown.
    """

    def get_next_state(self, index_next_state):
        return self.tree.nodes[index_next_state]

    def __init__(self, parameters):
        input("DEPRECATED CLASS")
        self.parameters = parameters
        self.tree = None
        self.end_novelty = False

    def __len__(self):
        return len(self.tree)

    def __str__(self):
        return str(self.tree)

    def reset(self, initial_state):
        self.end_novelty = False
        if self.tree is None:
            self.tree = Tree(initial_state)

        if not obs_equal(self.tree.root.data, initial_state):
            raise ValueError("The initial observation is not always the same ! " +
                             "Maybe a Graph structure could be more adapted")
        else:
            self.tree.reset()

    def find_best_action(self, train_episode=None):
        """
        todo take into account train_episode
        returns the best value and the best action. If best action is None, then explore.
        :return: best_option_index, terminal_state
        """
        values = self.tree.get_children_values()
        if train_episode and (not values or np.random.rand() < self.parameters["probability_random_action_agent"] *
                              np.exp(-train_episode * self.parameters["probability_random_action_agent_decay"])):
            return None, None

        if all(val == values[0] for val in values):  # all the values are the same

            # choose a random action with probability distribution computed from the leaves' depths
            best_option_index = self.tree.get_random_child_index()

        else:  # there exists a proper best action
            best_reward = max(values)
            best_option_index = values.index(best_reward)

        return best_option_index, self.tree.get_child_data_from_index(best_option_index)

    def update_policy(self, new_state, reward, action, train_episode=None):
        """
        todo take into account train_episode
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_(new_position, action)]

        and update the current_node position of variable self.tree
        """
        if train_episode is None:
            raise NotImplementedError()

        else:
            if action is None:  # we update from an exploration
                found = self.tree.move_if_node_with_state(new_state)
                if not found:
                    self.end_novelty = self.tree.add_node(new_state)

            else:  # we update from a regular option

                # move to node which value attribute is Q_t(current_position, action)
                self.tree.move_to_child_node_from_index(action)
                node_activated = self.tree.current_node

                # if the action performed well, take the best value
                if obs_equal(node_activated.data, new_state):

                    if node_activated.get_children_values():
                        best_value = max(node_activated.get_children_values())
                    else:
                        best_value = 0

                    node_activated.value *= (1 - self.parameters["learning_rate"])
                    node_activated.value += self.parameters["learning_rate"] * (reward + best_value)

                else:  # set best value to zero, add the node if the new_state is new and update tree.current_state
                    found = self.tree.move_if_node_with_state(new_state)
                    if not found:
                        self.tree.add_node(new_state)

    def get_max_number_successors(self):
        return self.tree.get_max_width()

    def get_current_state(self):
        return self.tree.get_current_state()
