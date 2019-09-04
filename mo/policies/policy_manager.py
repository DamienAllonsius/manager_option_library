"""
Policies that can be applied only on agents
"""
from abc import ABCMeta, abstractmethod
from mo.utils.miscellaneous import red, white
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


class GraphPolicy(AbstractPolicyManager):
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
        self.edge_cost = -1  # todo: put it in parameters

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
                s += str(idx) + ": " + str(self.transitions[idx]) + "\n"
        return s

    def reset(self, new_state):
        if not self.states:  # first state added
            self.states = [new_state]
            self.transitions = [[]]
            self.number_explorations = [0]
            self.current_state_index = 0

        else:
            self._update_graph(new_state, 0)

    def _update_graph(self, new_state, value):
        edge_value = value + self.edge_cost

        if new_state not in self.states:
            self.states.append(new_state)
            self.number_explorations.append(0)

            self.transitions.append([])  # new edge without vertex
            new_state_index = len(self.states) - 1
            self.transitions[self.current_state_index].append((new_state_index, edge_value))  # new vertex with a value

        else:
            new_state_index = self.states.index(new_state)
            if new_state_index not in map(lambda t: t[0], self.transitions[self.current_state_index]):
                self.transitions[self.current_state_index].append((new_state_index, edge_value))

        # update max_degree and the current state
        degree = len(self.transitions[self.current_state_index])
        if degree > self.max_degree:
            assert degree == self.max_degree + 1, "max_degree is not updated correctly"
            self.max_degree += 1

        self.current_state_index = new_state_index

    def update_policy(self, state, option_score):
        self._update_graph(state, option_score)

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
        next_state_index = self.transitions[self.current_state_index][option_index]
        return self.states[next_state_index]

    def find_best_action(self, train_episode=None):
        if self.current_state_index is None or self.transitions[self.current_state_index] == []:
            return None

        str(self.transitions[self.current_state_index])
        target_state_index = input("which state should I go ? (-1 = explore)")
        if target_state_index == -1:
            return None

        else:
            if target_state_index not in self.transitions[self.current_state_index]:
                self.find_best_path(train_episode)

            else:
                return self.transitions[self.current_state_index].index(target_state_index)

        # todo : find a good strategy
        # if self.current_state_index is None:
        #     return None  # explore
        #
        # if not self.current_path:  # if the current path is empty
        #     # compute the global best path from the current state index
        #     successors = self.find_best_path(self.current_state_index)
        #
        #     # make a new path to follow
        #     if train_episode is not None and np.random.rand() < 0.1:  # todo: put this in the parameters
        #         unexplored_state = np.argmin(self.number_explorations)
        #         if self.number_explorations[unexplored_state] < self.max_explore:  # exploration is needed
        #             self.current_path = self.make_sub_path(successors, self.current_state_index, unexplored_state)
        #
        #     else:
        #         self.current_path = self.make_sub_path(successors, self.current_state_index, successors.index(None))
        #
        #     return self.current_path.pop()
        #
        # else:  # follow the path
        #     if self.current_state_index != self.current_path[0]:  # if the current state is out of the path
        #         self.current_path = []
        #         return self.find_best_action(train_episode)  # make a new path
        #
        #     else:
        #         self.current_path.pop(0)  # remove the first element
        #         next_state = self.current_path[0]  # return the next state of the path
        #         return self.transitions[self.current_state_index].index(next_state)  # index of the next option

    def find_best_path(self, root):
        """
        Bellman-Ford algorithm to get the longest path (highest value)
        :param root: the origin of the path
        :return: the longest path from the root.
        """
        number_vertices = len(self.states)
        distances = [-float("inf")] * number_vertices
        distances[root] = 0
        successors = [None] * number_vertices

        for _ in range(number_vertices - 1):
            for origin in range(number_vertices):
                for (target, value) in self.transitions[origin]:
                    if distances[target] < distances[origin] + value:
                        distances[target] = distances[origin] + value
                        successors[origin] = target

        return successors

    def make_sub_path(self, successors, origin, target):
        if origin == target:
            return [origin]
        else:
            return [origin] + self.make_sub_path(successors, successors[origin], target)
