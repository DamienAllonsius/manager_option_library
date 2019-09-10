import unittest
import numpy as np


class Graph:
    def __init__(self, states, transitions):
        self.states = states
        self.transitions = transitions
        self.number_explorations = []
        self.max_explore = 10
        self.number_explorations = [10] * len(self.states)
        # self.number_explorations[0] = 7
        # self.number_explorations[1] = 4
        # self.number_explorations[2] = 10
        # self.number_explorations[3] = 10
        # self.number_explorations[4] = 9

    def find_best_action(self, train_episode=None):
        if self.current_state_index is None or not self.transitions[self.current_state_index]:
            return None  # explore

        if len(self.current_path) <= 1:  # The current path is empty, make a new path.

            # compute the global best path from the current state index
            predecessors, distances = self.find_best_path(self.current_state_index)

            if train_episode is not None and np.random.rand() < 0.1:  # todo: put this in the parameters
                # make a new path for exploring
                state_to_explore = self.get_state_to_explore(distances)
                if state_to_explore is not None:
                    self.current_path = self.make_sub_path(predecessors, self.current_state_index, state_to_explore) + [
                        None]
                    # print(self)
                    # print("target state to explore " + str(state_to_explore))
                    # print("new_path " + str(self.current_path))

            else:
                # make a new path for exploiting
                self.current_path = self.make_sub_path(predecessors, self.current_state_index, max_valued_vertex)
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
            if self.current_state_index == a:  # we are in the right state, follow the rest of the path
                next_state = self.current_path[0]  # next state of the path
                # print(self)
                # print("next target " + str(next_state))
                # print("current path " + str(self.current_path))
                if next_state is None:
                    return None
                else:  # return the option to go toward the next state
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

    def _get_path(self, distances):
        return np.nonzero(np.array(distances) != -float("inf"))[0]

    def _get_unexplored_states(self, path):
        return [state for state in path if self.number_explorations[state] < self.max_explore]

    def get_state_to_explore(self, distances):
        """
        return None if no state to explore available
        return the state index to explore which is a state accessible from current_state and is the least explored
        :param predecessors:
        :param current_state:
        :return:
        """
        path = self._get_path(distances)
        unexplored_states = self._get_unexplored_states(path)
        if unexplored_states:
            number_explorations = [self.number_explorations[explorable_state] for explorable_state in unexplored_states]
            return unexplored_states[int(np.argmin(number_explorations))]

        else:
            return None


class TestGraph(unittest.TestCase):

    def setUp(self):
        s = ["0", "1", "2", "3", "4", "0.2", "1.2"]
        t = [[(1, -1)], [(2, 3), (3, -1)], [], [(4, 4)], [], [(6, 0)], []]
        self.g = Graph(s, t)

    def test_path(self):
        initial_state = 0
        _, d = self.g.find_best_path(initial_state)
        print(d)
        print(self.g._get_path(d))