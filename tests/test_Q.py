import unittest
from ao.examples.policy_examples_agent import QGraph
from ao.examples.policy_examples_option import QArray
import numpy as np
from copy import deepcopy

class QArrayTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.parameters = {"probability_random_action_option": 0.1,
                           "penalty_option_action": -1,
                           "learning_rate": 0.1,
                           "random_decay": 0.01}

        self.q = QArray(action_space=range(2), parameters=self.parameters)
        self.q._update_states("state 0")
        self.q._update_states("state 1")
        self.q._update_states("state 2")
        self.q._update_states("state 0")
        self.q._update_states("state 3")
        self.q._update_states("state 4")
        self.q._update_states("state 4")
        self.q._update_states("state 4")
        self.q._update_states("state 5")
        self.q._update_states("state 2")
        self.q._update_states("state 5")

        self.q2 = QArray(action_space=range(2), parameters=self.parameters)
        self.q2._update_states("state 0")
        self.q2._update_states("state 1")
        self.q2._update_states("state 2")
        self.q2._update_states("state 0")
        self.q2._update_states("state 3")
        self.q2._update_states("state 0")
        self.q2._update_states("state 4")

        self.q3 = QArray(action_space=range(2), parameters=self.parameters)
        self.q3._update_states("state 0")
        self.q3._update_states("state 1")
        self.q3._update_states("state 2")
        self.q3._update_states("state 0")
        self.q3._update_states("state 3")
        self.q3._update_states("state 0")

    def test_str(self):
        message = ""
        for k in range(6):
            message += "state " + str(k) + " values: " + str(np.zeros(2, dtype=np.float64)) + "\n"

        self.assertEqual(message, str(self.q))

    def test_update_states(self):

        self.assertEqual(self.q.len_state_list, 6)
        self.assertEqual(self.q2.len_state_list, 5)
        self.assertEqual(self.q3.len_state_list, 4)

        self.assertEqual(self.q.current_state_index, 5)
        self.assertEqual(self.q2.current_state_index, 4)
        self.assertEqual(self.q3.current_state_index, 0)

        self.assertEqual(self.q.state_list, ["state 0", "state 1", "state 2", "state 3", "state 4", "state 5"])
        self.assertEqual(self.q2.state_list, ["state 0", "state 1", "state 2", "state 3", "state 4"])
        self.assertEqual(self.q3.state_list, ["state 0", "state 1", "state 2", "state 3"])

        self.assertListEqual([list(self.q.values[0]) for _ in range(self.q.len_state_list)],
                             [list(np.zeros(2, dtype=np.float64)) for _ in range(self.q.len_state_list)])
        self.assertListEqual([list(self.q2.values[0]) for _ in range(self.q2.len_state_list)],
                             [list(np.zeros(2, dtype=np.float64)) for _ in range(self.q2.len_state_list)])
        self.assertListEqual([list(self.q3.values[0]) for _ in range(self.q3.len_state_list)],
                             [list(np.zeros(2, dtype=np.float64)) for _ in range(self.q3.len_state_list)])

    def test_get_random_action(self):
        s = set()
        s2 = set()
        s3 = set()
        for k in range(100):
            s.add(self.q.get_random_action())
            s2.add(self.q.get_random_action())
            s3.add(self.q.get_random_action())

        self.assertEqual(s, {0,1})
        self.assertEqual(s2, {0, 1})
        self.assertEqual(s3, {0, 1})

    def test_reset(self):
        self.q.reset("current_state")
        self.assertEqual(self.q.current_state_index, self.q.len_state_list - 1)
        self.q.reset("state 0")
        self.assertEqual(self.q.current_state_index, 0)

        self.q2.reset("state 3")
        self.assertEqual(self.q2.current_state_index, 3)

        self.q3.reset("Fluke et schmurk")
        self.assertEqual(self.q3.current_state_index, self.q3.len_state_list - 1)

    def test_find_best_action(self):
        self.q.reset("state 5")
        self.q.values[5][0] = 100
        action = self.q.find_best_action()
        self.assertEqual(action, 0)

        self.q2.reset("state 3")
        self.q2.values[3][1] = 100
        action = self.q2.find_best_action()
        self.assertEqual(action, 1)

        self.q3.reset("SBLORG")
        self.q3.values[self.q3.len_state_list - 1][1] = 100
        action = self.q3.find_best_action()
        self.assertEqual(action, 1)

        self.q3.reset("state 0")
        self.q3.values[self.q3.current_state_index][1] = 100
        action = 0
        N = 67
        for k in range(N):
            action += self.q3.find_best_action(train_episode=1)

        action /= N
        self.assertTrue(action < 1)
        self.assertTrue(action > 0.9)

    def test_update_policy(self):
        # first test
        self.q.reset("state 0")
        new_state1 = "state 1"
        new_state1_index = 1
        reward = 10
        action = 1
        other_action = 0
        best_value = 0
        end_option = False

        self.q.update_policy(new_state1, reward, action, end_option, 0)
        self.assertEqual(self.q.values[0][other_action], 0)
        self.assertEqual(self.q.values[0][action], self.q.parameters["learning_rate"] * (reward + best_value))
        self.assertEqual(self.q.current_state_index, new_state1_index)

        # second test
        new_state2 = "state 2"
        new_state2_index = 2
        action = 1
        other_action = 0
        end_option = False

        val_new_state1 = np.random.rand(self.q.number_actions)
        val_new_state2 = np.random.rand(self.q.number_actions)

        self.q.values[new_state2_index] = val_new_state2.copy()
        self.q.values[new_state1_index] = val_new_state1.copy()

        best_value = max(val_new_state2)
        reward = 10

        self.q.update_policy(new_state2, reward, action, end_option, 0)
        self.assertEqual(self.q.values[new_state1_index][other_action], val_new_state1[other_action])
        self.assertEqual(self.q.current_state_index, new_state2_index)

        self.assertEqual(self.q.values[new_state1_index][action],
                         (1 - self.q.parameters["learning_rate"]) * val_new_state1[action]
                         + self.q.parameters["learning_rate"] * (reward + best_value))
        # third test
        new_state3 = "state 3"
        new_state3_index = 3
        action = 0
        other_action = 1
        end_option = True

        val_new_state3 = np.random.rand(self.q.number_actions)
        val_new_state2 = np.random.rand(self.q.number_actions)

        self.q.values[new_state3_index] = val_new_state3.copy()
        self.q.values[new_state2_index] = val_new_state2.copy()

        best_value = 0  # end_option is True !
        reward = 10

        self.q.update_policy(new_state3, reward, action, end_option, 0)
        self.assertEqual(self.q.values[new_state2_index][other_action], val_new_state2[other_action])
        self.assertEqual(self.q.current_state_index, new_state3_index)

        self.assertEqual(self.q.values[new_state2_index][action],
                         (1 - self.q.parameters["learning_rate"]) * val_new_state2[action]
                         + self.q.parameters["learning_rate"] * (reward + best_value))

        # other test
        val3 = deepcopy(self.q.values)
        self.q.update_policy(new_state2, reward, action, end_option, None)
        self.assertEqual(self.q.current_state_index, 2)
        self.assertTrue(np.array_equal(self.q.values, val3))


class QGraphTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.parameters = {"probability_random_action_agent": 0.1,
                           "learning_rate": 0.1,
                           "penalty_option_action": -1}

        self.q = QGraph(parameters=self.parameters)
        self.q._update_states("state 0")
        self.q._update_states("state 1")
        self.q._update_states("state 2")
        self.q._update_states("state 0")
        self.q._update_states("state 3")
        self.q._update_states("state 4")
        self.q._update_states("state 5")
        self.q._update_states("state 1")
        self.q._update_states("state 5")
        self.q.values = list(map(lambda x: list(np.random.randint(10, size=len(x))), self.q.state_graph))

        self.q2 = QGraph(parameters=self.parameters)
        self.q2._update_states("state 0")
        self.q2._update_states("state 1")
        self.q2._update_states("state 2")
        self.q2._update_states("state 0")
        self.q2._update_states("state 3")
        self.q2._update_states("state 0")
        self.q2._update_states("state 4")
        self.q2.values = list(map(lambda x: list(np.random.randint(10, size=len(x))), self.q2.state_graph))

        self.q3 = QGraph(parameters=self.parameters)
        self.q3._update_states("state 0")
        self.q3._update_states("state 1")
        self.q3._update_states("state 2")
        self.q3._update_states("state 0")
        self.q3._update_states("state 3")
        self.q3._update_states("state 0")
        self.q3.values = list(map(lambda x: list(np.random.randint(10, size=len(x))), self.q3.state_graph))

    def test_str(self):
        self.assertEqual(str(self.q), "0: 1, 3, \n1: 2, 5, \n2: 0, \n3: 4, \n4: 5, \n5: 1, \n")
        self.assertEqual(str(self.q2), "0: 1, 3, 4, \n1: 2, \n2: 0, \n3: 0, \n4: \n")
        self.assertEqual(str(self.q3), "0: 1, 3, \n1: 2, \n2: 0, \n3: 0, \n")

    def test_update_states(self):
        self.assertRaises(AssertionError, self.q._update_states, "state 5")
        self.assertRaises(AssertionError, self.q2._update_states, "state 4")
        self.assertRaises(AssertionError, self.q3._update_states, "state 0")

        self.assertEqual(self.q.max_states, 2)
        self.assertEqual(self.q2.max_states, 3)
        self.assertEqual(self.q3.max_states, 2)

        self.assertEqual(self.q.len_state_list, 6)
        self.assertEqual(self.q2.len_state_list, 5)
        self.assertEqual(self.q3.len_state_list, 4)

        self.assertEqual(self.q.current_state_index, 5)
        self.assertEqual(self.q2.current_state_index, 4)
        self.assertEqual(self.q3.current_state_index, 0)

        self.assertEqual(self.q.states, ["state 0", "state 1", "state 2", "state 3", "state 4", "state 5"])
        self.assertEqual(self.q2.states, ["state 0", "state 1", "state 2", "state 3", "state 4"])
        self.assertEqual(self.q3.states, ["state 0", "state 1", "state 2", "state 3"])

        self.assertEqual(self.q.state_graph, [[1, 3], [2, 5], [0], [4], [5], [1]])
        self.assertEqual(self.q2.state_graph, [[1, 3, 4], [2], [0], [0], []])
        self.assertEqual(self.q3.state_graph, [[1, 3], [2], [0], [0]])

    def test_reset(self):
        l_state_list = self.q.len_state_list
        graph = self.q.state_graph.copy()
        self.assertEqual(self.q.current_state_index, 5)
        self.q.reset("state 3")
        self.assertEqual(self.q.current_state_index, 3)
        self.assertEqual(self.q.len_state_list, l_state_list)
        self.assertEqual(self.q.state_graph, graph)

        self.q2.reset("state schmurk")
        self.assertEqual(self.q2.current_state_index, self.q2.len_state_list - 1)
        self.q3.reset("state schmurk")
        self.assertEqual(self.q3.current_state_index, 4)

    def test_find_best_action(self):
        self.q.current_state_index = 0
        self.q.values[0][1] = 100
        self.assertEqual(self.q.find_best_action(), (1, "state 3"))

        action = set()
        for k in range(1000):
            action.add(self.q.find_best_action(train_episode=1))

        expected_returned = set()
        expected_returned.add((None, None))
        expected_returned.add((1, "state 3"))

        self.assertEqual(action, expected_returned)

        self.q.parameters["probability_random_action_agent"] = 0
        idmax = np.argmax(np.array([self.q.values[self.q.current_state_index]]))
        self.assertEqual((idmax, self.q.states[self.q.state_graph[self.q.current_state_index][idmax]]),
                         self.q.find_best_action())

        self.q2.parameters["probability_random_action_agent"] = 0
        if len(self.q2.values[self.q2.current_state_index]) == 0:
            self.assertEqual((None, None), self.q2.find_best_action())
        else:
            idmax = np.argmax(np.array(self.q2.values[self.q2.current_state_index]))

            self.assertEqual((idmax, self.q2.states[self.q2.state_graph[self.q2.current_state_index][idmax]]),
                             self.q2.find_best_action())

        self.q3.parameters["probability_random_action_agent"] = 0
        idmax = np.argmax(np.array(self.q3.values[self.q3.current_state_index]))
        self.assertEqual((idmax, self.q3.states[self.q3.state_graph[self.q3.current_state_index][idmax]]),
                         self.q3.find_best_action())

    def test_update_policy(self):

        # first test
        self.q.reset("state 0")
        new_state1 = "state 1"
        new_state1_index = 1
        reward = 10
        action = 1
        other_action = 0
        best_value = 0
        self.q.values = list(map(lambda x: [0]*len(x), self.q.values))

        self.q.update_policy(new_state1, reward, action)
        self.assertEqual(self.q.values[0][other_action], 0)
        self.assertEqual(self.q.values[0][action], self.q.parameters["learning_rate"] * (reward + best_value))
        self.assertEqual(self.q.current_state_index, new_state1_index)

        # second test
        new_state2 = "state 2"
        new_state2_index = 2
        action = 1
        other_action = 0

        val_new_state1 = np.random.rand(len(self.q.values[new_state1_index]))
        val_new_state2 = np.random.rand(len(self.q.values[new_state2_index]))

        self.q.values[new_state2_index] = val_new_state2.copy()
        self.q.values[new_state1_index] = val_new_state1.copy()

        best_value = max(val_new_state2)
        reward = 10

        self.q.update_policy(new_state2, reward, action)
        self.assertEqual(self.q.values[new_state1_index][other_action], val_new_state1[other_action])

        self.assertEqual(self.q.current_state_index, new_state2_index)

        self.assertEqual(self.q.values[new_state1_index][action],
                         (1-self.q.parameters["learning_rate"]) * val_new_state1[action]
                         + self.q.parameters["learning_rate"] * (reward + best_value))

    def test_get_random_action(self):
        randomq = set()
        randomq2 = set()
        randomq3 = set()
        self.q.current_state_index = 0
        self.q2.current_state_index = 0
        for k in range(100):
            randomq.add(self.q.get_random_action())
            randomq2.add(self.q2.get_random_action())
            randomq3.add(self.q3.get_random_action())

        self.assertEqual(randomq, {1, 3})
        self.assertEqual(randomq2, {1, 3, 4})
        self.assertEqual(randomq3, {1, 3})

    def test_max_successors(self):
        self.assertEqual(self.q.get_max_number_successors(), 2)
        self.assertEqual(self.q2.get_max_number_successors(), 3)
        self.assertEqual(self.q.get_max_number_successors(), 2)

    def test_get_current_state(self):
        self.assertEqual(self.q.get_current_state(), "state 5")
        self.assertEqual(self.q2.get_current_state(), "state 4")
        self.assertEqual(self.q3.get_current_state(), "state 0")

# class QTreeTest(unittest.TestCase):
#     def setUp(self):
#         parameters = {"probability_random_action_option": 0.1,
#                       "learning_rate": 0.1,
#                       "random_decay": 0.01}
#         self.q = QTree(parameters, range(2))
#         tree_test = TreeTest()
#         tree_test.setUp()
#
#         self.q.tree = tree_test.tree
#
#     def test_len(self):
#         self.assertEqual(len(self.q), 7)
#
#     def test_str(self):
#         s = "\n"
#         s += green + "|0. depth : 0\n"
#         s += green + tab + "|1. depth : 1\n"
#         s += green + tab + tab + "|4. depth : 2\n"
#         s += red + tab + tab + tab + "|7. depth : 3\n"
#         s += green + tab + tab + "|5. depth : 2\n"
#         s += green + tab + "|2. depth : 1\n"
#         s += green + tab + "|3. depth : 1\n"
#         s += green + tab + tab + "|6. depth : 2\n"
#         s += white
#         self.assertEqual(str(self.q), s)
#
#     def test_reset(self):
#         self.q.reset(0)
#         self.assertEqual(self.q.get_current_state(), 0)
#         self.q.tree = None
#         self.q.reset("state")
#         self.assertEqual(self.q.tree, Tree("state"))
