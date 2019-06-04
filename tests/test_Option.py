import unittest
from agents.examples.options_examples import OptionQArray
import numpy as np


class OptionTest(unittest.TestCase):
    def setUp(self):
        self.parameters = {"probability_random_action_option": 0.1,
                           "reward_end_option": 10,
                           "penalty_end_option": -10,
                           "penalty_option_action": -1,
                           "learning_rate": 0.1}

        self.o_r_d_i1 = [{"agent": "state 1", "option": "state option 1"}, 7, "true", "no-info"]
        self.o_r_d_i2 = [{"agent": "terminal state", "option": "state option 2"}, 2, "False", "no-info"]
        self.o_r_d_i3 = [{"agent": "state 3", "option": "state option 3"}, 17, "true", "no-info"]

        self.option1 = OptionQArray(action_space=range(2), parameters=self.parameters, index=0)
        self.option1.reset("state 0", "state option 0", "terminal state")

        self.option1.policy._update_states("state option 1")
        self.option1.policy._update_states("state option 2")
        self.option1.policy._update_states("state option 0")
        self.option1.policy._update_states("state option 3")
        self.option1.policy._update_states("state option 4")
        self.option1.policy._update_states("state option 4")
        self.option1.policy._update_states("state option 4")
        self.option1.policy._update_states("state option 5")
        self.option1.policy.values = [np.array([1, 15], dtype="float64"),
                                      np.array([9, 24], dtype="float64"),
                                      np.array([55, 55], dtype="float64"),
                                      np.array([11, 32], dtype="float64"),
                                      np.array([10, 9], dtype="float64"),
                                      np.array([1, 2], dtype="float64")]

    def test_update_option(self):
        self.option1.reset("state 1", "state option 1", "terminal state")
        self.option1.update_option(self.o_r_d_i1, 0, train_episode=None)
        self.assertEqual(self.option1.score, self.o_r_d_i1[1])

        # test 2
        action = 0
        self.option1.reset("state 1", "state option 1", "terminal state")
        current_state_idx = self.option1.policy.current_state_index
        values = list(self.option1.policy.values[current_state_idx].copy())

        self.option1.update_option(self.o_r_d_i1, action, train_episode=1)
        total_reward = self.o_r_d_i1[1]
        total_reward += (action != 0) * self.parameters["penalty_option_action"]

        best_value = np.max(self.option1.policy.values[self.option1.policy.current_state_index])

        value = values[action] * (1 - self.option1.policy.parameters["learning_rate"]) + \
            self.option1.policy.parameters["learning_rate"] * (total_reward + best_value)

        values[action] = value

        # update the policy values
        self.assertListEqual(list(self.option1.policy.values[current_state_idx]), values)
        # update current state
        self.assertEqual(self.option1.policy.current_state_index, 1)
        # update score
        self.assertEqual(self.option1.score, self.o_r_d_i1[1])

        # test 3

        action = 1
        self.option1.reset("state 0", "state option 0", "terminal state")
        current_state_idx = self.option1.policy.current_state_index
        values = list(self.option1.policy.values[current_state_idx].copy())

        self.option1.update_option(self.o_r_d_i1, action, train_episode=1)
        total_reward = self.o_r_d_i1[1]
        total_reward += self.parameters["penalty_end_option"]
        total_reward += (action != 0) * self.parameters["penalty_option_action"]

        best_value = 0

        value = values[action] * (1 - self.option1.policy.parameters["learning_rate"]) + \
            self.option1.policy.parameters["learning_rate"] * (total_reward + best_value)

        values[action] = value

        # update the policy values
        self.assertListEqual(list(self.option1.policy.values[current_state_idx]), values)
        # update current state
        self.assertEqual(self.option1.policy.current_state_index, 1)
        # update score
        self.assertEqual(self.option1.score, self.o_r_d_i1[1])

    def test_act(self):
        self.assertEqual(self.option1.act(), 1)
        self.option1.reset("state 3", "state option 3", "terminal state")
        self.assertEqual(self.option1.act(), 1)
        self.option1.reset("state 4", "state option 4", "terminal state")
        self.assertEqual(self.option1.act(), 0)
        self.option1.reset("state 2", "state option 2", "terminal state")
        self.assertEqual(self.option1.act(), 0)

    def test_reset(self):
        self.option1.reset("initial state", "state option initial", "terminal state 2")
        self.assertTrue(self.option1.activated)

        self.option1.terminal_state = "terminal state"
        end = self.option1.check_end_option(new_state="terminal state")
        self.assertTrue(end)
        self.assertFalse(self.option1.activated)

    def test_compute_total_reward(self):
        action1 = 0
        action2 = 1
        total_reward = list()
        total_reward.append(self.option1.compute_total_reward(self.o_r_d_i1, action1, end_option=True))
        total_reward.append(self.option1.compute_total_reward(self.o_r_d_i1, action1, end_option=False))
        total_reward.append(self.option1.compute_total_reward(self.o_r_d_i2, action2, end_option=True))
        total_reward.append(self.option1.compute_total_reward(self.o_r_d_i3, action2, end_option=True))

        total_reward_expected = list()
        total_reward_expected.append(self.o_r_d_i1[1] + self.parameters["penalty_end_option"])
        total_reward_expected.append(self.o_r_d_i1[1])
        total_reward_expected.append(self.o_r_d_i2[1] +
                                     self.parameters["penalty_option_action"] +
                                     self.parameters["reward_end_option"])
        total_reward_expected.append(self.o_r_d_i3[1] +
                                     self.parameters["penalty_option_action"] +
                                     self.parameters["penalty_end_option"])

        for k in range(len(total_reward)):
            self.assertEqual(total_reward[k], total_reward_expected[k])

    def test_compute_total_score(self):
        self.assertEqual(self.option1.compute_total_score(self.o_r_d_i1, 0,
                                                          end_option=True,
                                                          train_episode=None), self.o_r_d_i1[1])
        self.assertEqual(self.option1.compute_total_score(self.o_r_d_i1, 0,
                                                          end_option=True,
                                                          train_episode=1), self.o_r_d_i1[1])
