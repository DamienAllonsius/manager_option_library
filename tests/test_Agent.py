import unittest
from mo.examples.Q_learning import ManagerMontezuma
from mo.examples.policy_examples_manager import QGraph
import numpy as np
import copy


class AgentMontezumaTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.parameters = {"probability_random_action_option": 0.1,
                           "probability_random_action_agent": 0.1,
                           "reward_end_option": 10,
                           "penalty_end_option": -10,
                           "penalty_option_action": -1,
                           "learning_rate": 0.1}

        self.action_space = range(4)

        self.agent = ManagerMontezuma(action_space=self.action_space, parameters=self.parameters)
        self.q = QGraph(parameters=self.parameters)
        self.q._update_states("state 0")
        self.q._update_states("state 1")
        self.q._update_states("state 2")
        self.q._update_states("state 0")
        self.q._update_states("state 3")
        self.q._update_states("state 4")
        self.q._update_states("state 5")
        self.q.values = list(map(lambda x: list(np.random.rand(len(x))), self.q.state_graph))

        self.o_r_d_i1 = [{"manager": "state 2", "option": "initial state option"}, 10, False, None]
        self.o_r_d_i2 = [{"manager": "state 0", "option": "state0 option"}, 0, True, None]
        self.o_r_d_i3 = [{"manager": "state 3", "option": "state0 option"}, 4, False, None]

        self.agent.policy = self.q
        self.agent.option_list = list()
        for k in range(self.q.get_max_number_successors()):
            self.agent.option_list.append(self.agent.new_option())

    def test_act(self):
        self.agent.policy.parameters["probability_random_action_agent"] = 1
        option = self.agent.act(self.o_r_d_i1, train_episode=1)
        self.assertEqual(type(option).__name__, "OptionRandomExplore")
        self.assertEqual(option.initial_state, self.q.get_current_state())

        self.agent.policy.parameters["probability_random_action_agent"] = 0
        option = self.agent.act(self.o_r_d_i1, train_episode=0)
        self.assertEqual(type(option).__name__, "OptionRandomExplore")

        self.agent.policy._update_states("state 0")
        option = self.agent.act(self.o_r_d_i1, train_episode=0)
        self.assertEqual(type(option).__name__, "OptionQArray")

        # are the current states updated before to activate the option ?
        self.assertEqual(option.initial_state, self.o_r_d_i1[0]["manager"])
        self.assertEqual(option.policy.state_list[option.policy.current_state_index], self.o_r_d_i1[0]["option"])

        idmax = np.argmax(np.array(self.q.values[self.q.current_state_index]))
        self.assertEqual(option.terminal_state, self.q.states[self.q.state_graph[self.q.current_state_index][idmax]])

    def test_update_policy(self):
        option_index = 0
        current_state_index = self.agent.policy.current_state_index
        v_copy = copy.deepcopy(list(self.agent.policy.values.copy()))

        self.agent._update_policy(self.o_r_d_i1, None, None)
        v_copy[current_state_index].append(0)
        for k in range(len(v_copy)):
            self.assertListEqual(self.q.values[k], v_copy[k])

        for ordi in [self.o_r_d_i2, self.o_r_d_i3]:
            state_index = self.q.states.index(ordi[0]["manager"])
            best_value = np.max(v_copy[state_index])

            total_reward = self.agent.compute_total_reward(ordi, self.agent.option_list[option_index],0)
            v_copy[self.q.current_state_index][option_index] *= (1 - self.parameters["learning_rate"])
            v_copy[self.q.current_state_index][option_index] += self.parameters["learning_rate"] * (total_reward +
                                                                                                    best_value)

            self.agent._update_policy(ordi, option_index, None)

            for k in range(len(v_copy)):
                self.assertListEqual(list(self.agent.policy.values[k]), list(v_copy[k]))

            self.assertEqual(self.agent.policy.get_current_state(), ordi[0]["manager"])

    def test_update_agent(self):
        n = self.agent.policy.get_max_number_successors()
        option_index = 1
        total_reward = self.agent.compute_total_reward(self.o_r_d_i1, self.agent.option_list[option_index], None)
        self.agent.policy._update_states("state 0")
        current_idx = self.agent.policy.current_state_index
        val = self.agent.policy.values[self.agent.policy.current_state_index][option_index]

        best_val = np.max(self.agent.policy.values[self.agent.policy.states.index(self.o_r_d_i1[0]["manager"])])

        val *= (1 - self.parameters["learning_rate"])
        val += self.parameters["learning_rate"]*(total_reward + best_val)

        self.agent.update_agent(self.o_r_d_i1, self.agent.option_list[1], 1)
        self.assertEqual(len(self.agent.option_list), n+1)

        self.assertEqual(val, self.agent.policy.values[current_idx][option_index])

    def test_get_option_state(self):
        self.agent.policy.current_state_index = 0
        self.agent.update_agent(self.o_r_d_i1, self.agent.option_list[0], 1)
        option_states = self.agent.get_option_states(self.o_r_d_i1, "terminal state")

        self.assertEqual(option_states, (self.o_r_d_i1[0]["manager"], self.o_r_d_i1[0]["option"], "terminal state"))
        self.assertEqual(option_states,
                         (self.agent.policy.get_current_state(), self.o_r_d_i1[0]["option"], "terminal state"))
        self.assertEqual(option_states, (self.agent.get_current_state(), self.o_r_d_i1[0]["option"], "terminal state"))

    def test_get_current_state(self):
        self.assertEqual(self.agent.get_current_state(), "state 5")

        self.agent.policy.current_state_index = 0
        self.agent.update_agent(self.o_r_d_i1, self.agent.option_list[0], 1)
        self.assertEqual(self.agent.get_current_state(), "state 2")

        self.agent.update_agent(self.o_r_d_i2, self.agent.option_list[0], 1)
        self.assertEqual(self.agent.get_current_state(), "state 0")
