from ao.structures.tree import Node, Tree
import numpy as np
import unittest
from ao.utils.miscellaneous import red, white, yellow, tab, green


class TreeTest(unittest.TestCase):

    def setUp(self):
        """
        We define here a Tree to test its functions
        """
        np.random.seed(0)
        self.tree = Tree(root_data=0)
        self.node_1 = Node(data=1)
        self.node_2 = Node(data=2)
        self.node_3 = Node(data=3)
        self.node_4 = Node(data=4)
        self.node_5 = Node(data=5)
        self.node_6 = Node(data=6)
        self.node_7 = Node(data=7)
        self.node_8 = Node(data=8)

        self.set_parents_children(self.tree)
        self.set_values()

    def set_values(self):
        self.tree.root.value = 0
        self.node_1.value = 1
        self.node_2.value = 10
        self.node_3.value = 11
        self.node_4.value = 100
        self.node_5.value = 101
        self.node_6.value = 111
        self.node_7.value = 1000

    def set_parents_children(self, tree):
        """
        Defines a Tree with the nodes
        :return:
        """
        tree.add_node(tree.root, self.node_1)
        tree.add_node(tree.root, self.node_2)
        tree.add_node(tree.root, self.node_3)

        tree.add_node(self.node_1, self.node_4)
        tree.add_node(self.node_1, self.node_5)

        tree.add_node(self.node_3, self.node_6)

        tree.add_node(self.node_4, self.node_7)

    # ------------- The tests are defined here --------------

    def test_eq(self):
        tree = Tree(root_data=0)
        self.set_parents_children(tree)
        self.assertEqual(tree, self.tree)

        tree2 = Tree(root_data=0)
        tree2.add_node(tree.root, self.node_1)
        tree2.add_node(tree.root, self.node_2)
        tree2.add_node(tree.root, self.node_3)

        self.assertNotEqual(tree2, self.tree)
        self.assertNotEqual(self.tree, tree2)
        self.assertNotEqual(tree2, tree)
        self.assertNotEqual(tree, tree2)

        self.tree.root.data = "toto"
        self.assertNotEqual(tree, self.tree)

    # def test_new_root(self):
    #     self.tree.new_root(self.node_3)
    #
    #     tree = Tree(0)
    #     tree.root = self.node_3
    #     tree.nodes = [self.node_3, self.node_6]
    #     tree.depth[0].append(self.node_3)
    #     tree.depth[1].append(self.node_6)
    #     tree.max_depth = 1
    #
    #     self.assertEqual(self.tree.root, tree.root)
    #     self.assertEqual(self.tree.nodes, tree.nodes)
    #     self.assertEqual(self.tree.depth, tree.depth)
    #     self.assertEqual(self.tree.get_max_depth(), tree.max_depth)

    def test_update(self):
        depth = 3
        self.node_8.depth = depth
        self.tree._update_characteristics(self.node_8)
        self.assertEqual(self.tree.depth[depth], [self.node_7, self.node_8])
        self.assertEqual(self.tree.current_node, self.node_7)
        self.assertEqual(self.tree.nodes[-1], self.node_8)
        self.assertEqual(self.tree.get_max_depth(), self.node_8.depth)

    def test_add_node(self):
        self.tree.add_node(parent_node=self.node_6, node=self.node_8)
        self.assertEqual(self.tree.depth[3], [self.node_7, self.node_8])
        self.assertEqual(self.node_1.parent, self.tree.root)
        self.assertEqual(self.node_2.parent, self.tree.root)
        self.assertEqual(self.node_3.parent, self.tree.root)
        self.assertEqual(self.node_4.parent, self.node_1)
        self.assertEqual(self.node_5.parent, self.node_1)
        self.assertEqual(self.node_6.parent, self.node_3)
        self.assertEqual(self.node_7.parent, self.node_4)

        self.assertEqual(self.tree.root.children, [self.node_1, self.node_2, self.node_3])
        self.assertEqual(self.node_1.children, [self.node_4, self.node_5])
        self.assertEqual(self.node_3.children, [self.node_6])
        self.assertEqual(self.node_4.children, [self.node_7])

    def test_get_leaves(self):
        leaves = self.tree._get_leaves(node=self.tree.root)
        leaves_1 = self.tree._get_leaves(node=self.tree.root.children[0])
        self.assertEqual(leaves, [self.node_7, self.node_5, self.node_2, self.node_6])
        self.assertEqual(leaves_1, [self.node_7, self.node_5])

    def test_get_child_index(self):
        next_node_index_1 = Tree._get_child_index(self.tree.root, self.node_4)
        next_node_index_4 = Tree._get_child_index(self.node_1, self.node_7)
        next_node_index_3 = Tree._get_child_index(self.tree.root, self.node_6)

        self.assertEqual(next_node_index_1, 0)
        self.assertEqual(next_node_index_4, 0)
        self.assertEqual(next_node_index_3, 2)

    def test_get_probability_leaves(self):
        leaves_0, _ = Tree._get_probability_leaves(self.tree.root)
        leaves_1, _ = Tree._get_probability_leaves(self.node_1)
        leaves_3, _ = Tree._get_probability_leaves(self.node_3)
        leaves_4, _ = Tree._get_probability_leaves(self.node_4)

        with self.assertRaises(Exception):
            leaves_2, _ = self.tree._get_probability_leaves(self.node_2)
        with self.assertRaises(Exception):
            leaves_5, _ = self.tree._get_probability_leaves(self.node_5)
        with self.assertRaises(Exception):
            leaves_7, _ = self.tree._get_probability_leaves(self.node_7)
        with self.assertRaises(Exception):
            leaves_6, _ = self.tree._get_probability_leaves(self.node_7)

        np.testing.assert_array_equal(leaves_0, np.array([3 / 8, 2 / 8, 1 / 8, 2 / 8]))
        np.testing.assert_array_equal(leaves_1, np.array([2 / 3, 1 / 3]))
        np.testing.assert_array_equal(leaves_3, np.array([1]))
        np.testing.assert_array_equal(leaves_4, np.array([1]))

    def test_tree_to_string(self):
        s = "\n"
        s += green + "|0. depth : 0\n"
        s += yellow + tab + "|1. depth : 1\n"
        s += green + tab + tab + "|4. depth : 2\n"
        s += red + tab + tab + tab + "|7. depth : 3\n"
        s += green + tab + tab + "|5. depth : 2\n"
        s += green + tab + "|2. depth : 1\n"
        s += green + tab + "|3. depth : 1\n"
        s += green + tab + tab + "|6. depth : 2\n"
        s += white

        self.assertEqual(s, self.tree.tree_to_string(self.node_1))

    def test_get_random_child_index(self):
        from collections import defaultdict
        index = defaultdict(int)
        self.tree.current_node = self.tree.root
        n = 100
        for k in range(n):
            index[self.tree.get_random_child_index()] += 1

        self.assertAlmostEqual(index[0] / n, 5 / 8, delta=0.05)
        self.assertAlmostEqual(index[1] / n, 1 / 8, delta=0.05)
        self.assertAlmostEqual(index[2] / n, 2 / 8, delta=0.05)

    def test_get_max_width(self):
        self.assertEqual(self.tree.get_max_width(), 3)

    def test_get_current_node(self):
        self.assertEqual(self.tree.get_current_node(), self.node_7)
        self.tree.add_node(self.node_4, Node(12))
        self.assertEqual(self.tree.get_current_node(), Node(12))
