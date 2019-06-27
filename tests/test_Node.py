import unittest
from ao.structures.tree import Node


class NodeTest(unittest.TestCase):
    def setUp(self):
        """
        We define here some nodes to test their functions
        """
        self.node_0 = Node(data=0)
        self.node_1 = Node(data=1, parent=self.node_0)
        self.node_2 = Node(data=2, parent=self.node_0)
        self.node_3 = Node(data=3, parent=self.node_0)
        self.node_4 = Node(data=4, parent=self.node_1)
        self.node_5 = Node(data=5, parent=self.node_1)
        self.node_6 = Node(data=6, parent=self.node_3)
        self.node_7 = Node(data=7, parent=self.node_4)

        self.set_parents_children()
        self.set_values()

    def set_parents_children(self):
        """
        Defines a Tree with the nodes
        :return:
        """
        self.node_0.children = [self.node_1, self.node_2, self.node_3]
        self.node_1.children = [self.node_4, self.node_5]
        self.node_3.children = [self.node_6]
        self.node_4.children = [self.node_7]

    def set_values(self):
        self.node_0.value = 0
        self.node_1.value = 1
        self.node_2.value = 10
        self.node_3.value = 11
        self.node_4.value = 100
        self.node_5.value = 101
        self.node_6.value = 111
        self.node_7.value = 1000

    def test_str(self):
        s0 = str(self.node_0)
        s5 = str(self.node_5)

        s0_test = "\n" + "node 0 at depth 0" + "\n" + "value: 0" + "\n" + "no parent" + "\n" + "children: [1, 2, 3]" \
            + "\n" + "action: 0 with value: 1" + "\n" + "action: 1 with value: 10" + "\n" + "action: 2 with value: 11\n"

        self.assertEqual(s0, s0_test)
        s5_test = "\n" + "node 5 at depth 2\n" + "value: 101\n" + "parent: 1\n" + "no child\n"

        self.assertEqual(s0, s0_test)
        self.assertEqual(s5, s5_test)

    def test_depth_first(self):
        nodes = [node for node in self.node_0.depth_first()]
        self.assertEqual(nodes, [self.node_0, self.node_1, self.node_4, self.node_7,
                                 self.node_5, self.node_2, self.node_3, self.node_6])

    def test_breadth_first(self):
        nodes = [node for node in self.node_0.breadth_first()]
        self.assertEqual(nodes, [self.node_0, self.node_1, self.node_2, self.node_3,
                                 self.node_4, self.node_5, self.node_6, self.node_7])

    def test_make_root(self):
        self.node_4.make_root()
        self.assertTrue(self.node_4.is_root())
        self.assertFalse(self.node_4 in self.node_1.children)

    def test_find_root(self):
        self.assertEqual(self.node_0, self.node_5.find_root())

    def test_is_root(self):
        for node in self.node_0.depth_first():
            if node == self.node_0:
                self.assertTrue(node.is_root())
            else:
                self.assertFalse(node.is_root())

    def test_is_leaf(self):
        for node in self.node_0.depth_first():
            if node.data in [2, 5, 6, 7]:
                self.assertTrue(node.is_leaf())
            else:
                self.assertFalse(node.is_leaf())

    def test_get_values(self):
        values_0 = self.node_0.get_children_values()
        values_1 = self.node_1.get_children_values()
        values_2 = self.node_2.get_children_values()
        values_3 = self.node_3.get_children_values()
        values_7 = self.node_7.get_children_values()

        self.assertEqual(values_0, [1, 10, 11])
        self.assertEqual(values_1, [100, 101])
        self.assertEqual(values_2, [])
        self.assertEqual(values_3, [111])
        self.assertEqual(values_7, [])
