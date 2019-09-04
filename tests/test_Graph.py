import unittest
from mo.structures.graph import GraphNode


class GraphNodeTest(unittest.TestCase):
    def setUp(self):
        """
        We define here some nodes to test their functions
        """
        self.set_cycle()
        self.set_linear()

    def set_linear(self):
        self.nodes_linear = []
        self.nodes_linear.append(GraphNode(index=0))
        self.nodes_linear.append(GraphNode(index=1))
        self.nodes_linear.append(GraphNode(index=2))
        self.nodes_linear.append(GraphNode(index=3))
        self.nodes_linear.append(GraphNode(index=4))

        self.number_nodes = len(self.nodes_linear)

        self.nodes_linear[0].nodes_out = [self.nodes_linear[1]]

        self.nodes_linear[1].nodes_in = [self.nodes_linear[0]]
        self.nodes_linear[1].nodes_out = [self.nodes_linear[2], self.nodes_linear[0], self.nodes_linear[4]]

        self.nodes_linear[2].nodes_in = [self.nodes_linear[1]]
        self.nodes_linear[2].nodes_out = [self.nodes_linear[3], self.nodes_linear[0], self.nodes_linear[0]]

        self.nodes_linear[3].nodes_in = [self.nodes_linear[2]]
        self.nodes_linear[3].nodes_out = [self.nodes_linear[1], self.nodes_linear[1]]

        self.number_nodes_linear = len(self.nodes_linear)

        for k in range(self.number_nodes):
            self.nodes_linear[k].value = 10 * k

    def set_cycle(self):
        self.nodes_cycle = []
        self.nodes_cycle.append(GraphNode(index=0))
        self.nodes_cycle.append(GraphNode(index=1))
        self.nodes_cycle.append(GraphNode(index=2))
        self.nodes_cycle.append(GraphNode(index=3))
        self.nodes_cycle.append(GraphNode(index=4))

        self.number_nodes = len(self.nodes_cycle)

        for k in range(self.number_nodes):
            self.nodes_cycle[k].nodes_in = [self.nodes_cycle[(k - 1) % self.number_nodes], self.nodes_cycle[k]]
            self.nodes_cycle[k].nodes_out = [self.nodes_cycle[(k + 1) % self.number_nodes], self.nodes_cycle[k]]

        self.number_nodes_cycle = len(self.nodes_cycle)

        for k in range(self.number_nodes):
            self.nodes_cycle[k].value = 10 * k

    def p(self):
        print("\n graph cycle \n")
        for node in self.nodes_cycle:
            print(node)

        print("\n graph linear")
        for node in self.nodes_linear:
            print(node)

    def depth_first(self, node, index_nodes):
        index_depths = []
        for n in node.depth_first():
            index_depths.append(n.index)

        self.assertListEqual(index_nodes, index_depths)

    def breadth_first(self, node, index_nodes):
        index_breadth = []
        for n in node.breadth_first():
            index_breadth.append(n.index)

        self.assertListEqual(index_nodes, index_breadth)

    def test_print(self):
        self.p()

    # cycle
    def test_depth_first_cycle(self):
        self.depth_first(self.nodes_cycle[0], [0, 1, 2, 3, 4])

    def test_breadth_first_cycle(self):
        self.breadth_first(self.nodes_cycle[0], [0, 1, 4, 2, 3])

    # linear
    def test_depth_first_linear(self):
        self.depth_first(self.nodes_linear[0], [0, 1, 2, 3, 4])

    def test_breadth_first_linear(self):
        self.breadth_first(self.nodes_linear[0], [0, 1, 2, 4, 3])

# class GraphTest(unittest.TestCase):
#     def setUp(self):
#         gn = GraphNodeTest()
#         gn.setUp()
#         self.graph = Graph(gn.nodes[0])
#
#     def p(self):
#         print(self.graph)