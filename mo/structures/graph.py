class GraphNode(object):
    """
    oriented graph
    """
    def __init__(self, index, nodes_out=None):
        """
        :param index: state id, an integer
        :param nodes_out: edges between self and nodes in nodes_out are directed towards nodes_out
        """
        self.value = 0
        self.index = index
        self.nodes_out = list()

        # nodes_in
        if nodes_in is not None:
            self.nodes_in = nodes_in
            for n in nodes_in:
                n.nodes_out.append(self)

        # nodes_out
        if nodes_out is not None:
            self.nodes_out = nodes_out
            for n in nodes_out:
                n.nodes_in.append(self)

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return self.index

    def __repr__(self):
        return "index: " + str(self.index)

    def __str__(self):
        s = str()
        if self.nodes_in:
            for n in self.nodes_in:
                s += str(n.index) + " "

        else:
            s += "_ "

        s += " --> "
        s += str(self.index)
        s += " --> "

        if self.nodes_out:
            for n in self.nodes_out:
                s += str(n.index) + " "

        else:
            s += "_ "

        return s

    def depth_first(self, visited_nodes=None):
        """
        Depth first search.
        We first visit the nodes_out list, then the nodes_in list (this choice is arbitrary).
        :param visited_nodes:
        :return:
        """
        if visited_nodes is None:
            visited_nodes = [self]

        yield self
        for neighbour in self.nodes_out + self.nodes_in:
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                for node in neighbour.depth_first(visited_nodes):
                    yield node

    def breadth_first(self):
        visited_nodes = []
        current_neighbours = [self]
        while current_neighbours:
            neighbours = []
            for node in current_neighbours:
                if node not in visited_nodes:
                    visited_nodes.append(node)
                    yield node
                    neighbours.extend(node.nodes_out + node.nodes_in)
            current_neighbours = neighbours

    def complete_search(self, k):
        yield self, k
        for neighbour in self.nodes_out:
            for node, depth in neighbour.depth_first(k + 1):
                yield node, depth

        for neighbour in self.nodes_in:
            for node, depth in neighbour.depth_first(k + 1):
                yield node, depth


    def get_out_values(self):
        return [n.value for n in self.nodes_out]

    def get_in_values(self):
        return [n.value for n in self.nodes_in]


class Graph:
    """
    """
    def __init__(self, data):
        self.nodes = [GraphNode(0)]  # list of all the nodes in the Graph
        self.node_data = [data]  # list of all nodes data in the Graph

        self.current_index = 0
        self.max_out = 0  # maximum_{n \in nodes of Graph} len(n.nodes_out)

    def add_node(self, data):
        try:
            # this data is already in list node_data
            self.current_index = self.node_data.index(data)

        except ValueError:
            # this data does not already exist
            # update the list of data of nodes
            self.node_data.append(data)

            # update the current index
            self.current_index = len(self.node_data)

            # create a new node with the appropriate index
            new_node = GraphNode(self.current_index)

            # update the current node
            self.nodes[self.current_index].append_node_out(new_node)

            # add this new node in the list
            self.nodes.append(new_node)

    def graph_to_string(self, current_index, visited_index):
        visited_index.append(current_index)
        s = str()
        for n in self.nodes[current_index].nodes_out:
            s += "-->"
            new_index = n.index
            if new_index == self.current_index:
                s += red
            s += str(new_index) + white
            if new_index not in visited_index
                s += self.graph_to_string(n.index, visited_index)


    # def __str__(self):
    #     s = str()
    #     s += "\n"
    #
    #     for node in self.current_node.breadth_first():
    #         s += red
    #         s += str(node.data)
    #
    #         else:
    #             s += green
    #         s += "".join([tab] * node.depth + ["|", str(id(node.data)) +
    #                                             " value : " + str(node.value), '\n'])
    #     return s + white

#     def tree_to_string(self, next_node):
#         """
#         transforms the tree into a string. The difference with __str__ is that we can give a different color to
#         a particular node (next_node)
#         :return: a string representing the tree
#         """
#
#         s = str()
#         s += "\n"
#         for node in self.root.depth_first():
#             if node == self.current_node:
#                 s += red
#
#             elif node == next_node:
#                 s += yellow
#
#             else:
#                 s += green
#
#             s += "".join([tab] * node.depth + ["|", str(node.data) + ". depth : " + str(node.depth), '\n'])
#         return s + white
#
#     def reset(self):
#         self.current_node = self.root
#
#     def get_max_width(self):
#         return self.max_width
#
#     def get_current_state(self):
#         return self.current_node.data
#
#     def move_to_child_node_from_index(self, index):
#         self.current_node = self.current_node.children[index]
#
#     def move_if_node_with_state(self, state):
#         """
#         update self.current_state if there exists a node with state
#         :param state:
#         :return True iff node with node.data == state is found
#         """
#         # to improve performances: first check the children
#         for node in self.current_node.children:
#             if obs_equal(node.data, state):
#                 self.current_node = node
#                 return True
#
#         # then check all nodes
#         for node in self.root.breadth_first():
#             if obs_equal(node.data, state):
#                 self.current_node = node
#                 return True
#
#         # if not found, return False
#         return False
#
#     def add_node(self, data):
#         """
#         Add the tree under the current_node if it is not novel (IW). Then, updates the tree characteristics.
#         :param data: the data contained in the node that we have to add to the tree.
#         :return the novelty
#         """
#         novel = self.update_novelty_table(data)
#         if novel:
#             node = Node(data, self.current_node)
#             self._update_characteristics(node)
#             self.current_node = node
#
#         return novel
#
#     def _update_characteristics(self, node: Node):
#         """
#         updates the depth, the nodes list, max_width and the current node
#         :param node:
#         """
#         self.depth[node.depth].append(node)
#         self.nodes.append(node)
#
#         # update max_width
#         if node.parent is not None and len(node.parent.children) > self.max_width:
#             self.max_width += 1
#
#     def update_novelty_table(self, state):
#         """
#         todo make tests
#         updates the novelty table by including the elements of state in the novelty table if needed.
#         :param state:
#         :return: True iff the state is novel
#         """
#         novel = False
#         for i in range(self.shape[0]):
#             for j in range(self.shape[1]):
#
#                 t = tuple(state[i, j])
#                 if t not in self.novelty_table[i, j]:
#                     novel = True
#                     self.novelty_table[i, j].add(t)
#
#         return novel
#
#     def get_children_values(self):
#         return self.current_node.get_children_values()
#
#     @staticmethod
#     def _get_leaves(node: Node):
#         """
#         :param: node: if root : return all the leaves.
#         else: get all the parents for each leaf and compare them to node.
#         :return the leaves of the given node input.
#         """
#         leaves = []
#         iterator = node.depth_first()
#
#         # remove the first element if it is the root
#         if node.is_root():
#             next(iterator)
#
#         for child in iterator:
#             if child.is_leaf():
#                 leaves.append(child)
#
#         return leaves
#
#     def _get_child_index_to_leaf(self, leaf: Node):
#         """
#         This function gets a child index of current_node.
#         This child is a parent of leaf.
#         :param leaf:
#         :return: an integer between 0 and len(self.current_node.children) - 1
#         """
#         while leaf.parent != self.current_node:
#             leaf = leaf.parent
#
#         return leaf.parent.children.index(leaf)
#
#     def _get_probability_leaves(self):
#         """
#         todo : possible to put another distribution
#         for all leaves from current_node, computes the probability of selecting a leaf.
#         These probabilities are proportional to the depth of the leaf (computed from input node)
#         :return: the probabilities of selecting a leaf, all the leaves of the tree which has input node as a parent
#         """
#         assert not(self.current_node.is_leaf())
#
#         leaves = Tree._get_leaves(self.current_node)
#         probability_leaves = np.zeros(len(leaves))
#         idx = -1
#         for leaf in leaves:
#             idx += 1
#             probability_leaves[idx] = (leaf.depth - self.current_node.depth)
#
#         probability_leaves /= np.sum(probability_leaves)
#
#         return probability_leaves, leaves
#
#     def get_random_child_index(self):
#         """
#         gives the index of a child, selected according to its probability, computed with _get_probability_leaves
#         :return: an index from list self.children
#         """
#         probability_leaves, leaves = self._get_probability_leaves()
#         selected_leaf = leaves[sample_pmf(probability_leaves)]
#         return self._get_child_index_to_leaf(selected_leaf)
#
#     def get_child_data_from_index(self, child_index):
#         return self.current_node.children[child_index].data
