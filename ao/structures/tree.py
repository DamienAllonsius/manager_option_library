from collections import defaultdict
from ao.utils.utils import *
import numpy as np


class Node(object):
    """
    Tests: OK
    This class creates a Node.
    A Node is connected to its parent node via the variable self.parent.
    A Node has children, stored in the list self.children.
    A node has a value and a variable data which is used to store a state representation.
    """
    def __init__(self, data, parent=None):
        self.value = 0
        self.data = data  # a.k.a state

        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1

        else:
            self.depth = 0

        self.children = list()

    def __eq__(self, other):
        return self.data == other.data

    def __repr__(self):
        return "data: " + str(self.data)

    def __str__(self):
        s = "\n" + "node " + str(self.data) + " at depth " + str(self.depth) + "\n"
        s += "value: " + str(self.value) + "\n"
        if self.parent is not None:
            s += "parent: " + str(self.parent.data) + "\n"

        else:
            s += "no parent\n"

        if self.children != list():
            s += "children: " + "["
            for child in self.children:
                s += str(child.data) + ", "

            s = s[:-2]
            s += "]" + "\n"
            for k in range(len(self.children)):
                s += "action: " + str(k) + \
                     " with value: " + str(self.children[k].value) \
                     + "\n"

        else:
            s += "no child\n"

        return s

    def depth_first(self):
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def breadth_first(self):
        current_nodes = [self]
        while len(current_nodes) > 0:
            children = []
            for node in current_nodes:
                yield node
                children.extend(node.children)
            current_nodes = children

    def make_root(self):
        if not self.is_root():
            self.parent.children.remove(self)  # just to be consistent
            self.parent = None
            old_depth = self.depth
            for node in self.breadth_first():
                node.depth -= old_depth

    def find_root(self):
        if self.is_root():
            return self

        else:
            return self.parent.find_root()

    def get_values(self):
        return [child.value for child in self.children]


class Tree:
    """
    Tests: OK
    Although a Node is a tree by itself, this class provides more iterators and
    quick access to the different depths of
    the tree, and keeps track of the root node
    """
    def __init__(self, root_data):
        self.root = Node(root_data)
        self.current_node = self.root
        self.max_width = 0
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)

    def __eq__(self, other_tree):
        iter_self = self.root.depth_first()
        iter_other = other_tree.root.depth_first()
        for n_self, n_other in zip(iter_self, iter_other):
            if n_other != n_self:
                return False

        # check if one of those iterators is not empty
        try:
            next(iter_self)
            return False
        except StopIteration:
            try:
                next(iter_other)
                return False
            except StopIteration:
                return True

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        s = str()
        s += "\n"
        for node in self.root.depth_first():
            if node == self.current_node:
                s += red

            else:
                s += green

            s += "".join([tab] * node.depth + ["|", str(node.data) + ". depth : " + str(node.depth), '\n'])
        return s + white

    def tree_to_string(self, next_node):
        """
        transforms the tree into a string. The difference with __str__ is that we can give a different color to
        a particular node (next_node)
        :return: a string representing the tree
        """

        s = str()
        s += "\n"
        for node in self.root.depth_first():
            if node == self.current_node:
                s += red

            elif node == next_node:
                s += yellow

            else:
                s += green

            s += "".join([tab] * node.depth + ["|", str(node.data) + ". depth : " + str(node.depth), '\n'])
        return s + white

    def new_root(self, node: Node):
        node.make_root()
        self.root = node
        self.current_node = self.root
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)

        for n in self.root.breadth_first():
            # iterate through children nodes and add them to the depth list
            self._update(n)

    def _update(self, node: Node):
        """
        updates the depth, the nodes list, the max_depth, max_width and the current node
        :param node:
        """
        self.current_node = node
        self.depth[node.depth].append(node)
        self.nodes.append(node)

        # update max_depth
        if node.depth > self.max_depth:
            self.max_depth = node.depth

        # update max_width
        if node.parent is not None and len(node.parent.children) > self.max_width:
            self.max_width += 1

    def add_node(self, parent_node: Node, node: Node):
        """
        add the tree under the parent_node if it does not already exist. Then, updates the tree parameters.
        :param parent_node: the node from where you attach the input node
        :param node: the node to add to the tree
        :return the new node
        """
        node.parent = parent_node
        node.depth = parent_node.depth + 1
        parent_node.children.append(node)
        self._update(node)

    @staticmethod
    def _get_leaves(node: Node):
        """
        :param: node: if root : return all the leaves.
        else: get all the parents for each leaf and compare them to node.
        :return the leaves of the given node input.
        """
        leaves = []
        for child in node.depth_first():
            if child.is_leaf() and (not child.is_root()):
                leaves.append(child)

        return leaves

    @staticmethod
    def _get_child_index(node: Node, leaf: Node):
        """
        This function gets the child of input node which is a parent of input leaf.
        :param node:
        :param leaf:
        :return: an index of list node.children
        """
        while leaf.parent != node:
            leaf = leaf.parent

        return leaf.parent.children.index(leaf)

    @staticmethod
    def _get_probability_leaves(node):
        """
        for all leaves from node, computes the probability of selecting a leaf.
        These probabilities are proportional to the depth of the leaf (computed from input node)
        :param node:
        :return: the probabilities of selecting a leaf, all the leaves of the tree which has input node as a parent
        """
        assert not(node.is_leaf())

        leaves = Tree._get_leaves(node)
        probability_leaves = np.zeros(len(leaves))
        idx = -1
        for leaf in leaves:
            idx += 1
            probability_leaves[idx] = (leaf.depth - node.depth)

        probability_leaves /= np.sum(probability_leaves)

        return probability_leaves, leaves

    def get_random_child_index(self):
        """
        gives the index of a child, selected according to its probability, computed with _get_probability_leaves
        :return: an index from list self.children
        """
        probability_leaves, leaves = Tree._get_probability_leaves(self.current_node)
        selected_leaf = leaves[sample_pmf(probability_leaves)]
        # selected_leaf = np.randome.choice(leaves, 1, p=probability_leaves)
        return Tree._get_child_index(self.current_node, selected_leaf)

    def get_max_width(self):
        return self.max_width

    def get_current_node(self):
        return self.current_node

    def set_current_node(self, node: Node):
        self.current_node = node

    def get_values_children(self):
        return self.current_node.get_values()

    def get_node_from_state(self, state):
        """
        :param state:
        :return: the corresponding node with node.data == state
        :exception if the state does not exist
        """
        for node in self.root.depth_first():
            if node.data == state:
                return node

        raise ValueError("state does not exist in the tree")

    def get_child_node(self, feature):
        """
        :param feature: the node feature we are looking for (can be node.data or the index of the child)
        :return: a child of self.current_node with child.data == state
        """
        if type(feature) == str:
            for child in self.current_node.children:
                if child.data == feature:
                    return child

            raise ValueError("None of my children have this state")

        elif type(feature) == int:
            return self.current_node.children[feature]

        else:
            raise Exception("feature type is not recognized")
