import numpy as np
import itertools
import random

"""
Simple Graph class that stores adjacency matrix of graph

:param num_of_nodes: number of nodes in Graph
:type num_of_nodes: int
:param adj_matrix: adjacency matrix
:type adj_matrix: np.array
:param edges: all edges in graph
:type edges: list
:param used_nodes: currently used nodes
:type used_nodes: set

"""

class Graph():
    
    def __init__(self, num_of_nodes=None, adj_matrix=None) -> None:
        if num_of_nodes:
            self.num_of_nodes = num_of_nodes
            self.adj_matrix = np.zeros((self.num_of_nodes, self.num_of_nodes), dtype=int)
            self.edges = []
        if adj_matrix is not None:
            rows, cols = np.where(adj_matrix == 1)
            self.edges = list(zip(rows.tolist(), cols.tolist()))
        self.used_nodes = set()

"""
Simple class of Random Graph. It is type of complete graph
with particular probability of every edge to exist. With probability
of 100%, it becomes a complete graph. If partition is provided 
graph becomes bipartited.

:param perc_of_edges: percentage of edges in graph
:type perc_of_edges: int
:param partition: node that parts graph
:type partition: int

"""
class RandomGraph(Graph):

    def __init__(self, perc_of_edges=None, num_of_nodes=None, adj_matrix=None, partition=None) -> None:
        Graph.__init__(self, num_of_nodes, adj_matrix)
        self.perc_of_edges = perc_of_edges
        self.partition = partition

        if adj_matrix is None:
            self.gen_adj_matrix()

    def gen_adj_matrix(self):
        # Randomly choose edges symmetrically
        if not self.partition:
            for row in range(1, self.num_of_nodes):
                for col in range(row):
                    choice = np.random.choice(2, 1, p=[1 - (self.perc_of_edges * 0.01), self.perc_of_edges * 0.01])
                    if choice[0] == 1:
                        self.add_edge(row, col)
        else:
            for row in range(self.partition, self.num_of_nodes):
                for col in range(self.partition):
                    choice = np.random.choice(2, 1, p=[1 - (self.perc_of_edges * 0.01), self.perc_of_edges * 0.01])
                    if choice[0] == 1:
                        self.add_edge(row, col)
        
        self.assure_consistency()
        
    """
    Assures consistency of graph

    """
    def assure_consistency(self):

        while len(self.edges) < self.num_of_nodes - 1:
            # Construct dictionary with number of vertex as key
            # and number of occurrences in self.edges as value
            a = dict()
            for edge in self.edges:
                if edge[0] in a.keys():
                    a[edge[0]] += 1
                else:
                    a[edge[0]] = 1
                if edge[1] in a.keys():
                    a[edge[1]] += 1
                else:
                    a[edge[1]] = 1

            # Collect every vertex with only one occurrence
            tmp = []
            for i, j in a.items():
                if j == 1:
                    tmp.append(a)

            # Make every possible pair with vertexes with 
            # simple occurrences. If any of this pair is an edge,
            # add extra edge to graph to make it consistent
            for pair in itertools.combinations(tmp, 2):
                for edge in self.edges:
                    if list(pair) == list(edge):
                        other_vert = random.choice(self.used_nodes)
                        while other_vert != edge[0] and other_vert != edge[1]:
                            self.add_edge(edge[0], other_vert)
        

    def add_edge(self, row, col):
        self.adj_matrix[row][col] = 1
        self.adj_matrix[col][row] = 1
        self.edges.append((row, col))
        self.used_nodes.add(row)
        self.used_nodes.add(col)
    
        