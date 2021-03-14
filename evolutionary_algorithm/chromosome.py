import random
import numpy as np

"""
Class of one chromosome in Evolutionary Algorithm
which is array of bites

:param num_of_bits: number of bits in chromosome
:type num_of_bits: int
:param gr_edges: edges in graph
:type gr_edges: list
:param bits: array of bits
:type bits: list
:param chosen_vert: vertices in graph chosen to vertex cover problem
:type chosen_vert: list
:param fitness: fitness score of chromosome, the lower the better
:type fitness: int
:param omited_edges: edges not covered in vertex cover problem
:type omited_edges: list

"""
class Chromosome():

    def __init__(self, num_of_bits, gr_edges, bits=None) -> None:
        self.num_of_bits = num_of_bits  
        self.gr_edges = gr_edges
        self.bits = [] if not bits else bits
        self.chosen_vert = []
        self.fitness = 0
        self.omited_edges = []

        if not bits:
            # Make random array of bitss
            for i in range(self.num_of_bits):
                choice = random.randint(0, 1)
                self.bits.append(choice)
                if choice:
                    self.chosen_vert.append(i)

    """
    Calculates fitness score of chromosome. The lower score
    the better chromosome. Every chosen vertex count as 1 and 
    every not covered edge count as 1.
    
    """
    def calc_fitness(self):
        # Every chosen vertex count as 1 in fitness score
        self.fitness = 0
        self.fitness += len(self.chosen_vert)

        # Calculate omitted edges
        self.omited_edges.clear()
        for edge in range(len(self.gr_edges)):
            if (self.gr_edges[edge][0] not in self.chosen_vert) and \
                (self.gr_edges[edge][1] not in self.chosen_vert):
                self.fitness += 1
                self.omited_edges.append(self.gr_edges[edge])

    def get_fitness(self):
        return self.fitness

    def visualize(self):
        print(np.array(self.bits))

    def change_bit(self, pos):
        if self.bits[pos] == 1:
            self.bits[pos] = 0
        else:
            self.bits[pos] = 1
        