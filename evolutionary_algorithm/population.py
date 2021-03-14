# from evolutionary_algorithm.chromosome import Chromosome 

"""
Class representing population in Evolutionary Algorithm.

:param num_of_bits: number of bits in single chromosome
:type num_of_bits: int
:param pop_init: initial number of chromosomes in population
:type pop_init: int
:param gr_edges: edges in graph
:type gr_edges: list
:param lst: array of chromosomes in population
:type lst: list
:param fitness: fitness score of population, the lower the better
:type fitness: int
:param vertexes: number of total vertexes in population
:type vertexes: int
:param uncovered_edges: number of total uncovered edges in graph
:type uncovered_edges: int

"""
class Population():

    def __init__(self, num_of_bits, pop_init, gr_edges, pop=None) -> None:
        self.num_of_bits = num_of_bits
        self.pop_init = pop_init    # Number of initial population
        self.gr_edges = gr_edges    # Edges in Graph
        self.lst = [] if not pop else pop  # Current population
        self.fitness = 0
        self.omited_edges = 0
        self.used_nodes = 0

        if not pop:
            for i in range(self.pop_init):
                chromosome = Chromosome(self.num_of_bits, self.gr_edges)
                self.lst.append(chromosome)


    def calc_fitness(self):
        for chromosome in self.lst:
            chromosome.calc_fitness()
            self.fitness += chromosome.get_fitness()
            self.omited_edges += len(chromosome.omited_edges)
            self.used_nodes += len(chromosome.chosen_vert)

    def get_fitness(self):
        return self.fitness

    def visualize(self):
        for i in range(len(self.lst)):
            self.lst[i].visualize()

    def sort_by_fitness(self):
        self.lst = sorted(self.lst, key=lambda x: x.fitness, reverse=False)