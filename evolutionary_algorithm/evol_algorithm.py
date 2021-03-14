from evolutionary_algorithm.population import Population
import random
import numpy as np

"""
Simple evolutionary algorithm without crossing, with tournament selection.
"""

def evol_algorithm(num_of_nodes, pop_init, max_iter, prob_mutation, g_edges):

    # Data for figures
    fitness_values = []
    uncovered_edges = []
    used_vertexes = []
    worst_population = None
    t = 0
    # Initialize the population
    curr_population = Population(num_of_nodes, pop_init, g_edges)
    curr_population.calc_fitness()
    worst_population = curr_population

    while t < max_iter:
        
        # Sorting population before selection
        curr_population.sort_by_fitness()

        # Get data for figures
        fitness_values.append(round(curr_population.fitness / pop_init, 2))
        uncovered_edges.append(round(curr_population.omited_edges / pop_init, 2))
        used_vertexes.append(round(curr_population.used_nodes / pop_init, 2))
        
        # Making next population using tournament selection
        next_population = make_next_population(curr_population, len(curr_population.lst), num_of_nodes, pop_init, g_edges)

        # Performing mutation of bits
        mutate_population(next_population, prob_mutation)

        # Calculate fitness of new population
        next_population.calc_fitness()
        curr_population = next_population
        t += 1
    
    curr_population.sort_by_fitness()

    return curr_population, worst_population, fitness_values, uncovered_edges, used_vertexes


def mutate_population(pop, prob_mutation):
    for chromosome in pop.lst:
        for bit in range(len(chromosome.bits)):
            mutate = np.random.choice(2, 1, p=[1 - prob_mutation, prob_mutation])
            if mutate:
                chromosome.change_bit(bit)
                
def make_next_population(pop, size_of_pop, num_of_bits, pop_init, g_edges):
    i = 0
    next_pop = []

    while i < size_of_pop:
        picked = tournament_selection(pop)
        next_pop.append(picked)
        i += 1

    return Population(num_of_bits, pop_init, g_edges, next_pop)


def tournament_selection(pop):
    # Randomly pick two chromosomes from population
    chromosome0 = random.choice(pop.lst)
    chromosome1 = random.choice(pop.lst)
    # Choose and return one with better fitness score
    fitness0 = chromosome0.fitness
    fitness1 = chromosome1.fitness

    return chromosome0 if fitness0 < fitness1 else chromosome1
