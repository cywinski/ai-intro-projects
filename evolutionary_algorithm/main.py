from evolutionary_algorithm.evol_algorithm import evol_algorithm
from evolutionary_algorithm.graphs import RandomGraph
from evolutionary_algorithm.figures import make_figures, draw_graph
import numpy as np
import time
import math

### Parameters for graph ###
num_of_nodes = 25   # Number of nodes in graph
perc_of_edges = 100   # Percentage of nodes comparing to complete graph

# Generate graph from csv file
adj_matrix = np.genfromtxt('star_graph.csv', delimiter=' ')


# Generate graph manually
# g = RandomGraph(num_of_nodes=num_of_nodes, perc_of_edges=perc_of_edges)

# Generate graph from csv file
g = RandomGraph(adj_matrix=adj_matrix)

### Parameters for algorithm ###
pop_init = 10 # Initial Population
max_iterate = 10  # Max iterations or population allowed
prob_mutation = 0.55 # Probability of mutation

iter = [i for i in range(max_iterate)]

# Get average results of algorithm executed num times
def get_avg_scores(num):
    fitness = []
    edges = []
    vertexes = []
    avg_time = 0
    best_fitness, worst_fitness = float('inf'), 0

    for i in range(num):
        start_time = time.time()
        best_population, worst_population, fitness_values, uncovered_edges, used_vertexes = evol_algorithm(num_of_nodes, pop_init, max_iterate, prob_mutation, g.edges)
        avg_time += time.time() - start_time
        if best_population.fitness / pop_init < best_fitness:
            best_fitness = best_population.fitness / pop_init
        if worst_population.fitness  / pop_init > worst_fitness:
            worst_fitness = worst_population.fitness / pop_init
        
        fitness.append(fitness_values)
        edges.append(uncovered_edges)
        vertexes.append(used_vertexes)

    avg_fitness, std_dev_fit = cal_avg(fitness, num)
    avg_edges, std_dev_edg = cal_avg(edges, num)
    avg_vertexes, std_dev_ver = cal_avg(vertexes, num)

    return avg_fitness, avg_edges, avg_vertexes, avg_time / num, std_dev_fit, best_fitness, worst_fitness

def cal_avg(ls, num):
    avg = [sum(i) for i in zip(*ls)]
    avg = [round(x / num, 2) for x in avg]
    total_avg =  round(sum(avg) / num, 2)
    std_dev = cal_dev_std(total_avg, avg)
    return avg, std_dev

def cal_dev_std(avg, a):
    std_dev = 0
    for i in range(len(a)):
        std_dev += round((a[i] - avg)**2, 2)
    std_dev = round(std_dev / (len(a) - 1), 2)
    return round(math.sqrt(std_dev), 2)


def main():
    
    avg_fitness, avg_edges, avg_vertexes, avg_time, std_dev_fit, best_fitness, worst_fitness = get_avg_scores(25)
    print(f"total_edges={len(g.edges)}")
    print(f"time={avg_time:.2f}")
    print(f"fitness={avg_fitness[max_iterate-1]}")
    print(f"best_fitness={best_fitness}")
    print(f"worst_fitness={worst_fitness}")
    print(f"std_dev_fitness={std_dev_fit}")
    print(f"edges={avg_edges[max_iterate-1]}")
    print(f"vertexes={avg_vertexes[max_iterate-1]}")
    draw_graph(g.edges)
    make_figures(avg_fitness, avg_vertexes, avg_edges, iter)

if __name__ == "__main__":
    main()