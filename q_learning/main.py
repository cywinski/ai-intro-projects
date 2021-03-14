import numpy as np
from q_learning.maze import Maze
from q_learning.q_learning import Q_Learning
from matplotlib import pyplot as plt


def main():
    
    ### PARAMETRY ###
    start = (0, 4)
    end = (7, 0)
    maze = Maze(8, start, end, matrix=np.loadtxt("test_maze3.txt"))
    min_paths = [
    [start,(1, 4),(2, 4),(3, 4),(4, 4),(4, 3),(5, 3),(6, 3),(6, 2),(6, 1),(7, 1),end],
    [start,(1, 4),(2, 4),(3, 4),(3, 3),(4, 3),(5, 3),(6, 3),(6, 2),(6, 1),(7, 1),end],
    [start,(1, 4),(2, 4),(3, 4),(4, 4),(4, 3),(5, 3),(6, 3),(6, 2),(7, 2),(7, 1),end],
    [start,(1, 4),(2, 4),(3, 4),(3, 3),(4, 3),(5, 3),(6, 3),(6, 2),(7, 2),(7, 1),end]]

    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay_rate = 0.1
    beta = 0.8
    gamma = 0.8
    total_epochs = 1000
    max_steps = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]


    brain_ends = list()
    pinky_ends = list()
    brain_min_p = list()
    pinky_min_p = list()
    brain_len_avg = list()
    pinky_brain_avg = list()
    
    for m in max_steps:

        algorithm = Q_Learning(start, end, maze, epsilon, max_epsilon, min_epsilon, decay_rate, beta, gamma, total_epochs, m, min_paths)

        brain_end, brain_min_paths, brain_paths_len, \
        pinky_end, pinky_min_paths, pinky_paths_len = algorithm.learn()

        brain_ends.append(brain_end)
        pinky_ends.append(pinky_end)
        brain_min_p.append(brain_min_paths)
        pinky_min_p.append(pinky_min_paths)
        brain_len_avg.append(sum(brain_paths_len) / total_epochs)
        pinky_brain_avg.append(sum(pinky_paths_len) / total_epochs)
    print(f"brain_found_cheese={brain_ends}")
    print(f"pinky_found_cheese={pinky_ends}")
    print(f"brain_minimal_paths={brain_min_p}")
    print(f"pinky_minimal_paths={pinky_min_p}")
    print(f"brain_average_length={brain_len_avg}")
    print(f"pinky_average_length={pinky_brain_avg}")
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax1.scatter(max_steps, brain_ends, c='blue', label='Brain', edgecolors='black', linewidths=0.5, s=50)
    ax1.scatter(max_steps, pinky_ends, c='green', label='Pinky', edgecolors='black', linewidths=0.5, s=50)
    ax2.scatter(max_steps, brain_min_p, c='blue', label='Brain', edgecolors='black', linewidths=0.5, s=50)
    ax2.scatter(max_steps, pinky_min_p, c='green', label='Pinky', edgecolors='black', linewidths=0.5, s=50)
    ax3.scatter(max_steps, brain_len_avg, c='blue', label='Brain', edgecolors='black', linewidths=0.5, s=50)
    ax3.scatter(max_steps, pinky_brain_avg, c='green', label='Pinky', edgecolors='black', linewidths=0.5, s=50)
    ax1.set_xticks(max_steps)
    ax2.set_xticks(max_steps)
    ax3.set_xticks(max_steps)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title('How many times mice have found the cheese')
    ax2.set_title('How many times mice have found minimal path')
    ax3.set_title('Average of length of mice\'s paths')
    ax3.set_xlabel('Max steps')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
