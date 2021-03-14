from optimization.math_calculations import (
    gradient, f
)
import numpy as np


"""
Gradient descent method that finds global minimum of loss function.

:param x0: coordinate x of starting point
:param y0: coordinate y of starting point
:param alpha: learning rate
:param num_of_iterations: maximal number of steps 
:param epsilon: stop parameter - when step is smaller than epsilon 
                                algorithm stops

:returns: list of coordinates of particular points and steps
"""
def gradient_descent(x0, y0, alpha, num_of_iterations, epsilon):
    # Data for figures
    coords = []
    f_values = []
    steps = []
    xs = []
    ys = []

    x = x0
    y = y0
    i = 0
    step = 1

    while i < num_of_iterations and step > epsilon:
        # Calculates gradient of Rosenbrock function
        gradient_of_x = gradient(x, y)[0]
        gradient_of_y = gradient(x, y)[1]

        # Stores data to lists
        coords.append(np.array([round(x, 2), round(y, 2)]))
        xs.append(round(x, 2))
        ys.append(round(y, 2))
        f_values.append(f(coords[i][0], coords[i][1]))
        steps.append(i)

        # Takes step multiplied by factor alpha
        step = abs(gradient_of_x * alpha)
        x -= gradient_of_x * alpha
        y -= gradient_of_y * alpha

        i += 1

    return coords, steps, f_values, xs, ys