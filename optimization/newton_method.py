from optimization.math_calculations import (
    gradient, inverted_hessian, f
)
import numpy as np


"""
Newton method that finds global minimum of loss function.

:param x0: coordinate x of starting point
:param y0: coordinate y of starting point
:param alpha: learning rate
:param num_of_iterations: maximal number of steps 
:param epsilon: stop parameter - when step is smaller than epsilon 
                                algorithm stops

:returns: list of coordinates of particular points and steps
"""
def newton(x0, y0, alpha, num_of_iterations, epsilon):
    # Data for figures
    coords = []
    xs = []
    ys = []
    steps = []
    f_values = []

    x = x0
    y = y0
    i = 0
    step = 1
    

    while i < num_of_iterations and step > epsilon:
        # d = inverted hessian * gradient
        d = np.matmul(inverted_hessian(x, y), gradient(x, y))

        coords.append(np.array([round(x, 2), round(y, 2)]))
        xs.append(round(x, 2))
        ys.append(round(y, 2))
        f_values.append(f(coords[i][0], coords[i][1]))
        steps.append(i)

        
        step = abs(d[0] * alpha)
        x -= d[0] * alpha
        y -= d[1] * alpha

        i += 1


    return coords, steps, f_values, xs, ys
