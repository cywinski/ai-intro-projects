import numpy as np

# Rosenbrock function that will be used as loss function
def f(x, y):
    return (1 - x)**2 + 100*((y - x**2)**2)

# Derivatives of Rosenbrock function
def f_prime_x(x, y):
    return 400*(x**3) - 400*x*y + 2*x - 2

def f_prime_y(x, y):
    return 200*y - 200*(x**2)

def f_prime_prime_x_x(x, y):
    return 1200*(x**2) - 400*y + 2

def f_prime_prime_x_y(x, y):
    return -400*x

def f_prime_prime_y_y(x, y):
    return 200


# Gradient of Rosenbrock function
def gradient(x, y):
    return np.array([f_prime_x(x, y), f_prime_y(x, y)])


# Calculates inverted hessian
def inverted_hessian(x, y):
    hessian = np.array([[f_prime_prime_x_x(x, y), f_prime_prime_x_y(x, y)], [f_prime_prime_x_y(x, y), f_prime_prime_y_y(x, y)]])
    return np.array(np.linalg.inv(hessian))
