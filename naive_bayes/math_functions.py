import numpy as np
import math

def mean(numbers):
    return sum(numbers) / len(numbers)

def std_dev(numbers):
    return np.std(np.array(numbers))

"""
Calculates Gaussian probability density function for given x

"""
def pdf(x, mean, std_dev):
    exp = math.exp(-((x - mean)**2 / (2*std_dev**2)))
    return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exp