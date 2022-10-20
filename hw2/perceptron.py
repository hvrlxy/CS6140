
import numpy as np
def distance_to_hyperplane(x, w, d):
    """
    Compute the distance of a point x to the hyperplane defined by w.
    :param x: a data point
    :param w: a vector of weights
    :return: the distance of x to the hyperplane defined by w
    """
    return np.abs((np.dot(x, w) + d) / np.linalg.norm(w))

#generate x and w for testing
x = [1, 2]
w = [1, 1]
d = 0

print(distance_to_hyperplane(x, w, d))
print(distance_to_hyperplane(x, np.add(w, x), d))