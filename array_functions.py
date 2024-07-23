import numpy as np
from numpy import ndarray

def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray.
    '''

    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
    Compute "Leaky ReLU" function across all elements in the
    input ndarray. "Leaky ReLU" takes the maximum between 0.2x
    and x.
    '''

    return np.maximum(0.2 * x, x)


    