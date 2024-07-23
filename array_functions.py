import numpy as np
from numpy import ndarray

from typing import Callable

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


def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001 ) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the
    "input_" array.
    '''

    return ((func(input_ + delta) - func(input_)) / (2 * delta)) 

    