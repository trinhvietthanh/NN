import math
import numpy as np
from math import exp


def hard_limit(n):
    """[Hard limit activation function]

    Args:
        n ([float]): [description]
    """
    return 0 if n < 0 else 1


def symmetrical_hardlimit(n):
    """[symmetrical_hardlimit]

    Args:
        n ([float]): [input activation]
    """
    return -1 if n < 0 else 1


def linear(n):
    """[Linear activation function]
  
    Args:
        n ([float]): [description]

    Returns:
        [float]: [return value input]
    """
    return n


def saturating_linear(n):
    """[activation function Saturating linear]

    Args:
        n ([float]): [description]

    Returns:
        [float]: [0 if n < 0, 1 if n > 1, n if 0<= n <= 1]
    """
    if n < 0:
        return 0
    if n > 1:
        return 1
    return n


def symmetrical_saturating_linear(n):
    """[summary]

    Args:
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    if n < -1:
        return 0
    if n > 1:
        return 1
    return n


def sigmoid(n):
    """[activation sigmoil: 1 / (1 + exp(-n))]

    Args:
        n ([float]): [description]

    Returns:
        [float]: [description]
    """
    return 1/(1 + exp(-n))

def tanh(n):
    """[Hyperbolic tangent activation function]

    Args:
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return ((exp(n) - exp(-n))/(exp(n) + exp(-n)))

def positive_linear(n):
    """[Positive linear activation function]

    Args:
        n ([float]): [description]

    Returns:
        [type]: [description]
    """
    return 0 if n < 0 else n

def relu(n, alpha=0.0, max_value=None, threshold=0):
    """[summary]

    Args:
        n ([type]): [description]
        alpha (float, optional): [ A float that governs the slope for values lower than the threshold]. Defaults to 0.0.
        max_value ([float], optional): [A float that sets the saturation threshold]. Defaults to None.
        threshold (int, optional): [A float giving the threshold value of the activation function below 
        which values will be damped or set to zero]]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    x = max(0, n)
    return x



def deserialize(name):
    globs = globals()
    obj = globs.get(name)
    if obj is None:
        raise ValueError(
            'Unknown activation name: {}.'.format(name)
        )
    return obj

def get(identifier):
    """[Returns function activation]

    Args:
        identifier : [Function or string]

    Raises:
        TypeError: [Input is an unknown function or string, i.e., the input does
        not denote any defined function]

    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        identifier = str(identifier)
        return deserialize(identifier)
    else:
        raise TypeError(
            'Could not interpret activation function identifier: {}'.format(
                identifier))

