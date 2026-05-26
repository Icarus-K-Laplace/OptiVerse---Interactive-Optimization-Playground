import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable, Optional


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    grad: np.ndarray,
    alpha: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4
) -> float:
    """
    Backtracking line search satisfying the Armijo condition.
    
    Args:
        f: Objective function
        x: Current point
        direction: Search direction
        grad: Gradient at x
        alpha: Initial step size
        rho: Reduction factor (0 < rho < 1)
        c: Armijo condition constant (0 < c < 1)
    
    Returns:
        Step size satisfying Armijo condition
    """
    f_x = f(x)
    grad_dot_direction = np.dot(grad, direction)
    
    while f(x + alpha * direction) > f_x + c * alpha * grad_dot_direction:
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha


def exact_line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    bounds: tuple = (0, 10)
) -> float:
    """
    Exact line search using scipy's minimize_scalar.
    
    Args:
        f: Objective function
        x: Current point
        direction: Search direction
        bounds: Search interval for alpha
    
    Returns:
        Optimal step size
    """
    def phi(alpha):
        return f(x + alpha * direction)
    
    result = minimize_scalar(phi, bounds=bounds, method='bounded')
    return result.x
