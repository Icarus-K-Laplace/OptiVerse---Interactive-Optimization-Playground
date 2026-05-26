from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Literal
from line_search import backtracking_line_search, exact_line_search


class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform one optimization step.
        
        Args:
            x: Current point
            grad: Gradient at current point
            hessian: Hessian matrix at current point (optional)
        
        Returns:
            tuple: (new_point, info_dict)
        """
        pass
    
    def reset(self):
        """Reset optimizer state."""
        pass


class GradientDescent(Optimizer):
    """Gradient Descent optimizer with optional line search."""
    
    def __init__(self, lr: float = 0.01, line_search: Literal['none', 'backtrack', 'exact'] = 'none'):
        super().__init__("Gradient Descent")
        self.lr = lr
        self.line_search = line_search
        self.f = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        direction = -grad
        
        if self.line_search == 'none':
            step_size = self.lr
            x_new = x + step_size * direction
        elif self.line_search == 'backtrack':
            if self.f is None:
                step_size = self.lr
                x_new = x + step_size * direction
            else:
                step_size = backtracking_line_search(self.f, x, direction, grad)
                x_new = x + step_size * direction
        elif self.line_search == 'exact':
            if self.f is None:
                step_size = self.lr
                x_new = x + step_size * direction
            else:
                step_size = exact_line_search(self.f, x, direction)
                x_new = x + step_size * direction
        else:
            step_size = self.lr
            x_new = x + step_size * direction
        
        info = {
            "step_size": step_size,
            "direction_norm": np.linalg.norm(direction)
        }
        return x_new, info


class Momentum(Optimizer):
    """Momentum optimizer with optional line search."""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, line_search: Literal['none', 'backtrack', 'exact'] = 'none'):
        super().__init__("Momentum")
        self.lr = lr
        self.momentum = momentum
        self.line_search = line_search
        self.v = None
        self.f = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if self.v is None:
            self.v = np.zeros_like(x)
        
        self.v = self.momentum * self.v + self.lr * grad
        direction = -self.v
        
        if self.line_search == 'none':
            step_size = 1.0
            x_new = x + step_size * direction
        elif self.line_search == 'backtrack':
            if self.f is None:
                step_size = 1.0
                x_new = x + step_size * direction
            else:
                step_size = backtracking_line_search(self.f, x, direction, grad)
                x_new = x + step_size * direction
        elif self.line_search == 'exact':
            if self.f is None:
                step_size = 1.0
                x_new = x + step_size * direction
            else:
                step_size = exact_line_search(self.f, x, direction)
                x_new = x + step_size * direction
        else:
            step_size = 1.0
            x_new = x + step_size * direction
        
        info = {
            "step_size": step_size,
            "velocity_norm": np.linalg.norm(self.v),
            "direction_norm": np.linalg.norm(direction)
        }
        return x_new, info
    
    def reset(self):
        self.v = None


class NewtonMethod(Optimizer):
    """Newton's Method optimizer with damping and optional line search."""
    
    def __init__(self, damping: float = 0.0, line_search: Literal['none', 'backtrack'] = 'none'):
        super().__init__("Newton Method")
        self.damping = damping
        self.line_search = line_search
        self.f = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if hessian is None:
            return None, {"error": "Hessian matrix is required for Newton's method"}
        
        n = len(x)
        H = hessian
        
        if self.damping > 0:
            H = H + self.damping * np.eye(n)
        
        try:
            H_inv = np.linalg.inv(H)
            direction = -H_inv @ grad
            
            if self.line_search == 'none':
                step_size = 1.0
                x_new = x + step_size * direction
            elif self.line_search == 'backtrack':
                if self.f is None:
                    step_size = 1.0
                    x_new = x + step_size * direction
                else:
                    step_size = backtracking_line_search(self.f, x, direction, grad)
                    x_new = x + step_size * direction
            else:
                step_size = 1.0
                x_new = x + step_size * direction
            
            info = {
                "step_size": step_size,
                "direction_norm": np.linalg.norm(direction),
                "condition_number": np.linalg.cond(H)
            }
            return x_new, info
        except np.linalg.LinAlgError:
            return None, {"error": "Hessian matrix is singular"}


class Nesterov(Optimizer):
    """Nesterov Accelerated Gradient optimizer with optional line search."""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, line_search: Literal['none', 'backtrack', 'exact'] = 'none'):
        super().__init__("Nesterov")
        self.lr = lr
        self.momentum = momentum
        self.line_search = line_search
        self.v = None
        self.f = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if self.v is None:
            self.v = np.zeros_like(x)
        
        v_prev = self.v.copy()
        self.v = self.momentum * self.v + self.lr * grad
        direction = -self.momentum * v_prev - (1 + self.momentum) * (self.v - self.momentum * v_prev)
        
        if self.line_search == 'none':
            step_size = 1.0
            x_new = x + step_size * direction
        elif self.line_search == 'backtrack':
            if self.f is None:
                step_size = 1.0
                x_new = x + step_size * direction
            else:
                step_size = backtracking_line_search(self.f, x, direction, grad)
                x_new = x + step_size * direction
        elif self.line_search == 'exact':
            if self.f is None:
                step_size = 1.0
                x_new = x + step_size * direction
            else:
                step_size = exact_line_search(self.f, x, direction)
                x_new = x + step_size * direction
        else:
            step_size = 1.0
            x_new = x + step_size * direction
        
        info = {
            "step_size": step_size,
            "velocity_norm": np.linalg.norm(self.v),
            "direction_norm": np.linalg.norm(direction)
        }
        return x_new, info
    
    def reset(self):
        self.v = None


class BFGS(Optimizer):
    """BFGS Quasi-Newton optimizer."""
    
    def __init__(self, lr: float = 1.0, damping: float = 1e-6, line_search: Literal['none', 'backtrack', 'exact'] = 'exact'):
        super().__init__("BFGS")
        self.lr = lr
        self.damping = damping
        self.line_search = line_search
        self.H = None
        self.f = None
        self.grad_prev = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        n = len(x)
        
        if self.H is None:
            self.H = np.eye(n)
        
        direction = -self.H @ grad
        
        if self.line_search == 'none':
            step_size = self.lr
            x_new = x + step_size * direction
        elif self.line_search == 'backtrack':
            if self.f is None:
                step_size = self.lr
                x_new = x + step_size * direction
            else:
                step_size = backtracking_line_search(self.f, x, direction, grad)
                x_new = x + step_size * direction
        elif self.line_search == 'exact':
            if self.f is None:
                step_size = self.lr
                x_new = x + step_size * direction
            else:
                step_size = exact_line_search(self.f, x, direction)
                x_new = x + step_size * direction
        else:
            step_size = self.lr
            x_new = x + step_size * direction
        
        if self.grad_prev is not None and step_size > 0:
            s = x_new - x
            y = grad - self.grad_prev
            
            ys = y @ s
            if ys > 1e-10:
                Hy = self.H @ y
                yHy = y @ Hy
                
                self.H = self.H + (ys + yHy) * np.outer(s, s) / (ys ** 2) - \
                         (np.outer(Hy, s) + np.outer(s, Hy)) / ys
                
                self.H = (self.H + self.H.T) / 2
                
                if self.damping > 0:
                    eigvals = np.linalg.eigvalsh(self.H)
                    min_eig = np.min(eigvals)
                    if min_eig < self.damping:
                        self.H = self.H + (self.damping - min_eig) * np.eye(n)
        
        self.grad_prev = grad.copy()
        
        info = {
            "step_size": step_size,
            "direction_norm": np.linalg.norm(direction),
            "condition_number": np.linalg.cond(self.H)
        }
        return x_new, info
    
    def reset(self):
        self.H = None
        self.grad_prev = None


class LBFGS(Optimizer):
    """Limited-memory BFGS optimizer."""
    
    def __init__(self, m: int = 10, lr: float = 1.0, line_search: Literal['none', 'backtrack', 'exact'] = 'exact'):
        super().__init__("L-BFGS")
        self.m = m
        self.lr = lr
        self.line_search = line_search
        self.s_list = []
        self.y_list = []
        self.rho_list = []
        self.f = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def two_loop_recursion(self, grad: np.ndarray) -> np.ndarray:
        """Two-loop recursion to compute search direction."""
        q = grad.copy()
        alpha_list = []
        
        for i in range(len(self.s_list) - 1, -1, -1):
            alpha = self.rho_list[i] * (self.s_list[i] @ q)
            alpha_list.append(alpha)
            q = q - alpha * self.y_list[i]
        
        if len(self.s_list) > 0:
            gamma = (self.s_list[-1] @ self.y_list[-1]) / (self.y_list[-1] @ self.y_list[-1])
        else:
            gamma = 1.0
        
        r = gamma * q
        
        alpha_list.reverse()
        for i in range(len(self.s_list)):
            beta = self.rho_list[i] * (self.y_list[i] @ r)
            r = r + alpha_list[i] * self.s_list[i] - beta * self.y_list[i]
        
        return -r
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        direction = self.two_loop_recursion(grad)
        
        if self.line_search == 'none':
            step_size = self.lr
            x_new = x + step_size * direction
        elif self.line_search == 'backtrack':
            if self.f is None:
                step_size = self.lr
                x_new = x + step_size * direction
            else:
                step_size = backtracking_line_search(self.f, x, direction, grad)
                x_new = x + step_size * direction
        elif self.line_search == 'exact':
            if self.f is None:
                step_size = self.lr
                x_new = x + step_size * direction
            else:
                step_size = exact_line_search(self.f, x, direction)
                x_new = x + step_size * direction
        else:
            step_size = self.lr
            x_new = x + step_size * direction
        
        if step_size > 0:
            s = x_new - x
            y = grad - self.grad_prev if hasattr(self, 'grad_prev') else np.zeros_like(grad)
            
            ys = y @ s
            if ys > 1e-10:
                self.s_list.append(s)
                self.y_list.append(y)
                self.rho_list.append(1.0 / ys)
                
                if len(self.s_list) > self.m:
                    self.s_list.pop(0)
                    self.y_list.pop(0)
                    self.rho_list.pop(0)
        
        self.grad_prev = grad.copy()
        
        info = {
            "step_size": step_size,
            "direction_norm": np.linalg.norm(direction),
            "memory_used": len(self.s_list)
        }
        return x_new, info
    
    def reset(self):
        self.s_list = []
        self.y_list = []
        self.rho_list = []
        self.grad_prev = None


class ConjugateGradient(Optimizer):
    """Conjugate Gradient optimizer."""
    
    def __init__(self, method: Literal['FR', 'PR'] = 'FR', restart_period: int = None, line_search: Literal['exact'] = 'exact'):
        super().__init__("Conjugate Gradient")
        self.method = method
        self.restart_period = restart_period
        self.line_search = line_search
        self.direction = None
        self.grad_norm_sq_prev = None
        self.iteration = 0
        self.f = None
    
    def set_objective(self, f):
        """Set the objective function for line search."""
        self.f = f
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        self.iteration += 1
        grad_norm_sq = grad @ grad
        
        if self.direction is None or (self.restart_period and self.iteration % self.restart_period == 0):
            self.direction = -grad
            beta = 0.0
        else:
            if self.method == 'FR':
                beta = grad_norm_sq / self.grad_norm_sq_prev
            else:
                grad_diff = grad - self.grad_prev
                beta = max(0.0, (grad @ grad_diff) / self.grad_norm_sq_prev)
            
            self.direction = -grad + beta * self.direction
        
        if self.line_search == 'exact' and self.f is not None:
            step_size = exact_line_search(self.f, x, self.direction)
        else:
            step_size = 1.0
        
        x_new = x + step_size * self.direction
        
        self.grad_norm_sq_prev = grad_norm_sq
        self.grad_prev = grad.copy()
        
        info = {
            "step_size": step_size,
            "direction_norm": np.linalg.norm(self.direction),
            "beta": beta,
            "iteration": self.iteration
        }
        return x_new, info
    
    def reset(self):
        self.direction = None
        self.grad_norm_sq_prev = None
        self.grad_prev = None
        self.iteration = 0


class TrustRegion(Optimizer):
    """Trust Region optimizer using dogleg method."""
    
    def __init__(self, radius: float = 1.0, eta: float = 0.15, max_radius: float = 10.0, min_radius: float = 1e-4):
        super().__init__("Trust Region")
        self.radius = radius
        self.eta = eta
        self.max_radius = max_radius
        self.min_radius = min_radius
    
    def dogleg(self, grad: np.ndarray, hessian: np.ndarray, delta: float) -> np.ndarray:
        """Dogleg method for solving trust region subproblem."""
        n = len(grad)
        
        try:
            H_inv = np.linalg.inv(hessian)
            p_cauchy = -(grad @ grad) / (grad @ hessian @ grad) * grad
            p_newton = -H_inv @ grad
            
            p_cauchy_norm = np.linalg.norm(p_cauchy)
            
            if p_cauchy_norm >= delta:
                return (delta / p_cauchy_norm) * p_cauchy
            
            p_newton_norm = np.linalg.norm(p_newton)
            if p_newton_norm <= delta:
                return p_newton
            
            a = np.linalg.norm(p_newton - p_cauchy) ** 2
            b = 2 * (p_cauchy @ (p_newton - p_cauchy))
            c = p_cauchy_norm ** 2 - delta ** 2
            
            tau = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            
            return p_cauchy + tau * (p_newton - p_cauchy)
        except np.linalg.LinAlgError:
            return -(delta / np.linalg.norm(grad)) * grad
    
    def step(self, x: np.ndarray, grad: np.ndarray, hessian: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if hessian is None:
            return None, {"error": "Hessian matrix is required for Trust Region method"}
        
        p = self.dogleg(grad, hessian, self.radius)
        p_norm = np.linalg.norm(p)
        
        if self.f is None:
            return x + p, {
                "step_size": p_norm,
                "direction_norm": p_norm,
                "radius": self.radius,
                "rho": None
            }
        
        f_x = self.f(x)
        f_x_new = self.f(x + p)
        
        m_p = f_x + grad @ p + 0.5 * p @ hessian @ p
        
        if m_p < f_x:
            rho = (f_x - f_x_new) / (f_x - m_p)
        else:
            rho = 0.0
        
        if rho < 0.25:
            self.radius = max(self.min_radius, 0.25 * self.radius)
        elif rho > 0.75 and p_norm >= 0.95 * self.radius:
            self.radius = min(self.max_radius, 2.0 * self.radius)
        
        if rho > self.eta:
            x_new = x + p
        else:
            x_new = x
        
        info = {
            "step_size": p_norm,
            "direction_norm": p_norm,
            "radius": self.radius,
            "rho": rho,
            "accepted": rho > self.eta
        }
        return x_new, info
    
    def set_objective(self, f):
        """Set the objective function for trust region."""
        self.f = f
    
    def reset(self):
        self.radius = 1.0


def create_optimizers() -> list[Optimizer]:
    """Create default instances of all optimizers."""
    return [
        GradientDescent(lr=0.01),
        Momentum(lr=0.01, momentum=0.9),
        NewtonMethod(damping=0.0),
        Nesterov(lr=0.01, momentum=0.9),
        BFGS(lr=1.0),
        LBFGS(m=10),
        ConjugateGradient(method='FR'),
        TrustRegion(radius=1.0)
    ]
