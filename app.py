import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy
import time
import json
import csv
from io import StringIO
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any, List
from optimizers import (Optimizer, GradientDescent, Momentum, NewtonMethod, Nesterov,
                       BFGS, LBFGS, ConjugateGradient, TrustRegion)


@dataclass
class LossFunction:
    """Data class representing a loss function and its properties."""
    name: str
    latex_expr: str
    f: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]
    hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None
    global_min: Optional[Tuple[np.ndarray, float]] = None
    is_custom: bool = False

    def __call__(self, x: np.ndarray) -> float:
        return self.f(x)


OPTIMIZER_INFO = {
    "Gradient Descent": {
        "name": "Gradient Descent",
        "formula": r"x_{k+1} = x_k - \alpha \nabla f(x_k)",
        "description": "最基本的优化算法。每次迭代沿着负梯度方向移动，步长由学习率 α 控制。收敛速度较慢，但简单稳定。",
        "pros": ["实现简单", "计算量小", "稳定性好"],
        "cons": ["收敛慢", "步长选择敏感", "对鞍点敏感"],
        "best_for": ["凸函数", "低维问题", "快速原型"]
    },
    "Momentum": {
        "name": "Momentum",
        "formula": r"v_k = \beta v_{k-1} + \alpha \nabla f(x_k)\nx_{k+1} = x_k - v_k",
        "description": "引入动量项 v，模拟物理惯性。有助于加速收敛，尤其是在狭长山谷中。动量系数 β 通常设为 0.9。",
        "pros": ["加速收敛", "克服局部震荡", "适合非凸问题"],
        "cons": ["超参数增多", "可能过冲", "需要调参"],
        "best_for": ["非凸函数", "狭长山谷", "深度学习"]
    },
    "Newton Method": {
        "name": "Newton's Method",
        "formula": r"x_{k+1} = x_k - H(x_k)^{-1} \nabla f(x_k)",
        "description": "利用 Hessian 矩阵进行二阶优化。在二次函数上可一步收敛，但计算成本高。需要正定 Hessian。",
        "pros": ["二次收敛", "精度高", "适合强凸函数"],
        "cons": ["计算量大", "需要 Hessian", "非正定问题不稳定"],
        "best_for": ["二次函数", "局部精细搜索", "小规模问题"]
    },
    "Nesterov": {
        "name": "Nesterov Accelerated Gradient",
        "formula": r"v_k = \beta v_{k-1} + \alpha \nabla f(x_k - \beta v_{k-1})\nx_{k+1} = x_k - v_k",
        "description": "在计算梯度前先向前看一步，获得更好的收敛保证。理论收敛速率优于普通动量法。",
        "pros": ["更快收敛", "理论保证好", "凸函数最优速率"],
        "cons": ["实现稍复杂", "非凸表现一般", "需要调参"],
        "best_for": ["凸函数", "理论分析", "需要保证收敛速率"]
    },
    "BFGS": {
        "name": "BFGS",
        "formula": r"H_{k+1} = H_k + \frac{(y_k y_k^T)}{y_k^T s_k} - \frac{(H_k s_k s_k^T H_k)}{s_k^T H_k s_k}",
        "description": "拟牛顿法，通过拟牛顿条件近似 Hessian 逆矩阵。不需要显式计算 Hessian，收敛快。",
        "pros": ["超线性收敛", "无需 Hessian", "实践表现优秀"],
        "cons": ["内存 O(n²)", "可能非正定", "线搜索敏感"],
        "best_for": ["中等规模", "无 Hessian", "通用优化"]
    },
    "L-BFGS": {
        "name": "Limited-memory BFGS",
        "formula": "存储最近 m 对 (s,y)，用 two-loop recursion 计算方向",
        "description": "BFGS 的内存高效版本，只存储最近 m 次迭代信息。适合大规模优化问题。",
        "pros": ["内存 O(nm)", "大规模适用", "收敛快"],
        "cons": ["需要调 m", "不如 BFGS 精确", "线搜索重要"],
        "best_for": ["大规模问题", "深度学习", "内存受限"]
    },
    "Conjugate Gradient": {
        "name": "Conjugate Gradient",
        "formula": r"\beta_{k+1} = \frac{\|\nabla f(x_{k+1})\|^2}{\|\nabla f(x_k)\|^2}\nd_{k+1} = -\nabla f(x_{k+1}) + \beta_{k+1} d_k",
        "description": "利用方向共轭性加速收敛。在二次函数上最多 n 步收敛，不需要存储完整矩阵。",
        "pros": ["内存 O(n)", "二次收敛", "无需 Hessian"],
        "cons": ["对非二次函数效果下降", "需要精确线搜索", "可能发散"],
        "best_for": ["二次函数", "大规模问题", "对称正定"]
    },
    "Trust Region": {
        "name": "Trust Region",
        "formula": r"\min_p m(p) = f(x_k) + p^T \nabla f(x_k) + \frac{1}{2} p^T H(x_k) p\n\text{s.t. } \|p\| \leq \Delta_k",
        "description": "在信赖域内求解二次模型，根据实际下降与模型下降的比值调整信赖域半径。",
        "pros": ["全局收敛保证", "自动调整步长", "数值稳定"],
        "cons": ["需要 Hessian", "每次迭代求解子问题", "计算量大"],
        "best_for": ["需要全局收敛", "病态问题", "高精度需求"]
    }
}


def create_loss_functions() -> list[LossFunction]:
    """Create a list of built-in loss functions."""
    x, y = sympy.symbols('x y')
    loss_functions = []

    # Rosenbrock function
    a, b = 1, 100
    rosenbrock_expr = (a - x)**2 + b * (y - x**2)**2
    rosenbrock_f = sympy.lambdify((x, y), rosenbrock_expr, 'numpy')
    rosenbrock_grad = sympy.lambdify((x, y), [sympy.diff(rosenbrock_expr, x), sympy.diff(rosenbrock_expr, y)], 'numpy')
    rosenbrock_hess = sympy.lambdify((x, y), sympy.hessian(rosenbrock_expr, (x, y)), 'numpy')
    loss_functions.append(LossFunction(
        name="Rosenbrock",
        latex_expr=r"$f(x,y)=(1-x)^2 + 100(y-x^2)^2$",
        f=lambda v: rosenbrock_f(v[0], v[1]),
        gradient=lambda v: np.array(rosenbrock_grad(v[0], v[1])),
        hessian=lambda v: np.array(rosenbrock_hess(v[0], v[1])),
        global_min=(np.array([1.0, 1.0]), 0.0)
    ))

    # Beale function
    beale_expr = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    beale_f = sympy.lambdify((x, y), beale_expr, 'numpy')
    beale_grad = sympy.lambdify((x, y), [sympy.diff(beale_expr, x), sympy.diff(beale_expr, y)], 'numpy')
    beale_hess = sympy.lambdify((x, y), sympy.hessian(beale_expr, (x, y)), 'numpy')
    loss_functions.append(LossFunction(
        name="Beale",
        latex_expr=r"$f(x,y)=(1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2$",
        f=lambda v: beale_f(v[0], v[1]),
        gradient=lambda v: np.array(beale_grad(v[0], v[1])),
        hessian=lambda v: np.array(beale_hess(v[0], v[1])),
        global_min=(np.array([3.0, 0.5]), 0.0)
    ))

    # Himmelblau function
    himmelblau_expr = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    himmelblau_f = sympy.lambdify((x, y), himmelblau_expr, 'numpy')
    himmelblau_grad = sympy.lambdify((x, y), [sympy.diff(himmelblau_expr, x), sympy.diff(himmelblau_expr, y)], 'numpy')
    himmelblau_hess = sympy.lambdify((x, y), sympy.hessian(himmelblau_expr, (x, y)), 'numpy')
    loss_functions.append(LossFunction(
        name="Himmelblau",
        latex_expr=r"$f(x,y)=(x^2+y-11)^2 + (x+y^2-7)^2$",
        f=lambda v: himmelblau_f(v[0], v[1]),
        gradient=lambda v: np.array(himmelblau_grad(v[0], v[1])),
        hessian=lambda v: np.array(himmelblau_hess(v[0], v[1])),
        global_min=None
    ))

    # Rastrigin function
    A = 10
    rastrigin_expr = 2*A + x**2 - A*sympy.cos(2*sympy.pi*x) + y**2 - A*sympy.cos(2*sympy.pi*y)
    rastrigin_f = sympy.lambdify((x, y), rastrigin_expr, 'numpy')
    rastrigin_grad = sympy.lambdify((x, y), [sympy.diff(rastrigin_expr, x), sympy.diff(rastrigin_expr, y)], 'numpy')
    rastrigin_hess = sympy.lambdify((x, y), sympy.hessian(rastrigin_expr, (x, y)), 'numpy')
    loss_functions.append(LossFunction(
        name="Rastrigin",
        latex_expr=r"$f(x,y)=20 + x^2 - 10\cos(2\pi x) + y^2 - 10\cos(2\pi y)$",
        f=lambda v: rastrigin_f(v[0], v[1]),
        gradient=lambda v: np.array(rastrigin_grad(v[0], v[1])),
        hessian=lambda v: np.array(rastrigin_hess(v[0], v[1])),
        global_min=(np.array([0.0, 0.0]), 0.0)
    ))

    # Booth function
    booth_expr = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    booth_f = sympy.lambdify((x, y), booth_expr, 'numpy')
    booth_grad = sympy.lambdify((x, y), [sympy.diff(booth_expr, x), sympy.diff(booth_expr, y)], 'numpy')
    booth_hess = sympy.lambdify((x, y), sympy.hessian(booth_expr, (x, y)), 'numpy')
    loss_functions.append(LossFunction(
        name="Booth",
        latex_expr=r"$f(x,y)=(x+2y-7)^2 + (2x+y-5)^2$",
        f=lambda v: booth_f(v[0], v[1]),
        gradient=lambda v: np.array(booth_grad(v[0], v[1])),
        hessian=lambda v: np.array(booth_hess(v[0], v[1])),
        global_min=(np.array([1.0, 3.0]), 0.0)
    ))

    # Custom function
    custom_expr = (x - 1)**2 + (y - 2)**2
    custom_f = sympy.lambdify((x, y), custom_expr, 'numpy')
    custom_grad = sympy.lambdify((x, y), [sympy.diff(custom_expr, x), sympy.diff(custom_expr, y)], 'numpy')
    custom_hess = sympy.lambdify((x, y), sympy.hessian(custom_expr, (x, y)), 'numpy')
    loss_functions.append(LossFunction(
        name="Custom Parabola",
        latex_expr=r"$f(x,y)=(x-1)^2 + (y-2)^2$",
        f=lambda v: custom_f(v[0], v[1]),
        gradient=lambda v: np.array(custom_grad(v[0], v[1])),
        hessian=lambda v: np.array(custom_hess(v[0], v[1])),
        global_min=(np.array([1.0, 2.0]), 0.0)
    ))

    return loss_functions


def parse_custom_function(expr_str: str) -> Optional[LossFunction]:
    """Parse a custom function string and create a LossFunction object."""
    try:
        x, y = sympy.symbols('x y')
        expr = sympy.sympify(expr_str)
        
        f = sympy.lambdify((x, y), expr, 'numpy')
        grad_x = sympy.diff(expr, x)
        grad_y = sympy.diff(expr, y)
        grad_func = sympy.lambdify((x, y), [grad_x, grad_y], 'numpy')
        hess_expr = sympy.hessian(expr, (x, y))
        hess_func = sympy.lambdify((x, y), hess_expr, 'numpy')
        
        return LossFunction(
            name=f"Custom: {expr_str}",
            latex_expr=f"$f(x,y)={sympy.latex(expr)}$",
            f=lambda v: float(f(v[0], v[1])),
            gradient=lambda v: np.array(grad_func(v[0], v[1])),
            hessian=lambda v: np.array(hess_func(v[0], v[1])),
            global_min=None,
            is_custom=True
        )
    except Exception as e:
        st.error(f"Invalid expression: {str(e)}")
        return None


def diagnose_optimization(opt_name, stats):
    """Diagnose potential issues in the optimization process."""
    if not stats or not stats['gradient_norms']:
        return None
    
    grad_norms = stats['gradient_norms']
    fvals = stats['function_values']
    
    if len(grad_norms) >= 5:
        recent_grads = grad_norms[-5:]
        avg_grad = np.mean(recent_grads)
        max_grad = np.max(recent_grads)
        
        if max_grad > 1e5:
            return f"⚠️ **梯度爆炸警告**：梯度范数过大 ({max_grad:.2e})，建议降低学习率或使用动量法。"
        
        if avg_grad < 1e-6:
            return "✅ **已收敛**：梯度范数小于 1e-6，优化已完成。"
        
        if len(fvals) >= 10:
            recent_fvals = fvals[-10:]
            if np.std(recent_fvals) < 1e-10 and avg_grad > 1e-3:
                return f"⚠️ **可能处于鞍点**：函数值变化极小但梯度不为零，建议尝试动量法或拟牛顿法。"
        
        if len(grad_norms) >= 20:
            if all(g > 0.1 for g in grad_norms[-10:]):
                return "⚠️ **收敛缓慢**：梯度范数持续较高，考虑增大学习率或使用二阶方法。"
    
    return None


OPTIMIZER_COLORS = {
    "Gradient Descent": "#1f77b4",
    "Momentum": "#ff7f0e",
    "Newton Method": "#2ca02c",
    "Nesterov": "#d62728",
    "BFGS": "#9467bd",
    "L-BFGS": "#8c564b",
    "Conjugate Gradient": "#e377c2",
    "Trust Region": "#7f7f7f"
}


def suggest_parameters(func_name: str):
    """Suggest parameters based on the loss function."""
    params = {
        "Gradient Descent": {"lr": 0.01, "line_search": "none"},
        "Momentum": {"lr": 0.01, "momentum": 0.9, "line_search": "none"},
        "Newton Method": {"damping": 0.0, "line_search": "none"},
        "Nesterov": {"lr": 0.01, "momentum": 0.9, "line_search": "none"},
        "BFGS": {"lr": 1.0, "damping": 1e-6, "line_search": "exact"},
        "L-BFGS": {"m": 10, "lr": 1.0, "line_search": "exact"},
        "Conjugate Gradient": {"method": "FR", "restart_period": None, "line_search": "exact"},
        "Trust Region": {"radius": 1.0, "eta": 0.15, "max_radius": 10.0, "min_radius": 1e-4}
    }
    
    if func_name == "Rosenbrock":
        params["Gradient Descent"]["lr"] = 0.001
        params["Momentum"]["lr"] = 0.001
        params["Nesterov"]["lr"] = 0.001
    elif func_name == "Rastrigin":
        params["Gradient Descent"]["lr"] = 0.01
        params["Momentum"]["lr"] = 0.01
        params["Nesterov"]["lr"] = 0.01
        params["Conjugate Gradient"]["restart_period"] = 2
    elif func_name == "Beale":
        params["Gradient Descent"]["lr"] = 0.005
        params["BFGS"]["line_search"] = "backtrack"
    elif func_name == "Himmelblau":
        params["Trust Region"]["radius"] = 2.0
    
    return params


def plot_contour(loss_func: LossFunction, 
                 x_range: Tuple[float, float] = (-5, 5), 
                 y_range: Tuple[float, float] = (-5, 5), 
                 num_points: int = 100) -> go.Figure:
    """Create elegant contour plot of the loss function."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[loss_func(np.array([xi, yi])) for xi in x] for yi in y])

    colorscale = [
        [0.0, '#0a0a25'],
        [0.1, '#0d1a45'],
        [0.2, '#102a65'],
        [0.3, '#153a85'],
        [0.4, '#1a4aa5'],
        [0.5, '#2560c0'],
        [0.6, '#3075dc'],
        [0.7, '#4a90e6'],
        [0.8, '#6ab0f0'],
        [0.9, '#8fd0fa'],
        [1.0, '#b8e0ff']
    ]

    fig = go.Figure(data=go.Contour(
        x=x,
        y=y,
        z=Z,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='rgba(255,255,255,0.8)', family='Arial'),
            start=np.min(Z),
            end=np.percentile(Z, 95),
            size=(np.percentile(Z, 95) - np.min(Z)) / 15,
            coloring='heatmap'
        ),
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text='f(x,y)', font=dict(color='rgba(255,255,255,0.8)', size=12)),
            tickfont=dict(color='rgba(255,255,255,0.6)', size=10),
            thickness=15,
            len=0.7,
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        ),
        line=dict(color='rgba(0, 212, 255, 0.15)', width=0.5),
        hoverinfo='z',
        hoverlabel=dict(
            bgcolor='rgba(10, 10, 30, 0.95)',
            bordercolor='rgba(0, 212, 255, 0.5)',
            font=dict(color='white', size=12)
        )
    ))

    if loss_func.global_min is not None:
        min_point, min_val = loss_func.global_min
        fig.add_trace(go.Scatter(
            x=[min_point[0]],
            y=[min_point[1]],
            mode='markers',
            marker=dict(
                color='rgba(255, 107, 107, 1)',
                size=14,
                symbol='star',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                opacity=1
            ),
            name=f'Global Min',
            hovertemplate=f'Global Minimum<br>x: {min_point[0]:.4f}<br>y: {min_point[1]:.4f}<br>f(x,y): {min_val:.6f}',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text=f'<span style="color:#00d4ff; font-size: 18px; font-weight: bold;">{loss_func.name}</span>',
            x=0.5,
            y=0.98
        ),
        xaxis_title=dict(text='x', font=dict(color='rgba(255,255,255,0.7)', size=12)),
        yaxis_title=dict(text='y', font=dict(color='rgba(255,255,255,0.7)', size=12)),
        xaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)',
            mirror=True
        ),
        yaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)',
            mirror=True
        ),
        width=380,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def plot_3d_surface(loss_func: LossFunction, 
                    x_range: Tuple[float, float] = (-5, 5), 
                    y_range: Tuple[float, float] = (-5, 5), 
                    num_points: int = 60) -> go.Figure:
    """Create elegant 3D surface plot of the loss function."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[loss_func(np.array([xi, yi])) for xi in x] for yi in y])

    colorscale = [
        [0.0, '#0a0a25'],
        [0.1, '#0d1a45'],
        [0.2, '#102a65'],
        [0.3, '#153a85'],
        [0.4, '#1a4aa5'],
        [0.5, '#2560c0'],
        [0.6, '#3075dc'],
        [0.7, '#4a90e6'],
        [0.8, '#6ab0f0'],
        [0.9, '#8fd0fa'],
        [1.0, '#b8e0ff']
    ]

    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text='f(x,y)', font=dict(color='rgba(255,255,255,0.8)', size=12)),
            tickfont=dict(color='rgba(255,255,255,0.6)', size=10),
            thickness=15,
            len=0.8,
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        ),
        opacity=0.9,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.5,
            roughness=0.2,
            fresnel=0.1
        ),
        lightposition=dict(x=100, y=100, z=50),
        hoverinfo='z',
        hoverlabel=dict(
            bgcolor='rgba(10, 10, 30, 0.95)',
            bordercolor='rgba(0, 212, 255, 0.5)',
            font=dict(color='white', size=12)
        )
    )])

    if loss_func.global_min is not None:
        min_point, min_val = loss_func.global_min
        fig.add_trace(go.Scatter3d(
            x=[min_point[0]],
            y=[min_point[1]],
            z=[min_val],
            mode='markers',
            marker=dict(
                color='rgba(255, 107, 107, 1)',
                size=14,
                symbol='diamond',
                line=dict(color='rgba(255, 255, 255, 0.9)', width=2),
                opacity=1
            ),
            name=f'Global Min',
            hovertemplate=f'Global Minimum<br>x: {min_point[0]:.4f}<br>y: {min_point[1]:.4f}<br>f(x,y): {min_val:.6f}',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text=f'<span style="color:#00d4ff; font-size: 20px; font-weight: bold;">{loss_func.name} - 3D Surface</span>',
            x=0.5,
            y=0.98
        ),
        scene=dict(
            xaxis_title=dict(text='x', font=dict(color='rgba(255,255,255,0.7)', size=12)),
            yaxis_title=dict(text='y', font=dict(color='rgba(255,255,255,0.7)', size=12)),
            zaxis_title=dict(text='f(x,y)', font=dict(color='rgba(255,255,255,0.7)', size=12)),
            xaxis=dict(
                gridcolor='rgba(0, 212, 255, 0.1)',
                gridwidth=1,
                tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
                linecolor='rgba(0, 212, 255, 0.3)',
                backgroundcolor='rgba(10, 10, 30, 0.5)',
                mirror=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 212, 255, 0.1)',
                gridwidth=1,
                tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
                linecolor='rgba(0, 212, 255, 0.3)',
                backgroundcolor='rgba(10, 10, 30, 0.5)',
                mirror=True
            ),
            zaxis=dict(
                gridcolor='rgba(0, 212, 255, 0.1)',
                gridwidth=1,
                tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
                linecolor='rgba(0, 212, 255, 0.3)',
                backgroundcolor='rgba(10, 10, 30, 0.5)',
                mirror=True
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgba(10, 10, 30, 0.8)'
        ),
        width=850,
        height=650,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(10, 10, 30, 0.95)',
        paper_bgcolor='rgba(10, 10, 30, 0.95)'
    )

    return fig


def plot_convergence_curves(selected_optimizers, optimizer_stats, highlight_step=None):
    """Create elegant convergence curves for function values, gradient norms, and step sizes."""
    if not selected_optimizers or not optimizer_stats:
        return None, None, None
    
    max_steps = max(len(stats.get('function_values', [])) for stats in optimizer_stats.values())
    if max_steps == 0:
        return None, None, None
    
    fig_fval = go.Figure()
    for opt_name in selected_optimizers:
        if opt_name in optimizer_stats:
            fvals = optimizer_stats[opt_name].get('function_values', [])
            steps = list(range(len(fvals)))
            fig_fval.add_trace(go.Scatter(
                x=steps,
                y=fvals,
                mode='lines',
                name=opt_name,
                line=dict(color=OPTIMIZER_COLORS[opt_name], width=3),
                hovertemplate=f'<b>{opt_name}</b><br>Step: %{{x}}<br>f(x): %{{y:.6f}}',
                hoverlabel=dict(
                    bgcolor='rgba(10, 10, 30, 0.95)',
                    bordercolor=OPTIMIZER_COLORS[opt_name],
                    font=dict(color='white', size=12)
                )
            ))
    
    if highlight_step is not None:
        fig_fval.add_vline(
            x=highlight_step,
            line_dash="dash",
            line_color='rgba(0, 212, 255, 0.5)',
            annotation_text=f"Step {highlight_step}",
            annotation=dict(
                font=dict(color='rgba(0, 212, 255, 0.8)', size=12),
                bgcolor='rgba(10, 10, 30, 0.8)'
            )
        )
    
    fig_fval.update_layout(
        title=dict(
            text=f'<span style="color:#00d4ff; font-size: 16px; font-weight: bold;">Function Value</span>',
            x=0.5,
            y=0.98
        ),
        xaxis_title=dict(text='Iteration', font=dict(color='rgba(255,255,255,0.7)', size=11)),
        yaxis_title=dict(text='log(f(x))', font=dict(color='rgba(255,255,255,0.7)', size=11)),
        yaxis_type='log',
        xaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)'
        ),
        width=420,
        height=280,
        margin=dict(l=15, r=15, t=40, b=15),
        plot_bgcolor='rgba(10, 10, 30, 0.95)',
        paper_bgcolor='rgba(10, 10, 30, 0.95)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='rgba(255,255,255,0.7)', size=10),
            bgcolor='rgba(10, 10, 30, 0.8)',
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        )
    )

    fig_grad = go.Figure()
    for opt_name in selected_optimizers:
        if opt_name in optimizer_stats:
            grad_norms = optimizer_stats[opt_name].get('gradient_norms', [])
            steps = list(range(len(grad_norms)))
            fig_grad.add_trace(go.Scatter(
                x=steps,
                y=grad_norms,
                mode='lines',
                name=opt_name,
                line=dict(color=OPTIMIZER_COLORS[opt_name], width=3),
                hovertemplate=f'<b>{opt_name}</b><br>Step: %{{x}}<br>||grad||: %{{y:.6f}}',
                hoverlabel=dict(
                    bgcolor='rgba(10, 10, 30, 0.95)',
                    bordercolor=OPTIMIZER_COLORS[opt_name],
                    font=dict(color='white', size=12)
                )
            ))
    
    if highlight_step is not None:
        fig_grad.add_vline(
            x=highlight_step,
            line_dash="dash",
            line_color='rgba(0, 212, 255, 0.5)',
            annotation_text=f"Step {highlight_step}",
            annotation=dict(
                font=dict(color='rgba(0, 212, 255, 0.8)', size=12),
                bgcolor='rgba(10, 10, 30, 0.8)'
            )
        )
    
    fig_grad.update_layout(
        title=dict(
            text=f'<span style="color:#00d4ff; font-size: 16px; font-weight: bold;">Gradient Norm</span>',
            x=0.5,
            y=0.98
        ),
        xaxis_title=dict(text='Iteration', font=dict(color='rgba(255,255,255,0.7)', size=11)),
        yaxis_title=dict(text='||grad(f(x))||', font=dict(color='rgba(255,255,255,0.7)', size=11)),
        yaxis_type='log',
        xaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)'
        ),
        width=420,
        height=280,
        margin=dict(l=15, r=15, t=40, b=15),
        plot_bgcolor='rgba(10, 10, 30, 0.95)',
        paper_bgcolor='rgba(10, 10, 30, 0.95)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='rgba(255,255,255,0.7)', size=10),
            bgcolor='rgba(10, 10, 30, 0.8)',
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        )
    )

    fig_step = go.Figure()
    for opt_name in selected_optimizers:
        if opt_name in optimizer_stats:
            step_sizes = optimizer_stats[opt_name].get('step_sizes', [])
            steps = list(range(1, len(step_sizes) + 1))
            fig_step.add_trace(go.Scatter(
                x=steps,
                y=step_sizes,
                mode='lines+markers',
                name=opt_name,
                line=dict(color=OPTIMIZER_COLORS[opt_name], width=3),
                marker=dict(size=5, color=OPTIMIZER_COLORS[opt_name]),
                hovertemplate=f'<b>{opt_name}</b><br>Step: %{{x}}<br>Step size: %{{y:.6f}}',
                hoverlabel=dict(
                    bgcolor='rgba(10, 10, 30, 0.95)',
                    bordercolor=OPTIMIZER_COLORS[opt_name],
                    font=dict(color='white', size=12)
                )
            ))
    
    if highlight_step is not None:
        fig_step.add_vline(
            x=highlight_step,
            line_dash="dash",
            line_color='rgba(0, 212, 255, 0.5)',
            annotation_text=f"Step {highlight_step}",
            annotation=dict(
                font=dict(color='rgba(0, 212, 255, 0.8)', size=12),
                bgcolor='rgba(10, 10, 30, 0.8)'
            )
        )
    
    fig_step.update_layout(
        title=dict(
            text=f'<span style="color:#00d4ff; font-size: 16px; font-weight: bold;">Step Size</span>',
            x=0.5,
            y=0.98
        ),
        xaxis_title=dict(text='Iteration', font=dict(color='rgba(255,255,255,0.7)', size=11)),
        yaxis_title=dict(text='Step Size', font=dict(color='rgba(255,255,255,0.7)', size=11)),
        xaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 212, 255, 0.08)',
            gridwidth=1,
            tickfont=dict(color='rgba(255,255,255,0.5)', size=10),
            linecolor='rgba(0, 212, 255, 0.2)'
        ),
        width=420,
        height=280,
        margin=dict(l=15, r=15, t=40, b=15),
        plot_bgcolor='rgba(10, 10, 30, 0.95)',
        paper_bgcolor='rgba(10, 10, 30, 0.95)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='rgba(255,255,255,0.7)', size=10),
            bgcolor='rgba(10, 10, 30, 0.8)',
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        )
    )

    return fig_fval, fig_grad, fig_step


def generate_csv_download():
    """Generate CSV download of optimization results."""
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Optimizer', 'Step', 'x', 'y', 'f(x,y)', 'Gradient Norm'])
    
    for opt_name in st.session_state.selected_optimizers:
        if opt_name in st.session_state.optimizer_stats:
            stats = st.session_state.optimizer_stats[opt_name]
            for i, (point, fval, grad_norm) in enumerate(zip(
                stats['points'], 
                stats['function_values'], 
                stats['gradient_norms']
            )):
                writer.writerow([opt_name, i, point[0], point[1], fval, grad_norm])
    
    return output.getvalue()


def generate_config_download():
    """Generate JSON download of optimizer configuration."""
    config = {
        'loss_function': st.session_state.loss_functions[st.session_state.selected_func_index].name,
        'start_point': st.session_state.start_point.tolist() if st.session_state.start_point is not None else None,
        'selected_optimizers': st.session_state.selected_optimizers,
        'optimizer_params': st.session_state.optimizer_params,
        'view_mode': st.session_state.view_mode,
        'learning_mode': st.session_state.get('learning_mode', False)
    }
    return json.dumps(config, indent=2)


def main():
    st.set_page_config(page_title="OptiVerse - Interactive Optimization Playground", layout="wide")
    st.title("🎯 OptiVerse - Interactive Optimization Playground")

    if 'loss_functions' not in st.session_state:
        st.session_state.loss_functions = create_loss_functions()
    if 'custom_func_history' not in st.session_state:
        st.session_state.custom_func_history = []
    if 'selected_func_index' not in st.session_state:
        st.session_state.selected_func_index = 0
    if 'start_point' not in st.session_state:
        st.session_state.start_point = None
    if 'optimizer_states' not in st.session_state:
        st.session_state.optimizer_states = {}
    if 'optimizer_params' not in st.session_state:
        st.session_state.optimizer_params = {
            "Gradient Descent": {"lr": 0.01, "line_search": "none"},
            "Momentum": {"lr": 0.01, "momentum": 0.9, "line_search": "none"},
            "Newton Method": {"damping": 0.0, "line_search": "none"},
            "Nesterov": {"lr": 0.01, "momentum": 0.9, "line_search": "none"},
            "BFGS": {"lr": 1.0, "damping": 1e-6, "line_search": "exact"},
            "L-BFGS": {"m": 10, "lr": 1.0, "line_search": "exact"},
            "Conjugate Gradient": {"method": "FR", "restart_period": None, "line_search": "exact"},
            "Trust Region": {"radius": 1.0, "eta": 0.15, "max_radius": 10.0, "min_radius": 1e-4}
        }
    if 'selected_optimizers' not in st.session_state:
        st.session_state.selected_optimizers = ["Gradient Descent"]
    if 'optimizer_paths' not in st.session_state:
        st.session_state.optimizer_paths = {}
    if 'optimizer_stats' not in st.session_state:
        st.session_state.optimizer_stats = {}
    if 'current_display_step' not in st.session_state:
        st.session_state.current_display_step = 0
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = '2D'
    if 'learning_mode' not in st.session_state:
        st.session_state.learning_mode = False

    st.markdown("""
    <style>
    body {
        overflow-x: hidden;
    }
    
    .stApp {
        background: 
            radial-gradient(ellipse at 20% 20%, rgba(0, 212, 255, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(168, 85, 247, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(255, 107, 107, 0.05) 0%, transparent 70%),
            linear-gradient(135deg, #050510 0%, #0a0a25 25%, #151535 50%, #0a0a25 75%, #050510 100%);
        min-height: 100vh;
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite, pulseGlow 4s ease-in-out infinite;
        position: relative;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulseGlow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    .stHeader {
        background: rgba(0, 212, 255, 0.08);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
    }
    
    .stSidebar {
        background: rgba(8, 8, 25, 0.98);
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(0, 212, 255, 0.15);
        box-shadow: 
            10px 0 40px rgba(0, 212, 255, 0.05),
            inset -5px 0 30px rgba(0, 212, 255, 0.02);
        position: relative;
    }
    
    .stSidebar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0077aa 50%, #005588 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        color: white;
        font-weight: bold;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 20px rgba(0, 212, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        padding: 10px 24px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 14px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00e5ff 0%, #0099cc 50%, #0066aa 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 10px 40px rgba(0, 212, 255, 0.4),
            0 0 60px rgba(0, 212, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
        box-shadow: 
            0 4px 15px rgba(0, 212, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .stExpander {
        background: rgba(0, 212, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 20px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .stExpander:hover {
        border-color: rgba(0, 212, 255, 0.3);
        box-shadow: 0 5px 25px rgba(0, 212, 255, 0.08);
    }
    
    .stExpander > div:first-child {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-radius: 20px 20px 0 0;
        padding: 12px 16px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #a855f7);
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }
    
    .stSlider > div > div > div {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .stSelectbox > div > div {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 16px;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .stMultiselect > div > div {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 16px;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 16px;
        color: white;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
        padding: 12px 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 
            inset 0 2px 10px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    .stToggle > label {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 30px;
        border: 2px solid rgba(0, 212, 255, 0.25);
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .stToggle > label > div {
        background: linear-gradient(135deg, #00d4ff, #a855f7);
        box-shadow: 
            0 0 20px rgba(0, 212, 255, 0.6),
            0 0 40px rgba(0, 212, 255, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-left: 4px solid #00d4ff;
        padding: 20px;
        border-radius: 0 16px 16px 0;
        margin: 15px 0;
        box-shadow: 
            0 5px 25px rgba(0, 212, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%);
        border-left: 4px solid #ff6b6b;
        padding: 20px;
        border-radius: 0 16px 16px 0;
        margin: 15px 0;
        box-shadow: 0 5px 25px rgba(255, 107, 107, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(78, 205, 196, 0.05) 100%);
        border-left: 4px solid #4ecdc4;
        padding: 20px;
        border-radius: 0 16px 16px 0;
        margin: 15px 0;
        box-shadow: 0 5px 25px rgba(78, 205, 196, 0.1);
    }
    
    .glow-text {
        text-shadow: 
            0 0 10px currentColor,
            0 0 20px currentColor,
            0 0 40px currentColor,
            0 0 80px currentColor;
    }
    
    .card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(0, 212, 255, 0.02) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        padding: 24px;
        margin: 15px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 5px 30px rgba(0, 212, 255, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .card:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-color: rgba(0, 212, 255, 0.4);
        transform: translateY(-4px);
        box-shadow: 
            0 15px 50px rgba(0, 212, 255, 0.15),
            0 0 60px rgba(0, 212, 255, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .neon-border {
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 
            0 0 10px rgba(0, 212, 255, 0.2),
            inset 0 0 10px rgba(0, 212, 255, 0.05);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00d4ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 30px 0; margin-bottom: 15px; position: relative;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="position: relative; display: inline-block;">
            <h1 style="font-size: 36px; background: linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #ff6b6b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 8px; animation: textGlow 3s ease-in-out infinite;">🎯 OptiVerse</h1>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 120%; height: 120%; background: radial-gradient(circle, rgba(0, 212, 255, 0.3) 0%, transparent 70%); filter: blur(20px); z-index: -1;"></div>
        </div>
        <style>
            @keyframes textGlow {
                0%, 100% { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5)) drop-shadow(0 0 20px rgba(0, 212, 255, 0.3)); }
                50% { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8)) drop-shadow(0 0 40px rgba(0, 212, 255, 0.5)) drop-shadow(0 0 60px rgba(168, 85, 247, 0.3)); }
            }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255,255,255,0.6); font-size: 15px;">Interactive Optimization Playground</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.session_state.learning_mode = st.toggle("📚 Learning Mode", value=st.session_state.learning_mode)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<h3 style="color: #00d4ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px;">⚙️</span> Loss Function</h3>', unsafe_allow_html=True)
        func_names = [lf.name for lf in st.session_state.loss_functions]
        selected_name = st.selectbox("", func_names, 
                                    index=st.session_state.selected_func_index,
                                    key='func_select')
        st.session_state.selected_func_index = func_names.index(selected_name)
        
        st.markdown("---")
        
        st.markdown('<h3 style="color: #00d4ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px;">✏️</span> Custom Function</h3>', unsafe_allow_html=True)
        custom_expr = st.text_input("Enter expression (x, y):", 
                                   placeholder="e.g., sin(x)*cos(y) + x**2")
        
        if st.button("➕ Add Function"):
            if custom_expr.strip():
                custom_func = parse_custom_function(custom_expr)
                if custom_func:
                    st.session_state.loss_functions.append(custom_func)
                    st.session_state.custom_func_history.append(custom_expr)
                    st.session_state.selected_func_index = len(st.session_state.loss_functions) - 1
                    reset_optimizers()
                    st.success("✅ Custom function added!")
        
        if st.session_state.custom_func_history:
            st.markdown('<p style="color: rgba(255,255,255,0.6); font-size: 13px; margin-bottom: 5px;">History:</p>', unsafe_allow_html=True)
            for i, expr in enumerate(st.session_state.custom_func_history[-3:], 1):
                if st.button(f"↺ {expr}", key=f"history_{i}"):
                    custom_func = parse_custom_function(expr)
                    if custom_func:
                        st.session_state.loss_functions.append(custom_func)
                        st.session_state.selected_func_index = len(st.session_state.loss_functions) - 1
                        reset_optimizers()
        
        selected_func = st.session_state.loss_functions[st.session_state.selected_func_index]
        st.latex(selected_func.latex_expr)

        st.markdown("---")
        
        st.markdown('<h3 style="color: #00d4ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px;">👁️</span> View Mode</h3>', unsafe_allow_html=True)
        view_mode = st.radio("", ['2D', '3D'], index=['2D', '3D'].index(st.session_state.view_mode), horizontal=True)
        st.session_state.view_mode = view_mode

        st.markdown("---")
        
        st.markdown('<h3 style="color: #00d4ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px;">📍</span> Start Point</h3>', unsafe_allow_html=True)
        if st.button("🎲 Random Point", type="primary"):
            st.session_state.start_point = np.random.uniform(-4, 4, 2)
            reset_optimizers()
            st.success(f"✅ Generated: ({st.session_state.start_point[0]:.2f}, {st.session_state.start_point[1]:.2f})")
        
        if st.session_state.start_point is not None:
            st.markdown(f'<p style="color: rgba(255,255,255,0.8);">Current: <span style="color:#00d4ff; font-weight:bold;">({st.session_state.start_point[0]:.4f}, {st.session_state.start_point[1]:.4f})</span></p>', unsafe_allow_html=True)

        st.markdown("---")
        
        st.markdown('<h3 style="color: #00d4ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px;">⚡</span> Optimizers</h3>', unsafe_allow_html=True)
        optimizer_names = ["Gradient Descent", "Momentum", "Newton Method", "Nesterov",
                          "BFGS", "L-BFGS", "Conjugate Gradient", "Trust Region"]
        selected = st.multiselect("", optimizer_names, 
                                 default=st.session_state.selected_optimizers)
        st.session_state.selected_optimizers = selected
        
        if st.button("💡 Suggest Params"):
            suggested = suggest_parameters(selected_func.name)
            for opt_name in selected:
                if opt_name in suggested:
                    st.session_state.optimizer_params[opt_name] = suggested[opt_name]
            st.success("✅ Suggested parameters applied!")
        
        st.markdown('<h4 style="color: rgba(255,255,255,0.7); margin-top: 15px; margin-bottom: 10px;">Hyperparameters</h4>', unsafe_allow_html=True)
        for opt_name in selected:
            with st.expander(opt_name):
                if opt_name == "Gradient Descent":
                    ls_options = ["none", "backtrack", "exact"]
                    line_search = st.selectbox("Line search", ls_options, 
                                              index=ls_options.index(st.session_state.optimizer_params[opt_name]["line_search"]))
                    st.session_state.optimizer_params[opt_name]["line_search"] = line_search
                    use_lr = line_search == "none"
                    if use_lr:
                        lr = st.slider("Learning rate", 0.0001, 0.1, 
                                       st.session_state.optimizer_params[opt_name]["lr"], 0.0001, format="%.4f")
                        st.session_state.optimizer_params[opt_name]["lr"] = lr
                    else:
                        st.info("Line search active - learning rate not used")
                        
                elif opt_name == "Momentum":
                    ls_options = ["none", "backtrack", "exact"]
                    line_search = st.selectbox("Line search", ls_options, 
                                              index=ls_options.index(st.session_state.optimizer_params[opt_name]["line_search"]))
                    st.session_state.optimizer_params[opt_name]["line_search"] = line_search
                    use_lr = line_search == "none"
                    if use_lr:
                        lr = st.slider("Learning rate", 0.0001, 0.1, 
                                       st.session_state.optimizer_params[opt_name]["lr"], 0.0001, format="%.4f")
                        st.session_state.optimizer_params[opt_name]["lr"] = lr
                    else:
                        st.info("Line search active - learning rate not used")
                    momentum = st.slider("Momentum", 0.0, 0.99, 
                                         st.session_state.optimizer_params[opt_name]["momentum"], 0.01)
                    st.session_state.optimizer_params[opt_name]["momentum"] = momentum
                    
                elif opt_name == "Newton Method":
                    ls_options = ["none", "backtrack"]
                    line_search = st.selectbox("Line search", ls_options, 
                                              index=ls_options.index(st.session_state.optimizer_params[opt_name]["line_search"]))
                    st.session_state.optimizer_params[opt_name]["line_search"] = line_search
                    damping = st.slider("Damping", 0.0, 1.0, 
                                        st.session_state.optimizer_params[opt_name]["damping"], 0.001)
                    st.session_state.optimizer_params[opt_name]["damping"] = damping
                    
                elif opt_name == "Nesterov":
                    ls_options = ["none", "backtrack", "exact"]
                    line_search = st.selectbox("Line search", ls_options, 
                                              index=ls_options.index(st.session_state.optimizer_params[opt_name]["line_search"]))
                    st.session_state.optimizer_params[opt_name]["line_search"] = line_search
                    use_lr = line_search == "none"
                    if use_lr:
                        lr = st.slider("Learning rate", 0.0001, 0.1, 
                                       st.session_state.optimizer_params[opt_name]["lr"], 0.0001, format="%.4f")
                        st.session_state.optimizer_params[opt_name]["lr"] = lr
                    else:
                        st.info("Line search active - learning rate not used")
                    momentum = st.slider("Momentum", 0.0, 0.99, 
                                         st.session_state.optimizer_params[opt_name]["momentum"], 0.01)
                    st.session_state.optimizer_params[opt_name]["momentum"] = momentum
                
                elif opt_name == "BFGS":
                    ls_options = ["none", "backtrack", "exact"]
                    line_search = st.selectbox("Line search", ls_options, 
                                              index=ls_options.index(st.session_state.optimizer_params[opt_name]["line_search"]))
                    st.session_state.optimizer_params[opt_name]["line_search"] = line_search
                    use_lr = line_search == "none"
                    if use_lr:
                        lr = st.slider("Learning rate", 0.1, 2.0, 
                                       st.session_state.optimizer_params[opt_name]["lr"], 0.1)
                        st.session_state.optimizer_params[opt_name]["lr"] = lr
                    else:
                        st.info("Line search active - learning rate not used")
                    damping = st.slider("Damping", 0.0, 1e-4, 
                                        st.session_state.optimizer_params[opt_name]["damping"], 1e-7, format="%.1e")
                    st.session_state.optimizer_params[opt_name]["damping"] = damping
                
                elif opt_name == "L-BFGS":
                    ls_options = ["none", "backtrack", "exact"]
                    line_search = st.selectbox("Line search", ls_options, 
                                              index=ls_options.index(st.session_state.optimizer_params[opt_name]["line_search"]))
                    st.session_state.optimizer_params[opt_name]["line_search"] = line_search
                    use_lr = line_search == "none"
                    if use_lr:
                        lr = st.slider("Learning rate", 0.1, 2.0, 
                                       st.session_state.optimizer_params[opt_name]["lr"], 0.1)
                        st.session_state.optimizer_params[opt_name]["lr"] = lr
                    else:
                        st.info("Line search active - learning rate not used")
                    m = st.slider("Memory size (m)", 5, 30, 
                                   st.session_state.optimizer_params[opt_name]["m"], 1)
                    st.session_state.optimizer_params[opt_name]["m"] = m
                
                elif opt_name == "Conjugate Gradient":
                    method_options = ["FR", "PR"]
                    method = st.selectbox("Method", method_options, 
                                         index=method_options.index(st.session_state.optimizer_params[opt_name]["method"]))
                    st.session_state.optimizer_params[opt_name]["method"] = method
                    restart_period = st.slider("Restart period (0 = no restart)", 0, 20, 
                                               st.session_state.optimizer_params[opt_name]["restart_period"] or 0, 1)
                    st.session_state.optimizer_params[opt_name]["restart_period"] = restart_period if restart_period > 0 else None
                
                elif opt_name == "Trust Region":
                    radius = st.slider("Initial radius", 0.1, 5.0, 
                                       st.session_state.optimizer_params[opt_name]["radius"], 0.1)
                    st.session_state.optimizer_params[opt_name]["radius"] = radius
                    eta = st.slider("Eta", 0.01, 0.5, 
                                     st.session_state.optimizer_params[opt_name]["eta"], 0.01)
                    st.session_state.optimizer_params[opt_name]["eta"] = eta
                    max_radius = st.slider("Max radius", 1.0, 20.0, 
                                           st.session_state.optimizer_params[opt_name]["max_radius"], 1.0)
                    st.session_state.optimizer_params[opt_name]["max_radius"] = max_radius
                    min_radius = st.slider("Min radius", 1e-6, 0.1, 
                                           st.session_state.optimizer_params[opt_name]["min_radius"], 1e-5, format="%.1e")
                    st.session_state.optimizer_params[opt_name]["min_radius"] = min_radius
                    if selected_func.hessian is None:
                        st.warning("Hessian not available!")

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Step", type="primary", disabled=st.session_state.is_running):
                if st.session_state.start_point is not None:
                    perform_step(selected_func)
                else:
                    st.warning("Please set a start point first!")
        
        with col2:
            if st.button("🔄 Reset", disabled=st.session_state.is_running):
                reset_optimizers()
                st.success("Optimizers reset")
        
        if not st.session_state.is_running:
            if st.button("🚀 Auto Run", type="primary"):
                st.session_state.is_running = True
                auto_run_optimizers(selected_func)
        else:
            if st.button("⏹️ Stop"):
                st.session_state.is_running = False
        
        st.markdown("---")
        
        st.markdown('<h3 style="color: #00d4ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px;">📥</span> Export Results</h3>', unsafe_allow_html=True)
        csv_data = generate_csv_download()
        st.download_button(
            "📊 Download CSV",
            csv_data,
            "optimization_results.csv",
            "text/csv"
        )
        
        config_data = generate_config_download()
        st.download_button(
            "⚙️ Download Config",
            config_data,
            "optimizer_config.json",
            "application/json"
        )

        st.markdown("---")
        
        st.markdown('<div style="text-align: center; padding: 25px 0; border-top: 1px solid rgba(0, 212, 255, 0.2); position: relative;">', unsafe_allow_html=True)
        st.markdown("""
        <a href="https://github.com" target="_blank" style="display: inline-flex; align-items: center; gap: 12px; color: white; text-decoration: none; padding: 14px 30px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%); border-radius: 20px; border: 1px solid rgba(0, 212, 255, 0.3); transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2);">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1-1-4-1-4 0-5 1-5 3 0 1.15.28 2.35 1 3.5-1.5 1-2 2.28-2 3.5v1c0 3 3 5 6 5.5.65.08 1.31-.23 1.85-.75" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span style="font-size: 16px; font-weight: bold; letter-spacing: 1px;">GitHub</span>
        </a>
        <style>
            a:hover {
                transform: translateY(-3px) scale(1.05) !important;
                box-shadow: 0 10px 40px rgba(0, 212, 255, 0.3) !important;
                border-color: rgba(0, 212, 255, 0.6) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.learning_mode and st.session_state.selected_optimizers:
        st.markdown("""
        <div style="margin-bottom: 25px;">
            <h2 style="color: transparent; background: linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #ff6b6b 100%); -webkit-background-clip: text; background-clip: text; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; font-size: 28px;">
                <span style="color: #00d4ff;">📚</span> Learning Mode - Algorithm Cards
            </h2>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(2)
        for i, opt_name in enumerate(st.session_state.selected_optimizers):
            info = OPTIMIZER_INFO.get(opt_name)
            if info:
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card" style="border-color: {OPTIMIZER_COLORS[opt_name]}30; box-shadow: 0 5px 30px {OPTIMIZER_COLORS[opt_name]}10;">
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background: {OPTIMIZER_COLORS[opt_name]}; box-shadow: 0 0 15px {OPTIMIZER_COLORS[opt_name]};"></div>
                            <h3 style="color: {OPTIMIZER_COLORS[opt_name]}; font-size: 20px; margin: 0;">{info['name']}</h3>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 15px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1);">
                            {info['formula']}
                        </div>
                        <p style="color: rgba(255,255,255,0.9); line-height: 1.6; margin-bottom: 15px;">
                            <span style="color: #00d4ff; font-weight: bold;">核心思想：</span>{info['description']}
                        </p>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                            <div style="background: rgba(78, 205, 196, 0.08); border-radius: 12px; padding: 12px; border-left: 3px solid #4ecdc4;">
                                <p style="color: #4ecdc4; font-weight: bold; margin: 0 0 8px 0;">优点</p>
                                {''.join([f'<p style="color: rgba(255,255,255,0.7); margin: 4px 0; font-size: 13px;">• {p}</p>' for p in info['pros']])}
                            </div>
                            <div style="background: rgba(255, 107, 107, 0.08); border-radius: 12px; padding: 12px; border-left: 3px solid #ff6b6b;">
                                <p style="color: #ff6b6b; font-weight: bold; margin: 0 0 8px 0;">缺点</p>
                                {''.join([f'<p style="color: rgba(255,255,255,0.7); margin: 4px 0; font-size: 13px;">• {c}</p>' for c in info['cons']])}
                            </div>
                        </div>
                        <div style="background: rgba(0, 212, 255, 0.08); border-radius: 12px; padding: 12px;">
                            <p style="color: rgba(255,255,255,0.8); margin: 0;">
                                <span style="color: #00d4ff; font-weight: bold;">适用场景：</span>
                                <span style="color: {OPTIMIZER_COLORS[opt_name]};">{', '.join(info['best_for'])}</span>
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    if st.session_state.selected_optimizers and st.session_state.optimizer_stats:
        st.markdown('<h2 style="color: #00d4ff; margin-bottom: 20px; display: flex; align-items: center; gap: 12px;"><span style="font-size: 28px;">🔍</span> Diagnostics</h2>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, opt_name in enumerate(st.session_state.selected_optimizers):
            if opt_name in st.session_state.optimizer_stats:
                diagnosis = diagnose_optimization(opt_name, st.session_state.optimizer_stats[opt_name])
                if diagnosis:
                    with cols[i % 2]:
                        st.info(diagnosis)

    if st.session_state.view_mode == '3D':
        row1_col1, row1_col2 = st.columns([3, 1])
        
        with row1_col1:
            st.markdown('<h2 style="color: #00d4ff; margin-bottom: 20px; display: flex; align-items: center; gap: 12px;"><span style="font-size: 28px;">📊</span> 3D Surface Visualization</h2>', unsafe_allow_html=True)
            fig_3d = plot_3d_surface(selected_func)
            
            if st.session_state.start_point is not None:
                z_start = selected_func(st.session_state.start_point)
                fig_3d.add_trace(go.Scatter3d(
                    x=[st.session_state.start_point[0]],
                    y=[st.session_state.start_point[1]],
                    z=[z_start],
                    mode='markers',
                    marker=dict(color='orange', size=6),
                    name=f'Start Point'
                ))
            
            display_step = st.session_state.current_display_step
            
            for opt_name in st.session_state.selected_optimizers:
                if opt_name in st.session_state.optimizer_paths:
                    path = st.session_state.optimizer_paths[opt_name]
                    display_idx = min(display_step, len(path) - 1)
                    current_point = path[display_idx]
                    z_val = selected_func(current_point)
                    
                    if len(path) > 1 and display_idx > 0:
                        path_array = np.array(path[:display_idx+1])
                        z_path = np.array([selected_func(p) for p in path[:display_idx+1]])
                        fig_3d.add_trace(go.Scatter3d(
                            x=path_array[:, 0],
                            y=path_array[:, 1],
                            z=z_path,
                            mode='lines',
                            line=dict(color=OPTIMIZER_COLORS[opt_name], width=3),
                            name=f'{opt_name} Path',
                            showlegend=False,
                            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>f(x,y): %{z:.6f}'
                        ))
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=[current_point[0]],
                        y=[current_point[1]],
                        z=[z_val],
                        mode='markers',
                        marker=dict(color=OPTIMIZER_COLORS[opt_name], size=8),
                        name=f'{opt_name}',
                        hovertemplate=f'{opt_name}<br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<br>f(x,y): %{{z:.6f}}'
                    ))
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with row1_col2:
            st.markdown('<h3 style="color: #00d4ff; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;"><span style="font-size: 22px;">📍</span> 2D Contour</h3>', unsafe_allow_html=True)
            fig_2d = plot_contour(selected_func)
            
            if st.session_state.start_point is not None:
                fig_2d.add_trace(go.Scatter(
                    x=[st.session_state.start_point[0]],
                    y=[st.session_state.start_point[1]],
                    mode='markers',
                    marker=dict(color='#ffe66d', size=12),
                    name='Start'
                ))
            
            for opt_name in st.session_state.selected_optimizers:
                if opt_name in st.session_state.optimizer_paths:
                    path = st.session_state.optimizer_paths[opt_name]
                    display_idx = min(display_step, len(path) - 1)
                    current_point = path[display_idx]
                    
                    if len(path) > 1 and display_idx > 0:
                        path_array = np.array(path[:display_idx+1])
                        fig_2d.add_trace(go.Scatter(
                            x=path_array[:, 0],
                            y=path_array[:, 1],
                            mode='lines',
                            line=dict(color=OPTIMIZER_COLORS[opt_name], width=2, dash='dash'),
                            showlegend=False
                        ))
                    
                    fig_2d.add_trace(go.Scatter(
                        x=[current_point[0]],
                        y=[current_point[1]],
                        mode='markers',
                        marker=dict(color=OPTIMIZER_COLORS[opt_name], size=10),
                        name=opt_name
                    ))
            
            st.plotly_chart(fig_2d, use_container_width=True)
            
            st.markdown('<h3 style="color: #00d4ff; margin-top: 20px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;"><span style="font-size: 22px;">📋</span> Information</h3>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**Function:** <span style="color:#00d4ff">{selected_func.name}</span></p>', unsafe_allow_html=True)
            
            if selected_func.global_min is not None:
                min_point, min_val = selected_func.global_min
                st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**Global min:** <span style="color:#4ecdc4">({min_point[0]:.2f}, {min_point[1]:.2f})</span></p>', unsafe_allow_html=True)
            
            if st.session_state.start_point is not None:
                sp = st.session_state.start_point
                st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**Start:** <span style="color:#ffe66d">({sp[0]:.4f}, {sp[1]:.4f})</span></p>', unsafe_allow_html=True)
    
    else:
        row1_col1, row1_col2 = st.columns([2, 1])

        with row1_col1:
            st.markdown('<h2 style="color: #00d4ff; margin-bottom: 20px; display: flex; align-items: center; gap: 12px;"><span style="font-size: 28px;">📊</span> Contour Plot</h2>', unsafe_allow_html=True)
            fig = plot_contour(selected_func)
            fig.update_layout(width=700, height=600)

            if st.session_state.start_point is not None:
                fig.add_trace(go.Scatter(
                    x=[st.session_state.start_point[0]],
                    y=[st.session_state.start_point[1]],
                    mode='markers',
                    marker=dict(color='orange', size=15, symbol='circle'),
                    name=f'Start Point'
                ))

            max_steps = 0
            for opt_name in st.session_state.selected_optimizers:
                if opt_name in st.session_state.optimizer_paths:
                    path_length = len(st.session_state.optimizer_paths[opt_name])
                    max_steps = max(max_steps, path_length)

            display_step = st.session_state.current_display_step
            
            for opt_name in st.session_state.selected_optimizers:
                if opt_name in st.session_state.optimizer_paths:
                    path = st.session_state.optimizer_paths[opt_name]
                    display_idx = min(display_step, len(path) - 1)
                    current_point = path[display_idx]
                    
                    if len(path) > 1 and display_idx > 0:
                        path_array = np.array(path[:display_idx+1])
                        fig.add_trace(go.Scatter(
                            x=path_array[:, 0],
                            y=path_array[:, 1],
                            mode='lines',
                            line=dict(color=OPTIMIZER_COLORS[opt_name], width=2, dash='dash', opacity=0.6),
                            showlegend=False
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=[current_point[0]],
                        y=[current_point[1]],
                        mode='markers',
                        marker=dict(color=OPTIMIZER_COLORS[opt_name], size=12, symbol='circle'),
                        name=f'{opt_name}'
                    ))

            st.plotly_chart(fig, use_container_width=True)

        with row1_col2:
            st.markdown('<h3 style="color: #00d4ff; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;"><span style="font-size: 22px;">📋</span> Information</h3>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**Function:** <span style="color:#00d4ff">{selected_func.name}</span></p>', unsafe_allow_html=True)
            
            if selected_func.global_min is not None:
                min_point, min_val = selected_func.global_min
                st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**Global min:** <span style="color:#4ecdc4">({min_point[0]:.2f}, {min_point[1]:.2f})</span></p>', unsafe_allow_html=True)

            if st.session_state.start_point is not None:
                sp = st.session_state.start_point
                st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**Start:** <span style="color:#ffe66d">({sp[0]:.4f}, {sp[1]:.4f})</span></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**f(start):** <span style="color:#a855f7">{selected_func(sp):.4f}</span></p>', unsafe_allow_html=True)
                grad = selected_func.gradient(sp)
                st.markdown(f'<p style="color: rgba(255,255,255,0.8);">**grad(start):** <span style="color:#f472b6">({grad[0]:.4f}, {grad[1]:.4f})</span></p>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<h3 style="color: #00d4ff; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;"><span style="font-size: 22px;">📈</span> Progress</h3>', unsafe_allow_html=True)
            
            for opt_name in st.session_state.selected_optimizers:
                if opt_name in st.session_state.optimizer_stats:
                    stats = st.session_state.optimizer_stats[opt_name]
                    if stats['function_values']:
                        idx = min(display_step, len(stats['function_values']) - 1)
                        func_val = stats['function_values'][idx]
                        grad_norm = stats['gradient_norms'][idx]
                        
                        st.markdown(f'<p style="color: {OPTIMIZER_COLORS[opt_name]}; font-weight: bold;">{opt_name}:</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: rgba(255,255,255,0.7);">  f(x): {func_val:.6f}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: rgba(255,255,255,0.7);">  ||grad||: {grad_norm:.6f}</p>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<h3 style="color: #00d4ff; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;"><span style="font-size: 22px;">🏆</span> Summary</h3>', unsafe_allow_html=True)
            for opt_name in st.session_state.selected_optimizers:
                if opt_name in st.session_state.optimizer_stats:
                    stats = st.session_state.optimizer_stats[opt_name]
                    if len(stats['function_values']) > 0:
                        final_point = stats['points'][-1]
                        final_fval = stats['function_values'][-1]
                        final_grad_norm = stats['gradient_norms'][-1]
                        num_steps = len(stats['function_values']) - 1
                        
                        converged = final_grad_norm < 1e-6
                        status_color = "#4ecdc4" if converged else "#ff6b6b"
                        status = "✓ Converged" if converged else "◌ Not converged"
                        
                        st.markdown(f'<p style="color: {OPTIMIZER_COLORS[opt_name]}; font-weight: bold;">{opt_name}:</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: {status_color};">  {status}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: rgba(255,255,255,0.7);">  Final: ({final_point[0]:.4f}, {final_point[1]:.4f})</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: rgba(255,255,255,0.7);">  Iterations: {num_steps}</p>', unsafe_allow_html=True)

    max_steps = max(len(path) for path in st.session_state.optimizer_paths.values()) if st.session_state.optimizer_paths else 0
    if max_steps > 1:
        st.markdown('<h2 style="color: #00d4ff; margin-bottom: 20px; display: flex; align-items: center; gap: 12px;"><span style="font-size: 28px;">📈</span> Convergence Analysis</h2>', unsafe_allow_html=True)
        st.slider("Step playback", 0, max_steps - 1, 
                  value=st.session_state.current_display_step,
                  key='current_display_step')
        
        fig_fval, fig_grad, fig_step = plot_convergence_curves(
            st.session_state.selected_optimizers,
            st.session_state.optimizer_stats,
            st.session_state.current_display_step
        )
        
        if fig_fval:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig_fval, use_container_width=True)
            with col2:
                st.plotly_chart(fig_grad, use_container_width=True)
            with col3:
                st.plotly_chart(fig_step, use_container_width=True)


def reset_optimizers():
    """Reset all optimizers to the start point."""
    if st.session_state.start_point is not None:
        st.session_state.optimizer_states = {}
        st.session_state.optimizer_paths = {}
        st.session_state.optimizer_stats = {}
        st.session_state.current_display_step = 0
        
        for opt_name in st.session_state.selected_optimizers:
            st.session_state.optimizer_states[opt_name] = st.session_state.start_point.copy()
            st.session_state.optimizer_paths[opt_name] = [st.session_state.start_point.copy()]
            st.session_state.optimizer_stats[opt_name] = {
                'function_values': [],
                'gradient_norms': [],
                'step_sizes': [],
                'points': []
            }


def perform_step(loss_func: LossFunction):
    """Perform one optimization step for all selected optimizers."""
    for opt_name in st.session_state.selected_optimizers:
        if opt_name not in st.session_state.optimizer_states:
            st.session_state.optimizer_states[opt_name] = st.session_state.start_point.copy()
            st.session_state.optimizer_paths[opt_name] = [st.session_state.start_point.copy()]
            st.session_state.optimizer_stats[opt_name] = {
                'function_values': [],
                'gradient_norms': [],
                'step_sizes': [],
                'points': []
            }
        
        current_point = st.session_state.optimizer_states[opt_name]
        grad = loss_func.gradient(current_point)
        hess = loss_func.hessian(current_point) if loss_func.hessian else None
        
        params = st.session_state.optimizer_params[opt_name]
        
        if opt_name == "Gradient Descent":
            optimizer = GradientDescent(lr=params["lr"], line_search=params["line_search"])
            optimizer.set_objective(loss_func.f)
        elif opt_name == "Momentum":
            optimizer = Momentum(lr=params["lr"], momentum=params["momentum"], line_search=params["line_search"])
            optimizer.set_objective(loss_func.f)
            if opt_name in st.session_state.get("_optimizer_instances", {}):
                optimizer.v = st.session_state["_optimizer_instances"][opt_name].v
        elif opt_name == "Newton Method":
            optimizer = NewtonMethod(damping=params["damping"], line_search=params["line_search"])
            optimizer.set_objective(loss_func.f)
        elif opt_name == "Nesterov":
            optimizer = Nesterov(lr=params["lr"], momentum=params["momentum"], line_search=params["line_search"])
            optimizer.set_objective(loss_func.f)
            if opt_name in st.session_state.get("_optimizer_instances", {}):
                optimizer.v = st.session_state["_optimizer_instances"][opt_name].v
        elif opt_name == "BFGS":
            optimizer = BFGS(lr=params["lr"], damping=params["damping"], line_search=params["line_search"])
            optimizer.set_objective(loss_func.f)
            if opt_name in st.session_state.get("_optimizer_instances", {}):
                optimizer.H = st.session_state["_optimizer_instances"][opt_name].H
                optimizer.grad_prev = st.session_state["_optimizer_instances"][opt_name].grad_prev
        elif opt_name == "L-BFGS":
            optimizer = LBFGS(m=params["m"], lr=params["lr"], line_search=params["line_search"])
            optimizer.set_objective(loss_func.f)
            if opt_name in st.session_state.get("_optimizer_instances", {}):
                optimizer.s_list = st.session_state["_optimizer_instances"][opt_name].s_list
                optimizer.y_list = st.session_state["_optimizer_instances"][opt_name].y_list
                optimizer.rho_list = st.session_state["_optimizer_instances"][opt_name].rho_list
                optimizer.grad_prev = st.session_state["_optimizer_instances"][opt_name].grad_prev
        elif opt_name == "Conjugate Gradient":
            optimizer = ConjugateGradient(method=params["method"], restart_period=params["restart_period"])
            optimizer.set_objective(loss_func.f)
            if opt_name in st.session_state.get("_optimizer_instances", {}):
                optimizer.direction = st.session_state["_optimizer_instances"][opt_name].direction
                optimizer.grad_norm_sq_prev = st.session_state["_optimizer_instances"][opt_name].grad_norm_sq_prev
                optimizer.grad_prev = st.session_state["_optimizer_instances"][opt_name].grad_prev
                optimizer.iteration = st.session_state["_optimizer_instances"][opt_name].iteration
        elif opt_name == "Trust Region":
            optimizer = TrustRegion(radius=params["radius"], eta=params["eta"], 
                                    max_radius=params["max_radius"], min_radius=params["min_radius"])
            optimizer.set_objective(loss_func.f)
        
        result = optimizer.step(current_point, grad, hess)
        
        if result[0] is not None:
            st.session_state.optimizer_states[opt_name] = result[0]
            st.session_state.optimizer_paths[opt_name].append(result[0].copy())
            
            stats = st.session_state.optimizer_stats[opt_name]
            stats['function_values'].append(loss_func(result[0]))
            stats['gradient_norms'].append(np.linalg.norm(loss_func.gradient(result[0])))
            stats['step_sizes'].append(result[1].get("step_size", 1.0))
            stats['points'].append(result[0].copy())
            
            if opt_name in ["Momentum", "Nesterov", "BFGS", "L-BFGS", "Conjugate Gradient", "Trust Region"]:
                if "_optimizer_instances" not in st.session_state:
                    st.session_state["_optimizer_instances"] = {}
                st.session_state["_optimizer_instances"][opt_name] = optimizer


def auto_run_optimizers(loss_func: LossFunction):
    """Run all optimizers until convergence or max steps."""
    max_steps = 1000
    convergence_threshold = 1e-6
    progress_bar = st.progress(0)
    
    for step in range(max_steps):
        if not st.session_state.is_running:
            break
        
        perform_step(loss_func)
        
        all_converged = True
        for opt_name in st.session_state.selected_optimizers:
            if opt_name in st.session_state.optimizer_stats:
                stats = st.session_state.optimizer_stats[opt_name]
                if stats['gradient_norms']:
                    if stats['gradient_norms'][-1] >= convergence_threshold:
                        all_converged = False
        
        progress_bar.progress((step + 1) / max_steps)
        st.session_state.current_display_step = step + 1
        
        if all_converged:
            break
        
        time.sleep(0.01)
    
    st.session_state.is_running = False
    progress_bar.empty()
    st.success("Auto run completed!")


if __name__ == "__main__":
    main()
