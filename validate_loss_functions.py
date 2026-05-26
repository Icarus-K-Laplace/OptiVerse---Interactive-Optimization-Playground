import numpy as np
from typing import Tuple
from app import LossFunction, create_loss_functions


def numerical_gradient(f: callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """
    计算数值梯度（中心差分）
    
    Args:
        f: 目标函数
        x: 点坐标
        h: 步长
    
    Returns:
        数值梯度
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def numerical_hessian(f: callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """
    计算数值 Hessian 矩阵（中心差分）
    
    Args:
        f: 目标函数
        x: 点坐标
        h: 步长
    
    Returns:
        数值 Hessian 矩阵
    """
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_ii = x.copy()
            x_ii[i] += h
            x_ii[j] += h
            
            x_ij = x.copy()
            x_ij[i] += h
            x_ij[j] -= h
            
            x_ji = x.copy()
            x_ji[i] -= h
            x_ji[j] += h
            
            x_jj = x.copy()
            x_jj[i] -= h
            x_jj[j] -= h
            
            hessian[i, j] = (f(x_ii) - f(x_ij) - f(x_ji) + f(x_jj)) / (4 * h**2)
    return hessian


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个数组的相对误差
    
    Args:
        a: 数组 1
        b: 数组 2
    
    Returns:
        相对误差
    """
    if np.linalg.norm(b) < 1e-12:
        return np.linalg.norm(a - b)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def validate_loss_function(loss_func: LossFunction) -> Tuple[bool, list]:
    """
    验证单个损失函数
    
    Args:
        loss_func: 要验证的损失函数
    
    Returns:
        (是否通过, 警告信息列表)
    """
    print(f"\n{'='*60}")
    print(f"验证函数: {loss_func.name}")
    print(f"{'='*60}")
    
    warnings = []
    passed = True
    
    np.random.seed(42)
    test_points = np.random.uniform(-4, 4, (3, 2))
    
    print(f"\n[1/3] 验证梯度 (3 个随机测试点)")
    for i, point in enumerate(test_points, 1):
        sym_grad = loss_func.gradient(point)
        num_grad = numerical_gradient(loss_func.f, point)
        err = relative_error(sym_grad, num_grad)
        
        if err > 1e-5:
            warning_msg = f"WARNING: 梯度验证失败 - 点 {i} ({point[0]:.4f}, {point[1]:.4f}): 相对误差 = {err:.2e} > 1e-5"
            print(warning_msg)
            warnings.append(warning_msg)
            passed = False
        else:
            print(f"OK: 点 {i} ({point[0]:.4f}, {point[1]:.4f}): 相对误差 = {err:.2e}")
    
    print(f"\n[2/3] 验证 Hessian")
    if loss_func.hessian is not None:
        for i, point in enumerate(test_points, 1):
            sym_hess = loss_func.hessian(point)
            num_hess = numerical_hessian(loss_func.f, point)
            err = relative_error(sym_hess, num_hess)
            
            if err > 1e-5:
                warning_msg = f"WARNING: Hessian 验证失败 - 点 {i} ({point[0]:.4f}, {point[1]:.4f}): 相对误差 = {err:.2e} > 1e-5"
                print(warning_msg)
                warnings.append(warning_msg)
                passed = False
            else:
                print(f"OK: 点 {i} ({point[0]:.4f}, {point[1]:.4f}): 相对误差 = {err:.2e}")
    else:
        print("Hessian 不存在，跳过")
    
    print(f"\n[3/3] 验证全局最小值")
    if loss_func.global_min is not None:
        min_point, expected_val = loss_func.global_min
        actual_val = loss_func(min_point)
        grad_at_min = loss_func.gradient(min_point)
        grad_norm = np.linalg.norm(grad_at_min)
        
        if grad_norm > 1e-6:
            warning_msg = f"WARNING: 最小值点梯度范数过大: ||grad|| = {grad_norm:.2e} > 1e-6"
            print(warning_msg)
            warnings.append(warning_msg)
            passed = False
        else:
            print(f"OK: 最小值点梯度范数: ||grad|| = {grad_norm:.2e}")
        
        val_err = abs(actual_val - expected_val)
        if val_err > 1e-6:
            warning_msg = f"WARNING: 最小值点函数值错误: f(x) = {actual_val:.6e} != 期望值 {expected_val:.6e}, 绝对误差 = {val_err:.2e}"
            print(warning_msg)
            warnings.append(warning_msg)
            passed = False
        else:
            print(f"OK: 最小值点函数值: f(x) = {actual_val:.6e}, 期望值 = {expected_val:.6e}, 绝对误差 = {val_err:.2e}")
    else:
        print("全局最小值未知，跳过")
    
    return passed, warnings


def main():
    print("=" * 60)
    print("  OptiVerse 损失函数验证工具")
    print("=" * 60)
    
    loss_functions = create_loss_functions()
    total_tests = len(loss_functions)
    passed_tests = 0
    all_warnings = []
    
    for func in loss_functions:
        func_passed, func_warnings = validate_loss_function(func)
        all_warnings.extend(func_warnings)
        if func_passed:
            passed_tests += 1
    
    print("\n" + "=" * 60)
    print("  验证总结")
    print("=" * 60)
    print(f"总函数数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {total_tests - passed_tests}")
    
    if all_warnings:
        print("\n警告信息:")
        for warning in all_warnings:
            print(warning)
    
    if passed_tests == total_tests:
        print("\n[PASS] 所有验证通过！")
    else:
        print(f"\n[FAIL] {total_tests - passed_tests} 个验证失败。")


if __name__ == "__main__":
    main()
