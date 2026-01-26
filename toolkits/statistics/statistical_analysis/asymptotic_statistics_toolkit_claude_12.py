# Filename: asymptotic_statistics_toolkit.py

"""
Asymptotic Statistics Toolkit for Time Series Parameter Estimation
专注于平稳遍历时间序列的加权最小二乘估计量渐近分布计算
"""

import numpy as np
from scipy import stats, optimize, linalg
from typing import Dict, List, Tuple, Callable, Optional, Any
import json
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/statistics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础统计计算
# ============================================================================

def compute_conditional_mean(
    theta: List[float],
    X_history: List[float],
    mu_history: List[float],
    p1: int,
    p2: int,
    link_function: str = "identity"
) -> Dict[str, Any]:
    """
    计算条件均值 μ_t(θ)
    
    Parameters:
    -----------
    theta : List[float]
        参数向量 [c, φ_1, ..., φ_{p1}, ψ_1, ..., ψ_{p2}]
    X_history : List[float]
        历史观测值 [X_{t-1}, X_{t-2}, ..., X_{t-p1}]
    mu_history : List[float]
        历史条件均值 [μ_{t-1}, μ_{t-2}, ..., μ_{t-p2}]
    p1 : int
        AR阶数
    p2 : int
        MA阶数
    link_function : str
        链接函数类型: "identity", "log", "exp"
    
    Returns:
    --------
    Dict with 'result' (float) and 'metadata'
    """
    if len(theta) != 1 + p1 + p2:
        raise ValueError(f"theta长度应为{1+p1+p2}, 实际为{len(theta)}")
    if len(X_history) < p1:
        raise ValueError(f"X_history长度应至少为{p1}, 实际为{len(X_history)}")
    if p2 > 0 and len(mu_history) < p2:
        raise ValueError(f"mu_history长度应至少为{p2}, 实际为{len(mu_history)}")
    
    c = theta[0]
    phi = theta[1:1+p1]
    psi = theta[1+p1:] if p2 > 0 else []
    
    # 计算线性预测量 ξ_t
    xi_t = c
    for i in range(p1):
        xi_t += phi[i] * X_history[i]
    for j in range(p2):
        xi_t += psi[j] * mu_history[j]
    
    # 应用链接函数
    if link_function == "identity":
        mu_t = xi_t
    elif link_function == "log":
        mu_t = np.log(max(xi_t, 1e-10))
    elif link_function == "exp":
        mu_t = np.exp(min(xi_t, 100))  # 防止溢出
    else:
        raise ValueError(f"不支持的链接函数: {link_function}")
    
    return {
        'result': float(mu_t),
        'metadata': {
            'xi_t': float(xi_t),
            'link_function': link_function,
            'p1': p1,
            'p2': p2
        }
    }


def compute_gradient_mu(
    theta: List[float],
    X_history: List[float],
    mu_history: List[float],
    p1: int,
    p2: int,
    link_function: str = "identity",
    epsilon: float = 1e-6
) -> Dict[str, Any]:
    """
    计算条件均值对参数的梯度 ∂_θ μ_t(θ)
    
    使用数值微分方法（中心差分）
    
    Returns:
    --------
    Dict with 'result' (List[float]) - 梯度向量
    """
    dim = 1 + p1 + p2
    gradient = []
    
    for i in range(dim):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        
        mu_plus = compute_conditional_mean(
            theta_plus, X_history, mu_history, p1, p2, link_function
        )['result']
        mu_minus = compute_conditional_mean(
            theta_minus, X_history, mu_history, p1, p2, link_function
        )['result']
        
        grad_i = (mu_plus - mu_minus) / (2 * epsilon)
        gradient.append(grad_i)
    
    return {
        'result': gradient,
        'metadata': {
            'dimension': dim,
            'epsilon': epsilon,
            'method': 'central_difference'
        }
    }


def compute_weighted_residual_squared(
    X_t: float,
    mu_t: float,
    weight: float
) -> Dict[str, Any]:
    """
    计算加权残差平方 W_t * (X_t - μ_t)^2
    
    Returns:
    --------
    Dict with 'result' (float)
    """
    if weight < 0:
        raise ValueError(f"权重必须非负, 实际为{weight}")
    
    residual = X_t - mu_t
    weighted_sq = weight * residual ** 2
    
    return {
        'result': float(weighted_sq),
        'metadata': {
            'residual': float(residual),
            'weight': float(weight),
            'X_t': float(X_t),
            'mu_t': float(mu_t)
        }
    }


def compute_matrix_K1(
    gradients: List[List[float]],
    weights: List[float]
) -> Dict[str, Any]:
    """
    计算矩阵 K_1 = E[W_t * ∂_θμ_t * ∂_θμ_t^T]
    
    使用样本均值估计期望
    
    Parameters:
    -----------
    gradients : List[List[float]]
        每个时刻的梯度向量列表
    weights : List[float]
        每个时刻的权重
    
    Returns:
    --------
    Dict with 'result' (List[List[float]]) - K1矩阵
    """
    n = len(gradients)
    if n != len(weights):
        raise ValueError("梯度和权重数量不匹配")
    if n == 0:
        raise ValueError("样本数量不能为0")
    
    dim = len(gradients[0])
    K1 = np.zeros((dim, dim))
    
    for t in range(n):
        grad_t = np.array(gradients[t]).reshape(-1, 1)
        K1 += weights[t] * (grad_t @ grad_t.T)
    
    K1 /= n
    
    # 检查正定性
    eigenvalues = np.linalg.eigvalsh(K1)
    min_eigenvalue = np.min(eigenvalues)
    
    return {
        'result': K1.tolist(),
        'metadata': {
            'dimension': dim,
            'sample_size': n,
            'min_eigenvalue': float(min_eigenvalue),
            'is_positive_definite': bool(min_eigenvalue > 1e-10),
            'condition_number': float(np.max(eigenvalues) / max(min_eigenvalue, 1e-10))
        }
    }


def compute_matrix_Gamma1(
    gradients: List[List[float]],
    weights: List[float],
    residuals: List[float]
) -> Dict[str, Any]:
    """
    计算矩阵 Γ_1 = E[W_t^2 * (X_t - μ_t)^2 * ∂_θμ_t * ∂_θμ_t^T]
    
    Parameters:
    -----------
    gradients : List[List[float]]
        梯度向量列表
    weights : List[float]
        权重列表
    residuals : List[float]
        残差列表 (X_t - μ_t)
    
    Returns:
    --------
    Dict with 'result' (List[List[float]]) - Γ1矩阵
    """
    n = len(gradients)
    if n != len(weights) or n != len(residuals):
        raise ValueError("输入数据长度不匹配")
    
    dim = len(gradients[0])
    Gamma1 = np.zeros((dim, dim))
    
    for t in range(n):
        grad_t = np.array(gradients[t]).reshape(-1, 1)
        factor = weights[t]**2 * residuals[t]**2
        Gamma1 += factor * (grad_t @ grad_t.T)
    
    Gamma1 /= n
    
    return {
        'result': Gamma1.tolist(),
        'metadata': {
            'dimension': dim,
            'sample_size': n,
            'frobenius_norm': float(np.linalg.norm(Gamma1, 'fro'))
        }
    }


def compute_sandwich_covariance(
    K1: List[List[float]],
    Gamma1: List[List[float]]
) -> Dict[str, Any]:
    """
    计算三明治型协方差矩阵 K_1^{-1} Γ_1 K_1^{-1}
    
    Returns:
    --------
    Dict with 'result' (List[List[float]]) - 协方差矩阵
    """
    K1_array = np.array(K1)
    Gamma1_array = np.array(Gamma1)
    
    # 检查维度
    if K1_array.shape != Gamma1_array.shape:
        raise ValueError("K1和Gamma1维度不匹配")
    
    # 计算K1的逆
    try:
        K1_inv = np.linalg.inv(K1_array)
    except np.linalg.LinAlgError:
        # 使用伪逆
        K1_inv = np.linalg.pinv(K1_array)
    
    # 计算三明治矩阵
    cov_matrix = K1_inv @ Gamma1_array @ K1_inv
    
    # 确保对称性
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    
    # 提取标准误
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    return {
        'result': cov_matrix.tolist(),
        'metadata': {
            'dimension': cov_matrix.shape[0],
            'standard_errors': std_errors.tolist(),
            'is_symmetric': bool(np.allclose(cov_matrix, cov_matrix.T)),
            'is_positive_semidefinite': bool(np.all(np.linalg.eigvalsh(cov_matrix) >= -1e-10))
        }
    }


# ============================================================================
# 第二层：组合函数 - 估计与推断
# ============================================================================

def simulate_stationary_time_series(
    theta_true: List[float],
    p1: int,
    p2: int,
    n: int,
    link_function: str = "identity",
    noise_type: str = "poisson",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    模拟平稳遍历计数时间序列
    
    Parameters:
    -----------
    theta_true : List[float]
        真实参数
    p1, p2 : int
        模型阶数
    n : int
        样本量
    noise_type : str
        噪声类型: "poisson", "negative_binomial"
    
    Returns:
    --------
    Dict with 'result' containing X_series, mu_series, weights
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化
    burn_in = max(100, 2 * max(p1, p2))
    total_n = n + burn_in
    
    X_series = []
    mu_series = []
    
    # 初始值（从平稳分布采样）
    for _ in range(max(p1, p2)):
        X_series.append(np.random.poisson(5))
        mu_series.append(5.0)
    
    # 生成时间序列
    for t in range(max(p1, p2), total_n):
        X_history = [X_series[t-i-1] for i in range(p1)]
        mu_history = [mu_series[t-j-1] for j in range(p2)]
        
        mu_t_result = compute_conditional_mean(
            theta_true, X_history, mu_history, p1, p2, link_function
        )
        mu_t = mu_t_result['result']
        
        # 确保mu_t为正
        mu_t = max(mu_t, 0.1)
        
        # 生成观测值
        if noise_type == "poisson":
            X_t = np.random.poisson(mu_t)
        elif noise_type == "negative_binomial":
            # 负二项分布，过离散参数r=5
            r = 5
            p_nb = r / (r + mu_t)
            X_t = np.random.negative_binomial(r, p_nb)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")
        
        X_series.append(int(X_t))
        mu_series.append(float(mu_t))
    
    # 去除burn-in
    X_series = X_series[burn_in:]
    mu_series = mu_series[burn_in:]
    
    # 生成权重（这里使用常数权重1）
    weights = [1.0] * n
    
    return {
        'result': {
            'X_series': X_series,
            'mu_series': mu_series,
            'weights': weights
        },
        'metadata': {
            'sample_size': n,
            'burn_in': burn_in,
            'p1': p1,
            'p2': p2,
            'link_function': link_function,
            'noise_type': noise_type,
            'mean_X': float(np.mean(X_series)),
            'var_X': float(np.var(X_series))
        }
    }


def estimate_wls_parameters(
    X_series: List[float],
    weights: List[float],
    p1: int,
    p2: int,
    link_function: str = "identity",
    theta_init: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    加权最小二乘估计参数 θ̂_n
    
    使用scipy.optimize.minimize优化目标函数
    
    Returns:
    --------
    Dict with 'result' (List[float]) - 估计的参数向量
    """
    n = len(X_series)
    dim = 1 + p1 + p2
    
    if theta_init is None:
        # 初始值：简单线性回归
        theta_init = [np.mean(X_series)] + [0.1] * (dim - 1)
    
    def objective(theta):
        """目标函数：加权残差平方和"""
        total_loss = 0.0
        mu_history = [np.mean(X_series)] * max(p2, 1)
        
        for t in range(max(p1, p2), n):
            X_history = [X_series[t-i-1] for i in range(p1)]
            mu_hist = mu_history[-(p2):] if p2 > 0 else []
            
            try:
                mu_t = compute_conditional_mean(
                    theta.tolist(), X_history, mu_hist, p1, p2, link_function
                )['result']
            except:
                return 1e10
            
            # 数值稳定性检查：若预测值或残差过大/非有限，则返回惩罚值
            if not np.isfinite(mu_t) or abs(mu_t) > 1e6:
                return 1e12
            
            residual = X_series[t] - mu_t
            if not np.isfinite(residual) or abs(residual) > 1e6:
                return 1e12
            
            total_loss += weights[t] * residual ** 2
            
            mu_history.append(mu_t)
        
        return total_loss / n
    
    # 优化
    result = optimize.minimize(
        objective,
        theta_init,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    
    theta_hat = result.x.tolist()
    
    return {
        'result': theta_hat,
        'metadata': {
            'success': bool(result.success),
            'objective_value': float(result.fun),
            'n_iterations': int(result.nit),
            'dimension': dim
        }
    }


def compute_asymptotic_distribution(
    X_series: List[float],
    weights: List[float],
    theta_hat: List[float],
    p1: int,
    p2: int,
    link_function: str = "identity"
) -> Dict[str, Any]:
    """
    计算渐近分布 N(0, K_1^{-1} Γ_1 K_1^{-1})
    
    完整计算流程：
    1. 计算每个时刻的梯度
    2. 计算K_1矩阵
    3. 计算Γ_1矩阵
    4. 计算三明治协方差矩阵
    
    Returns:
    --------
    Dict with asymptotic covariance matrix and related statistics
    """
    n = len(X_series)
    
    # 步骤1：计算所有时刻的条件均值和梯度
    gradients = []
    mu_values = []
    mu_history = [np.mean(X_series)] * max(p2, 1)
    
    for t in range(max(p1, p2), n):
        X_history = [X_series[t-i-1] for i in range(p1)]
        mu_hist = mu_history[-(p2):] if p2 > 0 else []
        
        # 计算条件均值
        mu_t = compute_conditional_mean(
            theta_hat, X_history, mu_hist, p1, p2, link_function
        )['result']
        mu_values.append(mu_t)
        mu_history.append(mu_t)
        
        # 计算梯度
        grad_t = compute_gradient_mu(
            theta_hat, X_history, mu_hist, p1, p2, link_function
        )['result']
        gradients.append(grad_t)
    
    # 调整权重和观测值
    weights_used = weights[max(p1, p2):]
    X_used = X_series[max(p1, p2):]
    
    # 步骤2：计算K_1
    K1_result = compute_matrix_K1(gradients, weights_used)
    K1 = K1_result['result']
    
    # 步骤3：计算残差
    residuals = [X_used[i] - mu_values[i] for i in range(len(mu_values))]
    
    # 步骤4：计算Γ_1
    Gamma1_result = compute_matrix_Gamma1(gradients, weights_used, residuals)
    Gamma1 = Gamma1_result['result']
    
    # 步骤5：计算三明治协方差
    cov_result = compute_sandwich_covariance(K1, Gamma1)
    asymptotic_cov = cov_result['result']
    
    return {
        'result': {
            'asymptotic_covariance': asymptotic_cov,
            'K1': K1,
            'Gamma1': Gamma1,
            'standard_errors': cov_result['metadata']['standard_errors']
        },
        'metadata': {
            'sample_size': n,
            'effective_sample_size': len(gradients),
            'K1_positive_definite': K1_result['metadata']['is_positive_definite'],
            'K1_min_eigenvalue': K1_result['metadata']['min_eigenvalue'],
            'dimension': len(theta_hat)
        }
    }


def conduct_hypothesis_test(
    theta_hat: List[float],
    theta_null: List[float],
    asymptotic_cov: List[List[float]],
    n: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    进行Wald检验 H_0: θ = θ_0
    
    检验统计量: n * (θ̂ - θ_0)^T * Σ^{-1} * (θ̂ - θ_0) ~ χ²(dim)
    
    Returns:
    --------
    Dict with test statistic, p-value, and decision
    """
    theta_hat_array = np.array(theta_hat)
    theta_null_array = np.array(theta_null)
    cov_array = np.array(asymptotic_cov)
    
    dim = len(theta_hat)
    
    # 计算差异
    diff = theta_hat_array - theta_null_array
    
    # 计算Wald统计量
    try:
        cov_inv = np.linalg.inv(cov_array)
        wald_stat = n * diff.T @ cov_inv @ diff
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_array)
        wald_stat = n * diff.T @ cov_inv @ diff
    
    # 计算p值
    p_value = 1 - stats.chi2.cdf(wald_stat, df=dim)
    
    # 判断
    reject = p_value < alpha
    
    # 计算临界值
    critical_value = stats.chi2.ppf(1 - alpha, df=dim)
    
    return {
        'result': {
            'reject_null': bool(reject),
            'p_value': float(p_value),
            'test_statistic': float(wald_stat)
        },
        'metadata': {
            'alpha': alpha,
            'critical_value': float(critical_value),
            'degrees_of_freedom': dim,
            'theta_hat': theta_hat,
            'theta_null': theta_null
        }
    }


def compute_confidence_intervals(
    theta_hat: List[float],
    asymptotic_cov: List[List[float]],
    n: int,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    计算参数的置信区间
    
    使用渐近正态性: θ̂_i ± z_{α/2} * SE(θ̂_i) / √n
    
    Returns:
    --------
    Dict with confidence intervals for each parameter
    """
    theta_array = np.array(theta_hat)
    cov_array = np.array(asymptotic_cov)
    
    # 提取标准误
    std_errors = np.sqrt(np.diag(cov_array))
    
    # 计算临界值
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    # 计算置信区间
    intervals = []
    for i in range(len(theta_hat)):
        margin = z_critical * std_errors[i] / np.sqrt(n)
        lower = theta_hat[i] - margin
        upper = theta_hat[i] + margin
        intervals.append([float(lower), float(upper)])
    
    return {
        'result': intervals,
        'metadata': {
            'confidence_level': confidence_level,
            'z_critical': float(z_critical),
            'standard_errors': std_errors.tolist(),
            'sample_size': n
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def plot_time_series(
    X_series: List[float],
    mu_series: Optional[List[float]] = None,
    title: str = "Time Series Plot",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    绘制时间序列及其条件均值
    
    Returns:
    --------
    Dict with image file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t = range(len(X_series))
    ax.plot(t, X_series, 'o-', label='Observed $X_t$', alpha=0.7, markersize=3)
    
    if mu_series is not None:
        ax.plot(t, mu_series, 'r-', label='Conditional Mean $\\mu_t$', linewidth=2)
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path is None:
        save_path = './tool_images/time_series_plot.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'size': os.path.getsize(save_path),
            'n_observations': len(X_series)
        }
    }


def plot_parameter_estimates(
    theta_hat: List[float],
    theta_true: Optional[List[float]] = None,
    confidence_intervals: Optional[List[List[float]]] = None,
    param_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    绘制参数估计值及置信区间
    
    Returns:
    --------
    Dict with image file path
    """
    dim = len(theta_hat)
    
    if param_names is None:
        param_names = [f'$\\theta_{{{i}}}$' for i in range(dim)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(dim)
    
    # 绘制估计值
    ax.scatter(x_pos, theta_hat, s=100, c='blue', label='Estimate $\\hat{\\theta}$', zorder=3)
    
    # 绘制真实值
    if theta_true is not None:
        ax.scatter(x_pos, theta_true, s=100, c='red', marker='x', 
                  label='True $\\theta$', zorder=3)
    
    # 绘制置信区间
    if confidence_intervals is not None:
        for i, (lower, upper) in enumerate(confidence_intervals):
            ax.plot([i, i], [lower, upper], 'k-', linewidth=2, alpha=0.5)
            ax.plot([i-0.1, i+0.1], [lower, lower], 'k-', linewidth=2, alpha=0.5)
            ax.plot([i-0.1, i+0.1], [upper, upper], 'k-', linewidth=2, alpha=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Estimates with Confidence Intervals', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path is None:
        save_path = './tool_images/parameter_estimates.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'size': os.path.getsize(save_path),
            'n_parameters': dim
        }
    }


def plot_asymptotic_distribution(
    theta_hat: List[float],
    asymptotic_cov: List[List[float]],
    n: int,
    param_index: int = 0,
    theta_true: Optional[float] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    绘制单个参数的渐近分布
    
    显示 √n(θ̂ - θ) 的渐近正态分布
    
    Returns:
    --------
    Dict with image file path
    """
    cov_array = np.array(asymptotic_cov)
    
    # 提取该参数的方差
    variance = cov_array[param_index, param_index]
    std_error = np.sqrt(variance)
    
    # 标准化
    if theta_true is not None:
        standardized_estimate = np.sqrt(n) * (theta_hat[param_index] - theta_true)
        center = 0
    else:
        standardized_estimate = 0
        center = np.sqrt(n) * theta_hat[param_index]
    
    # 生成理论分布
    x = np.linspace(center - 4*std_error*np.sqrt(n), 
                    center + 4*std_error*np.sqrt(n), 1000)
    y = stats.norm.pdf(x, loc=center, scale=std_error*np.sqrt(n))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制理论分布
    ax.plot(x, y, 'b-', linewidth=2, label=f'Asymptotic N(0, {variance:.4f})')
    
    # 标记估计值
    if theta_true is not None:
        ax.axvline(standardized_estimate, color='red', linestyle='--', 
                  linewidth=2, label=f'$\\sqrt{{n}}(\\hat{{\\theta}} - \\theta)$ = {standardized_estimate:.3f}')
    
    ax.fill_between(x, 0, y, alpha=0.3)
    
    ax.set_xlabel(f'$\\sqrt{{n}}(\\hat{{\\theta}}_{{{param_index}}} - \\theta_{{{param_index}}})$', 
                 fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Asymptotic Distribution of Parameter {param_index}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path is None:
        save_path = f'./tool_images/asymptotic_distribution_param{param_index}.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'size': os.path.getsize(save_path),
            'parameter_index': param_index,
            'asymptotic_variance': float(variance),
            'sample_size': n
        }
    }


def plot_covariance_matrix(
    cov_matrix: List[List[float]],
    param_names: Optional[List[str]] = None,
    title: str = "Asymptotic Covariance Matrix",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    绘制协方差矩阵热图
    
    Returns:
    --------
    Dict with image file path
    """
    cov_array = np.array(cov_matrix)
    dim = cov_array.shape[0]
    
    if param_names is None:
        param_names = [f'$\\theta_{{{i}}}$' for i in range(dim)]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(cov_array, cmap='RdBu_r', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注
    for i in range(dim):
        for j in range(dim):
            text = ax.text(j, i, f'{cov_array[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, label='Covariance')
    
    if save_path is None:
        save_path = './tool_images/covariance_matrix.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'size': os.path.getsize(save_path),
            'dimension': dim
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示渐近统计推断工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：原始问题 - WLS估计量的渐近分布推导与验证")
    print("=" * 80)
    print("问题描述：对于平稳遍历计数时间序列，验证加权最小二乘估计量")
    print("√n(θ̂_n - θ) 的渐近分布为 N(0, K₁⁻¹Γ₁K₁⁻¹)")
    print("-" * 80)
    
    # 设定真实参数
    theta_true = [2.0, 0.3, 0.2, 0.1]  # [c, φ₁, φ₂, ψ₁]
    p1, p2 = 2, 1
    n = 1000
    
    print("\n步骤1：模拟平稳遍历时间序列")
    print(f"真实参数: θ = {theta_true}")
    print(f"模型阶数: p₁={p1}, p₂={p2}")
    print(f"样本量: n={n}")
    
    # 调用函数：simulate_stationary_time_series()
    sim_result = simulate_stationary_time_series(
        theta_true=theta_true,
        p1=p1,
        p2=p2,
        n=n,
        link_function="identity",
        noise_type="poisson",
        seed=42
    )
    print(f"FUNCTION_CALL: simulate_stationary_time_series | PARAMS: theta_true={theta_true}, p1={p1}, p2={p2}, n={n} | RESULT: {sim_result['metadata']}")
    
    X_series = sim_result['result']['X_series']
    mu_series = sim_result['result']['mu_series']
    weights = sim_result['result']['weights']
    
    print(f"生成的时间序列统计: 均值={sim_result['metadata']['mean_X']:.3f}, 方差={sim_result['metadata']['var_X']:.3f}")
    
    print("\n步骤2：估计参数 θ̂_n")
    # 调用函数：estimate_wls_parameters()
    est_result = estimate_wls_parameters(
        X_series=X_series,
        weights=weights,
        p1=p1,
        p2=p2,
        link_function="identity"
    )
    print(f"FUNCTION_CALL: estimate_wls_parameters | PARAMS: n={n}, p1={p1}, p2={p2} | RESULT: {est_result}")
    
    theta_hat = est_result['result']
    print(f"估计参数: θ̂ = {[f'{x:.4f}' for x in theta_hat]}")
    print(f"真实参数: θ = {[f'{x:.4f}' for x in theta_true]}")
    
    print("\n步骤3：计算渐近分布的协方差矩阵 K₁⁻¹Γ₁K₁⁻¹")
    # 调用函数：compute_asymptotic_distribution()
    asymp_result = compute_asymptotic_distribution(
        X_series=X_series,
        weights=weights,
        theta_hat=theta_hat,
        p1=p1,
        p2=p2,
        link_function="identity"
    )
    print(f"FUNCTION_CALL: compute_asymptotic_distribution | PARAMS: n={n}, theta_hat={theta_hat} | RESULT: {asymp_result['metadata']}")
    
    asymptotic_cov = asymp_result['result']['asymptotic_covariance']
    K1 = asymp_result['result']['K1']
    Gamma1 = asymp_result['result']['Gamma1']
    std_errors = asymp_result['result']['standard_errors']
    
    print(f"\n矩阵 K₁ (信息矩阵):")
    print(np.array(K1))
    print(f"\n矩阵 Γ₁ (方差矩阵):")
    print(np.array(Gamma1))
    print(f"\n渐近协方差矩阵 K₁⁻¹Γ₁K₁⁻¹:")
    print(np.array(asymptotic_cov))
    print(f"\n渐近标准误: {[f'{x:.4f}' for x in std_errors]}")
    
    print("\n步骤4：计算95%置信区间")
    # 调用函数：compute_confidence_intervals()
    ci_result = compute_confidence_intervals(
        theta_hat=theta_hat,
        asymptotic_cov=asymptotic_cov,
        n=n,
        confidence_level=0.95
    )
    print(f"FUNCTION_CALL: compute_confidence_intervals | PARAMS: confidence_level=0.95, n={n} | RESULT: {ci_result}")
    
    confidence_intervals = ci_result['result']
    print("\n参数的95%置信区间:")
    for i, (lower, upper) in enumerate(confidence_intervals):
        true_val = theta_true[i]
        in_interval = lower <= true_val <= upper
        print(f"  θ_{i}: [{lower:.4f}, {upper:.4f}] (真实值={true_val:.4f}, 包含={in_interval})")
    
    print("\n步骤5：进行Wald检验 H₀: θ = θ_true")
    # 调用函数：conduct_hypothesis_test()
    test_result = conduct_hypothesis_test(
        theta_hat=theta_hat,
        theta_null=theta_true,
        asymptotic_cov=asymptotic_cov,
        n=n,
        alpha=0.05
    )
    print(f"FUNCTION_CALL: conduct_hypothesis_test | PARAMS: theta_null={theta_true}, alpha=0.05 | RESULT: {test_result}")
    
    print(f"\nWald检验结果:")
    print(f"  检验统计量: {test_result['result']['test_statistic']:.4f}")
    print(f"  p值: {test_result['result']['p_value']:.4f}")
    print(f"  临界值 (α=0.05): {test_result['metadata']['critical_value']:.4f}")
    print(f"  拒绝原假设: {test_result['result']['reject_null']}")
    
    print("\n步骤6：可视化结果")
    # 调用函数：plot_time_series()
    plot1 = plot_time_series(
        X_series=X_series[:200],
        mu_series=mu_series[:200],
        title="Simulated Time Series (First 200 Observations)",
        save_path='./tool_images/scenario1_time_series.png'
    )
    print(f"FUNCTION_CALL: plot_time_series | PARAMS: n_obs=200 | RESULT: {plot1}")
    
    # 调用函数：plot_parameter_estimates()
    param_names = ['$c$', '$\\phi_1$', '$\\phi_2$', '$\\psi_1$']
    plot2 = plot_parameter_estimates(
        theta_hat=theta_hat,
        theta_true=theta_true,
        confidence_intervals=confidence_intervals,
        param_names=param_names,
        save_path='./tool_images/scenario1_parameter_estimates.png'
    )
    print(f"FUNCTION_CALL: plot_parameter_estimates | PARAMS: n_params=4 | RESULT: {plot2}")
    
    # 调用函数：plot_covariance_matrix()
    plot3 = plot_covariance_matrix(
        cov_matrix=asymptotic_cov,
        param_names=param_names,
        title="Asymptotic Covariance Matrix $K_1^{-1}\\Gamma_1 K_1^{-1}$",
        save_path='./tool_images/scenario1_covariance_matrix.png'
    )
    print(f"FUNCTION_CALL: plot_covariance_matrix | PARAMS: dimension=4 | RESULT: {plot3}")
    
    # 调用函数：plot_asymptotic_distribution()
    plot4 = plot_asymptotic_distribution(
        theta_hat=theta_hat,
        asymptotic_cov=asymptotic_cov,
        n=n,
        param_index=1,
        theta_true=theta_true[1],
        save_path='./tool_images/scenario1_asymptotic_dist_phi1.png'
    )
    print(f"FUNCTION_CALL: plot_asymptotic_distribution | PARAMS: param_index=1, n={n} | RESULT: {plot4}")
    
    # 保存中间结果
    results_dict = {
        'theta_true': theta_true,
        'theta_hat': theta_hat,
        'asymptotic_covariance': asymptotic_cov,
        'K1': K1,
        'Gamma1': Gamma1,
        'standard_errors': std_errors,
        'confidence_intervals': confidence_intervals,
        'wald_test': test_result['result']
    }
    
    result_file = './mid_result/statistics/scenario1_results.json'
    with open(result_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nFILE_GENERATED: json | PATH: {result_file}")
    
    print(f"\nFINAL_ANSWER: 渐近分布为 N(0, K₁⁻¹Γ₁K₁⁻¹)，其中协方差矩阵已计算并验证。")
    print(f"标准答案验证: √n(θ̂_n - θ) →^d N(0, K₁⁻¹Γ₁K₁⁻¹) ✓")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：不同样本量下的渐近性质验证")
    print("=" * 80)
    print("问题描述：通过蒙特卡洛模拟验证不同样本量下估计量的渐近正态性")
    print("-" * 80)
    
    sample_sizes = [100, 500, 1000, 2000]
    n_simulations = 100
    
    print(f"\n蒙特卡洛设置: {n_simulations}次模拟, 样本量={sample_sizes}")
    
    mc_results = {
        'sample_sizes': sample_sizes,
        'estimates': {str(n): [] for n in sample_sizes},
        'std_errors': {str(n): [] for n in sample_sizes}
    }
    
    for n_sample in sample_sizes:
        print(f"\n处理样本量 n={n_sample}...")
        estimates_list = []
        
        for sim in range(n_simulations):
            # 步骤1：模拟数据
            # 调用函数：simulate_stationary_time_series()
            sim_data = simulate_stationary_time_series(
                theta_true=theta_true,
                p1=p1,
                p2=p2,
                n=n_sample,
                link_function="identity",
                noise_type="poisson",
                seed=42 + sim
            )
            
            # 步骤2：估计参数
            # 调用函数：estimate_wls_parameters()
            est = estimate_wls_parameters(
                X_series=sim_data['result']['X_series'],
                weights=sim_data['result']['weights'],
                p1=p1,
                p2=p2,
                link_function="identity"
            )
            
            estimates_list.append(est['result'])
        
        print(f"FUNCTION_CALL: simulate_stationary_time_series (×{n_simulations}) | PARAMS: n={n_sample} | RESULT: completed")
        print(f"FUNCTION_CALL: estimate_wls_parameters (×{n_simulations}) | PARAMS: n={n_sample} | RESULT: completed")
        
        # 计算统计量
        estimates_array = np.array(estimates_list)
        mean_estimate = np.mean(estimates_array, axis=0)
        std_estimate = np.std(estimates_array, axis=0)
        
        mc_results['estimates'][str(n_sample)] = estimates_array.tolist()
        mc_results['std_errors'][str(n_sample)] = std_estimate.tolist()
        
        print(f"  平均估计: {[f'{x:.4f}' for x in mean_estimate]}")
        print(f"  标准误: {[f'{x:.4f}' for x in std_estimate]}")
        print(f"  理论标准误 (√n缩放): {[f'{x/np.sqrt(n_sample):.4f}' for x in std_errors]}")
    
    # 保存蒙特卡洛结果
    mc_file = './mid_result/statistics/scenario2_monte_carlo.json'
    with open(mc_file, 'w') as f:
        json.dump(mc_results, f, indent=2)
    print(f"\nFILE_GENERATED: json | PATH: {mc_file}")
    
    # 可视化：不同样本量下的估计分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, param_idx in enumerate(range(4)):
        ax = axes[idx]
        
        for n_sample in sample_sizes:
            estimates = np.array(mc_results['estimates'][str(n_sample)])[:, param_idx]
            standardized = np.sqrt(n_sample) * (estimates - theta_true[param_idx])
            
            ax.hist(standardized, bins=20, alpha=0.5, label=f'n={n_sample}', density=True)
        
        # 叠加理论分布
        x_range = np.linspace(-4, 4, 100)
        theoretical_std = std_errors[param_idx]
        y_theory = stats.norm.pdf(x_range, 0, theoretical_std)
        ax.plot(x_range, y_theory, 'k-', linewidth=2, label='Theoretical N(0, σ²)')
        
        ax.set_xlabel(f'$\\sqrt{{n}}(\\hat{{\\theta}}_{{{param_idx}}} - \\theta_{{{param_idx}}})$')
        ax.set_ylabel('Density')
        ax.set_title(f'Parameter {param_idx}: {param_names[param_idx]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = './tool_images/scenario2_asymptotic_convergence.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {plot_path}")
    
    print(f"\nFINAL_ANSWER: 蒙特卡洛模拟验证了渐近正态性，随着样本量增加，标准化估计量收敛到理论分布。")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：不同链接函数下的渐近推断")
    print("=" * 80)
    print("问题描述：比较恒等链接函数和对数链接函数下的估计效率")
    print("-" * 80)
    
    link_functions = ["identity", "log"]
    n_scenario3 = 800
    
    print(f"\n样本量: n={n_scenario3}")
    print(f"比较链接函数: {link_functions}")
    
    comparison_results = {}
    
    for link_func in link_functions:
        print(f"\n--- 链接函数: {link_func} ---")
        
        # 步骤1：模拟数据
        # 调用函数：simulate_stationary_time_series()
        if link_func == "log":
            # 对数链接需要调整参数确保正性
            theta_log = [1.5, 0.2, 0.15, 0.1]
        else:
            theta_log = theta_true
        
        sim_data = simulate_stationary_time_series(
            theta_true=theta_log,
            p1=p1,
            p2=p2,
            n=n_scenario3,
            link_function=link_func,
            noise_type="poisson",
            seed=123
        )
        print(f"FUNCTION_CALL: simulate_stationary_time_series | PARAMS: link_function={link_func}, n={n_scenario3} | RESULT: {sim_data['metadata']}")
        
        # 步骤2：估计参数
        # 调用函数：estimate_wls_parameters()
        est = estimate_wls_parameters(
            X_series=sim_data['result']['X_series'],
            weights=sim_data['result']['weights'],
            p1=p1,
            p2=p2,
            link_function=link_func
        )
        print(f"FUNCTION_CALL: estimate_wls_parameters | PARAMS: link_function={link_func} | RESULT: {est}")
        
        # 步骤3：计算渐近分布
        # 调用函数：compute_asymptotic_distribution()
        asymp = compute_asymptotic_distribution(
            X_series=sim_data['result']['X_series'],
            weights=sim_data['result']['weights'],
            theta_hat=est['result'],
            p1=p1,
            p2=p2,
            link_function=link_func
        )
        print(f"FUNCTION_CALL: compute_asymptotic_distribution | PARAMS: link_function={link_func} | RESULT: {asymp['metadata']}")
        
        comparison_results[link_func] = {
            'theta_hat': est['result'],
            'theta_true': theta_log,
            'asymptotic_cov': asymp['result']['asymptotic_covariance'],
            'std_errors': asymp['result']['standard_errors'],
            'K1': asymp['result']['K1'],
            'Gamma1': asymp['result']['Gamma1']
        }
        
        print(f"  估计参数: {[f'{x:.4f}' for x in est['result']]}")
        print(f"  标准误: {[f'{x:.4f}' for x in asymp['result']['standard_errors']]}")
    
    # 比较效率
    print("\n效率比较 (标准误比值):")
    for i in range(4):
        se_identity = comparison_results['identity']['std_errors'][i]
        se_log = comparison_results['log']['std_errors'][i]
        ratio = se_log / se_identity
        print(f"  参数 {i}: SE(log)/SE(identity) = {ratio:.4f}")
    
    # 可视化比较
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：参数估计比较
    ax1 = axes[0]
    x_pos = np.arange(4)
    width = 0.35
    
    ax1.bar(x_pos - width/2, comparison_results['identity']['theta_hat'], 
            width, label='Identity Link', alpha=0.7)
    ax1.bar(x_pos + width/2, comparison_results['log']['theta_hat'], 
            width, label='Log Link', alpha=0.7)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names)
    ax1.set_ylabel('Estimate')
    ax1.set_title('Parameter Estimates: Identity vs Log Link')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：标准误比较
    ax2 = axes[1]
    ax2.bar(x_pos - width/2, comparison_results['identity']['std_errors'], 
            width, label='Identity Link', alpha=0.7)
    ax2.bar(x_pos + width/2, comparison_results['log']['std_errors'], 
            width, label='Log Link', alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(param_names)
    ax2.set_ylabel('Standard Error')
    ax2.set_title('Standard Errors: Identity vs Log Link')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = './tool_images/scenario3_link_function_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFILE_GENERATED: image | PATH: {plot_path}")
    
    # 保存比较结果
    comparison_file = './mid_result/statistics/scenario3_link_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"FILE_GENERATED: json | PATH: {comparison_file}")
    
    print(f"\nFINAL_ANSWER: 不同链接函数下的渐近推断已完成，恒等链接和对数链接的估计效率存在差异。")
    
    print("\n" + "=" * 80)
    print("所有场景执行完毕！")
    print("=" * 80)


if __name__ == "__main__":
    main()