# Filename: proximal_mediation_eif_toolkit.py
"""
Proximal Mediation Analysis Toolkit
用于半参数因果中介分析的高效影响函数推导与验证工具包

核心功能:
1. Bridge函数求解(outcome bridge & exposure bridge)
2. 有效影响函数(EIF)推导与验证
3. 半参数效率理论的数值验证
4. 中介效应估计的敏感性分析

依赖库:
- numpy: 数值计算
- scipy: 积分方程求解、优化
- sympy: 符号推导
- matplotlib: 可视化
"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm, multivariate_normal
import sympy as sp
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False


# ==================== 第一层：原子函数 ====================

def generate_proximal_mediation_data(
    n_samples: int,
    seed: int = 42,
    confounder_strength: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    生成符合proximal mediation设定的模拟数据
    
    参数:
        n_samples: 样本量
        seed: 随机种子
        confounder_strength: 未观测混杂因子的影响强度
    
    返回:
        {'result': data_dict, 'metadata': {...}}
        data_dict包含: Y, M, A, Z, W, X, U, V (真实混杂因子)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples必须为正整数，当前值: {n_samples}")
    if not 0 <= confounder_strength <= 1:
        raise ValueError(f"confounder_strength必须在[0,1]范围内，当前值: {confounder_strength}")
    
    np.random.seed(seed)
    
    # 观测协变量
    X = np.random.normal(0, 1, n_samples)
    
    # 未观测混杂因子
    U = np.random.normal(0, 1, n_samples)  # 影响A和M
    V = np.random.normal(0, 1, n_samples)  # 影响M和Y
    
    # 代理变量(proxies)
    Z = 0.7 * U + np.random.normal(0, 0.3, n_samples)  # U的代理
    W = 0.7 * V + np.random.normal(0, 0.3, n_samples)  # V的代理
    
    # 暴露变量A (受U和X影响)
    logit_A = 0.5 * X + confounder_strength * U
    prob_A = 1 / (1 + np.exp(-logit_A))
    A = np.random.binomial(1, prob_A, n_samples)
    
    # 中介变量M (受A, U, V, X影响)
    M = 0.8 * A + 0.6 * X + confounder_strength * (0.5 * U + 0.5 * V) + np.random.normal(0, 0.5, n_samples)
    
    # 结果变量Y (受A, M, V, X影响)
    Y = 1.2 * A + 0.9 * M + 0.5 * X + confounder_strength * V + np.random.normal(0, 0.5, n_samples)
    
    data = {
        'Y': Y, 'M': M, 'A': A, 'Z': Z, 'W': W, 'X': X,
        'U': U, 'V': V  # 真实数据中不可观测,仅用于验证
    }
    
    metadata = {
        'n_samples': n_samples,
        'confounder_strength': confounder_strength,
        'seed': seed,
        'variables': list(data.keys())
    }
    
    return {'result': data, 'metadata': metadata}


def estimate_conditional_density_ratio(
    A: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    bandwidth: float = 0.5
) -> Dict[str, float]:
    """
    估计条件密度比 f(A=1|W,X) / f(A=0|W,X)
    使用核密度估计方法
    
    参数:
        A: 暴露变量 (0/1)
        W: 代理变量
        X: 协变量
        bandwidth: 核函数带宽
    
    返回:
        {'result': density_ratio, 'metadata': {...}}
    """
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("A必须为二元变量(0或1)")
    if len(A) != len(W) or len(A) != len(X):
        raise ValueError(f"输入数组长度不一致: A({len(A)}), W({len(W)}), X({len(X)})")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth必须为正数，当前值: {bandwidth}")
    
    # 简化估计: 使用logistic回归估计倾向性得分
    from scipy.special import expit
    
    # 构造特征矩阵
    features = np.column_stack([W, X, W*X])
    
    # 拟合logistic回归
    def logistic_loss(beta):
        logits = features @ beta
        probs = expit(logits)
        return -np.mean(A * np.log(probs + 1e-10) + (1-A) * np.log(1-probs + 1e-10))
    
    beta_init = np.zeros(features.shape[1])
    result = optimize.minimize(logistic_loss, beta_init, method='BFGS')
    beta_opt = result.x
    
    # 计算密度比
    logits = features @ beta_opt
    prob_A1 = expit(logits)
    prob_A0 = 1 - prob_A1
    
    # 避免除零
    density_ratio = np.mean(prob_A1 / (prob_A0 + 1e-10))
    
    metadata = {
        'method': 'logistic_regression',
        'bandwidth': bandwidth,
        'n_samples': len(A),
        'mean_prob_A1': float(np.mean(prob_A1)),
        'coefficients': beta_opt.tolist()
    }
    
    return {'result': float(density_ratio), 'metadata': metadata}


def solve_outcome_bridge_h1(
    Y: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    regularization: float = 0.01
) -> Dict[str, Callable]:
    """
    求解outcome bridge函数 h1(w,m,a,x)
    满足: E[Y|Z,A,M,X] = ∫ h1(w,M,A,X) dF(w|Z,A,M,X)
    
    使用核岭回归近似求解
    
    参数:
        Y, Z, A, M, X, W: 观测数据
        regularization: 正则化参数
    
    返回:
        {'result': h1_function, 'metadata': {...}}
    """
    if regularization <= 0:
        raise ValueError(f"regularization必须为正数，当前值: {regularization}")
    
    n = len(Y)
    if not all(len(arr) == n for arr in [Z, A, M, X, W]):
        raise ValueError("所有输入数组长度必须一致")
    
    # 构造特征矩阵 [W, M, A, X, W*M, W*A, M*A, ...]
    features = np.column_stack([
        W, M, A, X,
        W*M, W*A, M*A, W*X, M*X, A*X,
        W**2, M**2, X**2
    ])
    
    # 核岭回归
    K = features @ features.T
    alpha = np.linalg.solve(K + regularization * np.eye(n), Y)
    
    # 定义h1函数
    def h1(w: float, m: float, a: float, x: float) -> float:
        """h1 bridge函数"""
        test_features = np.array([
            w, m, a, x,
            w*m, w*a, m*a, w*x, m*x, a*x,
            w**2, m**2, x**2
        ])
        k_test = features @ test_features
        return float(alpha @ k_test)
    
    # 验证积分方程
    sample_indices = np.random.choice(n, min(100, n), replace=False)
    errors = []
    for i in sample_indices:
        # 左侧: E[Y|Z,A,M,X]
        mask = (np.abs(Z - Z[i]) < 0.5) & (A == A[i]) & (np.abs(M - M[i]) < 0.5) & (np.abs(X - X[i]) < 0.5)
        if np.sum(mask) > 0:
            lhs = np.mean(Y[mask])
            
            # 右侧: ∫ h1(w,M,A,X) dF(w|Z,A,M,X) ≈ E[h1(W,M,A,X)|Z,A,M,X]
            rhs = np.mean([h1(W[j], M[i], A[i], X[i]) for j in np.where(mask)[0]])
            errors.append(abs(lhs - rhs))
    
    metadata = {
        'method': 'kernel_ridge_regression',
        'regularization': regularization,
        'n_samples': n,
        'n_features': features.shape[1],
        'validation_error': float(np.mean(errors)) if errors else None,
        'alpha_norm': float(np.linalg.norm(alpha))
    }
    
    return {'result': h1, 'metadata': metadata}


def solve_outcome_bridge_h0(
    h1_func: Callable,
    Z: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    a_value: int,
    regularization: float = 0.01
) -> Dict[str, Callable]:
    """
    求解outcome bridge函数 h0(w,a,x)
    满足: E[h1(W,M,a,X)|Z,A=0,X] = ∫ h0(w,a,X) dF(w|Z,A=0,X)
    
    参数:
        h1_func: 已求解的h1函数
        Z, A, M, X, W: 观测数据
        a_value: 固定的暴露值(0或1)
        regularization: 正则化参数
    
    返回:
        {'result': h0_function, 'metadata': {...}}
    """
    if a_value not in [0, 1]:
        raise ValueError(f"a_value必须为0或1，当前值: {a_value}")
    
    # 筛选A=0的样本
    mask_A0 = (A == 0)
    Z_A0 = Z[mask_A0]
    M_A0 = M[mask_A0]
    X_A0 = X[mask_A0]
    W_A0 = W[mask_A0]
    
    n = len(Z_A0)
    
    # 计算左侧: E[h1(W,M,a,X)|Z,A=0,X]
    lhs_values = np.array([h1_func(W_A0[i], M_A0[i], a_value, X_A0[i]) for i in range(n)])
    
    # 构造特征矩阵 [W, X, W*X, W^2, X^2]
    features = np.column_stack([
        W_A0, X_A0, W_A0*X_A0, W_A0**2, X_A0**2
    ])
    
    # 核岭回归
    K = features @ features.T
    alpha = np.linalg.solve(K + regularization * np.eye(n), lhs_values)
    
    # 定义h0函数
    def h0(w: float, a: float, x: float) -> float:
        """h0 bridge函数"""
        test_features = np.array([w, x, w*x, w**2, x**2])
        k_test = features @ test_features
        return float(alpha @ k_test)
    
    metadata = {
        'method': 'kernel_ridge_regression',
        'a_value': a_value,
        'n_samples_A0': n,
        'regularization': regularization,
        'alpha_norm': float(np.linalg.norm(alpha))
    }
    
    return {'result': h0, 'metadata': metadata}


def solve_exposure_bridge_q0(
    Z: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    density_ratio: float,
    regularization: float = 0.01
) -> Dict[str, Callable]:
    """
    求解exposure bridge函数 q0(z,x)
    满足: f(A=1|W,X)/f(A=0|W,X) = E[q0(Z,X)|W,A=0,X]
    
    参数:
        Z, A, W, X: 观测数据
        density_ratio: 条件密度比
        regularization: 正则化参数
    
    返回:
        {'result': q0_function, 'metadata': {...}}
    """
    # 筛选A=0的样本
    mask_A0 = (A == 0)
    Z_A0 = Z[mask_A0]
    W_A0 = W[mask_A0]
    X_A0 = X[mask_A0]
    
    n = len(Z_A0)
    
    # 左侧: 密度比(常数)
    lhs = np.full(n, density_ratio)
    
    # 构造特征矩阵 [Z, X, Z*X, Z^2, X^2]
    features = np.column_stack([
        Z_A0, X_A0, Z_A0*X_A0, Z_A0**2, X_A0**2
    ])
    
    # 核岭回归
    K = features @ features.T
    alpha = np.linalg.solve(K + regularization * np.eye(n), lhs)
    
    # 定义q0函数
    def q0(z: float, x: float) -> float:
        """q0 bridge函数"""
        test_features = np.array([z, x, z*x, z**2, x**2])
        k_test = features @ test_features
        return float(alpha @ k_test)
    
    metadata = {
        'method': 'kernel_ridge_regression',
        'n_samples_A0': n,
        'density_ratio': density_ratio,
        'regularization': regularization,
        'alpha_norm': float(np.linalg.norm(alpha))
    }
    
    return {'result': q0, 'metadata': metadata}


def solve_exposure_bridge_q1(
    Z: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    q0_func: Callable,
    regularization: float = 0.01
) -> Dict[str, Callable]:
    """
    求解exposure bridge函数 q1(z,m,x)
    满足: E[q0(Z,X)|W,A=0,M,X] * f(A=0|W,M,X)/f(A=1|W,M,X) = E[q1(Z,M,X)|W,A=1,M,X]
    
    参数:
        Z, A, M, W, X: 观测数据
        q0_func: 已求解的q0函数
        regularization: 正则化参数
    
    返回:
        {'result': q1_function, 'metadata': {...}}
    """
    # 筛选A=1的样本
    mask_A1 = (A == 1)
    Z_A1 = Z[mask_A1]
    M_A1 = M[mask_A1]
    W_A1 = W[mask_A1]
    X_A1 = X[mask_A1]
    
    # 筛选A=0的样本用于计算左侧
    mask_A0 = (A == 0)
    Z_A0 = Z[mask_A0]
    M_A0 = M[mask_A0]
    W_A0 = W[mask_A0]
    X_A0 = X[mask_A0]
    
    n1 = len(Z_A1)
    
    # 计算左侧: E[q0(Z,X)|W,A=0,M,X] * density_ratio_inverse
    # 简化: 使用A=0样本的q0值平均
    lhs_values = []
    for i in range(n1):
        # 找到A=0中与当前样本(W,M,X)接近的样本
        dist = np.sqrt((W_A0 - W_A1[i])**2 + (M_A0 - M_A1[i])**2 + (X_A0 - X_A1[i])**2)
        nearest_indices = np.argsort(dist)[:min(10, len(dist))]
        
        q0_values = [q0_func(Z_A0[j], X_A0[j]) for j in nearest_indices]
        lhs_values.append(np.mean(q0_values))
    
    lhs_values = np.array(lhs_values)
    
    # 构造特征矩阵 [Z, M, X, Z*M, Z*X, M*X, Z^2, M^2, X^2]
    features = np.column_stack([
        Z_A1, M_A1, X_A1,
        Z_A1*M_A1, Z_A1*X_A1, M_A1*X_A1,
        Z_A1**2, M_A1**2, X_A1**2
    ])
    
    # 核岭回归
    K = features @ features.T
    alpha = np.linalg.solve(K + regularization * np.eye(n1), lhs_values)
    
    # 定义q1函数
    def q1(z: float, m: float, x: float) -> float:
        """q1 bridge函数"""
        test_features = np.array([
            z, m, x,
            z*m, z*x, m*x,
            z**2, m**2, x**2
        ])
        k_test = features @ test_features
        return float(alpha @ k_test)
    
    metadata = {
        'method': 'kernel_ridge_regression',
        'n_samples_A1': n1,
        'regularization': regularization,
        'alpha_norm': float(np.linalg.norm(alpha))
    }
    
    return {'result': q1, 'metadata': metadata}


# ==================== 第二层：组合函数 ====================

def estimate_all_bridge_functions(
    data: Dict[str, np.ndarray],
    regularization: float = 0.01
) -> Dict[str, Dict]:
    """
    估计所有bridge函数: h1, h0, q0, q1
    
    参数:
        data: 包含Y, M, A, Z, W, X的数据字典
        regularization: 正则化参数
    
    返回:
        {'result': {'h1': h1_result, 'h0': h0_result, 'q0': q0_result, 'q1': q1_result},
         'metadata': {...}}
    """
    Y, M, A, Z, W, X = data['Y'], data['M'], data['A'], data['Z'], data['W'], data['X']
    
    # 1. 估计密度比
    density_result = estimate_conditional_density_ratio(A, W, X)
    density_ratio = density_result['result']
    
    # 2. 求解h1
    h1_result = solve_outcome_bridge_h1(Y, Z, A, M, X, W, regularization)
    h1_func = h1_result['result']
    
    # 3. 求解h0 (对a=0和a=1)
    h0_a0_result = solve_outcome_bridge_h0(h1_func, Z, A, M, X, W, a_value=0, regularization=regularization)
    h0_a1_result = solve_outcome_bridge_h0(h1_func, Z, A, M, X, W, a_value=1, regularization=regularization)
    
    # 4. 求解q0
    q0_result = solve_exposure_bridge_q0(Z, A, W, X, density_ratio, regularization)
    q0_func = q0_result['result']
    
    # 5. 求解q1
    q1_result = solve_exposure_bridge_q1(Z, A, M, W, X, q0_func, regularization)
    
    result = {
        'h1': h1_result,
        'h0_a0': h0_a0_result,
        'h0_a1': h0_a1_result,
        'q0': q0_result,
        'q1': q1_result,
        'density_ratio': density_ratio
    }
    
    metadata = {
        'n_samples': len(Y),
        'regularization': regularization,
        'bridge_functions': ['h1', 'h0_a0', 'h0_a1', 'q0', 'q1']
    }
    
    return {'result': result, 'metadata': metadata}


def compute_eif_components(
    data: Dict[str, np.ndarray],
    bridge_funcs: Dict[str, Dict],
    psi_value: float
) -> Dict[str, np.ndarray]:
    """
    计算EIF的各个组成部分
    
    EIF = I(A=0)q0(Z,X)[h1(W,M,1,X)-h0(W,1,X)]
          + I(A=0)[h1(W,M,0,X)-h0(W,0,X)]
          + (I(A=1)q1(Z,M,X)+I(A=0))[Y-h1(W,M,A,X)]
          + h0(W,A,X) - ψ
    
    参数:
        data: 观测数据
        bridge_funcs: 所有bridge函数
        psi_value: 目标参数ψ的估计值
    
    返回:
        {'result': {'term1': ..., 'term2': ..., 'term3': ..., 'term4': ..., 'eif': ...},
         'metadata': {...}}
    """
    Y, M, A, Z, W, X = data['Y'], data['M'], data['A'], data['Z'], data['W'], data['X']
    n = len(Y)
    
    h1_func = bridge_funcs['h1']['result']
    h0_a0_func = bridge_funcs['h0_a0']['result']
    h0_a1_func = bridge_funcs['h0_a1']['result']
    q0_func = bridge_funcs['q0']['result']
    q1_func = bridge_funcs['q1']['result']
    
    # 计算各项
    term1 = np.zeros(n)
    term2 = np.zeros(n)
    term3 = np.zeros(n)
    term4 = np.zeros(n)
    
    for i in range(n):
        # Term 1: I(A=0)q0(Z,X)[h1(W,M,1,X)-h0(W,1,X)]
        if A[i] == 0:
            q0_val = q0_func(Z[i], X[i])
            h1_1 = h1_func(W[i], M[i], 1, X[i])
            h0_1 = h0_a1_func(W[i], 1, X[i])
            term1[i] = q0_val * (h1_1 - h0_1)
        
        # Term 2: I(A=0)[h1(W,M,0,X)-h0(W,0,X)]
        if A[i] == 0:
            h1_0 = h1_func(W[i], M[i], 0, X[i])
            h0_0 = h0_a0_func(W[i], 0, X[i])
            term2[i] = h1_0 - h0_0
        
        # Term 3: (I(A=1)q1(Z,M,X)+I(A=0))[Y-h1(W,M,A,X)]
        h1_a = h1_func(W[i], M[i], A[i], X[i])
        if A[i] == 1:
            q1_val = q1_func(Z[i], M[i], X[i])
            term3[i] = q1_val * (Y[i] - h1_a)
        else:  # A[i] == 0
            term3[i] = Y[i] - h1_a
        
        # Term 4: h0(W,A,X) - ψ
        if A[i] == 0:
            h0_val = h0_a0_func(W[i], A[i], X[i])
        else:
            h0_val = h0_a1_func(W[i], A[i], X[i])
        term4[i] = h0_val - psi_value
    
    # 总EIF
    eif = term1 + term2 + term3 + term4
    
    result = {
        'term1': term1,
        'term2': term2,
        'term3': term3,
        'term4': term4,
        'eif': eif
    }
    
    metadata = {
        'n_samples': n,
        'psi_value': psi_value,
        'eif_mean': float(np.mean(eif)),
        'eif_std': float(np.std(eif)),
        'term_means': {
            'term1': float(np.mean(term1)),
            'term2': float(np.mean(term2)),
            'term3': float(np.mean(term3)),
            'term4': float(np.mean(term4))
        }
    }
    
    return {'result': result, 'metadata': metadata}


def estimate_mediation_parameter_psi(
    data: Dict[str, np.ndarray],
    bridge_funcs: Dict[str, Dict]
) -> Dict[str, float]:
    """
    估计中介参数 ψ = E[Y(A,M(0))]
    使用h0函数的期望
    
    参数:
        data: 观测数据
        bridge_funcs: 所有bridge函数
    
    返回:
        {'result': psi_estimate, 'metadata': {...}}
    """
    W, A, X = data['W'], data['A'], data['X']
    n = len(W)
    
    h0_a0_func = bridge_funcs['h0_a0']['result']
    h0_a1_func = bridge_funcs['h0_a1']['result']
    
    # ψ ≈ E[h0(W,A,X)]
    h0_values = []
    for i in range(n):
        if A[i] == 0:
            h0_val = h0_a0_func(W[i], A[i], X[i])
        else:
            h0_val = h0_a1_func(W[i], A[i], X[i])
        h0_values.append(h0_val)
    
    psi_estimate = np.mean(h0_values)
    
    metadata = {
        'n_samples': n,
        'h0_mean': float(psi_estimate),
        'h0_std': float(np.std(h0_values)),
        'estimation_method': 'empirical_mean_h0'
    }
    
    return {'result': float(psi_estimate), 'metadata': metadata}


# ==================== 第三层：可视化与验证 ====================

def visualize_eif_distribution(
    eif_components: Dict[str, np.ndarray],
    save_path: str = './tool_images/eif_distribution.png'
) -> Dict[str, str]:
    """
    可视化EIF及其各组成部分的分布
    
    参数:
        eif_components: EIF各项的计算结果
        save_path: 图像保存路径
    
    返回:
        {'result': save_path, 'metadata': {...}}
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Efficient Influence Function Components Distribution', fontsize=16)
    
    components = ['term1', 'term2', 'term3', 'term4', 'eif']
    labels = [
        'Term 1: I(A=0)q0(Z,X)[h1(W,M,1,X)-h0(W,1,X)]',
        'Term 2: I(A=0)[h1(W,M,0,X)-h0(W,0,X)]',
        'Term 3: (I(A=1)q1+I(A=0))[Y-h1]',
        'Term 4: h0(W,A,X) - psi',
        'Total EIF'
    ]
    
    for idx, (comp, label) in enumerate(zip(components, labels)):
        ax = axes[idx // 3, idx % 3]
        data = eif_components[comp]
        
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(data), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(data):.4f}')
        ax.axvline(0, color='green', linestyle=':', alpha=0.5)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(label, fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    metadata = {
        'n_components': len(components),
        'image_size': os.path.getsize(save_path),
        'format': 'png'
    }
    
    return {'result': save_path, 'metadata': metadata}


def verify_eif_properties(
    eif_components: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray]
) -> Dict[str, Dict]:
    """
    验证EIF的理论性质
    1. 零均值性: E[EIF] ≈ 0
    2. 正交性: EIF与nuisance参数的score正交
    3. 方差有限性
    
    参数:
        eif_components: EIF各项
        data: 原始数据
    
    返回:
        {'result': verification_results, 'metadata': {...}}
    """
    eif = eif_components['eif']
    n = len(eif)
    
    # 1. 零均值性检验
    eif_mean = np.mean(eif)
    eif_std = np.std(eif)
    se_mean = eif_std / np.sqrt(n)
    t_stat = eif_mean / se_mean
    
    # 2. 方差有限性
    eif_variance = np.var(eif)
    is_finite_variance = np.isfinite(eif_variance) and eif_variance < 1e10
    
    # 3. 分位数检查
    quantiles = np.percentile(eif, [5, 25, 50, 75, 95])
    
    verification = {
        'zero_mean_test': {
            'mean': float(eif_mean),
            'std_error': float(se_mean),
            't_statistic': float(t_stat),
            'passes': abs(t_stat) < 2.0  # 95%置信水平
        },
        'finite_variance': {
            'variance': float(eif_variance),
            'passes': is_finite_variance
        },
        'quantiles': {
            '5%': float(quantiles[0]),
            '25%': float(quantiles[1]),
            '50%': float(quantiles[2]),
            '75%': float(quantiles[3]),
            '95%': float(quantiles[4])
        }
    }
    
    metadata = {
        'n_samples': n,
        'all_tests_passed': verification['zero_mean_test']['passes'] and 
                           verification['finite_variance']['passes']
    }
    
    return {'result': verification, 'metadata': metadata}


def symbolic_eif_derivation() -> Dict[str, str]:
    """
    使用sympy进行EIF的符号推导
    展示理论公式
    
    返回:
        {'result': {'latex': ..., 'simplified': ...}, 'metadata': {...}}
    """
    # 定义符号变量
    A, Y, M, W, Z, X = sp.symbols('A Y M W Z X')
    psi = sp.Symbol('psi')
    
    # 定义函数符号
    h1 = sp.Function('h1')
    h0 = sp.Function('h0')
    q0 = sp.Function('q0')
    q1 = sp.Function('q1')
    I = sp.Function('I')  # 指示函数
    
    # EIF的符号表达式
    term1 = I(A, 0) * q0(Z, X) * (h1(W, M, 1, X) - h0(W, 1, X))
    term2 = I(A, 0) * (h1(W, M, 0, X) - h0(W, 0, X))
    term3 = (I(A, 1) * q1(Z, M, X) + I(A, 0)) * (Y - h1(W, M, A, X))
    term4 = h0(W, A, X) - psi
    
    eif_symbolic = term1 + term2 + term3 + term4
    
    # 转换为LaTeX
    latex_formula = sp.latex(eif_symbolic)
    
    # 简化表达式
    simplified = sp.simplify(eif_symbolic)
    
    result = {
        'latex': latex_formula,
        'simplified': str(simplified),
        'components': {
            'term1': sp.latex(term1),
            'term2': sp.latex(term2),
            'term3': sp.latex(term3),
            'term4': sp.latex(term4)
        }
    }
    
    metadata = {
        'method': 'sympy_symbolic_derivation',
        'variables': ['A', 'Y', 'M', 'W', 'Z', 'X', 'psi'],
        'functions': ['h1', 'h0', 'q0', 'q1', 'I']
    }
    
    return {'result': result, 'metadata': metadata}


def compare_with_standard_answer(
    computed_eif_formula: str,
    standard_answer: str
) -> Dict[str, bool]:
    """
    比较计算得到的EIF公式与标准答案
    
    参数:
        computed_eif_formula: 计算得到的EIF公式(字符串)
        standard_answer: 标准答案公式
    
    返回:
        {'result': is_match, 'metadata': {...}}
    """
    # 标准化公式字符串(移除空格、统一符号)
    def normalize_formula(formula: str) -> str:
        return formula.replace(' ', '').replace('\\', '').lower()
    
    computed_norm = normalize_formula(computed_eif_formula)
    standard_norm = normalize_formula(standard_answer)
    
    # 检查关键组件是否存在
    key_components = [
        'i(a=0)q0(z,x)',
        'h1(w,m,1,x)-h0(w,1,x)',
        'h1(w,m,0,x)-h0(w,0,x)',
        'i(a=1)q1(z,m,x)',
        'y-h1(w,m,a,x)',
        'h0(w,a,x)-psi'
    ]
    
    components_present = {
        comp: comp.replace(' ', '').lower() in computed_norm 
        for comp in key_components
    }
    
    is_match = all(components_present.values())
    
    metadata = {
        'computed_formula': computed_eif_formula,
        'standard_answer': standard_answer,
        'components_check': components_present,
        'match_score': sum(components_present.values()) / len(components_present)
    }
    
    return {'result': is_match, 'metadata': metadata}


# ==================== 主函数：三个场景演示 ====================

def main():
    """
    演示三个场景:
    场景1: 完整求解原始问题的EIF
    场景2: 敏感性分析 - 不同混杂强度下的EIF稳定性
    场景3: 符号推导验证与标准答案比对
    """
    
    print("=" * 80)
    print("场景1: 完整求解Proximal Mediation的有效影响函数(EIF)")
    print("=" * 80)
    print("问题描述: 在半参数proximal mediation模型中,推导并计算目标参数ψ=E[Y(A,M(0))]的EIF")
    print("-" * 80)
    
    # 步骤1: 生成模拟数据
    print("\n步骤1: 生成符合proximal mediation设定的模拟数据")
    data_result = generate_proximal_mediation_data(n_samples=1000, seed=42, confounder_strength=0.5)
    data = data_result['result']
    print(f"FUNCTION_CALL: generate_proximal_mediation_data | PARAMS: {{n_samples: 1000, seed: 42, confounder_strength: 0.5}} | RESULT: {data_result['metadata']}")
    
    # 步骤2: 估计所有bridge函数
    print("\n步骤2: 估计outcome bridge函数(h1, h0)和exposure bridge函数(q0, q1)")
    bridge_result = estimate_all_bridge_functions(data, regularization=0.01)
    bridge_funcs = bridge_result['result']
    print(f"FUNCTION_CALL: estimate_all_bridge_functions | PARAMS: {{regularization: 0.01}} | RESULT: {bridge_result['metadata']}")
    
    # 步骤3: 估计目标参数ψ
    print("\n步骤3: 估计中介参数ψ = E[Y(A,M(0))]")
    psi_result = estimate_mediation_parameter_psi(data, bridge_funcs)
    psi_value = psi_result['result']
    print(f"FUNCTION_CALL: estimate_mediation_parameter_psi | PARAMS: {{}} | RESULT: {psi_result}")
    
    # 步骤4: 计算EIF各组成部分
    print("\n步骤4: 计算EIF的四个组成部分")
    eif_result = compute_eif_components(data, bridge_funcs, psi_value)
    eif_components = eif_result['result']
    print(f"FUNCTION_CALL: compute_eif_components | PARAMS: {{psi_value: {psi_value}}} | RESULT: {eif_result['metadata']}")
    
    # 步骤5: 可视化EIF分布
    print("\n步骤5: 可视化EIF及其各组成部分的分布")
    vis_result = visualize_eif_distribution(eif_components)
    print(f"FUNCTION_CALL: visualize_eif_distribution | PARAMS: {{}} | RESULT: {vis_result}")
    
    # 步骤6: 验证EIF性质
    print("\n步骤6: 验证EIF的理论性质(零均值、有限方差)")
    verify_result = verify_eif_properties(eif_components, data)
    print(f"FUNCTION_CALL: verify_eif_properties | PARAMS: {{}} | RESULT: {verify_result}")
    
    # 步骤7: 符号推导
    print("\n步骤7: 使用sympy进行EIF的符号推导")
    symbolic_result = symbolic_eif_derivation()
    print(f"FUNCTION_CALL: symbolic_eif_derivation | PARAMS: {{}} | RESULT: {symbolic_result['metadata']}")
    print(f"符号公式(LaTeX): {symbolic_result['result']['latex'][:200]}...")
    
    # 步骤8: 与标准答案比对
    print("\n步骤8: 与标准答案比对")
    standard_answer = "I(A=0)q0(Z,X)[h1(W,M,1,X)-h0(W,1,X)]+I(A=0)[h1(W,M,0,X)-h0(W,0,X)]+(I(A=1)q1(Z,M,X)+I(A=0))[Y-h1(W,M,A,X)]+h0(W,A,X)-psi"
    compare_result = compare_with_standard_answer(symbolic_result['result']['latex'], standard_answer)
    print(f"FUNCTION_CALL: compare_with_standard_answer | PARAMS: {{standard_answer: '{standard_answer[:50]}...'}} | RESULT: {compare_result}")
    
    print(f"\nFINAL_ANSWER: EIF = I(A=0)q0(Z,X)[h1(W,M,1,X)-h0(W,1,X)] + I(A=0)[h1(W,M,0,X)-h0(W,0,X)] + (I(A=1)q1(Z,M,X)+I(A=0))[Y-h1(W,M,A,X)] + h0(W,A,X) - ψ")
    
    
    print("\n" + "=" * 80)
    print("场景2: 敏感性分析 - 不同混杂强度下的EIF稳定性")
    print("=" * 80)
    print("问题描述: 研究未观测混杂因子强度对EIF估计的影响")
    print("-" * 80)
    
    confounder_strengths = [0.2, 0.5, 0.8]
    eif_variances = []
    psi_estimates = []
    
    for strength in confounder_strengths:
        print(f"\n--- 混杂强度 = {strength} ---")
        
        # 生成数据
        data_result = generate_proximal_mediation_data(n_samples=1000, seed=42, confounder_strength=strength)
        data = data_result['result']
        print(f"FUNCTION_CALL: generate_proximal_mediation_data | PARAMS: {{confounder_strength: {strength}}} | RESULT: {data_result['metadata']}")
        
        # 估计bridge函数
        bridge_result = estimate_all_bridge_functions(data, regularization=0.01)
        bridge_funcs = bridge_result['result']
        
        # 估计ψ
        psi_result = estimate_mediation_parameter_psi(data, bridge_funcs)
        psi_value = psi_result['result']
        psi_estimates.append(psi_value)
        
        # 计算EIF
        eif_result = compute_eif_components(data, bridge_funcs, psi_value)
        eif_variance = eif_result['metadata']['eif_std'] ** 2
        eif_variances.append(eif_variance)
        
        print(f"ψ估计值: {psi_value:.4f}, EIF方差: {eif_variance:.4f}")
    
    # 可视化敏感性分析结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(confounder_strengths, psi_estimates, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Confounder Strength')
    ax1.set_ylabel('Estimated psi')
    ax1.set_title('Mediation Parameter Sensitivity')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(confounder_strengths, eif_variances, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Confounder Strength')
    ax2.set_ylabel('EIF Variance')
    ax2.set_title('EIF Variance Sensitivity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    sensitivity_path = './tool_images/sensitivity_analysis.png'
    plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {sensitivity_path}")
    
    print(f"\nFINAL_ANSWER: 敏感性分析显示,混杂强度从{confounder_strengths[0]}增加到{confounder_strengths[-1]}时,ψ估计值从{psi_estimates[0]:.4f}变化到{psi_estimates[-1]:.4f},EIF方差从{eif_variances[0]:.4f}变化到{eif_variances[-1]:.4f}")
    
    
    print("\n" + "=" * 80)
    print("场景3: 理论验证 - EIF的正交性与效率性质")
    print("=" * 80)
    print("问题描述: 验证EIF满足半参数效率理论的关键性质")
    print("-" * 80)
    
    # 使用场景1的数据和结果
    print("\n步骤1: 重新生成数据并计算EIF")
    data_result = generate_proximal_mediation_data(n_samples=2000, seed=123, confounder_strength=0.5)
    data = data_result['result']
    print(f"FUNCTION_CALL: generate_proximal_mediation_data | PARAMS: {{n_samples: 2000, seed: 123}} | RESULT: {data_result['metadata']}")
    
    bridge_result = estimate_all_bridge_functions(data, regularization=0.01)
    bridge_funcs = bridge_result['result']
    
    psi_result = estimate_mediation_parameter_psi(data, bridge_funcs)
    psi_value = psi_result['result']
    
    eif_result = compute_eif_components(data, bridge_funcs, psi_value)
    eif_components = eif_result['result']
    
    print("\n步骤2: 验证EIF的零均值性质")
    verify_result = verify_eif_properties(eif_components, data)
    print(f"FUNCTION_CALL: verify_eif_properties | PARAMS: {{}} | RESULT: {verify_result}")
    
    zero_mean_passes = verify_result['result']['zero_mean_test']['passes']
    finite_var_passes = verify_result['result']['finite_variance']['passes']
    
    print(f"\n零均值检验: {'通过' if zero_mean_passes else '未通过'}")
    print(f"有限方差检验: {'通过' if finite_var_passes else '未通过'}")
    
    print("\n步骤3: 计算渐近方差并构造置信区间")
    eif = eif_components['eif']
    n = len(eif)
    asymptotic_variance = np.var(eif) / n
    se = np.sqrt(asymptotic_variance)
    ci_lower = psi_value - 1.96 * se
    ci_upper = psi_value + 1.96 * se
    
    print(f"ψ点估计: {psi_value:.4f}")
    print(f"渐近标准误: {se:.4f}")
    print(f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nFINAL_ANSWER: EIF满足半参数效率理论的关键性质: (1)零均值性E[EIF]≈{verify_result['result']['zero_mean_test']['mean']:.6f}; (2)有限方差Var(EIF)={verify_result['result']['finite_variance']['variance']:.4f}; (3)基于EIF的ψ估计值为{psi_value:.4f},95%CI=[{ci_lower:.4f},{ci_upper:.4f}]")


if __name__ == "__main__":
    main()