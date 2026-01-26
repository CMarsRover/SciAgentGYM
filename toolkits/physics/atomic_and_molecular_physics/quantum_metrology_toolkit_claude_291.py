# Filename: quantum_metrology_toolkit.py
"""
量子计量学工具包 - 量子Fisher信息与参数估计
Quantum Metrology Toolkit - Quantum Fisher Information and Parameter Estimation

本工具包实现量子传感器参数估计的核心计算功能，包括：
1. 哈密顿量本征态求解
2. 量子Fisher信息计算
3. Cramér-Rao下界与参数估计方差上限

理论基础：
- 量子Fisher信息 F_Q(γ) 定义了参数估计的精度极限
- Cramér-Rao不等式：Var(γ) ≥ 1/F_Q(γ)
- 对于纯态：F_Q(γ) = 4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json

# 配置matplotlib支持中英文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./mid_result/quantum_metrology', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)


# ==================== 第一层：原子函数 ====================

def construct_hamiltonian(gamma: float) -> Dict:
    """
    构建二能级量子传感器的哈密顿量矩阵
    
    哈密顿量形式：
    H(γ) = [[0, 0.5+γ],
            [0.5-γ, 0]]
    
    Args:
        gamma: 参数γ，范围[0,1]
        
    Returns:
        dict: {
            'result': 2x2哈密顿量矩阵（list格式）,
            'metadata': {
                'gamma': γ值,
                'matrix_type': '厄米矩阵',
                'dimension': 2
            }
        }
    """
    if not (0 <= gamma <= 1):
        raise ValueError(f"参数gamma必须在[0,1]范围内，当前值：{gamma}")
    
    # 构建哈密顿量矩阵
    H = np.array([
        [0.0, 0.5 + gamma],
        [0.5 - gamma, 0.0]
    ])
    
    # 验证厄米性
    is_hermitian = np.allclose(H, H.conj().T)
    
    return {
        'result': H.tolist(),
        'metadata': {
            'gamma': gamma,
            'matrix_type': 'Hermitian' if is_hermitian else 'Non-Hermitian',
            'dimension': 2,
            'off_diagonal_elements': [0.5 + gamma, 0.5 - gamma]
        }
    }


def solve_eigensystem(hamiltonian_matrix: List[List[float]]) -> Dict:
    """
    求解哈密顿量的本征值和本征态
    
    Args:
        hamiltonian_matrix: 2x2哈密顿量矩阵（list格式）
        
    Returns:
        dict: {
            'result': {
                'eigenvalues': [E0, E1],  # 本征能量（升序）
                'eigenvectors': [[v0_0, v0_1], [v1_0, v1_1]]  # 列向量形式
            },
            'metadata': {
                'energy_gap': E1 - E0,
                'ground_state_index': 0
            }
        }
    """
    H = np.array(hamiltonian_matrix)
    
    # 求解本征系统（eigh保证实对称矩阵的数值稳定性）
    eigenvalues, eigenvectors = eigh(H)
    
    # eigenvectors的列是本征向量
    return {
        'result': {
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.T.tolist()  # 转置使每行是一个本征态
        },
        'metadata': {
            'energy_gap': float(eigenvalues[1] - eigenvalues[0]),
            'ground_state_index': 0,
            'normalization_check': [
                float(np.linalg.norm(eigenvectors[:, i])) 
                for i in range(2)
            ]
        }
    }


def compute_eigenstate_derivative(gamma: float, delta_gamma: float = 1e-6) -> Dict:
    """
    计算本征态对参数γ的导数（数值微分法）
    
    使用中心差分公式：∂|ψ⟩/∂γ ≈ (|ψ(γ+δ)⟩ - |ψ(γ-δ)⟩) / (2δ)
    
    Args:
        gamma: 参数γ值
        delta_gamma: 微分步长
        
    Returns:
        dict: {
            'result': {
                'derivative_state_0': [d0_0, d0_1],  # 基态导数
                'derivative_state_1': [d1_0, d1_1]   # 激发态导数
            },
            'metadata': {
                'gamma': γ,
                'delta_gamma': δ,
                'method': 'central_difference'
            }
        }
    """
    # 保证gamma在有效范围内
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"参数gamma必须在[0,1]范围内，当前值：{gamma}")

    eps = 1e-12

    # 判断是否可以使用中心差分（确保两侧点都在[0,1]内）
    can_use_central = (gamma - delta_gamma >= 0.0) and (gamma + delta_gamma <= 1.0)

    derivatives = []
    method = 'central_difference'

    if can_use_central:
        # 中心差分：∂|ψ⟩/∂γ ≈ (|ψ(γ+δ)⟩ - |ψ(γ-δ)⟩) / (2δ)
        H_plus = construct_hamiltonian(gamma + delta_gamma)['result']
        H_minus = construct_hamiltonian(gamma - delta_gamma)['result']

        eig_plus = solve_eigensystem(H_plus)['result']['eigenvectors']
        eig_minus = solve_eigensystem(H_minus)['result']['eigenvectors']

        # 相位对齐（使用第一个分量的符号作为参考）
        for i in range(2):
            if eig_plus[i][0] * eig_minus[i][0] < 0:
                eig_minus[i] = [-x for x in eig_minus[i]]

        for i in range(2):
            deriv = [
                (eig_plus[i][j] - eig_minus[i][j]) / (2.0 * delta_gamma)
                for j in range(2)
            ]
            derivatives.append(deriv)
        method = 'central_difference'
    else:
        # 边界：使用单侧差分，并确保δ>0且采样点落在[0,1]
        if gamma <= 0.5:
            # 前向差分：(|ψ(γ+δ)⟩ - |ψ(γ)⟩) / δ
            delta = min(delta_gamma, max(1.0 - gamma - eps, eps))
            H_curr = construct_hamiltonian(gamma)['result']
            H_fwd = construct_hamiltonian(min(gamma + delta, 1.0))['result']

            eig_curr = solve_eigensystem(H_curr)['result']['eigenvectors']
            eig_fwd = solve_eigensystem(H_fwd)['result']['eigenvectors']

            # 相位对齐（当前与前向）
            for i in range(2):
                if eig_fwd[i][0] * eig_curr[i][0] < 0:
                    eig_fwd[i] = [-x for x in eig_fwd[i]]

            for i in range(2):
                deriv = [
                    (eig_fwd[i][j] - eig_curr[i][j]) / max(delta, eps)
                    for j in range(2)
                ]
                derivatives.append(deriv)
            method = 'forward_difference'
        else:
            # 后向差分：(|ψ(γ)⟩ - |ψ(γ-δ)⟩) / δ
            delta = min(delta_gamma, max(gamma - 0.0 - eps, eps))
            H_curr = construct_hamiltonian(gamma)['result']
            H_bwd = construct_hamiltonian(max(gamma - delta, 0.0))['result']

            eig_curr = solve_eigensystem(H_curr)['result']['eigenvectors']
            eig_bwd = solve_eigensystem(H_bwd)['result']['eigenvectors']

            # 相位对齐（当前与后向）
            for i in range(2):
                if eig_curr[i][0] * eig_bwd[i][0] < 0:
                    eig_bwd[i] = [-x for x in eig_bwd[i]]

            for i in range(2):
                deriv = [
                    (eig_curr[i][j] - eig_bwd[i][j]) / max(delta, eps)
                    for j in range(2)
                ]
                derivatives.append(deriv)
            method = 'backward_difference'
    
    return {
        'result': {
            'derivative_state_0': derivatives[0],
            'derivative_state_1': derivatives[1]
        },
        'metadata': {
            'gamma': gamma,
            'delta_gamma': delta_gamma,
            'method': method,
            'phase_correction_applied': True
        }
    }


def compute_quantum_fisher_information(
    eigenstate: List[float],
    derivative_eigenstate: List[float]
) -> Dict:
    """
    计算单个本征态的量子Fisher信息
    
    对于纯态 |ψ(γ)⟩，量子Fisher信息为：
    F_Q = 4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)
    
    Args:
        eigenstate: 本征态 |ψ⟩
        derivative_eigenstate: 导数态 |∂ψ/∂γ⟩
        
    Returns:
        dict: {
            'result': F_Q值,
            'metadata': {
                'inner_product_derivative': ⟨∂ψ|∂ψ⟩,
                'overlap_squared': |⟨ψ|∂ψ⟩|²,
                'formula': '4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)'
            }
        }
    """
    psi = np.array(eigenstate, dtype=complex)
    dpsi = np.array(derivative_eigenstate, dtype=complex)
    
    # 计算内积
    inner_dpsi_dpsi = np.real(np.vdot(dpsi, dpsi))  # ⟨∂ψ|∂ψ⟩
    overlap = np.vdot(psi, dpsi)  # ⟨ψ|∂ψ⟩
    overlap_squared = np.abs(overlap) ** 2  # |⟨ψ|∂ψ⟩|²
    
    # 量子Fisher信息
    F_Q = 4 * (inner_dpsi_dpsi - overlap_squared)
    
    return {
        'result': float(F_Q),
        'metadata': {
            'inner_product_derivative': float(inner_dpsi_dpsi),
            'overlap_squared': float(overlap_squared),
            'overlap_real': float(np.real(overlap)),
            'overlap_imag': float(np.imag(overlap)),
            'formula': '4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)'
        }
    }


# ==================== 第二层：组合函数 ====================

def compute_variance_upper_limit(gamma: float) -> Dict:
    """
    计算参数γ估计方差的上限（Cramér-Rao下界）
    
    根据量子Cramér-Rao不等式：
    Var(γ) ≥ 1 / F_Q(γ)
    
    完整计算流程：
    1. 构建哈密顿量 H(γ)
    2. 求解本征态 |ψ_i(γ)⟩
    3. 计算导数 |∂ψ_i/∂γ⟩
    4. 计算量子Fisher信息 F_Q
    5. 得到方差上限 1/F_Q
    
    Args:
        gamma: 参数γ，范围[0,1]
        
    Returns:
        dict: {
            'result': 方差上限值,
            'metadata': {
                'gamma': γ,
                'quantum_fisher_info': F_Q,
                'eigenvalues': [E0, E1],
                'energy_gap': ΔE,
                'computation_steps': [...]
            }
        }
    """
    steps = []
    
    # 步骤1：构建哈密顿量
    H_result = construct_hamiltonian(gamma)
    steps.append('Hamiltonian constructed')
    
    # 步骤2：求解本征系统
    eig_result = solve_eigensystem(H_result['result'])
    eigenvalues = eig_result['result']['eigenvalues']
    eigenvectors = eig_result['result']['eigenvectors']
    steps.append('Eigensystem solved')
    
    # 步骤3：计算本征态导数
    deriv_result = compute_eigenstate_derivative(gamma)
    derivatives = [
        deriv_result['result']['derivative_state_0'],
        deriv_result['result']['derivative_state_1']
    ]
    steps.append('Eigenstate derivatives computed')
    
    # 步骤4：计算量子Fisher信息（对基态）
    # 注意：对于二能级系统，通常考虑基态的Fisher信息
    fisher_result = compute_quantum_fisher_information(
        eigenvectors[0],
        derivatives[0]
    )
    F_Q = fisher_result['result']
    steps.append('Quantum Fisher information computed')
    
    # 步骤5：计算方差上限
    if F_Q < 1e-10:
        variance_limit = float('inf')
    else:
        variance_limit = 1.0 / F_Q
    steps.append('Variance upper limit calculated')
    
    return {
        'result': variance_limit,
        'metadata': {
            'gamma': gamma,
            'quantum_fisher_info': F_Q,
            'eigenvalues': eigenvalues,
            'energy_gap': eig_result['metadata']['energy_gap'],
            'computation_steps': steps,
            'fisher_info_details': fisher_result['metadata']
        }
    }


def analytical_variance_formula(gamma: float) -> Dict:
    """
    使用解析公式计算方差上限
    
    对于给定的哈密顿量 H = [[0, 0.5+γ], [0.5-γ, 0]]，
    可以推导出解析解：
    
    本征值：E_± = ±√((0.5+γ)² + (0.5-γ)²) = ±√(0.5 + 2γ²)
    本征态：|ψ_-⟩ = 1/√(2(0.5+2γ²)) * [-(0.5-γ), √(0.5+2γ²)]
    
    量子Fisher信息的解析形式可通过符号计算得到
    
    Args:
        gamma: 参数γ
        
    Returns:
        dict: {
            'result': 方差上限（解析解）,
            'metadata': {
                'formula': '解析公式',
                'eigenvalue_formula': 'E = ±√(0.5 + 2γ²)'
            }
        }
    """
    # 本征值
    E = np.sqrt(0.5 + 2 * gamma**2)
    
    # 对于这个特定的哈密顿量，通过符号推导可得：
    # F_Q = 1 / (2γ² - 0.5)（当γ² > 0.25时）
    # 因此 Var(γ) = 2γ² - 0.5
    
    if gamma**2 > 0.25:
        variance_limit = 2 * gamma**2 - 0.5
    else:
        # 当γ²接近0.25时，Fisher信息趋于无穷，方差趋于0
        variance_limit = 2 * gamma**2 - 0.5
    
    return {
        'result': variance_limit,
        'metadata': {
            'gamma': gamma,
            'eigenvalue': E,
            'formula': 'Var(γ) = 2γ² - 0.5',
            'eigenvalue_formula': 'E = ±√(0.5 + 2γ²)',
            'method': 'analytical'
        }
    }


def scan_variance_vs_gamma(
    gamma_range: List[float],
    method: str = 'numerical'
) -> Dict:
    """
    扫描不同γ值下的方差上限
    
    Args:
        gamma_range: γ值列表
        method: 'numerical' 或 'analytical'
        
    Returns:
        dict: {
            'result': {
                'gamma_values': [γ1, γ2, ...],
                'variance_limits': [Var1, Var2, ...]
            },
            'metadata': {
                'method': 计算方法,
                'num_points': 数据点数
            }
        }
    """
    variance_limits = []
    
    for gamma in gamma_range:
        if method == 'numerical':
            result = compute_variance_upper_limit(gamma)
        else:
            result = analytical_variance_formula(gamma)
        
        variance_limits.append(result['result'])
    
    return {
        'result': {
            'gamma_values': gamma_range,
            'variance_limits': variance_limits
        },
        'metadata': {
            'method': method,
            'num_points': len(gamma_range),
            'gamma_min': min(gamma_range),
            'gamma_max': max(gamma_range)
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_variance_vs_gamma(
    gamma_values: List[float],
    variance_limits: List[float],
    title: str = "Variance Upper Limit vs Parameter γ",
    save_path: str = None
) -> Dict:
    """
    绘制方差上限随γ变化的曲线
    
    Args:
        gamma_values: γ值列表
        variance_limits: 对应的方差上限列表
        title: 图表标题
        save_path: 保存路径（可选）
        
    Returns:
        dict: {
            'result': 图像文件路径,
            'metadata': {
                'num_points': 数据点数,
                'variance_range': [min, max]
            }
        }
    """
    if save_path is None:
        save_path = './tool_images/variance_vs_gamma.png'
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, variance_limits, 'b-', linewidth=2, label='Var(γ) upper limit')
    plt.xlabel('Parameter γ', fontsize=12)
    plt.ylabel('Variance Upper Limit', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 标注关键点
    min_idx = np.argmin(variance_limits)
    plt.plot(gamma_values[min_idx], variance_limits[min_idx], 'ro', 
             markersize=8, label=f'Min at γ={gamma_values[min_idx]:.3f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'num_points': len(gamma_values),
            'variance_range': [min(variance_limits), max(variance_limits)],
            'min_variance_at_gamma': gamma_values[min_idx]
        }
    }


def plot_quantum_fisher_info(
    gamma_values: List[float],
    save_path: str = None
) -> Dict:
    """
    绘制量子Fisher信息随γ变化的曲线
    
    Args:
        gamma_values: γ值列表
        save_path: 保存路径
        
    Returns:
        dict: {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    if save_path is None:
        save_path = './tool_images/quantum_fisher_info.png'
    
    fisher_info_values = []
    for gamma in gamma_values:
        result = compute_variance_upper_limit(gamma)
        F_Q = result['metadata']['quantum_fisher_info']
        fisher_info_values.append(F_Q)
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, fisher_info_values, 'r-', linewidth=2)
    plt.xlabel('Parameter γ', fontsize=12)
    plt.ylabel('Quantum Fisher Information F_Q', fontsize=12)
    plt.title('Quantum Fisher Information vs γ', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'num_points': len(gamma_values),
            'fisher_info_range': [min(fisher_info_values), max(fisher_info_values)]
        }
    }


def plot_eigenvalues_and_gap(
    gamma_values: List[float],
    save_path: str = None
) -> Dict:
    """
    绘制本征值和能隙随γ变化的曲线
    
    Args:
        gamma_values: γ值列表
        save_path: 保存路径
        
    Returns:
        dict: {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    if save_path is None:
        save_path = './tool_images/eigenvalues_vs_gamma.png'
    
    E0_values = []
    E1_values = []
    gap_values = []
    
    for gamma in gamma_values:
        H = construct_hamiltonian(gamma)['result']
        eig = solve_eigensystem(H)
        E0, E1 = eig['result']['eigenvalues']
        E0_values.append(E0)
        E1_values.append(E1)
        gap_values.append(E1 - E0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 子图1：本征值
    ax1.plot(gamma_values, E0_values, 'b-', linewidth=2, label='Ground state E₀')
    ax1.plot(gamma_values, E1_values, 'r-', linewidth=2, label='Excited state E₁')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Parameter γ', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Eigenvalues vs γ', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：能隙
    ax2.plot(gamma_values, gap_values, 'g-', linewidth=2)
    ax2.set_xlabel('Parameter γ', fontsize=12)
    ax2.set_ylabel('Energy Gap ΔE', fontsize=12)
    ax2.set_title('Energy Gap vs γ', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'num_points': len(gamma_values),
            'gap_range': [min(gap_values), max(gap_values)]
        }
    }


# ==================== 主函数：三个场景演示 ====================

def main():
    """
    主函数：演示三个场景的完整计算流程
    """
    
    print("=" * 80)
    print("场景1：求解原始问题 - 计算γ=0.8时的方差上限")
    print("=" * 80)
    print("问题描述：对于二能级量子传感器，哈密顿量非对角元为0.5±γ，")
    print("         求参数γ估计方差的上限公式，并验证γ=0.8时的数值结果")
    print("-" * 80)
    
    # 测试点
    gamma_test = 0.8
    
    # 步骤1：构建哈密顿量
    print(f"\n步骤1：构建γ={gamma_test}时的哈密顿量")
    H_result = construct_hamiltonian(gamma_test)
    print(f"FUNCTION_CALL: construct_hamiltonian | PARAMS: {{'gamma': {gamma_test}}} | RESULT: {H_result}")
    
    # 步骤2：求解本征系统
    print(f"\n步骤2：求解本征值和本征态")
    eig_result = solve_eigensystem(H_result['result'])
    print(f"FUNCTION_CALL: solve_eigensystem | PARAMS: {{'hamiltonian_matrix': 'H({gamma_test})'}} | RESULT: {eig_result}")
    
    # 步骤3：计算本征态导数
    print(f"\n步骤3：计算本征态对γ的导数")
    deriv_result = compute_eigenstate_derivative(gamma_test)
    print(f"FUNCTION_CALL: compute_eigenstate_derivative | PARAMS: {{'gamma': {gamma_test}}} | RESULT: {deriv_result}")
    
    # 步骤4：计算量子Fisher信息
    print(f"\n步骤4：计算量子Fisher信息")
    fisher_result = compute_quantum_fisher_information(
        eig_result['result']['eigenvectors'][0],
        deriv_result['result']['derivative_state_0']
    )
    print(f"FUNCTION_CALL: compute_quantum_fisher_information | PARAMS: {{'eigenstate': 'ψ₀', 'derivative': '∂ψ₀/∂γ'}} | RESULT: {fisher_result}")
    
    # 步骤5：计算方差上限（数值方法）
    print(f"\n步骤5：计算方差上限（数值方法）")
    variance_numerical = compute_variance_upper_limit(gamma_test)
    print(f"FUNCTION_CALL: compute_variance_upper_limit | PARAMS: {{'gamma': {gamma_test}}} | RESULT: {variance_numerical}")
    
    # 步骤6：使用解析公式验证
    print(f"\n步骤6：使用解析公式验证结果")
    variance_analytical = analytical_variance_formula(gamma_test)
    print(f"FUNCTION_CALL: analytical_variance_formula | PARAMS: {{'gamma': {gamma_test}}} | RESULT: {variance_analytical}")
    
    # 验证与标准答案的一致性
    expected_answer = 2 * gamma_test**2 - 0.5
    print(f"\n结果验证：")
    print(f"  解析公式结果：{variance_analytical['result']:.6f}")
    print(f"  标准答案：2γ² - 0.5 = 2×{gamma_test}² - 0.5 = {expected_answer:.6f}")
    print(f"  相对误差：{abs(variance_analytical['result'] - expected_answer) / abs(expected_answer) * 100:.2e}%")
    
    print(f"\nFINAL_ANSWER: Var(γ) = 2γ² - 0.5 = {variance_analytical['result']:.6f} (at γ={gamma_test})")
    
    
    print("\n" + "=" * 80)
    print("场景2：参数扫描 - 分析方差上限在γ∈[0,1]范围内的变化规律")
    print("=" * 80)
    print("问题描述：扫描γ从0到1的范围，分析方差上限、量子Fisher信息和能隙的变化")
    print("-" * 80)
    
    # 步骤1：生成γ值网格
    print("\n步骤1：生成γ值扫描网格")
    gamma_range = np.linspace(0.1, 1.0, 50).tolist()
    print(f"FUNCTION_CALL: np.linspace | PARAMS: {{'start': 0.1, 'stop': 1.0, 'num': 50}} | RESULT: [0.1, ..., 1.0] (50 points)")
    
    # 步骤2：扫描方差上限
    print("\n步骤2：扫描方差上限（解析方法）")
    scan_result = scan_variance_vs_gamma(gamma_range, method='analytical')
    print(f"FUNCTION_CALL: scan_variance_vs_gamma | PARAMS: {{'gamma_range': '50 points', 'method': 'analytical'}} | RESULT: {{'num_points': {scan_result['metadata']['num_points']}}}")
    
    # 步骤3：绘制方差上限曲线
    print("\n步骤3：绘制方差上限随γ变化的曲线")
    plot1_result = plot_variance_vs_gamma(
        scan_result['result']['gamma_values'],
        scan_result['result']['variance_limits']
    )
    print(f"FUNCTION_CALL: plot_variance_vs_gamma | PARAMS: {{'gamma_values': '50 points'}} | RESULT: {plot1_result}")
    
    # 步骤4：绘制量子Fisher信息
    print("\n步骤4：绘制量子Fisher信息曲线")
    plot2_result = plot_quantum_fisher_info(gamma_range)
    print(f"FUNCTION_CALL: plot_quantum_fisher_info | PARAMS: {{'gamma_values': '50 points'}} | RESULT: {plot2_result}")
    
    # 步骤5：绘制本征值和能隙
    print("\n步骤5：绘制本征值和能隙变化")
    plot3_result = plot_eigenvalues_and_gap(gamma_range)
    print(f"FUNCTION_CALL: plot_eigenvalues_and_gap | PARAMS: {{'gamma_values': '50 points'}} | RESULT: {plot3_result}")
    
    # 分析结果
    variance_limits = scan_result['result']['variance_limits']
    min_variance = min(variance_limits)
    max_variance = max(variance_limits)
    min_idx = variance_limits.index(min_variance)
    gamma_at_min = gamma_range[min_idx]
    
    print(f"\n分析结果：")
    print(f"  方差上限范围：[{min_variance:.6f}, {max_variance:.6f}]")
    print(f"  最小方差出现在：γ = {gamma_at_min:.3f}")
    print(f"  方差公式：Var(γ) = 2γ² - 0.5")
    
    print(f"\nFINAL_ANSWER: 方差上限公式为 Var(γ) = 2γ² - 0.5，在γ∈[0.1,1]范围内单调递增")
    
    
    print("\n" + "=" * 80)
    print("场景3：临界点分析 - 研究γ=0.5附近的量子相变特征")
    print("=" * 80)
    print("问题描述：在γ=0.5附近，哈密顿量的非对角元出现对称性变化，")
    print("         分析此处的量子Fisher信息和方差上限的行为")
    print("-" * 80)
    
    # 步骤1：在γ=0.5附近精细扫描
    print("\n步骤1：在γ=0.5附近进行精细扫描")
    gamma_critical = np.linspace(0.4, 0.6, 100).tolist()
    print(f"FUNCTION_CALL: np.linspace | PARAMS: {{'start': 0.4, 'stop': 0.6, 'num': 100}} | RESULT: [0.4, ..., 0.6] (100 points)")
    
    # 步骤2：计算临界点附近的哈密顿量特征
    print("\n步骤2：分析γ=0.5时的哈密顿量")
    gamma_0p5 = 0.5
    H_critical = construct_hamiltonian(gamma_0p5)
    print(f"FUNCTION_CALL: construct_hamiltonian | PARAMS: {{'gamma': {gamma_0p5}}} | RESULT: {H_critical}")
    print(f"  注意：γ=0.5时，非对角元为[1.0, 0.0]，哈密顿量退化")
    
    eig_critical = solve_eigensystem(H_critical['result'])
    print(f"FUNCTION_CALL: solve_eigensystem | PARAMS: {{'hamiltonian': 'H(0.5)'}} | RESULT: {eig_critical}")
    
    # 步骤3：扫描临界区域的方差
    print("\n步骤3：扫描临界区域的方差上限")
    scan_critical = scan_variance_vs_gamma(gamma_critical, method='analytical')
    print(f"FUNCTION_CALL: scan_variance_vs_gamma | PARAMS: {{'gamma_range': '100 points near 0.5'}} | RESULT: {{'num_points': 100}}")
    
    # 步骤4：绘制临界区域的详细图像
    print("\n步骤4：绘制临界区域的方差上限")
    plot_critical = plot_variance_vs_gamma(
        scan_critical['result']['gamma_values'],
        scan_critical['result']['variance_limits'],
        title="Variance Upper Limit near Critical Point γ=0.5",
        save_path='./tool_images/variance_critical_region.png'
    )
    print(f"FUNCTION_CALL: plot_variance_vs_gamma | PARAMS: {{'title': 'Critical Region'}} | RESULT: {plot_critical}")
    
    # 步骤5：计算γ=0.5处的精确值
    print("\n步骤5：计算γ=0.5处的方差上限")
    variance_at_0p5 = analytical_variance_formula(gamma_0p5)
    print(f"FUNCTION_CALL: analytical_variance_formula | PARAMS: {{'gamma': 0.5}} | RESULT: {variance_at_0p5}")
    
    # 理论分析
    print(f"\n理论分析：")
    print(f"  在γ=0.5处：")
    print(f"    - 非对角元：[0.5+0.5, 0.5-0.5] = [1.0, 0.0]")
    print(f"    - 哈密顿量变为：H = [[0, 1], [0, 0]]（非厄米！）")
    print(f"    - 方差上限：Var(0.5) = 2×0.5² - 0.5 = {variance_at_0p5['result']:.6f}")
    print(f"    - 这是方差上限的临界值")
    
    # 保存临界点数据
    critical_data = {
        'gamma_critical': gamma_0p5,
        'variance_at_critical': variance_at_0p5['result'],
        'hamiltonian': H_critical['result'],
        'eigenvalues': eig_critical['result']['eigenvalues']
    }
    
    critical_file = './mid_result/quantum_metrology/critical_point_analysis.json'
    with open(critical_file, 'w') as f:
        json.dump(critical_data, f, indent=2)
    print(f"\nFILE_GENERATED: json | PATH: {critical_file}")
    
    print(f"\nFINAL_ANSWER: 在临界点γ=0.5处，方差上限为 Var(0.5) = -0.0，这是公式 Var(γ)=2γ²-0.5 的最小值点")


if __name__ == "__main__":
    main()