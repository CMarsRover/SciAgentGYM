# Filename: quantum_mechanics_toolkit.py

"""
Quantum Mechanics Toolkit for Spin State Analysis
专业量子力学计算工具包 - 自旋态分析

核心功能：
1. 自旋态归一化与表示
2. 算符本征值/本征态求解
3. 测量概率计算
4. 算符期望值计算

依赖库：
- numpy: 数值计算
- scipy.linalg: 线性代数运算（本征值求解）
- sympy: 符号计算（验证）
- qutip: 量子工具箱（专业量子计算库）
"""

import numpy as np
from scipy import linalg
import json
import os
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt

# 全局常量
HBAR = 1.0  # 约化普朗克常数（自然单位制）
TOLERANCE = 1e-10  # 数值精度容差

# ============================================================================
# 第一层：原子函数（基础量子态操作）
# ============================================================================

def normalize_state_vector(coefficients: List[complex]) -> Dict:
    """
    归一化量子态矢量
    
    参数：
        coefficients: 复数系数列表，如 [(1+1j), (2-1j)]
    
    返回：
        {
            'result': {
                'normalized_coeffs': List[complex],  # 归一化后的系数
                'norm_factor': float  # 归一化因子
            },
            'metadata': {
                'original_norm': float,
                'dimension': int
            }
        }
    """
    if not coefficients:
        raise ValueError("系数列表不能为空")
    
    # 转换为numpy数组
    state = np.array(coefficients, dtype=complex)
    
    # 计算范数
    norm = np.sqrt(np.sum(np.abs(state)**2))
    
    if norm < TOLERANCE:
        raise ValueError(f"态矢量范数过小: {norm}")
    
    # 归一化
    normalized_state = state / norm
    
    return {
        'result': {
            'normalized_coeffs': normalized_state.tolist(),
            'norm_factor': float(norm)
        },
        'metadata': {
            'original_norm': float(norm),
            'dimension': len(coefficients),
            'is_normalized': True
        }
    }


def construct_operator_matrix(matrix_elements: Dict[str, Union[float, complex]], 
                              dimension: int) -> Dict:
    """
    根据矩阵元素规则构造算符矩阵
    
    参数：
        matrix_elements: 矩阵元素规则字典
            {
                'diagonal': float/complex,  # 对角元素值
                'off_diagonal': float/complex  # 非对角元素值
            }
        dimension: 矩阵维度
    
    返回：
        {
            'result': List[List[complex]],  # 算符矩阵（列表形式）
            'metadata': {
                'dimension': int,
                'is_hermitian': bool,
                'trace': complex
            }
        }
    """
    if dimension < 2:
        raise ValueError(f"矩阵维度必须 >= 2，当前: {dimension}")
    
    diag_val = matrix_elements.get('diagonal', 0)
    off_diag_val = matrix_elements.get('off_diagonal', 0)
    
    # 构造矩阵
    matrix = np.zeros((dimension, dimension), dtype=complex)
    
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                matrix[i, j] = diag_val
            else:
                matrix[i, j] = off_diag_val
    
    # 检查厄米性
    is_hermitian = np.allclose(matrix, matrix.conj().T, atol=TOLERANCE)
    
    return {
        'result': matrix.tolist(),
        'metadata': {
            'dimension': dimension,
            'is_hermitian': is_hermitian,
            'trace': complex(np.trace(matrix)),
            'matrix_type': 'hermitian' if is_hermitian else 'general'
        }
    }


def solve_eigenvalue_problem(matrix: List[List[complex]]) -> Dict:
    """
    求解算符矩阵的本征值和本征态
    
    参数：
        matrix: 算符矩阵（列表形式）
    
    返回：
        {
            'result': {
                'eigenvalues': List[float],  # 本征值（实数）
                'eigenvectors': List[List[complex]]  # 本征矢量（列向量）
            },
            'metadata': {
                'num_eigenvalues': int,
                'eigenvalue_degeneracy': Dict[float, int]
            }
        }
    """
    # 转换为numpy数组
    A = np.array(matrix, dtype=complex)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"矩阵必须是方阵，当前形状: {A.shape}")
    
    # 求解本征值问题（使用厄米矩阵专用算法）
    if np.allclose(A, A.conj().T, atol=TOLERANCE):
        eigenvalues, eigenvectors = linalg.eigh(A)  # 厄米矩阵
    else:
        eigenvalues, eigenvectors = linalg.eig(A)  # 一般矩阵
        eigenvalues = eigenvalues.real  # 取实部
    
    # 按本征值排序（降序）
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 检查简并度
    degeneracy = {}
    for ev in eigenvalues:
        ev_rounded = round(float(ev), 10)
        degeneracy[ev_rounded] = degeneracy.get(ev_rounded, 0) + 1
    
    return {
        'result': {
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.T.tolist()  # 转置为行向量列表
        },
        'metadata': {
            'num_eigenvalues': len(eigenvalues),
            'eigenvalue_degeneracy': degeneracy,
            'all_real': np.allclose(eigenvalues.imag, 0, atol=TOLERANCE)
        }
    }


def compute_inner_product(state1: List[complex], state2: List[complex]) -> Dict:
    """
    计算两个量子态的内积 <state1|state2>
    
    参数：
        state1: 第一个态矢量（bra）
        state2: 第二个态矢量（ket）
    
    返回：
        {
            'result': complex,  # 内积值
            'metadata': {
                'magnitude': float,
                'phase': float  # 相位（弧度）
            }
        }
    """
    if len(state1) != len(state2):
        raise ValueError(f"态矢量维度不匹配: {len(state1)} vs {len(state2)}")
    
    s1 = np.array(state1, dtype=complex)
    s2 = np.array(state2, dtype=complex)
    
    # 内积 = <s1|s2> = s1^† · s2
    inner_prod = np.vdot(s1, s2)
    
    magnitude = np.abs(inner_prod)
    phase = np.angle(inner_prod)
    
    return {
        'result': complex(inner_prod),
        'metadata': {
            'magnitude': float(magnitude),
            'phase': float(phase),
            'probability': float(magnitude**2)
        }
    }


# ============================================================================
# 第二层：组合函数（量子测量与期望值）
# ============================================================================

def calculate_measurement_probabilities(initial_state: List[complex],
                                       eigenstates: List[List[complex]]) -> Dict:
    """
    计算在给定本征态基底下的测量概率
    
    参数：
        initial_state: 初始量子态（归一化）
        eigenstates: 本征态列表（每个本征态是一个列表）
    
    返回：
        {
            'result': {
                'probabilities': List[float],  # 各本征态的测量概率
                'probability_sum': float  # 概率和（应为1）
            },
            'metadata': {
                'num_eigenstates': int,
                'inner_products': List[complex]
            }
        }
    """
    if not eigenstates:
        raise ValueError("本征态列表不能为空")
    
    probabilities = []
    inner_products = []
    
    for eigenstate in eigenstates:
        # 调用函数：compute_inner_product()
        inner_prod_result = compute_inner_product(eigenstate, initial_state)
        inner_prod = inner_prod_result['result']
        inner_products.append(inner_prod)
        
        # 概率 = |<eigenstate|initial_state>|^2
        prob = inner_prod_result['metadata']['probability']
        probabilities.append(prob)
    
    prob_sum = sum(probabilities)
    
    if abs(prob_sum - 1.0) > TOLERANCE:
        print(f"警告：概率和 = {prob_sum:.6f}，偏离1.0")
    
    return {
        'result': {
            'probabilities': probabilities,
            'probability_sum': float(prob_sum)
        },
        'metadata': {
            'num_eigenstates': len(eigenstates),
            'inner_products': [complex(ip) for ip in inner_products]
        }
    }


def calculate_expectation_value(state: List[complex],
                               operator_matrix: List[List[complex]]) -> Dict:
    """
    计算算符在给定态下的期望值 <state|A|state>
    
    参数：
        state: 量子态（归一化）
        operator_matrix: 算符矩阵
    
    返回：
        {
            'result': float,  # 期望值（实数）
            'metadata': {
                'state_norm': float,
                'operator_trace': complex
            }
        }
    """
    psi = np.array(state, dtype=complex)
    A = np.array(operator_matrix, dtype=complex)
    
    if len(psi) != A.shape[0] or A.shape[0] != A.shape[1]:
        raise ValueError(f"维度不匹配: state={len(psi)}, operator={A.shape}")
    
    # 期望值 = <psi|A|psi> = psi^† · A · psi
    expectation = np.vdot(psi, A @ psi)
    
    # 对于厄米算符，期望值应为实数
    if abs(expectation.imag) > TOLERANCE:
        print(f"警告：期望值虚部非零: {expectation.imag}")
    
    state_norm = np.linalg.norm(psi)
    
    return {
        'result': float(expectation.real),
        'metadata': {
            'state_norm': float(state_norm),
            'operator_trace': complex(np.trace(A)),
            'expectation_complex': complex(expectation)
        }
    }


def project_state_to_eigenbasis(state: List[complex],
                               eigenstates: List[List[complex]],
                               eigenvalues: List[float]) -> Dict:
    """
    将量子态投影到本征态基底，并计算各分量
    
    参数：
        state: 初始量子态
        eigenstates: 本征态列表
        eigenvalues: 对应的本征值列表
    
    返回：
        {
            'result': {
                'coefficients': List[complex],  # 投影系数
                'probabilities': List[float],  # 测量概率
                'weighted_eigenvalues': List[float]  # 加权本征值
            },
            'metadata': {
                'reconstruction_error': float
            }
        }
    """
    if len(eigenstates) != len(eigenvalues):
        raise ValueError("本征态和本征值数量不匹配")
    
    # 调用函数：calculate_measurement_probabilities()
    prob_result = calculate_measurement_probabilities(state, eigenstates)
    probabilities = prob_result['result']['probabilities']
    coefficients = prob_result['metadata']['inner_products']
    
    # 加权本征值（用于期望值计算）
    weighted_eigenvalues = [prob * ev for prob, ev in zip(probabilities, eigenvalues)]
    
    # 重构态矢量以验证完备性
    reconstructed = np.zeros(len(state), dtype=complex)
    for coeff, eigenstate in zip(coefficients, eigenstates):
        reconstructed += coeff * np.array(eigenstate)
    
    reconstruction_error = np.linalg.norm(reconstructed - np.array(state))
    
    return {
        'result': {
            'coefficients': [complex(c) for c in coefficients],
            'probabilities': probabilities,
            'weighted_eigenvalues': weighted_eigenvalues
        },
        'metadata': {
            'reconstruction_error': float(reconstruction_error),
            'basis_complete': reconstruction_error < TOLERANCE
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_probability_distribution(probabilities: List[float],
                                      eigenvalues: List[float],
                                      title: str = "Measurement Probability Distribution") -> Dict:
    """
    可视化测量概率分布
    
    参数：
        probabilities: 概率列表
        eigenvalues: 对应的本征值
        title: 图表标题
    
    返回：
        {
            'result': str,  # 图像文件路径
            'metadata': {
                'num_states': int,
                'max_probability': float
            }
        }
    """
    # 确保中英文显示正常
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：柱状图
    x_pos = np.arange(len(probabilities))
    bars = ax1.bar(x_pos, probabilities, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Eigenstate Index', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'|ψ{i}⟩' for i in range(len(probabilities))])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # 子图2：本征值-概率散点图
    ax2.scatter(eigenvalues, probabilities, s=200, alpha=0.6, 
               c=probabilities, cmap='viridis', edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Eigenvalue (ℏ/2 units)', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Eigenvalue vs Probability', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(probabilities), 
                                                 vmax=max(probabilities)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Probability', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('./tool_images', exist_ok=True)
    filepath = './tool_images/probability_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'num_states': len(probabilities),
            'max_probability': float(max(probabilities)),
            'min_probability': float(min(probabilities))
        }
    }


def visualize_quantum_state(state_coeffs: List[complex],
                           basis_labels: List[str] = None,
                           title: str = "Quantum State Representation") -> Dict:
    """
    可视化量子态的复数系数（幅值和相位）
    
    参数：
        state_coeffs: 量子态系数列表
        basis_labels: 基底标签
        title: 图表标题
    
    返回：
        {
            'result': str,  # 图像文件路径
            'metadata': {
                'num_components': int
            }
        }
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    coeffs = np.array(state_coeffs, dtype=complex)
    amplitudes = np.abs(coeffs)
    phases = np.angle(coeffs)
    
    if basis_labels is None:
        basis_labels = [f'|{i}⟩' for i in range(len(coeffs))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：幅值
    x_pos = np.arange(len(amplitudes))
    bars1 = ax1.bar(x_pos, amplitudes, alpha=0.7, color='coral', edgecolor='black')
    ax1.set_xlabel('Basis State', fontsize=12)
    ax1.set_ylabel('Amplitude |c_i|', fontsize=12)
    ax1.set_title(f'{title} - Amplitudes', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(basis_labels)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, amp in zip(bars1, amplitudes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{amp:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 子图2：相位
    bars2 = ax2.bar(x_pos, phases, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Basis State', fontsize=12)
    ax2.set_ylabel('Phase (radians)', fontsize=12)
    ax2.set_title(f'{title} - Phases', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(basis_labels)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, phase in zip(bars2, phases):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{phase:.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs('./tool_images', exist_ok=True)
    filepath = './tool_images/quantum_state_representation.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'num_components': len(state_coeffs)
        }
    }


# ============================================================================
# 主函数：三个场景演示
# ============================================================================

def main():
    """
    演示量子力学计算工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 自旋态测量概率与算符期望值")
    print("=" * 80)
    print("问题描述：")
    print("给定电子自旋态 |α⟩ ∝ (1+i)|up⟩ + (2-i)|down⟩")
    print("算符矩阵 A_ij = ℏ/2 (i≠j), 0 (i=j)")
    print("求：1) 在算符本征态下的测量概率  2) 算符期望值")
    print("-" * 80)
    
    # 步骤1：归一化初始态
    print("\n步骤1：归一化初始自旋态")
    initial_coeffs = [1+1j, 2-1j]  # (1+i)|up⟩ + (2-i)|down⟩
    # 调用函数：normalize_state_vector()
    norm_result = normalize_state_vector(initial_coeffs)
    normalized_state = norm_result['result']['normalized_coeffs']
    print(f"FUNCTION_CALL: normalize_state_vector | PARAMS: {initial_coeffs} | RESULT: {norm_result}")
    print(f"归一化态: {normalized_state}")
    print(f"归一化因子: {norm_result['result']['norm_factor']:.6f}")
    
    # 步骤2：构造算符矩阵
    print("\n步骤2：构造算符矩阵 A")
    matrix_params = {
        'diagonal': 0,
        'off_diagonal': HBAR / 2
    }
    # 调用函数：construct_operator_matrix()
    operator_result = construct_operator_matrix(matrix_params, dimension=2)
    operator_matrix = operator_result['result']
    print(f"FUNCTION_CALL: construct_operator_matrix | PARAMS: {matrix_params} | RESULT: {operator_result}")
    print(f"算符矩阵 A:")
    for row in operator_matrix:
        print(f"  {row}")
    print(f"是否厄米: {operator_result['metadata']['is_hermitian']}")
    
    # 步骤3：求解本征值和本征态
    print("\n步骤3：求解算符A的本征值和本征态")
    # 调用函数：solve_eigenvalue_problem()
    eigen_result = solve_eigenvalue_problem(operator_matrix)
    eigenvalues = eigen_result['result']['eigenvalues']
    eigenvectors = eigen_result['result']['eigenvectors']
    print(f"FUNCTION_CALL: solve_eigenvalue_problem | PARAMS: operator_matrix | RESULT: {eigen_result}")
    print(f"本征值: {eigenvalues}")
    print(f"本征态:")
    for i, ev in enumerate(eigenvectors):
        print(f"  |ψ{i}⟩ = {ev}")
    
    # 步骤4：计算测量概率
    print("\n步骤4：计算在本征态基底下的测量概率")
    # 调用函数：calculate_measurement_probabilities()
    prob_result = calculate_measurement_probabilities(normalized_state, eigenvectors)
    probabilities = prob_result['result']['probabilities']
    print(f"FUNCTION_CALL: calculate_measurement_probabilities | PARAMS: normalized_state, eigenvectors | RESULT: {prob_result}")
    print(f"测量概率:")
    for i, (prob, ev) in enumerate(zip(probabilities, eigenvalues)):
        print(f"  P(λ={ev:.4f}) = {prob:.6f}")
    print(f"概率和: {prob_result['result']['probability_sum']:.6f}")
    
    # 步骤5：计算算符期望值
    print("\n步骤5：计算算符A的期望值")
    # 调用函数：calculate_expectation_value()
    expect_result = calculate_expectation_value(normalized_state, operator_matrix)
    expectation_value = expect_result['result']
    print(f"FUNCTION_CALL: calculate_expectation_value | PARAMS: normalized_state, operator_matrix | RESULT: {expect_result}")
    print(f"期望值 ⟨A⟩ = {expectation_value:.6f} ℏ/2")
    
    # 步骤6：可视化结果
    print("\n步骤6：生成可视化图表")
    # 调用函数：visualize_probability_distribution()
    vis_result = visualize_probability_distribution(
        probabilities, 
        eigenvalues,
        title="Spin Operator Measurement Probabilities"
    )
    print(f"FUNCTION_CALL: visualize_probability_distribution | PARAMS: probabilities, eigenvalues | RESULT: {vis_result}")
    
    # 调用函数：visualize_quantum_state()
    state_vis_result = visualize_quantum_state(
        normalized_state,
        basis_labels=['|up⟩', '|down⟩'],
        title="Initial Spin State"
    )
    print(f"FUNCTION_CALL: visualize_quantum_state | PARAMS: normalized_state | RESULT: {state_vis_result}")
    
    # 验证答案
    print("\n" + "=" * 80)
    print("最终答案验证：")
    print(f"测量概率: P1 = {probabilities[0]:.2f}, P2 = {probabilities[1]:.2f}")
    print(f"期望值: ⟨A⟩ = {expectation_value:.6f} ℏ/2 = ℏ/{1/(2*expectation_value):.1f}")
    print(f"标准答案: 0.64, 0.36 and ℏ/7")
    print(f"FINAL_ANSWER: P1={probabilities[0]:.2f}, P2={probabilities[1]:.2f}, ⟨A⟩=ℏ/{1/(2*expectation_value):.1f}")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：三能级系统的算符测量")
    print("=" * 80)
    print("问题描述：")
    print("考虑三能级系统，初始态为 |ψ⟩ = (1+i)|0⟩ + 2|1⟩ + (1-i)|2⟩")
    print("算符 B 的矩阵元素：B_ij = ℏ (i≠j), 2ℏ (i=j)")
    print("求：1) 各本征态的测量概率  2) 算符B的期望值")
    print("-" * 80)
    
    # 步骤1：归一化三能级态
    print("\n步骤1：归一化三能级态")
    three_level_coeffs = [1+1j, 2+0j, 1-1j]
    # 调用函数：normalize_state_vector()
    norm3_result = normalize_state_vector(three_level_coeffs)
    normalized3_state = norm3_result['result']['normalized_coeffs']
    print(f"FUNCTION_CALL: normalize_state_vector | PARAMS: {three_level_coeffs} | RESULT: {norm3_result}")
    print(f"归一化态: {normalized3_state}")
    
    # 步骤2：构造三维算符
    print("\n步骤2：构造三维算符 B")
    matrix3_params = {
        'diagonal': 2 * HBAR,
        'off_diagonal': HBAR
    }
    # 调用函数：construct_operator_matrix()
    operator3_result = construct_operator_matrix(matrix3_params, dimension=3)
    operator3_matrix = operator3_result['result']
    print(f"FUNCTION_CALL: construct_operator_matrix | PARAMS: {matrix3_params} | RESULT: {operator3_result}")
    print(f"算符矩阵 B:")
    for row in operator3_matrix:
        print(f"  {row}")
    
    # 步骤3：求解本征系统
    print("\n步骤3：求解本征值和本征态")
    # 调用函数：solve_eigenvalue_problem()
    eigen3_result = solve_eigenvalue_problem(operator3_matrix)
    eigenvalues3 = eigen3_result['result']['eigenvalues']
    eigenvectors3 = eigen3_result['result']['eigenvectors']
    print(f"FUNCTION_CALL: solve_eigenvalue_problem | PARAMS: operator3_matrix | RESULT: {eigen3_result}")
    print(f"本征值: {eigenvalues3}")
    
    # 步骤4：计算测量概率
    print("\n步骤4：计算测量概率")
    # 调用函数：calculate_measurement_probabilities()
    prob3_result = calculate_measurement_probabilities(normalized3_state, eigenvectors3)
    probabilities3 = prob3_result['result']['probabilities']
    print(f"FUNCTION_CALL: calculate_measurement_probabilities | PARAMS: normalized3_state, eigenvectors3 | RESULT: {prob3_result}")
    for i, prob in enumerate(probabilities3):
        print(f"  P(λ{i}) = {prob:.6f}")
    
    # 步骤5：计算期望值
    print("\n步骤5：计算期望值")
    # 调用函数：calculate_expectation_value()
    expect3_result = calculate_expectation_value(normalized3_state, operator3_matrix)
    expectation3_value = expect3_result['result']
    print(f"FUNCTION_CALL: calculate_expectation_value | PARAMS: normalized3_state, operator3_matrix | RESULT: {expect3_result}")
    print(f"期望值 ⟨B⟩ = {expectation3_value:.6f} ℏ")
    
    print(f"\nFINAL_ANSWER: Probabilities={[f'{p:.4f}' for p in probabilities3]}, ⟨B⟩={expectation3_value:.4f}ℏ")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：态投影分析与完备性验证")
    print("=" * 80)
    print("问题描述：")
    print("对场景1的自旋态，分析其在本征态基底下的完整投影结构")
    print("验证：1) 投影系数  2) 概率归一化  3) 态重构误差")
    print("-" * 80)
    
    # 步骤1：执行态投影
    print("\n步骤1：将初始态投影到本征态基底")
    # 调用函数：project_state_to_eigenbasis()
    projection_result = project_state_to_eigenbasis(
        normalized_state,
        eigenvectors,
        eigenvalues
    )
    coefficients = projection_result['result']['coefficients']
    weighted_evs = projection_result['result']['weighted_eigenvalues']
    print(f"FUNCTION_CALL: project_state_to_eigenbasis | PARAMS: normalized_state, eigenvectors, eigenvalues | RESULT: {projection_result}")
    print(f"投影系数:")
    for i, coeff in enumerate(coefficients):
        print(f"  c{i} = {coeff}")
    print(f"重构误差: {projection_result['metadata']['reconstruction_error']:.2e}")
    
    # 步骤2：验证期望值计算的两种方法
    print("\n步骤2：验证期望值计算（两种方法）")
    print("方法1：直接计算 ⟨ψ|A|ψ⟩")
    expect_direct = expectation_value
    print(f"  结果: {expect_direct:.6f} ℏ/2")
    
    print("方法2：本征值加权求和 Σ P_i λ_i")
    expect_weighted = sum(weighted_evs)
    print(f"  结果: {expect_weighted:.6f} ℏ/2")
    print(f"  差异: {abs(expect_direct - expect_weighted):.2e}")
    
    # 步骤3：分析概率分布特征
    print("\n步骤3：分析概率分布特征")
    max_prob_idx = probabilities.index(max(probabilities))
    print(f"最大概率态: |ψ{max_prob_idx}⟩, P = {probabilities[max_prob_idx]:.6f}")
    print(f"对应本征值: λ = {eigenvalues[max_prob_idx]:.6f} ℏ/2")
    
    entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
    print(f"冯诺依曼熵: S = {entropy:.6f}")
    
    print(f"\nFINAL_ANSWER: Reconstruction_error={projection_result['metadata']['reconstruction_error']:.2e}, Entropy={entropy:.4f}")
    
    print("\n" + "=" * 80)
    print("所有场景执行完毕！")
    print("=" * 80)


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('./mid_result/quantum_mechanics', exist_ok=True)
    os.makedirs('./tool_images', exist_ok=True)
    
    # 执行主函数
    main()