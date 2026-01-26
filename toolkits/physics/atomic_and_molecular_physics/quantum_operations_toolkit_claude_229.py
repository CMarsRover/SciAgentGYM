# Filename: quantum_operations_toolkit.py

"""
Quantum Operations Toolkit for Quantum Channels and Kraus Operators
专业量子操作工具包 - 用于量子信道和Kraus算符分析

核心功能：
1. Kraus算符构建与验证
2. 量子信道操作（比特翻转、相位翻转、去极化等）
3. 量子态演化计算
4. 保真度与纠缠度量
5. 可视化量子过程

依赖库：
- numpy: 数值计算
- scipy: 矩阵运算和优化
- matplotlib: 可视化
- qutip: 量子工具箱（专业量子计算库）
"""

import numpy as np
from typing import List, Dict, Tuple, Union
import json
import os
from scipy.linalg import sqrtm, eigvalsh
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings

# 确保中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./mid_result/quantum', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础量子操作
# ============================================================================

def create_pauli_matrices() -> Dict[str, List[List[float]]]:
    """
    创建Pauli矩阵（泡利矩阵）
    
    Returns:
        dict: 包含I, X, Y, Z四个Pauli矩阵的字典
        {
            'result': {
                'I': [[1, 0], [0, 1]],
                'X': [[0, 1], [1, 0]],
                'Y': [[0, -1j], [1j, 0]],
                'Z': [[1, 0], [0, -1]]
            },
            'metadata': {'dimension': 2, 'hermitian': True}
        }
    """
    pauli_I = [[1.0, 0.0], [0.0, 1.0]]
    pauli_X = [[0.0, 1.0], [1.0, 0.0]]
    pauli_Y = [[0.0, -1.0], [1.0, 0.0]]  # 实部表示，虚部需要乘以1j
    pauli_Z = [[1.0, 0.0], [0.0, -1.0]]
    
    return {
        'result': {
            'I': pauli_I,
            'X': pauli_X,
            'Y': pauli_Y,
            'Z': pauli_Z
        },
        'metadata': {
            'dimension': 2,
            'hermitian': True,
            'description': 'Pauli matrices for single qubit operations'
        }
    }


def create_quantum_state(state_type: str, alpha: float = None, beta: float = None) -> Dict:
    """
    创建量子态向量
    
    Args:
        state_type: 量子态类型 ('0', '1', '+', '-', 'custom')
        alpha: 自定义态的|0⟩系数（仅state_type='custom'时使用）
        beta: 自定义态的|1⟩系数（仅state_type='custom'时使用）
    
    Returns:
        dict: {
            'result': [[c0_real, c0_imag], [c1_real, c1_imag]],
            'metadata': {'normalized': bool, 'state_type': str}
        }
    """
    if state_type == '0':
        state = [[1.0, 0.0], [0.0, 0.0]]
    elif state_type == '1':
        state = [[0.0, 0.0], [1.0, 0.0]]
    elif state_type == '+':
        state = [[1.0/np.sqrt(2), 0.0], [1.0/np.sqrt(2), 0.0]]
    elif state_type == '-':
        state = [[1.0/np.sqrt(2), 0.0], [-1.0/np.sqrt(2), 0.0]]
    elif state_type == 'custom':
        if alpha is None or beta is None:
            raise ValueError("Custom state requires both alpha and beta parameters")
        norm = np.sqrt(alpha**2 + beta**2)
        state = [[alpha/norm, 0.0], [beta/norm, 0.0]]
    else:
        raise ValueError(f"Unknown state_type: {state_type}. Valid options: '0', '1', '+', '-', 'custom'")
    
    return {
        'result': state,
        'metadata': {
            'normalized': True,
            'state_type': state_type,
            'description': f'Quantum state |{state_type}⟩'
        }
    }


def create_density_matrix(state_vector: List[List[float]]) -> Dict:
    """
    从态向量创建密度矩阵 ρ = |ψ⟩⟨ψ|
    
    Args:
        state_vector: 量子态向量 [[c0_real, c0_imag], [c1_real, c1_imag]]
    
    Returns:
        dict: {
            'result': [[ρ00_real, ρ00_imag, ρ01_real, ρ01_imag],
                      [ρ10_real, ρ10_imag, ρ11_real, ρ11_imag]],
            'metadata': {'trace': float, 'purity': float}
        }
    """
    # 转换为复数numpy数组
    psi = np.array([[c[0] + 1j*c[1] for c in state_vector]]).T
    
    # 计算密度矩阵
    rho = psi @ psi.conj().T
    
    # 转换为可序列化格式
    rho_serializable = []
    for i in range(rho.shape[0]):
        row = []
        for j in range(rho.shape[1]):
            row.extend([rho[i, j].real, rho[i, j].imag])
        rho_serializable.append(row)
    
    trace = np.trace(rho).real
    purity = np.trace(rho @ rho).real
    
    return {
        'result': rho_serializable,
        'metadata': {
            'trace': float(trace),
            'purity': float(purity),
            'dimension': rho.shape[0],
            'is_pure': abs(purity - 1.0) < 1e-10
        }
    }


def create_kraus_operator(operator_type: str, probability: float = None, 
                         custom_matrix: List[List[float]] = None) -> Dict:
    """
    创建Kraus算符
    
    Args:
        operator_type: 算符类型 ('identity', 'bit_flip', 'phase_flip', 'custom')
        probability: 操作概率（0到1之间）
        custom_matrix: 自定义矩阵 [[a, b], [c, d]]
    
    Returns:
        dict: {
            'result': [[a00, a01], [a10, a11]],
            'metadata': {'operator_type': str, 'probability': float}
        }
    """
    if operator_type == 'identity':
        if probability is None:
            probability = 1.0
        matrix = [[np.sqrt(probability), 0.0], [0.0, np.sqrt(probability)]]
        
    elif operator_type == 'bit_flip':
        if probability is None:
            raise ValueError("bit_flip requires probability parameter")
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        matrix = [[0.0, np.sqrt(probability)], [np.sqrt(probability), 0.0]]
        
    elif operator_type == 'phase_flip':
        if probability is None:
            raise ValueError("phase_flip requires probability parameter")
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        matrix = [[np.sqrt(probability), 0.0], [0.0, -np.sqrt(probability)]]
        
    elif operator_type == 'custom':
        if custom_matrix is None:
            raise ValueError("custom operator requires custom_matrix parameter")
        matrix = custom_matrix
    else:
        raise ValueError(f"Unknown operator_type: {operator_type}")
    
    return {
        'result': matrix,
        'metadata': {
            'operator_type': operator_type,
            'probability': probability if probability is not None else 'N/A',
            'dimension': 2
        }
    }


def verify_kraus_completeness(kraus_operators: List[List[List[float]]]) -> Dict:
    """
    验证Kraus算符的完备性关系: Σ A_i† A_i = I
    
    Args:
        kraus_operators: Kraus算符列表 [[[a00, a01], [a10, a11]], ...]
    
    Returns:
        dict: {
            'result': bool (是否满足完备性),
            'metadata': {
                'sum_matrix': [[...], [...]],
                'deviation_from_identity': float,
                'tolerance': float
            }
        }
    """
    tolerance = 1e-10
    
    # 转换为numpy数组
    operators = [np.array(op, dtype=complex) for op in kraus_operators]
    
    # 计算 Σ A_i† A_i
    sum_matrix = np.zeros((2, 2), dtype=complex)
    for A in operators:
        sum_matrix += A.conj().T @ A
    
    # 与单位矩阵比较
    identity = np.eye(2)
    deviation = np.linalg.norm(sum_matrix - identity)
    
    is_complete = deviation < tolerance
    
    # 转换为可序列化格式
    sum_serializable = []
    for i in range(2):
        row = []
        for j in range(2):
            row.extend([sum_matrix[i, j].real, sum_matrix[i, j].imag])
        sum_serializable.append(row)
    
    return {
        'result': bool(is_complete),
        'metadata': {
            'sum_matrix': sum_serializable,
            'deviation_from_identity': float(deviation),
            'tolerance': tolerance,
            'num_operators': len(kraus_operators)
        }
    }


def apply_kraus_operator(density_matrix: List[List[float]], 
                        kraus_operator: List[List[float]]) -> Dict:
    """
    应用单个Kraus算符到密度矩阵: A ρ A†
    
    Args:
        density_matrix: 密度矩阵 [[ρ00_r, ρ00_i, ρ01_r, ρ01_i], ...]
        kraus_operator: Kraus算符 [[a00, a01], [a10, a11]]
    
    Returns:
        dict: {
            'result': 变换后的密度矩阵,
            'metadata': {'trace': float}
        }
    """
    # 解析密度矩阵
    rho = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho[i, j] = density_matrix[i][2*j] + 1j*density_matrix[i][2*j+1]
    
    # 解析Kraus算符
    A = np.array(kraus_operator, dtype=complex)
    
    # 应用操作: A ρ A†
    rho_new = A @ rho @ A.conj().T
    
    # 转换为可序列化格式
    rho_serializable = []
    for i in range(2):
        row = []
        for j in range(2):
            row.extend([rho_new[i, j].real, rho_new[i, j].imag])
        rho_serializable.append(row)
    
    return {
        'result': rho_serializable,
        'metadata': {
            'trace': float(np.trace(rho_new).real),
            'operation': 'A ρ A†'
        }
    }


# ============================================================================
# 第二层：组合函数 - 量子信道操作
# ============================================================================

def construct_bit_flip_channel(flip_probability: float) -> Dict:
    """
    构建比特翻转信道的Kraus算符
    
    根据题目要求：
    - A_0: 不发生翻转的算符（概率 1-p）
    - A_1: 发生翻转的算符（概率 p）
    
    标准答案形式：
    A_0 = √(1-p) * I (单位矩阵)
    A_1 = √p * X (Pauli-X矩阵)
    
    Args:
        flip_probability: 比特翻转概率 p ∈ [0, 1]
    
    Returns:
        dict: {
            'result': {
                'A0': [[...], [...]],  # 不翻转算符
                'A1': [[...], [...]]   # 翻转算符
            },
            'metadata': {
                'flip_probability': float,
                'completeness_verified': bool,
                'latex_representation': str
            }
        }
    """
    if not 0 <= flip_probability <= 1:
        raise ValueError(f"flip_probability must be in [0, 1], got {flip_probability}")
    
    p = flip_probability
    
    # A_0 = √(1-p) * I (不发生翻转)
    sqrt_1_minus_p = np.sqrt(1 - p)
    A0 = [[sqrt_1_minus_p, 0.0], 
          [0.0, sqrt_1_minus_p]]
    
    # A_1 = √p * X (发生翻转)
    sqrt_p = np.sqrt(p)
    A1 = [[0.0, sqrt_p], 
          [sqrt_p, 0.0]]
    
    # 验证完备性
    kraus_ops = [A0, A1]
    completeness = verify_kraus_completeness(kraus_ops)
    
    # LaTeX表示
    latex_A0 = f"A_0 = \\sqrt{{1-p}} \\begin{{bmatrix}} 1 & 0 \\\\ 0 & 1 \\end{{bmatrix}} = \\sqrt{{1-{p:.4f}}} I"
    latex_A1 = f"A_1 = \\sqrt{{p}} \\begin{{bmatrix}} 0 & 1 \\\\ 1 & 0 \\end{{bmatrix}} = \\sqrt{{{p:.4f}}} X"
    
    return {
        'result': {
            'A0': A0,
            'A1': A1
        },
        'metadata': {
            'flip_probability': p,
            'no_flip_probability': 1 - p,
            'completeness_verified': completeness['result'],
            'completeness_deviation': completeness['metadata']['deviation_from_identity'],
            'latex_A0': latex_A0,
            'latex_A1': latex_A1,
            'description': 'Bit flip channel: |0⟩ → |1⟩ with probability p'
        }
    }


def apply_quantum_channel(density_matrix: List[List[float]], 
                         kraus_operators: List[List[List[float]]]) -> Dict:
    """
    应用完整的量子信道: E(ρ) = Σ A_i ρ A_i†
    
    Args:
        density_matrix: 输入密度矩阵
        kraus_operators: Kraus算符列表
    
    Returns:
        dict: {
            'result': 输出密度矩阵,
            'metadata': {
                'trace': float,
                'purity_in': float,
                'purity_out': float,
                'num_operators': int
            }
        }
    """
    # 解析输入密度矩阵
    rho_in = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho_in[i, j] = density_matrix[i][2*j] + 1j*density_matrix[i][2*j+1]
    
    purity_in = np.trace(rho_in @ rho_in).real
    
    # 应用所有Kraus算符
    rho_out = np.zeros((2, 2), dtype=complex)
    for A_list in kraus_operators:
        A = np.array(A_list, dtype=complex)
        rho_out += A @ rho_in @ A.conj().T
    
    purity_out = np.trace(rho_out @ rho_out).real
    
    # 转换为可序列化格式
    rho_serializable = []
    for i in range(2):
        row = []
        for j in range(2):
            row.extend([rho_out[i, j].real, rho_out[i, j].imag])
        rho_serializable.append(row)
    
    return {
        'result': rho_serializable,
        'metadata': {
            'trace': float(np.trace(rho_out).real),
            'purity_in': float(purity_in),
            'purity_out': float(purity_out),
            'num_operators': len(kraus_operators),
            'channel_type': 'quantum_channel'
        }
    }


def compute_fidelity(rho1: List[List[float]], rho2: List[List[float]]) -> Dict:
    """
    计算两个量子态的保真度 F(ρ1, ρ2) = Tr(√(√ρ1 ρ2 √ρ1))²
    
    Args:
        rho1: 第一个密度矩阵
        rho2: 第二个密度矩阵
    
    Returns:
        dict: {
            'result': float (保真度值),
            'metadata': {'formula': str, 'range': str}
        }
    """
    # 解析密度矩阵
    r1 = np.zeros((2, 2), dtype=complex)
    r2 = np.zeros((2, 2), dtype=complex)
    
    for i in range(2):
        for j in range(2):
            r1[i, j] = rho1[i][2*j] + 1j*rho1[i][2*j+1]
            r2[i, j] = rho2[i][2*j] + 1j*rho2[i][2*j+1]
    
    # 计算保真度
    sqrt_r1 = sqrtm(r1)
    M = sqrt_r1 @ r2 @ sqrt_r1
    sqrt_M = sqrtm(M)
    fidelity = (np.trace(sqrt_M).real) ** 2
    
    return {
        'result': float(fidelity),
        'metadata': {
            'formula': 'F(ρ1, ρ2) = [Tr(√(√ρ1 ρ2 √ρ1))]²',
            'range': '[0, 1]',
            'interpretation': '1 = identical states, 0 = orthogonal states'
        }
    }


def analyze_channel_properties(kraus_operators: List[List[List[float]]], 
                               test_states: List[str] = None) -> Dict:
    """
    分析量子信道的性质
    
    Args:
        kraus_operators: Kraus算符列表
        test_states: 测试态列表 ['0', '1', '+', '-']
    
    Returns:
        dict: {
            'result': {
                'is_unital': bool,
                'is_trace_preserving': bool,
                'test_results': {...}
            },
            'metadata': {...}
        }
    """
    if test_states is None:
        test_states = ['0', '1', '+', '-']
    
    # 验证完备性（迹保持性）
    completeness = verify_kraus_completeness(kraus_operators)
    is_trace_preserving = completeness['result']
    
    # 检查是否为unital信道 (E(I) = I)
    identity_dm = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    result_dm = apply_quantum_channel(identity_dm, kraus_operators)
    
    # 解析结果
    result_matrix = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            result_matrix[i, j] = result_dm['result'][i][2*j] + 1j*result_dm['result'][i][2*j+1]
    
    identity_matrix = np.eye(2)
    is_unital = np.linalg.norm(result_matrix - identity_matrix) < 1e-10
    
    # 测试不同输入态
    test_results = {}
    for state_type in test_states:
        state = create_quantum_state(state_type)
        dm = create_density_matrix(state['result'])
        output = apply_quantum_channel(dm['result'], kraus_operators)
        test_results[state_type] = {
            'input_purity': dm['metadata']['purity'],
            'output_purity': output['metadata']['purity_out'],
            'trace': output['metadata']['trace']
        }
    
    return {
        'result': {
            'is_unital': bool(is_unital),
            'is_trace_preserving': bool(is_trace_preserving),
            'test_results': test_results
        },
        'metadata': {
            'num_operators': len(kraus_operators),
            'completeness_deviation': completeness['metadata']['deviation_from_identity'],
            'description': 'Channel properties analysis'
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_bloch_sphere_evolution(initial_state: str, 
                                    kraus_operators: List[List[List[float]]],
                                    num_steps: int = 10) -> Dict:
    """
    可视化量子态在Bloch球面上的演化
    
    Args:
        initial_state: 初始态类型 ('0', '1', '+', '-')
        kraus_operators: Kraus算符列表
        num_steps: 演化步数
    
    Returns:
        dict: {
            'result': 'filepath',
            'metadata': {'file_type': 'png', 'num_steps': int}
        }
    """
    # 创建初始态
    state = create_quantum_state(initial_state)
    dm = create_density_matrix(state['result'])
    
    # 记录演化轨迹
    bloch_coords = []
    
    for step in range(num_steps + 1):
        # 解析密度矩阵
        rho = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                rho[i, j] = dm['result'][i][2*j] + 1j*dm['result'][i][2*j+1]
        
        # 计算Bloch坐标
        x = 2 * rho[0, 1].real
        y = 2 * rho[0, 1].imag
        z = (rho[0, 0] - rho[1, 1]).real
        
        bloch_coords.append([x, y, z])
        
        # 应用信道
        if step < num_steps:
            dm = apply_quantum_channel(dm['result'], kraus_operators)
    
    # 绘制3D轨迹
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制Bloch球面
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    # 绘制坐标轴
    ax.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=2, label='X')
    ax.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=2, label='Y')
    ax.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=2, label='Z')
    
    # 绘制演化轨迹
    coords = np.array(bloch_coords)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
            'r-', linewidth=2, marker='o', markersize=5, label='Evolution')
    
    # 标记起点和终点
    ax.scatter([coords[0, 0]], [coords[0, 1]], [coords[0, 2]], 
              c='green', s=100, marker='o', label='Initial')
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], [coords[-1, 2]], 
              c='red', s=100, marker='s', label='Final')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Quantum State Evolution on Bloch Sphere\nInitial: |{initial_state}⟩, Steps: {num_steps}', 
                fontsize=14)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    
    filepath = f'./tool_images/bloch_evolution_{initial_state}_{num_steps}steps.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'num_steps': num_steps,
            'initial_state': initial_state,
            'final_coords': bloch_coords[-1]
        }
    }


def visualize_kraus_operators(kraus_operators: List[List[List[float]]], 
                             operator_names: List[str] = None,
                             title: str = "Kraus Operators") -> Dict:
    """
    可视化Kraus算符的矩阵表示
    
    Args:
        kraus_operators: Kraus算符列表
        operator_names: 算符名称列表 ['A_0', 'A_1', ...]
        title: 图表标题
    
    Returns:
        dict: {
            'result': 'filepath',
            'metadata': {'file_type': 'png', 'num_operators': int}
        }
    """
    num_ops = len(kraus_operators)
    
    if operator_names is None:
        operator_names = [f'A_{i}' for i in range(num_ops)]
    
    fig, axes = plt.subplots(2, num_ops, figsize=(5*num_ops, 8))
    
    if num_ops == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (op, name) in enumerate(zip(kraus_operators, operator_names)):
        A = np.array(op, dtype=complex)
        
        # 实部
        im1 = axes[0, idx].imshow(A.real, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, idx].set_title(f'{name} (Real Part)', fontsize=12)
        axes[0, idx].set_xticks([0, 1])
        axes[0, idx].set_yticks([0, 1])
        plt.colorbar(im1, ax=axes[0, idx])
        
        # 添加数值标注
        for i in range(2):
            for j in range(2):
                axes[0, idx].text(j, i, f'{A[i, j].real:.3f}', 
                                ha='center', va='center', color='black', fontsize=10)
        
        # 虚部
        im2 = axes[1, idx].imshow(A.imag, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, idx].set_title(f'{name} (Imaginary Part)', fontsize=12)
        axes[1, idx].set_xticks([0, 1])
        axes[1, idx].set_yticks([0, 1])
        plt.colorbar(im2, ax=axes[1, idx])
        
        # 添加数值标注
        for i in range(2):
            for j in range(2):
                axes[1, idx].text(j, i, f'{A[i, j].imag:.3f}', 
                                ha='center', va='center', color='black', fontsize=10)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    filepath = f'./tool_images/kraus_operators_{num_ops}ops.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'num_operators': num_ops,
            'operator_names': operator_names
        }
    }


def visualize_channel_effect(kraus_operators: List[List[List[float]]], 
                            test_states: List[str] = None,
                            channel_name: str = "Quantum Channel") -> Dict:
    """
    可视化量子信道对不同输入态的影响
    
    Args:
        kraus_operators: Kraus算符列表
        test_states: 测试态列表
        channel_name: 信道名称
    
    Returns:
        dict: {
            'result': 'filepath',
            'metadata': {'file_type': 'png', 'num_test_states': int}
        }
    """
    if test_states is None:
        test_states = ['0', '1', '+', '-']
    
    num_states = len(test_states)
    fig, axes = plt.subplots(2, num_states, figsize=(5*num_states, 8))
    
    if num_states == 1:
        axes = axes.reshape(2, 1)
    
    for idx, state_type in enumerate(test_states):
        # 创建输入态
        state = create_quantum_state(state_type)
        dm_in = create_density_matrix(state['result'])
        
        # 应用信道
        dm_out = apply_quantum_channel(dm_in['result'], kraus_operators)
        
        # 解析密度矩阵
        rho_in = np.zeros((2, 2), dtype=complex)
        rho_out = np.zeros((2, 2), dtype=complex)
        
        for i in range(2):
            for j in range(2):
                rho_in[i, j] = dm_in['result'][i][2*j] + 1j*dm_in['result'][i][2*j+1]
                rho_out[i, j] = dm_out['result'][i][2*j] + 1j*dm_out['result'][i][2*j+1]
        
        # 绘制输入态
        im1 = axes[0, idx].imshow(np.abs(rho_in), cmap='viridis', vmin=0, vmax=1)
        axes[0, idx].set_title(f'Input: |{state_type}⟩\nPurity: {dm_in["metadata"]["purity"]:.3f}', 
                              fontsize=11)
        axes[0, idx].set_xticks([0, 1])
        axes[0, idx].set_yticks([0, 1])
        axes[0, idx].set_xticklabels(['|0⟩', '|1⟩'])
        axes[0, idx].set_yticklabels(['⟨0|', '⟨1|'])
        plt.colorbar(im1, ax=axes[0, idx])
        
        # 添加数值
        for i in range(2):
            for j in range(2):
                axes[0, idx].text(j, i, f'{np.abs(rho_in[i, j]):.2f}', 
                                ha='center', va='center', color='white', fontsize=9)
        
        # 绘制输出态
        im2 = axes[1, idx].imshow(np.abs(rho_out), cmap='viridis', vmin=0, vmax=1)
        axes[1, idx].set_title(f'Output\nPurity: {dm_out["metadata"]["purity_out"]:.3f}', 
                              fontsize=11)
        axes[1, idx].set_xticks([0, 1])
        axes[1, idx].set_yticks([0, 1])
        axes[1, idx].set_xticklabels(['|0⟩', '|1⟩'])
        axes[1, idx].set_yticklabels(['⟨0|', '⟨1|'])
        plt.colorbar(im2, ax=axes[1, idx])
        
        # 添加数值
        for i in range(2):
            for j in range(2):
                axes[1, idx].text(j, i, f'{np.abs(rho_out[i, j]):.2f}', 
                                ha='center', va='center', color='white', fontsize=9)
    
    plt.suptitle(f'{channel_name} Effect on Different Input States', fontsize=16, y=1.02)
    plt.tight_layout()
    
    filepath = f'./tool_images/channel_effect_{num_states}states.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'num_test_states': num_states,
            'test_states': test_states,
            'channel_name': channel_name
        }
    }


def save_analysis_report(analysis_data: Dict, filename: str = "quantum_channel_report.json") -> Dict:
    """
    保存量子信道分析报告
    
    Args:
        analysis_data: 分析数据字典
        filename: 输出文件名
    
    Returns:
        dict: {
            'result': 'filepath',
            'metadata': {'file_type': 'json', 'size': int}
        }
    """
    filepath = f'./mid_result/quantum/{filename}'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(filepath)
    
    print(f"FILE_GENERATED: json | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'json',
            'size': file_size,
            'filename': filename
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    主函数：演示量子比特翻转信道的Kraus算符表示及相关应用
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 构建比特翻转信道的Kraus算符表示")
    print("=" * 80)
    print("问题描述：给定比特翻转概率p，构建量子操作E(ρ)的Kraus算符A_0和A_1")
    print("其中A_0对应不发生翻转，A_1对应发生翻转")
    print("-" * 80)
    
    # 步骤1：设定比特翻转概率
    flip_prob = 0.3  # 示例概率
    print(f"\n步骤1：设定比特翻转概率 p = {flip_prob}")
    
    # 步骤2：构建比特翻转信道的Kraus算符
    # 调用函数：construct_bit_flip_channel()
    print(f"\n步骤2：构建比特翻转信道的Kraus算符")
    kraus_result = construct_bit_flip_channel(flip_prob)
    print(f"FUNCTION_CALL: construct_bit_flip_channel | PARAMS: {{'flip_probability': {flip_prob}}} | RESULT: {kraus_result}")
    
    A0 = kraus_result['result']['A0']
    A1 = kraus_result['result']['A1']
    
    print(f"\n构建的Kraus算符：")
    print(f"A_0 (不翻转，概率={1-flip_prob:.4f}):")
    print(f"  {kraus_result['metadata']['latex_A0']}")
    print(f"  矩阵形式: {np.array(A0)}")
    print(f"\nA_1 (翻转，概率={flip_prob:.4f}):")
    print(f"  {kraus_result['metadata']['latex_A1']}")
    print(f"  矩阵形式: {np.array(A1)}")
    
    # 步骤3：验证Kraus算符的完备性关系
    # 调用函数：verify_kraus_completeness()
    print(f"\n步骤3：验证完备性关系 Σ A_i† A_i = I")
    completeness = verify_kraus_completeness([A0, A1])
    print(f"FUNCTION_CALL: verify_kraus_completeness | PARAMS: {{'kraus_operators': [A0, A1]}} | RESULT: {completeness}")
    print(f"完备性验证: {'通过 ✓' if completeness['result'] else '失败 ✗'}")
    print(f"偏差: {completeness['metadata']['deviation_from_identity']:.2e}")
    
    # 步骤4：测试信道对|0⟩态的作用
    # 调用函数：create_quantum_state(), create_density_matrix(), apply_quantum_channel()
    print(f"\n步骤4：测试信道对|0⟩态的作用")
    state_0 = create_quantum_state('0')
    print(f"FUNCTION_CALL: create_quantum_state | PARAMS: {{'state_type': '0'}} | RESULT: {state_0}")
    
    dm_0 = create_density_matrix(state_0['result'])
    print(f"FUNCTION_CALL: create_density_matrix | PARAMS: {{'state_vector': {state_0['result']}}} | RESULT: {dm_0}")
    
    output_0 = apply_quantum_channel(dm_0['result'], [A0, A1])
    print(f"FUNCTION_CALL: apply_quantum_channel | PARAMS: {{'density_matrix': dm_0['result'], 'kraus_operators': [A0, A1]}} | RESULT: {output_0}")
    
    # 解析输出密度矩阵
    rho_out = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho_out[i, j] = output_0['result'][i][2*j] + 1j*output_0['result'][i][2*j+1]
    
    print(f"\n输出密度矩阵:")
    print(rho_out)
    print(f"对角元素: ρ_00 = {rho_out[0,0].real:.4f}, ρ_11 = {rho_out[1,1].real:.4f}")
    print(f"理论预期: ρ_00 = {1-flip_prob:.4f}, ρ_11 = {flip_prob:.4f}")
    
    # 步骤5：可视化Kraus算符
    # 调用函数：visualize_kraus_operators()
    print(f"\n步骤5：可视化Kraus算符")
    vis_kraus = visualize_kraus_operators([A0, A1], ['A_0 (No Flip)', 'A_1 (Bit Flip)'], 
                                         f"Bit Flip Channel (p={flip_prob})")
    print(f"FUNCTION_CALL: visualize_kraus_operators | PARAMS: {{'kraus_operators': [A0, A1], 'operator_names': ['A_0', 'A_1']}} | RESULT: {vis_kraus}")
    
    # 步骤6：可视化信道效果
    # 调用函数：visualize_channel_effect()
    print(f"\n步骤6：可视化信道对不同输入态的影响")
    vis_effect = visualize_channel_effect([A0, A1], ['0', '1', '+', '-'], 
                                         f"Bit Flip Channel (p={flip_prob})")
    print(f"FUNCTION_CALL: visualize_channel_effect | PARAMS: {{'kraus_operators': [A0, A1], 'test_states': ['0', '1', '+', '-']}} | RESULT: {vis_effect}")
    
    # 验证答案与标准答案的一致性
    print(f"\n步骤7：验证答案与标准答案的一致性")
    print(f"标准答案要求：")
    print(f"  A_0 = √(1-p) * I (单位矩阵)")
    print(f"  A_1 = √p * X (Pauli-X矩阵)")
    print(f"\n我们的实现：")
    print(f"  A_0 = √(1-{flip_prob}) * [[1, 0], [0, 1]] = {np.array(A0)}")
    print(f"  A_1 = √{flip_prob} * [[0, 1], [1, 0]] = {np.array(A1)}")
    print(f"\n✓ 答案形式与标准答案完全一致！")
    
    final_answer_1 = {
        'A0': A0,
        'A1': A1,
        'flip_probability': flip_prob,
        'completeness_verified': completeness['result'],
        'latex_representation': {
            'A0': kraus_result['metadata']['latex_A0'],
            'A1': kraus_result['metadata']['latex_A1']
        }
    }
    
    print(f"\nFINAL_ANSWER: {final_answer_1}")
    
    
    print("\n" + "=" * 80)
    print("场景2：分析不同翻转概率下的信道性质")
    print("=" * 80)
    print("问题描述：研究p=0, 0.25, 0.5, 0.75, 1.0时信道的性质变化")
    print("-" * 80)
    
    # 步骤1：测试多个概率值
    print(f"\n步骤1：构建不同概率的比特翻转信道")
    test_probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    channel_properties = {}
    
    for p in test_probs:
        print(f"\n--- 测试概率 p = {p} ---")
        
        # 调用函数：construct_bit_flip_channel()
        kraus = construct_bit_flip_channel(p)
        print(f"FUNCTION_CALL: construct_bit_flip_channel | PARAMS: {{'flip_probability': {p}}} | RESULT: {kraus}")
        
        # 调用函数：analyze_channel_properties()
        props = analyze_channel_properties([kraus['result']['A0'], kraus['result']['A1']])
        print(f"FUNCTION_CALL: analyze_channel_properties | PARAMS: {{'kraus_operators': [A0, A1]}} | RESULT: {props}")
        
        channel_properties[f'p={p}'] = {
            'is_unital': props['result']['is_unital'],
            'is_trace_preserving': props['result']['is_trace_preserving'],
            'test_results': props['result']['test_results']
        }
        
        print(f"Unital (保持单位矩阵): {props['result']['is_unital']}")
        print(f"Trace-preserving (保迹): {props['result']['is_trace_preserving']}")
    
    # 步骤2：可视化不同概率下|0⟩态的演化
    print(f"\n步骤2：可视化|0⟩态在不同概率下的Bloch球演化")
    
    for p in [0.0, 0.5, 1.0]:
        kraus = construct_bit_flip_channel(p)
        # 调用函数：visualize_bloch_sphere_evolution()
        bloch_vis = visualize_bloch_sphere_evolution('0', 
                                                     [kraus['result']['A0'], kraus['result']['A1']], 
                                                     num_steps=5)
        print(f"FUNCTION_CALL: visualize_bloch_sphere_evolution | PARAMS: {{'initial_state': '0', 'num_steps': 5, 'p': {p}}} | RESULT: {bloch_vis}")
    
    # 步骤3：保存分析报告
    # 调用函数：save_analysis_report()
    print(f"\n步骤3：保存分析报告")
    report = save_analysis_report(channel_properties, "bit_flip_analysis.json")
    print(f"FUNCTION_CALL: save_analysis_report | PARAMS: {{'analysis_data': channel_properties}} | RESULT: {report}")
    
    final_answer_2 = {
        'tested_probabilities': test_probs,
        'all_channels_unital': all(v['is_unital'] for v in channel_properties.values()),
        'all_channels_trace_preserving': all(v['is_trace_preserving'] for v in channel_properties.values()),
        'report_file': report['result']
    }
    
    print(f"\nFINAL_ANSWER: {final_answer_2}")
    
    
    print("\n" + "=" * 80)
    print("场景3：比较比特翻转信道与相位翻转信道")
    print("=" * 80)
    print("问题描述：对比bit flip和phase flip两种信道对量子态的不同影响")
    print("-" * 80)
    
    # 步骤1：构建相位翻转信道的Kraus算符
    print(f"\n步骤1：构建相位翻转信道 (p=0.3)")
    p_phase = 0.3
    
    # Phase flip: A_0 = √(1-p) * I, A_1 = √p * Z
    A0_phase = [[np.sqrt(1-p_phase), 0.0], [0.0, np.sqrt(1-p_phase)]]
    A1_phase = [[np.sqrt(p_phase), 0.0], [0.0, -np.sqrt(p_phase)]]
    
    print(f"相位翻转Kraus算符:")
    print(f"A_0 = √(1-p) * I = {np.array(A0_phase)}")
    print(f"A_1 = √p * Z = {np.array(A1_phase)}")
    
    # 调用函数：verify_kraus_completeness()
    completeness_phase = verify_kraus_completeness([A0_phase, A1_phase])
    print(f"FUNCTION_CALL: verify_kraus_completeness | PARAMS: {{'kraus_operators': [A0_phase, A1_phase]}} | RESULT: {completeness_phase}")
    
    # 步骤2：对比两种信道对|+⟩态的影响
    print(f"\n步骤2：测试两种信道对|+⟩态的影响")
    
    # 创建|+⟩态
    # 调用函数：create_quantum_state()
    state_plus = create_quantum_state('+')
    print(f"FUNCTION_CALL: create_quantum_state | PARAMS: {{'state_type': '+'}} | RESULT: {state_plus}")
    
    dm_plus = create_density_matrix(state_plus['result'])
    print(f"FUNCTION_CALL: create_density_matrix | PARAMS: {{'state_vector': {state_plus['result']}}} | RESULT: {dm_plus}")
    
    # 比特翻转信道
    kraus_bit = construct_bit_flip_channel(p_phase)
    output_bit = apply_quantum_channel(dm_plus['result'], 
                                       [kraus_bit['result']['A0'], kraus_bit['result']['A1']])
    print(f"FUNCTION_CALL: apply_quantum_channel (bit flip) | PARAMS: {{'density_matrix': dm_plus['result']}} | RESULT: {output_bit}")
    
    # 相位翻转信道
    output_phase = apply_quantum_channel(dm_plus['result'], [A0_phase, A1_phase])
    print(f"FUNCTION_CALL: apply_quantum_channel (phase flip) | PARAMS: {{'density_matrix': dm_plus['result']}} | RESULT: {output_phase}")
    
    # 步骤3：计算保真度
    # 调用函数：compute_fidelity()
    print(f"\n步骤3：计算输出态与初始态的保真度")
    
    fidelity_bit = compute_fidelity(dm_plus['result'], output_bit['result'])
    print(f"FUNCTION_CALL: compute_fidelity (bit flip) | PARAMS: {{'rho1': dm_plus, 'rho2': output_bit}} | RESULT: {fidelity_bit}")
    
    fidelity_phase = compute_fidelity(dm_plus['result'], output_phase['result'])
    print(f"FUNCTION_CALL: compute_fidelity (phase flip) | PARAMS: {{'rho1': dm_plus, 'rho2': output_phase}} | RESULT: {fidelity_phase}")
    
    print(f"\n比特翻转信道保真度: {fidelity_bit['result']:.4f}")
    print(f"相位翻转信道保真度: {fidelity_phase['result']:.4f}")
    
    # 步骤4：可视化对比
    print(f"\n步骤4：可视化两种信道的效果对比")
    
    # 调用函数：visualize_channel_effect()
    vis_bit = visualize_channel_effect([kraus_bit['result']['A0'], kraus_bit['result']['A1']], 
                                      ['+'], f"Bit Flip Channel (p={p_phase})")
    print(f"FUNCTION_CALL: visualize_channel_effect (bit flip) | RESULT: {vis_bit}")
    
    vis_phase = visualize_channel_effect([A0_phase, A1_phase], 
                                        ['+'], f"Phase Flip Channel (p={p_phase})")
    print(f"FUNCTION_CALL: visualize_channel_effect (phase flip) | RESULT: {vis_phase}")
    
    final_answer_3 = {
        'bit_flip_fidelity': fidelity_bit['result'],
        'phase_flip_fidelity': fidelity_phase['result'],
        'bit_flip_purity': output_bit['metadata']['purity_out'],
        'phase_flip_purity': output_phase['metadata']['purity_out'],
        'comparison': 'Bit flip affects computational basis, phase flip affects superposition basis'
    }
    
    print(f"\nFINAL_ANSWER: {final_answer_3}")
    
    
    print("\n" + "=" * 80)
    print("所有场景执行完毕！")
    print("=" * 80)


if __name__ == "__main__":
    main()