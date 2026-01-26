# Filename: quantum_mechanics_toolkit.py

"""
Quantum Mechanics Toolkit for Spin Systems
专业量子力学计算工具包 - 自旋系统

核心功能：
1. 量子态构建与操作
2. 密度矩阵计算
3. 算符期望值计算
4. 混合态分析

使用的专业库：
- qutip: 量子工具箱（Quantum Toolbox in Python）
- numpy: 数值计算
- scipy: 科学计算
- matplotlib: 可视化
"""

import numpy as np
from typing import Dict, List, Tuple, Union
import json
import os

import qutip as qt
QUTIP_AVAILABLE = True

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./mid_result/quantum', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础量子态和算符操作
# ============================================================================

def create_spin_state(coeff_up: float, coeff_down: float, 
                     normalize: bool = True) -> Dict:
    """
    创建自旋态的状态向量
    
    参数:
        coeff_up: 自旋向上态的系数（实数）
        coeff_down: 自旋向下态的系数（实数）
        normalize: 是否归一化
    
    返回:
        {'result': [c_up, c_down], 'metadata': {...}}
    """
    if not isinstance(coeff_up, (int, float)):
        raise TypeError(f"coeff_up must be numeric, got {type(coeff_up)}")
    if not isinstance(coeff_down, (int, float)):
        raise TypeError(f"coeff_down must be numeric, got {type(coeff_down)}")
    
    state = np.array([coeff_up, coeff_down], dtype=complex)
    
    if normalize:
        norm = np.sqrt(np.abs(coeff_up)**2 + np.abs(coeff_down)**2)
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero state")
        state = state / norm
    
    # 计算归一化检查
    norm_check = np.sum(np.abs(state)**2)
    
    return {
        'result': state.tolist(),
        'metadata': {
            'dimension': 2,
            'normalized': normalize,
            'norm': float(norm_check),
            'is_valid_state': abs(norm_check - 1.0) < 1e-6 if normalize else True
        }
    }


def get_pauli_matrix(matrix_type: str) -> Dict:
    """
    获取泡利矩阵
    
    参数:
        matrix_type: 'x', 'y', 'z' 或 'identity'
    
    返回:
        {'result': [[...], [...]], 'metadata': {...}}
    """
    valid_types = ['x', 'y', 'z', 'identity', 'I']
    if matrix_type not in valid_types:
        raise ValueError(f"matrix_type must be one of {valid_types}, got {matrix_type}")
    
    pauli_matrices = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
        'identity': np.array([[1, 0], [0, 1]], dtype=complex),
        'I': np.array([[1, 0], [0, 1]], dtype=complex)
    }
    
    matrix = pauli_matrices[matrix_type]
    
    # 检查厄米性
    is_hermitian = np.allclose(matrix, matrix.conj().T)
    
    # 计算本征值
    eigenvalues = np.linalg.eigvalsh(matrix.real if matrix_type != 'y' else matrix)
    
    return {
        'result': matrix.tolist(),
        'metadata': {
            'type': matrix_type,
            'dimension': '2x2',
            'is_hermitian': bool(is_hermitian),
            'eigenvalues': eigenvalues.tolist(),
            'trace': float(np.trace(matrix).real)
        }
    }


def compute_density_matrix(state_vector: List[complex], 
                          is_pure_state: bool = True) -> Dict:
    """
    从状态向量计算密度矩阵
    
    参数:
        state_vector: 状态向量 [c1, c2, ...]
        is_pure_state: 是否为纯态
    
    返回:
        {'result': [[...], [...]], 'metadata': {...}}
    """
    if not isinstance(state_vector, list):
        raise TypeError("state_vector must be a list")
    
    state = np.array(state_vector, dtype=complex)
    
    if len(state.shape) != 1:
        raise ValueError("state_vector must be 1-dimensional")
    
    # 密度矩阵 ρ = |ψ⟩⟨ψ|
    rho = np.outer(state, state.conj())
    
    # 计算密度矩阵性质
    trace = np.trace(rho)
    purity = np.trace(rho @ rho).real
    
    return {
        'result': rho.tolist(),
        'metadata': {
            'dimension': f'{len(state)}x{len(state)}',
            'trace': float(trace.real),
            'purity': float(purity),
            'is_pure': bool(abs(purity - 1.0) < 1e-6),
            'is_hermitian': bool(np.allclose(rho, rho.conj().T))
        }
    }


def compute_expectation_value(density_matrix: List[List[complex]], 
                             operator: List[List[complex]]) -> Dict:
    """
    计算算符在给定密度矩阵下的期望值
    
    参数:
        density_matrix: 密度矩阵 [[...], [...]]
        operator: 算符矩阵 [[...], [...]]
    
    返回:
        {'result': float, 'metadata': {...}}
    """
    if not isinstance(density_matrix, list) or not isinstance(operator, list):
        raise TypeError("Both inputs must be lists")
    
    rho = np.array(density_matrix, dtype=complex)
    op = np.array(operator, dtype=complex)
    
    if rho.shape != op.shape:
        raise ValueError(f"Matrix dimensions must match: {rho.shape} vs {op.shape}")
    
    # 期望值 ⟨A⟩ = Tr(ρA)
    expectation = np.trace(rho @ op)
    
    # 检查结果是否为实数（对于厄米算符应该是）
    is_real = abs(expectation.imag) < 1e-10
    
    return {
        'result': float(expectation.real),
        'metadata': {
            'complex_value': complex(expectation),
            'is_real': bool(is_real),
            'imaginary_part': float(expectation.imag),
            'operator_trace': float(np.trace(op).real)
        }
    }


# ============================================================================
# 第二层：组合函数 - 混合态和复合算符
# ============================================================================

def create_mixed_state_density_matrix(states: List[Dict], 
                                     probabilities: List[float]) -> Dict:
    """
    创建混合态的密度矩阵
    
    参数:
        states: 状态向量列表，每个元素是 {'state': [c1, c2], ...}
        probabilities: 对应的概率列表
    
    返回:
        {'result': [[...], [...]], 'metadata': {...}}
    """
    if not isinstance(states, list) or not isinstance(probabilities, list):
        raise TypeError("states and probabilities must be lists")
    
    if len(states) != len(probabilities):
        raise ValueError(f"Number of states ({len(states)}) must match number of probabilities ({len(probabilities)})")
    
    # 检查概率归一化
    prob_sum = sum(probabilities)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")
    
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("All probabilities must be between 0 and 1")
    
    # 初始化混合态密度矩阵
    dim = len(states[0]['state'])
    rho_mixed = np.zeros((dim, dim), dtype=complex)
    
    individual_purities = []
    
    # 计算混合态密度矩阵: ρ = Σ p_i |ψ_i⟩⟨ψ_i|
    for i, (state_dict, prob) in enumerate(zip(states, probabilities)):
        state = np.array(state_dict['state'], dtype=complex)
        rho_i = np.outer(state, state.conj())
        rho_mixed += prob * rho_i
        
        purity_i = np.trace(rho_i @ rho_i).real
        individual_purities.append(float(purity_i))
    
    # 计算混合态的纯度
    purity_mixed = np.trace(rho_mixed @ rho_mixed).real
    
    return {
        'result': rho_mixed.tolist(),
        'metadata': {
            'num_states': len(states),
            'probabilities': probabilities,
            'individual_purities': individual_purities,
            'mixed_purity': float(purity_mixed),
            'is_pure': bool(abs(purity_mixed - 1.0) < 1e-6),
            'trace': float(np.trace(rho_mixed).real)
        }
    }


def create_linear_combination_operator(operators: List[Dict], 
                                      coefficients: List[float]) -> Dict:
    """
    创建算符的线性组合
    
    参数:
        operators: 算符列表，每个元素是 {'matrix': [[...], [...]]}
        coefficients: 对应的系数列表
    
    返回:
        {'result': [[...], [...]], 'metadata': {...}}
    """
    if not isinstance(operators, list) or not isinstance(coefficients, list):
        raise TypeError("operators and coefficients must be lists")
    
    if len(operators) != len(coefficients):
        raise ValueError(f"Number of operators ({len(operators)}) must match number of coefficients ({len(coefficients)})")
    
    # 初始化组合算符
    first_op = np.array(operators[0]['matrix'], dtype=complex)
    combined_op = np.zeros_like(first_op)
    
    operator_info = []
    
    # 计算线性组合: A = Σ c_i A_i
    for i, (op_dict, coeff) in enumerate(zip(operators, coefficients)):
        op_matrix = np.array(op_dict['matrix'], dtype=complex)
        
        if op_matrix.shape != first_op.shape:
            raise ValueError(f"All operators must have the same shape")
        
        combined_op += coeff * op_matrix
        
        operator_info.append({
            'index': i,
            'coefficient': float(coeff),
            'operator_type': op_dict.get('type', 'unknown'),
            'trace': float(np.trace(op_matrix).real)
        })
    
    # 检查组合算符的性质
    is_hermitian = np.allclose(combined_op, combined_op.conj().T)
    
    return {
        'result': combined_op.tolist(),
        'metadata': {
            'num_operators': len(operators),
            'coefficients': coefficients,
            'operator_info': operator_info,
            'is_hermitian': bool(is_hermitian),
            'trace': float(np.trace(combined_op).real)
        }
    }


def analyze_quantum_state(state_vector: List[complex]) -> Dict:
    """
    分析量子态的各种性质
    
    参数:
        state_vector: 状态向量
    
    返回:
        {'result': {...}, 'metadata': {...}}
    """
    state = np.array(state_vector, dtype=complex)
    
    # 归一化检查
    norm = np.sqrt(np.sum(np.abs(state)**2))
    
    # 计算各个基态的概率
    probabilities = np.abs(state)**2
    
    # 计算相位
    phases = np.angle(state)
    
    # 计算密度矩阵
    rho = np.outer(state, state.conj())
    purity = np.trace(rho @ rho).real
    
    # 计算冯诺依曼熵（对于纯态应该为0）
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # 过滤数值误差
    von_neumann_entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
    
    analysis = {
        'norm': float(norm),
        'is_normalized': bool(abs(norm - 1.0) < 1e-6),
        'basis_probabilities': probabilities.tolist(),
        'phases_radians': phases.tolist(),
        'phases_degrees': (phases * 180 / np.pi).tolist(),
        'purity': float(purity),
        'von_neumann_entropy': float(von_neumann_entropy)
    }
    
    return {
        'result': analysis,
        'metadata': {
            'state_dimension': len(state),
            'is_pure_state': bool(abs(purity - 1.0) < 1e-6)
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_bloch_sphere(state_vector: List[complex], 
                          title: str = "Bloch Sphere Representation",
                          save_path: str = None) -> Dict:
    """
    在布洛赫球上可视化自旋-1/2态
    
    参数:
        state_vector: 状态向量 [c_up, c_down]
        title: 图表标题
        save_path: 保存路径（可选）
    
    返回:
        {'result': 'file_path', 'metadata': {...}}
    """
    state = np.array(state_vector, dtype=complex)
    
    if len(state) != 2:
        raise ValueError("Bloch sphere visualization only supports spin-1/2 (2D) states")
    
    # 归一化
    state = state / np.linalg.norm(state)
    
    # 计算布洛赫向量坐标
    # |ψ⟩ = cos(θ/2)|↑⟩ + e^(iφ)sin(θ/2)|↓⟩
    c_up, c_down = state[0], state[1]
    
    # 计算θ和φ
    theta = 2 * np.arccos(np.abs(c_up))
    if abs(c_down) > 1e-10:
        phi = np.angle(c_down / c_up) if abs(c_up) > 1e-10 else np.angle(c_down)
    else:
        phi = 0
    
    # 布洛赫向量
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制布洛赫球
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    # 绘制坐标轴
    ax.plot([0, 1.2], [0, 0], [0, 0], 'r-', linewidth=2, label='X')
    ax.plot([0, 0], [0, 1.2], [0, 0], 'g-', linewidth=2, label='Y')
    ax.plot([0, 0], [0, 0], [0, 1.2], 'b-', linewidth=2, label='Z')
    
    # 绘制状态向量
    ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
    
    # 标注
    ax.text(1.3, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.3, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.3, 'Z (|↑⟩)', fontsize=12)
    ax.text(0, 0, -1.3, '-Z (|↓⟩)', fontsize=12)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, pad=20)
    
    # 设置相同的刻度范围
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    
    if save_path is None:
        save_path = './tool_images/bloch_sphere.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'bloch_vector': [float(x), float(y), float(z)],
            'theta_radians': float(theta),
            'phi_radians': float(phi),
            'theta_degrees': float(theta * 180 / np.pi),
            'phi_degrees': float(phi * 180 / np.pi)
        }
    }


def visualize_density_matrix(density_matrix: List[List[complex]], 
                            title: str = "Density Matrix",
                            save_path: str = None) -> Dict:
    """
    可视化密度矩阵（实部和虚部）
    
    参数:
        density_matrix: 密度矩阵
        title: 图表标题
        save_path: 保存路径
    
    返回:
        {'result': 'file_path', 'metadata': {...}}
    """
    rho = np.array(density_matrix, dtype=complex)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 实部
    im1 = axes[0].imshow(rho.real, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title(f'{title} - Real Part', fontsize=12)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])
    
    # 在每个格子中显示数值
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            text = axes[0].text(j, i, f'{rho[i, j].real:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    # 虚部
    im2 = axes[1].imshow(rho.imag, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title(f'{title} - Imaginary Part', fontsize=12)
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1])
    
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            text = axes[1].text(j, i, f'{rho[i, j].imag:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    if save_path is None:
        save_path = './tool_images/density_matrix.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'matrix_shape': list(rho.shape),
            'trace': float(np.trace(rho).real),
            'max_real': float(np.max(rho.real)),
            'max_imag': float(np.max(np.abs(rho.imag)))
        }
    }


def visualize_expectation_values(operators: List[str], 
                                 expectation_values: List[float],
                                 title: str = "Expectation Values",
                                 save_path: str = None) -> Dict:
    """
    可视化多个算符的期望值
    
    参数:
        operators: 算符名称列表
        expectation_values: 对应的期望值列表
        title: 图表标题
        save_path: 保存路径
    
    返回:
        {'result': 'file_path', 'metadata': {...}}
    """
    if len(operators) != len(expectation_values):
        raise ValueError("Number of operators must match number of expectation values")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(operators, expectation_values, 
                  color=colors[:len(operators)], alpha=0.7, edgecolor='black')
    
    # 在柱子上显示数值
    for bar, val in zip(bars, expectation_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Expectation Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    if save_path is None:
        save_path = './tool_images/expectation_values.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'num_operators': len(operators),
            'mean_expectation': float(np.mean(expectation_values)),
            'std_expectation': float(np.std(expectation_values)),
            'min_expectation': float(np.min(expectation_values)),
            'max_expectation': float(np.max(expectation_values))
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示量子力学计算工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始混合态自旋系统问题")
    print("=" * 80)
    print("问题描述：计算混合态系统中算符 10σz + 5σx 的期望值")
    print("系统由两个纯态组成：")
    print("  |ψ1⟩ = (1/2)|↑⟩ + (√3/2)|↓⟩，概率 p1 = 1/3")
    print("  |ψ2⟩ = (1/√2)|↑⟩ + (1/√2)|↓⟩，概率 p2 = 2/3")
    print("-" * 80)
    
    # 步骤1：创建第一个纯态 |ψ1⟩
    print("\n步骤1：创建纯态 |ψ1⟩ = (1/2)|↑⟩ + (√3/2)|↓⟩")
    # 调用函数：create_spin_state()
    psi1_result = create_spin_state(
        coeff_up=0.5,
        coeff_down=np.sqrt(3)/2,
        normalize=True
    )
    psi1 = psi1_result['result']
    print(f"FUNCTION_CALL: create_spin_state | PARAMS: {{coeff_up: 0.5, coeff_down: {np.sqrt(3)/2:.6f}}} | RESULT: {psi1_result}")
    
    # 步骤2：创建第二个纯态 |ψ2⟩
    print("\n步骤2：创建纯态 |ψ2⟩ = (1/√2)|↑⟩ + (1/√2)|↓⟩")
    # 调用函数：create_spin_state()
    psi2_result = create_spin_state(
        coeff_up=1/np.sqrt(2),
        coeff_down=1/np.sqrt(2),
        normalize=True
    )
    psi2 = psi2_result['result']
    print(f"FUNCTION_CALL: create_spin_state | PARAMS: {{coeff_up: {1/np.sqrt(2):.6f}, coeff_down: {1/np.sqrt(2):.6f}}} | RESULT: {psi2_result}")
    
    # 步骤3：分析两个纯态的性质
    print("\n步骤3：分析 |ψ1⟩ 的量子态性质")
    # 调用函数：analyze_quantum_state()
    psi1_analysis = analyze_quantum_state(psi1)
    print(f"FUNCTION_CALL: analyze_quantum_state | PARAMS: {{state_vector: {psi1}}} | RESULT: {psi1_analysis}")
    
    print("\n步骤4：分析 |ψ2⟩ 的量子态性质")
    # 调用函数：analyze_quantum_state()
    psi2_analysis = analyze_quantum_state(psi2)
    print(f"FUNCTION_CALL: analyze_quantum_state | PARAMS: {{state_vector: {psi2}}} | RESULT: {psi2_analysis}")
    
    # 步骤5：创建混合态密度矩阵
    print("\n步骤5：创建混合态密度矩阵 ρ = (1/3)|ψ1⟩⟨ψ1| + (2/3)|ψ2⟩⟨ψ2|")
    # 调用函数：create_mixed_state_density_matrix()
    states_list = [
        {'state': psi1},
        {'state': psi2}
    ]
    probabilities = [1/3, 2/3]
    rho_mixed_result = create_mixed_state_density_matrix(states_list, probabilities)
    rho_mixed = rho_mixed_result['result']
    print(f"FUNCTION_CALL: create_mixed_state_density_matrix | PARAMS: {{states: {states_list}, probabilities: {probabilities}}} | RESULT: {rho_mixed_result}")
    
    # 步骤6：获取泡利矩阵 σz
    print("\n步骤6：获取泡利矩阵 σz")
    # 调用函数：get_pauli_matrix()
    sigma_z_result = get_pauli_matrix('z')
    sigma_z = sigma_z_result['result']
    print(f"FUNCTION_CALL: get_pauli_matrix | PARAMS: {{matrix_type: 'z'}} | RESULT: {sigma_z_result}")
    
    # 步骤7：获取泡利矩阵 σx
    print("\n步骤7：获取泡利矩阵 σx")
    # 调用函数：get_pauli_matrix()
    sigma_x_result = get_pauli_matrix('x')
    sigma_x = sigma_x_result['result']
    print(f"FUNCTION_CALL: get_pauli_matrix | PARAMS: {{matrix_type: 'x'}} | RESULT: {sigma_x_result}")
    
    # 步骤8：创建组合算符 10σz + 5σx
    print("\n步骤8：创建组合算符 A = 10σz + 5σx")
    # 调用函数：create_linear_combination_operator()
    operators_list = [
        {'matrix': sigma_z, 'type': 'sigma_z'},
        {'matrix': sigma_x, 'type': 'sigma_x'}
    ]
    coefficients = [10, 5]
    combined_op_result = create_linear_combination_operator(operators_list, coefficients)
    combined_op = combined_op_result['result']
    print(f"FUNCTION_CALL: create_linear_combination_operator | PARAMS: {{operators: {operators_list}, coefficients: {coefficients}}} | RESULT: {combined_op_result}")
    
    # 步骤9：计算期望值 ⟨A⟩ = Tr(ρA)
    print("\n步骤9：计算算符 A 在混合态下的期望值")
    # 调用函数：compute_expectation_value()
    expectation_result = compute_expectation_value(rho_mixed, combined_op)
    expectation_value = expectation_result['result']
    print(f"FUNCTION_CALL: compute_expectation_value | PARAMS: {{density_matrix: {rho_mixed}, operator: {combined_op}}} | RESULT: {expectation_result}")
    
    # 步骤10：可视化密度矩阵
    print("\n步骤10：可视化混合态密度矩阵")
    # 调用函数：visualize_density_matrix()
    vis_rho_result = visualize_density_matrix(
        rho_mixed,
        title="Mixed State Density Matrix",
        save_path='./tool_images/scenario1_density_matrix.png'
    )
    print(f"FUNCTION_CALL: visualize_density_matrix | PARAMS: {{density_matrix: ..., title: 'Mixed State Density Matrix'}} | RESULT: {vis_rho_result}")
    
    # 步骤11：可视化两个纯态在布洛赫球上
    print("\n步骤11：可视化 |ψ1⟩ 在布洛赫球上")
    # 调用函数：visualize_bloch_sphere()
    bloch1_result = visualize_bloch_sphere(
        psi1,
        title="State |ψ1⟩ on Bloch Sphere",
        save_path='./tool_images/scenario1_bloch_psi1.png'
    )
    print(f"FUNCTION_CALL: visualize_bloch_sphere | PARAMS: {{state_vector: {psi1}, title: 'State |ψ1⟩'}} | RESULT: {bloch1_result}")
    
    print("\n步骤12：可视化 |ψ2⟩ 在布洛赫球上")
    # 调用函数：visualize_bloch_sphere()
    bloch2_result = visualize_bloch_sphere(
        psi2,
        title="State |ψ2⟩ on Bloch Sphere",
        save_path='./tool_images/scenario1_bloch_psi2.png'
    )
    print(f"FUNCTION_CALL: visualize_bloch_sphere | PARAMS: {{state_vector: {psi2}, title: 'State |ψ2⟩'}} | RESULT: {bloch2_result}")
    
    # 最终答案
    final_answer = round(expectation_value, 2)
    print("\n" + "=" * 80)
    print(f"场景1最终结果：算符 10σz + 5σx 的期望值 = {final_answer}")
    print(f"FINAL_ANSWER: {final_answer}")
    print("=" * 80)
    
    
    # ========================================================================
    print("\n\n")
    print("=" * 80)
    print("场景2：不同混合态概率下的期望值变化分析")
    print("=" * 80)
    print("问题描述：研究当两个纯态的混合概率变化时，期望值如何变化")
    print("固定 |ψ1⟩ 和 |ψ2⟩，改变概率 p1 从 0 到 1")
    print("-" * 80)
    
    # 步骤1：设置概率扫描范围
    print("\n步骤1：设置概率扫描参数")
    p1_values = np.linspace(0, 1, 21)  # 从0到1，21个点
    expectation_values_scan = []
    
    print(f"概率 p1 扫描范围: {p1_values[0]:.2f} 到 {p1_values[-1]:.2f}，共 {len(p1_values)} 个点")
    
    # 步骤2：对每个概率值计算期望值
    print("\n步骤2：计算不同概率下的期望值")
    for p1 in p1_values:
        p2 = 1 - p1
        
        # 调用函数：create_mixed_state_density_matrix()
        rho_scan_result = create_mixed_state_density_matrix(
            states_list,
            [float(p1), float(p2)]
        )
        rho_scan = rho_scan_result['result']
        
        # 调用函数：compute_expectation_value()
        exp_scan_result = compute_expectation_value(rho_scan, combined_op)
        expectation_values_scan.append(exp_scan_result['result'])
    
    print(f"FUNCTION_CALL: create_mixed_state_density_matrix (循环21次) | 计算完成")
    print(f"期望值范围: [{min(expectation_values_scan):.4f}, {max(expectation_values_scan):.4f}]")
    
    # 步骤3：可视化期望值随概率的变化
    print("\n步骤3：绘制期望值随概率变化的曲线")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p1_values, expectation_values_scan, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=final_answer, color='r', linestyle='--', linewidth=2, 
               label=f'Original (p1=1/3): {final_answer:.2f}')
    ax.axvline(x=1/3, color='g', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Probability p1 for |ψ1⟩', fontsize=12)
    ax.set_ylabel('Expectation Value ⟨10σz + 5σx⟩', fontsize=12)
    ax.set_title('Expectation Value vs Mixing Probability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    save_path_scan = './tool_images/scenario2_probability_scan.png'
    plt.tight_layout()
    plt.savefig(save_path_scan, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {save_path_scan}")
    
    # 找到最大和最小期望值
    max_exp_idx = np.argmax(expectation_values_scan)
    min_exp_idx = np.argmin(expectation_values_scan)
    
    print(f"\n最大期望值: {expectation_values_scan[max_exp_idx]:.4f} (p1 = {p1_values[max_exp_idx]:.2f})")
    print(f"最小期望值: {expectation_values_scan[min_exp_idx]:.4f} (p1 = {p1_values[min_exp_idx]:.2f})")
    
    scenario2_answer = f"期望值范围: [{expectation_values_scan[min_exp_idx]:.2f}, {expectation_values_scan[max_exp_idx]:.2f}]"
    print(f"\nFINAL_ANSWER: {scenario2_answer}")
    
    
    # ========================================================================
    print("\n\n")
    print("=" * 80)
    print("场景3：不同算符组合的期望值比较")
    print("=" * 80)
    print("问题描述：在同一混合态下，比较不同算符组合的期望值")
    print("计算以下算符的期望值：σx, σy, σz, 10σz, 5σx, 10σz+5σx")
    print("-" * 80)
    
    # 步骤1：获取所有泡利矩阵
    print("\n步骤1：获取泡利矩阵 σy")
    # 调用函数：get_pauli_matrix()
    sigma_y_result = get_pauli_matrix('y')
    sigma_y = sigma_y_result['result']
    print(f"FUNCTION_CALL: get_pauli_matrix | PARAMS: {{matrix_type: 'y'}} | RESULT: {sigma_y_result}")
    
    # 步骤2：定义要测试的算符列表
    print("\n步骤2：定义算符组合列表")
    test_operators = [
        {'name': 'σx', 'matrix': sigma_x},
        {'name': 'σy', 'matrix': sigma_y},
        {'name': 'σz', 'matrix': sigma_z},
        {'name': '10σz', 'matrix': create_linear_combination_operator(
            [{'matrix': sigma_z, 'type': 'sigma_z'}], [10])['result']},
        {'name': '5σx', 'matrix': create_linear_combination_operator(
            [{'matrix': sigma_x, 'type': 'sigma_x'}], [5])['result']},
        {'name': '10σz+5σx', 'matrix': combined_op}
    ]
    
    # 步骤3：计算每个算符的期望值
    print("\n步骤3：计算各算符的期望值")
    operator_names = []
    operator_expectations = []
    
    for op_dict in test_operators:
        # 调用函数：compute_expectation_value()
        exp_result = compute_expectation_value(rho_mixed, op_dict['matrix'])
        operator_names.append(op_dict['name'])
        operator_expectations.append(exp_result['result'])
        print(f"FUNCTION_CALL: compute_expectation_value | PARAMS: {{operator: '{op_dict['name']}'}} | RESULT: {exp_result['result']:.4f}")
    
    # 步骤4：可视化比较
    print("\n步骤4：可视化不同算符的期望值比较")
    # 调用函数：visualize_expectation_values()
    vis_exp_result = visualize_expectation_values(
        operator_names,
        operator_expectations,
        title="Expectation Values for Different Operators",
        save_path='./tool_images/scenario3_operator_comparison.png'
    )
    print(f"FUNCTION_CALL: visualize_expectation_values | PARAMS: {{operators: {operator_names}}} | RESULT: {vis_exp_result}")
    
    # 步骤5：创建详细的比较表格
    print("\n步骤5：生成算符期望值比较表")
    print("-" * 60)
    print(f"{'算符':<15} {'期望值':<15} {'相对于10σz+5σx':<20}")
    print("-" * 60)
    reference_value = operator_expectations[-1]  # 10σz+5σx的值
    for name, exp_val in zip(operator_names, operator_expectations):
        ratio = (exp_val / reference_value * 100) if reference_value != 0 else 0
        print(f"{name:<15} {exp_val:<15.4f} {ratio:<20.2f}%")
    print("-" * 60)
    
    # 保存结果到文件
    results_dict = {
        'operators': operator_names,
        'expectation_values': operator_expectations,
        'mixed_state_probabilities': probabilities,
        'reference_operator': '10σz+5σx',
        'reference_value': reference_value
    }
    
    result_file = './mid_result/quantum/scenario3_operator_comparison.json'
    with open(result_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n结果已保存到: {result_file}")
    
    scenario3_answer = f"目标算符 10σz+5σx 的期望值为 {reference_value:.2f}，在所有测试算符中排名第{sorted(operator_expectations, reverse=True).index(reference_value) + 1}"
    print(f"\nFINAL_ANSWER: {scenario3_answer}")
    
    print("\n" + "=" * 80)
    print("所有场景计算完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()