# Filename: quantum_mechanics_toolkit.py

"""
Quantum Mechanics Computational Toolkit
========================================
专业量子力学计算工具包，用于处理量子态、算符本征值问题和测量概率计算

核心功能：
1. 厄米算符本征值分解与简并度分析
2. 量子态在本征子空间的投影
3. 测量概率计算（包括简并情况）
4. 量子态可视化

依赖库：
- numpy: 数值计算和线性代数
- scipy.linalg: 高级线性代数运算
- matplotlib: 可视化
- sympy: 符号计算验证
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
import os
from collections import defaultdict

# 配置matplotlib支持中英文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局常量
TOLERANCE = 1e-10  # 数值容差
MID_RESULT_DIR = "./mid_result/quantum_mechanics"
IMAGE_DIR = "./tool_images"

# 创建必要的目录
os.makedirs(MID_RESULT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================================
# 第一层：原子函数（基础量子力学计算）
# ============================================================================

def create_quantum_state(components: List[float]) -> Dict[str, Any]:
    """
    创建归一化的量子态向量
    
    参数:
        components: 量子态的分量列表 [c1, c2, ..., cn]
    
    返回:
        {
            'result': 归一化后的量子态列表,
            'metadata': {
                'dimension': 希尔伯特空间维度,
                'norm': 原始态的模,
                'is_normalized': 是否已归一化
            }
        }
    """
    if not components or not isinstance(components, list):
        raise ValueError("components必须是非空列表")
    
    if not all(isinstance(c, (int, float, complex)) for c in components):
        raise ValueError("所有分量必须是数值类型")
    
    # 转换为numpy数组进行计算
    state_vector = np.array(components, dtype=complex)
    norm = np.linalg.norm(state_vector)
    
    if norm < TOLERANCE:
        raise ValueError("量子态的模不能为零")
    
    # 归一化
    normalized_state = state_vector / norm
    is_normalized = abs(norm - 1.0) < TOLERANCE
    
    # 转换回列表以便JSON序列化
    result_list = [complex(c) for c in normalized_state]
    
    return {
        'result': result_list,
        'metadata': {
            'dimension': len(components),
            'original_norm': float(norm),
            'is_normalized': is_normalized
        }
    }


def create_hermitian_operator(matrix_elements: List[List[float]]) -> Dict[str, Any]:
    """
    创建并验证厄米算符矩阵
    
    参数:
        matrix_elements: 矩阵元素的嵌套列表 [[row1], [row2], ...]
    
    返回:
        {
            'result': 矩阵元素列表,
            'metadata': {
                'dimension': 矩阵维度,
                'is_hermitian': 是否为厄米矩阵,
                'hermiticity_error': 厄米性误差
            }
        }
    """
    if not matrix_elements or not isinstance(matrix_elements, list):
        raise ValueError("matrix_elements必须是非空嵌套列表")
    
    # 转换为numpy数组
    matrix = np.array(matrix_elements, dtype=complex)
    
    if matrix.ndim != 2:
        raise ValueError("必须是二维矩阵")
    
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("必须是方阵")
    
    # 检查厄米性: A = A†
    hermitian_conjugate = matrix.conj().T
    hermiticity_error = np.linalg.norm(matrix - hermitian_conjugate)
    is_hermitian = hermiticity_error < TOLERANCE
    
    return {
        'result': matrix_elements,
        'metadata': {
            'dimension': matrix.shape[0],
            'is_hermitian': bool(is_hermitian),
            'hermiticity_error': float(hermiticity_error)
        }
    }


def compute_eigenvalues_eigenvectors(matrix_elements: List[List[float]]) -> Dict[str, Any]:
    """
    计算厄米算符的本征值和本征向量
    
    参数:
        matrix_elements: 厄米矩阵的元素
    
    返回:
        {
            'result': {
                'eigenvalues': 本征值列表（实数）,
                'eigenvectors': 本征向量列表（每个向量是一个列表）
            },
            'metadata': {
                'dimension': 矩阵维度,
                'eigenvalue_range': [最小值, 最大值]
            }
        }
    """
    matrix = np.array(matrix_elements, dtype=complex)
    
    # 使用scipy的eigh函数（专门用于厄米矩阵，更稳定）
    eigenvalues, eigenvectors = linalg.eigh(matrix)
    
    # 转换为可序列化格式
    eigenvalues_list = [float(ev) for ev in eigenvalues]
    eigenvectors_list = [[complex(c) for c in vec] for vec in eigenvectors.T]
    
    return {
        'result': {
            'eigenvalues': eigenvalues_list,
            'eigenvectors': eigenvectors_list
        },
        'metadata': {
            'dimension': len(eigenvalues_list),
            'eigenvalue_range': [float(min(eigenvalues)), float(max(eigenvalues))]
        }
    }


def analyze_degeneracy(eigenvalues: List[float], tolerance: float = TOLERANCE) -> Dict[str, Any]:
    """
    分析本征值的简并度
    
    参数:
        eigenvalues: 本征值列表
        tolerance: 判断简并的容差
    
    返回:
        {
            'result': {
                'unique_eigenvalues': 唯一本征值列表,
                'degeneracies': 对应的简并度列表,
                'degenerate_groups': {本征值: [索引列表]}
            },
            'metadata': {
                'total_levels': 总能级数,
                'degenerate_levels': 简并能级数,
                'max_degeneracy': 最大简并度
            }
        }
    """
    if not eigenvalues:
        raise ValueError("eigenvalues不能为空")
    
    # 按值分组
    groups = defaultdict(list)
    for idx, ev in enumerate(eigenvalues):
        # 找到最接近的已有组
        found = False
        for key in groups.keys():
            if abs(ev - key) < tolerance:
                groups[key].append(idx)
                found = True
                break
        if not found:
            groups[ev] = [idx]
    
    # 整理结果
    unique_eigenvalues = sorted(groups.keys())
    degeneracies = [len(groups[ev]) for ev in unique_eigenvalues]
    degenerate_groups = {float(ev): groups[ev] for ev in unique_eigenvalues}
    
    degenerate_count = sum(1 for d in degeneracies if d > 1)
    max_degeneracy = max(degeneracies) if degeneracies else 0
    
    return {
        'result': {
            'unique_eigenvalues': [float(ev) for ev in unique_eigenvalues],
            'degeneracies': degeneracies,
            'degenerate_groups': degenerate_groups
        },
        'metadata': {
            'total_levels': len(unique_eigenvalues),
            'degenerate_levels': degenerate_count,
            'max_degeneracy': max_degeneracy
        }
    }


def compute_projection_operator(eigenvectors: List[List[complex]], 
                                indices: List[int]) -> Dict[str, Any]:
    """
    计算投影到指定本征子空间的投影算符
    
    参数:
        eigenvectors: 所有本征向量列表
        indices: 要投影的本征向量索引列表
    
    返回:
        {
            'result': 投影算符矩阵元素（嵌套列表）,
            'metadata': {
                'subspace_dimension': 子空间维度,
                'is_projector': 是否满足投影算符性质 (P^2 = P)
            }
        }
    """
    if not indices:
        raise ValueError("indices不能为空")
    
    if not all(0 <= i < len(eigenvectors) for i in indices):
        raise ValueError(f"索引必须在0到{len(eigenvectors)-1}之间")
    
    # 构建投影算符: P = Σ|ψᵢ⟩⟨ψᵢ|
    dimension = len(eigenvectors[0])
    projector = np.zeros((dimension, dimension), dtype=complex)
    
    for idx in indices:
        vec = np.array(eigenvectors[idx], dtype=complex)
        projector += np.outer(vec, vec.conj())
    
    # 验证投影算符性质: P^2 = P
    projector_squared = projector @ projector
    is_projector = np.allclose(projector, projector_squared, atol=TOLERANCE)
    
    # 转换为可序列化格式
    projector_list = [[complex(c) for c in row] for row in projector]
    
    return {
        'result': projector_list,
        'metadata': {
            'subspace_dimension': len(indices),
            'is_projector': bool(is_projector),
            'trace': float(np.trace(projector).real)
        }
    }


def compute_measurement_probability(state: List[complex], 
                                   projector: List[List[complex]]) -> Dict[str, Any]:
    """
    计算量子态在投影算符对应子空间的测量概率
    
    参数:
        state: 量子态向量
        projector: 投影算符矩阵
    
    返回:
        {
            'result': 测量概率（0到1之间的实数）,
            'metadata': {
                'projected_state_norm': 投影态的模,
                'original_state_norm': 原始态的模
            }
        }
    """
    state_vec = np.array(state, dtype=complex)
    proj_matrix = np.array(projector, dtype=complex)
    
    # 计算投影态: P|ψ⟩
    projected_state = proj_matrix @ state_vec
    
    # 概率 = ⟨ψ|P|ψ⟩ = ||P|ψ⟩||²
    probability = np.abs(np.vdot(projected_state, projected_state))
    
    original_norm = np.linalg.norm(state_vec)
    projected_norm = np.linalg.norm(projected_state)
    
    return {
        'result': float(probability),
        'metadata': {
            'projected_state_norm': float(projected_norm),
            'original_state_norm': float(original_norm)
        }
    }


# ============================================================================
# 第二层：组合函数（高级量子力学分析）
# ============================================================================

def analyze_operator_spectrum(matrix_elements: List[List[float]]) -> Dict[str, Any]:
    """
    完整分析算符的谱结构（本征值、简并度、本征向量）
    
    参数:
        matrix_elements: 算符矩阵元素
    
    返回:
        {
            'result': {
                'eigenvalues': 本征值列表,
                'eigenvectors': 本征向量列表,
                'degeneracy_info': 简并度分析结果,
                'spectrum_summary': 谱结构摘要
            },
            'metadata': {...}
        }
    """
    # 步骤1：验证厄米性
    hermitian_check = create_hermitian_operator(matrix_elements)
    if not hermitian_check['metadata']['is_hermitian']:
        print(f"警告：矩阵的厄米性误差为 {hermitian_check['metadata']['hermiticity_error']}")
    
    # 步骤2：计算本征值和本征向量
    eigen_result = compute_eigenvalues_eigenvectors(matrix_elements)
    eigenvalues = eigen_result['result']['eigenvalues']
    eigenvectors = eigen_result['result']['eigenvectors']
    
    # 步骤3：分析简并度
    degeneracy_result = analyze_degeneracy(eigenvalues)
    
    # 生成谱结构摘要
    spectrum_summary = []
    for ev, deg in zip(degeneracy_result['result']['unique_eigenvalues'],
                       degeneracy_result['result']['degeneracies']):
        spectrum_summary.append({
            'eigenvalue': float(ev),
            'degeneracy': deg,
            'is_degenerate': deg > 1
        })
    
    return {
        'result': {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'degeneracy_info': degeneracy_result['result'],
            'spectrum_summary': spectrum_summary
        },
        'metadata': {
            'dimension': len(eigenvalues),
            'is_hermitian': hermitian_check['metadata']['is_hermitian'],
            'degenerate_levels': degeneracy_result['metadata']['degenerate_levels'],
            'max_degeneracy': degeneracy_result['metadata']['max_degeneracy']
        }
    }


def compute_degenerate_subspace_probabilities(state_components: List[float],
                                              matrix_elements: List[List[float]]) -> Dict[str, Any]:
    """
    计算量子态在所有简并子空间的测量概率
    
    参数:
        state_components: 量子态分量
        matrix_elements: 算符矩阵元素
    
    返回:
        {
            'result': {
                'probabilities': {本征值: 概率},
                'degenerate_probabilities': {本征值: [各简并态的概率]},
                'total_probability': 总概率（应为1）
            },
            'metadata': {...}
        }
    """
    # 步骤1：创建并归一化量子态
    state_result = create_quantum_state(state_components)
    state = state_result['result']
    
    # 步骤2：分析算符谱
    spectrum = analyze_operator_spectrum(matrix_elements)
    eigenvalues = spectrum['result']['eigenvalues']
    eigenvectors = spectrum['result']['eigenvectors']
    degeneracy_info = spectrum['result']['degeneracy_info']
    
    # 步骤3：计算每个本征值对应的测量概率
    probabilities = {}
    degenerate_probabilities = {}
    
    for ev, indices in degeneracy_info['degenerate_groups'].items():
        # 计算投影算符
        proj_result = compute_projection_operator(eigenvectors, indices)
        projector = proj_result['result']
        
        # 计算测量概率
        prob_result = compute_measurement_probability(state, projector)
        probability = prob_result['result']
        
        probabilities[float(ev)] = probability
        
        # 如果简并，计算各简并态的概率
        if len(indices) > 1:
            individual_probs = []
            for idx in indices:
                single_proj = compute_projection_operator(eigenvectors, [idx])
                single_prob = compute_measurement_probability(state, single_proj['result'])
                individual_probs.append(single_prob['result'])
            degenerate_probabilities[float(ev)] = individual_probs
    
    total_probability = sum(probabilities.values())
    
    return {
        'result': {
            'probabilities': probabilities,
            'degenerate_probabilities': degenerate_probabilities,
            'total_probability': float(total_probability)
        },
        'metadata': {
            'state_dimension': len(state),
            'number_of_eigenvalues': len(probabilities),
            'probability_sum_error': abs(total_probability - 1.0)
        }
    }


def find_degenerate_eigenvalues(matrix_elements: List[List[float]],
                               min_degeneracy: int = 2) -> Dict[str, Any]:
    """
    查找所有简并本征值及其简并度
    
    参数:
        matrix_elements: 算符矩阵元素
        min_degeneracy: 最小简并度阈值
    
    返回:
        {
            'result': {
                'degenerate_eigenvalues': 简并本征值列表,
                'degeneracies': 对应的简并度列表,
                'degenerate_subspaces': {本征值: 本征向量索引列表}
            },
            'metadata': {...}
        }
    """
    # 分析谱结构
    spectrum = analyze_operator_spectrum(matrix_elements)
    degeneracy_info = spectrum['result']['degeneracy_info']
    
    # 筛选简并本征值
    degenerate_eigenvalues = []
    degeneracies = []
    degenerate_subspaces = {}
    
    for ev, deg in zip(degeneracy_info['unique_eigenvalues'],
                       degeneracy_info['degeneracies']):
        if deg >= min_degeneracy:
            degenerate_eigenvalues.append(float(ev))
            degeneracies.append(deg)
            degenerate_subspaces[float(ev)] = degeneracy_info['degenerate_groups'][ev]
    
    return {
        'result': {
            'degenerate_eigenvalues': degenerate_eigenvalues,
            'degeneracies': degeneracies,
            'degenerate_subspaces': degenerate_subspaces
        },
        'metadata': {
            'total_degenerate_levels': len(degenerate_eigenvalues),
            'max_degeneracy': max(degeneracies) if degeneracies else 0,
            'min_degeneracy_threshold': min_degeneracy
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_energy_spectrum(spectrum_data: Dict[str, Any],
                              title: str = "Energy Spectrum") -> Dict[str, Any]:
    """
    可视化能级谱（包括简并度标注）
    
    参数:
        spectrum_data: analyze_operator_spectrum的返回结果
        title: 图表标题
    
    返回:
        {
            'result': 图像文件路径,
            'metadata': {
                'file_type': 'png',
                'figure_size': [width, height]
            }
        }
    """
    spectrum_summary = spectrum_data['result']['spectrum_summary']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制能级线
    for i, level in enumerate(spectrum_summary):
        ev = level['eigenvalue']
        deg = level['degeneracy']
        
        # 根据简并度设置颜色和线宽
        color = 'red' if deg > 1 else 'blue'
        linewidth = 2 if deg > 1 else 1
        
        ax.hlines(ev, i - 0.3, i + 0.3, colors=color, linewidth=linewidth)
        
        # 标注简并度
        if deg > 1:
            ax.text(i, ev, f'  g={deg}', fontsize=10, va='center', color='red')
    
    ax.set_xlabel('Level Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1, label='Non-degenerate'),
        Line2D([0], [0], color='red', lw=2, label='Degenerate')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    filepath = os.path.join(IMAGE_DIR, 'energy_spectrum.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'figure_size': [10, 6],
            'num_levels': len(spectrum_summary)
        }
    }


def visualize_measurement_probabilities(probability_data: Dict[str, Any],
                                       title: str = "Measurement Probabilities") -> Dict[str, Any]:
    """
    可视化测量概率分布
    
    参数:
        probability_data: compute_degenerate_subspace_probabilities的返回结果
        title: 图表标题
    
    返回:
        {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    probabilities = probability_data['result']['probabilities']
    degenerate_probs = probability_data['result']['degenerate_probabilities']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：总概率柱状图
    eigenvalues = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['red' if ev in degenerate_probs else 'blue' for ev in eigenvalues]
    
    ax1.bar(range(len(eigenvalues)), probs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Total Probabilities', fontsize=14)
    ax1.set_xticks(range(len(eigenvalues)))
    ax1.set_xticklabels([f'{ev:.3f}' for ev in eigenvalues], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：简并态的详细概率
    if degenerate_probs:
        ax2_data = []
        ax2_labels = []
        for ev, probs_list in degenerate_probs.items():
            for i, p in enumerate(probs_list):
                ax2_data.append(p)
                ax2_labels.append(f'λ={ev:.3f}\nstate {i+1}')
        
        ax2.bar(range(len(ax2_data)), ax2_data, color='orange', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Degenerate State', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Degenerate Subspace Probabilities', fontsize=14)
        ax2.set_xticks(range(len(ax2_data)))
        ax2.set_xticklabels(ax2_labels, rotation=45, fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No Degenerate States', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    filepath = os.path.join(IMAGE_DIR, 'measurement_probabilities.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'figure_size': [14, 5],
            'num_eigenvalues': len(eigenvalues),
            'num_degenerate': len(degenerate_probs)
        }
    }


def visualize_quantum_state(state_components: List[complex],
                           basis_labels: List[str] = None,
                           title: str = "Quantum State") -> Dict[str, Any]:
    """
    可视化量子态（幅值和相位）
    
    参数:
        state_components: 量子态分量
        basis_labels: 基矢标签列表
        title: 图表标题
    
    返回:
        {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    state = np.array(state_components, dtype=complex)
    amplitudes = np.abs(state)
    phases = np.angle(state)
    
    if basis_labels is None:
        basis_labels = [f'|{i}⟩' for i in range(len(state))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：幅值
    ax1.bar(range(len(amplitudes)), amplitudes, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Basis State', fontsize=12)
    ax1.set_ylabel('Amplitude |cᵢ|', fontsize=12)
    ax1.set_title('State Amplitudes', fontsize=14)
    ax1.set_xticks(range(len(basis_labels)))
    ax1.set_xticklabels(basis_labels)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：相位
    ax2.bar(range(len(phases)), phases, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Basis State', fontsize=12)
    ax2.set_ylabel('Phase (radians)', fontsize=12)
    ax2.set_title('State Phases', fontsize=14)
    ax2.set_xticks(range(len(basis_labels)))
    ax2.set_xticklabels(basis_labels)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=16, y=1.02)
    
    filepath = os.path.join(IMAGE_DIR, 'quantum_state.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'figure_size': [12, 5],
            'state_dimension': len(state)
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示量子力学计算工具包的三个应用场景
    """
    
    # ========================================================================
    # 场景1：解决原始问题 - 计算简并本征值和测量概率
    # ========================================================================
    print("=" * 80)
    print("场景1：量子测量中的简并本征值和概率计算")
    print("=" * 80)
    print("问题描述：")
    print("给定量子态 |ψ⟩ = (1/6, 0, 4/6)ᵀ")
    print("和算符 P = [[√2, 0, 0], [0, 1/√2, i/√2], [0, -i/√2, 1/√2]]")
    print("求：1) 简并本征值及其简并度")
    print("    2) 测量得到简并本征值的概率")
    print("-" * 80)
    
    # 定义问题参数
    state_components = [1/6, 0, 4/6]
    matrix_elements = [
        [np.sqrt(2), 0, 0],
        [0, 1/np.sqrt(2), 1j/np.sqrt(2)],
        [0, -1j/np.sqrt(2), 1/np.sqrt(2)]
    ]
    
    # 步骤1：创建并归一化量子态
    print("\n步骤1：创建并归一化量子态")
    state_result = create_quantum_state(state_components)
    print(f"FUNCTION_CALL: create_quantum_state | PARAMS: {state_components} | RESULT: {state_result}")
    
    # 步骤2：验证算符的厄米性
    print("\n步骤2：验证算符的厄米性")
    hermitian_result = create_hermitian_operator(matrix_elements)
    print(f"FUNCTION_CALL: create_hermitian_operator | PARAMS: matrix_elements | RESULT: {hermitian_result}")
    
    # 步骤3：分析算符的完整谱结构
    print("\n步骤3：分析算符的完整谱结构")
    spectrum_result = analyze_operator_spectrum(matrix_elements)
    print(f"FUNCTION_CALL: analyze_operator_spectrum | PARAMS: matrix_elements | RESULT: spectrum_result")
    print(f"本征值: {spectrum_result['result']['eigenvalues']}")
    print(f"简并度信息: {spectrum_result['result']['degeneracy_info']}")
    
    # 步骤4：查找简并本征值
    print("\n步骤4：查找简并本征值")
    degenerate_result = find_degenerate_eigenvalues(matrix_elements, min_degeneracy=2)
    print(f"FUNCTION_CALL: find_degenerate_eigenvalues | PARAMS: matrix_elements, min_degeneracy=2 | RESULT: {degenerate_result}")
    
    degenerate_eigenvalue = degenerate_result['result']['degenerate_eigenvalues'][0]
    degeneracy = degenerate_result['result']['degeneracies'][0]
    print(f"\n简并本征值: {degenerate_eigenvalue}")
    print(f"简并度: {degeneracy}")
    
    # 步骤5：计算所有本征值的测量概率
    print("\n步骤5：计算所有本征值的测量概率")
    probability_result = compute_degenerate_subspace_probabilities(state_components, matrix_elements)
    print(f"FUNCTION_CALL: compute_degenerate_subspace_probabilities | PARAMS: state_components, matrix_elements | RESULT: {probability_result}")
    
    # 提取简并本征值对应的概率
    degenerate_probs = probability_result['result']['degenerate_probabilities'][degenerate_eigenvalue]
    print(f"\n简并本征值 {degenerate_eigenvalue} 的各态测量概率: {degenerate_probs}")
    
    # 步骤6：可视化能级谱
    print("\n步骤6：可视化能级谱")
    spectrum_image = visualize_energy_spectrum(spectrum_result, title="Operator P Energy Spectrum")
    print(f"FUNCTION_CALL: visualize_energy_spectrum | PARAMS: spectrum_result | RESULT: {spectrum_image}")
    
    # 步骤7：可视化测量概率
    print("\n步骤7：可视化测量概率")
    prob_image = visualize_measurement_probabilities(probability_result, title="Measurement Probabilities for |ψ⟩")
    print(f"FUNCTION_CALL: visualize_measurement_probabilities | PARAMS: probability_result | RESULT: {prob_image}")
    
    # 步骤8：可视化量子态
    print("\n步骤8：可视化量子态")
    state_image = visualize_quantum_state(state_result['result'], title="Initial Quantum State |ψ⟩")
    print(f"FUNCTION_CALL: visualize_quantum_state | PARAMS: state_result['result'] | RESULT: {state_image}")
    
    # 验证答案
    print("\n" + "=" * 80)
    print("答案验证：")
    print(f"简并本征值: {degenerate_eigenvalue:.6f} (标准答案: √2 ≈ 1.414214)")
    print(f"简并度: {degeneracy} (标准答案: 2)")
    print(f"测量概率1: {degenerate_probs[0]:.6f} (标准答案: 8/17 ≈ 0.470588)")
    print(f"测量概率2: {degenerate_probs[1]:.6f} (标准答案: 1/17 ≈ 0.058824)")
    
    # 计算误差
    expected_eigenvalue = np.sqrt(2)
    expected_prob1 = 8/17
    expected_prob2 = 1/17
    
    eigenvalue_error = abs(degenerate_eigenvalue - expected_eigenvalue)
    prob1_error = abs(degenerate_probs[0] - expected_prob1)
    prob2_error = abs(degenerate_probs[1] - expected_prob2)
    
    print(f"\n误差分析：")
    print(f"本征值误差: {eigenvalue_error:.2e}")
    print(f"概率1误差: {prob1_error:.2e}")
    print(f"概率2误差: {prob2_error:.2e}")
    
    print(f"\nFINAL_ANSWER: 简并本征值={degenerate_eigenvalue:.6f}, 简并度={degeneracy}, 概率=[{degenerate_probs[0]:.6f}, {degenerate_probs[1]:.6f}]")
    
    
    # ========================================================================
    # 场景2：不同量子态在同一算符下的测量概率比较
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：不同量子态在同一算符下的测量概率比较")
    print("=" * 80)
    print("问题描述：")
    print("使用相同的算符P，比较三个不同初态的测量概率分布：")
    print("态1: |ψ₁⟩ = (1, 0, 0)ᵀ (第一个本征态)")
    print("态2: |ψ₂⟩ = (0, 1, 0)ᵀ (第二个基态)")
    print("态3: |ψ₃⟩ = (1, 1, 1)ᵀ/√3 (均匀叠加态)")
    print("-" * 80)
    
    test_states = [
        [1, 0, 0],
        [0, 1, 0],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    ]
    state_names = ['|ψ₁⟩ (First eigenstate)', '|ψ₂⟩ (Second basis)', '|ψ₃⟩ (Uniform superposition)']
    
    for i, (state, name) in enumerate(zip(test_states, state_names)):
        print(f"\n--- 分析 {name} ---")
        
        # 计算测量概率
        prob_result = compute_degenerate_subspace_probabilities(state, matrix_elements)
        print(f"FUNCTION_CALL: compute_degenerate_subspace_probabilities | PARAMS: {state}, matrix_elements | RESULT: prob_result")
        
        print(f"测量概率分布: {prob_result['result']['probabilities']}")
        if prob_result['result']['degenerate_probabilities']:
            print(f"简并态详细概率: {prob_result['result']['degenerate_probabilities']}")
        
        # 可视化
        prob_image = visualize_measurement_probabilities(prob_result, title=f"Probabilities for {name}")
        print(f"FUNCTION_CALL: visualize_measurement_probabilities | PARAMS: prob_result | RESULT: {prob_image}")
    
    print(f"\nFINAL_ANSWER: 场景2完成，已比较三个不同量子态的测量概率分布")
    
    
    # ========================================================================
    # 场景3：构造具有指定简并结构的算符
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：构造具有指定简并结构的算符并验证")
    print("=" * 80)
    print("问题描述：")
    print("构造一个4×4厄米算符，要求：")
    print("- 本征值为 {1, 2, 2, 3}（2是二重简并）")
    print("- 验证其简并结构")
    print("- 计算均匀叠加态 |ψ⟩ = (1, 1, 1, 1)ᵀ/2 的测量概率")
    print("-" * 80)
    
    # 构造对角算符（最简单的厄米算符）
    print("\n步骤1：构造具有指定本征值的对角算符")
    target_eigenvalues = [1, 2, 2, 3]
    custom_matrix = np.diag(target_eigenvalues).tolist()
    print(f"构造的矩阵（对角形式）: {custom_matrix}")
    
    # 验证厄米性
    print("\n步骤2：验证算符的厄米性")
    hermitian_check = create_hermitian_operator(custom_matrix)
    print(f"FUNCTION_CALL: create_hermitian_operator | PARAMS: custom_matrix | RESULT: {hermitian_check}")
    
    # 分析谱结构
    print("\n步骤3：分析谱结构")
    custom_spectrum = analyze_operator_spectrum(custom_matrix)
    print(f"FUNCTION_CALL: analyze_operator_spectrum | PARAMS: custom_matrix | RESULT: custom_spectrum")
    print(f"本征值: {custom_spectrum['result']['eigenvalues']}")
    print(f"简并度信息: {custom_spectrum['result']['degeneracy_info']}")
    
    # 查找简并本征值
    print("\n步骤4：查找简并本征值")
    custom_degenerate = find_degenerate_eigenvalues(custom_matrix, min_degeneracy=2)
    print(f"FUNCTION_CALL: find_degenerate_eigenvalues | PARAMS: custom_matrix | RESULT: {custom_degenerate}")
    print(f"简并本征值: {custom_degenerate['result']['degenerate_eigenvalues']}")
    print(f"简并度: {custom_degenerate['result']['degeneracies']}")
    
    # 计算均匀叠加态的测量概率
    print("\n步骤5：计算均匀叠加态的测量概率")
    uniform_state = [0.5, 0.5, 0.5, 0.5]
    custom_prob = compute_degenerate_subspace_probabilities(uniform_state, custom_matrix)
    print(f"FUNCTION_CALL: compute_degenerate_subspace_probabilities | PARAMS: {uniform_state}, custom_matrix | RESULT: {custom_prob}")
    print(f"测量概率分布: {custom_prob['result']['probabilities']}")
    print(f"简并态详细概率: {custom_prob['result']['degenerate_probabilities']}")
    
    # 可视化
    print("\n步骤6：可视化能级谱和测量概率")
    custom_spectrum_image = visualize_energy_spectrum(custom_spectrum, title="Custom Operator Spectrum (1, 2, 2, 3)")
    print(f"FUNCTION_CALL: visualize_energy_spectrum | PARAMS: custom_spectrum | RESULT: {custom_spectrum_image}")
    
    custom_prob_image = visualize_measurement_probabilities(custom_prob, title="Probabilities for Uniform State")
    print(f"FUNCTION_CALL: visualize_measurement_probabilities | PARAMS: custom_prob | RESULT: {custom_prob_image}")
    
    print(f"\nFINAL_ANSWER: 场景3完成，成功构造并验证了具有指定简并结构的算符")
    
    print("\n" + "=" * 80)
    print("所有场景演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()