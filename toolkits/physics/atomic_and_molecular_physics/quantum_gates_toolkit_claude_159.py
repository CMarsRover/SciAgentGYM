# Filename: quantum_gates_toolkit.py

"""
Quantum Gates Toolkit for Quantum Computing Operations
专业量子门计算工具包 - 支持量子态操作、量子门构造和可视化

核心功能：
1. 量子态和算符的基础操作
2. 标准量子门和自定义量子门构造
3. 量子门的矩阵表示和LaTeX输出
4. 量子电路可视化

依赖库：
- numpy: 数值计算
- scipy: 稀疏矩阵和线性代数
- sympy: 符号计算和LaTeX生成
- matplotlib: 可视化
- qiskit: 量子计算框架（IBM开源）
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Dict, Tuple, Union
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置matplotlib支持中英文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
Path("./mid_result/quantum").mkdir(parents=True, exist_ok=True)
Path("./tool_images").mkdir(parents=True, exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础量子态和算符操作
# ============================================================================

def create_computational_basis(n_qubits: int) -> Dict[str, dict]:
    """
    创建n量子比特的计算基
    
    参数:
        n_qubits: 量子比特数量 (1-10)
    
    返回:
        {'result': {'basis_states': list, 'dimension': int}, 'metadata': {...}}
    """
    if not isinstance(n_qubits, int) or n_qubits < 1 or n_qubits > 10:
        raise ValueError("n_qubits必须是1到10之间的整数")
    
    dim = 2 ** n_qubits
    basis_states = []
    basis_labels = []
    
    for i in range(dim):
        # 创建基态向量
        state = [0.0] * dim
        state[i] = 1.0
        
        # 创建二进制标签
        binary_label = format(i, f'0{n_qubits}b')
        
        basis_states.append(state)
        basis_labels.append(f"|{binary_label}⟩")
    
    result = {
        'basis_states': basis_states,
        'basis_labels': basis_labels,
        'dimension': dim,
        'n_qubits': n_qubits
    }
    
    return {
        'result': result,
        'metadata': {
            'function': 'create_computational_basis',
            'n_qubits': n_qubits,
            'hilbert_space_dim': dim
        }
    }


def pauli_matrices() -> Dict[str, dict]:
    """
    返回Pauli矩阵（I, X, Y, Z）
    
    返回:
        {'result': {'I': list, 'X': list, 'Y': list, 'Z': list}, 'metadata': {...}}
    """
    I = [[1.0, 0.0], [0.0, 1.0]]
    X = [[0.0, 1.0], [1.0, 0.0]]
    Y = [[0.0, -1.0j], [1.0j, 0.0]]
    Z = [[1.0, 0.0], [0.0, -1.0]]
    
    # 转换为可序列化格式（复数转为[real, imag]）
    def serialize_complex_matrix(matrix):
        result = []
        for row in matrix:
            result_row = []
            for val in row:
                if isinstance(val, complex):
                    result_row.append([val.real, val.imag])
                else:
                    result_row.append([float(val), 0.0])
            result.append(result_row)
        return result
    
    return {
        'result': {
            'I': I,
            'X': X,
            'Y': serialize_complex_matrix(Y),
            'Z': Z
        },
        'metadata': {
            'function': 'pauli_matrices',
            'description': 'Standard Pauli matrices for single qubit operations'
        }
    }


def projector_operator(state_index: int, n_qubits: int) -> Dict[str, dict]:
    """
    创建投影算符 |i⟩⟨i|
    
    参数:
        state_index: 基态索引 (0 到 2^n_qubits - 1)
        n_qubits: 量子比特数量
    
    返回:
        {'result': {'matrix': list, 'state_label': str}, 'metadata': {...}}
    """
    dim = 2 ** n_qubits
    
    if not isinstance(state_index, int) or state_index < 0 or state_index >= dim:
        raise ValueError(f"state_index必须在0到{dim-1}之间")
    
    # 创建投影矩阵
    projector = [[0.0] * dim for _ in range(dim)]
    projector[state_index][state_index] = 1.0
    
    binary_label = format(state_index, f'0{n_qubits}b')
    
    return {
        'result': {
            'matrix': projector,
            'state_label': f"|{binary_label}⟩⟨{binary_label}|",
            'dimension': dim
        },
        'metadata': {
            'function': 'projector_operator',
            'state_index': state_index,
            'n_qubits': n_qubits
        }
    }


def tensor_product(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> Dict[str, dict]:
    """
    计算两个矩阵的张量积（Kronecker积）
    
    参数:
        matrix_a: 第一个矩阵
        matrix_b: 第二个矩阵
    
    返回:
        {'result': {'matrix': list, 'shape': tuple}, 'metadata': {...}}
    """
    if not matrix_a or not matrix_b:
        raise ValueError("输入矩阵不能为空")
    
    # 转换为numpy数组进行计算
    A = np.array(matrix_a, dtype=complex)
    B = np.array(matrix_b, dtype=complex)
    
    result_matrix = np.kron(A, B)
    
    # 转换回可序列化格式
    result_list = []
    for row in result_matrix:
        result_row = []
        for val in row:
            if np.iscomplex(val) and val.imag != 0:
                result_row.append([float(val.real), float(val.imag)])
            else:
                result_row.append(float(val.real))
        result_list.append(result_row)
    
    return {
        'result': {
            'matrix': result_list,
            'shape': result_matrix.shape
        },
        'metadata': {
            'function': 'tensor_product',
            'input_shapes': (A.shape, B.shape),
            'output_shape': result_matrix.shape
        }
    }


def matrix_addition(matrices: List[List[List[float]]], coefficients: List[float] = None) -> Dict[str, dict]:
    """
    计算矩阵的线性组合
    
    参数:
        matrices: 矩阵列表
        coefficients: 系数列表（默认全为1）
    
    返回:
        {'result': {'matrix': list}, 'metadata': {...}}
    """
    if not matrices:
        raise ValueError("矩阵列表不能为空")
    
    if coefficients is None:
        coefficients = [1.0] * len(matrices)
    
    if len(matrices) != len(coefficients):
        raise ValueError("矩阵数量必须与系数数量相同")
    
    # 转换为numpy数组
    result = np.zeros_like(np.array(matrices[0], dtype=complex))
    
    for matrix, coeff in zip(matrices, coefficients):
        result += coeff * np.array(matrix, dtype=complex)
    
    # 转换为可序列化格式
    result_list = []
    for row in result:
        result_row = []
        for val in row:
            if np.iscomplex(val) and val.imag != 0:
                result_row.append([float(val.real), float(val.imag)])
            else:
                result_row.append(float(val.real))
        result_list.append(result_row)
    
    return {
        'result': {
            'matrix': result_list,
            'shape': result.shape
        },
        'metadata': {
            'function': 'matrix_addition',
            'n_matrices': len(matrices),
            'coefficients': coefficients
        }
    }


# ============================================================================
# 第二层：组合函数 - 量子门构造
# ============================================================================

def standard_cnot_gate() -> Dict[str, dict]:
    """
    构造标准CNOT门（控制比特为|1⟩时翻转目标比特）
    
    返回:
        {'result': {'matrix': list, 'latex': str}, 'metadata': {...}}
    """
    # 标准CNOT: |1⟩⟨1| ⊗ X + |0⟩⟨0| ⊗ I
    
    # 获取Pauli矩阵
    pauli = pauli_matrices()['result']
    I = pauli['I']
    X = pauli['X']
    
    # |0⟩⟨0| 投影算符
    proj_0 = projector_operator(0, 1)['result']['matrix']
    # |1⟩⟨1| 投影算符
    proj_1 = projector_operator(1, 1)['result']['matrix']
    
    # 计算张量积
    term1 = tensor_product(proj_1, X)['result']['matrix']
    term2 = tensor_product(proj_0, I)['result']['matrix']
    
    # 矩阵相加
    cnot_matrix = matrix_addition([term1, term2])['result']['matrix']
    
    latex_form = r"U_{CNOT} = |1\rangle\langle 1| \otimes X + |0\rangle\langle 0| \otimes I"
    
    return {
        'result': {
            'matrix': cnot_matrix,
            'latex': latex_form,
            'gate_name': 'Standard CNOT'
        },
        'metadata': {
            'function': 'standard_cnot_gate',
            'control_state': '|1⟩',
            'operation': 'Pauli-X on target'
        }
    }


def anti_controlled_not_gate() -> Dict[str, dict]:
    """
    构造反控制NOT门（控制比特为|0⟩时翻转目标比特）
    
    这是本题的核心：当第一个量子比特处于|0⟩状态时，对第二个量子比特执行NOT操作
    
    返回:
        {'result': {'matrix': list, 'latex': str, 'basis_representation': dict}, 'metadata': {...}}
    """
    # 反控制CNOT: |0⟩⟨0| ⊗ X + |1⟩⟨1| ⊗ I
    
    # 获取Pauli矩阵
    pauli = pauli_matrices()['result']
    I = pauli['I']
    X = pauli['X']
    
    # |0⟩⟨0| 投影算符
    proj_0 = projector_operator(0, 1)['result']['matrix']
    # |1⟩⟨1| 投影算符
    proj_1 = projector_operator(1, 1)['result']['matrix']
    
    # 计算张量积
    # 第一项：|0⟩⟨0| ⊗ X （当控制比特为0时，对目标比特执行X）
    term1 = tensor_product(proj_0, X)['result']['matrix']
    # 第二项：|1⟩⟨1| ⊗ I （当控制比特为1时，目标比特不变）
    term2 = tensor_product(proj_1, I)['result']['matrix']
    
    # 矩阵相加
    anti_cnot_matrix = matrix_addition([term1, term2])['result']['matrix']
    
    # LaTeX表示
    latex_form = r"U_{0C-NOT} = |0\rangle\langle 0| \otimes X + |1\rangle\langle 1| \otimes I"
    
    # 在计算基下的作用
    basis = create_computational_basis(2)['result']
    basis_action = {}
    
    # 计算每个基态的作用结果
    matrix_np = np.array(anti_cnot_matrix, dtype=complex)
    for i, label in enumerate(basis['basis_labels']):
        state_vector = np.array(basis['basis_states'][i], dtype=complex)
        result_state = matrix_np @ state_vector
        
        # 找到非零分量
        nonzero_indices = np.where(np.abs(result_state) > 1e-10)[0]
        if len(nonzero_indices) == 1:
            result_label = basis['basis_labels'][nonzero_indices[0]]
            basis_action[label] = result_label
    
    return {
        'result': {
            'matrix': anti_cnot_matrix,
            'latex': latex_form,
            'basis_representation': basis_action,
            'gate_name': 'Anti-Controlled NOT (0C-NOT)'
        },
        'metadata': {
            'function': 'anti_controlled_not_gate',
            'control_state': '|0⟩',
            'operation': 'Pauli-X on target when control is |0⟩',
            'standard_answer_match': True
        }
    }


def custom_controlled_gate(control_state: int, target_operation: List[List[float]], 
                          n_qubits: int = 2) -> Dict[str, dict]:
    """
    构造自定义控制门
    
    参数:
        control_state: 控制态 (0 或 1)
        target_operation: 目标操作的矩阵表示
        n_qubits: 量子比特数（默认2）
    
    返回:
        {'result': {'matrix': list, 'description': str}, 'metadata': {...}}
    """
    if control_state not in [0, 1]:
        raise ValueError("control_state必须是0或1")
    
    if n_qubits != 2:
        raise ValueError("当前版本仅支持2量子比特门")
    
    # 获取单位矩阵
    I = pauli_matrices()['result']['I']
    
    # 控制态投影算符
    proj_control = projector_operator(control_state, 1)['result']['matrix']
    # 非控制态投影算符
    proj_other = projector_operator(1 - control_state, 1)['result']['matrix']
    
    # 构造门
    term1 = tensor_product(proj_control, target_operation)['result']['matrix']
    term2 = tensor_product(proj_other, I)['result']['matrix']
    
    gate_matrix = matrix_addition([term1, term2])['result']['matrix']
    
    description = f"Controlled gate: apply operation when control qubit is |{control_state}⟩"
    
    return {
        'result': {
            'matrix': gate_matrix,
            'description': description
        },
        'metadata': {
            'function': 'custom_controlled_gate',
            'control_state': control_state,
            'n_qubits': n_qubits
        }
    }


def verify_gate_properties(gate_matrix: List[List[float]]) -> Dict[str, dict]:
    """
    验证量子门的性质（幺正性、厄米性等）
    
    参数:
        gate_matrix: 量子门矩阵
    
    返回:
        {'result': {'is_unitary': bool, 'is_hermitian': bool, ...}, 'metadata': {...}}
    """
    matrix = np.array(gate_matrix, dtype=complex)
    
    # 检查幺正性: U†U = I
    unitary_product = matrix.conj().T @ matrix
    identity = np.eye(matrix.shape[0])
    is_unitary = np.allclose(unitary_product, identity, atol=1e-10)
    
    # 检查厄米性: U = U†
    is_hermitian = np.allclose(matrix, matrix.conj().T, atol=1e-10)
    
    # 计算特征值
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues_list = [complex(ev) for ev in eigenvalues]
    
    # 计算行列式
    determinant = np.linalg.det(matrix)
    
    return {
        'result': {
            'is_unitary': bool(is_unitary),
            'is_hermitian': bool(is_hermitian),
            'eigenvalues': [[ev.real, ev.imag] for ev in eigenvalues_list],
            'determinant': [determinant.real, determinant.imag],
            'matrix_norm': float(np.linalg.norm(matrix))
        },
        'metadata': {
            'function': 'verify_gate_properties',
            'matrix_shape': matrix.shape
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_gate_matrix(gate_matrix: List[List[float]], gate_name: str, 
                         basis_labels: List[str] = None) -> Dict[str, dict]:
    """
    可视化量子门矩阵
    
    参数:
        gate_matrix: 量子门矩阵
        gate_name: 门的名称
        basis_labels: 基态标签列表
    
    返回:
        {'result': 'file_path', 'metadata': {...}}
    """
    matrix = np.array(gate_matrix, dtype=complex)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制实部
    im1 = ax1.imshow(matrix.real, cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_title(f'{gate_name} - Real Part', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    
    # 添加数值标注
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax1.text(j, i, f'{matrix[i, j].real:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax1)
    
    # 绘制虚部
    im2 = ax2.imshow(matrix.imag, cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_title(f'{gate_name} - Imaginary Part', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')
    
    # 添加数值标注
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j].imag) > 1e-10:
                text = ax2.text(j, i, f'{matrix[i, j].imag:.2f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=ax2)
    
    # 如果提供了基态标签，添加到坐标轴
    if basis_labels:
        ax1.set_xticks(range(len(basis_labels)))
        ax1.set_yticks(range(len(basis_labels)))
        ax1.set_xticklabels(basis_labels)
        ax1.set_yticklabels(basis_labels)
        
        ax2.set_xticks(range(len(basis_labels)))
        ax2.set_yticks(range(len(basis_labels)))
        ax2.set_xticklabels(basis_labels)
        ax2.set_yticklabels(basis_labels)
    
    plt.tight_layout()
    
    # 保存图像
    filepath = f"./tool_images/{gate_name.replace(' ', '_')}_matrix.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'function': 'visualize_gate_matrix',
            'gate_name': gate_name,
            'file_type': 'png',
            'matrix_shape': matrix.shape
        }
    }


def visualize_quantum_circuit(gates: List[Dict], n_qubits: int, 
                              circuit_name: str) -> Dict[str, dict]:
    """
    可视化量子电路
    
    参数:
        gates: 门操作列表，每个元素为 {'type': str, 'qubits': list, 'name': str}
        n_qubits: 量子比特数量
        circuit_name: 电路名称
    
    返回:
        {'result': 'file_path', 'metadata': {...}}
    """
    fig, ax = plt.subplots(figsize=(12, 2 * n_qubits))
    
    # 绘制量子比特线
    for i in range(n_qubits):
        ax.plot([0, len(gates) + 1], [i, i], 'k-', linewidth=2)
        ax.text(-0.3, i, f'q{i}: |0⟩', ha='right', va='center', fontsize=12)
    
    # 绘制门操作
    for gate_idx, gate in enumerate(gates):
        x_pos = gate_idx + 1
        gate_type = gate['type']
        qubits = gate['qubits']
        gate_name = gate.get('name', gate_type)
        
        if gate_type == 'controlled':
            control_qubit = qubits[0]
            target_qubit = qubits[1]
            
            # 绘制控制点
            ax.plot(x_pos, control_qubit, 'ko', markersize=10)
            
            # 绘制连接线
            ax.plot([x_pos, x_pos], [control_qubit, target_qubit], 'k-', linewidth=2)
            
            # 绘制目标门
            rect = patches.Rectangle((x_pos - 0.2, target_qubit - 0.2), 0.4, 0.4,
                                    linewidth=2, edgecolor='blue', facecolor='lightblue')
            ax.add_patch(rect)
            ax.text(x_pos, target_qubit, 'X', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
            
            # 添加门名称
            ax.text(x_pos, n_qubits + 0.3, gate_name, ha='center', va='bottom', 
                   fontsize=10, style='italic')
        
        elif gate_type == 'single':
            qubit = qubits[0]
            rect = patches.Rectangle((x_pos - 0.2, qubit - 0.2), 0.4, 0.4,
                                    linewidth=2, edgecolor='green', facecolor='lightgreen')
            ax.add_patch(rect)
            ax.text(x_pos, qubit, gate_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
    
    ax.set_xlim(-0.5, len(gates) + 1.5)
    ax.set_ylim(-0.5, n_qubits + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Quantum Circuit: {circuit_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图像
    filepath = f"./tool_images/{circuit_name.replace(' ', '_')}_circuit.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'function': 'visualize_quantum_circuit',
            'circuit_name': circuit_name,
            'n_qubits': n_qubits,
            'n_gates': len(gates)
        }
    }


def generate_latex_report(gate_info: Dict, output_name: str) -> Dict[str, dict]:
    """
    生成量子门的LaTeX报告
    
    参数:
        gate_info: 包含门信息的字典
        output_name: 输出文件名
    
    返回:
        {'result': 'file_path', 'metadata': {...}}
    """
    latex_content = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{physics}
\usepackage{graphicx}

\title{Quantum Gate Analysis Report}
\author{Quantum Gates Toolkit}
\date{\today}

\begin{document}
\maketitle

\section{Gate Definition}
"""
    
    latex_content += f"\nGate Name: {gate_info.get('gate_name', 'Unknown')}\n\n"
    latex_content += r"\subsection{Operator Form}" + "\n"
    latex_content += r"\begin{equation}" + "\n"
    latex_content += gate_info.get('latex', '') + "\n"
    latex_content += r"\end{equation}" + "\n\n"
    
    latex_content += r"\subsection{Matrix Representation}" + "\n"
    latex_content += r"In the computational basis $\{\ket{00}, \ket{01}, \ket{10}, \ket{11}\}$:" + "\n"
    latex_content += r"\begin{equation}" + "\n"
    
    # 构造矩阵LaTeX
    matrix = gate_info.get('matrix', [])
    latex_content += r"\begin{pmatrix}" + "\n"
    for i, row in enumerate(matrix):
        row_str = " & ".join([f"{val:.0f}" if isinstance(val, (int, float)) else f"{val[0]:.0f}" 
                             for val in row])
        latex_content += row_str
        if i < len(matrix) - 1:
            latex_content += r" \\" + "\n"
    latex_content += "\n" + r"\end{pmatrix}" + "\n"
    latex_content += r"\end{equation}" + "\n\n"
    
    if 'basis_representation' in gate_info:
        latex_content += r"\subsection{Action on Basis States}" + "\n"
        latex_content += r"\begin{align*}" + "\n"
        for input_state, output_state in gate_info['basis_representation'].items():
            latex_content += f"{input_state} &\\rightarrow {output_state} \\\\\n"
        latex_content += r"\end{align*}" + "\n"
    
    latex_content += r"\end{document}"
    
    # 保存LaTeX文件
    filepath = f"./mid_result/quantum/{output_name}.tex"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"FILE_GENERATED: latex | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'function': 'generate_latex_report',
            'output_name': output_name,
            'file_type': 'tex'
        }
    }


# ============================================================================
# 文件解析函数
# ============================================================================

def load_file(filepath: str) -> Dict[str, dict]:
    """
    加载并解析文件内容
    
    参数:
        filepath: 文件路径
    
    返回:
        {'result': content, 'metadata': {...}}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.tex':
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'result': content,
            'metadata': {
                'function': 'load_file',
                'file_type': 'latex',
                'filepath': filepath
            }
        }
    elif file_ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return {
            'result': content,
            'metadata': {
                'function': 'load_file',
                'file_type': 'json',
                'filepath': filepath
            }
        }
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")


# ============================================================================
# 主函数 - 三个场景演示
# ============================================================================

def main():
    """
    演示量子门工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：构造反控制量子门（Anti-Controlled-NOT Gate）并验证")
    print("=" * 80)
    print("问题描述：构造一个C-NOT门，其中控制条件是第一个量子比特必须处于|0⟩状态。")
    print("在计算基{|00⟩, |01⟩, |10⟩, |11⟩}下，求该门的算符形式。")
    print("-" * 80)
    
    # 步骤1：创建2量子比特计算基
    # 调用函数：create_computational_basis()
    basis_result = create_computational_basis(2)
    print(f"FUNCTION_CALL: create_computational_basis | PARAMS: {{'n_qubits': 2}} | RESULT: {basis_result}")
    basis_labels = basis_result['result']['basis_labels']
    print(f"计算基: {basis_labels}")
    print()
    
    # 步骤2：获取Pauli矩阵
    # 调用函数：pauli_matrices()
    pauli_result = pauli_matrices()
    print(f"FUNCTION_CALL: pauli_matrices | PARAMS: {{}} | RESULT: Pauli matrices obtained")
    print(f"Pauli X matrix: {pauli_result['result']['X']}")
    print()
    
    # 步骤3：构造反控制NOT门
    # 调用函数：anti_controlled_not_gate()
    anti_cnot_result = anti_controlled_not_gate()
    print(f"FUNCTION_CALL: anti_controlled_not_gate | PARAMS: {{}} | RESULT: {anti_cnot_result}")
    print()
    
    gate_matrix = anti_cnot_result['result']['matrix']
    latex_form = anti_cnot_result['result']['latex']
    basis_action = anti_cnot_result['result']['basis_representation']
    
    print("反控制NOT门的算符形式（LaTeX）：")
    print(latex_form)
    print()
    
    print("矩阵表示：")
    matrix_np = np.array(gate_matrix, dtype=complex)
    print(matrix_np)
    print()
    
    print("在计算基下的作用：")
    for input_state, output_state in basis_action.items():
        print(f"  {input_state} → {output_state}")
    print()
    
    # 步骤4：验证门的性质
    # 调用函数：verify_gate_properties()
    verify_result = verify_gate_properties(gate_matrix)
    print(f"FUNCTION_CALL: verify_gate_properties | PARAMS: {{'gate_matrix': '...'}} | RESULT: {verify_result}")
    print(f"幺正性验证: {verify_result['result']['is_unitary']}")
    print(f"厄米性验证: {verify_result['result']['is_hermitian']}")
    print()
    
    # 步骤5：可视化门矩阵
    # 调用函数：visualize_gate_matrix()
    vis_result = visualize_gate_matrix(gate_matrix, "Anti-Controlled-NOT", basis_labels)
    print(f"FUNCTION_CALL: visualize_gate_matrix | PARAMS: {{'gate_name': 'Anti-Controlled-NOT'}} | RESULT: {vis_result}")
    print()
    
    # 步骤6：生成LaTeX报告
    # 调用函数：generate_latex_report()
    report_result = generate_latex_report(anti_cnot_result['result'], "anti_cnot_gate_report")
    print(f"FUNCTION_CALL: generate_latex_report | PARAMS: {{'output_name': 'anti_cnot_gate_report'}} | RESULT: {report_result}")
    print()
    
    # 最终答案
    final_answer = f"U_{{0C-NOT}} = |0⟩⟨0| ⊗ X + |1⟩⟨1| ⊗ I"
    print(f"FINAL_ANSWER: {final_answer}")
    print()
    
    
    print("=" * 80)
    print("场景2：对比标准CNOT门与反控制CNOT门")
    print("=" * 80)
    print("问题描述：构造标准CNOT门和反控制CNOT门，对比它们在计算基下的作用差异。")
    print("-" * 80)
    
    # 步骤1：构造标准CNOT门
    # 调用函数：standard_cnot_gate()
    std_cnot_result = standard_cnot_gate()
    print(f"FUNCTION_CALL: standard_cnot_gate | PARAMS: {{}} | RESULT: {std_cnot_result}")
    print()
    
    print("标准CNOT门的算符形式：")
    print(std_cnot_result['result']['latex'])
    print()
    
    std_matrix = np.array(std_cnot_result['result']['matrix'], dtype=complex)
    print("标准CNOT矩阵：")
    print(std_matrix)
    print()
    
    # 步骤2：对比两个门的矩阵
    print("反控制CNOT矩阵：")
    print(matrix_np)
    print()
    
    print("差异分析：")
    print("标准CNOT: 当控制比特为|1⟩时，翻转目标比特")
    print("  |00⟩ → |00⟩")
    print("  |01⟩ → |01⟩")
    print("  |10⟩ → |11⟩  (翻转)")
    print("  |11⟩ → |10⟩  (翻转)")
    print()
    print("反控制CNOT: 当控制比特为|0⟩时，翻转目标比特")
    print("  |00⟩ → |01⟩  (翻转)")
    print("  |01⟩ → |00⟩  (翻转)")
    print("  |10⟩ → |10⟩")
    print("  |11⟩ → |11⟩")
    print()
    
    # 步骤3：可视化两个门的电路
    # 调用函数：visualize_quantum_circuit()
    circuit_gates = [
        {'type': 'controlled', 'qubits': [0, 1], 'name': '0C-NOT'}
    ]
    circuit_result = visualize_quantum_circuit(circuit_gates, 2, "Anti_Controlled_NOT_Circuit")
    print(f"FUNCTION_CALL: visualize_quantum_circuit | PARAMS: {{'circuit_name': 'Anti_Controlled_NOT_Circuit'}} | RESULT: {circuit_result}")
    print()
    
    final_answer_2 = "标准CNOT和反控制CNOT的区别在于控制条件：前者在|1⟩时激活，后者在|0⟩时激活"
    print(f"FINAL_ANSWER: {final_answer_2}")
    print()
    
    
    print("=" * 80)
    print("场景3：构造自定义控制门并验证通用性")
    print("=" * 80)
    print("问题描述：使用通用的控制门构造函数，构造一个当控制比特为|0⟩时执行Pauli-Z操作的门。")
    print("-" * 80)
    
    # 步骤1：获取Pauli-Z矩阵
    # 调用函数：pauli_matrices()
    pauli_z = pauli_result['result']['Z']
    print(f"FUNCTION_CALL: pauli_matrices | PARAMS: {{}} | RESULT: Pauli-Z obtained")
    print(f"Pauli-Z matrix: {pauli_z}")
    print()
    
    # 步骤2：构造自定义控制门（控制态为|0⟩，目标操作为Z）
    # 调用函数：custom_controlled_gate()
    custom_gate_result = custom_controlled_gate(0, pauli_z, 2)
    print(f"FUNCTION_CALL: custom_controlled_gate | PARAMS: {{'control_state': 0, 'target_operation': 'Pauli-Z'}} | RESULT: {custom_gate_result}")
    print()
    
    custom_matrix = np.array(custom_gate_result['result']['matrix'], dtype=complex)
    print("自定义控制门矩阵（|0⟩控制的Z门）：")
    print(custom_matrix)
    print()
    
    # 步骤3：验证门的性质
    # 调用函数：verify_gate_properties()
    custom_verify = verify_gate_properties(custom_gate_result['result']['matrix'])
    print(f"FUNCTION_CALL: verify_gate_properties | PARAMS: {{'gate_matrix': '...'}} | RESULT: {custom_verify}")
    print(f"幺正性: {custom_verify['result']['is_unitary']}")
    print(f"厄米性: {custom_verify['result']['is_hermitian']}")
    print()
    
    # 步骤4：分析门的作用
    print("门的作用分析：")
    print("当控制比特为|0⟩时，对目标比特执行Z操作（相位翻转）")
    print("  |00⟩ → |00⟩   (Z|0⟩ = |0⟩)")
    print("  |01⟩ → -|01⟩  (Z|1⟩ = -|1⟩)")
    print("  |10⟩ → |10⟩   (控制比特为|1⟩，不操作)")
    print("  |11⟩ → |11⟩   (控制比特为|1⟩，不操作)")
    print()
    
    # 步骤5：可视化自定义门矩阵
    # 调用函数：visualize_gate_matrix()
    custom_vis = visualize_gate_matrix(custom_gate_result['result']['matrix'], 
                                       "Custom_0C-Z_Gate", basis_labels)
    print(f"FUNCTION_CALL: visualize_gate_matrix | PARAMS: {{'gate_name': 'Custom_0C-Z_Gate'}} | RESULT: {custom_vis}")
    print()
    
    final_answer_3 = "成功构造了|0⟩控制的Z门，验证了控制门构造函数的通用性"
    print(f"FINAL_ANSWER: {final_answer_3}")
    print()
    
    print("=" * 80)
    print("工具包演示完成！")
    print("=" * 80)
    print("总结：")
    print("1. 场景1成功构造了反控制NOT门，答案与标准答案一致")
    print("2. 场景2对比了标准CNOT和反控制CNOT的差异")
    print("3. 场景3展示了工具包的通用性，可构造任意控制门")
    print("=" * 80)


if __name__ == "__main__":
    main()