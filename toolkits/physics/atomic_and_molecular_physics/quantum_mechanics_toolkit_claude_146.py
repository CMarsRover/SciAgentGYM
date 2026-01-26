# Filename: quantum_mechanics_toolkit.py

"""
Quantum Mechanics Matrix Analysis Toolkit
用于分析量子力学中矩阵的性质，包括厄米性、酉性、正定性等
"""

import numpy as np
from scipy.linalg import expm, logm, eigvalsh
from typing import Dict, List, Tuple, Union
import json
import os
import re
import matplotlib.pyplot as plt

# 设置matplotlib支持中英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./mid_result/quantum', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
TOLERANCE = 1e-10  # 数值容差


# ==================== 第一层：原子函数 ====================

def parse_matrix_string(matrix_str: str) -> List[List]:
    """
    解析矩阵字符串为列表格式
    
    Args:
        matrix_str: 矩阵字符串，行用分号分隔，如 "1,2;3,4" 或 "(1,2;3,4)"
                    支持带括号的格式，如 "(i, -1, 2i; 1, 0, 1; 2i, -1, -i)"
    
    Returns:
        dict: {'result': [[row1], [row2], ...], 'metadata': {...}}
    """
    try:
        # 去除首尾的括号和空格（如果存在）
        matrix_str = matrix_str.strip()
        if matrix_str.startswith('(') and matrix_str.endswith(')'):
            matrix_str = matrix_str[1:-1].strip()
        
        rows = matrix_str.split(';')
        matrix_list = []
        for row in rows:
            # 替换虚数单位 i 为 j（Python中的虚数单位）
            # 先处理数字后跟i的情况（如2i, -2i），但保留负号
            # 匹配模式：数字（可能带负号）后跟i
            row = re.sub(r'(-?\d+)i', r'\1j', row)
            # 然后处理单独的i（前后是逗号、空格或行首尾）
            # 注意：要先处理-i，再处理i，避免重复替换
            # 使用命名组或不同的替换方式来避免\1j被解释为\11
            row = re.sub(r'([,;\s]|^)-i([,;\s]|$)', lambda m: m.group(1) + '-1j' + m.group(2), row)
            row = re.sub(r'([,;\s]|^)i([,;\s]|$)', lambda m: m.group(1) + '1j' + m.group(2), row)
            
            elements = [eval(x.strip()) for x in row.split(',')]
            matrix_list.append(elements)
        
        return {
            'result': matrix_list,
            'metadata': {
                'shape': (len(matrix_list), len(matrix_list[0])),
                'description': 'Parsed matrix from string'
            }
        }
    except Exception as e:
        raise ValueError(f"矩阵解析失败: {str(e)}")


def is_hermitian(matrix_list: List[List], tolerance: float = TOLERANCE) -> Dict:
    """
    检查矩阵是否为厄米矩阵（Hermitian）
    厄米矩阵满足: A = A^†（共轭转置）
    
    Args:
        matrix_list: 矩阵的列表表示
        tolerance: 数值容差
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    try:
        A = np.array(matrix_list, dtype=complex)
        A_dagger = A.conj().T
        
        is_herm = np.allclose(A, A_dagger, atol=tolerance)
        max_diff = np.max(np.abs(A - A_dagger))
        
        return {
            'result': bool(is_herm),
            'metadata': {
                'max_difference': float(max_diff),
                'tolerance': tolerance,
                'description': 'Hermitian matrix check (A = A†)'
            }
        }
    except Exception as e:
        raise ValueError(f"厄米性检查失败: {str(e)}")


def is_unitary(matrix_list: List[List], tolerance: float = TOLERANCE) -> Dict:
    """
    检查矩阵是否为酉矩阵（Unitary）
    酉矩阵满足: U * U^† = I
    
    Args:
        matrix_list: 矩阵的列表表示
        tolerance: 数值容差
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    try:
        U = np.array(matrix_list, dtype=complex)
        n = U.shape[0]
        
        if U.shape[0] != U.shape[1]:
            return {
                'result': False,
                'metadata': {
                    'reason': 'Not a square matrix',
                    'description': 'Unitary matrix check (U*U† = I)'
                }
            }
        
        U_dagger = U.conj().T
        product = U @ U_dagger
        identity = np.eye(n)
        
        is_unit = np.allclose(product, identity, atol=tolerance)
        max_diff = np.max(np.abs(product - identity))
        
        return {
            'result': bool(is_unit),
            'metadata': {
                'max_difference': float(max_diff),
                'tolerance': tolerance,
                'description': 'Unitary matrix check (U*U† = I)'
            }
        }
    except Exception as e:
        raise ValueError(f"酉性检查失败: {str(e)}")


def is_positive_semidefinite(matrix_list: List[List], tolerance: float = TOLERANCE) -> Dict:
    """
    检查矩阵是否为半正定矩阵
    半正定矩阵的所有特征值 >= 0
    
    Args:
        matrix_list: 矩阵的列表表示
        tolerance: 数值容差
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    try:
        A = np.array(matrix_list, dtype=complex)
        
        # 首先检查是否为厄米矩阵（半正定矩阵必须是厄米矩阵）
        herm_check = is_hermitian(matrix_list, tolerance)
        if not herm_check['result']:
            return {
                'result': False,
                'metadata': {
                    'reason': 'Not Hermitian',
                    'eigenvalues': [],
                    'description': 'Positive semi-definite check'
                }
            }
        
        # 计算特征值
        eigenvalues = eigvalsh(A)
        min_eigenvalue = np.min(eigenvalues)
        
        is_psd = min_eigenvalue >= -tolerance
        
        return {
            'result': bool(is_psd),
            'metadata': {
                'eigenvalues': [float(ev) for ev in eigenvalues],
                'min_eigenvalue': float(min_eigenvalue),
                'tolerance': tolerance,
                'description': 'Positive semi-definite check (all eigenvalues >= 0)'
            }
        }
    except Exception as e:
        raise ValueError(f"半正定性检查失败: {str(e)}")


def calculate_trace(matrix_list: List[List]) -> Dict:
    """
    计算矩阵的迹（对角元素之和）
    
    Args:
        matrix_list: 矩阵的列表表示
    
    Returns:
        dict: {'result': complex, 'metadata': {...}}
    """
    try:
        A = np.array(matrix_list, dtype=complex)
        trace_value = np.trace(A)
        
        return {
            'result': complex(trace_value),
            'metadata': {
                'real_part': float(trace_value.real),
                'imag_part': float(trace_value.imag),
                'description': 'Matrix trace (sum of diagonal elements)'
            }
        }
    except Exception as e:
        raise ValueError(f"迹计算失败: {str(e)}")


def matrix_exponential(matrix_list: List[List]) -> Dict:
    """
    计算矩阵指数 e^A
    
    Args:
        matrix_list: 矩阵的列表表示
    
    Returns:
        dict: {'result': List[List], 'metadata': {...}}
    """
    try:
        A = np.array(matrix_list, dtype=complex)
        exp_A = expm(A)
        
        # 转换为列表格式
        result_list = exp_A.tolist()
        
        return {
            'result': result_list,
            'metadata': {
                'shape': exp_A.shape,
                'norm': float(np.linalg.norm(exp_A)),
                'description': 'Matrix exponential e^A'
            }
        }
    except Exception as e:
        raise ValueError(f"矩阵指数计算失败: {str(e)}")


def matrix_multiply(matrix1_list: List[List], matrix2_list: List[List]) -> Dict:
    """
    矩阵乘法 A * B
    
    Args:
        matrix1_list: 第一个矩阵
        matrix2_list: 第二个矩阵
    
    Returns:
        dict: {'result': List[List], 'metadata': {...}}
    """
    try:
        A = np.array(matrix1_list, dtype=complex)
        B = np.array(matrix2_list, dtype=complex)
        
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"矩阵维度不匹配: {A.shape} 和 {B.shape}")
        
        C = A @ B
        result_list = C.tolist()
        
        return {
            'result': result_list,
            'metadata': {
                'shape': C.shape,
                'description': 'Matrix multiplication A*B'
            }
        }
    except Exception as e:
        raise ValueError(f"矩阵乘法失败: {str(e)}")


def conjugate_transpose(matrix_list: List[List]) -> Dict:
    """
    计算矩阵的共轭转置 A^†
    
    Args:
        matrix_list: 矩阵的列表表示
    
    Returns:
        dict: {'result': List[List], 'metadata': {...}}
    """
    try:
        A = np.array(matrix_list, dtype=complex)
        A_dagger = A.conj().T
        
        result_list = A_dagger.tolist()
        
        return {
            'result': result_list,
            'metadata': {
                'shape': A_dagger.shape,
                'description': 'Conjugate transpose A†'
            }
        }
    except Exception as e:
        raise ValueError(f"共轭转置计算失败: {str(e)}")


# ==================== 第二层：组合函数 ====================

def is_density_matrix(matrix_list: List[List], tolerance: float = TOLERANCE) -> Dict:
    """
    检查矩阵是否为有效的密度矩阵（量子态）
    密度矩阵必须满足：
    1. 厄米性 (Hermitian)
    2. 半正定性 (Positive semi-definite)
    3. 迹为1 (Trace = 1)
    
    Args:
        matrix_list: 矩阵的列表表示
        tolerance: 数值容差
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    try:
        # 检查1: 厄米性
        herm_result = is_hermitian(matrix_list, tolerance)
        
        # 检查2: 半正定性
        psd_result = is_positive_semidefinite(matrix_list, tolerance)
        
        # 检查3: 迹为1
        trace_result = calculate_trace(matrix_list)
        trace_value = trace_result['result']
        trace_is_one = abs(trace_value - 1.0) < tolerance
        
        # 综合判断
        is_valid = herm_result['result'] and psd_result['result'] and trace_is_one
        
        return {
            'result': bool(is_valid),
            'metadata': {
                'hermitian': herm_result['result'],
                'positive_semidefinite': psd_result['result'],
                'trace_is_one': trace_is_one,
                'trace_value': {
                    'real': float(trace_value.real),
                    'imag': float(trace_value.imag)
                },
                'eigenvalues': psd_result['metadata'].get('eigenvalues', []),
                'description': 'Density matrix validation (Hermitian + PSD + Tr=1)'
            }
        }
    except Exception as e:
        raise ValueError(f"密度矩阵检查失败: {str(e)}")


def similarity_transformation(matrix_list: List[List], transform_matrix_list: List[List]) -> Dict:
    """
    相似变换: U * A * U^†
    在量子力学中，酉变换保持密度矩阵的性质
    
    Args:
        matrix_list: 要变换的矩阵 A
        transform_matrix_list: 变换矩阵 U
    
    Returns:
        dict: {'result': List[List], 'metadata': {...}}
    """
    try:
        # 计算 U^†
        U_dagger_result = conjugate_transpose(transform_matrix_list)
        U_dagger = U_dagger_result['result']
        
        # 计算 U * A
        UA_result = matrix_multiply(transform_matrix_list, matrix_list)
        UA = UA_result['result']
        
        # 计算 (U * A) * U^†
        result = matrix_multiply(UA, U_dagger)
        
        return {
            'result': result['result'],
            'metadata': {
                'transformation': 'U * A * U†',
                'description': 'Similarity transformation'
            }
        }
    except Exception as e:
        raise ValueError(f"相似变换失败: {str(e)}")


def exponential_similarity_transformation(matrix_list: List[List], exponent_matrix_list: List[List]) -> Dict:
    """
    指数相似变换: e^X * A * e^(-X)
    
    Args:
        matrix_list: 要变换的矩阵 A
        exponent_matrix_list: 指数矩阵 X
    
    Returns:
        dict: {'result': List[List], 'metadata': {...}}
    """
    try:
        # 计算 e^X
        exp_X_result = matrix_exponential(exponent_matrix_list)
        exp_X = exp_X_result['result']
        
        # 计算 -X
        neg_X = [[-x for x in row] for row in exponent_matrix_list]
        
        # 计算 e^(-X)
        exp_neg_X_result = matrix_exponential(neg_X)
        exp_neg_X = exp_neg_X_result['result']
        
        # 计算 e^X * A
        temp_result = matrix_multiply(exp_X, matrix_list)
        temp = temp_result['result']
        
        # 计算 (e^X * A) * e^(-X)
        final_result = matrix_multiply(temp, exp_neg_X)
        
        return {
            'result': final_result['result'],
            'metadata': {
                'transformation': 'e^X * A * e^(-X)',
                'exp_X_norm': exp_X_result['metadata']['norm'],
                'description': 'Exponential similarity transformation'
            }
        }
    except Exception as e:
        raise ValueError(f"指数相似变换失败: {str(e)}")


def analyze_matrix_properties(matrix_list: List[List], matrix_name: str = "Matrix") -> Dict:
    """
    全面分析矩阵的量子力学性质
    
    Args:
        matrix_list: 矩阵的列表表示
        matrix_name: 矩阵名称
    
    Returns:
        dict: {'result': dict, 'metadata': {...}}
    """
    try:
        properties = {}
        
        # 基本性质
        properties['hermitian'] = is_hermitian(matrix_list)['result']
        properties['unitary'] = is_unitary(matrix_list)['result']
        
        psd_result = is_positive_semidefinite(matrix_list)
        properties['positive_semidefinite'] = psd_result['result']
        properties['eigenvalues'] = psd_result['metadata'].get('eigenvalues', [])
        
        trace_result = calculate_trace(matrix_list)
        properties['trace'] = {
            'value': complex(trace_result['result']),
            'real': float(trace_result['result'].real),
            'imag': float(trace_result['result'].imag)
        }
        
        # 密度矩阵检查
        density_result = is_density_matrix(matrix_list)
        properties['is_density_matrix'] = density_result['result']
        
        return {
            'result': properties,
            'metadata': {
                'matrix_name': matrix_name,
                'description': 'Comprehensive quantum matrix property analysis'
            }
        }
    except Exception as e:
        raise ValueError(f"矩阵性质分析失败: {str(e)}")


# ==================== 第三层：可视化函数 ====================

def visualize_matrix_properties(properties_dict: dict, matrix_name: str, 
                                save_path: str = './tool_images/matrix_properties.png') -> Dict:
    """
    可视化矩阵性质
    
    Args:
        properties_dict: 矩阵性质字典
        matrix_name: 矩阵名称
        save_path: 保存路径
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：性质检查结果
        ax1 = axes[0]
        properties_to_check = ['hermitian', 'unitary', 'positive_semidefinite', 'is_density_matrix']
        property_names = ['Hermitian', 'Unitary', 'Positive\nSemi-definite', 'Density\nMatrix']
        colors = ['green' if properties_dict.get(prop, False) else 'red' 
                 for prop in properties_to_check]
        values = [1 if properties_dict.get(prop, False) else 0 
                 for prop in properties_to_check]
        
        bars = ax1.bar(property_names, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylim([0, 1.2])
        ax1.set_ylabel('Status (1=True, 0=False)', fontsize=12)
        ax1.set_title(f'{matrix_name} Properties', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    'True' if val == 1 else 'False',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 右图：特征值分布
        ax2 = axes[1]
        eigenvalues = properties_dict.get('eigenvalues', [])
        if eigenvalues:
            x_pos = range(len(eigenvalues))
            colors_ev = ['green' if ev >= 0 else 'red' for ev in eigenvalues]
            bars = ax2.bar(x_pos, eigenvalues, color=colors_ev, alpha=0.7, edgecolor='black')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel('Eigenvalue Index', fontsize=12)
            ax2.set_ylabel('Eigenvalue', fontsize=12)
            ax2.set_title(f'{matrix_name} Eigenvalues', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, ev in zip(bars, eigenvalues):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ev:.4f}',
                        ha='center', va='bottom' if ev >= 0 else 'top', 
                        fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No eigenvalues available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'{matrix_name} Eigenvalues', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"FILE_GENERATED: image | PATH: {save_path}")
        
        return {
            'result': save_path,
            'metadata': {
                'file_type': 'png',
                'description': 'Matrix properties visualization'
            }
        }
    except Exception as e:
        raise ValueError(f"可视化失败: {str(e)}")


def visualize_eigenvalue_comparison(matrices_dict: dict, 
                                   save_path: str = './tool_images/eigenvalue_comparison.png') -> Dict:
    """
    比较多个矩阵的特征值
    
    Args:
        matrices_dict: {matrix_name: properties_dict}
        save_path: 保存路径
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_offset = 0
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, (name, props) in enumerate(matrices_dict.items()):
            eigenvalues = props.get('eigenvalues', [])
            if eigenvalues:
                x_pos = [x_offset + i for i in range(len(eigenvalues))]
                color = colors[idx % len(colors)]
                ax.bar(x_pos, eigenvalues, width=0.8, label=name, 
                      color=color, alpha=0.7, edgecolor='black')
                x_offset += len(eigenvalues) + 1
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Eigenvalue Index (grouped by matrix)', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title('Eigenvalue Comparison Across Matrices', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"FILE_GENERATED: image | PATH: {save_path}")
        
        return {
            'result': save_path,
            'metadata': {
                'file_type': 'png',
                'description': 'Eigenvalue comparison visualization'
            }
        }
    except Exception as e:
        raise ValueError(f"特征值比较可视化失败: {str(e)}")


# ==================== 主函数：三个场景演示 ====================

def main():
    """
    演示量子力学矩阵分析工具包的三个场景
    """
    
    # 定义题目中的矩阵（使用问题中的格式，包含括号和虚数单位i）
    W_str = "(0, 0, 1; 0, 1, 0; 1, 0, 0)"
    X_str = "(i, -1, 2i; 1, 0, 1; 2i, -1, -i)"
    Y_str = "(0.5, 0.1, 0.2; 0.1, 0.25, 0.1; 0.2, 0.1, 0.25)"
    Z_str = "(3, 2i, 5; -2i, -2, -4i; 5, 4i, 4)"
    
    print("=" * 80)
    print("场景1：分析 (e^X)*Y*(e^{-X}) 是否为量子态")
    print("=" * 80)
    print("问题描述：根据题目给定的矩阵W, X, Y, Z，分析 (e^X)*Y*(e^{-X}) 是否为有效的密度矩阵（量子态）")
    print("-" * 80)
    print(f"给定矩阵：")
    print(f"  W = {W_str}")
    print(f"  X = {X_str}")
    print(f"  Y = {Y_str}")
    print(f"  Z = {Z_str}")
    print("-" * 80)
    
    # 步骤1：解析矩阵X
    print("\n步骤1：解析矩阵X")
    # 调用函数：parse_matrix_string()
    X_parsed = parse_matrix_string(X_str)
    X_matrix = X_parsed['result']
    print(f"FUNCTION_CALL: parse_matrix_string | PARAMS: {{'matrix_str': 'X_str'}} | RESULT: shape={X_parsed['metadata']['shape']}")
    print(f"矩阵X解析结果：{X_matrix}")
    
    # 步骤2：解析矩阵Y
    print("\n步骤2：解析矩阵Y")
    # 调用函数：parse_matrix_string()
    Y_parsed = parse_matrix_string(Y_str)
    Y_matrix = Y_parsed['result']
    print(f"FUNCTION_CALL: parse_matrix_string | PARAMS: {{'matrix_str': 'Y_str'}} | RESULT: shape={Y_parsed['metadata']['shape']}")
    print(f"矩阵Y解析结果：{Y_matrix}")
    
    # 步骤3：分析Y的性质
    print("\n步骤3：分析矩阵Y的量子力学性质")
    # 调用函数：analyze_matrix_properties()
    Y_properties = analyze_matrix_properties(Y_matrix, "Y")
    print(f"FUNCTION_CALL: analyze_matrix_properties | PARAMS: {{'matrix_name': 'Y'}} | RESULT: {Y_properties['result']}")
    
    # 步骤4：执行指数相似变换 (e^X)*Y*(e^{-X})
    print("\n步骤4：计算 (e^X)*Y*(e^{-X})")
    # 调用函数：exponential_similarity_transformation()
    transformed_Y = exponential_similarity_transformation(Y_matrix, X_matrix)
    result_matrix = transformed_Y['result']
    print(f"FUNCTION_CALL: exponential_similarity_transformation | PARAMS: {{'exponent': 'X', 'matrix': 'Y'}} | RESULT: transformation_complete")
    
    # 步骤5：分析变换后矩阵的性质
    print("\n步骤5：分析变换后矩阵 (e^X)*Y*(e^{-X}) 的性质")
    # 调用函数：analyze_matrix_properties()
    result_properties = analyze_matrix_properties(result_matrix, "(e^X)*Y*(e^{-X})")
    print(f"FUNCTION_CALL: analyze_matrix_properties | PARAMS: {{'matrix_name': '(e^X)*Y*(e^{{-X}})'}} | RESULT: {result_properties['result']}")
    
    # 步骤6：检查是否为密度矩阵
    print("\n步骤6：验证 (e^X)*Y*(e^{-X}) 是否为有效的密度矩阵")
    # 调用函数：is_density_matrix()
    density_check = is_density_matrix(result_matrix)
    print(f"FUNCTION_CALL: is_density_matrix | PARAMS: {{'matrix': '(e^X)*Y*(e^{{-X}})'}} | RESULT: {density_check}")
    
    # 步骤7：可视化结果
    print("\n步骤7：可视化矩阵性质")
    # 调用函数：visualize_matrix_properties()
    vis_result = visualize_matrix_properties(result_properties['result'], "(e^X)*Y*(e^{-X})", 
                                            './tool_images/scenario1_result.png')
    print(f"FUNCTION_CALL: visualize_matrix_properties | PARAMS: {{'matrix_name': '(e^X)*Y*(e^{{-X}})'}} | RESULT: {vis_result}")
    
    print("\n" + "=" * 80)
    print(f"场景1结论：(e^X)*Y*(e^{{-X}}) {'是' if density_check['result'] else '不是'}有效的量子态（密度矩阵）")
    print(f"  - 厄米性: {result_properties['result']['hermitian']}")
    print(f"  - 半正定性: {result_properties['result']['positive_semidefinite']}")
    print(f"  - 迹为1: {abs(result_properties['result']['trace']['real'] - 1.0) < TOLERANCE}")
    print(f"  - 特征值: {result_properties['result']['eigenvalues']}")
    print(f"FINAL_ANSWER: (e^X)*Y*(e^{{-X}}) represents a quantum state: {density_check['result']}")
    
    
    print("\n\n" + "=" * 80)
    print("场景2：分析所有给定矩阵的量子力学性质")
    print("=" * 80)
    print("问题描述：系统分析矩阵W, X, Y, Z的厄米性、酉性、半正定性等性质")
    print("-" * 80)
    
    # 解析所有矩阵
    matrices = {
        'W': parse_matrix_string(W_str)['result'],
        'X': parse_matrix_string(X_str)['result'],
        'Y': parse_matrix_string(Y_str)['result'],
        'Z': parse_matrix_string(Z_str)['result']
    }
    
    all_properties = {}
    
    for name, matrix in matrices.items():
        print(f"\n分析矩阵 {name}:")
        # 调用函数：analyze_matrix_properties()
        props = analyze_matrix_properties(matrix, name)
        all_properties[name] = props['result']
        print(f"FUNCTION_CALL: analyze_matrix_properties | PARAMS: {{'matrix_name': '{name}'}} | RESULT: {props['result']}")
        
        # 可视化每个矩阵
        # 调用函数：visualize_matrix_properties()
        vis = visualize_matrix_properties(props['result'], name, 
                                         f'./tool_images/scenario2_{name}.png')
        print(f"FUNCTION_CALL: visualize_matrix_properties | PARAMS: {{'matrix_name': '{name}'}} | RESULT: {vis}")
    
    # 比较所有矩阵的特征值
    print("\n生成特征值比较图:")
    # 调用函数：visualize_eigenvalue_comparison()
    comparison_vis = visualize_eigenvalue_comparison(all_properties, 
                                                    './tool_images/scenario2_comparison.png')
    print(f"FUNCTION_CALL: visualize_eigenvalue_comparison | PARAMS: {{'matrices': ['W','X','Y','Z']}} | RESULT: {comparison_vis}")
    
    print("\n" + "=" * 80)
    print("场景2总结：")
    for name, props in all_properties.items():
        print(f"  矩阵{name}: 厄米={props['hermitian']}, 酉={props['unitary']}, "
              f"半正定={props['positive_semidefinite']}, 密度矩阵={props['is_density_matrix']}")
    print(f"FINAL_ANSWER: Matrix property analysis completed for W, X, Y, Z")
    
    
    print("\n\n" + "=" * 80)
    print("场景3：验证酉变换保持密度矩阵性质")
    print("=" * 80)
    print("问题描述：如果W是酉矩阵且Y是密度矩阵，验证 W*Y*W† 是否仍为密度矩阵")
    print("-" * 80)
    
    W_matrix = matrices['W']
    Y_matrix = matrices['Y']
    
    # 步骤1：检查W是否为酉矩阵
    print("\n步骤1：检查矩阵W是否为酉矩阵")
    # 调用函数：is_unitary()
    W_unitary = is_unitary(W_matrix)
    print(f"FUNCTION_CALL: is_unitary | PARAMS: {{'matrix': 'W'}} | RESULT: {W_unitary}")
    
    # 步骤2：检查Y是否为密度矩阵
    print("\n步骤2：检查矩阵Y是否为密度矩阵")
    # 调用函数：is_density_matrix()
    Y_density = is_density_matrix(Y_matrix)
    print(f"FUNCTION_CALL: is_density_matrix | PARAMS: {{'matrix': 'Y'}} | RESULT: {Y_density}")
    
    # 步骤3：执行相似变换 W*Y*W†
    print("\n步骤3：计算 W*Y*W†")
    # 调用函数：similarity_transformation()
    transformed_by_W = similarity_transformation(Y_matrix, W_matrix)
    WYW_matrix = transformed_by_W['result']
    print(f"FUNCTION_CALL: similarity_transformation | PARAMS: {{'matrix': 'Y', 'transform': 'W'}} | RESULT: transformation_complete")
    
    # 步骤4：检查变换后是否仍为密度矩阵
    print("\n步骤4：验证 W*Y*W† 是否为密度矩阵")
    # 调用函数：is_density_matrix()
    WYW_density = is_density_matrix(WYW_matrix)
    print(f"FUNCTION_CALL: is_density_matrix | PARAMS: {{'matrix': 'W*Y*W†'}} | RESULT: {WYW_density}")
    
    # 步骤5：分析变换后的性质
    print("\n步骤5：分析 W*Y*W† 的详细性质")
    # 调用函数：analyze_matrix_properties()
    WYW_properties = analyze_matrix_properties(WYW_matrix, "W*Y*W†")
    print(f"FUNCTION_CALL: analyze_matrix_properties | PARAMS: {{'matrix_name': 'W*Y*W†'}} | RESULT: {WYW_properties['result']}")
    
    # 步骤6：可视化
    print("\n步骤6：可视化变换前后的对比")
    # 调用函数：visualize_matrix_properties()
    vis_WYW = visualize_matrix_properties(WYW_properties['result'], "W*Y*W†", 
                                         './tool_images/scenario3_WYW.png')
    print(f"FUNCTION_CALL: visualize_matrix_properties | PARAMS: {{'matrix_name': 'W*Y*W†'}} | RESULT: {vis_WYW}")
    
    print("\n" + "=" * 80)
    print(f"场景3结论：")
    print(f"  - W是酉矩阵: {W_unitary['result']}")
    print(f"  - Y是密度矩阵: {Y_density['result']}")
    print(f"  - W*Y*W†是密度矩阵: {WYW_density['result']}")
    print(f"  验证了量子力学中的重要性质：酉变换保持密度矩阵的性质")
    print(f"FINAL_ANSWER: Unitary transformation preserves density matrix properties: {WYW_density['result']}")


if __name__ == "__main__":
    main()