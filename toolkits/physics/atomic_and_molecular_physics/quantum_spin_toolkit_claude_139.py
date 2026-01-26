# Filename: quantum_spin_toolkit.py

"""
Quantum Spin State Analysis Toolkit
用于分析复数向量是否满足n电子系统自旋态的条件
"""

import numpy as np
from typing import List, Dict, Tuple, Union
import json
import os
from pathlib import Path

# 创建结果保存目录
Path("./mid_result/physics").mkdir(parents=True, exist_ok=True)
Path("./tool_images").mkdir(parents=True, exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础向量操作
# ============================================================================

def parse_complex_vector(vector_data: List[Union[str, float, int]]) -> Dict:
    """
    解析复数向量字符串或数值列表为numpy数组
    
    Args:
        vector_data: 向量数据，可以是字符串列表或数值列表
                    例如: ["1/sqrt(2)", "-i/sqrt(2)"] 或 [0.707, -0.707j]
    
    Returns:
        dict: {
            'result': list of complex numbers,
            'metadata': {
                'dimension': int,
                'has_complex': bool,
                'original_format': str
            }
        }
    """
    try:
        parsed_vector = []
        
        for element in vector_data:
            if isinstance(element, str):
                # 处理字符串表达式
                element = element.strip()
                
                # 替换常见数学符号
                element = element.replace('sqrt', 'np.sqrt')
                element = element.replace('i/', '1j/')
                element = element.replace('/i', '/1j')
                
                # 处理单独的i
                if element == 'i':
                    value = 1j
                elif element == '-i':
                    value = -1j
                elif element.startswith('i/'):
                    value = 1j / eval(element[2:])
                elif element.startswith('-i/'):
                    value = -1j / eval(element[3:])
                else:
                    value = complex(eval(element))
                
                parsed_vector.append(value)
            else:
                # 直接是数值
                parsed_vector.append(complex(element))
        
        has_complex = any(np.imag(v) != 0 for v in parsed_vector)
        
        return {
            'result': parsed_vector,
            'metadata': {
                'dimension': len(parsed_vector),
                'has_complex': has_complex,
                'original_format': 'string' if isinstance(vector_data[0], str) else 'numeric'
            }
        }
    
    except Exception as e:
        raise ValueError(f"Failed to parse vector: {str(e)}")


def compute_norm(vector: List[complex]) -> Dict:
    """
    计算复数向量的范数（模）
    
    Args:
        vector: 复数向量列表
    
    Returns:
        dict: {
            'result': float (范数值),
            'metadata': {
                'is_normalized': bool,
                'tolerance': float
            }
        }
    """
    if not vector:
        raise ValueError("Vector cannot be empty")
    
    vec_array = np.array(vector, dtype=complex)
    norm_value = float(np.linalg.norm(vec_array))
    
    tolerance = 1e-10
    is_normalized = abs(norm_value - 1.0) < tolerance
    
    return {
        'result': norm_value,
        'metadata': {
            'is_normalized': is_normalized,
            'tolerance': tolerance,
            'dimension': len(vector)
        }
    }


def compute_inner_product(vector1: List[complex], vector2: List[complex]) -> Dict:
    """
    计算两个复数向量的内积 <v1|v2>
    
    Args:
        vector1: 第一个复数向量
        vector2: 第二个复数向量
    
    Returns:
        dict: {
            'result': complex (内积值),
            'metadata': {
                'are_orthogonal': bool,
                'magnitude': float
            }
        }
    """
    if len(vector1) != len(vector2):
        raise ValueError(f"Vector dimensions must match: {len(vector1)} vs {len(vector2)}")
    
    vec1 = np.array(vector1, dtype=complex)
    vec2 = np.array(vector2, dtype=complex)
    
    # 内积: <v1|v2> = sum(conj(v1_i) * v2_i)
    inner_prod = complex(np.vdot(vec1, vec2))
    magnitude = abs(inner_prod)
    
    tolerance = 1e-10
    are_orthogonal = magnitude < tolerance
    
    return {
        'result': inner_prod,
        'metadata': {
            'are_orthogonal': are_orthogonal,
            'magnitude': magnitude,
            'tolerance': tolerance
        }
    }


def check_dimension_power_of_2(dimension: int) -> Dict:
    """
    检查维度是否为2的幂次（n电子系统的自旋态维度必须是2^n）
    
    Args:
        dimension: 向量维度
    
    Returns:
        dict: {
            'result': bool (是否为2的幂次),
            'metadata': {
                'dimension': int,
                'n_electrons': int or None,
                'explanation': str
            }
        }
    """
    if dimension <= 0:
        raise ValueError("Dimension must be positive")
    
    # 检查是否为2的幂次
    is_power_of_2 = (dimension & (dimension - 1)) == 0 and dimension != 0
    
    n_electrons = None
    if is_power_of_2:
        n_electrons = int(np.log2(dimension))
    
    explanation = (
        f"Dimension {dimension} = 2^{n_electrons} (valid for {n_electrons}-electron system)"
        if is_power_of_2
        else f"Dimension {dimension} is not a power of 2 (invalid for n-electron system)"
    )
    
    return {
        'result': is_power_of_2,
        'metadata': {
            'dimension': dimension,
            'n_electrons': n_electrons,
            'explanation': explanation
        }
    }


# ============================================================================
# 第二层：组合函数 - 自旋态验证
# ============================================================================

def validate_spin_state(vector_data: List[Union[str, float, int]], 
                       vector_name: str) -> Dict:
    """
    验证向量是否满足n电子系统自旋态的所有条件
    
    Args:
        vector_data: 向量数据
        vector_name: 向量名称（用于标识）
    
    Returns:
        dict: {
            'result': bool (是否为有效自旋态),
            'metadata': {
                'vector_name': str,
                'dimension': int,
                'is_normalized': bool,
                'is_power_of_2': bool,
                'n_electrons': int or None,
                'norm_value': float,
                'checks_passed': list of str,
                'checks_failed': list of str
            }
        }
    """
    checks_passed = []
    checks_failed = []
    
    # 步骤1：解析向量
    parsed = parse_complex_vector(vector_data)
    vector = parsed['result']
    dimension = parsed['metadata']['dimension']
    checks_passed.append(f"Vector parsing (dimension={dimension})")
    
    # 步骤2：检查归一化
    norm_result = compute_norm(vector)
    norm_value = norm_result['result']
    is_normalized = norm_result['metadata']['is_normalized']
    
    if is_normalized:
        checks_passed.append(f"Normalization (||ψ|| = {norm_value:.10f} ≈ 1)")
    else:
        checks_failed.append(f"Normalization (||ψ|| = {norm_value:.10f} ≠ 1)")
    
    # 步骤3：检查维度是否为2的幂次
    dim_check = check_dimension_power_of_2(dimension)
    is_power_of_2 = dim_check['result']
    n_electrons = dim_check['metadata']['n_electrons']
    
    if is_power_of_2:
        checks_passed.append(f"Dimension check (2^{n_electrons} = {dimension})")
    else:
        checks_failed.append(f"Dimension check ({dimension} is not 2^n)")
    
    # 最终判断
    is_valid = is_normalized and is_power_of_2
    
    return {
        'result': is_valid,
        'metadata': {
            'vector_name': vector_name,
            'dimension': dimension,
            'is_normalized': is_normalized,
            'is_power_of_2': is_power_of_2,
            'n_electrons': n_electrons,
            'norm_value': norm_value,
            'checks_passed': checks_passed,
            'checks_failed': checks_failed
        }
    }


def analyze_vector_set(vectors_dict: Dict[str, List[Union[str, float, int]]]) -> Dict:
    """
    分析一组向量，判断哪些可以作为n电子系统的自旋态
    
    Args:
        vectors_dict: 向量字典 {name: vector_data}
    
    Returns:
        dict: {
            'result': {
                'valid_count': int,
                'valid_vectors': list of str,
                'invalid_vectors': list of str
            },
            'metadata': {
                'total_vectors': int,
                'detailed_results': dict
            }
        }
    """
    valid_vectors = []
    invalid_vectors = []
    detailed_results = {}
    
    for name, vector_data in vectors_dict.items():
        validation = validate_spin_state(vector_data, name)
        detailed_results[name] = validation['metadata']
        
        if validation['result']:
            valid_vectors.append(name)
        else:
            invalid_vectors.append(name)
    
    return {
        'result': {
            'valid_count': len(valid_vectors),
            'valid_vectors': valid_vectors,
            'invalid_vectors': invalid_vectors
        },
        'metadata': {
            'total_vectors': len(vectors_dict),
            'detailed_results': detailed_results
        }
    }


def compute_orthogonality_matrix(vectors_dict: Dict[str, List[Union[str, float, int]]]) -> Dict:
    """
    计算向量集合的正交性矩阵
    
    Args:
        vectors_dict: 向量字典 {name: vector_data}
    
    Returns:
        dict: {
            'result': 'file_path',
            'metadata': {
                'file_type': 'txt',
                'matrix_size': tuple,
                'orthogonal_pairs': list of tuples
            }
        }
    """
    names = list(vectors_dict.keys())
    n = len(names)
    
    # 解析所有向量
    parsed_vectors = {}
    for name, vector_data in vectors_dict.items():
        parsed = parse_complex_vector(vector_data)
        parsed_vectors[name] = parsed['result']
    
    # 计算内积矩阵
    orthogonal_pairs = []
    matrix_data = []
    
    for i, name1 in enumerate(names):
        row = []
        for j, name2 in enumerate(names):
            if i == j:
                # 对角线：范数
                norm_result = compute_norm(parsed_vectors[name1])
                value = norm_result['result']
                row.append(f"{value:.6f}")
            else:
                # 非对角线：内积
                vec1 = parsed_vectors[name1]
                vec2 = parsed_vectors[name2]
                
                # 检查维度是否相同
                if len(vec1) == len(vec2):
                    inner_prod = compute_inner_product(vec1, vec2)
                    magnitude = inner_prod['metadata']['magnitude']
                    row.append(f"{magnitude:.6e}")
                    
                    if inner_prod['metadata']['are_orthogonal']:
                        orthogonal_pairs.append((name1, name2))
                else:
                    row.append("N/A (diff dim)")
        
        matrix_data.append(row)
    
    # 保存到文件
    filepath = "./mid_result/physics/orthogonality_matrix.txt"
    with open(filepath, 'w') as f:
        f.write("Orthogonality Matrix (|<vi|vj>|)\n")
        f.write("=" * 60 + "\n\n")
        
        # 表头
        f.write(f"{'':>10}")
        for name in names:
            f.write(f"{name:>12}")
        f.write("\n" + "-" * 60 + "\n")
        
        # 矩阵内容
        for i, name in enumerate(names):
            f.write(f"{name:>10}")
            for value in matrix_data[i]:
                f.write(f"{value:>12}")
            f.write("\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Orthogonal pairs: {orthogonal_pairs}\n")
    
    print(f"FILE_GENERATED: txt | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'txt',
            'matrix_size': (n, n),
            'orthogonal_pairs': orthogonal_pairs,
            'vector_names': names
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_validation_results(analysis_result: Dict, 
                                 save_path: str = "./tool_images/spin_state_validation.png") -> Dict:
    """
    可视化自旋态验证结果
    
    Args:
        analysis_result: analyze_vector_set的返回结果
        save_path: 图像保存路径
    
    Returns:
        dict: {
            'result': str (图像路径),
            'metadata': {
                'image_type': str,
                'vectors_analyzed': int
            }
        }
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    detailed = analysis_result['metadata']['detailed_results']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1：验证结果总览
    valid_count = analysis_result['result']['valid_count']
    invalid_count = len(analysis_result['result']['invalid_vectors'])
    
    colors = ['#2ecc71', '#e74c3c']
    labels = [f'Valid Spin States\n({valid_count})', 
              f'Invalid\n({invalid_count})']
    sizes = [valid_count, invalid_count]
    
    if valid_count > 0 or invalid_count > 0:
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax1.set_title('Spin State Validation Results', fontsize=14, weight='bold')
    
    # 子图2：详细检查结果
    vector_names = list(detailed.keys())
    y_pos = np.arange(len(vector_names))
    
    # 准备数据
    dimensions = [detailed[name]['dimension'] for name in vector_names]
    is_normalized = [detailed[name]['is_normalized'] for name in vector_names]
    is_power_of_2 = [detailed[name]['is_power_of_2'] for name in vector_names]
    
    bar_width = 0.25
    
    # 绘制条形图
    bars1 = ax2.barh(y_pos - bar_width, dimensions, bar_width, 
                     label='Dimension', color='#3498db', alpha=0.8)
    
    norm_values = [1 if n else 0 for n in is_normalized]
    bars2 = ax2.barh(y_pos, norm_values, bar_width,
                     label='Normalized', color='#2ecc71', alpha=0.8)
    
    power2_values = [1 if p else 0 for p in is_power_of_2]
    bars3 = ax2.barh(y_pos + bar_width, power2_values, bar_width,
                     label='Power of 2', color='#9b59b6', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(vector_names, fontsize=11)
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_title('Detailed Validation Checks', fontsize=14, weight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加n-electron标注
    for i, name in enumerate(vector_names):
        n_electrons = detailed[name]['n_electrons']
        if n_electrons is not None:
            ax2.text(max(dimensions) * 0.5, i, f'n={n_electrons}',
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'image_type': 'validation_summary',
            'vectors_analyzed': len(vector_names),
            'format': 'png'
        }
    }


def create_dimension_analysis_plot(vectors_dict: Dict[str, List[Union[str, float, int]]],
                                  save_path: str = "./tool_images/dimension_analysis.png") -> Dict:
    """
    创建维度分析图，展示各向量维度与n电子系统的关系
    
    Args:
        vectors_dict: 向量字典
        save_path: 图像保存路径
    
    Returns:
        dict: {
            'result': str (图像路径),
            'metadata': {
                'image_type': str
            }
        }
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 准备数据
    names = []
    dimensions = []
    colors = []
    n_electrons_list = []
    
    for name, vector_data in vectors_dict.items():
        parsed = parse_complex_vector(vector_data)
        dim = parsed['metadata']['dimension']
        dim_check = check_dimension_power_of_2(dim)
        
        names.append(name)
        dimensions.append(dim)
        n_electrons_list.append(dim_check['metadata']['n_electrons'])
        
        if dim_check['result']:
            colors.append('#2ecc71')  # 绿色：有效
        else:
            colors.append('#e74c3c')  # 红色：无效
    
    # 绘制条形图
    bars = ax.bar(names, dimensions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 添加2^n参考线
    max_dim = max(dimensions)
    n_max = int(np.ceil(np.log2(max_dim))) + 1
    
    for n in range(n_max + 1):
        power_of_2 = 2 ** n
        if power_of_2 <= max_dim * 1.2:
            ax.axhline(y=power_of_2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(len(names) - 0.5, power_of_2, f'2^{n}={power_of_2}',
                   ha='right', va='bottom', fontsize=9, color='gray')
    
    # 标注n值
    for i, (bar, n_e) in enumerate(zip(bars, n_electrons_list)):
        height = bar.get_height()
        if n_e is not None:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={n_e}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   'Invalid',
                   ha='center', va='bottom', fontsize=10, color='red')
    
    ax.set_xlabel('Vector Name', fontsize=13, weight='bold')
    ax.set_ylabel('Dimension', fontsize=13, weight='bold')
    ax.set_title('Vector Dimensions vs n-Electron System Requirements\n(Valid dimensions must be 2^n)',
                fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 图例
    valid_patch = mpatches.Patch(color='#2ecc71', label='Valid (2^n)')
    invalid_patch = mpatches.Patch(color='#e74c3c', label='Invalid')
    ax.legend(handles=[valid_patch, invalid_patch], loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'image_type': 'dimension_analysis',
            'format': 'png'
        }
    }


# ============================================================================
# 文件解析函数
# ============================================================================

def load_file(filepath: str) -> Dict:
    """
    加载并解析文本文件内容
    
    Args:
        filepath: 文件路径
    
    Returns:
        dict: {
            'result': str (文件内容),
            'metadata': {
                'file_size': int,
                'line_count': int
            }
        }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    line_count = len(content.split('\n'))
    file_size = os.path.getsize(filepath)
    
    return {
        'result': content,
        'metadata': {
            'file_size': file_size,
            'line_count': line_count,
            'filepath': filepath
        }
    }


# ============================================================================
# 主函数：3个场景演示
# ============================================================================

def main():
    """
    演示量子自旋态分析工具包的三个场景
    """
    
    # 定义问题中的向量
    W = ["1/sqrt(2)", "-i/sqrt(2)"]
    X = ["1/sqrt(3)", "i/sqrt(3)", "-i/sqrt(3)"]
    Y = ["1/2", "-1/2", "1/2", "-1/2"]
    Z = ["-1/sqrt(5)", "sqrt(2/5)", "0", "sqrt(1/5)", "i/sqrt(5)"]
    
    vectors = {
        'W': W,
        'X': X,
        'Y': Y,
        'Z': Z
    }
    
    print("=" * 80)
    print("场景1：分析原始问题 - 判断哪些向量可以作为n电子系统的自旋态")
    print("=" * 80)
    print("问题描述：给定4个复数向量W, X, Y, Z，判断哪些可以作为n电子系统的自旋态")
    print("自旋态必须满足：1) 归一化 ||ψ||=1  2) 维度为2^n (n≥1)")
    print("-" * 80)
    
    # 步骤1：逐个验证向量
    print("\n步骤1：逐个验证向量的自旋态条件")
    for name, vector_data in vectors.items():
        print(f"\n--- 验证向量 {name} ---")
        # 调用函数：validate_spin_state()
        result = validate_spin_state(vector_data, name)
        print(f"FUNCTION_CALL: validate_spin_state | PARAMS: {{'vector_name': '{name}', 'dimension': {len(vector_data)}}} | RESULT: {result}")
        
        metadata = result['metadata']
        print(f"  维度: {metadata['dimension']}")
        print(f"  归一化: {metadata['is_normalized']} (||ψ|| = {metadata['norm_value']:.10f})")
        print(f"  2的幂次: {metadata['is_power_of_2']}")
        if metadata['n_electrons'] is not None:
            print(f"  对应电子数: n = {metadata['n_electrons']}")
        print(f"  结论: {'✓ 有效自旋态' if result['result'] else '✗ 无效自旋态'}")
    
    # 步骤2：汇总分析
    print("\n" + "-" * 80)
    print("步骤2：汇总分析所有向量")
    # 调用函数：analyze_vector_set()
    analysis = analyze_vector_set(vectors)
    print(f"FUNCTION_CALL: analyze_vector_set | PARAMS: {{'vector_count': {len(vectors)}}} | RESULT: {analysis}")
    
    valid_count = analysis['result']['valid_count']
    valid_vectors = analysis['result']['valid_vectors']
    invalid_vectors = analysis['result']['invalid_vectors']
    
    print(f"\n有效自旋态数量: {valid_count}")
    print(f"有效向量: {valid_vectors}")
    print(f"无效向量: {invalid_vectors}")
    
    # 步骤3：生成可视化
    print("\n" + "-" * 80)
    print("步骤3：生成验证结果可视化")
    # 调用函数：visualize_validation_results()
    viz_result = visualize_validation_results(analysis)
    print(f"FUNCTION_CALL: visualize_validation_results | PARAMS: {{'analysis_result': 'dict'}} | RESULT: {viz_result}")
    
    # 调用函数：create_dimension_analysis_plot()
    dim_plot = create_dimension_analysis_plot(vectors)
    print(f"FUNCTION_CALL: create_dimension_analysis_plot | PARAMS: {{'vectors_dict': 'dict'}} | RESULT: {dim_plot}")
    
    # 步骤4：计算正交性矩阵
    print("\n" + "-" * 80)
    print("步骤4：计算向量间的正交性")
    # 调用函数：compute_orthogonality_matrix()
    ortho_result = compute_orthogonality_matrix(vectors)
    print(f"FUNCTION_CALL: compute_orthogonality_matrix | PARAMS: {{'vectors_dict': 'dict'}} | RESULT: {ortho_result}")
    
    # 调用函数：load_file() 读取正交性矩阵
    file_content = load_file(ortho_result['result'])
    print(f"FUNCTION_CALL: load_file | PARAMS: {{'filepath': '{ortho_result['result']}'}} | RESULT: {{'line_count': {file_content['metadata']['line_count']}}}")
    print("\n正交性矩阵内容：")
    print(file_content['result'])
    
    print("\n" + "=" * 80)
    print(f"FINAL_ANSWER: {valid_count} of the vectors can be the spin states of some n-electron system for n>=1")
    print("=" * 80)
    
    
    # ========================================================================
    print("\n\n")
    print("=" * 80)
    print("场景2：验证单电子自旋态（n=1）的标准基矢")
    print("=" * 80)
    print("问题描述：验证单电子自旋向上|↑⟩和向下|↓⟩态是否满足自旋态条件")
    print("-" * 80)
    
    # 单电子自旋态的标准基矢
    spin_up = ["1", "0"]      # |↑⟩ = (1, 0)
    spin_down = ["0", "1"]    # |↓⟩ = (0, 1)
    
    single_electron_states = {
        'spin_up': spin_up,
        'spin_down': spin_down
    }
    
    print("\n步骤1：验证单电子自旋态")
    for name, vector_data in single_electron_states.items():
        print(f"\n--- 验证 {name} ---")
        # 调用函数：validate_spin_state()
        result = validate_spin_state(vector_data, name)
        print(f"FUNCTION_CALL: validate_spin_state | PARAMS: {{'vector_name': '{name}'}} | RESULT: {result}")
        print(f"  维度: {result['metadata']['dimension']} = 2^1 (单电子)")
        print(f"  归一化: {result['metadata']['is_normalized']}")
        print(f"  结论: {'✓ 有效' if result['result'] else '✗ 无效'}")
    
    print("\n" + "-" * 80)
    print("步骤2：计算自旋向上和向下态的内积（验证正交性）")
    parsed_up = parse_complex_vector(spin_up)
    parsed_down = parse_complex_vector(spin_down)
    # 调用函数：compute_inner_product()
    inner_prod = compute_inner_product(parsed_up['result'], parsed_down['result'])
    print(f"FUNCTION_CALL: compute_inner_product | PARAMS: {{'vec1': 'spin_up', 'vec2': 'spin_down'}} | RESULT: {inner_prod}")
    print(f"  内积 ⟨↑|↓⟩ = {inner_prod['result']}")
    print(f"  正交性: {inner_prod['metadata']['are_orthogonal']}")
    
    print("\n" + "-" * 80)
    print("步骤3：汇总分析")
    # 调用函数：analyze_vector_set()
    analysis_s2 = analyze_vector_set(single_electron_states)
    print(f"FUNCTION_CALL: analyze_vector_set | PARAMS: {{'vector_count': 2}} | RESULT: {analysis_s2}")
    
    print(f"\n有效自旋态数量: {analysis_s2['result']['valid_count']}")
    
    print(f"\nFINAL_ANSWER: Both single-electron spin states are valid (n=1), total valid count = {analysis_s2['result']['valid_count']}")
    
    
    # ========================================================================
    print("\n\n")
    print("=" * 80)
    print("场景3：分析双电子系统的Bell态")
    print("=" * 80)
    print("问题描述：验证量子纠缠态（Bell态）是否满足双电子自旋态条件")
    print("-" * 80)
    
    # Bell态：最大纠缠态
    # |Φ+⟩ = (|↑↑⟩ + |↓↓⟩)/√2 = (1/√2, 0, 0, 1/√2)
    # |Ψ-⟩ = (|↑↓⟩ - |↓↑⟩)/√2 = (0, 1/√2, -1/√2, 0)
    
    bell_phi_plus = ["1/sqrt(2)", "0", "0", "1/sqrt(2)"]
    bell_psi_minus = ["0", "1/sqrt(2)", "-1/sqrt(2)", "0"]
    
    # 非归一化的态（用于对比）
    non_normalized = ["1", "0", "0", "1"]
    
    bell_states = {
        'Bell_Phi_plus': bell_phi_plus,
        'Bell_Psi_minus': bell_psi_minus,
        'Non_normalized': non_normalized
    }
    
    print("\n步骤1：验证Bell态和非归一化态")
    for name, vector_data in bell_states.items():
        print(f"\n--- 验证 {name} ---")
        # 调用函数：validate_spin_state()
        result = validate_spin_state(vector_data, name)
        print(f"FUNCTION_CALL: validate_spin_state | PARAMS: {{'vector_name': '{name}'}} | RESULT: {result}")
        print(f"  维度: {result['metadata']['dimension']} = 2^2 (双电子)")
        print(f"  归一化: {result['metadata']['is_normalized']} (||ψ|| = {result['metadata']['norm_value']:.6f})")
        print(f"  结论: {'✓ 有效' if result['result'] else '✗ 无效'}")
    
    print("\n" + "-" * 80)
    print("步骤2：计算两个Bell态之间的内积（验证正交性）")
    parsed_phi = parse_complex_vector(bell_phi_plus)
    parsed_psi = parse_complex_vector(bell_psi_minus)
    # 调用函数：compute_inner_product()
    inner_prod_bell = compute_inner_product(parsed_phi['result'], parsed_psi['result'])
    print(f"FUNCTION_CALL: compute_inner_product | PARAMS: {{'vec1': 'Phi+', 'vec2': 'Psi-'}} | RESULT: {inner_prod_bell}")
    print(f"  内积 ⟨Φ+|Ψ-⟩ = {inner_prod_bell['result']}")
    print(f"  正交性: {inner_prod_bell['metadata']['are_orthogonal']}")
    
    print("\n" + "-" * 80)
    print("步骤3：汇总分析Bell态集合")
    # 调用函数：analyze_vector_set()
    analysis_s3 = analyze_vector_set(bell_states)
    print(f"FUNCTION_CALL: analyze_vector_set | PARAMS: {{'vector_count': 3}} | RESULT: {analysis_s3}")
    
    valid_count_s3 = analysis_s3['result']['valid_count']
    print(f"\n有效自旋态数量: {valid_count_s3}")
    print(f"有效向量: {analysis_s3['result']['valid_vectors']}")
    print(f"无效向量: {analysis_s3['result']['invalid_vectors']}")
    
    print("\n" + "-" * 80)
    print("步骤4：生成Bell态可视化")
    # 调用函数：visualize_validation_results()
    viz_s3 = visualize_validation_results(analysis_s3, 
                                         save_path="./tool_images/bell_states_validation.png")
    print(f"FUNCTION_CALL: visualize_validation_results | PARAMS: {{'save_path': 'bell_states_validation.png'}} | RESULT: {viz_s3}")
    
    print(f"\nFINAL_ANSWER: {valid_count_s3} Bell states are valid 2-electron spin states (n=2)")


if __name__ == "__main__":
    main()