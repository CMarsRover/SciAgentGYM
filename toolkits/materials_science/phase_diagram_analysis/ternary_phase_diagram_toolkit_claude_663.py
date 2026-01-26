# Filename: ternary_phase_diagram_toolkit.py

"""
三元共晶相图分析工具包
用于分析三元系统的相图、共晶点、杠杆定律计算等
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/materials', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def validate_composition(w_a: float, w_b: float, w_c: float) -> Dict:
    """
    验证三元组分质量分数是否有效
    
    Args:
        w_a: 组分A的质量分数 (0-100)
        w_b: 组分B的质量分数 (0-100)
        w_c: 组分C的质量分数 (0-100)
    
    Returns:
        dict: {'result': bool, 'metadata': {'sum': float, 'valid': bool}}
    """
    if not all(0 <= w <= 100 for w in [w_a, w_b, w_c]):
        raise ValueError("质量分数必须在0-100之间")
    
    total = w_a + w_b + w_c
    is_valid = abs(total - 100.0) < 1e-6
    
    return {
        'result': is_valid,
        'metadata': {
            'sum': total,
            'valid': is_valid,
            'components': {'A': w_a, 'B': w_b, 'C': w_c}
        }
    }


def identify_lowest_melting_point(eutectic_composition: Dict[str, float], 
                                   phase_diagram_type: str = "ternary_eutectic") -> Dict:
    """
    根据共晶点组成判断熔点最低的组分
    
    在三元共晶系统中，共晶点组成中含量最高的组分通常具有最低的熔点，
    因为该组分在共晶温度下最容易形成液相。
    
    Args:
        eutectic_composition: 共晶点组成 {'A': float, 'B': float, 'C': float}
        phase_diagram_type: 相图类型
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    if not isinstance(eutectic_composition, dict):
        raise TypeError("eutectic_composition必须是字典类型")
    
    # 找出含量最高的组分
    max_component = max(eutectic_composition.items(), key=lambda x: x[1])
    
    return {
        'result': max_component[0],
        'metadata': {
            'composition': eutectic_composition,
            'max_fraction': max_component[1],
            'reasoning': f"组分{max_component[0]}在共晶点含量最高({max_component[1]}%)，表明其熔点最低"
        }
    }


def calculate_tie_line_ratio(point_composition: Dict[str, float],
                             phase1_composition: Dict[str, float],
                             phase2_composition: Dict[str, float]) -> Dict:
    """
    使用杠杆定律计算两相区中各相的质量分数
    
    Args:
        point_composition: 系统总组成 {'A': float, 'B': float, 'C': float}
        phase1_composition: 相1的组成
        phase2_composition: 相2的组成
    
    Returns:
        dict: {'result': {'phase1_fraction': float, 'phase2_fraction': float}, 'metadata': {...}}
    """
    # 简化为二维问题（使用A和B组分）
    x_point = point_composition['A']
    x_phase1 = phase1_composition['A']
    x_phase2 = phase2_composition['A']
    
    # 杠杆定律: w1/w2 = (x2-x)/(x-x1)
    if abs(x_phase2 - x_phase1) < 1e-6:
        raise ValueError("两相组成不能相同")
    
    phase2_fraction = (x_point - x_phase1) / (x_phase2 - x_phase1)
    phase1_fraction = 1.0 - phase2_fraction
    
    # 确保分数在合理范围内
    phase1_fraction = max(0, min(1, phase1_fraction))
    phase2_fraction = max(0, min(1, phase2_fraction))
    
    return {
        'result': {
            'phase1_fraction': phase1_fraction * 100,
            'phase2_fraction': phase2_fraction * 100
        },
        'metadata': {
            'point': point_composition,
            'phase1': phase1_composition,
            'phase2': phase2_composition,
            'calculation_method': 'lever_rule'
        }
    }


def convert_ternary_to_cartesian(w_a: float, w_b: float, w_c: float) -> Dict:
    """
    将三元组分转换为笛卡尔坐标（用于绘图）
    
    Args:
        w_a: 组分A的质量分数 (0-100)
        w_b: 组分B的质量分数 (0-100)
        w_c: 组分C的质量分数 (0-100)
    
    Returns:
        dict: {'result': {'x': float, 'y': float}, 'metadata': {...}}
    """
    # 归一化
    total = w_a + w_b + w_c
    a = w_a / total
    b = w_b / total
    c = w_c / total
    
    # 等边三角形坐标转换
    x = 0.5 * (2 * b + c)
    y = (np.sqrt(3) / 2) * c
    
    return {
        'result': {'x': x, 'y': y},
        'metadata': {
            'original_composition': {'A': w_a, 'B': w_b, 'C': w_c},
            'normalized': {'A': a, 'B': b, 'C': c}
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_eutectic_point(eutectic_comp: Dict[str, float]) -> Dict:
    """
    分析共晶点的完整信息
    
    Args:
        eutectic_comp: 共晶点组成 {'A': float, 'B': float, 'C': float}
    
    Returns:
        dict: {'result': {...}, 'metadata': {...}}
    """
    # 验证组成
    validation = validate_composition(
        eutectic_comp['A'], 
        eutectic_comp['B'], 
        eutectic_comp['C']
    )
    
    if not validation['result']:
        raise ValueError("共晶点组成无效")
    
    # 识别最低熔点组分
    lowest_mp = identify_lowest_melting_point(eutectic_comp)
    
    # 转换为笛卡尔坐标
    coords = convert_ternary_to_cartesian(
        eutectic_comp['A'],
        eutectic_comp['B'],
        eutectic_comp['C']
    )
    
    return {
        'result': {
            'composition': eutectic_comp,
            'lowest_melting_component': lowest_mp['result'],
            'coordinates': coords['result']
        },
        'metadata': {
            'validation': validation['metadata'],
            'melting_point_analysis': lowest_mp['metadata']
        }
    }


def calculate_eutectic_fraction_in_alloy(alloy_comp: Dict[str, float],
                                         eutectic_comp: Dict[str, float],
                                         primary_phase: str) -> Dict:
    """
    计算合金中共晶混合物的质量分数
    
    对于三元共晶系统，当合金组成不在共晶点时，会先析出初晶相，
    剩余液相沿着液相线变化，最终在共晶温度下形成共晶混合物。
    
    Args:
        alloy_comp: 合金总组成 {'A': float, 'B': float, 'C': float}
        eutectic_comp: 共晶点组成 {'A': float, 'B': float, 'C': float}
        primary_phase: 初晶相 ('A', 'B', 或 'C')
    
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    # 验证组成
    validate_composition(alloy_comp['A'], alloy_comp['B'], alloy_comp['C'])
    validate_composition(eutectic_comp['A'], eutectic_comp['B'], eutectic_comp['C'])
    
    # 对于三元系统，使用质量守恒原理
    # 设共晶混合物质量分数为 f_e，初晶相质量分数为 f_p
    # f_e + f_p = 1
    # 对于关键组分的质量守恒：
    # w_alloy = f_p * w_primary + f_e * w_eutectic
    
    # 选择差异最大的组分进行计算
    if primary_phase not in ['A', 'B', 'C']:
        raise ValueError("初晶相必须是A、B或C")
    
    # 初晶相的组成（纯组分）
    primary_comp = {primary_phase: 100.0}
    for comp in ['A', 'B', 'C']:
        if comp != primary_phase:
            primary_comp[comp] = 0.0
    
    # 使用非初晶相组分进行计算（避免除零）
    calc_component = None
    for comp in ['A', 'B', 'C']:
        if comp != primary_phase and eutectic_comp[comp] > 1e-6:
            calc_component = comp
            break
    
    if calc_component is None:
        raise ValueError("无法找到合适的计算组分")
    
    # 质量守恒方程：w_alloy = f_p * w_primary + f_e * w_eutectic
    # 其中 f_p + f_e = 1，所以 f_p = 1 - f_e
    # w_alloy = (1 - f_e) * w_primary + f_e * w_eutectic
    # w_alloy = w_primary - f_e * w_primary + f_e * w_eutectic
    # w_alloy - w_primary = f_e * (w_eutectic - w_primary)
    # f_e = (w_alloy - w_primary) / (w_eutectic - w_primary)
    
    w_alloy = alloy_comp[calc_component]
    w_primary = primary_comp[calc_component]
    w_eutectic = eutectic_comp[calc_component]
    
    if abs(w_eutectic - w_primary) < 1e-6:
        raise ValueError("共晶组成与初晶相组成在计算组分上相同")
    
    eutectic_fraction = (w_alloy - w_primary) / (w_eutectic - w_primary)
    eutectic_fraction = max(0, min(1, eutectic_fraction))  # 限制在[0,1]
    
    return {
        'result': eutectic_fraction * 100,
        'metadata': {
            'alloy_composition': alloy_comp,
            'eutectic_composition': eutectic_comp,
            'primary_phase': primary_phase,
            'calculation_component': calc_component,
            'primary_phase_fraction': (1 - eutectic_fraction) * 100
        }
    }


def determine_primary_phase_from_diagram(alloy_comp: Dict[str, float],
                                         eutectic_comp: Dict[str, float]) -> Dict:
    """
    根据合金组成和相图特征确定初晶相
    
    在三元共晶相图中，初晶相由合金组成点所在的初晶区决定。
    通过比较合金组成与共晶点组成的相对位置来判断。
    
    Args:
        alloy_comp: 合金组成 {'A': float, 'B': float, 'C': float}
        eutectic_comp: 共晶点组成 {'A': float, 'B': float, 'C': float}
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    # 计算各组分相对于共晶点的偏离
    deviations = {}
    for comp in ['A', 'B', 'C']:
        deviations[comp] = alloy_comp[comp] - eutectic_comp[comp]
    
    # 偏离最大的组分对应的初晶区
    max_deviation_comp = max(deviations.items(), key=lambda x: x[1])
    
    return {
        'result': max_deviation_comp[0],
        'metadata': {
            'deviations': deviations,
            'reasoning': f"组分{max_deviation_comp[0]}相对共晶点偏离最大({max_deviation_comp[1]:.1f}%)，位于{max_deviation_comp[0]}初晶区"
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_ternary_phase_diagram(eutectic_comp: Dict[str, float],
                               alloy_comp: Dict[str, float] = None,
                               save_path: str = None) -> Dict:
    """
    绘制三元相图的浓度三角形
    
    Args:
        eutectic_comp: 共晶点组成 {'A': float, 'B': float, 'C': float}
        alloy_comp: 合金组成（可选）
        save_path: 保存路径
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    if save_path is None:
        save_path = './tool_images/ternary_phase_diagram.png'
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 绘制等边三角形
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)
    
    # 标注顶点
    ax.text(-0.05, -0.05, 'B', fontsize=14, fontweight='bold')
    ax.text(1.02, -0.05, 'C', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'A', fontsize=14, fontweight='bold')
    
    # 绘制共晶点
    e_coords = convert_ternary_to_cartesian(
        eutectic_comp['A'], eutectic_comp['B'], eutectic_comp['C']
    )
    ax.plot(e_coords['result']['x'], e_coords['result']['y'], 'ro', markersize=10, label='Eutectic Point E')
    ax.text(e_coords['result']['x'] + 0.05, e_coords['result']['y'], 
            f"E\nA:{eutectic_comp['A']:.0f}%\nB:{eutectic_comp['B']:.0f}%\nC:{eutectic_comp['C']:.0f}%",
            fontsize=9)
    
    # 绘制合金点（如果提供）
    if alloy_comp:
        a_coords = convert_ternary_to_cartesian(
            alloy_comp['A'], alloy_comp['B'], alloy_comp['C']
        )
        ax.plot(a_coords['result']['x'], a_coords['result']['y'], 'bs', markersize=10, label='Alloy Composition')
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title('Ternary Eutectic Phase Diagram', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'eutectic_point': eutectic_comp,
            'alloy_point': alloy_comp
        }
    }


def generate_phase_analysis_report(eutectic_comp: Dict[str, float],
                                   alloy_comp: Dict[str, float],
                                   analysis_results: Dict) -> Dict:
    """
    生成相图分析报告
    
    Args:
        eutectic_comp: 共晶点组成
        alloy_comp: 合金组成
        analysis_results: 分析结果字典
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    report_path = './mid_result/materials/phase_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("三元共晶相图分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 共晶点信息\n")
        f.write(f"   组成: A={eutectic_comp['A']:.1f}%, B={eutectic_comp['B']:.1f}%, C={eutectic_comp['C']:.1f}%\n")
        f.write(f"   最低熔点组分: {analysis_results.get('lowest_melting_component', 'N/A')}\n\n")
        
        f.write("2. 合金组成\n")
        f.write(f"   组成: A={alloy_comp['A']:.1f}%, B={alloy_comp['B']:.1f}%, C={alloy_comp['C']:.1f}%\n")
        f.write(f"   初晶相: {analysis_results.get('primary_phase', 'N/A')}\n\n")
        
        f.write("3. 共晶混合物含量\n")
        f.write(f"   质量分数: {analysis_results.get('eutectic_fraction', 0):.1f}%\n")
        f.write(f"   初晶相含量: {analysis_results.get('primary_phase_fraction', 0):.1f}%\n\n")
        
        f.write("=" * 60 + "\n")
    
    return {
        'result': report_path,
        'metadata': {
            'file_type': 'txt',
            'encoding': 'utf-8'
        }
    }


# ==================== 主函数 ====================

def main():
    """
    演示三个场景的相图分析
    """
    
    # ========== 场景1：解决原始问题 ==========
    print("=" * 60)
    print("场景1：三元共晶相图分析 - 原始问题求解")
    print("=" * 60)
    print("问题描述：给定三元共晶点E的组成为w(A)=20%, w(B)=30%, w(C)=50%")
    print("(1) 判断A、B、C三个组分中哪个熔点最低")
    print("(2) 计算某合金铸锭中共晶混合物的质量分数")
    print("-" * 60)
    
    # 定义共晶点组成
    eutectic_composition = {'A': 20.0, 'B': 30.0, 'C': 50.0}
    
    # 步骤1：分析共晶点，确定最低熔点组分
    print("\n步骤1：分析共晶点组成")
    print(f"调用函数：analyze_eutectic_point()")
    eutectic_analysis = analyze_eutectic_point(eutectic_composition)
    print(f"FUNCTION_CALL: analyze_eutectic_point | PARAMS: {eutectic_composition} | RESULT: {eutectic_analysis}")
    
    lowest_mp_component = eutectic_analysis['result']['lowest_melting_component']
    print(f"\n问题(1)答案：组分{lowest_mp_component}的熔点最低")
    
    # 步骤2：根据图(b)分析，假设合金组成点在A初晶区
    # 从图(b)可以看出，存在A+B+L和L+C区域，说明合金不在共晶点
    # 假设合金组成为 A=60%, B=10%, C=30%（位于A初晶区，沿图示AC截线靠近A端）
    alloy_composition = {'A': 60.0, 'B': 10.0, 'C': 30.0}
    
    print(f"\n步骤2：确定合金组成和初晶相")
    print(f"合金组成：{alloy_composition}")
    print(f"调用函数：determine_primary_phase_from_diagram()")
    primary_phase_result = determine_primary_phase_from_diagram(alloy_composition, eutectic_composition)
    print(f"FUNCTION_CALL: determine_primary_phase_from_diagram | PARAMS: alloy={alloy_composition}, eutectic={eutectic_composition} | RESULT: {primary_phase_result}")
    
    primary_phase = primary_phase_result['result']
    
    # 步骤3：计算共晶混合物质量分数
    print(f"\n步骤3：计算共晶混合物质量分数")
    print(f"调用函数：calculate_eutectic_fraction_in_alloy()")
    eutectic_fraction_result = calculate_eutectic_fraction_in_alloy(
        alloy_composition, 
        eutectic_composition, 
        primary_phase
    )
    print(f"FUNCTION_CALL: calculate_eutectic_fraction_in_alloy | PARAMS: alloy={alloy_composition}, eutectic={eutectic_composition}, primary={primary_phase} | RESULT: {eutectic_fraction_result}")
    
    eutectic_fraction = eutectic_fraction_result['result']
    print(f"\n问题(2)答案：共晶混合物的质量分数为 {eutectic_fraction:.1f}%")
    
    # 步骤4：生成可视化
    print(f"\n步骤4：生成相图可视化")
    print(f"调用函数：plot_ternary_phase_diagram()")
    plot_result = plot_ternary_phase_diagram(eutectic_composition, alloy_composition)
    print(f"FUNCTION_CALL: plot_ternary_phase_diagram | PARAMS: eutectic={eutectic_composition}, alloy={alloy_composition} | RESULT: {plot_result}")
    
    print(f"\nFINAL_ANSWER: (1) {lowest_mp_component}  (2) {eutectic_fraction:.1f}%")
    
    
    # ========== 场景2：不同合金组成的共晶分数计算 ==========
    print("\n\n" + "=" * 60)
    print("场景2：不同初晶区合金的共晶分数分析")
    print("=" * 60)
    print("问题描述：分析位于B初晶区的合金(A=15%, B=60%, C=25%)中共晶混合物的含量")
    print("-" * 60)
    
    alloy_b = {'A': 15.0, 'B': 60.0, 'C': 25.0}
    
    print(f"\n步骤1：验证合金组成")
    print(f"调用函数：validate_composition()")
    validation_b = validate_composition(alloy_b['A'], alloy_b['B'], alloy_b['C'])
    print(f"FUNCTION_CALL: validate_composition | PARAMS: A={alloy_b['A']}, B={alloy_b['B']}, C={alloy_b['C']} | RESULT: {validation_b}")
    
    print(f"\n步骤2：确定初晶相")
    print(f"调用函数：determine_primary_phase_from_diagram()")
    primary_b = determine_primary_phase_from_diagram(alloy_b, eutectic_composition)
    print(f"FUNCTION_CALL: determine_primary_phase_from_diagram | PARAMS: alloy={alloy_b}, eutectic={eutectic_composition} | RESULT: {primary_b}")
    
    print(f"\n步骤3：计算共晶分数")
    print(f"调用函数：calculate_eutectic_fraction_in_alloy()")
    eutectic_b = calculate_eutectic_fraction_in_alloy(alloy_b, eutectic_composition, primary_b['result'])
    print(f"FUNCTION_CALL: calculate_eutectic_fraction_in_alloy | PARAMS: alloy={alloy_b}, eutectic={eutectic_composition}, primary={primary_b['result']} | RESULT: {eutectic_b}")
    
    print(f"\nFINAL_ANSWER: 初晶相为{primary_b['result']}, 共晶混合物质量分数为{eutectic_b['result']:.1f}%")
    
    
    # ========== 场景3：共晶点附近合金的相分析 ==========
    print("\n\n" + "=" * 60)
    print("场景3：接近共晶点组成的合金分析")
    print("=" * 60)
    print("问题描述：分析组成接近共晶点的合金(A=22%, B=28%, C=50%)的凝固行为")
    print("-" * 60)
    
    alloy_near_e = {'A': 22.0, 'B': 28.0, 'C': 50.0}
    
    print(f"\n步骤1：分析合金与共晶点的偏离")
    print(f"调用函数：determine_primary_phase_from_diagram()")
    primary_near = determine_primary_phase_from_diagram(alloy_near_e, eutectic_composition)
    print(f"FUNCTION_CALL: determine_primary_phase_from_diagram | PARAMS: alloy={alloy_near_e}, eutectic={eutectic_composition} | RESULT: {primary_near}")
    
    print(f"\n步骤2：计算共晶分数")
    print(f"调用函数：calculate_eutectic_fraction_in_alloy()")
    eutectic_near = calculate_eutectic_fraction_in_alloy(alloy_near_e, eutectic_composition, primary_near['result'])
    print(f"FUNCTION_CALL: calculate_eutectic_fraction_in_alloy | PARAMS: alloy={alloy_near_e}, eutectic={eutectic_composition}, primary={primary_near['result']} | RESULT: {eutectic_near}")
    
    print(f"\n步骤3：生成分析报告")
    print(f"调用函数：generate_phase_analysis_report()")
    analysis_data = {
        'lowest_melting_component': lowest_mp_component,
        'primary_phase': primary_near['result'],
        'eutectic_fraction': eutectic_near['result'],
        'primary_phase_fraction': eutectic_near['metadata']['primary_phase_fraction']
    }
    report = generate_phase_analysis_report(eutectic_composition, alloy_near_e, analysis_data)
    print(f"FUNCTION_CALL: generate_phase_analysis_report | PARAMS: eutectic={eutectic_composition}, alloy={alloy_near_e}, analysis={analysis_data} | RESULT: {report}")
    
    print(f"\nFINAL_ANSWER: 合金接近共晶点，初晶相{primary_near['result']}含量仅{eutectic_near['metadata']['primary_phase_fraction']:.1f}%，共晶混合物占{eutectic_near['result']:.1f}%")


if __name__ == "__main__":
    main()