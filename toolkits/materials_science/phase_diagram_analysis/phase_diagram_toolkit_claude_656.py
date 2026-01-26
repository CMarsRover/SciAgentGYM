# Filename: phase_diagram_toolkit.py
"""
Materials Science Toolkit for Binary Phase Diagram Analysis
专注于二元相图分析，特别是含不一致熔化化合物的共晶系统

核心功能：
1. 杠杆定律计算（Lever Rule）
2. 初晶析出量计算
3. 相图组成分析
4. 可视化相图和计算结果
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# 配置matplotlib字体，避免中英文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 全局常量
MID_RESULT_DIR = Path("./mid_result/materials")
IMAGE_DIR = Path("./tool_images")

# 创建必要的目录
MID_RESULT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 第一层：原子函数 ====================

def lever_rule_calculation(
    overall_composition: float,
    phase1_composition: float,
    phase2_composition: float
) -> Dict:
    """
    应用杠杆定律计算两相的质量分数
    
    Lever Rule: w1/w2 = (x_overall - x2)/(x1 - x_overall)
    
    Args:
        overall_composition: 整体组成（B的质量分数，%）
        phase1_composition: 相1的组成（B的质量分数，%）
        phase2_composition: 相2的组成（B的质量分数，%）
    
    Returns:
        dict: {
            'result': {
                'phase1_fraction': float,  # 相1的质量分数
                'phase2_fraction': float   # 相2的质量分数
            },
            'metadata': {
                'overall_composition': float,
                'phase1_composition': float,
                'phase2_composition': float,
                'calculation_method': str
            }
        }
    """
    # 参数验证
    if not (0 <= overall_composition <= 100):
        raise ValueError(f"Overall composition must be between 0 and 100%, got {overall_composition}%")
    if not (0 <= phase1_composition <= 100):
        raise ValueError(f"Phase1 composition must be between 0 and 100%, got {phase1_composition}%")
    if not (0 <= phase2_composition <= 100):
        raise ValueError(f"Phase2 composition must be between 0 and 100%, got {phase2_composition}%")
    
    # 确保overall_composition在两相之间
    min_comp = min(phase1_composition, phase2_composition)
    max_comp = max(phase1_composition, phase2_composition)
    if not (min_comp <= overall_composition <= max_comp):
        raise ValueError(
            f"Overall composition ({overall_composition}%) must be between "
            f"phase compositions ({min_comp}% and {max_comp}%)"
        )
    
    # 杠杆定律计算
    total_length = abs(phase2_composition - phase1_composition)
    if total_length == 0:
        raise ValueError("Phase compositions cannot be identical")
    
    # 计算各相的质量分数
    phase2_fraction = abs(overall_composition - phase1_composition) / total_length
    phase1_fraction = abs(phase2_composition - overall_composition) / total_length
    
    # 验证质量守恒
    assert abs(phase1_fraction + phase2_fraction - 1.0) < 1e-10, "Mass balance error"
    
    return {
        'result': {
            'phase1_fraction': round(phase1_fraction, 6),
            'phase2_fraction': round(phase2_fraction, 6)
        },
        'metadata': {
            'overall_composition': overall_composition,
            'phase1_composition': phase1_composition,
            'phase2_composition': phase2_composition,
            'calculation_method': 'lever_rule',
            'unit': 'mass_fraction'
        }
    }


def primary_crystal_amount(
    initial_composition: float,
    eutectic_composition: float,
    primary_phase_composition: float,
    total_mass: float = 100.0
) -> Dict:
    """
    计算初晶析出量（在达到共晶温度之前析出的晶体量）
    
    在冷却过程中，当液相组成达到共晶点时，初晶停止析出
    
    Args:
        initial_composition: 初始混合物组成（B的质量分数，%）
        eutectic_composition: 共晶点组成（B的质量分数，%）
        primary_phase_composition: 初晶相的组成（B的质量分数，%）
        total_mass: 总质量（默认100g）
    
    Returns:
        dict: {
            'result': float,  # 初晶析出量（g）
            'metadata': {
                'initial_composition': float,
                'eutectic_composition': float,
                'primary_phase_composition': float,
                'total_mass': float,
                'remaining_liquid_mass': float
            }
        }
    """
    # 参数验证
    if not (0 <= initial_composition <= 100):
        raise ValueError(f"Initial composition must be between 0 and 100%, got {initial_composition}%")
    if not (0 <= eutectic_composition <= 100):
        raise ValueError(f"Eutectic composition must be between 0 and 100%, got {eutectic_composition}%")
    if not (0 <= primary_phase_composition <= 100):
        raise ValueError(f"Primary phase composition must be between 0 and 100%, got {primary_phase_composition}%")
    if total_mass <= 0:
        raise ValueError(f"Total mass must be positive, got {total_mass}")
    
    # 使用杠杆定律计算初晶析出量
    # 当液相达到共晶组成时，系统由初晶和共晶液相组成
    lever_result = lever_rule_calculation(
        overall_composition=initial_composition,
        phase1_composition=primary_phase_composition,
        phase2_composition=eutectic_composition
    )
    
    # 初晶的质量分数
    primary_fraction = lever_result['result']['phase1_fraction']
    
    # 初晶的质量
    primary_mass = primary_fraction * total_mass
    
    # 剩余液相质量
    remaining_liquid = (1 - primary_fraction) * total_mass
    
    return {
        'result': round(primary_mass, 4),
        'metadata': {
            'initial_composition': initial_composition,
            'eutectic_composition': eutectic_composition,
            'primary_phase_composition': primary_phase_composition,
            'total_mass': total_mass,
            'primary_fraction': round(primary_fraction, 6),
            'remaining_liquid_mass': round(remaining_liquid, 4),
            'unit': 'g'
        }
    }


def solve_composition_system(
    eutectic_b: float,
    compound_b: float,
    ratio_c1_to_c2: float,
    phase_a_b: float = 0.0
) -> Dict:
    """
    求解两个混合物组成的方程组
    
    已知条件：
    1. C1的B含量是C2的ratio倍
    2. 两个混合物的初晶析出量相等
    3. C1和C2分别位于共晶点两侧
    
    Args:
        eutectic_b: 共晶点E的B含量（%）
        compound_b: 化合物A_mB_n的B含量（%）
        ratio_c1_to_c2: C1的B含量与C2的B含量之比
        phase_a_b: 纯A相的B含量（默认0%）
    
    Returns:
        dict: {
            'result': {
                'c1_composition': float,  # C1的B含量（%）
                'c2_composition': float   # C2的B含量（%）
            },
            'metadata': {
                'eutectic_b': float,
                'compound_b': float,
                'ratio': float,
                'equations': list,
                'solution_method': str
            }
        }
    """
    # 参数验证
    if not (0 <= eutectic_b <= 100):
        raise ValueError(f"Eutectic B content must be between 0 and 100%, got {eutectic_b}%")
    if not (0 <= compound_b <= 100):
        raise ValueError(f"Compound B content must be between 0 and 100%, got {compound_b}%")
    if ratio_c1_to_c2 <= 0:
        raise ValueError(f"Ratio must be positive, got {ratio_c1_to_c2}")
    if not (0 <= phase_a_b <= 100):
        raise ValueError(f"Phase A B content must be between 0 and 100%, got {phase_a_b}%")
    
    # 设C2的B含量为x，则C1的B含量为ratio*x
    # C2 < E < C1（C2在左侧析出A，C1在右侧析出化合物）
    
    # 对于C2（析出纯A相）：
    # 初晶析出量 = (E - x) / (E - phase_a_b) * 100
    
    # 对于C1（析出化合物相）：
    # 初晶析出量 = (ratio*x - E) / (compound_b - E) * 100
    
    # 两者相等：
    # (E - x) / (E - phase_a_b) = (ratio*x - E) / (compound_b - E)
    
    # 交叉相乘求解：
    # (E - x) * (compound_b - E) = (ratio*x - E) * (E - phase_a_b)
    
    # 展开：
    # E*compound_b - E^2 - x*compound_b + x*E = ratio*x*E - ratio*x*phase_a_b - E^2 + E*phase_a_b
    
    # 整理：
    # -x*compound_b + x*E = ratio*x*E - ratio*x*phase_a_b + E*phase_a_b - E*compound_b
    # x*E - x*compound_b - ratio*x*E + ratio*x*phase_a_b = E*phase_a_b - E*compound_b
    # x*(E - compound_b - ratio*E + ratio*phase_a_b) = E*phase_a_b - E*compound_b
    # x*(E*(1-ratio) - compound_b + ratio*phase_a_b) = E*(phase_a_b - compound_b)
    
    numerator = eutectic_b * (phase_a_b - compound_b)
    denominator = eutectic_b * (1 - ratio_c1_to_c2) - compound_b + ratio_c1_to_c2 * phase_a_b
    
    if abs(denominator) < 1e-10:
        raise ValueError("No unique solution exists for the given parameters")
    
    c2_composition = numerator / denominator
    c1_composition = ratio_c1_to_c2 * c2_composition
    
    # 验证解的合理性
    if c2_composition < 0 or c2_composition > eutectic_b:
        raise ValueError(f"C2 composition ({c2_composition:.2f}%) is outside valid range [0, {eutectic_b}%]")
    if c1_composition < eutectic_b or c1_composition > compound_b:
        raise ValueError(f"C1 composition ({c1_composition:.2f}%) is outside valid range [{eutectic_b}%, {compound_b}%]")
    
    # 验证初晶析出量确实相等
    primary_c2 = primary_crystal_amount(c2_composition, eutectic_b, phase_a_b)
    primary_c1 = primary_crystal_amount(c1_composition, eutectic_b, compound_b)
    
    if abs(primary_c2['result'] - primary_c1['result']) > 0.1:
        raise ValueError(
            f"Primary crystal amounts do not match: "
            f"C2={primary_c2['result']:.2f}g, C1={primary_c1['result']:.2f}g"
        )
    
    return {
        'result': {
            'c1_composition': round(c1_composition, 2),
            'c2_composition': round(c2_composition, 2)
        },
        'metadata': {
            'eutectic_b': eutectic_b,
            'compound_b': compound_b,
            'phase_a_b': phase_a_b,
            'ratio_c1_to_c2': ratio_c1_to_c2,
            'primary_c1': round(primary_c1['result'], 4),
            'primary_c2': round(primary_c2['result'], 4),
            'equations': [
                f"C1 = {ratio_c1_to_c2} * C2",
                f"(E - C2)/(E - A) = (C1 - E)/(Compound - E)"
            ],
            'solution_method': 'algebraic_substitution'
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_binary_eutectic_system(
    eutectic_b: float,
    compound_b: float,
    ratio_c1_to_c2: float,
    phase_a_b: float = 0.0,
    total_mass: float = 100.0
) -> Dict:
    """
    完整分析二元共晶系统，包括组成计算和初晶析出分析
    
    Args:
        eutectic_b: 共晶点E的B含量（%）
        compound_b: 化合物A_mB_n的B含量（%）
        ratio_c1_to_c2: C1的B含量与C2的B含量之比
        phase_a_b: 纯A相的B含量（默认0%）
        total_mass: 总质量（默认100g）
    
    Returns:
        dict: {
            'result': {
                'compositions': dict,
                'primary_crystals': dict,
                'phase_analysis': dict
            },
            'metadata': {...}
        }
    """
    # 步骤1：求解组成
    composition_result = solve_composition_system(
        eutectic_b=eutectic_b,
        compound_b=compound_b,
        ratio_c1_to_c2=ratio_c1_to_c2,
        phase_a_b=phase_a_b
    )
    
    c1_comp = composition_result['result']['c1_composition']
    c2_comp = composition_result['result']['c2_composition']
    
    # 步骤2：计算C2的初晶析出（纯A相）
    primary_c2 = primary_crystal_amount(
        initial_composition=c2_comp,
        eutectic_composition=eutectic_b,
        primary_phase_composition=phase_a_b,
        total_mass=total_mass
    )
    
    # 步骤3：计算C1的初晶析出（化合物相）
    primary_c1 = primary_crystal_amount(
        initial_composition=c1_comp,
        eutectic_composition=eutectic_b,
        primary_phase_composition=compound_b,
        total_mass=total_mass
    )
    
    # 步骤4：分析共晶反应后的相组成
    # C2最终组成：A + (A + A_mB_n)共晶混合物
    eutectic_c2 = lever_rule_calculation(
        overall_composition=eutectic_b,
        phase1_composition=phase_a_b,
        phase2_composition=compound_b
    )
    
    # C1最终组成：A_mB_n + (A + A_mB_n)共晶混合物
    eutectic_c1 = lever_rule_calculation(
        overall_composition=eutectic_b,
        phase1_composition=phase_a_b,
        phase2_composition=compound_b
    )
    
    return {
        'result': {
            'compositions': {
                'c1_b_content': c1_comp,
                'c2_b_content': c2_comp,
                'ratio': ratio_c1_to_c2
            },
            'primary_crystals': {
                'c1_primary_mass': primary_c1['result'],
                'c1_primary_phase': 'A_mB_n (compound)',
                'c2_primary_mass': primary_c2['result'],
                'c2_primary_phase': 'A (pure)',
                'amounts_equal': abs(primary_c1['result'] - primary_c2['result']) < 0.1
            },
            'phase_analysis': {
                'c1_final_phases': ['A_mB_n', 'A+A_mB_n eutectic'],
                'c2_final_phases': ['A', 'A+A_mB_n eutectic'],
                'eutectic_composition_in_A': eutectic_c1['result']['phase1_fraction'],
                'eutectic_composition_in_compound': eutectic_c1['result']['phase2_fraction']
            }
        },
        'metadata': {
            'system_type': 'binary_eutectic_with_incongruent_compound',
            'eutectic_point': eutectic_b,
            'compound_composition': compound_b,
            'total_mass': total_mass,
            'calculation_steps': [
                'solve_composition_system',
                'primary_crystal_amount (C2)',
                'primary_crystal_amount (C1)',
                'lever_rule_calculation (eutectic)'
            ]
        }
    }


def validate_phase_diagram_constraints(
    eutectic_b: float,
    compound_b: float,
    c1_b: float,
    c2_b: float
) -> Dict:
    """
    验证相图约束条件是否满足
    
    Args:
        eutectic_b: 共晶点B含量（%）
        compound_b: 化合物B含量（%）
        c1_b: C1的B含量（%）
        c2_b: C2的B含量（%）
    
    Returns:
        dict: {
            'result': bool,  # 是否满足所有约束
            'metadata': {
                'constraints': list,
                'violations': list
            }
        }
    """
    constraints = []
    violations = []
    
    # 约束1：C2 < E < C1
    constraint1 = "C2 < Eutectic < C1"
    if c2_b < eutectic_b < c1_b:
        constraints.append({'constraint': constraint1, 'satisfied': True})
    else:
        constraints.append({'constraint': constraint1, 'satisfied': False})
        violations.append(f"{constraint1} violated: C2={c2_b}%, E={eutectic_b}%, C1={c1_b}%")
    
    # 约束2：E < Compound
    constraint2 = "Eutectic < Compound"
    if eutectic_b < compound_b:
        constraints.append({'constraint': constraint2, 'satisfied': True})
    else:
        constraints.append({'constraint': constraint2, 'satisfied': False})
        violations.append(f"{constraint2} violated: E={eutectic_b}%, Compound={compound_b}%")
    
    # 约束3：C1 < Compound
    constraint3 = "C1 < Compound"
    if c1_b < compound_b:
        constraints.append({'constraint': constraint3, 'satisfied': True})
    else:
        constraints.append({'constraint': constraint3, 'satisfied': False})
        violations.append(f"{constraint3} violated: C1={c1_b}%, Compound={compound_b}%")
    
    # 约束4：所有组成在0-100%之间
    constraint4 = "All compositions in [0, 100]%"
    if all(0 <= x <= 100 for x in [eutectic_b, compound_b, c1_b, c2_b]):
        constraints.append({'constraint': constraint4, 'satisfied': True})
    else:
        constraints.append({'constraint': constraint4, 'satisfied': False})
        violations.append(f"{constraint4} violated")
    
    all_satisfied = len(violations) == 0
    
    return {
        'result': all_satisfied,
        'metadata': {
            'constraints': constraints,
            'violations': violations,
            'total_constraints': len(constraints),
            'satisfied_count': sum(1 for c in constraints if c['satisfied'])
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_binary_phase_diagram(
    eutectic_b: float,
    compound_b: float,
    c1_b: float,
    c2_b: float,
    save_path: Optional[str] = None
) -> Dict:
    """
    绘制二元相图并标注关键点
    
    Args:
        eutectic_b: 共晶点B含量（%）
        compound_b: 化合物B含量（%）
        c1_b: C1的B含量（%）
        c2_b: C2的B含量（%）
        save_path: 图像保存路径（可选）
    
    Returns:
        dict: {
            'result': str,  # 图像文件路径
            'metadata': {...}
        }
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 简化的相图绘制（示意图）
    # 液相线
    liquidus_x = [0, eutectic_b, compound_b, 100]
    liquidus_y = [1200, 800, 1000, 1400]  # 示意温度
    ax.plot(liquidus_x, liquidus_y, 'b-', linewidth=2, label='Liquidus')
    
    # 共晶线
    ax.axhline(y=800, xmin=0, xmax=1, color='r', linestyle='--', 
               linewidth=1.5, label='Eutectic Temperature')
    
    # 标注关键点
    ax.plot(eutectic_b, 800, 'ro', markersize=10, label=f'Eutectic E ({eutectic_b}% B)')
    ax.plot(compound_b, 1000, 'gs', markersize=10, label=f'Compound ({compound_b}% B)')
    ax.plot(c1_b, 900, 'b^', markersize=10, label=f'C1 ({c1_b}% B)')
    ax.plot(c2_b, 900, 'm^', markersize=10, label=f'C2 ({c2_b}% B)')
    
    # 标注相区
    ax.text(eutectic_b/2, 1000, 'L + A', fontsize=12, ha='center')
    ax.text((eutectic_b + compound_b)/2, 900, 'L + A_mB_n', fontsize=12, ha='center')
    ax.text(eutectic_b/2, 600, 'A + A_mB_n', fontsize=12, ha='center')
    ax.text((compound_b + 100)/2, 1100, 'L + B', fontsize=12, ha='center')
    
    # 设置坐标轴
    ax.set_xlabel('B Content (%)', fontsize=14)
    ax.set_ylabel('Temperature (arbitrary units)', fontsize=14)
    ax.set_title('Binary Phase Diagram with Incongruent Melting Compound', fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(500, 1500)
    
    # 保存图像
    if save_path is None:
        save_path = IMAGE_DIR / "binary_phase_diagram.png"
    else:
        save_path = Path(save_path)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': str(save_path),
        'metadata': {
            'file_type': 'png',
            'dpi': 300,
            'eutectic_b': eutectic_b,
            'compound_b': compound_b,
            'c1_b': c1_b,
            'c2_b': c2_b,
            'figure_size': '10x8 inches'
        }
    }


def plot_primary_crystal_comparison(
    c1_b: float,
    c2_b: float,
    eutectic_b: float,
    compound_b: float,
    phase_a_b: float = 0.0,
    save_path: Optional[str] = None
) -> Dict:
    """
    绘制初晶析出量对比图
    
    Args:
        c1_b: C1的B含量（%）
        c2_b: C2的B含量（%）
        eutectic_b: 共晶点B含量（%）
        compound_b: 化合物B含量（%）
        phase_a_b: 纯A相B含量（%）
        save_path: 图像保存路径（可选）
    
    Returns:
        dict: {
            'result': str,  # 图像文件路径
            'metadata': {...}
        }
    """
    # 计算初晶析出量
    primary_c1 = primary_crystal_amount(c1_b, eutectic_b, compound_b)
    primary_c2 = primary_crystal_amount(c2_b, eutectic_b, phase_a_b)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：柱状图对比
    mixtures = ['C1', 'C2']
    amounts = [primary_c1['result'], primary_c2['result']]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax1.bar(mixtures, amounts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Primary Crystal Amount (g)', fontsize=12)
    ax1.set_title('Primary Crystal Precipitation Comparison', fontsize=14)
    ax1.set_ylim(0, max(amounts) * 1.2)
    
    # 在柱子上标注数值
    for bar, amount in zip(bars, amounts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{amount:.2f}g',
                ha='center', va='bottom', fontsize=11)
    
    # 右图：组成对比
    compositions = [c1_b, c2_b]
    bars2 = ax2.bar(mixtures, compositions, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=eutectic_b, color='red', linestyle='--', linewidth=2, 
                label=f'Eutectic ({eutectic_b}% B)')
    ax2.set_ylabel('B Content (%)', fontsize=12)
    ax2.set_title('Composition Comparison', fontsize=14)
    ax2.legend()
    
    # 在柱子上标注数值
    for bar, comp in zip(bars2, compositions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{comp:.2f}%',
                ha='center', va='bottom', fontsize=11)
    
    # 保存图像
    if save_path is None:
        save_path = IMAGE_DIR / "primary_crystal_comparison.png"
    else:
        save_path = Path(save_path)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': str(save_path),
        'metadata': {
            'file_type': 'png',
            'dpi': 300,
            'c1_primary': primary_c1['result'],
            'c2_primary': primary_c2['result'],
            'c1_composition': c1_b,
            'c2_composition': c2_b,
            'difference': abs(primary_c1['result'] - primary_c2['result'])
        }
    }


def save_calculation_report(
    analysis_result: Dict,
    filepath: Optional[str] = None
) -> Dict:
    """
    保存完整的计算报告为JSON文件
    
    Args:
        analysis_result: analyze_binary_eutectic_system的返回结果
        filepath: 保存路径（可选）
    
    Returns:
        dict: {
            'result': str,  # 文件路径
            'metadata': {...}
        }
    """
    if filepath is None:
        filepath = MID_RESULT_DIR / "calculation_report.json"
    else:
        filepath = Path(filepath)
    
    # 确保目录存在
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"FILE_GENERATED: json | PATH: {filepath}")
    
    return {
        'result': str(filepath),
        'metadata': {
            'file_type': 'json',
            'encoding': 'utf-8',
            'size_bytes': filepath.stat().st_size
        }
    }


# ==================== 主函数：演示场景 ====================

def main():
    """
    演示三个场景的计算
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 计算C1和C2的组成")
    print("=" * 80)
    print("问题描述：")
    print("- 共晶点E的B含量：20%")
    print("- 化合物A_mB_n的B含量：64%")
    print("- C1的B含量是C2的1.5倍")
    print("- 两个混合物的初晶析出量相等")
    print("- 求C1和C2的组成")
    print("-" * 80)
    
    # 已知参数
    eutectic_b = 20.0  # 共晶点B含量
    compound_b = 64.0  # 化合物B含量
    ratio = 1.5        # C1/C2的比值
    phase_a_b = 0.0    # 纯A相的B含量
    
    # 步骤1：求解组成
    print("\n步骤1：求解C1和C2的组成")
    print("调用函数：solve_composition_system()")
    composition_result = solve_composition_system(
        eutectic_b=eutectic_b,
        compound_b=compound_b,
        ratio_c1_to_c2=ratio,
        phase_a_b=phase_a_b
    )
    print(f"FUNCTION_CALL: solve_composition_system | PARAMS: {{eutectic_b: {eutectic_b}, compound_b: {compound_b}, ratio: {ratio}}} | RESULT: {composition_result['result']}")
    
    c1_composition = composition_result['result']['c1_composition']
    c2_composition = composition_result['result']['c2_composition']
    
    # 步骤2：验证初晶析出量
    print("\n步骤2：验证C2的初晶析出量")
    print("调用函数：primary_crystal_amount() for C2")
    primary_c2 = primary_crystal_amount(
        initial_composition=c2_composition,
        eutectic_composition=eutectic_b,
        primary_phase_composition=phase_a_b
    )
    print(f"FUNCTION_CALL: primary_crystal_amount | PARAMS: {{initial: {c2_composition}, eutectic: {eutectic_b}, primary_phase: {phase_a_b}}} | RESULT: {primary_c2['result']}g")
    
    print("\n步骤3：验证C1的初晶析出量")
    print("调用函数：primary_crystal_amount() for C1")
    primary_c1 = primary_crystal_amount(
        initial_composition=c1_composition,
        eutectic_composition=eutectic_b,
        primary_phase_composition=compound_b
    )
    print(f"FUNCTION_CALL: primary_crystal_amount | PARAMS: {{initial: {c1_composition}, eutectic: {eutectic_b}, primary_phase: {compound_b}}} | RESULT: {primary_c1['result']}g")
    
    # 步骤4：完整系统分析
    print("\n步骤4：完整系统分析")
    print("调用函数：analyze_binary_eutectic_system()")
    analysis = analyze_binary_eutectic_system(
        eutectic_b=eutectic_b,
        compound_b=compound_b,
        ratio_c1_to_c2=ratio,
        phase_a_b=phase_a_b
    )
    print(f"FUNCTION_CALL: analyze_binary_eutectic_system | PARAMS: {{eutectic_b: {eutectic_b}, compound_b: {compound_b}, ratio: {ratio}}} | RESULT: {analysis['result']['compositions']}")
    
    # 步骤5：验证相图约束
    print("\n步骤5：验证相图约束条件")
    print("调用函数：validate_phase_diagram_constraints()")
    validation = validate_phase_diagram_constraints(
        eutectic_b=eutectic_b,
        compound_b=compound_b,
        c1_b=c1_composition,
        c2_b=c2_composition
    )
    print(f"FUNCTION_CALL: validate_phase_diagram_constraints | PARAMS: {{eutectic: {eutectic_b}, compound: {compound_b}, c1: {c1_composition}, c2: {c2_composition}}} | RESULT: {validation['result']}")
    
    # 步骤6：生成可视化
    print("\n步骤6：生成相图可视化")
    print("调用函数：plot_binary_phase_diagram()")
    phase_diagram = plot_binary_phase_diagram(
        eutectic_b=eutectic_b,
        compound_b=compound_b,
        c1_b=c1_composition,
        c2_b=c2_composition
    )
    print(f"FUNCTION_CALL: plot_binary_phase_diagram | PARAMS: {{eutectic: {eutectic_b}, compound: {compound_b}, c1: {c1_composition}, c2: {c2_composition}}} | RESULT: {phase_diagram['result']}")
    
    print("\n步骤7：生成初晶析出对比图")
    print("调用函数：plot_primary_crystal_comparison()")
    comparison_plot = plot_primary_crystal_comparison(
        c1_b=c1_composition,
        c2_b=c2_composition,
        eutectic_b=eutectic_b,
        compound_b=compound_b,
        phase_a_b=phase_a_b
    )
    print(f"FUNCTION_CALL: plot_primary_crystal_comparison | PARAMS: {{c1: {c1_composition}, c2: {c2_composition}}} | RESULT: {comparison_plot['result']}")
    
    # 步骤8：保存计算报告
    print("\n步骤8：保存完整计算报告")
    print("调用函数：save_calculation_report()")
    report = save_calculation_report(analysis)
    print(f"FUNCTION_CALL: save_calculation_report | PARAMS: {{analysis_result}} | RESULT: {report['result']}")
    
    print("\n" + "=" * 80)
    print(f"FINAL_ANSWER: C1组成 = {c1_composition}% B, C2组成 = {c2_composition}% B")
    print("=" * 80)
    
    
    print("\n\n" + "=" * 80)
    print("场景2：杠杆定律应用 - 计算不同组成下的相比例")
    print("=" * 80)
    print("问题描述：对于组成为30% B的合金，在共晶温度时计算液相和固相的比例")
    print("-" * 80)
    
    # 步骤1：计算液相和化合物相的比例
    print("\n步骤1：应用杠杆定律计算相比例")
    print("调用函数：lever_rule_calculation()")
    alloy_composition = 30.0
    lever_result = lever_rule_calculation(
        overall_composition=alloy_composition,
        phase1_composition=eutectic_b,
        phase2_composition=compound_b
    )
    print(f"FUNCTION_CALL: lever_rule_calculation | PARAMS: {{overall: {alloy_composition}, phase1: {eutectic_b}, phase2: {compound_b}}} | RESULT: {lever_result['result']}")
    
    # 步骤2：计算初晶析出量
    print("\n步骤2：计算该合金的初晶析出量")
    print("调用函数：primary_crystal_amount()")
    primary_30 = primary_crystal_amount(
        initial_composition=alloy_composition,
        eutectic_composition=eutectic_b,
        primary_phase_composition=compound_b
    )
    print(f"FUNCTION_CALL: primary_crystal_amount | PARAMS: {{initial: {alloy_composition}, eutectic: {eutectic_b}, primary_phase: {compound_b}}} | RESULT: {primary_30['result']}g")
    
    liquid_fraction = lever_result['result']['phase1_fraction']
    compound_fraction = lever_result['result']['phase2_fraction']
    
    print(f"\nFINAL_ANSWER: 在共晶温度时，液相占{liquid_fraction*100:.2f}%，化合物相占{compound_fraction*100:.2f}%；初晶析出量为{primary_30['result']:.2f}g")
    
    
    print("\n\n" + "=" * 80)
    print("场景3：参数敏感性分析 - 不同比值下的组成变化")
    print("=" * 80)
    print("问题描述：分析当C1/C2的比值从1.2变化到2.0时，组成如何变化")
    print("-" * 80)
    
    ratios = [1.2, 1.5, 1.8, 2.0]
    results = []
    
    print("\n步骤1：计算不同比值下的组成")
    for r in ratios:
        print(f"\n  比值 = {r}")
        print(f"  调用函数：solve_composition_system()")
        comp_result = solve_composition_system(
            eutectic_b=eutectic_b,
            compound_b=compound_b,
            ratio_c1_to_c2=r,
            phase_a_b=phase_a_b
        )
        print(f"  FUNCTION_CALL: solve_composition_system | PARAMS: {{ratio: {r}}} | RESULT: {comp_result['result']}")
        results.append({
            'ratio': r,
            'c1': comp_result['result']['c1_composition'],
            'c2': comp_result['result']['c2_composition']
        })
    
    # 步骤2：可视化参数敏感性
    print("\n步骤2：绘制参数敏感性曲线")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ratios_plot = [r['ratio'] for r in results]
    c1_values = [r['c1'] for r in results]
    c2_values = [r['c2'] for r in results]
    
    ax.plot(ratios_plot, c1_values, 'o-', linewidth=2, markersize=8, label='C1 composition')
    ax.plot(ratios_plot, c2_values, 's-', linewidth=2, markersize=8, label='C2 composition')
    ax.axhline(y=eutectic_b, color='r', linestyle='--', linewidth=1.5, label=f'Eutectic ({eutectic_b}% B)')
    
    ax.set_xlabel('Ratio (C1/C2)', fontsize=12)
    ax.set_ylabel('B Content (%)', fontsize=12)
    ax.set_title('Sensitivity Analysis: Composition vs. Ratio', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    sensitivity_path = IMAGE_DIR / "sensitivity_analysis.png"
    plt.tight_layout()
    plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {sensitivity_path}")
    print(f"FUNCTION_CALL: plot_sensitivity_analysis | PARAMS: {{ratios: {ratios}}} | RESULT: {sensitivity_path}")
    
    print(f"\nFINAL_ANSWER: 当比值从1.2增加到2.0时，C1从{results[0]['c1']:.2f}%增加到{results[-1]['c1']:.2f}%，C2从{results[0]['c2']:.2f}%减少到{results[-1]['c2']:.2f}%")


if __name__ == "__main__":
    main()