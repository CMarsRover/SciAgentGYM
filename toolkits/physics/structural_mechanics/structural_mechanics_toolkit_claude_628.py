# Filename: structural_mechanics_toolkit.py

"""
结构力学计算工具包 - 简支梁弯矩分析
Structural Mechanics Toolkit - Simply Supported Beam Bending Moment Analysis

集成库：
- scipy: 数值计算和优化
- sympy: 符号计算和解析解
- matplotlib: 工程图表绘制
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, integrate, lambdify, Piecewise

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/structural_mechanics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)


# ==================== 第一层：原子函数 ====================

def calculate_support_reactions(load: float, load_position: float, span: float) -> Dict:
    """
    计算简支梁支座反力
    
    Args:
        load: 集中荷载 (kN)
        load_position: 荷载作用位置距左端距离 (m)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': {'R_A': float, 'R_B': float}, 'metadata': {...}}
    """
    if load <= 0:
        raise ValueError(f"荷载必须为正值，当前值: {load}")
    if not (0 <= load_position <= span):
        raise ValueError(f"荷载位置必须在 [0, {span}] 范围内，当前值: {load_position}")
    if span <= 0:
        raise ValueError(f"跨度必须为正值，当前值: {span}")
    
    # 力矩平衡方程：ΣM_A = 0
    R_B = load * load_position / span
    # 力平衡方程：ΣF_y = 0
    R_A = load - R_B
    
    return {
        'result': {
            'R_A': round(R_A, 4),
            'R_B': round(R_B, 4)
        },
        'metadata': {
            'method': 'equilibrium_equations',
            'load': load,
            'load_position': load_position,
            'span': span
        }
    }


def calculate_shear_force(x: float, load: float, load_position: float, 
                         R_A: float, span: float) -> Dict:
    """
    计算指定位置的剪力
    
    Args:
        x: 计算位置距左端距离 (m)
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        R_A: 左支座反力 (kN)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if not (0 <= x <= span):
        raise ValueError(f"计算位置必须在 [0, {span}] 范围内，当前值: {x}")
    
    if x < load_position:
        V = R_A
    else:
        V = R_A - load
    
    return {
        'result': round(V, 4),
        'metadata': {
            'position': x,
            'section': 'left' if x < load_position else 'right'
        }
    }


def calculate_bending_moment(x: float, load: float, load_position: float,
                            R_A: float, span: float) -> Dict:
    """
    计算指定位置的弯矩
    
    Args:
        x: 计算位置距左端距离 (m)
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        R_A: 左支座反力 (kN)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if not (0 <= x <= span):
        raise ValueError(f"计算位置必须在 [0, {span}] 范围内，当前值: {x}")
    
    if x < load_position:
        M = R_A * x
    else:
        M = R_A * x - load * (x - load_position)
    
    return {
        'result': round(M, 4),
        'metadata': {
            'position': x,
            'formula': 'M = R_A * x' if x < load_position else 'M = R_A * x - P * (x - a)'
        }
    }


def find_maximum_bending_moment(load: float, load_position: float,
                               R_A: float, span: float) -> Dict:
    """
    求解最大弯矩及其位置
    
    Args:
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        R_A: 左支座反力 (kN)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': {'M_max': float, 'x_max': float}, 'metadata': {...}}
    """
    # 对于集中荷载，最大弯矩发生在荷载作用点
    M_max_result = calculate_bending_moment(load_position, load, load_position, R_A, span)
    M_max = M_max_result['result']
    
    return {
        'result': {
            'M_max': round(M_max, 4),
            'x_max': load_position
        },
        'metadata': {
            'criterion': 'concentrated_load_maximum_at_load_point',
            'verification': 'shear_force_changes_sign'
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_beam_complete(load: float, load_position: float, span: float) -> Dict:
    """
    完整分析简支梁（支座反力 + 最大弯矩）
    
    Args:
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': {...}, 'metadata': {...}}
    """
    # 步骤1：计算支座反力
    reactions = calculate_support_reactions(load, load_position, span)
    R_A = reactions['result']['R_A']
    R_B = reactions['result']['R_B']
    
    # 步骤2：求最大弯矩
    max_moment = find_maximum_bending_moment(load, load_position, R_A, span)
    
    return {
        'result': {
            'support_reactions': reactions['result'],
            'max_bending_moment': max_moment['result']
        },
        'metadata': {
            'load': load,
            'load_position': load_position,
            'span': span,
            'analysis_type': 'complete_beam_analysis'
        }
    }


def generate_beam_diagrams(load: float, load_position: float, span: float,
                          num_points: int = 100) -> Dict:
    """
    生成剪力图和弯矩图数据
    
    Args:
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        span: 梁跨度 (m)
        num_points: 采样点数
    
    Returns:
        dict: {'result': 'filepath', 'metadata': {...}}
    """
    # 计算支座反力
    reactions = calculate_support_reactions(load, load_position, span)
    R_A = reactions['result']['R_A']
    
    # 生成位置数组
    x_left = np.linspace(0, load_position, num_points // 2)
    x_right = np.linspace(load_position, span, num_points // 2)
    x_all = np.concatenate([x_left, x_right])
    
    # 计算剪力和弯矩
    shear_forces = []
    bending_moments = []
    
    for x in x_all:
        V = calculate_shear_force(x, load, load_position, R_A, span)['result']
        M = calculate_bending_moment(x, load, load_position, R_A, span)['result']
        shear_forces.append(V)
        bending_moments.append(M)
    
    # 保存数据
    data = {
        'x': x_all.tolist(),
        'shear_force': shear_forces,
        'bending_moment': bending_moments
    }
    
    filepath = './mid_result/structural_mechanics/beam_diagrams_data.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'json',
            'num_points': len(x_all),
            'load': load,
            'span': span
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_beam_diagrams(load: float, load_position: float, span: float) -> Dict:
    """
    绘制简支梁的剪力图和弯矩图
    
    Args:
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': 'filepath', 'metadata': {...}}
    """
    # 生成数据
    data_result = generate_beam_diagrams(load, load_position, span)
    
    with open(data_result['result'], 'r') as f:
        data = json.load(f)
    
    x = np.array(data['x'])
    V = np.array(data['shear_force'])
    M = np.array(data['bending_moment'])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 剪力图
    ax1.plot(x, V, 'b-', linewidth=2, label='Shear Force')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax1.axvline(x=load_position, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Load Position')
    ax1.set_xlabel('Position x (m)', fontsize=11)
    ax1.set_ylabel('Shear Force V (kN)', fontsize=11)
    ax1.set_title('Shear Force Diagram', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 弯矩图
    ax2.plot(x, M, 'r-', linewidth=2, label='Bending Moment')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.axvline(x=load_position, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Load Position')
    M_max_idx = np.argmax(M)
    ax2.plot(x[M_max_idx], M[M_max_idx], 'go', markersize=10, label=f'M_max = {M[M_max_idx]:.2f} kN·m')
    ax2.set_xlabel('Position x (m)', fontsize=11)
    ax2.set_ylabel('Bending Moment M (kN·m)', fontsize=11)
    ax2.set_title('Bending Moment Diagram', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    filepath = './tool_images/beam_diagrams.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'M_max': float(M[M_max_idx]),
            'M_max_position': float(x[M_max_idx])
        }
    }


def plot_beam_schematic(load: float, load_position: float, span: float) -> Dict:
    """
    绘制简支梁示意图
    
    Args:
        load: 集中荷载 (kN)
        load_position: 荷载作用位置 (m)
        span: 梁跨度 (m)
    
    Returns:
        dict: {'result': 'filepath', 'metadata': {...}}
    """
    reactions = calculate_support_reactions(load, load_position, span)
    R_A = reactions['result']['R_A']
    R_B = reactions['result']['R_B']
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 绘制梁
    ax.plot([0, span], [0, 0], 'k-', linewidth=3)
    
    # 绘制支座
    ax.plot(0, 0, 'b^', markersize=15, label=f'Support A: R_A = {R_A:.2f} kN')
    ax.plot(span, 0, 'b^', markersize=15, label=f'Support B: R_B = {R_B:.2f} kN')
    
    # 绘制荷载
    ax.arrow(load_position, 0.5, 0, -0.4, head_width=0.1, head_length=0.08, 
             fc='red', ec='red', linewidth=2)
    ax.text(load_position, 0.6, f'P = {load} kN', ha='center', fontsize=11, fontweight='bold')
    
    # 标注尺寸
    ax.annotate('', xy=(load_position, -0.3), xytext=(0, -0.3),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(load_position/2, -0.45, f'{load_position} m', ha='center', fontsize=10, color='green')
    
    ax.annotate('', xy=(span, -0.3), xytext=(load_position, -0.3),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text((span + load_position)/2, -0.45, f'{span - load_position} m', ha='center', fontsize=10, color='green')
    
    ax.set_xlim(-0.5, span + 0.5)
    ax.set_ylim(-0.7, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Simply Supported Beam Schematic', fontsize=13, fontweight='bold', pad=20)
    
    filepath = './tool_images/beam_schematic.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'R_A': R_A,
            'R_B': R_B
        }
    }


# ==================== 文件解析工具 ====================

def load_file(filepath: str) -> Dict:
    """
    加载JSON格式的计算结果文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        dict: {'result': data, 'metadata': {...}}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'result': data,
        'metadata': {
            'filepath': filepath,
            'file_size': os.path.getsize(filepath)
        }
    }


# ==================== 主函数：3个场景演示 ====================

def main():
    """
    演示3个场景的完整计算流程
    """
    
    print("=" * 60)
    print("场景1：原始问题 - 简支梁中点集中荷载最大弯矩计算")
    print("=" * 60)
    print("问题描述：梁AB长4.0m，中点C处施加P=6kN向下集中力，求最大弯矩M_max")
    print("-" * 60)
    
    # 步骤1：计算支座反力
    # 调用函数：calculate_support_reactions()
    load_1 = 6.0  # kN
    load_pos_1 = 2.0  # m (中点)
    span_1 = 4.0  # m
    reactions_1 = calculate_support_reactions(load_1, load_pos_1, span_1)
    print(f"FUNCTION_CALL: calculate_support_reactions | PARAMS: {{load: {load_1}, load_position: {load_pos_1}, span: {span_1}}} | RESULT: {reactions_1}")
    
    # 步骤2：求最大弯矩
    # 调用函数：find_maximum_bending_moment()
    R_A_1 = reactions_1['result']['R_A']
    max_moment_1 = find_maximum_bending_moment(load_1, load_pos_1, R_A_1, span_1)
    print(f"FUNCTION_CALL: find_maximum_bending_moment | PARAMS: {{load: {load_1}, load_position: {load_pos_1}, R_A: {R_A_1}, span: {span_1}}} | RESULT: {max_moment_1}")
    
    # 步骤3：绘制示意图
    # 调用函数：plot_beam_schematic()
    schematic_1 = plot_beam_schematic(load_1, load_pos_1, span_1)
    print(f"FUNCTION_CALL: plot_beam_schematic | PARAMS: {{load: {load_1}, load_position: {load_pos_1}, span: {span_1}}} | RESULT: {schematic_1}")
    
    # 步骤4：绘制剪力弯矩图
    # 调用函数：plot_beam_diagrams()
    diagrams_1 = plot_beam_diagrams(load_1, load_pos_1, span_1)
    print(f"FUNCTION_CALL: plot_beam_diagrams | PARAMS: {{load: {load_1}, load_position: {load_pos_1}, span: {span_1}}} | RESULT: {diagrams_1}")
    
    answer_1 = f"M_max = {max_moment_1['result']['M_max']} kN·m"
    print(f"FINAL_ANSWER: {answer_1}")
    
    
    print("\n" + "=" * 60)
    print("场景2：非对称荷载 - 1/4跨度位置集中荷载")
    print("=" * 60)
    print("问题描述：梁长6m，距左端1.5m处施加P=12kN荷载，求最大弯矩及位置")
    print("-" * 60)
    
    # 步骤1：完整分析（组合函数）
    # 调用函数：analyze_beam_complete()
    load_2 = 12.0  # kN
    load_pos_2 = 1.5  # m
    span_2 = 6.0  # m
    analysis_2 = analyze_beam_complete(load_2, load_pos_2, span_2)
    print(f"FUNCTION_CALL: analyze_beam_complete | PARAMS: {{load: {load_2}, load_position: {load_pos_2}, span: {span_2}}} | RESULT: {analysis_2}")
    
    # 步骤2：验证特定位置弯矩
    # 调用函数：calculate_bending_moment()
    x_check = 3.0  # m
    R_A_2 = analysis_2['result']['support_reactions']['R_A']
    moment_at_3m = calculate_bending_moment(x_check, load_2, load_pos_2, R_A_2, span_2)
    print(f"FUNCTION_CALL: calculate_bending_moment | PARAMS: {{x: {x_check}, load: {load_2}, load_position: {load_pos_2}, R_A: {R_A_2}, span: {span_2}}} | RESULT: {moment_at_3m}")
    
    # 步骤3：生成图表数据
    # 调用函数：generate_beam_diagrams()
    data_file_2 = generate_beam_diagrams(load_2, load_pos_2, span_2)
    print(f"FUNCTION_CALL: generate_beam_diagrams | PARAMS: {{load: {load_2}, load_position: {load_pos_2}, span: {span_2}}} | RESULT: {data_file_2}")
    
    answer_2 = f"M_max = {analysis_2['result']['max_bending_moment']['M_max']} kN·m at x = {analysis_2['result']['max_bending_moment']['x_max']} m"
    print(f"FINAL_ANSWER: {answer_2}")
    
    
    print("\n" + "=" * 60)
    print("场景3：多位置剪力分析 - 剪力突变验证")
    print("=" * 60)
    print("问题描述：梁长5m，中点施加P=10kN，分析荷载前后剪力变化")
    print("-" * 60)
    
    # 步骤1：计算支座反力
    # 调用函数：calculate_support_reactions()
    load_3 = 10.0  # kN
    load_pos_3 = 2.5  # m
    span_3 = 5.0  # m
    reactions_3 = calculate_support_reactions(load_3, load_pos_3, span_3)
    print(f"FUNCTION_CALL: calculate_support_reactions | PARAMS: {{load: {load_3}, load_position: {load_pos_3}, span: {span_3}}} | RESULT: {reactions_3}")
    
    # 步骤2：计算荷载点左侧剪力
    # 调用函数：calculate_shear_force()
    x_left = 2.4  # m
    R_A_3 = reactions_3['result']['R_A']
    shear_left = calculate_shear_force(x_left, load_3, load_pos_3, R_A_3, span_3)
    print(f"FUNCTION_CALL: calculate_shear_force | PARAMS: {{x: {x_left}, load: {load_3}, load_position: {load_pos_3}, R_A: {R_A_3}, span: {span_3}}} | RESULT: {shear_left}")
    
    # 步骤3：计算荷载点右侧剪力
    # 调用函数：calculate_shear_force()
    x_right = 2.6  # m
    shear_right = calculate_shear_force(x_right, load_3, load_pos_3, R_A_3, span_3)
    print(f"FUNCTION_CALL: calculate_shear_force | PARAMS: {{x: {x_right}, load: {load_3}, load_position: {load_pos_3}, R_A: {R_A_3}, span: {span_3}}} | RESULT: {shear_right}")
    
    # 步骤4：计算剪力突变量
    shear_jump = shear_left['result'] - shear_right['result']
    print(f"剪力突变量: {shear_jump} kN (应等于荷载 {load_3} kN)")
    
    # 步骤5：求最大弯矩
    # 调用函数：find_maximum_bending_moment()
    max_moment_3 = find_maximum_bending_moment(load_3, load_pos_3, R_A_3, span_3)
    print(f"FUNCTION_CALL: find_maximum_bending_moment | PARAMS: {{load: {load_3}, load_position: {load_pos_3}, R_A: {R_A_3}, span: {span_3}}} | RESULT: {max_moment_3}")
    
    answer_3 = f"Shear jump = {shear_jump} kN, M_max = {max_moment_3['result']['M_max']} kN·m"
    print(f"FINAL_ANSWER: {answer_3}")


if __name__ == "__main__":
    main()