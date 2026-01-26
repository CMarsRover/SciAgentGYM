# Filename: structural_mechanics_toolkit.py

"""
结构力学计算工具包 - 刚架弯矩分析
Structural Mechanics Toolkit - Rigid Frame Bending Moment Analysis

支持功能：
1. 刚架静力分析（支座反力计算）
2. 截面弯矩计算
3. 弯矩图绘制
4. 剪力图绘制
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
Path("./mid_result/structural_mechanics").mkdir(parents=True, exist_ok=True)
Path("./tool_images").mkdir(parents=True, exist_ok=True)


# ==================== 第一层：原子函数 ====================

def calculate_support_reactions(
    frame_height: float,
    frame_width: float,
    distributed_load: float,
    load_position: str = "right_vertical"
) -> Dict:
    """
    计算刚架支座反力
    
    Parameters:
    -----------
    frame_height : float
        刚架高度 h (m)
    frame_width : float
        刚架宽度 L (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    load_position : str
        荷载位置 ("right_vertical" 表示右侧竖向构件)
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        包含支座A和B的反力分量
    """
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("刚架尺寸必须为正值")
    if distributed_load < 0:
        raise ValueError("荷载强度不能为负值")
    
    # 右侧竖向构件受均布水平荷载
    total_load = distributed_load * frame_height  # 总荷载 F = q*h
    
    # 静力平衡方程求解
    # ΣFx = 0: HA - q*h = 0
    HA = total_load  # A点水平反力
    
    # ΣMa = 0 (对A点取矩): VB*L - q*h*(h/2) = 0
    VB = (total_load * frame_height / 2) / frame_width  # B点竖向反力
    
    # ΣFy = 0: VA + VB = 0
    VA = -VB  # A点竖向反力
    
    result = {
        'HA': HA,
        'VA': VA,
        'VB': VB,
        'total_load': total_load,
        'load_centroid_height': frame_height / 2
    }
    
    metadata = {
        'frame_height': frame_height,
        'frame_width': frame_width,
        'distributed_load': distributed_load,
        'equilibrium_check': {
            'sum_Fx': HA - total_load,
            'sum_Fy': VA + VB,
            'sum_Ma': VB * frame_width - total_load * frame_height / 2
        }
    }
    
    return {'result': result, 'metadata': metadata}


def calculate_section_moment(
    section_position: str,
    x_coord: float,
    y_coord: float,
    reactions: Dict,
    frame_height: float,
    distributed_load: float
) -> Dict:
    """
    计算刚架任意截面的弯矩
    
    Parameters:
    -----------
    section_position : str
        截面位置 ("left_vertical", "top_horizontal", "right_vertical")
    x_coord : float
        截面x坐标 (m)
    y_coord : float
        截面y坐标 (m)
    reactions : dict
        支座反力字典
    frame_height : float
        刚架高度 h (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        截面弯矩值 (kN·m)
    """
    HA = reactions['HA']
    VA = reactions['VA']
    
    if section_position == "left_vertical":
        # 左侧竖向构件 (x=0, 0≤y≤h)
        if not (0 <= y_coord <= frame_height):
            raise ValueError(f"y坐标必须在[0, {frame_height}]范围内")
        # 从A点向上取截面，弯矩 M = HA * y
        moment = HA * y_coord
        
    elif section_position == "top_horizontal":
        # 顶部水平构件 (0≤x≤L, y=h)
        # 从左端C点向右取截面，弯矩 M = HA * h - VA * x
        moment = HA * frame_height - VA * x_coord
        
    elif section_position == "right_vertical":
        # 右侧竖向构件 (x=L, 0≤y≤h)
        if not (0 <= y_coord <= frame_height):
            raise ValueError(f"y坐标必须在[0, {frame_height}]范围内")
        # 从B点向上取截面，考虑均布荷载
        # 截面以上荷载 = q * (h - y)
        load_above = distributed_load * (frame_height - y_coord)
        # 弯矩 M = load_above * (h - y) / 2
        moment = load_above * (frame_height - y_coord) / 2
        
    else:
        raise ValueError("无效的截面位置")
    
    metadata = {
        'section_position': section_position,
        'coordinates': {'x': x_coord, 'y': y_coord},
        'applied_reactions': {'HA': HA, 'VA': VA}
    }
    
    return {'result': moment, 'metadata': metadata}


def calculate_shear_force(
    section_position: str,
    x_coord: float,
    y_coord: float,
    reactions: Dict,
    frame_height: float,
    distributed_load: float
) -> Dict:
    """
    计算刚架任意截面的剪力
    
    Parameters:
    -----------
    section_position : str
        截面位置
    x_coord : float
        截面x坐标 (m)
    y_coord : float
        截面y坐标 (m)
    reactions : dict
        支座反力字典
    frame_height : float
        刚架高度 h (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        截面剪力值 (kN)
    """
    HA = reactions['HA']
    VA = reactions['VA']
    
    if section_position == "left_vertical":
        shear = -HA  # 竖向构件的剪力
        
    elif section_position == "top_horizontal":
        shear = VA  # 水平构件的剪力
        
    elif section_position == "right_vertical":
        load_above = distributed_load * (frame_height - y_coord)
        shear = load_above
        
    else:
        raise ValueError("无效的截面位置")
    
    return {'result': shear, 'metadata': {'section_position': section_position}}


# ==================== 第二层：组合函数 ====================

def analyze_rigid_frame(
    frame_height: float,
    frame_width: float,
    distributed_load: float
) -> Dict:
    """
    完整分析刚架结构
    
    Parameters:
    -----------
    frame_height : float
        刚架高度 h (m)
    frame_width : float
        刚架宽度 L (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    
    Returns:
    --------
    dict : 包含支座反力、关键截面弯矩和剪力
    """
    # 计算支座反力
    reactions_result = calculate_support_reactions(
        frame_height, frame_width, distributed_load
    )
    reactions = reactions_result['result']
    
    # 计算关键截面弯矩
    # C点 (左上角)
    moment_C = calculate_section_moment(
        "left_vertical", 0, frame_height, reactions, frame_height, distributed_load
    )
    
    # A点 (左下角)
    moment_A = calculate_section_moment(
        "left_vertical", 0, 0, reactions, frame_height, distributed_load
    )
    
    # B点 (右下角)
    moment_B = calculate_section_moment(
        "right_vertical", frame_width, 0, reactions, frame_height, distributed_load
    )
    
    result = {
        'reactions': reactions,
        'moments': {
            'C': moment_C['result'],
            'A': moment_A['result'],
            'B': moment_B['result']
        }
    }
    
    metadata = {
        'frame_geometry': {
            'height': frame_height,
            'width': frame_width
        },
        'loading': {
            'distributed_load': distributed_load,
            'total_load': reactions['total_load']
        }
    }
    
    return {'result': result, 'metadata': metadata}


def generate_moment_diagram_data(
    frame_height: float,
    frame_width: float,
    distributed_load: float,
    num_points: int = 50
) -> Dict:
    """
    生成弯矩图数据
    
    Parameters:
    -----------
    frame_height : float
        刚架高度 h (m)
    frame_width : float
        刚架宽度 L (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    num_points : int
        每段的采样点数
    
    Returns:
    --------
    dict : 包含各构件的弯矩分布数据
    """
    reactions_result = calculate_support_reactions(
        frame_height, frame_width, distributed_load
    )
    reactions = reactions_result['result']
    
    # 左侧竖向构件
    y_left = np.linspace(0, frame_height, num_points)
    moments_left = [
        calculate_section_moment(
            "left_vertical", 0, y, reactions, frame_height, distributed_load
        )['result'] for y in y_left
    ]
    
    # 顶部水平构件
    x_top = np.linspace(0, frame_width, num_points)
    moments_top = [
        calculate_section_moment(
            "top_horizontal", x, frame_height, reactions, frame_height, distributed_load
        )['result'] for x in x_top
    ]
    
    # 右侧竖向构件
    y_right = np.linspace(0, frame_height, num_points)
    moments_right = [
        calculate_section_moment(
            "right_vertical", frame_width, y, reactions, frame_height, distributed_load
        )['result'] for y in y_right
    ]
    
    result = {
        'left_member': {'y': y_left.tolist(), 'moments': moments_left},
        'top_member': {'x': x_top.tolist(), 'moments': moments_top},
        'right_member': {'y': y_right.tolist(), 'moments': moments_right}
    }
    
    metadata = {
        'num_points': num_points,
        'max_moment': max(max(moments_left), max(moments_top), max(moments_right)),
        'min_moment': min(min(moments_left), min(moments_top), min(moments_right))
    }
    
    return {'result': result, 'metadata': metadata}


# ==================== 第三层：可视化函数 ====================

def plot_moment_diagram(
    frame_height: float,
    frame_width: float,
    distributed_load: float,
    save_path: str = "./tool_images/moment_diagram.png"
) -> Dict:
    """
    绘制刚架弯矩图
    
    Parameters:
    -----------
    frame_height : float
        刚架高度 h (m)
    frame_width : float
        刚架宽度 L (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    save_path : str
        图像保存路径
    
    Returns:
    --------
    dict : 包含图像路径
    """
    diagram_data = generate_moment_diagram_data(
        frame_height, frame_width, distributed_load
    )
    data = diagram_data['result']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制刚架轮廓
    frame_x = [0, 0, frame_width, frame_width]
    frame_y = [0, frame_height, frame_height, 0]
    ax.plot(frame_x, frame_y, 'k-', linewidth=2, label='Frame')
    
    # 绘制弯矩图（在受拉侧）
    scale = 0.3 * frame_width / abs(diagram_data['metadata']['max_moment'])
    
    # 左侧构件
    y_left = np.array(data['left_member']['y'])
    m_left = np.array(data['left_member']['moments'])
    ax.fill_betweenx(y_left, 0, -m_left * scale, alpha=0.3, color='blue')
    ax.plot(-m_left * scale, y_left, 'b-', linewidth=1.5)
    
    # 顶部构件
    x_top = np.array(data['top_member']['x'])
    m_top = np.array(data['top_member']['moments'])
    ax.fill_between(x_top, frame_height, frame_height - m_top * scale, 
                     alpha=0.3, color='red')
    ax.plot(x_top, frame_height - m_top * scale, 'r-', linewidth=1.5)
    
    # 右侧构件
    y_right = np.array(data['right_member']['y'])
    m_right = np.array(data['right_member']['moments'])
    ax.fill_betweenx(y_right, frame_width, frame_width + m_right * scale, 
                      alpha=0.3, color='green')
    ax.plot(frame_width + m_right * scale, y_right, 'g-', linewidth=1.5)
    
    # 标注关键点
    moment_C = m_left[-1]
    ax.plot(0, frame_height, 'ro', markersize=8)
    ax.text(-0.1 * frame_width, frame_height + 0.05 * frame_height, 
            f'C: M={moment_C:.2f} kN·m', fontsize=10, ha='right')
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f'Bending Moment Diagram (q={distributed_load} kN/m, h={frame_height} m)', 
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'max_moment': diagram_data['metadata']['max_moment'],
            'min_moment': diagram_data['metadata']['min_moment']
        }
    }


def plot_frame_with_loads(
    frame_height: float,
    frame_width: float,
    distributed_load: float,
    save_path: str = "./tool_images/frame_structure.png"
) -> Dict:
    """
    绘制刚架结构及荷载示意图
    
    Parameters:
    -----------
    frame_height : float
        刚架高度 h (m)
    frame_width : float
        刚架宽度 L (m)
    distributed_load : float
        均布荷载强度 q (kN/m)
    save_path : str
        图像保存路径
    
    Returns:
    --------
    dict : 包含图像路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制刚架
    frame_x = [0, 0, frame_width, frame_width]
    frame_y = [0, frame_height, frame_height, 0]
    ax.plot(frame_x, frame_y, 'k-', linewidth=3)
    
    # 绘制支座
    # A点固定铰支座
    ax.plot(0, 0, 'ks', markersize=12)
    ax.text(-0.1 * frame_width, -0.1 * frame_height, 'A', fontsize=14, ha='right')
    
    # B点铰支座
    ax.plot(frame_width, 0, 'ko', markersize=12)
    ax.text(frame_width + 0.05 * frame_width, -0.1 * frame_height, 'B', 
            fontsize=14, ha='left')
    
    # C点
    ax.plot(0, frame_height, 'ro', markersize=10)
    ax.text(-0.1 * frame_width, frame_height + 0.05 * frame_height, 'C', 
            fontsize=14, ha='right', color='red')
    
    # 绘制均布荷载
    num_arrows = 10
    y_arrows = np.linspace(0.05 * frame_height, 0.95 * frame_height, num_arrows)
    arrow_length = 0.15 * frame_width
    
    for y in y_arrows:
        ax.arrow(frame_width + arrow_length, y, -arrow_length * 0.8, 0,
                head_width=0.05 * frame_height, head_length=0.03 * frame_width,
                fc='blue', ec='blue', linewidth=1.5)
    
    ax.text(frame_width + arrow_length * 1.2, frame_height / 2, 
            f'q = {distributed_load} kN/m', fontsize=12, rotation=-90, 
            va='center', color='blue')
    
    # 标注尺寸
    ax.annotate('', xy=(0, -0.2 * frame_height), xytext=(frame_width, -0.2 * frame_height),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(frame_width / 2, -0.25 * frame_height, f'L = {frame_width} m', 
            fontsize=12, ha='center')
    
    ax.annotate('', xy=(-0.2 * frame_width, 0), xytext=(-0.2 * frame_width, frame_height),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(-0.3 * frame_width, frame_height / 2, f'h = {frame_height} m', 
            fontsize=12, rotation=90, va='center')
    
    ax.set_xlim(-0.5 * frame_width, frame_width + arrow_length * 1.5)
    ax.set_ylim(-0.4 * frame_height, frame_height + 0.2 * frame_height)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Rigid Frame Structure with Distributed Load', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {'file_type': 'png'}
    }


# ==================== 主函数：3个场景演示 ====================

def main():
    """
    演示刚架弯矩计算工具包的三个应用场景
    """
    
    print("=" * 60)
    print("场景1：原始问题 - 计算C点弯矩")
    print("=" * 60)
    print("问题描述：矩形刚架，左下角A为固定铰支座，右下角B为铰支座")
    print("右侧竖向构件受均布水平荷载q，高度为h，求左上角C点的弯矩")
    print("-" * 60)
    
    # 设定参数（使用符号值进行计算）
    h = 4.0  # 刚架高度 (m)
    L = 6.0  # 刚架宽度 (m)
    q = 10.0  # 均布荷载强度 (kN/m)
    
    # 步骤1：绘制结构示意图
    print("\n步骤1：绘制刚架结构及荷载示意图")
    print("调用函数：plot_frame_with_loads()")
    structure_plot = plot_frame_with_loads(h, L, q)
    print(f"FUNCTION_CALL: plot_frame_with_loads | PARAMS: {{h={h}, L={L}, q={q}}} | RESULT: {structure_plot}")
    
    # 步骤2：计算支座反力
    print("\n步骤2：计算支座反力")
    print("调用函数：calculate_support_reactions()")
    reactions_result = calculate_support_reactions(h, L, q)
    reactions = reactions_result['result']
    print(f"FUNCTION_CALL: calculate_support_reactions | PARAMS: {{h={h}, L={L}, q={q}}} | RESULT: {reactions}")
    print(f"  支座A水平反力 HA = {reactions['HA']:.2f} kN")
    print(f"  支座A竖向反力 VA = {reactions['VA']:.2f} kN")
    print(f"  支座B竖向反力 VB = {reactions['VB']:.2f} kN")
    
    # 步骤3：计算C点弯矩
    print("\n步骤3：计算C点（左上角）的弯矩")
    print("调用函数：calculate_section_moment()")
    moment_C_result = calculate_section_moment(
        "left_vertical", 0, h, reactions, h, q
    )
    moment_C = moment_C_result['result']
    print(f"FUNCTION_CALL: calculate_section_moment | PARAMS: {{section='left_vertical', x=0, y={h}, reactions={reactions}}} | RESULT: {moment_C_result}")
    print(f"  C点弯矩 M_C = {moment_C:.2f} kN·m")
    
    # 步骤4：验证标准答案
    print("\n步骤4：验证标准答案 M_C = q*h²")
    expected_moment = q * h**2
    print(f"  理论值：M_C = q*h² = {q} × {h}² = {expected_moment:.2f} kN·m")
    print(f"  计算值：M_C = {moment_C:.2f} kN·m")
    print(f"  误差：{abs(moment_C - expected_moment):.6f} kN·m")
    
    # 步骤5：绘制弯矩图
    print("\n步骤5：绘制完整弯矩图")
    print("调用函数：plot_moment_diagram()")
    moment_diagram = plot_moment_diagram(h, L, q)
    print(f"FUNCTION_CALL: plot_moment_diagram | PARAMS: {{h={h}, L={L}, q={q}}} | RESULT: {moment_diagram}")
    
    print(f"\nFINAL_ANSWER: M_C = {moment_C:.2f} kN·m = {q}*{h}² kN·m")
    
    
    print("\n" + "=" * 60)
    print("场景2：参数敏感性分析 - 不同高度对C点弯矩的影响")
    print("=" * 60)
    print("问题描述：固定荷载强度q和宽度L，改变刚架高度h，分析C点弯矩变化")
    print("-" * 60)
    
    # 步骤1：设定参数范围
    q_fixed = 10.0  # kN/m
    L_fixed = 6.0   # m
    heights = [2.0, 3.0, 4.0, 5.0, 6.0]  # m
    
    print(f"\n步骤1：设定参数 q={q_fixed} kN/m, L={L_fixed} m")
    print(f"  高度范围：h = {heights} m")
    
    # 步骤2：计算不同高度下的C点弯矩
    print("\n步骤2：计算不同高度下的C点弯矩")
    moments_C = []
    for h_var in heights:
        reactions_var = calculate_support_reactions(h_var, L_fixed, q_fixed)['result']
        moment_var = calculate_section_moment(
            "left_vertical", 0, h_var, reactions_var, h_var, q_fixed
        )['result']
        moments_C.append(moment_var)
        print(f"  h={h_var} m: M_C={moment_var:.2f} kN·m (理论值={q_fixed*h_var**2:.2f} kN·m)")
    
    print(f"FUNCTION_CALL: calculate_section_moment (multiple) | PARAMS: {{heights={heights}}} | RESULT: {{moments={moments_C}}}")
    
    # 步骤3：绘制参数敏感性曲线
    print("\n步骤3：绘制M_C vs h曲线")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(heights, moments_C, 'bo-', linewidth=2, markersize=8, label='Calculated M_C')
    theoretical = [q_fixed * h**2 for h in heights]
    ax.plot(heights, theoretical, 'r--', linewidth=2, label='Theoretical q*h²')
    ax.set_xlabel('Frame Height h (m)', fontsize=12)
    ax.set_ylabel('Moment at C (kN·m)', fontsize=12)
    ax.set_title('Sensitivity Analysis: M_C vs Frame Height', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    sensitivity_path = "./tool_images/sensitivity_analysis.png"
    plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {sensitivity_path}")
    
    print(f"\nFINAL_ANSWER: C点弯矩随高度呈二次方关系增长，符合 M_C = q*h² 理论公式")
    
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()