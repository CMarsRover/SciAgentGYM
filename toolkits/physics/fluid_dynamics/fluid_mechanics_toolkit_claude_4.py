# Filename: fluid_mechanics_toolkit.py

"""
流体力学工具包 - 加速容器中的液体行为分析
Fluid Mechanics Toolkit - Liquid Behavior in Accelerating Containers

本工具包用于分析容器在加速运动时液体表面的倾斜现象，
并计算防止液体溢出所需的最小容器高度。

核心物理原理：
1. 牛顿第二定律：系统加速度计算
2. 非惯性参考系：液体表面倾斜角度
3. 几何关系：临界高度计算
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

# 全局常量
GRAVITY = 9.8  # m/s^2, 重力加速度

# 创建输出目录
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础物理计算
# ============================================================================

def calculate_system_acceleration(m1: float, m2: float, mu: float, g: float = GRAVITY) -> Dict:
    """
    计算滑轮系统的加速度
    
    物理原理：
    - 对m2：T - m2*g = -m2*a (向下为正)
    - 对m1+水：T - f = (m1+m_water)*a
    - 摩擦力：f = μ*N = μ*(m1+m_water)*g
    
    联立求解：a = (m2*g - μ*(m1+m_water)*g) / (m1+m_water+m2)
    
    Parameters:
    -----------
    m1 : float
        容器质量 (kg)
    m2 : float
        重物质量 (kg)
    mu : float
        摩擦系数
    g : float
        重力加速度 (m/s^2)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        加速度值及计算详情
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("质量必须为正值")
    if mu < 0 or mu > 1:
        raise ValueError("摩擦系数必须在[0,1]范围内")
    
    # 计算水的质量
    # 注意：这里需要从外部传入水的质量，或者在组合函数中计算
    # 为了保持原子函数的独立性，这里假设传入的是总质量
    
    numerator = m2 * g - mu * m1 * g
    denominator = m1 + m2
    acceleration = numerator / denominator
    
    metadata = {
        'driving_force': m2 * g,
        'friction_force': mu * m1 * g,
        'total_mass': m1 + m2,
        'formula': 'a = (m2*g - μ*m1*g) / (m1 + m2)'
    }
    
    return {
        'result': acceleration,
        'metadata': metadata
    }


def calculate_water_mass(base_area: float, height: float, density: float = 1000.0) -> Dict:
    """
    计算容器中水的质量
    
    Parameters:
    -----------
    base_area : float
        容器底面积 (m^2)
    height : float
        水位高度 (m)
    density : float
        水的密度 (kg/m^3), 默认1000
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
    """
    if base_area <= 0 or height <= 0:
        raise ValueError("面积和高度必须为正值")
    
    volume = base_area * height
    mass = density * volume
    
    return {
        'result': mass,
        'metadata': {
            'volume': volume,
            'density': density,
            'formula': 'm = ρ * V = ρ * A * h'
        }
    }


def calculate_tilt_angle(acceleration: float, g: float = GRAVITY) -> Dict:
    """
    计算液体表面相对于水平面的倾斜角度
    
    物理原理：
    在加速参考系中，液体受到惯性力和重力的合力
    液面垂直于合力方向
    tan(θ) = a/g
    
    Parameters:
    -----------
    acceleration : float
        容器加速度 (m/s^2)
    g : float
        重力加速度 (m/s^2)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        倾斜角度(弧度)及相关信息
    """
    if acceleration < 0:
        raise ValueError("加速度不能为负值")
    
    tan_theta = acceleration / g
    theta_rad = np.arctan(tan_theta)
    theta_deg = np.degrees(theta_rad)
    
    return {
        'result': theta_rad,
        'metadata': {
            'angle_degrees': theta_deg,
            'tan_theta': tan_theta,
            'formula': 'tan(θ) = a/g'
        }
    }


def calculate_height_difference(base_width: float, tilt_angle: float) -> Dict:
    """
    计算液面两端的高度差
    
    几何关系：Δh = (a/2) * tan(θ)
    其中a是容器宽度，θ是倾斜角
    
    Parameters:
    -----------
    base_width : float
        容器宽度 (m)
    tilt_angle : float
        倾斜角度 (弧度)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        高度差(m)
    """
    if base_width <= 0:
        raise ValueError("容器宽度必须为正值")
    
    height_diff = (base_width / 2) * np.tan(tilt_angle)
    
    return {
        'result': height_diff,
        'metadata': {
            'half_width': base_width / 2,
            'tan_angle': np.tan(tilt_angle),
            'formula': 'Δh = (a/2) * tan(θ)'
        }
    }


def calculate_minimum_height(initial_height: float, height_difference: float) -> Dict:
    """
    计算防止溢出的最小容器高度
    
    临界条件：H = h + Δh
    其中h是初始水位，Δh是液面高度差
    
    Parameters:
    -----------
    initial_height : float
        初始水位高度 (m)
    height_difference : float
        液面高度差 (m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
    """
    if initial_height <= 0:
        raise ValueError("初始高度必须为正值")
    
    min_height = initial_height + height_difference
    
    return {
        'result': min_height,
        'metadata': {
            'initial_height': initial_height,
            'height_difference': height_difference,
            'formula': 'H = h + Δh'
        }
    }


# ============================================================================
# 第二层：组合函数 - 完整求解流程
# ============================================================================

def solve_minimum_container_height(
    container_mass: float,
    hanging_mass: float,
    friction_coef: float,
    base_width: float,
    initial_water_height: float,
    water_density: float = 1000.0
) -> Dict:
    """
    求解防止水溢出的最小容器高度（完整流程）
    
    Parameters:
    -----------
    container_mass : float
        容器质量 (kg)
    hanging_mass : float
        悬挂重物质量 (kg)
    friction_coef : float
        摩擦系数
    base_width : float
        容器底边长 (m)
    initial_water_height : float
        初始水位高度 (m)
    water_density : float
        水的密度 (kg/m^3)
    
    Returns:
    --------
    dict : 包含最终结果和所有中间步骤
    """
    results = {}
    
    # 步骤1：计算水的质量
    base_area = base_width ** 2
    water_mass_result = calculate_water_mass(base_area, initial_water_height, water_density)
    water_mass = water_mass_result['result']
    results['water_mass'] = water_mass_result
    
    # 步骤2：计算系统总质量和加速度
    total_mass = container_mass + water_mass
    accel_result = calculate_system_acceleration(total_mass, hanging_mass, friction_coef)
    acceleration = accel_result['result']
    results['acceleration'] = accel_result
    
    # 步骤3：计算液面倾斜角度
    tilt_result = calculate_tilt_angle(acceleration)
    tilt_angle = tilt_result['result']
    results['tilt_angle'] = tilt_result
    
    # 步骤4：计算液面高度差
    height_diff_result = calculate_height_difference(base_width, tilt_angle)
    height_diff = height_diff_result['result']
    results['height_difference'] = height_diff_result
    
    # 步骤5：计算最小容器高度
    min_height_result = calculate_minimum_height(initial_water_height, height_diff)
    min_height = min_height_result['result']
    results['minimum_height'] = min_height_result
    
    # 保存中间结果
    filepath = './mid_result/physics/container_calculation.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return {
        'result': min_height,
        'metadata': {
            'all_steps': results,
            'saved_to': filepath
        }
    }


def analyze_parameter_sensitivity(
    base_params: Dict,
    vary_param: str,
    param_range: List[float]
) -> Dict:
    """
    分析参数敏感性
    
    Parameters:
    -----------
    base_params : dict
        基准参数字典，包含：
        - container_mass, hanging_mass, friction_coef, 
          base_width, initial_water_height
    vary_param : str
        要变化的参数名称
    param_range : list
        参数变化范围
    
    Returns:
    --------
    dict : 敏感性分析结果
    """
    results = []
    
    for value in param_range:
        params = base_params.copy()
        params[vary_param] = value
        
        result = solve_minimum_container_height(**params)
        results.append({
            'param_value': value,
            'min_height': result['result']
        })
    
    # 保存结果
    filepath = './mid_result/physics/sensitivity_analysis.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'varied_parameter': vary_param,
            'results': results
        }, f, indent=2)
    
    return {
        'result': results,
        'metadata': {
            'varied_parameter': vary_param,
            'num_points': len(results),
            'saved_to': filepath
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_liquid_surface(
    base_width: float,
    initial_height: float,
    tilt_angle: float,
    min_height: float,
    save_path: str = './tool_images/liquid_surface.png'
) -> Dict:
    """
    可视化液体表面倾斜情况
    
    Parameters:
    -----------
    base_width : float
        容器宽度 (m)
    initial_height : float
        初始水位 (m)
    tilt_angle : float
        倾斜角度 (弧度)
    min_height : float
        最小容器高度 (m)
    save_path : str
        保存路径
    
    Returns:
    --------
    dict : 图像信息
    """
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 容器轮廓
    container_x = [0, base_width, base_width, 0, 0]
    container_y = [0, 0, min_height, min_height, 0]
    ax.plot(container_x, container_y, 'k-', linewidth=2, label='Container')
    
    # 初始水位（水平）
    ax.plot([0, base_width], [initial_height, initial_height], 
            'b--', linewidth=1.5, label='Initial Water Level')
    
    # 倾斜后的液面
    height_diff = (base_width / 2) * np.tan(tilt_angle)
    left_height = initial_height - height_diff
    right_height = initial_height + height_diff
    ax.plot([0, base_width], [left_height, right_height], 
            'r-', linewidth=2, label='Tilted Surface')
    
    # 填充水域
    water_x = [0, base_width, base_width, 0]
    water_y = [0, 0, right_height, left_height]
    ax.fill(water_x, water_y, color='cyan', alpha=0.3)
    
    # 标注关键尺寸
    ax.annotate(f'H = {min_height:.4f} m', 
                xy=(base_width/2, min_height), 
                xytext=(base_width/2, min_height + 0.02),
                ha='center', fontsize=12, fontweight='bold')
    
    ax.annotate(f'h = {initial_height:.3f} m', 
                xy=(0, initial_height), 
                xytext=(-0.03, initial_height),
                ha='right', fontsize=10)
    
    ax.annotate(f'Δh = {height_diff:.4f} m', 
                xy=(base_width, right_height), 
                xytext=(base_width + 0.03, right_height),
                ha='left', fontsize=10, color='red')
    
    # 标注倾斜角
    angle_deg = np.degrees(tilt_angle)
    ax.text(base_width/2, initial_height - 0.03, 
            f'θ = {angle_deg:.2f}°', 
            ha='center', fontsize=10, color='red')
    
    ax.set_xlabel('Width (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('Liquid Surface Tilt in Accelerating Container', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'dimensions': '10x8 inches',
            'dpi': 300
        }
    }


def plot_sensitivity_analysis(
    sensitivity_data: List[Dict],
    param_name: str,
    param_unit: str,
    save_path: str = './tool_images/sensitivity_analysis.png'
) -> Dict:
    """
    绘制参数敏感性分析图
    
    Parameters:
    -----------
    sensitivity_data : list
        敏感性数据，格式：[{'param_value': x, 'min_height': y}, ...]
    param_name : str
        参数名称
    param_unit : str
        参数单位
    save_path : str
        保存路径
    
    Returns:
    --------
    dict : 图像信息
    """
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    param_values = [d['param_value'] for d in sensitivity_data]
    min_heights = [d['min_height'] for d in sensitivity_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, min_heights, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel(f'{param_name} ({param_unit})', fontsize=12)
    ax.set_ylabel('Minimum Container Height (m)', fontsize=12)
    ax.set_title(f'Sensitivity Analysis: {param_name} vs Minimum Height', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'num_data_points': len(sensitivity_data)
        }
    }


def create_force_diagram(
    container_mass: float,
    water_mass: float,
    hanging_mass: float,
    friction_force: float,
    tension: float,
    acceleration: float,
    save_path: str = './tool_images/force_diagram.png'
) -> Dict:
    """
    绘制系统受力分析图
    
    Parameters:
    -----------
    container_mass : float
        容器质量 (kg)
    water_mass : float
        水的质量 (kg)
    hanging_mass : float
        悬挂物质量 (kg)
    friction_force : float
        摩擦力 (N)
    tension : float
        绳子张力 (N)
    acceleration : float
        加速度 (m/s^2)
    save_path : str
        保存路径
    
    Returns:
    --------
    dict : 图像信息
    """
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：容器受力
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    
    # 容器
    rect = plt.Rectangle((-0.5, -0.3), 1, 0.6, fill=True, 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    
    # 张力（向右）
    ax1.arrow(0.5, 0, 0.8, 0, head_width=0.15, head_length=0.15, 
              fc='red', ec='red', linewidth=2)
    ax1.text(1.2, 0.3, f'T = {tension:.1f} N', fontsize=11, color='red')
    
    # 摩擦力（向左）
    ax1.arrow(-0.5, 0, -0.6, 0, head_width=0.15, head_length=0.15, 
              fc='orange', ec='orange', linewidth=2)
    ax1.text(-1.3, 0.3, f'f = {friction_force:.1f} N', fontsize=11, color='orange')
    
    # 重力（向下）
    total_weight = (container_mass + water_mass) * GRAVITY
    ax1.arrow(0, -0.3, 0, -0.6, head_width=0.15, head_length=0.15, 
              fc='blue', ec='blue', linewidth=2)
    ax1.text(0.3, -0.8, f'W = {total_weight:.1f} N', fontsize=11, color='blue')
    
    # 支持力（向上）
    ax1.arrow(0, -0.3, 0, -0.05, head_width=0.15, head_length=0.05, 
              fc='green', ec='green', linewidth=2)
    ax1.text(0.3, -0.15, f'N = {total_weight:.1f} N', fontsize=11, color='green')
    
    ax1.text(0, 1.5, f'Container + Water\nm = {container_mass + water_mass:.1f} kg\na = {acceleration:.3f} m/s²', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_title('Container Forces', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # 右图：悬挂物受力
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    
    # 悬挂物
    circle = plt.Circle((0, 0), 0.3, fill=True, 
                        facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax2.add_patch(circle)
    
    # 张力（向上）
    ax2.arrow(0, 0.3, 0, 0.6, head_width=0.15, head_length=0.15, 
              fc='red', ec='red', linewidth=2)
    ax2.text(0.3, 0.8, f'T = {tension:.1f} N', fontsize=11, color='red')
    
    # 重力（向下）
    hanging_weight = hanging_mass * GRAVITY
    ax2.arrow(0, -0.3, 0, -0.8, head_width=0.15, head_length=0.15, 
              fc='blue', ec='blue', linewidth=2)
    ax2.text(0.3, -0.9, f'W = {hanging_weight:.1f} N', fontsize=11, color='blue')
    
    ax2.text(0, 1.5, f'Hanging Mass\nm = {hanging_mass:.1f} kg\na = {acceleration:.3f} m/s²', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_title('Hanging Mass Forces', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'diagram_type': 'force_analysis'
        }
    }


# ============================================================================
# 主函数 - 三个场景演示
# ============================================================================

def main():
    """
    主函数：演示三个不同场景的计算
    """
    
    print("=" * 80)
    print("场景1：求解原始问题 - 计算防止水溢出的最小容器高度")
    print("=" * 80)
    print("问题描述：")
    print("容器底面积：200×200 mm² = 0.04 m²")
    print("容器质量：m₁ = 4 kg")
    print("初始水位：h = 150 mm = 0.15 m")
    print("悬挂重物：m₂ = 25 kg")
    print("摩擦系数：μ = 0.3")
    print("求：最小容器高度 H")
    print("-" * 80)
    
    # 步骤1：计算水的质量
    print("\n步骤1：计算水的质量")
    base_area = 0.2 * 0.2  # m^2
    initial_height = 0.15  # m
    water_mass_result = calculate_water_mass(base_area, initial_height)
    water_mass = water_mass_result['result']
    print(f"FUNCTION_CALL: calculate_water_mass | PARAMS: {{base_area: {base_area}, height: {initial_height}}} | RESULT: {water_mass_result}")
    print(f"水的质量：{water_mass:.2f} kg")
    
    # 步骤2：计算系统加速度
    print("\n步骤2：计算系统加速度")
    container_mass = 4.0  # kg
    hanging_mass = 25.0  # kg
    friction_coef = 0.3
    total_mass = container_mass + water_mass
    accel_result = calculate_system_acceleration(total_mass, hanging_mass, friction_coef)
    acceleration = accel_result['result']
    print(f"FUNCTION_CALL: calculate_system_acceleration | PARAMS: {{m1: {total_mass}, m2: {hanging_mass}, mu: {friction_coef}}} | RESULT: {accel_result}")
    print(f"系统加速度：{acceleration:.4f} m/s²")
    
    # 步骤3：计算液面倾斜角度
    print("\n步骤3：计算液面倾斜角度")
    tilt_result = calculate_tilt_angle(acceleration)
    tilt_angle = tilt_result['result']
    print(f"FUNCTION_CALL: calculate_tilt_angle | PARAMS: {{acceleration: {acceleration}}} | RESULT: {tilt_result}")
    print(f"倾斜角度：{tilt_result['metadata']['angle_degrees']:.2f}°")
    
    # 步骤4：计算液面高度差
    print("\n步骤4：计算液面高度差")
    base_width = 0.2  # m
    height_diff_result = calculate_height_difference(base_width, tilt_angle)
    height_diff = height_diff_result['result']
    print(f"FUNCTION_CALL: calculate_height_difference | PARAMS: {{base_width: {base_width}, tilt_angle: {tilt_angle}}} | RESULT: {height_diff_result}")
    print(f"液面高度差：{height_diff:.4f} m")
    
    # 步骤5：计算最小容器高度
    print("\n步骤5：计算最小容器高度")
    min_height_result = calculate_minimum_height(initial_height, height_diff)
    min_height = min_height_result['result']
    print(f"FUNCTION_CALL: calculate_minimum_height | PARAMS: {{initial_height: {initial_height}, height_difference: {height_diff}}} | RESULT: {min_height_result}")
    print(f"最小容器高度：{min_height:.4f} m = {min_height*1000:.1f} mm")
    
    # 步骤6：可视化液面倾斜
    print("\n步骤6：可视化液面倾斜情况")
    vis_result = visualize_liquid_surface(base_width, initial_height, tilt_angle, min_height)
    print(f"FUNCTION_CALL: visualize_liquid_surface | PARAMS: {{base_width: {base_width}, initial_height: {initial_height}, tilt_angle: {tilt_angle}, min_height: {min_height}}} | RESULT: {vis_result}")
    
    # 步骤7：绘制受力分析图
    print("\n步骤7：绘制系统受力分析图")
    friction_force = friction_coef * total_mass * GRAVITY
    tension = hanging_mass * GRAVITY - hanging_mass * acceleration
    force_diagram_result = create_force_diagram(
        container_mass, water_mass, hanging_mass, 
        friction_force, tension, acceleration
    )
    print(f"FUNCTION_CALL: create_force_diagram | PARAMS: {{container_mass: {container_mass}, water_mass: {water_mass}, hanging_mass: {hanging_mass}, friction_force: {friction_force}, tension: {tension}, acceleration: {acceleration}}} | RESULT: {force_diagram_result}")
    
    print("\n" + "=" * 80)
    print(f"FINAL_ANSWER: {min_height:.3f} m (标准答案: 0.213 m)")
    print("=" * 80)
    
    
    print("\n\n")
    print("=" * 80)
    print("场景2：参数敏感性分析 - 摩擦系数对最小高度的影响")
    print("=" * 80)
    print("问题描述：")
    print("固定其他参数，改变摩擦系数μ从0.1到0.5，")
    print("分析最小容器高度H如何变化")
    print("-" * 80)
    
    # 步骤1：设置基准参数
    print("\n步骤1：设置基准参数")
    base_params = {
        'container_mass': 4.0,
        'hanging_mass': 25.0,
        'friction_coef': 0.3,
        'base_width': 0.2,
        'initial_water_height': 0.15
    }
    print(f"基准参数：{base_params}")
    
    # 步骤2：执行敏感性分析
    print("\n步骤2：执行摩擦系数敏感性分析")
    mu_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    sensitivity_result = analyze_parameter_sensitivity(
        base_params, 'friction_coef', mu_range
    )
    print(f"FUNCTION_CALL: analyze_parameter_sensitivity | PARAMS: {{base_params: {base_params}, vary_param: 'friction_coef', param_range: {mu_range}}} | RESULT: {sensitivity_result}")
    
    # 步骤3：可视化敏感性分析结果
    print("\n步骤3：可视化敏感性分析结果")
    plot_result = plot_sensitivity_analysis(
        sensitivity_result['result'], 
        'Friction Coefficient', 
        'dimensionless'
    )
    print(f"FUNCTION_CALL: plot_sensitivity_analysis | PARAMS: {{sensitivity_data: [...], param_name: 'Friction Coefficient', param_unit: 'dimensionless'}} | RESULT: {plot_result}")
    
    # 分析结果
    print("\n分析结果：")
    for data in sensitivity_result['result']:
        print(f"μ = {data['param_value']:.2f} → H = {data['min_height']:.4f} m")
    
    print("\n结论：摩擦系数越大，系统加速度越小，液面倾斜越小，所需最小高度越小")
    
    print("\n" + "=" * 80)
    final_mu_03 = [d for d in sensitivity_result['result'] if abs(d['param_value'] - 0.3) < 0.01][0]
    print(f"FINAL_ANSWER: 当μ=0.3时，H = {final_mu_03['min_height']:.3f} m")
    print("=" * 80)
    
    
    print("\n\n")
    print("=" * 80)
    print("场景3：不同初始水位的最小高度计算")
    print("=" * 80)
    print("问题描述：")
    print("保持其他参数不变，改变初始水位h从100mm到200mm，")
    print("计算每种情况下的最小容器高度H")
    print("-" * 80)
    
    # 步骤1：设置水位范围
    print("\n步骤1：设置初始水位范围")
    water_heights = [0.10, 0.12, 0.15, 0.18, 0.20]  # m
    print(f"水位范围：{[h*1000 for h in water_heights]} mm")
    
    # 步骤2：对每个水位计算最小高度
    print("\n步骤2：计算每个水位对应的最小容器高度")
    results_scenario3 = []
    
    for h in water_heights:
        # 使用完整求解函数
        result = solve_minimum_container_height(
            container_mass=4.0,
            hanging_mass=25.0,
            friction_coef=0.3,
            base_width=0.2,
            initial_water_height=h
        )
        
        min_h = result['result']
        results_scenario3.append({
            'initial_height': h,
            'minimum_height': min_h
        })
        
        print(f"FUNCTION_CALL: solve_minimum_container_height | PARAMS: {{container_mass: 4.0, hanging_mass: 25.0, friction_coef: 0.3, base_width: 0.2, initial_water_height: {h}}} | RESULT: {{result: {min_h:.4f}}}")
        print(f"  初始水位 h = {h*1000:.0f} mm → 最小高度 H = {min_h*1000:.1f} mm")
    
    # 步骤3：可视化结果
    print("\n步骤3：可视化不同水位的最小高度关系")
    sensitivity_data_h = [
        {'param_value': r['initial_height']*1000, 'min_height': r['minimum_height']}
        for r in results_scenario3
    ]
    plot_result_h = plot_sensitivity_analysis(
        sensitivity_data_h,
        'Initial Water Height',
        'mm',
        save_path='./tool_images/water_height_sensitivity.png'
    )
    print(f"FUNCTION_CALL: plot_sensitivity_analysis | PARAMS: {{sensitivity_data: [...], param_name: 'Initial Water Height', param_unit: 'mm'}} | RESULT: {plot_result_h}")
    
    # 分析趋势
    print("\n分析结果：")
    print("初始水位越高，所需的最小容器高度也越高（线性关系）")
    print("这是因为液面高度差Δh与初始水位h无关，仅取决于加速度和容器宽度")
    print("因此 H = h + Δh 呈线性增长")
    
    print("\n" + "=" * 80)
    target_result = [r for r in results_scenario3 if abs(r['initial_height'] - 0.15) < 0.01][0]
    print(f"FINAL_ANSWER: 当h=150mm时，H = {target_result['minimum_height']:.3f} m")
    print("=" * 80)


if __name__ == "__main__":
    main()