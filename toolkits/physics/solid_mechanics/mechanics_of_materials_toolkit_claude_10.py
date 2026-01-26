# Filename: mechanics_of_materials_toolkit.py
"""
Mechanics of Materials Toolkit for Stress Analysis
专业材料力学应力分析工具包

功能：
1. 薄壁压力容器应力计算
2. 斜截面应力变换
3. 焊缝应力分析
4. 应力可视化

依赖库：
- numpy: 数值计算
- matplotlib: 可视化
- scipy: 科学计算
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json

# 配置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 全局常量
RESULTS_DIR = "./mid_result/mechanics"
IMAGES_DIR = "./tool_images"

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ==================== 第一层：原子函数 ====================

def calculate_cross_sectional_area(outer_diameter: float, thickness: float) -> Dict:
    """
    计算薄壁圆管的横截面积
    
    Parameters:
    -----------
    outer_diameter : float
        外径 (mm)
    thickness : float
        壁厚 (mm)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        横截面积 (mm²) 和计算元数据
    """
    if outer_diameter <= 0:
        raise ValueError(f"外径必须为正数，当前值: {outer_diameter}")
    if thickness <= 0 or thickness >= outer_diameter / 2:
        raise ValueError(f"壁厚必须在 (0, {outer_diameter/2}) 范围内，当前值: {thickness}")
    
    # 薄壁假设：使用平均直径
    mean_diameter = outer_diameter - thickness
    area = np.pi * mean_diameter * thickness
    
    return {
        'result': area,
        'metadata': {
            'outer_diameter_mm': outer_diameter,
            'thickness_mm': thickness,
            'mean_diameter_mm': mean_diameter,
            'formula': 'A = π * d_mean * t'
        }
    }


def calculate_axial_stress(axial_force: float, area: float) -> Dict:
    """
    计算轴向正应力
    
    Parameters:
    -----------
    axial_force : float
        轴向力 (N)
    area : float
        横截面积 (mm²)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        轴向应力 (MPa) 和计算元数据
    """
    if area <= 0:
        raise ValueError(f"横截面积必须为正数，当前值: {area}")
    
    # 应力 = 力 / 面积，单位转换：N/mm² = MPa
    stress = axial_force / area
    
    return {
        'result': stress,
        'metadata': {
            'axial_force_N': axial_force,
            'area_mm2': area,
            'stress_MPa': stress,
            'formula': 'σ = F / A'
        }
    }


def calculate_hoop_stress(internal_pressure: float, mean_diameter: float, thickness: float) -> Dict:
    """
    计算环向应力（周向应力）
    
    Parameters:
    -----------
    internal_pressure : float
        内压 (MPa)
    mean_diameter : float
        平均直径 (mm)
    thickness : float
        壁厚 (mm)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        环向应力 (MPa) 和计算元数据
    """
    if internal_pressure < 0:
        raise ValueError(f"内压不能为负数，当前值: {internal_pressure}")
    if mean_diameter <= 0:
        raise ValueError(f"平均直径必须为正数，当前值: {mean_diameter}")
    if thickness <= 0:
        raise ValueError(f"壁厚必须为正数，当前值: {thickness}")
    
    # 薄壁圆筒环向应力公式
    hoop_stress = internal_pressure * mean_diameter / (2 * thickness)
    
    return {
        'result': hoop_stress,
        'metadata': {
            'internal_pressure_MPa': internal_pressure,
            'mean_diameter_mm': mean_diameter,
            'thickness_mm': thickness,
            'formula': 'σ_θ = p * d / (2 * t)'
        }
    }


def calculate_longitudinal_stress(internal_pressure: float, mean_diameter: float, thickness: float) -> Dict:
    """
    计算轴向应力（纵向应力，封闭端）
    
    Parameters:
    -----------
    internal_pressure : float
        内压 (MPa)
    mean_diameter : float
        平均直径 (mm)
    thickness : float
        壁厚 (mm)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        轴向应力 (MPa) 和计算元数据
    """
    if internal_pressure < 0:
        raise ValueError(f"内压不能为负数，当前值: {internal_pressure}")
    if mean_diameter <= 0:
        raise ValueError(f"平均直径必须为正数，当前值: {mean_diameter}")
    if thickness <= 0:
        raise ValueError(f"壁厚必须为正数，当前值: {thickness}")
    
    # 封闭端薄壁圆筒轴向应力公式
    longitudinal_stress = internal_pressure * mean_diameter / (4 * thickness)
    
    return {
        'result': longitudinal_stress,
        'metadata': {
            'internal_pressure_MPa': internal_pressure,
            'mean_diameter_mm': mean_diameter,
            'thickness_mm': thickness,
            'formula': 'σ_z = p * d / (4 * t)'
        }
    }


def transform_stress_to_inclined_plane(sigma_x: float, sigma_y: float, tau_xy: float, theta_deg: float) -> Dict:
    """
    应力变换：计算斜截面上的正应力和剪应力
    
    Parameters:
    -----------
    sigma_x : float
        x方向正应力 (MPa)
    sigma_y : float
        y方向正应力 (MPa)
    tau_xy : float
        xy平面剪应力 (MPa)
    theta_deg : float
        斜截面角度（相对于x轴，逆时针为正）(度)
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        斜截面应力分量和计算元数据
    """
    # 角度转弧度
    theta = np.radians(theta_deg)
    
    # 应力变换公式
    sigma_n = (sigma_x + sigma_y) / 2 + (sigma_x - sigma_y) / 2 * np.cos(2 * theta) + tau_xy * np.sin(2 * theta)
    tau_n = -(sigma_x - sigma_y) / 2 * np.sin(2 * theta) + tau_xy * np.cos(2 * theta)
    
    return {
        'result': {
            'normal_stress_MPa': sigma_n,
            'shear_stress_MPa': tau_n
        },
        'metadata': {
            'input_stresses': {
                'sigma_x_MPa': sigma_x,
                'sigma_y_MPa': sigma_y,
                'tau_xy_MPa': tau_xy
            },
            'angle_deg': theta_deg,
            'angle_rad': theta,
            'formulas': {
                'normal': 'σ_n = (σ_x + σ_y)/2 + (σ_x - σ_y)/2 * cos(2θ) + τ_xy * sin(2θ)',
                'shear': 'τ_n = -(σ_x - σ_y)/2 * sin(2θ) + τ_xy * cos(2θ)'
            }
        }
    }


def save_calculation_results(results: Dict, filename: str) -> Dict:
    """
    保存计算结果到JSON文件
    
    Parameters:
    -----------
    results : dict
        计算结果字典
    filename : str
        文件名（不含路径）
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        文件路径和保存信息
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'json',
            'size_bytes': os.path.getsize(filepath),
            'saved_at': filepath
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_weld_stress_under_axial_load(
    outer_diameter: float,
    thickness: float,
    axial_force: float,
    weld_angle_deg: float
) -> Dict:
    """
    分析轴向载荷下焊缝处的应力状态
    
    Parameters:
    -----------
    outer_diameter : float
        外径 (mm)
    thickness : float
        壁厚 (mm)
    axial_force : float
        轴向力 (N)
    weld_angle_deg : float
        焊缝螺旋角（相对于轴向）(度)
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        焊缝处应力分析结果
    """
    # 步骤1：计算横截面积
    area_result = calculate_cross_sectional_area(outer_diameter, thickness)
    area = area_result['result']
    
    # 步骤2：计算轴向正应力
    axial_stress_result = calculate_axial_stress(axial_force, area)
    sigma_z = axial_stress_result['result']
    
    # 步骤3：建立应力状态（轴向拉伸，单向应力状态）
    # 在圆柱坐标系中：σ_z = 轴向应力，σ_θ = 0（无内压），τ = 0
    sigma_x = sigma_z  # 轴向
    sigma_y = 0.0      # 环向
    tau_xy = 0.0       # 初始剪应力
    
    # 步骤4：应力变换到焊缝方向（焊缝与轴向成weld_angle_deg角）
    weld_stress_result = transform_stress_to_inclined_plane(
        sigma_x, sigma_y, tau_xy, weld_angle_deg
    )
    
    sigma_weld = weld_stress_result['result']['normal_stress_MPa']
    tau_weld = weld_stress_result['result']['shear_stress_MPa']
    
    return {
        'result': {
            'axial_stress_MPa': sigma_z,
            'weld_normal_stress_MPa': sigma_weld,
            'weld_shear_stress_MPa': tau_weld
        },
        'metadata': {
            'loading_condition': 'axial_load_only',
            'geometry': {
                'outer_diameter_mm': outer_diameter,
                'thickness_mm': thickness,
                'weld_angle_deg': weld_angle_deg
            },
            'load': {
                'axial_force_N': axial_force
            },
            'intermediate_results': {
                'cross_sectional_area_mm2': area,
                'stress_state': {
                    'sigma_axial_MPa': sigma_x,
                    'sigma_hoop_MPa': sigma_y,
                    'tau_initial_MPa': tau_xy
                }
            }
        }
    }


def analyze_weld_stress_under_internal_pressure(
    outer_diameter: float,
    thickness: float,
    internal_pressure: float,
    weld_angle_deg: float
) -> Dict:
    """
    分析内压载荷下焊缝处的应力状态（封闭端）
    
    Parameters:
    -----------
    outer_diameter : float
        外径 (mm)
    thickness : float
        壁厚 (mm)
    internal_pressure : float
        内压 (MPa)
    weld_angle_deg : float
        焊缝螺旋角（相对于轴向）(度)
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        焊缝处应力分析结果
    """
    # 步骤1：计算平均直径
    mean_diameter = outer_diameter - thickness
    
    # 步骤2：计算环向应力
    hoop_stress_result = calculate_hoop_stress(internal_pressure, mean_diameter, thickness)
    sigma_theta = hoop_stress_result['result']
    
    # 步骤3：计算轴向应力（封闭端）
    longitudinal_stress_result = calculate_longitudinal_stress(internal_pressure, mean_diameter, thickness)
    sigma_z = longitudinal_stress_result['result']
    
    # 步骤4：建立应力状态（双向应力状态）
    # 在圆柱坐标系中：σ_z = 轴向应力，σ_θ = 环向应力，τ = 0
    sigma_x = sigma_z      # 轴向
    sigma_y = sigma_theta  # 环向
    tau_xy = 0.0           # 初始剪应力
    
    # 步骤5：应力变换到焊缝方向
    weld_stress_result = transform_stress_to_inclined_plane(
        sigma_x, sigma_y, tau_xy, weld_angle_deg
    )
    
    sigma_weld = weld_stress_result['result']['normal_stress_MPa']
    tau_weld = weld_stress_result['result']['shear_stress_MPa']
    
    return {
        'result': {
            'axial_stress_MPa': sigma_z,
            'hoop_stress_MPa': sigma_theta,
            'weld_normal_stress_MPa': sigma_weld,
            'weld_shear_stress_MPa': tau_weld
        },
        'metadata': {
            'loading_condition': 'internal_pressure_closed_ends',
            'geometry': {
                'outer_diameter_mm': outer_diameter,
                'thickness_mm': thickness,
                'mean_diameter_mm': mean_diameter,
                'weld_angle_deg': weld_angle_deg
            },
            'load': {
                'internal_pressure_MPa': internal_pressure
            },
            'intermediate_results': {
                'stress_state': {
                    'sigma_axial_MPa': sigma_x,
                    'sigma_hoop_MPa': sigma_y,
                    'tau_initial_MPa': tau_xy
                }
            }
        }
    }


def compare_stress_states(
    stress_state_1: Dict,
    stress_state_2: Dict,
    labels: List[str]
) -> Dict:
    """
    比较两种应力状态
    
    Parameters:
    -----------
    stress_state_1 : dict
        第一种应力状态结果
    stress_state_2 : dict
        第二种应力状态结果
    labels : list of str
        两种状态的标签 [label1, label2]
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        比较结果
    """
    if len(labels) != 2:
        raise ValueError("必须提供两个标签")
    
    comparison = {
        'state_1': {
            'label': labels[0],
            'weld_normal_stress_MPa': stress_state_1['result']['weld_normal_stress_MPa'],
            'weld_shear_stress_MPa': stress_state_1['result']['weld_shear_stress_MPa']
        },
        'state_2': {
            'label': labels[1],
            'weld_normal_stress_MPa': stress_state_2['result']['weld_normal_stress_MPa'],
            'weld_shear_stress_MPa': stress_state_2['result']['weld_shear_stress_MPa']
        },
        'differences': {
            'normal_stress_diff_MPa': abs(
                stress_state_1['result']['weld_normal_stress_MPa'] - 
                stress_state_2['result']['weld_normal_stress_MPa']
            ),
            'shear_stress_diff_MPa': abs(
                stress_state_1['result']['weld_shear_stress_MPa'] - 
                stress_state_2['result']['weld_shear_stress_MPa']
            )
        }
    }
    
    return {
        'result': comparison,
        'metadata': {
            'comparison_type': 'stress_state_comparison',
            'states_compared': labels
        }
    }


# ==================== 第三层：可视化函数 ====================

def visualize_stress_transformation(
    sigma_x: float,
    sigma_y: float,
    tau_xy: float,
    theta_deg: float,
    title: str = "Stress Transformation"
) -> Dict:
    """
    可视化应力变换过程
    
    Parameters:
    -----------
    sigma_x : float
        x方向正应力 (MPa)
    sigma_y : float
        y方向正应力 (MPa)
    tau_xy : float
        xy平面剪应力 (MPa)
    theta_deg : float
        斜截面角度 (度)
    title : str
        图表标题
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：应力元素示意图
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Stress Element', fontsize=12, fontweight='bold')
    
    # 绘制应力元素
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(square)
    
    # 标注应力
    ax1.arrow(1, 0, 0.5, 0, head_width=0.15, head_length=0.1, fc='red', ec='red')
    ax1.text(1.8, 0, f'σx={sigma_x:.2f}', fontsize=10, ha='center')
    
    ax1.arrow(0, 1, 0, 0.5, head_width=0.15, head_length=0.1, fc='blue', ec='blue')
    ax1.text(0, 1.8, f'σy={sigma_y:.2f}', fontsize=10, ha='center')
    
    if abs(tau_xy) > 0.01:
        ax1.arrow(1, 1, 0.3, 0, head_width=0.1, head_length=0.08, fc='green', ec='green')
        ax1.text(1.5, 1.3, f'τxy={tau_xy:.2f}', fontsize=10, ha='center')
    
    # 绘制斜截面
    theta_rad = np.radians(theta_deg)
    x_line = np.array([-1.5, 1.5])
    y_line = x_line * np.tan(theta_rad)
    ax1.plot(x_line, y_line, 'k--', linewidth=2, label=f'Inclined plane: {theta_deg}°')
    ax1.legend(loc='lower right')
    
    # 右图：Mohr圆
    # 计算主应力和最大剪应力
    sigma_avg = (sigma_x + sigma_y) / 2
    R = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R
    
    # 绘制Mohr圆
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    sigma_circle = sigma_avg + R * np.cos(theta_circle)
    tau_circle = R * np.sin(theta_circle)
    
    ax2.plot(sigma_circle, tau_circle, 'b-', linewidth=2, label='Mohr Circle')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 标注原始应力状态
    ax2.plot(sigma_x, tau_xy, 'ro', markersize=8, label='Original state')
    ax2.plot(sigma_y, -tau_xy, 'ro', markersize=8)
    
    # 标注变换后的应力状态
    transform_result = transform_stress_to_inclined_plane(sigma_x, sigma_y, tau_xy, theta_deg)
    sigma_n = transform_result['result']['normal_stress_MPa']
    tau_n = transform_result['result']['shear_stress_MPa']
    ax2.plot(sigma_n, tau_n, 'gs', markersize=8, label=f'Transformed state ({theta_deg}°)')
    
    ax2.set_xlabel('Normal Stress σ (MPa)', fontsize=11)
    ax2.set_ylabel('Shear Stress τ (MPa)', fontsize=11)
    ax2.set_title('Mohr\'s Circle', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    filename = f"stress_transformation_{theta_deg}deg.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'title': title,
            'stress_state': {
                'sigma_x_MPa': sigma_x,
                'sigma_y_MPa': sigma_y,
                'tau_xy_MPa': tau_xy,
                'theta_deg': theta_deg
            },
            'transformed_stress': {
                'sigma_n_MPa': sigma_n,
                'tau_n_MPa': tau_n
            },
            'principal_stresses': {
                'sigma_1_MPa': sigma_1,
                'sigma_2_MPa': sigma_2,
                'max_shear_MPa': R
            }
        }
    }


def visualize_weld_stress_comparison(
    case1_result: Dict,
    case2_result: Dict,
    case1_label: str,
    case2_label: str
) -> Dict:
    """
    可视化两种工况下焊缝应力的比较
    
    Parameters:
    -----------
    case1_result : dict
        工况1的应力分析结果
    case2_result : dict
        工况2的应力分析结果
    case1_label : str
        工况1标签
    case2_label : str
        工况2标签
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 提取数据
    cases = [case1_result, case2_result]
    labels = [case1_label, case2_label]
    colors = ['steelblue', 'coral']
    
    # 左图：正应力比较
    ax1 = axes[0]
    normal_stresses = [case['result']['weld_normal_stress_MPa'] for case in cases]
    bars1 = ax1.bar(labels, normal_stresses, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Normal Stress (MPa)', fontsize=11)
    ax1.set_title('Normal Stress Perpendicular to Weld', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, normal_stresses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 右图：剪应力比较
    ax2 = axes[1]
    shear_stresses = [case['result']['weld_shear_stress_MPa'] for case in cases]
    bars2 = ax2.bar(labels, shear_stresses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Shear Stress (MPa)', fontsize=11)
    ax2.set_title('Shear Stress Along Weld Direction', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, shear_stresses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Weld Stress Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    filename = "weld_stress_comparison.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'comparison': {
                'case1': {
                    'label': case1_label,
                    'normal_stress_MPa': normal_stresses[0],
                    'shear_stress_MPa': shear_stresses[0]
                },
                'case2': {
                    'label': case2_label,
                    'normal_stress_MPa': normal_stresses[1],
                    'shear_stress_MPa': shear_stresses[1]
                }
            }
        }
    }


def visualize_stress_distribution_on_cylinder(
    outer_diameter: float,
    thickness: float,
    sigma_axial: float,
    sigma_hoop: float,
    weld_angle_deg: float
) -> Dict:
    """
    可视化圆柱体上的应力分布和焊缝位置
    
    Parameters:
    -----------
    outer_diameter : float
        外径 (mm)
    thickness : float
        壁厚 (mm)
    sigma_axial : float
        轴向应力 (MPa)
    sigma_hoop : float
        环向应力 (MPa)
    weld_angle_deg : float
        焊缝螺旋角 (度)
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    fig = plt.figure(figsize=(14, 10))
    
    # 创建3D子图
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # 子图1：3D圆柱体和螺旋焊缝
    mean_radius = (outer_diameter - thickness) / 2
    height = outer_diameter * 1.5
    
    # 生成圆柱体表面
    theta = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(0, height, 50)
    Theta, Z = np.meshgrid(theta, z)
    X = mean_radius * np.cos(Theta)
    Y = mean_radius * np.sin(Theta)
    
    ax1.plot_surface(X, Y, Z, alpha=0.3, color='lightblue')
    
    # 绘制螺旋焊缝
    t_weld = np.linspace(0, 4 * np.pi, 200)
    pitch = height / (4 * np.pi) * np.tan(np.radians(weld_angle_deg))
    x_weld = mean_radius * np.cos(t_weld)
    y_weld = mean_radius * np.sin(t_weld)
    z_weld = pitch * t_weld
    
    # 只显示在圆柱体高度范围内的焊缝
    mask = z_weld <= height
    ax1.plot(x_weld[mask], y_weld[mask], z_weld[mask], 'r-', linewidth=3, label='Weld seam')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Cylinder with Spiral Weld', fontweight='bold')
    ax1.legend()
    
    # 子图2：轴向应力分布
    ax2.barh(['Axial Stress'], [sigma_axial], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Stress (MPa)')
    ax2.set_title('Axial Stress Distribution', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(sigma_axial, 0, f'  {sigma_axial:.2f} MPa', va='center', fontsize=10)
    
    # 子图3：环向应力分布
    ax3.barh(['Hoop Stress'], [sigma_hoop], color='coral', alpha=0.7)
    ax3.set_xlabel('Stress (MPa)')
    ax3.set_title('Hoop Stress Distribution', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.text(sigma_hoop, 0, f'  {sigma_hoop:.2f} MPa', va='center', fontsize=10)
    
    # 子图4：焊缝处应力变换示意图
    theta_rad = np.radians(weld_angle_deg)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    
    # 绘制坐标轴
    ax4.arrow(0, 0, 1.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax4.text(1.3, 0, 'Axial', fontsize=10, ha='center')
    ax4.arrow(0, 0, 0, 1.2, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax4.text(0, 1.3, 'Hoop', fontsize=10, ha='center')
    
    # 绘制焊缝方向
    x_weld_dir = 1.0 * np.cos(theta_rad)
    y_weld_dir = 1.0 * np.sin(theta_rad)
    ax4.arrow(0, 0, x_weld_dir, y_weld_dir, head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2)
    ax4.text(x_weld_dir * 1.2, y_weld_dir * 1.2, f'Weld\n{weld_angle_deg}°', 
            fontsize=10, ha='center', color='red', fontweight='bold')
    
    ax4.set_title('Weld Orientation', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    plt.suptitle(f'Stress Distribution on Cylinder (D={outer_diameter}mm, t={thickness}mm)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    filename = "cylinder_stress_distribution.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'geometry': {
                'outer_diameter_mm': outer_diameter,
                'thickness_mm': thickness,
                'mean_radius_mm': mean_radius,
                'weld_angle_deg': weld_angle_deg
            },
            'stresses': {
                'sigma_axial_MPa': sigma_axial,
                'sigma_hoop_MPa': sigma_hoop
            }
        }
    }


# ==================== 主函数：演示场景 ====================

def main():
    """
    主函数：演示三个应用场景
    """
    
    # 问题参数
    OUTER_DIAMETER = 300.0  # mm
    THICKNESS = 8.0         # mm
    WELD_ANGLE = 20.0       # degrees
    AXIAL_FORCE = 250000.0  # N (250 kN)
    INTERNAL_PRESSURE = 5.0 # MPa
    
    print("=" * 80)
    print("场景1：原始问题完整求解 - 螺旋焊接钢管应力分析")
    print("=" * 80)
    print("问题描述：分析外径300mm、壁厚8mm、焊缝角度20°的螺旋焊接钢管")
    print("在两种工况下焊缝处的应力状态：")
    print("  工况1：仅受轴向载荷 Fp = 250 kN")
    print("  工况2：仅受内压 p = 5.0 MPa（两端封闭）")
    print("-" * 80)
    
    # ========== 工况1：轴向载荷 ==========
    print("\n步骤1.1：计算横截面积（工况1）")
    print("调用函数：calculate_cross_sectional_area()")
    area_result_case1 = calculate_cross_sectional_area(OUTER_DIAMETER, THICKNESS)
    area_case1 = area_result_case1['result']
    print(f"FUNCTION_CALL: calculate_cross_sectional_area | "
          f"PARAMS: {{outer_diameter: {OUTER_DIAMETER}, thickness: {THICKNESS}}} | "
          f"RESULT: {area_result_case1}")
    
    print("\n步骤1.2：计算轴向正应力（工况1）")
    print("调用函数：calculate_axial_stress()")
    axial_stress_result_case1 = calculate_axial_stress(AXIAL_FORCE, area_case1)
    sigma_z_case1 = axial_stress_result_case1['result']
    print(f"FUNCTION_CALL: calculate_axial_stress | "
          f"PARAMS: {{axial_force: {AXIAL_FORCE}, area: {area_case1:.2f}}} | "
          f"RESULT: {axial_stress_result_case1}")
    
    print("\n步骤1.3：建立应力状态（工况1）")
    # 在圆柱坐标系中：σ_z = 轴向应力，σ_θ = 0（无内压），τ = 0
    sigma_x_case1 = sigma_z_case1  # 轴向
    sigma_y_case1 = 0.0            # 环向
    tau_xy_case1 = 0.0             # 初始剪应力
    print(f"应力状态：σ_x = {sigma_x_case1:.2f} MPa, σ_y = {sigma_y_case1:.2f} MPa, τ_xy = {tau_xy_case1:.2f} MPa")
    
    print("\n步骤1.4：应力变换到焊缝方向（工况1）")
    print("调用函数：transform_stress_to_inclined_plane()")
    weld_stress_result_case1 = transform_stress_to_inclined_plane(
        sigma_x_case1, sigma_y_case1, tau_xy_case1, WELD_ANGLE
    )
    sigma_weld_case1 = weld_stress_result_case1['result']['normal_stress_MPa']
    tau_weld_case1 = weld_stress_result_case1['result']['shear_stress_MPa']
    print(f"FUNCTION_CALL: transform_stress_to_inclined_plane | "
          f"PARAMS: {{sigma_x: {sigma_x_case1:.2f}, sigma_y: {sigma_y_case1:.2f}, "
          f"tau_xy: {tau_xy_case1:.2f}, theta_deg: {WELD_ANGLE}}} | "
          f"RESULT: {weld_stress_result_case1}")
    
    # 组装工况1结果（保持与原函数接口兼容）
    case1_result = {
        'result': {
            'axial_stress_MPa': sigma_z_case1,
            'weld_normal_stress_MPa': sigma_weld_case1,
            'weld_shear_stress_MPa': tau_weld_case1
        },
        'metadata': {
            'loading_condition': 'axial_load_only',
            'geometry': {
                'outer_diameter_mm': OUTER_DIAMETER,
                'thickness_mm': THICKNESS,
                'weld_angle_deg': WELD_ANGLE
            },
            'load': {
                'axial_force_N': AXIAL_FORCE
            }
        }
    }
    
    # ========== 工况2：内压 ==========
    print("\n步骤2.1：计算平均直径（工况2）")
    mean_diameter_case2 = OUTER_DIAMETER - THICKNESS
    print(f"平均直径：{mean_diameter_case2:.2f} mm")
    
    print("\n步骤2.2：计算环向应力（工况2）")
    print("调用函数：calculate_hoop_stress()")
    hoop_stress_result_case2 = calculate_hoop_stress(INTERNAL_PRESSURE, mean_diameter_case2, THICKNESS)
    sigma_theta_case2 = hoop_stress_result_case2['result']
    print(f"FUNCTION_CALL: calculate_hoop_stress | "
          f"PARAMS: {{internal_pressure: {INTERNAL_PRESSURE}, mean_diameter: {mean_diameter_case2:.2f}, thickness: {THICKNESS}}} | "
          f"RESULT: {hoop_stress_result_case2}")
    
    print("\n步骤2.3：计算轴向应力（工况2，封闭端）")
    print("调用函数：calculate_longitudinal_stress()")
    longitudinal_stress_result_case2 = calculate_longitudinal_stress(INTERNAL_PRESSURE, mean_diameter_case2, THICKNESS)
    sigma_z_case2 = longitudinal_stress_result_case2['result']
    print(f"FUNCTION_CALL: calculate_longitudinal_stress | "
          f"PARAMS: {{internal_pressure: {INTERNAL_PRESSURE}, mean_diameter: {mean_diameter_case2:.2f}, thickness: {THICKNESS}}} | "
          f"RESULT: {longitudinal_stress_result_case2}")
    
    print("\n步骤2.4：建立应力状态（工况2）")
    # 在圆柱坐标系中：σ_z = 轴向应力，σ_θ = 环向应力，τ = 0
    sigma_x_case2 = sigma_z_case2      # 轴向
    sigma_y_case2 = sigma_theta_case2  # 环向
    tau_xy_case2 = 0.0                 # 初始剪应力
    print(f"应力状态：σ_x = {sigma_x_case2:.2f} MPa, σ_y = {sigma_y_case2:.2f} MPa, τ_xy = {tau_xy_case2:.2f} MPa")
    
    print("\n步骤2.5：应力变换到焊缝方向（工况2）")
    print("调用函数：transform_stress_to_inclined_plane()")
    weld_stress_result_case2 = transform_stress_to_inclined_plane(
        sigma_x_case2, sigma_y_case2, tau_xy_case2, WELD_ANGLE
    )
    sigma_weld_case2 = weld_stress_result_case2['result']['normal_stress_MPa']
    tau_weld_case2 = weld_stress_result_case2['result']['shear_stress_MPa']
    print(f"FUNCTION_CALL: transform_stress_to_inclined_plane | "
          f"PARAMS: {{sigma_x: {sigma_x_case2:.2f}, sigma_y: {sigma_y_case2:.2f}, "
          f"tau_xy: {tau_xy_case2:.2f}, theta_deg: {WELD_ANGLE}}} | "
          f"RESULT: {weld_stress_result_case2}")
    
    # 组装工况2结果（保持与原函数接口兼容）
    case2_result = {
        'result': {
            'axial_stress_MPa': sigma_z_case2,
            'hoop_stress_MPa': sigma_theta_case2,
            'weld_normal_stress_MPa': sigma_weld_case2,
            'weld_shear_stress_MPa': tau_weld_case2
        },
        'metadata': {
            'loading_condition': 'internal_pressure_closed_ends',
            'geometry': {
                'outer_diameter_mm': OUTER_DIAMETER,
                'thickness_mm': THICKNESS,
                'mean_diameter_mm': mean_diameter_case2,
                'weld_angle_deg': WELD_ANGLE
            },
            'load': {
                'internal_pressure_MPa': INTERNAL_PRESSURE
            }
        }
    }
    
    # 保存结果
    print("\n步骤3：保存计算结果")
    print("调用函数：save_calculation_results()")
    results_summary = {
        'problem': 'Spiral welded steel pipe stress analysis',
        'geometry': {
            'outer_diameter_mm': OUTER_DIAMETER,
            'thickness_mm': THICKNESS,
            'weld_angle_deg': WELD_ANGLE
        },
        'case1_axial_load': case1_result,
        'case2_internal_pressure': case2_result
    }
    save_result = save_calculation_results(results_summary, 'scenario1_results.json')
    print(f"FUNCTION_CALL: save_calculation_results | "
          f"PARAMS: {{filename: 'scenario1_results.json'}} | "
          f"RESULT: {save_result}")
    
    # 可视化比较
    print("\n步骤4：可视化焊缝应力比较")
    print("调用函数：visualize_weld_stress_comparison()")
    comparison_plot = visualize_weld_stress_comparison(
        case1_result=case1_result,
        case2_result=case2_result,
        case1_label="Axial Load (250 kN)",
        case2_label="Internal Pressure (5.0 MPa)"
    )
    print(f"FUNCTION_CALL: visualize_weld_stress_comparison | "
          f"PARAMS: {{case1_label: 'Axial Load', case2_label: 'Internal Pressure'}} | "
          f"RESULT: {comparison_plot}")
    
    # 输出最终答案
    print("\n" + "=" * 80)
    print("工况1（轴向载荷）焊缝应力：")
    print(f"  - 垂直于焊缝方向的正应力：{case1_result['result']['weld_normal_stress_MPa']:.2f} MPa")
    print(f"  - 沿焊缝方向的剪应力：{case1_result['result']['weld_shear_stress_MPa']:.2f} MPa")
    print("\n工况2（内压）焊缝应力：")
    print(f"  - 垂直于焊缝方向的正应力：{case2_result['result']['weld_normal_stress_MPa']:.2f} MPa")
    print(f"  - 沿焊缝方向的剪应力：{case2_result['result']['weld_shear_stress_MPa']:.2f} MPa")
    
    answer_case1 = (f"Normal stress: {case1_result['result']['weld_normal_stress_MPa']:.2f} MPa, "
                   f"Shear stress: {case1_result['result']['weld_shear_stress_MPa']:.2f} MPa")
    answer_case2 = (f"Normal stress: {case2_result['result']['weld_normal_stress_MPa']:.2f} MPa, "
                   f"Shear stress: {case2_result['result']['weld_shear_stress_MPa']:.2f} MPa")
    
    print(f"\nFINAL_ANSWER: Case 1 - {answer_case1}; Case 2 - {answer_case2}")
    
    
    # ========================================================================
    # 场景2：不同焊缝角度的应力分析
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：参数敏感性分析 - 焊缝角度对应力的影响")
    print("=" * 80)
    print("问题描述：分析相同几何尺寸和载荷条件下，不同焊缝角度（10°, 20°, 30°, 45°）")
    print("对焊缝处应力状态的影响（工况：轴向载荷 250 kN）")
    print("-" * 80)
    
    angles = [10.0, 20.0, 30.0, 45.0]
    angle_results = []
    
    for angle in angles:
        print(f"\n步骤：分析焊缝角度 {angle}° 的应力状态")
        print(f"调用函数：analyze_weld_stress_under_axial_load()")
        result = analyze_weld_stress_under_axial_load(
            outer_diameter=OUTER_DIAMETER,
            thickness=THICKNESS,
            axial_force=AXIAL_FORCE,
            weld_angle_deg=angle
        )
        angle_results.append(result)
        print(f"FUNCTION_CALL: analyze_weld_stress_under_axial_load | "
              f"PARAMS: {{weld_angle_deg: {angle}}} | "
              f"RESULT: {{normal_stress: {result['result']['weld_normal_stress_MPa']:.2f} MPa, "
              f"shear_stress: {result['result']['weld_shear_stress_MPa']:.2f} MPa}}")
    
    # 可视化角度影响
    print("\n步骤：可视化焊缝角度对应力的影响")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    normal_stresses = [r['result']['weld_normal_stress_MPa'] for r in angle_results]
    shear_stresses = [r['result']['weld_shear_stress_MPa'] for r in angle_results]
    
    ax1.plot(angles, normal_stresses, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Weld Angle (degrees)', fontsize=11)
    ax1.set_ylabel('Normal Stress (MPa)', fontsize=11)
    ax1.set_title('Normal Stress vs Weld Angle', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(angles, shear_stresses, 's-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Weld Angle (degrees)', fontsize=11)
    ax2.set_ylabel('Shear Stress (MPa)', fontsize=11)
    ax2.set_title('Shear Stress vs Weld Angle', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Weld Angle on Stress (Axial Load: 250 kN)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(IMAGES_DIR, "weld_angle_sensitivity.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    # 保存场景2结果
    scenario2_results = {
        'problem': 'Weld angle sensitivity analysis',
        'angles_analyzed': angles,
        'results': [
            {
                'angle_deg': angle,
                'normal_stress_MPa': normal_stresses[i],
                'shear_stress_MPa': shear_stresses[i]
            }
            for i, angle in enumerate(angles)
        ]
    }
    save_result2 = save_calculation_results(scenario2_results, 'scenario2_results.json')
    print(f"\nFUNCTION_CALL: save_calculation_results | "
          f"PARAMS: {{filename: 'scenario2_results.json'}} | "
          f"RESULT: {save_result2}")
    
    print(f"\nFINAL_ANSWER: Weld angle significantly affects stress distribution. "
          f"At 45°, normal stress = {normal_stresses[-1]:.2f} MPa, "
          f"shear stress = {shear_stresses[-1]:.2f} MPa")
    
    
    # ========================================================================
    # 场景3：组合载荷分析
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：组合载荷分析 - 同时受轴向载荷和内压")
    print("=" * 80)
    print("问题描述：分析钢管同时受到轴向载荷（250 kN）和内压（5.0 MPa）时")
    print("焊缝处的应力状态，并与单独载荷情况进行比较")
    print("-" * 80)
    
    # 步骤1：计算组合载荷下的应力状态
    print("\n步骤1：计算组合载荷下的基本应力")
    
    # 计算横截面积
    area_result = calculate_cross_sectional_area(OUTER_DIAMETER, THICKNESS)
    area = area_result['result']
    print(f"FUNCTION_CALL: calculate_cross_sectional_area | "
          f"PARAMS: {{outer_diameter: {OUTER_DIAMETER}, thickness: {THICKNESS}}} | "
          f"RESULT: {{area: {area:.2f} mm²}}")
    
    # 轴向载荷产生的轴向应力
    axial_stress_from_load = calculate_axial_stress(AXIAL_FORCE, area)
    sigma_z_load = axial_stress_from_load['result']
    print(f"FUNCTION_CALL: calculate_axial_stress | "
          f"PARAMS: {{axial_force: {AXIAL_FORCE}, area: {area:.2f}}} | "
          f"RESULT: {{axial_stress: {sigma_z_load:.2f} MPa}}")
    
    # 内压产生的应力
    mean_diameter = OUTER_DIAMETER - THICKNESS
    hoop_stress = calculate_hoop_stress(INTERNAL_PRESSURE, mean_diameter, THICKNESS)
    sigma_theta = hoop_stress['result']
    print(f"FUNCTION_CALL: calculate_hoop_stress | "
          f"PARAMS: {{internal_pressure: {INTERNAL_PRESSURE}, mean_diameter: {mean_diameter}, thickness: {THICKNESS}}} | "
          f"RESULT: {{hoop_stress: {sigma_theta:.2f} MPa}}")
    
    longitudinal_stress = calculate_longitudinal_stress(INTERNAL_PRESSURE, mean_diameter, THICKNESS)
    sigma_z_pressure = longitudinal_stress['result']
    print(f"FUNCTION_CALL: calculate_longitudinal_stress | "
          f"PARAMS: {{internal_pressure: {INTERNAL_PRESSURE}, mean_diameter: {mean_diameter}, thickness: {THICKNESS}}} | "
          f"RESULT: {{longitudinal_stress: {sigma_z_pressure:.2f} MPa}}")
    
    # 步骤2：叠加原理计算总应力
    print("\n步骤2：应用叠加原理计算组合载荷下的总应力")
    sigma_z_total = sigma_z_load + sigma_z_pressure
    sigma_theta_total = sigma_theta
    print(f"总轴向应力：{sigma_z_total:.2f} MPa = {sigma_z_load:.2f} (载荷) + {sigma_z_pressure:.2f} (内压)")
    print(f"总环向应力：{sigma_theta_total:.2f} MPa")
    
    # 步骤3：应力变换到焊缝方向
    print("\n步骤3：将应力变换到焊缝方向")
    print("调用函数：transform_stress_to_inclined_plane()")
    combined_weld_stress = transform_stress_to_inclined_plane(
        sigma_x=sigma_z_total,
        sigma_y=sigma_theta_total,
        tau_xy=0.0,
        theta_deg=WELD_ANGLE
    )
    print(f"FUNCTION_CALL: transform_stress_to_inclined_plane | "
          f"PARAMS: {{sigma_x: {sigma_z_total:.2f}, sigma_y: {sigma_theta_total:.2f}, "
          f"tau_xy: 0.0, theta_deg: {WELD_ANGLE}}} | "
          f"RESULT: {combined_weld_stress['result']}")
    
    # 步骤4：可视化应力变换
    print("\n步骤4：可视化组合载荷下的应力变换")
    print("调用函数：visualize_stress_transformation()")
    transform_plot = visualize_stress_transformation(
        sigma_x=sigma_z_total,
        sigma_y=sigma_theta_total,
        tau_xy=0.0,
        theta_deg=WELD_ANGLE,
        title="Combined Loading: Axial Load + Internal Pressure"
    )
    print(f"FUNCTION_CALL: visualize_stress_transformation | "
          f"PARAMS: {{sigma_x: {sigma_z_total:.2f}, sigma_y: {sigma_theta_total:.2f}, theta_deg: {WELD_ANGLE}}} | "
          f"RESULT: {transform_plot}")
    
    # 步骤5：可视化圆柱体应力分布
    print("\n步骤5：可视化圆柱体上的应力分布")
    print("调用函数：visualize_stress_distribution_on_cylinder()")
    cylinder_plot = visualize_stress_distribution_on_cylinder(
        outer_diameter=OUTER_DIAMETER,
        thickness=THICKNESS,
        sigma_axial=sigma_z_total,
        sigma_hoop=sigma_theta_total,
        weld_angle_deg=WELD_ANGLE
    )
    print(f"FUNCTION_CALL: visualize_stress_distribution_on_cylinder | "
          f"PARAMS: {{outer_diameter: {OUTER_DIAMETER}, thickness: {THICKNESS}, "
          f"sigma_axial: {sigma_z_total:.2f}, sigma_hoop: {sigma_theta_total:.2f}}} | "
          f"RESULT: {cylinder_plot}")
    
    # 步骤6：比较三种情况
    print("\n步骤6：比较三种载荷情况下的焊缝应力")
    comparison_data = {
        'case1_axial_only': {
            'normal_stress_MPa': case1_result['result']['weld_normal_stress_MPa'],
            'shear_stress_MPa': case1_result['result']['weld_shear_stress_MPa']
        },
        'case2_pressure_only': {
            'normal_stress_MPa': case2_result['result']['weld_normal_stress_MPa'],
            'shear_stress_MPa': case2_result['result']['weld_shear_stress_MPa']
        },
        'case3_combined': {
            'normal_stress_MPa': combined_weld_stress['result']['normal_stress_MPa'],
            'shear_stress_MPa': combined_weld_stress['result']['shear_stress_MPa']
        }
    }
    
    # 绘制三种情况的比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cases = ['Axial\nOnly', 'Pressure\nOnly', 'Combined']
    normal_values = [
        comparison_data['case1_axial_only']['normal_stress_MPa'],
        comparison_data['case2_pressure_only']['normal_stress_MPa'],
        comparison_data['case3_combined']['normal_stress_MPa']
    ]
    shear_values = [
        comparison_data['case1_axial_only']['shear_stress_MPa'],
        comparison_data['case2_pressure_only']['shear_stress_MPa'],
        comparison_data['case3_combined']['shear_stress_MPa']
    ]
    
    colors = ['steelblue', 'coral', 'green']
    
    bars1 = ax1.bar(cases, normal_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Normal Stress (MPa)', fontsize=11)
    ax1.set_title('Normal Stress Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars1, normal_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    bars2 = ax2.bar(cases, shear_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Shear Stress (MPa)', fontsize=11)
    ax2.set_title('Shear Stress Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars2, shear_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Weld Stress Comparison: Three Loading Conditions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(IMAGES_DIR, "three_cases_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    # 保存场景3结果
    scenario3_results = {
        'problem': 'Combined loading analysis',
        'loading': {
            'axial_force_N': AXIAL_FORCE,
            'internal_pressure_MPa': INTERNAL_PRESSURE
        },
        'stress_components': {
            'axial_from_load_MPa': sigma_z_load,
            'axial_from_pressure_MPa': sigma_z_pressure,
            'total_axial_MPa': sigma_z_total,
            'hoop_MPa': sigma_theta_total
        },
        'weld_stresses': combined_weld_stress['result'],
        'comparison': comparison_data
    }
    save_result3 = save_calculation_results(scenario3_results, 'scenario3_results.json')
    print(f"\nFUNCTION_CALL: save_calculation_results | "
          f"PARAMS: {{filename: 'scenario3_results.json'}} | "
          f"RESULT: {save_result3}")
    
    print("\n" + "=" * 80)
    print("场景3总结：")
    print(f"组合载荷下焊缝应力：")
    print(f"  - 垂直于焊缝方向的正应力：{combined_weld_stress['result']['normal_stress_MPa']:.2f} MPa")
    print(f"  - 沿焊缝方向的剪应力：{combined_weld_stress['result']['shear_stress_MPa']:.2f} MPa")
    print(f"\n与单独载荷相比：")
    print(f"  - 正应力增加：{combined_weld_stress['result']['normal_stress_MPa'] - case1_result['result']['weld_normal_stress_MPa']:.2f} MPa (相对于仅轴向载荷)")
    print(f"  - 剪应力变化：{combined_weld_stress['result']['shear_stress_MPa'] - case1_result['result']['weld_shear_stress_MPa']:.2f} MPa (相对于仅轴向载荷)")
    
    answer_combined = (f"Normal stress: {combined_weld_stress['result']['normal_stress_MPa']:.2f} MPa, "
                      f"Shear stress: {combined_weld_stress['result']['shear_stress_MPa']:.2f} MPa")
    
    print(f"\nFINAL_ANSWER: Combined loading - {answer_combined}")


if __name__ == "__main__":
    main()