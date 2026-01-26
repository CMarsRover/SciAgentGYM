# Filename: solid_mechanics_toolkit.py

"""
固体力学计算工具包 - 圆轴扭转与拉伸组合变形分析
Solid Mechanics Toolkit - Torsion and Tension Analysis of Circular Shafts

本工具包用于分析实心圆轴在扭转力偶和轴向拉力作用下的变形行为。
基于线弹性小变形理论，计算外力矩、拉力以及表面微元线段的旋转角度。

核心理论：
1. 扭转理论：τ = Tr/Ip, γ = τ/G, φ = Ml/(GIp)
2. 拉伸理论：σ = F/A, ε = σ/E
3. 应变组合：剪应变和正应变的耦合效应
4. 泊松效应：横向应变 = -μ × 轴向应变

主要功能：
- 计算扭转力矩
- 计算轴向拉力
- 分析应变状态
- 可视化变形过程
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json
from pathlib import Path

# 配置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
Path("./mid_result/mechanics").mkdir(parents=True, exist_ok=True)
Path("./tool_images").mkdir(parents=True, exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础力学计算
# ============================================================================

def calculate_polar_moment(diameter: float) -> Dict:
    """
    计算圆形截面的极惯性矩
    
    理论基础：
    对于实心圆截面，极惯性矩 Ip = πD⁴/32
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        极惯性矩 (m⁴) 及相关信息
    """
    if diameter <= 0:
        raise ValueError(f"直径必须为正数，当前值: {diameter}")
    
    Ip = np.pi * diameter**4 / 32
    
    return {
        'result': Ip,
        'metadata': {
            'diameter': diameter,
            'formula': 'Ip = π*D⁴/32',
            'unit': 'm⁴'
        }
    }


def calculate_cross_section_area(diameter: float) -> Dict:
    """
    计算圆形截面面积
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        截面积 (m²)
    """
    if diameter <= 0:
        raise ValueError(f"直径必须为正数，当前值: {diameter}")
    
    A = np.pi * diameter**2 / 4
    
    return {
        'result': A,
        'metadata': {
            'diameter': diameter,
            'formula': 'A = π*D²/4',
            'unit': 'm²'
        }
    }


def calculate_shear_modulus(E: float, mu: float) -> Dict:
    """
    根据弹性模量和泊松比计算剪切模量
    
    理论基础：
    对于各向同性材料，G = E / (2(1+μ))
    
    Parameters:
    -----------
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比 (无量纲)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        剪切模量 (Pa)
    """
    if E <= 0:
        raise ValueError(f"弹性模量必须为正数，当前值: {E}")
    if not -1 <= mu <= 0.5:
        raise ValueError(f"泊松比必须在[-1, 0.5]范围内，当前值: {mu}")
    
    G = E / (2 * (1 + mu))
    
    return {
        'result': G,
        'metadata': {
            'E': E,
            'mu': mu,
            'formula': 'G = E / (2(1+μ))',
            'unit': 'Pa'
        }
    }


def calculate_torsion_angle_per_length(M: float, G: float, Ip: float) -> Dict:
    """
    计算单位长度扭转角
    
    理论基础：
    φ/l = M/(G*Ip)，其中φ是总扭转角，l是长度
    
    Parameters:
    -----------
    M : float
        扭矩 (N·m)
    G : float
        剪切模量 (Pa)
    Ip : float
        极惯性矩 (m⁴)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        单位长度扭转角 (rad/m)
    """
    if G <= 0 or Ip <= 0:
        raise ValueError("剪切模量和极惯性矩必须为正数")
    
    phi_per_l = M / (G * Ip)
    
    return {
        'result': phi_per_l,
        'metadata': {
            'M': M,
            'G': G,
            'Ip': Ip,
            'formula': 'φ/l = M/(G*Ip)',
            'unit': 'rad/m'
        }
    }


def calculate_shear_strain_at_surface(diameter: float, phi_per_l: float) -> Dict:
    """
    计算圆轴表面的剪应变
    
    理论基础：
    γ = r * (dφ/dx)，其中r是半径，dφ/dx是单位长度扭转角
    对于表面点，r = D/2
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    phi_per_l : float
        单位长度扭转角 (rad/m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        表面剪应变 (无量纲)
    """
    if diameter <= 0:
        raise ValueError(f"直径必须为正数，当前值: {diameter}")
    
    r = diameter / 2
    gamma = r * phi_per_l
    
    return {
        'result': gamma,
        'metadata': {
            'diameter': diameter,
            'radius': r,
            'phi_per_l': phi_per_l,
            'formula': 'γ = r * (dφ/dx)',
            'unit': 'dimensionless'
        }
    }


def calculate_axial_strain(F: float, A: float, E: float) -> Dict:
    """
    计算轴向正应变
    
    理论基础：
    ε = σ/E = F/(A*E)
    
    Parameters:
    -----------
    F : float
        轴向拉力 (N)
    A : float
        截面积 (m²)
    E : float
        弹性模量 (Pa)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        轴向应变 (无量纲)
    """
    if A <= 0 or E <= 0:
        raise ValueError("截面积和弹性模量必须为正数")
    
    epsilon = F / (A * E)
    
    return {
        'result': epsilon,
        'metadata': {
            'F': F,
            'A': A,
            'E': E,
            'formula': 'ε = F/(A*E)',
            'unit': 'dimensionless'
        }
    }


def calculate_lateral_strain(epsilon_axial: float, mu: float) -> Dict:
    """
    计算横向应变（泊松效应）
    
    理论基础：
    ε_lateral = -μ * ε_axial
    
    Parameters:
    -----------
    epsilon_axial : float
        轴向应变 (无量纲)
    mu : float
        泊松比 (无量纲)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        横向应变 (无量纲)
    """
    if not -1 <= mu <= 0.5:
        raise ValueError(f"泊松比必须在[-1, 0.5]范围内，当前值: {mu}")
    
    epsilon_lateral = -mu * epsilon_axial
    
    return {
        'result': epsilon_lateral,
        'metadata': {
            'epsilon_axial': epsilon_axial,
            'mu': mu,
            'formula': 'ε_lateral = -μ * ε_axial',
            'unit': 'dimensionless'
        }
    }


# ============================================================================
# 第二层：组合函数 - 复杂力学分析
# ============================================================================

def calculate_line_rotation_pure_torsion(diameter: float, length: float, 
                                        E: float, mu: float, 
                                        beta: float, M: float) -> Dict:
    """
    计算纯扭转状态下表面微元线段的旋转角度
    
    理论推导：
    1. 扭转产生剪应变：γ = r*φ/l = r*M/(G*Ip)
    2. 表面点处剪应变：γ = (D/2) * M / (G * π*D⁴/32) = 16*M / (π*G*D³)
    3. 微元线段AB的旋转角度变化：Δβ = γ
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    length : float
        圆轴长度 (m)
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比 (无量纲)
    beta : float
        初始角度 (rad)
    M : float
        扭矩 (N·m)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        旋转角度 (rad)，正值表示顺时针
    """
    # 步骤1：计算剪切模量
    G_result = calculate_shear_modulus(E, mu)
    G = G_result['result']
    
    # 步骤2：计算极惯性矩
    Ip_result = calculate_polar_moment(diameter)
    Ip = Ip_result['result']
    
    # 步骤3：计算单位长度扭转角
    phi_per_l_result = calculate_torsion_angle_per_length(M, G, Ip)
    phi_per_l = phi_per_l_result['result']
    
    # 步骤4：计算表面剪应变
    gamma_result = calculate_shear_strain_at_surface(diameter, phi_per_l)
    gamma = gamma_result['result']
    
    # 步骤5：微元线段旋转角度等于剪应变
    delta_beta = gamma
    
    return {
        'result': delta_beta,
        'metadata': {
            'diameter': diameter,
            'length': length,
            'E': E,
            'mu': mu,
            'G': G,
            'beta': beta,
            'M': M,
            'Ip': Ip,
            'phi_per_l': phi_per_l,
            'gamma': gamma,
            'formula': 'Δβ = γ = (D/2) * M / (G*Ip)',
            'unit': 'rad'
        }
    }


def calculate_line_rotation_combined_loading(diameter: float, length: float,
                                             E: float, mu: float,
                                             beta: float, M: float, F: float) -> Dict:
    """
    计算扭转+拉伸组合载荷下表面微元线段的旋转角度
    
    理论推导：
    1. 扭转产生剪应变：γ_torsion
    2. 拉伸产生轴向应变：ε_axial = F/(A*E)
    3. 拉伸产生横向应变（泊松效应）：ε_lateral = -μ*ε_axial
    4. 横向应变导致半径变化，进而影响剪应变：γ_additional = ε_lateral * γ_torsion / ε_axial (近似)
    5. 更精确的分析：横向收缩改变了剪应变的几何关系
       Δβ_total = γ_torsion * (1 + ε_lateral) ≈ γ_torsion * (1 - μ*ε_axial)
    6. 额外旋转角：Δβ_2 = Δβ_total - Δβ_1 = -γ_torsion * μ * ε_axial
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    length : float
        圆轴长度 (m)
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比 (无量纲)
    beta : float
        初始角度 (rad)
    M : float
        扭矩 (N·m)
    F : float
        轴向拉力 (N)
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        包含总旋转角和额外旋转角
    """
    # 步骤1：计算纯扭转的旋转角
    torsion_result = calculate_line_rotation_pure_torsion(diameter, length, E, mu, beta, M)
    delta_beta_1 = torsion_result['result']
    gamma = torsion_result['metadata']['gamma']
    
    # 步骤2：计算截面积
    A_result = calculate_cross_section_area(diameter)
    A = A_result['result']
    
    # 步骤3：计算轴向应变
    epsilon_axial_result = calculate_axial_strain(F, A, E)
    epsilon_axial = epsilon_axial_result['result']
    
    # 步骤4：计算横向应变
    epsilon_lateral_result = calculate_lateral_strain(epsilon_axial, mu)
    epsilon_lateral = epsilon_lateral_result['result']
    
    # 步骤5：计算额外旋转角（由于横向收缩效应）
    # 理论：横向收缩使半径减小，剪应变的几何关系改变
    # Δβ_2 = -γ * μ * ε_axial
    delta_beta_2 = -gamma * mu * epsilon_axial
    
    # 步骤6：计算总旋转角
    delta_beta_total = delta_beta_1 + delta_beta_2
    
    return {
        'result': {
            'delta_beta_total': delta_beta_total,
            'delta_beta_1': delta_beta_1,
            'delta_beta_2': delta_beta_2
        },
        'metadata': {
            'diameter': diameter,
            'length': length,
            'E': E,
            'mu': mu,
            'beta': beta,
            'M': M,
            'F': F,
            'A': A,
            'gamma': gamma,
            'epsilon_axial': epsilon_axial,
            'epsilon_lateral': epsilon_lateral,
            'formula': 'Δβ_2 = -γ * μ * ε_axial',
            'unit': 'rad'
        }
    }


def solve_torque_from_rotation(diameter: float, length: float,
                               E: float, mu: float,
                               delta_beta_1: float) -> Dict:
    """
    根据测量的旋转角度反算扭矩
    
    理论推导：
    从 Δβ₁ = γ = (D/2) * M / (G*Ip)
    得到 M = Δβ₁ * G * Ip / (D/2)
         = Δβ₁ * G * (π*D⁴/32) / (D/2)
         = Δβ₁ * π * G * D³ / 16
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    length : float
        圆轴长度 (m)
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比 (无量纲)
    delta_beta_1 : float
        测量的旋转角度 (rad)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        扭矩 (N·m)
    """
    # 步骤1：计算剪切模量
    G_result = calculate_shear_modulus(E, mu)
    G = G_result['result']
    
    # 步骤2：计算极惯性矩
    Ip_result = calculate_polar_moment(diameter)
    Ip = Ip_result['result']
    
    # 步骤3：根据公式反算扭矩
    # M = Δβ₁ * G * Ip / (D/2)
    M = delta_beta_1 * G * Ip / (diameter / 2)
    
    # 验证计算
    verification = calculate_line_rotation_pure_torsion(diameter, length, E, mu, 0, M)
    calculated_delta_beta = verification['result']
    
    return {
        'result': M,
        'metadata': {
            'diameter': diameter,
            'length': length,
            'E': E,
            'mu': mu,
            'G': G,
            'Ip': Ip,
            'delta_beta_1': delta_beta_1,
            'calculated_delta_beta_1': calculated_delta_beta,
            'error': abs(calculated_delta_beta - delta_beta_1),
            'formula': 'M = Δβ₁ * G * Ip / (D/2) = Δβ₁ * π * G * D³ / 16',
            'unit': 'N·m'
        }
    }


def solve_axial_force_from_additional_rotation(diameter: float, length: float,
                                               E: float, mu: float,
                                               M: float, delta_beta_2: float) -> Dict:
    """
    根据额外旋转角度反算轴向拉力
    
    理论推导：
    从 Δβ₂ = -γ * μ * ε_axial
    其中 γ = (D/2) * M / (G*Ip)
         ε_axial = F / (A*E)
    得到 Δβ₂ = -(D/2) * M / (G*Ip) * μ * F / (A*E)
    因此 F = -Δβ₂ * G * Ip * A * E / ((D/2) * M * μ)
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    length : float
        圆轴长度 (m)
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比 (无量纲)
    M : float
        扭矩 (N·m)
    delta_beta_2 : float
        额外旋转角度 (rad)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        轴向拉力 (N)
    """
    if abs(mu) < 1e-10:
        raise ValueError("泊松比不能为零，否则无法计算轴向力")
    
    # 步骤1：计算剪切模量
    G_result = calculate_shear_modulus(E, mu)
    G = G_result['result']
    
    # 步骤2：计算极惯性矩
    Ip_result = calculate_polar_moment(diameter)
    Ip = Ip_result['result']
    
    # 步骤3：计算截面积
    A_result = calculate_cross_section_area(diameter)
    A = A_result['result']
    
    # 步骤4：计算剪应变
    phi_per_l = M / (G * Ip)
    gamma = (diameter / 2) * phi_per_l
    
    # 步骤5：根据公式反算轴向力
    # Δβ₂ = -γ * μ * ε_axial = -γ * μ * F / (A*E)
    # F = -Δβ₂ * A * E / (γ * μ)
    F = -delta_beta_2 * A * E / (gamma * mu)
    
    # 验证计算
    verification = calculate_line_rotation_combined_loading(diameter, length, E, mu, 0, M, F)
    calculated_delta_beta_2 = verification['result']['delta_beta_2']
    
    return {
        'result': F,
        'metadata': {
            'diameter': diameter,
            'length': length,
            'E': E,
            'mu': mu,
            'M': M,
            'G': G,
            'Ip': Ip,
            'A': A,
            'gamma': gamma,
            'delta_beta_2': delta_beta_2,
            'calculated_delta_beta_2': calculated_delta_beta_2,
            'error': abs(calculated_delta_beta_2 - delta_beta_2),
            'formula': 'F = -Δβ₂ * A * E / (γ * μ)',
            'unit': 'N'
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_shaft_deformation(diameter: float, length: float,
                               beta: float, delta_beta_1: float,
                               delta_beta_2: float = 0,
                               save_path: str = None) -> Dict:
    """
    可视化圆轴变形和微元线段旋转
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    length : float
        圆轴长度 (m)
    beta : float
        初始角度 (rad)
    delta_beta_1 : float
        纯扭转旋转角 (rad)
    delta_beta_2 : float, optional
        额外旋转角 (rad)
    save_path : str, optional
        保存路径
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    fig, axes = plt.subplots(1, 2 if delta_beta_2 == 0 else 3, figsize=(15, 5))
    
    # 子图1：初始状态
    ax1 = axes[0] if delta_beta_2 == 0 else axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = diameter/2 * np.cos(theta)
    y_circle = diameter/2 * np.sin(theta)
    
    ax1.plot(x_circle, y_circle, 'b-', linewidth=2, label='Shaft cross-section')
    ax1.plot([0, diameter/2], [0, 0], 'k--', linewidth=1, label='Horizontal line AC')
    
    # 绘制初始微元线段AB
    line_length = diameter/2 * 0.3
    x_B = diameter/2 + line_length * np.cos(beta)
    y_B = line_length * np.sin(beta)
    ax1.plot([diameter/2, x_B], [0, y_B], 'r-', linewidth=2, label=f'Line AB (β={np.degrees(beta):.1f}°)')
    ax1.plot(diameter/2, 0, 'ro', markersize=8, label='Point A')
    ax1.plot(x_B, y_B, 'rs', markersize=8, label='Point B')
    
    ax1.set_xlim(-diameter, diameter)
    ax1.set_ylim(-diameter, diameter)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title('Initial State', fontsize=10)
    ax1.set_xlabel('x (m)', fontsize=9)
    ax1.set_ylabel('y (m)', fontsize=9)
    
    # 子图2：纯扭转后
    ax2 = axes[1] if delta_beta_2 == 0 else axes[1]
    ax2.plot(x_circle, y_circle, 'b-', linewidth=2, label='Shaft cross-section')
    ax2.plot([0, diameter/2], [0, 0], 'k--', linewidth=1, label='Horizontal line AC')
    
    # 初始位置（虚线）
    ax2.plot([diameter/2, x_B], [0, y_B], 'r--', linewidth=1, alpha=0.5, label='Initial AB')
    
    # 旋转后位置
    beta_new = beta - delta_beta_1  # 顺时针旋转为负
    x_B_new = diameter/2 + line_length * np.cos(beta_new)
    y_B_new = line_length * np.sin(beta_new)
    ax2.plot([diameter/2, x_B_new], [0, y_B_new], 'g-', linewidth=2, 
             label=f'After torsion (Δβ₁={np.degrees(delta_beta_1):.4f}°)')
    ax2.plot(diameter/2, 0, 'go', markersize=8)
    ax2.plot(x_B_new, y_B_new, 'gs', markersize=8)
    
    # 绘制旋转角度弧线
    arc_angles = np.linspace(beta_new, beta, 50)
    arc_r = diameter/2 * 0.15
    arc_x = diameter/2 + arc_r * np.cos(arc_angles)
    arc_y = arc_r * np.sin(arc_angles)
    ax2.plot(arc_x, arc_y, 'm-', linewidth=1.5, label=f'Rotation angle')
    
    ax2.set_xlim(-diameter, diameter)
    ax2.set_ylim(-diameter, diameter)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title('After Pure Torsion (M_e)', fontsize=10)
    ax2.set_xlabel('x (m)', fontsize=9)
    ax2.set_ylabel('y (m)', fontsize=9)
    
    # 子图3：扭转+拉伸后（如果有）
    if delta_beta_2 != 0:
        ax3 = axes[2]
        ax3.plot(x_circle, y_circle, 'b-', linewidth=2, label='Shaft cross-section')
        ax3.plot([0, diameter/2], [0, 0], 'k--', linewidth=1, label='Horizontal line AC')
        
        # 纯扭转位置（虚线）
        ax3.plot([diameter/2, x_B_new], [0, y_B_new], 'g--', linewidth=1, alpha=0.5, 
                label='After torsion only')
        
        # 扭转+拉伸后位置
        beta_final = beta - delta_beta_1 - delta_beta_2
        x_B_final = diameter/2 + line_length * np.cos(beta_final)
        y_B_final = line_length * np.sin(beta_final)
        ax3.plot([diameter/2, x_B_final], [0, y_B_final], 'purple', linewidth=2,
                label=f'After torsion+tension (Δβ₂={np.degrees(delta_beta_2):.4f}°)')
        ax3.plot(diameter/2, 0, 'o', color='purple', markersize=8)
        ax3.plot(x_B_final, y_B_final, 's', color='purple', markersize=8)
        
        # 绘制额外旋转角度弧线
        arc_angles2 = np.linspace(beta_final, beta_new, 50)
        arc_r2 = diameter/2 * 0.2
        arc_x2 = diameter/2 + arc_r2 * np.cos(arc_angles2)
        arc_y2 = arc_r2 * np.sin(arc_angles2)
        ax3.plot(arc_x2, arc_y2, 'orange', linewidth=1.5, label='Additional rotation')
        
        ax3.set_xlim(-diameter, diameter)
        ax3.set_ylim(-diameter, diameter)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.set_title('After Torsion + Tension (M_e + F)', fontsize=10)
        ax3.set_xlabel('x (m)', fontsize=9)
        ax3.set_ylabel('y (m)', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = "./tool_images/shaft_deformation.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'diameter': diameter,
            'length': length,
            'beta': beta,
            'delta_beta_1': delta_beta_1,
            'delta_beta_2': delta_beta_2,
            'file_type': 'png',
            'description': 'Shaft deformation visualization'
        }
    }


def visualize_stress_strain_state(diameter: float, M: float, F: float,
                                  E: float, mu: float,
                                  save_path: str = None) -> Dict:
    """
    可视化应力应变状态
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    M : float
        扭矩 (N·m)
    F : float
        轴向力 (N)
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比
    save_path : str, optional
        保存路径
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    # 计算应力应变
    G_result = calculate_shear_modulus(E, mu)
    G = G_result['result']
    
    Ip_result = calculate_polar_moment(diameter)
    Ip = Ip_result['result']
    
    A_result = calculate_cross_section_area(diameter)
    A = A_result['result']
    
    # 剪应力分布（沿半径）
    r_array = np.linspace(0, diameter/2, 100)
    tau_array = M * r_array / Ip
    
    # 正应力（均匀分布）
    sigma_axial = F / A if F != 0 else 0
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图1：剪应力分布
    ax1 = axes[0, 0]
    ax1.plot(r_array * 1000, tau_array / 1e6, 'b-', linewidth=2)
    ax1.fill_between(r_array * 1000, 0, tau_array / 1e6, alpha=0.3)
    ax1.set_xlabel('Radius r (mm)', fontsize=10)
    ax1.set_ylabel('Shear Stress τ (MPa)', fontsize=10)
    ax1.set_title('Shear Stress Distribution (Torsion)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    
    # 子图2：正应力分布
    ax2 = axes[0, 1]
    if F != 0:
        ax2.barh([0], [sigma_axial / 1e6], height=0.5, color='green', alpha=0.6)
        ax2.set_xlabel('Axial Stress σ (MPa)', fontsize=10)
        ax2.set_ylabel('Cross-section', fontsize=10)
        ax2.set_title('Axial Stress Distribution (Tension)', fontsize=11)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='k', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'No Axial Force', ha='center', va='center',
                fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Axial Stress Distribution', fontsize=11)
    
    # 子图3：剪应变分布
    ax3 = axes[1, 0]
    gamma_array = tau_array / G
    ax3.plot(r_array * 1000, gamma_array * 1000, 'r-', linewidth=2)
    ax3.fill_between(r_array * 1000, 0, gamma_array * 1000, alpha=0.3, color='red')
    ax3.set_xlabel('Radius r (mm)', fontsize=10)
    ax3.set_ylabel('Shear Strain γ (×10⁻³)', fontsize=10)
    ax3.set_title('Shear Strain Distribution', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    
    # 子图4：应变状态总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    gamma_max = tau_array[-1] / G
    epsilon_axial = sigma_axial / E if F != 0 else 0
    epsilon_lateral = -mu * epsilon_axial
    
    summary_text = f"""
    Stress-Strain State Summary
    {'='*40}
    
    Torsion:
      Max Shear Stress:  τ_max = {tau_array[-1]/1e6:.2f} MPa
      Max Shear Strain:  γ_max = {gamma_max*1000:.4f} ×10⁻³
    
    Tension:
      Axial Stress:      σ = {sigma_axial/1e6:.2f} MPa
      Axial Strain:      ε_axial = {epsilon_axial*1e6:.2f} με
      Lateral Strain:    ε_lateral = {epsilon_lateral*1e6:.2f} με
    
    Material Properties:
      Elastic Modulus:   E = {E/1e9:.1f} GPa
      Shear Modulus:     G = {G/1e9:.1f} GPa
      Poisson's Ratio:   μ = {mu:.3f}
    
    Geometry:
      Diameter:          D = {diameter*1000:.1f} mm
      Polar Moment:      I_p = {Ip*1e12:.2f} mm⁴
      Cross-section:     A = {A*1e6:.2f} mm²
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = "./tool_images/stress_strain_state.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'diameter': diameter,
            'M': M,
            'F': F,
            'E': E,
            'mu': mu,
            'G': G,
            'tau_max': tau_array[-1],
            'gamma_max': gamma_max,
            'sigma_axial': sigma_axial,
            'epsilon_axial': epsilon_axial,
            'file_type': 'png',
            'description': 'Stress and strain state visualization'
        }
    }


def plot_parametric_study(diameter: float, length: float,
                         E: float, mu: float,
                         M_range: List[float],
                         save_path: str = None) -> Dict:
    """
    参数化研究：扭矩对旋转角的影响
    
    Parameters:
    -----------
    diameter : float
        圆轴直径 (m)
    length : float
        圆轴长度 (m)
    E : float
        弹性模量 (Pa)
    mu : float
        泊松比
    M_range : List[float]
        扭矩范围 (N·m)
    save_path : str, optional
        保存路径
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    delta_beta_list = []
    
    for M in M_range:
        result = calculate_line_rotation_pure_torsion(diameter, length, E, mu, 0, M)
        delta_beta_list.append(result['result'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(M_range, np.degrees(delta_beta_list), 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Torque M (N·m)', fontsize=11)
    ax.set_ylabel('Rotation Angle Δβ (degrees)', fontsize=11)
    ax.set_title('Parametric Study: Torque vs Rotation Angle', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加线性拟合
    coeffs = np.polyfit(M_range, np.degrees(delta_beta_list), 1)
    fit_line = np.poly1d(coeffs)
    ax.plot(M_range, fit_line(M_range), 'r--', linewidth=1.5, 
            label=f'Linear fit: Δβ = {coeffs[0]:.4e}·M + {coeffs[1]:.4e}')
    
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = "./tool_images/parametric_study.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'diameter': diameter,
            'length': length,
            'E': E,
            'mu': mu,
            'M_range': M_range,
            'delta_beta_range': delta_beta_list,
            'linear_coefficients': coeffs.tolist(),
            'file_type': 'png',
            'description': 'Parametric study of torque effect'
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    主函数：演示三个应用场景
    """
    
    print("="*60)
    print("场景1：解决原始问题 - 圆轴扭转与拉伸组合变形分析")
    print("="*60)
    print("问题描述：")
    print("给定实心圆轴：直径D=0.05m，长度l=1.0m")
    print("材料参数：E=200GPa，μ=0.3")
    print("初始角度：β=30°")
    print("测量数据：纯扭转时Δβ₁=0.001rad，加拉力后额外旋转Δβ₂=0.0002rad")
    print("求解：1) 外力偶矩Me；2) 轴向拉力F")
    print("-"*60)
    
    # 问题参数
    D = 0.05  # m
    l = 1.0   # m
    E = 200e9  # Pa
    mu = 0.3
    beta = np.radians(30)  # rad
    delta_beta_1_measured = 0.001  # rad
    delta_beta_2_measured = 0.0002  # rad
    
    # ========== 步骤1：根据Δβ₁计算外力偶矩Me ==========
    print("\n步骤1.1：计算剪切模量")
    print("调用函数：calculate_shear_modulus()")
    G_result = calculate_shear_modulus(E, mu)
    G = G_result['result']
    print(f"FUNCTION_CALL: calculate_shear_modulus | PARAMS: {{E={E}, mu={mu}}} | RESULT: {G_result}")
    
    print("\n步骤1.2：计算极惯性矩")
    print("调用函数：calculate_polar_moment()")
    Ip_result = calculate_polar_moment(D)
    Ip = Ip_result['result']
    print(f"FUNCTION_CALL: calculate_polar_moment | PARAMS: {{diameter={D}}} | RESULT: {Ip_result}")
    
    print("\n步骤1.3：根据旋转角反算扭矩")
    # M = Δβ₁ * G * Ip / (D/2)
    Me = delta_beta_1_measured * G * Ip / (D / 2)
    print(f"根据公式 M = Δβ₁ * G * Ip / (D/2) 计算得到：Me = {Me:.4f} N·m")
    
    # ========== 步骤2：根据Δβ₂计算轴向拉力F ==========
    print("\n步骤2.1：计算截面积")
    print("调用函数：calculate_cross_section_area()")
    A_result = calculate_cross_section_area(D)
    A = A_result['result']
    print(f"FUNCTION_CALL: calculate_cross_section_area | PARAMS: {{diameter={D}}} | RESULT: {A_result}")
    
    print("\n步骤2.2：计算单位长度扭转角")
    print("调用函数：calculate_torsion_angle_per_length()")
    phi_per_l_result = calculate_torsion_angle_per_length(Me, G, Ip)
    phi_per_l = phi_per_l_result['result']
    print(f"FUNCTION_CALL: calculate_torsion_angle_per_length | PARAMS: {{M={Me}, G={G}, Ip={Ip}}} | RESULT: {phi_per_l_result}")
    
    print("\n步骤2.3：计算表面剪应变")
    print("调用函数：calculate_shear_strain_at_surface()")
    gamma_result = calculate_shear_strain_at_surface(D, phi_per_l)
    gamma = gamma_result['result']
    print(f"FUNCTION_CALL: calculate_shear_strain_at_surface | PARAMS: {{diameter={D}, phi_per_l={phi_per_l}}} | RESULT: {gamma_result}")
    
    print("\n步骤2.4：根据额外旋转角反算轴向拉力")
    # Δβ₂ = -γ * μ * ε_axial = -γ * μ * F / (A*E)
    # F = -Δβ₂ * A * E / (γ * μ)
    F = -delta_beta_2_measured * A * E / (gamma * mu)
    print(f"根据公式 F = -Δβ₂ * A * E / (γ * μ) 计算得到：F = {F:.2f} N")
    
    # ========== 步骤3：验证计算结果 ==========
    print("\n步骤3.1：计算纯扭转的旋转角Δβ₁")
    # 使用已经计算的值：delta_beta_1 = gamma
    delta_beta_1_calculated = gamma
    print(f"计算得到Δβ₁ = {delta_beta_1_calculated:.6f} rad")
    
    print("\n步骤3.2：计算轴向应变")
    print("调用函数：calculate_axial_strain()")
    epsilon_axial_result = calculate_axial_strain(F, A, E)
    epsilon_axial = epsilon_axial_result['result']
    print(f"FUNCTION_CALL: calculate_axial_strain | PARAMS: {{F={F}, A={A:.6f}, E={E}}} | RESULT: {epsilon_axial_result}")
    
    print("\n步骤3.3：计算横向应变（泊松效应）")
    print("调用函数：calculate_lateral_strain()")
    epsilon_lateral_result = calculate_lateral_strain(epsilon_axial, mu)
    epsilon_lateral = epsilon_lateral_result['result']
    print(f"FUNCTION_CALL: calculate_lateral_strain | PARAMS: {{epsilon_axial={epsilon_axial}, mu={mu}}} | RESULT: {epsilon_lateral_result}")
    
    print("\n步骤3.4：计算额外旋转角Δβ₂")
    # Δβ₂ = -γ * μ * ε_axial
    delta_beta_2_calculated = -gamma * mu * epsilon_axial
    print(f"根据公式 Δβ₂ = -γ * μ * ε_axial 计算得到：Δβ₂ = {delta_beta_2_calculated:.6f} rad")
    
    print("\n步骤3.5：验证结果")
    print(f"验证Δβ₁：测量值={delta_beta_1_measured:.6f} rad，计算值={delta_beta_1_calculated:.6f} rad")
    print(f"验证Δβ₂：测量值={delta_beta_2_measured:.6f} rad，计算值={delta_beta_2_calculated:.6f} rad")
    
    # 组装验证结果（保持与原函数接口兼容）
    verification = {
        'result': {
            'delta_beta_total': delta_beta_1_calculated + delta_beta_2_calculated,
            'delta_beta_1': delta_beta_1_calculated,
            'delta_beta_2': delta_beta_2_calculated
        },
        'metadata': {
            'gamma': gamma,
            'epsilon_axial': epsilon_axial,
            'epsilon_lateral': epsilon_lateral
        }
    }
    
    # 步骤4：可视化变形过程
    print("\n步骤4：可视化圆轴变形过程")
    print("调用函数：visualize_shaft_deformation()")
    vis1_result = visualize_shaft_deformation(D, l, beta, delta_beta_1_measured, delta_beta_2_measured)
    print(f"FUNCTION_CALL: visualize_shaft_deformation | PARAMS: {{D={D}, l={l}, beta={beta}, delta_beta_1={delta_beta_1_measured}, delta_beta_2={delta_beta_2_measured}}} | RESULT: {vis1_result}")
    
    # 步骤5：可视化应力应变状态
    print("\n步骤5：可视化应力应变状态")
    print("调用函数：visualize_stress_strain_state()")
    vis2_result = visualize_stress_strain_state(D, Me, F, E, mu)
    print(f"FUNCTION_CALL: visualize_stress_strain_state | PARAMS: {{D={D}, Me={Me}, F={F}, E={E}, mu={mu}}} | RESULT: {vis2_result}")
    
    # 保存中间结果
    result_data = {
        'problem_parameters': {
            'diameter': D,
            'length': l,
            'E': E,
            'mu': mu,
            'beta': beta,
            'delta_beta_1_measured': delta_beta_1_measured,
            'delta_beta_2_measured': delta_beta_2_measured
        },
        'solution': {
            'Me': Me,
            'F': F
        },
        'verification': verification['result']
    }
    
    result_file = "./mid_result/mechanics/scenario1_solution.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nFILE_GENERATED: json | PATH: {result_file}")
    
    print(f"\nFINAL_ANSWER: 外力偶矩 Me = {Me:.4f} N·m，轴向拉力 F = {F:.2f} N")
    
    
    print("\n" + "="*60)
    print("场景2：参数敏感性分析 - 不同扭矩下的旋转角度")
    print("="*60)
    print("问题描述：")
    print("分析在相同几何和材料参数下，不同扭矩对旋转角度的影响")
    print("扭矩范围：0 ~ 100 N·m")
    print("-"*60)
    
    # 步骤1：定义扭矩范围
    print("\n步骤1：定义扭矩范围")
    M_range = np.linspace(0, 100, 20)
    print(f"扭矩范围：{M_range[0]} ~ {M_range[-1]} N·m，共{len(M_range)}个点")
    
    # 步骤2：计算每个扭矩对应的旋转角
    print("\n步骤2：计算每个扭矩对应的旋转角")
    print("调用函数：calculate_line_rotation_pure_torsion()（循环调用）")
    delta_beta_array = []
    for M in M_range:
        result = calculate_line_rotation_pure_torsion(D, l, E, mu, beta, M)
        delta_beta_array.append(result['result'])
    print(f"计算完成，得到{len(delta_beta_array)}个旋转角数据点")
    
    # 步骤3：绘制参数化研究图
    print("\n步骤3：绘制参数化研究图")
    print("调用函数：plot_parametric_study()")
    param_result = plot_parametric_study(D, l, E, mu, M_range.tolist())
    print(f"FUNCTION_CALL: plot_parametric_study | PARAMS: {{D={D}, l={l}, E={E}, mu={mu}, M_range=[{M_range[0]}, ..., {M_range[-1]}]}} | RESULT: {param_result}")
    
    # 步骤4：线性拟合分析
    print("\n步骤4：线性拟合分析")
    coeffs = np.polyfit(M_range, np.degrees(delta_beta_array), 1)
    print(f"线性关系：Δβ (degrees) = {coeffs[0]:.6e} × M + {coeffs[1]:.6e}")
    print(f"斜率物理意义：每增加1 N·m扭矩，旋转角增加{coeffs[0]:.6e}度")
    
    print(f"\nFINAL_ANSWER: 旋转角与扭矩呈线性关系，斜率为{coeffs[0]:.6e} deg/(N·m)")
    
    
    print("\n" + "="*60)
    print("场景3：材料对比分析 - 不同泊松比的影响")
    print("="*60)
    print("问题描述：")
    print("对比三种材料（钢、铝、橡胶）在相同载荷下的变形行为")
    print("固定参数：D=0.05m, l=1.0m, Me=50 N·m, F=10000 N")
    print("-"*60)
    
    # 步骤1：定义三种材料参数
    print("\n步骤1：定义三种材料参数")
    materials = {
        'Steel': {'E': 200e9, 'mu': 0.30},
        'Aluminum': {'E': 70e9, 'mu': 0.33},
        'Rubber': {'E': 0.01e9, 'mu': 0.48}
    }
    
    Me_test = 50  # N·m
    F_test = 10000  # N
    
    for name, props in materials.items():
        print(f"{name}: E={props['E']/1e9:.1f} GPa, μ={props['mu']:.2f}")
    
    # 步骤2：计算每种材料的变形
    print("\n步骤2：计算每种材料的变形")
    print("调用函数：calculate_line_rotation_combined_loading()（循环调用）")
    
    results_comparison = {}
    for name, props in materials.items():
        result = calculate_line_rotation_combined_loading(
            D, l, props['E'], props['mu'], beta, Me_test, F_test
        )
        results_comparison[name] = result['result']
        print(f"\n{name}:")
        print(f"  Δβ₁ = {np.degrees(result['result']['delta_beta_1']):.6f}°")
        print(f"  Δβ₂ = {np.degrees(result['result']['delta_beta_2']):.6f}°")
        print(f"  Δβ_total = {np.degrees(result['result']['delta_beta_total']):.6f}°")
    
    # 步骤3：可视化对比
    print("\n步骤3：可视化材料对比")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：Δβ₁对比
    ax1 = axes[0]
    names = list(materials.keys())
    delta_beta_1_values = [np.degrees(results_comparison[name]['delta_beta_1']) for name in names]
    colors = ['steelblue', 'orange', 'green']
    ax1.bar(names, delta_beta_1_values, color=colors, alpha=0.7)
    ax1.set_ylabel('Δβ₁ (degrees)', fontsize=11)
    ax1.set_title('Pure Torsion Rotation Angle Comparison', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2：Δβ₂对比
    ax2 = axes[1]
    delta_beta_2_values = [np.degrees(results_comparison[name]['delta_beta_2']) for name in names]
    ax2.bar(names, delta_beta_2_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Δβ₂ (degrees)', fontsize=11)
    ax2.set_title('Additional Rotation Angle Comparison (with Tension)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_path = "./tool_images/material_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {comparison_path}")
    
    # 步骤4：分析泊松比的影响
    print("\n步骤4：分析泊松比的影响")
    print("泊松比越大，横向收缩越明显，导致Δβ₂的绝对值越大")
    for name in names:
        mu_val = materials[name]['mu']
        delta_beta_2 = results_comparison[name]['delta_beta_2']
        print(f"{name}: μ={mu_val:.2f}, Δβ₂={np.degrees(delta_beta_2):.6f}°")
    
    print(f"\nFINAL_ANSWER: 泊松比越大，轴向拉伸引起的额外旋转角越大（橡胶 > 铝 > 钢）")


if __name__ == "__main__":
    main()