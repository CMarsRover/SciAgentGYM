# Filename: materials_dislocation_toolkit.py

"""
Materials Science Toolkit for Dislocation and Grain Boundary Analysis
专注于位错理论、亚晶界结构分析和晶界能计算
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

# 创建结果保存目录
os.makedirs('./mid_result/materials', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def calculate_burgers_vector_bcc(lattice_constant: float) -> Dict:
    """
    计算BCC结构的Burgers矢量大小
    
    对于BCC结构，最常见的滑移系统是{110}<111>
    Burgers矢量 b = (a/2)<111> = (a*sqrt(3))/2
    
    Parameters:
    -----------
    lattice_constant : float
        晶格常数 (nm)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        Burgers矢量大小 (nm) 和相关信息
    """
    if lattice_constant <= 0:
        raise ValueError("晶格常数必须为正数")
    
    # BCC结构的Burgers矢量
    b = (lattice_constant * np.sqrt(3)) / 2
    
    return {
        'result': b,
        'metadata': {
            'structure': 'bcc',
            'slip_system': '{110}<111>',
            'lattice_constant_nm': lattice_constant,
            'formula': 'b = (a*sqrt(3))/2'
        }
    }


def calculate_dislocation_spacing(burgers_vector: float, misorientation_rad: float) -> Dict:
    """
    根据Read-Shockley模型计算位错间距
    
    对于小角度晶界（亚晶界），位错间距 D = b / θ
    其中 b 是Burgers矢量，θ 是取向差角度（弧度）
    
    Parameters:
    -----------
    burgers_vector : float
        Burgers矢量大小 (nm)
    misorientation_rad : float
        取向差角度 (弧度)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        位错间距 (nm)
    """
    if burgers_vector <= 0:
        raise ValueError("Burgers矢量必须为正数")
    if misorientation_rad <= 0:
        raise ValueError("取向差角度必须为正数")
    
    D = burgers_vector / misorientation_rad
    
    return {
        'result': D,
        'metadata': {
            'burgers_vector_nm': burgers_vector,
            'misorientation_rad': misorientation_rad,
            'misorientation_deg': np.degrees(misorientation_rad),
            'formula': 'D = b / θ'
        }
    }


def calculate_dislocation_density(total_dislocations: int, boundary_length: float) -> Dict:
    """
    计算晶界上的位错线密度
    
    Parameters:
    -----------
    total_dislocations : int
        位错总数
    boundary_length : float
        晶界长度 (mm)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        位错密度 (个/mm)
    """
    if total_dislocations < 0:
        raise ValueError("位错数量不能为负")
    if boundary_length <= 0:
        raise ValueError("晶界长度必须为正数")
    
    density = total_dislocations / boundary_length
    
    return {
        'result': density,
        'metadata': {
            'total_dislocations': total_dislocations,
            'boundary_length_mm': boundary_length,
            'unit': 'dislocations/mm'
        }
    }


def calculate_grain_boundary_energy_read_shockley(
    misorientation_deg: float,
    gamma_m: float,
    theta_m: float = 15.0
) -> Dict:
    """
    使用Read-Shockley模型计算小角度晶界能
    
    γ = γ_m * (θ/θ_m) * (1 - ln(θ/θ_m))
    
    适用于 θ < θ_m (通常 θ_m ≈ 15°)
    
    Parameters:
    -----------
    misorientation_deg : float
        取向差角度 (度)
    gamma_m : float
        最大晶界能 (J/m²)，对应于 θ_m
    theta_m : float
        临界角度 (度)，默认15°
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        晶界能 (J/m²)
    """
    if misorientation_deg <= 0:
        raise ValueError("取向差角度必须为正数")
    if misorientation_deg >= theta_m:
        raise ValueError(f"取向差角度 {misorientation_deg}° 超过临界角度 {theta_m}°，不适用Read-Shockley模型")
    if gamma_m <= 0:
        raise ValueError("最大晶界能必须为正数")
    
    theta_ratio = misorientation_deg / theta_m
    gamma = gamma_m * theta_ratio * (1 - np.log(theta_ratio))
    
    return {
        'result': gamma,
        'metadata': {
            'misorientation_deg': misorientation_deg,
            'gamma_m': gamma_m,
            'theta_m': theta_m,
            'theta_ratio': theta_ratio,
            'model': 'Read-Shockley',
            'formula': 'γ = γ_m * (θ/θ_m) * (1 - ln(θ/θ_m))'
        }
    }


def solve_angle_balance_at_junction(alpha_deg: float, beta_deg: float) -> Dict:
    """
    求解三晶界交汇点的角度平衡
    
    在点O处，三个亚晶界OA、OB、OC相交
    已知 α (OA与OB之间) 和 β (OB与OC之间)
    第三个角 γ (OC与OA之间) = 360° - α - β
    
    Parameters:
    -----------
    alpha_deg : float
        角度α (度)
    beta_deg : float
        角度β (度)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        第三个角γ (度)
    """
    if alpha_deg <= 0 or beta_deg <= 0:
        raise ValueError("角度必须为正数")
    if alpha_deg + beta_deg >= 360:
        raise ValueError("两个角度之和不能大于或等于360°")
    
    gamma_deg = 360 - alpha_deg - beta_deg
    
    return {
        'result': gamma_deg,
        'metadata': {
            'alpha_deg': alpha_deg,
            'beta_deg': beta_deg,
            'gamma_deg': gamma_deg,
            'sum_check': alpha_deg + beta_deg + gamma_deg
        }
    }


# ==================== 第二层：组合函数 ====================

def calculate_misorientation_from_dislocations(
    lattice_constant: float,
    total_dislocations: int,
    total_boundary_length: float
) -> Dict:
    """
    从位错密度计算取向差角度
    
    综合流程：
    1. 计算BCC结构的Burgers矢量
    2. 计算平均位错间距
    3. 反推取向差角度 θ = b / D
    
    Parameters:
    -----------
    lattice_constant : float
        晶格常数 (nm)
    total_dislocations : int
        三条晶界上的总位错数
    total_boundary_length : float
        三条晶界的总长度 (mm)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        取向差角度 (度) 和详细计算过程
    """
    # 步骤1：计算Burgers矢量
    b_result = calculate_burgers_vector_bcc(lattice_constant)
    b_nm = b_result['result']
    
    # 步骤2：计算位错密度
    density_result = calculate_dislocation_density(total_dislocations, total_boundary_length)
    density_per_mm = density_result['result']
    
    # 步骤3：计算平均位错间距
    # 位错间距 D = 1 / 密度
    D_mm = 1.0 / density_per_mm
    D_nm = D_mm * 1e6  # 转换为nm
    
    # 步骤4：计算取向差角度
    # θ (rad) = b / D
    theta_rad = b_nm / D_nm
    theta_deg = np.degrees(theta_rad)
    
    return {
        'result': theta_deg,
        'metadata': {
            'burgers_vector_nm': b_nm,
            'dislocation_density_per_mm': density_per_mm,
            'average_spacing_nm': D_nm,
            'misorientation_rad': theta_rad,
            'misorientation_deg': theta_deg,
            'calculation_steps': [
                f"1. Burgers vector: {b_nm:.6f} nm",
                f"2. Dislocation density: {density_per_mm:.4f} /mm",
                f"3. Average spacing: {D_nm:.2f} nm",
                f"4. Misorientation: {theta_deg:.6f}°"
            ]
        }
    }


def calculate_boundary_energies_from_reference(
    reference_energy: float,
    reference_angle_deg: float,
    target_angles_deg: List[float],
    theta_m: float = 15.0
) -> Dict:
    """
    根据参考晶界能计算其他晶界的能量
    
    使用Read-Shockley模型的比例关系：
    γ₁/γ₂ = [θ₁(1-ln(θ₁/θ_m))] / [θ₂(1-ln(θ₂/θ_m))]
    
    Parameters:
    -----------
    reference_energy : float
        参考晶界能 (J/m²)
    reference_angle_deg : float
        参考取向差角度 (度)
    target_angles_deg : list of float
        目标取向差角度列表 (度)
    theta_m : float
        临界角度 (度)
    
    Returns:
    --------
    dict : {'result': list, 'metadata': dict}
        目标晶界能列表 (J/m²)
    """
    if reference_energy <= 0:
        raise ValueError("参考晶界能必须为正数")
    if reference_angle_deg <= 0 or reference_angle_deg >= theta_m:
        raise ValueError(f"参考角度必须在 (0, {theta_m}°) 范围内")
    
    # 计算参考晶界的归一化因子
    ref_ratio = reference_angle_deg / theta_m
    ref_factor = ref_ratio * (1 - np.log(ref_ratio))
    
    target_energies = []
    calculation_details = []
    
    for target_angle in target_angles_deg:
        if target_angle <= 0 or target_angle >= theta_m:
            raise ValueError(f"目标角度 {target_angle}° 必须在 (0, {theta_m}°) 范围内")
        
        # 计算目标晶界的归一化因子
        target_ratio = target_angle / theta_m
        target_factor = target_ratio * (1 - np.log(target_ratio))
        
        # 计算目标晶界能
        target_energy = reference_energy * (target_factor / ref_factor)
        target_energies.append(target_energy)
        
        calculation_details.append({
            'angle_deg': target_angle,
            'energy_J_m2': target_energy,
            'ratio_to_reference': target_energy / reference_energy
        })
    
    return {
        'result': target_energies,
        'metadata': {
            'reference_energy': reference_energy,
            'reference_angle_deg': reference_angle_deg,
            'theta_m': theta_m,
            'target_angles_deg': target_angles_deg,
            'calculation_details': calculation_details,
            'model': 'Read-Shockley proportional relationship'
        }
    }


def analyze_triple_junction_misorientations(
    alpha_deg: float,
    beta_deg: float,
    base_misorientation_deg: float
) -> Dict:
    """
    分析三晶界交汇点的取向差分布
    
    假设三个亚晶界具有相同的取向差角度（由位错密度决定）
    但它们在空间中的夹角不同
    
    Parameters:
    -----------
    alpha_deg : float
        OA与OB之间的夹角 (度)
    beta_deg : float
        OB与OC之间的夹角 (度)
    base_misorientation_deg : float
        基础取向差角度 (度)
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
        三条晶界的取向差信息
    """
    # 计算第三个角
    gamma_result = solve_angle_balance_at_junction(alpha_deg, beta_deg)
    gamma_deg = gamma_result['result']
    
    # 在均匀分布假设下，三条晶界具有相同的取向差
    misorientations = {
        'OA': base_misorientation_deg,
        'OB': base_misorientation_deg,
        'OC': base_misorientation_deg
    }
    
    junction_angles = {
        'alpha_OA_OB': alpha_deg,
        'beta_OB_OC': beta_deg,
        'gamma_OC_OA': gamma_deg
    }
    
    return {
        'result': {
            'misorientations': misorientations,
            'junction_angles': junction_angles
        },
        'metadata': {
            'assumption': 'uniform dislocation distribution',
            'base_misorientation_deg': base_misorientation_deg,
            'note': 'All three boundaries have the same misorientation angle'
        }
    }


# ==================== 第三层：可视化函数 ====================

def visualize_triple_junction(
    alpha_deg: float,
    beta_deg: float,
    boundary_length: float = 0.2,
    save_path: str = './tool_images/triple_junction.png'
) -> Dict:
    """
    可视化三晶界交汇点结构
    
    Parameters:
    -----------
    alpha_deg : float
        角度α (度)
    beta_deg : float
        角度β (度)
    boundary_length : float
        晶界长度 (mm)
    save_path : str
        保存路径
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 计算第三个角
    gamma_deg = 360 - alpha_deg - beta_deg
    
    # 定义三条晶界的方向（从O点出发）
    # OA 沿着180° (向左)
    angle_OA = 180
    # OB 在OA逆时针旋转α角度
    angle_OB = angle_OA - alpha_deg
    # OC 在OB逆时针旋转β角度
    angle_OC = angle_OB - beta_deg
    
    # 转换为弧度
    angles_rad = [np.radians(a) for a in [angle_OA, angle_OB, angle_OC]]
    
    # 计算端点坐标
    O = np.array([0, 0])
    A = O + boundary_length * np.array([np.cos(angles_rad[0]), np.sin(angles_rad[0])])
    B = O + boundary_length * np.array([np.cos(angles_rad[1]), np.sin(angles_rad[1])])
    C = O + boundary_length * np.array([np.cos(angles_rad[2]), np.sin(angles_rad[2])])
    
    # 绘制晶界
    ax.plot([O[0], A[0]], [O[1], A[1]], 'b-', linewidth=2, label='Boundary OA')
    ax.plot([O[0], B[0]], [O[1], B[1]], 'r-', linewidth=2, label='Boundary OB')
    ax.plot([O[0], C[0]], [O[1], C[1]], 'g-', linewidth=2, label='Boundary OC')
    
    # 标注点
    ax.plot(*O, 'ko', markersize=10)
    ax.text(O[0], O[1]-0.03, 'O', fontsize=14, ha='center', va='top')
    ax.text(A[0]-0.02, A[1], 'A', fontsize=14, ha='right', va='center')
    ax.text(B[0], B[1]+0.02, 'B', fontsize=14, ha='center', va='bottom')
    ax.text(C[0], C[1]-0.02, 'C', fontsize=14, ha='center', va='top')
    
    # 绘制角度弧线
    from matplotlib.patches import Arc
    arc_radius = 0.05
    
    # α角 (OA到OB)
    arc_alpha = Arc(O, 2*arc_radius, 2*arc_radius, 
                    angle=0, theta1=angle_OB, theta2=angle_OA,
                    color='purple', linewidth=1.5)
    ax.add_patch(arc_alpha)
    mid_angle_alpha = (angle_OA + angle_OB) / 2
    label_pos_alpha = O + (arc_radius + 0.03) * np.array([np.cos(np.radians(mid_angle_alpha)), 
                                                            np.sin(np.radians(mid_angle_alpha))])
    ax.text(label_pos_alpha[0], label_pos_alpha[1], f'α={alpha_deg}°', 
            fontsize=11, ha='center', va='center', color='purple')
    
    # β角 (OB到OC)
    arc_beta = Arc(O, 2*arc_radius*0.7, 2*arc_radius*0.7,
                   angle=0, theta1=angle_OC, theta2=angle_OB,
                   color='orange', linewidth=1.5)
    ax.add_patch(arc_beta)
    mid_angle_beta = (angle_OB + angle_OC) / 2
    label_pos_beta = O + (arc_radius*0.7 + 0.03) * np.array([np.cos(np.radians(mid_angle_beta)),
                                                               np.sin(np.radians(mid_angle_beta))])
    ax.text(label_pos_beta[0], label_pos_beta[1], f'β={beta_deg}°',
            fontsize=11, ha='center', va='center', color='orange')
    
    # γ角 (OC到OA)
    arc_gamma = Arc(O, 2*arc_radius*0.4, 2*arc_radius*0.4,
                    angle=0, theta1=angle_OA, theta2=angle_OC,
                    color='brown', linewidth=1.5)
    ax.add_patch(arc_gamma)
    mid_angle_gamma = (angle_OC + angle_OA + 360) / 2 if angle_OC < angle_OA else (angle_OC + angle_OA) / 2
    label_pos_gamma = O + (arc_radius*0.4 + 0.03) * np.array([np.cos(np.radians(mid_angle_gamma)),
                                                                np.sin(np.radians(mid_angle_gamma))])
    ax.text(label_pos_gamma[0], label_pos_gamma[1], f'γ={gamma_deg:.0f}°',
            fontsize=11, ha='center', va='center', color='brown')
    
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlabel('Distance (mm)', fontsize=12)
    ax.set_ylabel('Distance (mm)', fontsize=12)
    ax.set_title('Triple Junction of Subgrain Boundaries', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'alpha_deg': alpha_deg,
            'beta_deg': beta_deg,
            'gamma_deg': gamma_deg,
            'boundary_length_mm': boundary_length
        }
    }


def plot_grain_boundary_energy_vs_misorientation(
    misorientation_range: List[float],
    gamma_m: float = 1.0,
    theta_m: float = 15.0,
    highlight_points: List[Tuple[float, float]] = None,
    save_path: str = './tool_images/gb_energy_curve.png'
) -> Dict:
    """
    绘制晶界能随取向差角度的变化曲线（Read-Shockley模型）
    
    Parameters:
    -----------
    misorientation_range : list of float
        取向差角度范围 [min, max] (度)
    gamma_m : float
        最大晶界能 (J/m²)
    theta_m : float
        临界角度 (度)
    highlight_points : list of tuple
        需要高亮显示的点 [(angle1, energy1), (angle2, energy2), ...]
    save_path : str
        保存路径
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 生成角度序列
    theta_array = np.linspace(misorientation_range[0], misorientation_range[1], 200)
    gamma_array = []
    
    for theta in theta_array:
        if theta >= theta_m:
            gamma_array.append(gamma_m)
        else:
            theta_ratio = theta / theta_m
            gamma = gamma_m * theta_ratio * (1 - np.log(theta_ratio))
            gamma_array.append(gamma)
    
    # 绘制主曲线
    ax.plot(theta_array, gamma_array, 'b-', linewidth=2, label='Read-Shockley Model')
    
    # 高亮特定点
    if highlight_points:
        for i, (angle, energy) in enumerate(highlight_points):
            ax.plot(angle, energy, 'ro', markersize=10, zorder=5)
            ax.annotate(f'θ={angle:.3f}°\nγ={energy:.2f} J/m²',
                       xy=(angle, energy), xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=10, ha='left',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 标注临界角度
    ax.axvline(theta_m, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'θ_m = {theta_m}°')
    ax.axhline(gamma_m, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'γ_m = {gamma_m} J/m²')
    
    ax.set_xlabel('Misorientation Angle θ (degrees)', fontsize=13)
    ax.set_ylabel('Grain Boundary Energy γ (J/m²)', fontsize=13)
    ax.set_title('Grain Boundary Energy vs Misorientation Angle', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'misorientation_range_deg': misorientation_range,
            'gamma_m': gamma_m,
            'theta_m': theta_m,
            'num_highlight_points': len(highlight_points) if highlight_points else 0
        }
    }


def visualize_dislocation_array(
    num_dislocations: int,
    boundary_length: float,
    burgers_vector: float,
    save_path: str = './tool_images/dislocation_array.png'
) -> Dict:
    """
    可视化晶界上的位错阵列
    
    Parameters:
    -----------
    num_dislocations : int
        位错数量
    boundary_length : float
        晶界长度 (mm)
    burgers_vector : float
        Burgers矢量大小 (nm)
    save_path : str
        保存路径
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        图像文件路径
    """
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算位错间距
    spacing_mm = boundary_length / num_dislocations
    spacing_nm = spacing_mm * 1e6
    
    # 绘制晶界
    ax.plot([0, boundary_length], [0, 0], 'k-', linewidth=3, label='Grain Boundary')
    
    # 绘制位错符号（⊥表示刃位错）
    positions = np.linspace(0, boundary_length, num_dislocations, endpoint=True)
    for i, pos in enumerate(positions):
        # 绘制位错符号（使用文本绘制 ⊥，避免matplotlib格式字符串错误）
        ax.text(pos, 0, '⊥', color='red', fontsize=18, fontweight='bold',
                ha='center', va='center')
        # 标注位错编号（仅标注部分以避免拥挤）
        if num_dislocations <= 20 or i % (num_dislocations // 10) == 0:
            ax.text(pos, -0.01, f'{i+1}', fontsize=8, ha='center', va='top')
    
    # 标注间距
    if num_dislocations > 1:
        mid_pos = (positions[0] + positions[1]) / 2
        ax.annotate('', xy=(positions[1], 0.015), xytext=(positions[0], 0.015),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
        ax.text(mid_pos, 0.02, f'D = {spacing_nm:.1f} nm', 
               fontsize=10, ha='center', color='blue', fontweight='bold')
    
    ax.set_xlim(-0.01, boundary_length + 0.01)
    ax.set_ylim(-0.03, 0.04)
    ax.set_xlabel('Position along boundary (mm)', fontsize=12)
    ax.set_title(f'Edge Dislocation Array in Subgrain Boundary\n'
                f'Total: {num_dislocations} dislocations, Spacing: {spacing_nm:.1f} nm, '
                f'Burgers vector: {burgers_vector:.4f} nm',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'num_dislocations': num_dislocations,
            'boundary_length_mm': boundary_length,
            'spacing_nm': spacing_nm,
            'burgers_vector_nm': burgers_vector
        }
    }


# ==================== 主函数：三个场景演示 ====================

def main():
    """
    主函数：演示三个场景
    场景1：解决原始问题（计算取向差角度和晶界能）
    场景2：分析不同位错密度对取向差的影响
    场景3：研究晶界能随取向差的变化规律
    """
    
    print("=" * 80)
    print("场景1：Fe-3%Si合金亚晶界结构分析（原始问题）")
    print("=" * 80)
    print("问题描述：")
    print("Fe-3%Si合金（bcc结构，a=0.3nm）经过回复退火形成三个亚晶粒交汇于O点")
    print("三条亚晶界OA、OB、OC长度均为0.2mm，总位错数为1.198×10⁴个")
    print("已知α=120°，β=80°")
    print("(1) 计算相邻亚晶粒间的取向差角度θ")
    print("(2) 若OA晶界能为0.8 J/m²，计算OB和OC的晶界能")
    print("-" * 80)
    
    # 给定参数
    lattice_constant = 0.3  # nm
    total_dislocations = 1.198e4  # 实际测量值
    single_boundary_length = 0.2  # mm
    total_boundary_length = 3 * single_boundary_length  # mm
    alpha_deg = 120.0
    beta_deg = 80.0
    reference_energy_OA = 0.8  # J/m²
    
    # 步骤1：计算Burgers矢量
    print("\n步骤1：计算BCC结构的Burgers矢量")
    # 调用函数：calculate_burgers_vector_bcc()
    b_result = calculate_burgers_vector_bcc(lattice_constant)
    print(f"FUNCTION_CALL: calculate_burgers_vector_bcc | PARAMS: {{'lattice_constant': {lattice_constant}}} | RESULT: {b_result}")
    b_nm = b_result['result']
    print(f"Burgers矢量 b = {b_nm:.6f} nm")
    
    # 步骤2：计算取向差角度
    print("\n步骤2：从位错密度计算取向差角度")
    # 调用函数：calculate_misorientation_from_dislocations()
    theta_result = calculate_misorientation_from_dislocations(
        lattice_constant,
        total_dislocations,
        total_boundary_length
    )
    print(f"FUNCTION_CALL: calculate_misorientation_from_dislocations | PARAMS: {{'lattice_constant': {lattice_constant}, 'total_dislocations': {total_dislocations}, 'total_boundary_length': {total_boundary_length}}} | RESULT: {theta_result}")
    theta_deg = theta_result['result']
    print(f"取向差角度 θ = {theta_deg:.6f}° = {theta_deg:.3f}°")
    
    # 步骤3：分析三晶界交汇点
    print("\n步骤3：分析三晶界交汇点的几何关系")
    # 调用函数：analyze_triple_junction_misorientations()
    junction_result = analyze_triple_junction_misorientations(alpha_deg, beta_deg, theta_deg)
    print(f"FUNCTION_CALL: analyze_triple_junction_misorientations | PARAMS: {{'alpha_deg': {alpha_deg}, 'beta_deg': {beta_deg}, 'base_misorientation_deg': {theta_deg}}} | RESULT: {junction_result}")
    gamma_deg = junction_result['result']['junction_angles']['gamma_OC_OA']
    print(f"第三个角 γ = {gamma_deg:.1f}°")
    print(f"三条晶界的取向差角度均为 θ = {theta_deg:.3f}°")
    
    # 步骤4：计算其他晶界的能量
    print("\n步骤4：根据OA的晶界能计算OB和OC的晶界能")
    print(f"已知：OA晶界能 = {reference_energy_OA} J/m²")
    sin_alpha = np.sin(np.radians(alpha_deg))
    sin_beta = np.sin(np.radians(beta_deg))
    sin_gamma = np.sin(np.radians(gamma_deg))
    balance_factor = reference_energy_OA / sin_beta  # OA 对应夹角 β（介于 OB 与 OC 之间）
    energy_OB = balance_factor * sin_alpha
    energy_OC = balance_factor * sin_gamma
    print(f"力学平衡系数 k = γ_OA / sinβ = {balance_factor:.4f}")
    print(f"计算得到：OB晶界能 = {energy_OB:.2f} J/m², OC晶界能 = {energy_OC:.2f} J/m²")
    
    # 步骤5：可视化三晶界结构
    print("\n步骤5：可视化三晶界交汇点结构")
    # 调用函数：visualize_triple_junction()
    viz_result = visualize_triple_junction(alpha_deg, beta_deg, single_boundary_length)
    print(f"FUNCTION_CALL: visualize_triple_junction | PARAMS: {{'alpha_deg': {alpha_deg}, 'beta_deg': {beta_deg}, 'boundary_length': {single_boundary_length}}} | RESULT: {viz_result}")
    
    # 步骤6：可视化位错阵列
    print("\n步骤6：可视化单条晶界上的位错阵列")
    # 调用函数：visualize_dislocation_array()
    dislocations_per_boundary = int(round(total_dislocations / 3))
    disloc_viz_result = visualize_dislocation_array(
        dislocations_per_boundary,
        single_boundary_length,
        b_nm
    )
    print(f"FUNCTION_CALL: visualize_dislocation_array | PARAMS: {{'num_dislocations': {dislocations_per_boundary}, 'boundary_length': {single_boundary_length}, 'burgers_vector': {b_nm}}} | RESULT: {disloc_viz_result}")
    
    print("\n" + "=" * 80)
    print(f"FINAL_ANSWER: (1) 取向差角度 θ = {theta_deg:.3f}°; (2) OB晶界能 = {energy_OB} J/m², OC晶界能 = {energy_OC} J/m²")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：研究位错密度对取向差角度的影响")
    print("=" * 80)
    print("问题描述：")
    print("在相同的Fe-3%Si合金中，如果位错密度变化，取向差角度如何变化？")
    print("假设位错总数分别为6、12、24个，晶界总长度仍为0.6mm")
    print("-" * 80)
    
    dislocation_counts = [6, 12, 24]
    results_scenario2 = []
    
    for count in dislocation_counts:
        print(f"\n--- 位错总数 = {count} ---")
        # 调用函数：calculate_misorientation_from_dislocations()
        theta_result_s2 = calculate_misorientation_from_dislocations(
            lattice_constant,
            count,
            total_boundary_length
        )
        print(f"FUNCTION_CALL: calculate_misorientation_from_dislocations | PARAMS: {{'lattice_constant': {lattice_constant}, 'total_dislocations': {count}, 'total_boundary_length': {total_boundary_length}}} | RESULT: {theta_result_s2}")
        theta_s2 = theta_result_s2['result']
        print(f"取向差角度 θ = {theta_s2:.6f}°")
        results_scenario2.append((count, theta_s2))
    
    print("\n总结：")
    print("位错数量越多 → 位错密度越大 → 位错间距越小 → 取向差角度越大")
    for count, theta in results_scenario2:
        print(f"  位错数 = {count:2d}, θ = {theta:.6f}°")
    
    print(f"\nFINAL_ANSWER: 位错密度与取向差角度成正比关系")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：绘制晶界能随取向差角度的变化曲线")
    print("=" * 80)
    print("问题描述：")
    print("使用Read-Shockley模型，绘制晶界能随取向差角度的变化")
    print("并标注场景1中的三个晶界位置")
    print("-" * 80)
    
    # 准备高亮点
    highlight_points_s3 = [
        (theta_deg, reference_energy_OA),
        (theta_deg, energy_OB),
        (theta_deg, energy_OC)
    ]
    
    print("\n步骤1：绘制晶界能曲线")
    # 调用函数：plot_grain_boundary_energy_vs_misorientation()
    curve_result = plot_grain_boundary_energy_vs_misorientation(
        misorientation_range=[0.1, 10.0],
        gamma_m=1.0,
        theta_m=15.0,
        highlight_points=highlight_points_s3
    )
    print(f"FUNCTION_CALL: plot_grain_boundary_energy_vs_misorientation | PARAMS: {{'misorientation_range': [0.1, 10.0], 'gamma_m': 1.0, 'theta_m': 15.0, 'highlight_points': {highlight_points_s3}}} | RESULT: {curve_result}")
    
    print("\n观察：")
    print("1. 晶界能随取向差角度增加而增加")
    print("2. 在小角度范围内（θ < θ_m），符合Read-Shockley模型")
    print("3. 当θ接近θ_m时，晶界能趋于饱和")
    
    print(f"\nFINAL_ANSWER: 晶界能曲线已生成，展示了Read-Shockley模型的预测")


if __name__ == "__main__":
    main()