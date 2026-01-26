# Filename: composite_torsion_toolkit.py

"""
复合材料扭转力学计算工具包
用于分析由多种材料组成的复合扭转构件的强度和刚度问题
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

# ==================== 第一层：原子函数 ====================

def calculate_polar_moment_solid(diameter: float) -> Dict:
    """
    计算实心圆截面的极惯性矩
    
    Args:
        diameter: 圆截面直径 (mm)
    
    Returns:
        dict: {'result': Ip值(mm^4), 'metadata': {...}}
    """
    if diameter <= 0:
        raise ValueError(f"直径必须为正数，当前值: {diameter}")
    
    Ip = np.pi * diameter**4 / 32
    
    return {
        'result': Ip,
        'metadata': {
            'diameter': diameter,
            'formula': 'Ip = π*d^4/32',
            'unit': 'mm^4'
        }
    }


def calculate_polar_moment_hollow(outer_diameter: float, inner_diameter: float) -> Dict:
    """
    计算空心圆截面的极惯性矩
    
    Args:
        outer_diameter: 外径 (mm)
        inner_diameter: 内径 (mm)
    
    Returns:
        dict: {'result': Ip值(mm^4), 'metadata': {...}}
    """
    if outer_diameter <= 0 or inner_diameter <= 0:
        raise ValueError(f"直径必须为正数，外径: {outer_diameter}, 内径: {inner_diameter}")
    if inner_diameter >= outer_diameter:
        raise ValueError(f"内径必须小于外径，内径: {inner_diameter}, 外径: {outer_diameter}")
    
    Ip = np.pi * (outer_diameter**4 - inner_diameter**4) / 32
    
    return {
        'result': Ip,
        'metadata': {
            'outer_diameter': outer_diameter,
            'inner_diameter': inner_diameter,
            'formula': 'Ip = π*(D^4 - d^4)/32',
            'unit': 'mm^4'
        }
    }


def calculate_torsional_stiffness(shear_modulus: float, polar_moment: float, length: float) -> Dict:
    """
    计算扭转刚度 GIp/l
    
    Args:
        shear_modulus: 剪切模量 (GPa)
        polar_moment: 极惯性矩 (mm^4)
        length: 长度 (mm)
    
    Returns:
        dict: {'result': 扭转刚度(N·mm), 'metadata': {...}}
    """
    if shear_modulus <= 0 or polar_moment <= 0 or length <= 0:
        raise ValueError("所有参数必须为正数")
    
    # 单位转换: GPa -> N/mm^2
    G_N_mm2 = shear_modulus * 1000
    stiffness = G_N_mm2 * polar_moment / length
    
    return {
        'result': stiffness,
        'metadata': {
            'shear_modulus_GPa': shear_modulus,
            'polar_moment_mm4': polar_moment,
            'length_mm': length,
            'formula': 'k = G*Ip/l',
            'unit': 'N·mm'
        }
    }


def calculate_max_shear_stress(torque: float, polar_moment: float, radius: float) -> Dict:
    """
    计算最大剪应力 τ = T*r/Ip
    
    Args:
        torque: 扭矩 (N·mm)
        polar_moment: 极惯性矩 (mm^4)
        radius: 半径 (mm)
    
    Returns:
        dict: {'result': 剪应力(MPa), 'metadata': {...}}
    """
    if polar_moment <= 0 or radius <= 0:
        raise ValueError("极惯性矩和半径必须为正数")
    
    tau = torque * radius / polar_moment
    tau_MPa = tau  # N/mm^2 = MPa
    
    return {
        'result': tau_MPa,
        'metadata': {
            'torque_Nmm': torque,
            'polar_moment_mm4': polar_moment,
            'radius_mm': radius,
            'formula': 'τ = T*r/Ip',
            'unit': 'MPa'
        }
    }


def distribute_torque_by_stiffness(total_torque: float, stiffness_list: List[float]) -> Dict:
    """
    根据扭转刚度分配扭矩（变形协调条件）
    对于并联扭转构件：φ1 = φ2，即 T1/(k1) = T2/(k2)
    且 T1 + T2 = M_total
    
    Args:
        total_torque: 总扭矩 (N·mm)
        stiffness_list: 各构件的扭转刚度列表 (N·mm)
    
    Returns:
        dict: {'result': [T1, T2, ...], 'metadata': {...}}
    """
    if not stiffness_list or any(k <= 0 for k in stiffness_list):
        raise ValueError("刚度列表不能为空且所有刚度必须为正数")
    
    total_stiffness = sum(stiffness_list)
    torque_distribution = [total_torque * k / total_stiffness for k in stiffness_list]
    
    return {
        'result': torque_distribution,
        'metadata': {
            'total_torque_Nmm': total_torque,
            'stiffness_list': stiffness_list,
            'total_stiffness': total_stiffness,
            'principle': '变形协调条件: φ1 = φ2',
            'unit': 'N·mm'
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_composite_torsion_member(
    d1: float, d2: float, D2: float, length: float,
    G1: float, tau1_allow: float,
    G2: float, tau2_allow: float
) -> Dict:
    """
    分析复合扭转构件（实心钢杆+空心铝管）
    
    Args:
        d1: 钢杆直径 (mm)
        d2: 铝管内径 (mm)
        D2: 铝管外径 (mm)
        length: 构件长度 (mm)
        G1: 钢的剪切模量 (GPa)
        tau1_allow: 钢的许用剪应力 (MPa)
        G2: 铝的剪切模量 (GPa)
        tau2_allow: 铝的许用剪应力 (MPa)
    
    Returns:
        dict: 包含极惯性矩、扭转刚度等信息
    """
    # 计算钢杆极惯性矩
    Ip1_result = calculate_polar_moment_solid(d1)
    Ip1 = Ip1_result['result']
    
    # 计算铝管极惯性矩
    Ip2_result = calculate_polar_moment_hollow(D2, d2)
    Ip2 = Ip2_result['result']
    
    # 计算扭转刚度
    k1_result = calculate_torsional_stiffness(G1, Ip1, length)
    k1 = k1_result['result']
    
    k2_result = calculate_torsional_stiffness(G2, Ip2, length)
    k2 = k2_result['result']
    
    return {
        'result': {
            'steel_polar_moment': Ip1,
            'aluminum_polar_moment': Ip2,
            'steel_stiffness': k1,
            'aluminum_stiffness': k2,
            'steel_radius': d1 / 2,
            'aluminum_outer_radius': D2 / 2
        },
        'metadata': {
            'geometry': {
                'd1': d1, 'd2': d2, 'D2': D2, 'length': length
            },
            'material_properties': {
                'steel': {'G': G1, 'tau_allow': tau1_allow},
                'aluminum': {'G': G2, 'tau_allow': tau2_allow}
            }
        }
    }


def calculate_allowable_torque_by_strength(
    polar_moment: float, radius: float, tau_allow: float
) -> Dict:
    """
    根据强度条件计算许用扭矩
    τ_max = T*r/Ip ≤ [τ]
    T_allow = [τ]*Ip/r
    
    Args:
        polar_moment: 极惯性矩 (mm^4)
        radius: 半径 (mm)
        tau_allow: 许用剪应力 (MPa)
    
    Returns:
        dict: {'result': 许用扭矩(N·mm), 'metadata': {...}}
    """
    if polar_moment <= 0 or radius <= 0 or tau_allow <= 0:
        raise ValueError("所有参数必须为正数")
    
    T_allow = tau_allow * polar_moment / radius
    
    return {
        'result': T_allow,
        'metadata': {
            'polar_moment_mm4': polar_moment,
            'radius_mm': radius,
            'tau_allow_MPa': tau_allow,
            'formula': 'T_allow = [τ]*Ip/r',
            'unit': 'N·mm'
        }
    }


def determine_max_torque_composite(
    analysis_result: Dict,
    tau1_allow: float,
    tau2_allow: float
) -> Dict:
    """
    确定复合构件能承受的最大扭矩
    考虑两个材料的强度限制
    
    Args:
        analysis_result: analyze_composite_torsion_member的返回结果
        tau1_allow: 钢的许用剪应力 (MPa)
        tau2_allow: 铝的许用剪应力 (MPa)
    
    Returns:
        dict: 包含最大扭矩和控制因素
    """
    data = analysis_result['result']
    
    Ip1 = data['steel_polar_moment']
    Ip2 = data['aluminum_polar_moment']
    r1 = data['steel_radius']
    r2 = data['aluminum_outer_radius']
    k1 = data['steel_stiffness']
    k2 = data['aluminum_stiffness']
    
    # 假设总扭矩为M，根据刚度分配
    # T1 = M * k1/(k1+k2)
    # T2 = M * k2/(k1+k2)
    
    # 钢的强度条件: T1*r1/Ip1 ≤ [τ]1
    # M * k1/(k1+k2) * r1/Ip1 ≤ [τ]1
    # M ≤ [τ]1 * Ip1/r1 * (k1+k2)/k1
    
    M1_allow = tau1_allow * Ip1 / r1 * (k1 + k2) / k1
    
    # 铝的强度条件: T2*r2/Ip2 ≤ [τ]2
    # M * k2/(k1+k2) * r2/Ip2 ≤ [τ]2
    # M ≤ [τ]2 * Ip2/r2 * (k1+k2)/k2
    
    M2_allow = tau2_allow * Ip2 / r2 * (k1 + k2) / k2
    
    # 取较小值
    M_max = min(M1_allow, M2_allow)
    controlling_material = 'steel' if M1_allow < M2_allow else 'aluminum'
    
    # 计算实际扭矩分配
    T1_actual = M_max * k1 / (k1 + k2)
    T2_actual = M_max * k2 / (k1 + k2)
    
    # 计算实际应力
    tau1_actual = T1_actual * r1 / Ip1
    tau2_actual = T2_actual * r2 / Ip2
    
    return {
        'result': M_max,
        'metadata': {
            'M1_allow_Nmm': M1_allow,
            'M2_allow_Nmm': M2_allow,
            'controlling_material': controlling_material,
            'torque_distribution': {
                'steel_torque_Nmm': T1_actual,
                'aluminum_torque_Nmm': T2_actual
            },
            'actual_stresses': {
                'steel_stress_MPa': tau1_actual,
                'aluminum_stress_MPa': tau2_actual
            },
            'allowable_stresses': {
                'steel_allow_MPa': tau1_allow,
                'aluminum_allow_MPa': tau2_allow
            },
            'unit': 'N·mm'
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_stress_distribution(
    analysis_result: Dict,
    max_torque_result: Dict,
    save_path: str = './tool_images/stress_distribution.png'
) -> Dict:
    """
    绘制复合构件的剪应力分布图
    
    Args:
        analysis_result: 构件分析结果
        max_torque_result: 最大扭矩计算结果
        save_path: 图像保存路径
    
    Returns:
        dict: 包含文件路径信息
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = analysis_result['result']
    torque_data = max_torque_result['metadata']
    
    r1 = data['steel_radius']
    r2_inner = analysis_result['metadata']['geometry']['d2'] / 2
    r2_outer = data['aluminum_outer_radius']
    
    T1 = torque_data['torque_distribution']['steel_torque_Nmm']
    T2 = torque_data['torque_distribution']['aluminum_torque_Nmm']
    Ip1 = data['steel_polar_moment']
    Ip2 = data['aluminum_polar_moment']
    
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：径向应力分布
    r_steel = np.linspace(0, r1, 100)
    tau_steel = T1 * r_steel / Ip1
    
    r_aluminum = np.linspace(r2_inner, r2_outer, 100)
    tau_aluminum = T2 * r_aluminum / Ip2
    
    ax1.plot(r_steel, tau_steel, 'b-', linewidth=2, label='Steel')
    ax1.plot(r_aluminum, tau_aluminum, 'r-', linewidth=2, label='Aluminum')
    ax1.axhline(y=torque_data['allowable_stresses']['steel_allow_MPa'], 
                color='b', linestyle='--', label='Steel Allowable')
    ax1.axhline(y=torque_data['allowable_stresses']['aluminum_allow_MPa'], 
                color='r', linestyle='--', label='Aluminum Allowable')
    ax1.set_xlabel('Radius (mm)', fontsize=12)
    ax1.set_ylabel('Shear Stress (MPa)', fontsize=12)
    ax1.set_title('Radial Shear Stress Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：截面示意图
    circle_steel = plt.Circle((0, 0), r1, color='steelblue', alpha=0.6, label='Steel')
    circle_al_outer = plt.Circle((0, 0), r2_outer, color='lightcoral', alpha=0.6)
    circle_al_inner = plt.Circle((0, 0), r2_inner, color='white', alpha=1.0)
    
    ax2.add_patch(circle_steel)
    ax2.add_patch(circle_al_outer)
    ax2.add_patch(circle_al_inner)
    
    ax2.set_xlim(-r2_outer*1.2, r2_outer*1.2)
    ax2.set_ylim(-r2_outer*1.2, r2_outer*1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (mm)', fontsize=12)
    ax2.set_ylabel('Y (mm)', fontsize=12)
    ax2.set_title('Cross-Section View', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加标注
    ax2.annotate(f'd1={r1*2:.0f}mm', xy=(0, r1), xytext=(r1*1.5, r1*1.5),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    ax2.annotate(f'D2={r2_outer*2:.0f}mm', xy=(0, r2_outer), xytext=(-r2_outer*1.5, r2_outer*1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'description': 'Shear stress distribution in composite torsion member'
        }
    }


def plot_torque_capacity_comparison(
    analysis_result: Dict,
    max_torque_result: Dict,
    save_path: str = './tool_images/torque_capacity.png'
) -> Dict:
    """
    绘制各材料的扭矩承载能力对比图
    
    Args:
        analysis_result: 构件分析结果
        max_torque_result: 最大扭矩计算结果
        save_path: 图像保存路径
    
    Returns:
        dict: 包含文件路径信息
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metadata = max_torque_result['metadata']
    
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ['Steel Limit', 'Aluminum Limit', 'Actual Max']
    torques = [
        metadata['M1_allow_Nmm'] / 1e6,  # 转换为 kN·m
        metadata['M2_allow_Nmm'] / 1e6,
        max_torque_result['result'] / 1e6
    ]
    colors = ['steelblue', 'lightcoral', 'green']
    
    bars = ax.bar(materials, torques, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, torque in zip(bars, torques):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{torque:.2f} kN·m',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Torque Capacity (kN·m)', fontsize=12)
    ax.set_title('Torque Capacity Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 标注控制材料
    controlling = metadata['controlling_material']
    ax.text(0.5, 0.95, f'Controlling Material: {controlling.capitalize()}',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'description': 'Torque capacity comparison between materials'
        }
    }


def save_calculation_report(
    analysis_result: Dict,
    max_torque_result: Dict,
    save_path: str = './mid_result/materials/torsion_report.json'
) -> Dict:
    """
    保存详细计算报告
    
    Args:
        analysis_result: 构件分析结果
        max_torque_result: 最大扭矩计算结果
        save_path: 报告保存路径
    
    Returns:
        dict: 包含文件路径信息
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    report = {
        'problem_description': 'Composite torsion member analysis',
        'geometry': analysis_result['metadata']['geometry'],
        'material_properties': analysis_result['metadata']['material_properties'],
        'calculated_properties': analysis_result['result'],
        'max_torque_Nmm': max_torque_result['result'],
        'max_torque_kNm': max_torque_result['result'] / 1e6,
        'detailed_results': max_torque_result['metadata']
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"FILE_GENERATED: json | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'json',
            'description': 'Detailed calculation report'
        }
    }


# ==================== 主函数：演示场景 ====================

def main():
    """
    演示三个场景的计算
    """
    
    # ========== 场景1：解决原始问题 ==========
    print("=" * 60)
    print("场景1：复合扭转构件最大承载扭矩计算")
    print("=" * 60)
    print("问题描述：计算由实心钢杆和空心铝管组成的复合构件能承受的最大扭矩")
    print("-" * 60)
    
    # 给定参数
    d1 = 50.0  # mm
    d2 = 60.0  # mm
    D2 = 76.0  # mm
    length = 500.0  # mm
    G1 = 77.0  # GPa
    tau1_allow = 120.0  # MPa
    G2 = 27.0  # GPa
    tau2_allow = 70.0  # MPa
    
    # 步骤1：计算钢杆极惯性矩
    # 调用函数：calculate_polar_moment_solid()
    Ip1_result = calculate_polar_moment_solid(d1)
    Ip1 = Ip1_result['result']
    print(f"FUNCTION_CALL: calculate_polar_moment_solid | PARAMS: diameter={d1} | RESULT: {Ip1_result}")
    
    # 步骤2：计算铝管极惯性矩
    # 调用函数：calculate_polar_moment_hollow()
    Ip2_result = calculate_polar_moment_hollow(D2, d2)
    Ip2 = Ip2_result['result']
    print(f"FUNCTION_CALL: calculate_polar_moment_hollow | PARAMS: outer_diameter={D2}, inner_diameter={d2} | RESULT: {Ip2_result}")
    
    # 步骤3：计算钢杆扭转刚度
    # 调用函数：calculate_torsional_stiffness()
    k1_result = calculate_torsional_stiffness(G1, Ip1, length)
    k1 = k1_result['result']
    print(f"FUNCTION_CALL: calculate_torsional_stiffness | PARAMS: shear_modulus={G1}, polar_moment={Ip1}, length={length} | RESULT: {k1_result}")
    
    # 步骤4：计算铝管扭转刚度
    # 调用函数：calculate_torsional_stiffness()
    k2_result = calculate_torsional_stiffness(G2, Ip2, length)
    k2 = k2_result['result']
    print(f"FUNCTION_CALL: calculate_torsional_stiffness | PARAMS: shear_modulus={G2}, polar_moment={Ip2}, length={length} | RESULT: {k2_result}")
    
    # 步骤5：根据刚度分配扭矩（用于计算许用扭矩）
    # 钢的强度条件: T1*r1/Ip1 ≤ [τ]1
    # 假设总扭矩为M，根据刚度分配: T1 = M * k1/(k1+k2)
    # M * k1/(k1+k2) * r1/Ip1 ≤ [τ]1
    # M ≤ [τ]1 * Ip1/r1 * (k1+k2)/k1
    r1 = d1 / 2
    r2 = D2 / 2
    M1_allow = tau1_allow * Ip1 / r1 * (k1 + k2) / k1
    
    # 铝的强度条件: T2*r2/Ip2 ≤ [τ]2
    # M * k2/(k1+k2) * r2/Ip2 ≤ [τ]2
    # M ≤ [τ]2 * Ip2/r2 * (k1+k2)/k2
    M2_allow = tau2_allow * Ip2 / r2 * (k1 + k2) / k2
    
    # 步骤6：确定最大扭矩（取较小值）
    M_max = min(M1_allow, M2_allow)
    controlling_material = 'steel' if M1_allow < M2_allow else 'aluminum'
    
    # 步骤7：根据总扭矩和刚度分配各构件的扭矩
    # 调用函数：distribute_torque_by_stiffness()
    torque_dist_result = distribute_torque_by_stiffness(M_max, [k1, k2])
    T1_actual = torque_dist_result['result'][0]
    T2_actual = torque_dist_result['result'][1]
    print(f"FUNCTION_CALL: distribute_torque_by_stiffness | PARAMS: total_torque={M_max}, stiffness_list=[{k1}, {k2}] | RESULT: {torque_dist_result}")
    
    # 步骤8：计算钢杆实际最大剪应力
    # 调用函数：calculate_max_shear_stress()
    tau1_actual_result = calculate_max_shear_stress(T1_actual, Ip1, r1)
    tau1_actual = tau1_actual_result['result']
    print(f"FUNCTION_CALL: calculate_max_shear_stress | PARAMS: torque={T1_actual}, polar_moment={Ip1}, radius={r1} | RESULT: {tau1_actual_result}")
    
    # 步骤9：计算铝管实际最大剪应力
    # 调用函数：calculate_max_shear_stress()
    tau2_actual_result = calculate_max_shear_stress(T2_actual, Ip2, r2)
    tau2_actual = tau2_actual_result['result']
    print(f"FUNCTION_CALL: calculate_max_shear_stress | PARAMS: torque={T2_actual}, polar_moment={Ip2}, radius={r2} | RESULT: {tau2_actual_result}")
    
    # 组装分析结果用于可视化（保持与原函数接口兼容）
    analysis = {
        'result': {
            'steel_polar_moment': Ip1,
            'aluminum_polar_moment': Ip2,
            'steel_stiffness': k1,
            'aluminum_stiffness': k2,
            'steel_radius': r1,
            'aluminum_outer_radius': r2
        },
        'metadata': {
            'geometry': {
                'd1': d1, 'd2': d2, 'D2': D2, 'length': length
            },
            'material_properties': {
                'steel': {'G': G1, 'tau_allow': tau1_allow},
                'aluminum': {'G': G2, 'tau_allow': tau2_allow}
            }
        }
    }
    
    # 组装最大扭矩结果用于可视化（保持与原函数接口兼容）
    max_torque = {
        'result': M_max,
        'metadata': {
            'M1_allow_Nmm': M1_allow,
            'M2_allow_Nmm': M2_allow,
            'controlling_material': controlling_material,
            'torque_distribution': {
                'steel_torque_Nmm': T1_actual,
                'aluminum_torque_Nmm': T2_actual
            },
            'actual_stresses': {
                'steel_stress_MPa': tau1_actual,
                'aluminum_stress_MPa': tau2_actual
            },
            'allowable_stresses': {
                'steel_allow_MPa': tau1_allow,
                'aluminum_allow_MPa': tau2_allow
            },
            'unit': 'N·mm'
        }
    }
    
    # 步骤10：生成应力分布图
    # 调用函数：plot_stress_distribution()
    stress_plot = plot_stress_distribution(analysis, max_torque)
    print(f"FUNCTION_CALL: plot_stress_distribution | PARAMS: analysis, max_torque | RESULT: {stress_plot}")
    
    # 步骤11：生成扭矩承载能力对比图
    # 调用函数：plot_torque_capacity_comparison()
    capacity_plot = plot_torque_capacity_comparison(analysis, max_torque)
    print(f"FUNCTION_CALL: plot_torque_capacity_comparison | PARAMS: analysis, max_torque | RESULT: {capacity_plot}")
    
    # 步骤12：保存计算报告
    # 调用函数：save_calculation_report()
    report = save_calculation_report(analysis, max_torque)
    print(f"FUNCTION_CALL: save_calculation_report | PARAMS: analysis, max_torque | RESULT: {report}")
    
    M_max_kNm = M_max / 1e6
    print(f"\n最大承载扭矩: {M_max_kNm:.2f} kN·m")
    print(f"控制材料: {controlling_material}")
    print(f"FINAL_ANSWER: {M_max_kNm:.2f} kN·m")
    
    
    # ========== 场景2：单一材料扭转杆件强度分析 ==========
    print("\n" + "=" * 60)
    print("场景2：单一材料实心圆杆的许用扭矩计算")
    print("=" * 60)
    print("问题描述：计算直径50mm的实心钢杆在许用剪应力120MPa下能承受的最大扭矩")
    print("-" * 60)
    
    # 步骤1：计算极惯性矩
    # 调用函数：calculate_polar_moment_solid()
    Ip_solid = calculate_polar_moment_solid(d1)
    print(f"FUNCTION_CALL: calculate_polar_moment_solid | PARAMS: d={d1} | RESULT: {Ip_solid}")
    
    # 步骤2：计算许用扭矩
    # 调用函数：calculate_allowable_torque_by_strength()
    T_allow_solid = calculate_allowable_torque_by_strength(
        Ip_solid['result'], d1/2, tau1_allow
    )
    print(f"FUNCTION_CALL: calculate_allowable_torque_by_strength | PARAMS: Ip={Ip_solid['result']}, r={d1/2}, tau_allow={tau1_allow} | RESULT: {T_allow_solid}")
    
    T_allow_kNm = T_allow_solid['result'] / 1e6
    print(f"\n实心钢杆许用扭矩: {T_allow_kNm:.2f} kN·m")
    print(f"FINAL_ANSWER: {T_allow_kNm:.2f} kN·m")
    
    
    # ========== 场景3：空心圆管优化设计 ==========
    print("\n" + "=" * 60)
    print("场景3：空心铝管的扭转刚度与强度分析")
    print("=" * 60)
    print("问题描述：分析内径60mm、外径76mm的铝管在给定扭矩下的应力和变形")
    print("-" * 60)
    
    # 步骤1：计算空心圆管极惯性矩
    # 调用函数：calculate_polar_moment_hollow()
    Ip_hollow = calculate_polar_moment_hollow(D2, d2)
    print(f"FUNCTION_CALL: calculate_polar_moment_hollow | PARAMS: D={D2}, d={d2} | RESULT: {Ip_hollow}")
    
    # 步骤2：计算扭转刚度
    # 调用函数：calculate_torsional_stiffness()
    k_hollow = calculate_torsional_stiffness(G2, Ip_hollow['result'], length)
    print(f"FUNCTION_CALL: calculate_torsional_stiffness | PARAMS: G={G2}, Ip={Ip_hollow['result']}, l={length} | RESULT: {k_hollow}")
    
    # 步骤3：假设施加扭矩3 kN·m，计算最大剪应力
    T_test = 3e6  # N·mm
    # 调用函数：calculate_max_shear_stress()
    tau_max = calculate_max_shear_stress(T_test, Ip_hollow['result'], D2/2)
    print(f"FUNCTION_CALL: calculate_max_shear_stress | PARAMS: T={T_test}, Ip={Ip_hollow['result']}, r={D2/2} | RESULT: {tau_max}")
    
    # 步骤4：计算许用扭矩
    # 调用函数：calculate_allowable_torque_by_strength()
    T_allow_hollow = calculate_allowable_torque_by_strength(
        Ip_hollow['result'], D2/2, tau2_allow
    )
    print(f"FUNCTION_CALL: calculate_allowable_torque_by_strength | PARAMS: Ip={Ip_hollow['result']}, r={D2/2}, tau_allow={tau2_allow} | RESULT: {T_allow_hollow}")
    
    print(f"\n铝管扭转刚度: {k_hollow['result']:.2e} N·mm")
    print(f"在扭矩{T_test/1e6:.1f} kN·m下的最大剪应力: {tau_max['result']:.2f} MPa")
    print(f"铝管许用扭矩: {T_allow_hollow['result']/1e6:.2f} kN·m")
    print(f"FINAL_ANSWER: 许用扭矩 {T_allow_hollow['result']/1e6:.2f} kN·m")


if __name__ == "__main__":
    main()