# Filename: nucleation_thermodynamics_toolkit.py

"""
晶体成核热力学分析工具包
用于分析晶核形成过程中的吉布斯自由能变化与温度关系
基于经典成核理论（Classical Nucleation Theory, CNT）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar
from typing import Dict, List, Tuple, Optional
import json
import os

# 配置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/materials', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def calculate_volume_free_energy_change(radius: float, delta_gv: float) -> Dict:
    """
    计算体积自由能变化（成核驱动力）
    
    ΔG_volume = (4/3)πr³ΔG_v
    其中ΔG_v < 0 表示固相比液相更稳定
    
    Args:
        radius: 晶核半径 (nm)
        delta_gv: 单位体积自由能变化 (J/m³)，通常为负值
        
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if radius < 0:
        raise ValueError(f"半径必须为非负值，当前值: {radius}")
    
    volume = (4/3) * np.pi * (radius ** 3)
    delta_g_volume = volume * delta_gv
    
    return {
        'result': delta_g_volume,
        'metadata': {
            'radius_nm': radius,
            'volume_nm3': volume,
            'delta_gv_J_per_m3': delta_gv,
            'unit': 'J',
            'description': '体积自由能变化（驱动成核）'
        }
    }


def calculate_surface_free_energy(radius: float, surface_tension: float) -> Dict:
    """
    计算表面自由能（成核阻力）
    
    ΔG_surface = 4πr²σ
    其中σ是液固界面张力
    
    Args:
        radius: 晶核半径 (nm)
        surface_tension: 界面张力 (J/m²)
        
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if radius < 0:
        raise ValueError(f"半径必须为非负值，当前值: {radius}")
    if surface_tension < 0:
        raise ValueError(f"界面张力必须为非负值，当前值: {surface_tension}")
    
    surface_area = 4 * np.pi * (radius ** 2)
    delta_g_surface = surface_area * surface_tension
    
    return {
        'result': delta_g_surface,
        'metadata': {
            'radius_nm': radius,
            'surface_area_nm2': surface_area,
            'surface_tension_J_per_m2': surface_tension,
            'unit': 'J',
            'description': '表面自由能（阻碍成核）'
        }
    }


def calculate_total_gibbs_free_energy(radius: float, delta_gv: float, 
                                     surface_tension: float) -> Dict:
    """
    计算总吉布斯自由能变化
    
    ΔG(r) = (4/3)πr³ΔG_v + 4πr²σ
    
    Args:
        radius: 晶核半径 (nm)
        delta_gv: 单位体积自由能变化 (J/m³)
        surface_tension: 界面张力 (J/m²)
        
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    volume_term = calculate_volume_free_energy_change(radius, delta_gv)
    surface_term = calculate_surface_free_energy(radius, surface_tension)
    
    total_delta_g = volume_term['result'] + surface_term['result']
    
    return {
        'result': total_delta_g,
        'metadata': {
            'radius_nm': radius,
            'volume_contribution_J': volume_term['result'],
            'surface_contribution_J': surface_term['result'],
            'total_delta_g_J': total_delta_g,
            'unit': 'J',
            'description': '总吉布斯自由能变化'
        }
    }


def calculate_critical_radius(delta_gv: float, surface_tension: float) -> Dict:
    """
    计算临界晶核半径
    
    r* = -2σ/ΔG_v
    
    临界半径是ΔG(r)曲线的极大值点
    
    Args:
        delta_gv: 单位体积自由能变化 (J/m³)，必须为负值
        surface_tension: 界面张力 (J/m²)
        
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if delta_gv >= 0:
        raise ValueError(f"ΔG_v必须为负值（固相更稳定），当前值: {delta_gv}")
    if surface_tension <= 0:
        raise ValueError(f"界面张力必须为正值，当前值: {surface_tension}")
    
    r_critical = -2 * surface_tension / delta_gv
    
    return {
        'result': r_critical,
        'metadata': {
            'r_critical_nm': r_critical,
            'delta_gv_J_per_m3': delta_gv,
            'surface_tension_J_per_m2': surface_tension,
            'unit': 'nm',
            'description': '临界晶核半径'
        }
    }


def calculate_critical_free_energy(delta_gv: float, surface_tension: float) -> Dict:
    """
    计算临界成核能垒
    
    ΔG* = (16πσ³)/(3ΔG_v²)
    
    这是成核必须克服的能量障碍
    
    Args:
        delta_gv: 单位体积自由能变化 (J/m³)
        surface_tension: 界面张力 (J/m²)
        
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if delta_gv >= 0:
        raise ValueError(f"ΔG_v必须为负值，当前值: {delta_gv}")
    
    delta_g_critical = (16 * np.pi * (surface_tension ** 3)) / (3 * (delta_gv ** 2))
    
    return {
        'result': delta_g_critical,
        'metadata': {
            'delta_g_critical_J': delta_g_critical,
            'delta_gv_J_per_m3': delta_gv,
            'surface_tension_J_per_m2': surface_tension,
            'unit': 'J',
            'description': '临界成核能垒'
        }
    }


def calculate_delta_gv_from_temperature(temperature: float, melting_point: float, 
                                       latent_heat: float) -> Dict:
    """
    根据温度计算单位体积自由能变化
    
    ΔG_v ≈ -L_f * ΔT / (T_m * V_m)
    其中ΔT = T_m - T（过冷度）
    
    Args:
        temperature: 当前温度 (K)
        melting_point: 熔点温度 (K)
        latent_heat: 单位体积潜热 (J/m³)
        
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if temperature <= 0:
        raise ValueError(f"温度必须为正值，当前值: {temperature}")
    if melting_point <= 0:
        raise ValueError(f"熔点必须为正值，当前值: {melting_point}")
    
    undercooling = melting_point - temperature
    delta_gv = -latent_heat * undercooling / melting_point
    
    return {
        'result': delta_gv,
        'metadata': {
            'temperature_K': temperature,
            'melting_point_K': melting_point,
            'undercooling_K': undercooling,
            'latent_heat_J_per_m3': latent_heat,
            'delta_gv_J_per_m3': delta_gv,
            'unit': 'J/m³',
            'description': '单位体积自由能变化'
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_nucleation_curve(temperature: float, melting_point: float, 
                            latent_heat: float, surface_tension: float,
                            radius_range: List[float]) -> Dict:
    """
    分析特定温度下的成核曲线
    
    Args:
        temperature: 温度 (K)
        melting_point: 熔点 (K)
        latent_heat: 单位体积潜热 (J/m³)
        surface_tension: 界面张力 (J/m²)
        radius_range: 半径范围 [r_min, r_max] (nm)
        
    Returns:
        dict: 包含曲线数据和特征参数
    """
    # 计算ΔG_v
    delta_gv_result = calculate_delta_gv_from_temperature(
        temperature, melting_point, latent_heat
    )
    delta_gv = delta_gv_result['result']
    
    # 生成半径数组
    r_min, r_max = radius_range
    radii = np.linspace(r_min, r_max, 200)
    
    # 计算每个半径对应的ΔG
    delta_g_values = []
    for r in radii:
        dg = calculate_total_gibbs_free_energy(r, delta_gv, surface_tension)
        delta_g_values.append(dg['result'])
    
    # 计算临界参数
    if delta_gv < 0:  # 只有过冷时才有临界半径
        r_crit = calculate_critical_radius(delta_gv, surface_tension)
        dg_crit = calculate_critical_free_energy(delta_gv, surface_tension)
        has_critical_point = True
    else:
        r_crit = {'result': None}
        dg_crit = {'result': None}
        has_critical_point = False
    
    return {
        'result': {
            'radii': radii.tolist(),
            'delta_g_values': delta_g_values,
            'r_critical': r_crit['result'],
            'delta_g_critical': dg_crit['result'],
            'has_critical_point': has_critical_point
        },
        'metadata': {
            'temperature_K': temperature,
            'delta_gv_J_per_m3': delta_gv,
            'undercooling_K': melting_point - temperature,
            'surface_tension_J_per_m2': surface_tension,
            'description': '成核曲线分析结果'
        }
    }


def compare_temperatures(temperatures: List[float], melting_point: float,
                        latent_heat: float, surface_tension: float,
                        radius_range: List[float]) -> Dict:
    """
    比较不同温度下的成核曲线特征
    
    Args:
        temperatures: 温度列表 (K)
        melting_point: 熔点 (K)
        latent_heat: 单位体积潜热 (J/m³)
        surface_tension: 界面张力 (J/m²)
        radius_range: 半径范围 [r_min, r_max] (nm)
        
    Returns:
        dict: 包含所有温度的比较结果
    """
    results = []
    
    for temp in temperatures:
        curve_data = analyze_nucleation_curve(
            temp, melting_point, latent_heat, surface_tension, radius_range
        )
        
        results.append({
            'temperature': temp,
            'undercooling': melting_point - temp,
            'r_critical': curve_data['result']['r_critical'],
            'delta_g_critical': curve_data['result']['delta_g_critical'],
            'has_critical_point': curve_data['result']['has_critical_point'],
            'curve_data': curve_data['result']
        })
    
    # 排序：按临界能垒从小到大
    results_sorted = sorted(
        results, 
        key=lambda x: x['delta_g_critical'] if x['delta_g_critical'] is not None else float('inf')
    )
    
    return {
        'result': results_sorted,
        'metadata': {
            'num_temperatures': len(temperatures),
            'temperature_range_K': [min(temperatures), max(temperatures)],
            'melting_point_K': melting_point,
            'description': '多温度成核曲线比较'
        }
    }


def identify_temperature_order(curve_characteristics: List[Dict]) -> Dict:
    """
    根据曲线特征识别温度高低顺序
    
    理论依据：
    1. 温度越高，过冷度越小，|ΔG_v|越小
    2. r* = -2σ/ΔG_v，|ΔG_v|越小，r*越大
    3. ΔG* = 16πσ³/(3ΔG_v²)，|ΔG_v|越小，ΔG*越大
    4. 温度高于熔点时，无临界点（曲线单调上升）
    
    Args:
        curve_characteristics: 曲线特征列表，每个元素包含：
            - 'label': 曲线标签（如'T1', 'T2', 'T3'）
            - 'has_critical_point': 是否有临界点
            - 'r_critical': 临界半径（如果有）
            - 'delta_g_critical': 临界能垒（如果有）
            
    Returns:
        dict: {'result': {'highest': str, 'lowest': str}, 'metadata': {...}}
    """
    # 分类曲线
    curves_with_critical = []
    curves_without_critical = []
    
    for curve in curve_characteristics:
        if curve['has_critical_point']:
            curves_with_critical.append(curve)
        else:
            curves_without_critical.append(curve)
    
    # 无临界点的曲线对应最高温度（T > T_m）
    if curves_without_critical:
        highest_temp_label = curves_without_critical[0]['label']
    else:
        # 如果都有临界点，选择临界能垒最大的（过冷度最小）
        curves_with_critical_sorted = sorted(
            curves_with_critical,
            key=lambda x: x['delta_g_critical'],
            reverse=True
        )
        highest_temp_label = curves_with_critical_sorted[0]['label']
    
    # 有临界点的曲线中，临界能垒最小的对应最低温度（过冷度最大）
    if curves_with_critical:
        curves_with_critical_sorted = sorted(
            curves_with_critical,
            key=lambda x: x['delta_g_critical']
        )
        lowest_temp_label = curves_with_critical_sorted[0]['label']
    else:
        lowest_temp_label = None
    
    return {
        'result': {
            'highest': highest_temp_label,
            'lowest': lowest_temp_label
        },
        'metadata': {
            'num_curves': len(curve_characteristics),
            'num_with_critical': len(curves_with_critical),
            'num_without_critical': len(curves_without_critical),
            'reasoning': {
                'highest': f"{'无临界点（T>Tm）' if curves_without_critical else '临界能垒最大（过冷度最小）'}",
                'lowest': '临界能垒最小（过冷度最大）'
            },
            'description': '温度高低顺序识别'
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_nucleation_curves(comparison_data: Dict = None, labels: List[str] = None,
                          title: str = "Gibbs Free Energy vs Nucleus Radius",
                          precomputed_results: List[Dict] = None) -> Dict:
    """
    绘制多条成核曲线
    
    Args:
        comparison_data: compare_temperatures()的返回结果
        labels: 曲线标签列表（如['T1', 'T2', 'T3']）
        title: 图表标题
        
    Returns:
        dict: {'result': 'filepath', 'metadata': {...}}
    """
    if precomputed_results is not None:
        results = precomputed_results
    else:
        results = comparison_data['result']
        if labels is None:
            labels = [f"Curve {i+1}" for i in range(len(results))]
    
    plt.figure(figsize=(10, 7))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, (result, label) in enumerate(zip(results, labels)):
        curve_data = result['curve_data']
        radii = curve_data['radii']
        delta_g = curve_data['delta_g_values']
        
        plt.plot(radii, delta_g, label=label, color=colors[i % len(colors)], linewidth=2)
        
        # 标记临界点
        if result['has_critical_point']:
            r_crit = result['r_critical']
            dg_crit = result['delta_g_critical']
            plt.plot(r_crit, dg_crit, 'o', color=colors[i % len(colors)], 
                    markersize=8, label=f'{label} critical point')
    
    plt.xlabel('Nucleus Radius r (arb. units)', fontsize=12)
    plt.ylabel('Gibbs Free Energy Change ΔG (J)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    filepath = './tool_images/nucleation_curves_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'num_curves': len(results),
            'labels': labels,
            'description': '成核曲线对比图'
        }
    }


def plot_critical_parameters_vs_temperature(temperatures: List[float],
                                           r_criticals: List[float],
                                           dg_criticals: List[float]) -> Dict:
    """
    绘制临界参数随温度变化的关系
    
    Args:
        temperatures: 温度列表 (K)
        r_criticals: 临界半径列表 (nm)
        dg_criticals: 临界能垒列表 (J)
        
    Returns:
        dict: {'result': 'filepath', 'metadata': {...}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 临界半径 vs 温度
    ax1.plot(temperatures, r_criticals, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Critical Radius r* (nm)', fontsize=12)
    ax1.set_title('Critical Radius vs Temperature', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 临界能垒 vs 温度
    ax2.plot(temperatures, dg_criticals, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('Critical Free Energy ΔG* (J)', fontsize=12)
    ax2.set_title('Nucleation Barrier vs Temperature', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = './tool_images/critical_parameters_vs_temperature.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'num_points': len(temperatures),
            'temperature_range_K': [min(temperatures), max(temperatures)],
            'description': '临界参数随温度变化图'
        }
    }


def save_analysis_report(temperature_order: Dict, comparison_data: Dict,
                        labels: List[str]) -> Dict:
    """
    保存分析报告到JSON文件
    
    Args:
        temperature_order: identify_temperature_order()的返回结果
        comparison_data: compare_temperatures()的返回结果
        labels: 曲线标签列表
        
    Returns:
        dict: {'result': 'filepath', 'metadata': {...}}
    """
    report = {
        'temperature_identification': {
            'highest_temperature': temperature_order['result']['highest'],
            'lowest_temperature': temperature_order['result']['lowest'],
            'reasoning': temperature_order['metadata']['reasoning']
        },
        'curve_analysis': []
    }
    
    results = comparison_data['result']
    for result, label in zip(results, labels):
        curve_info = {
            'label': label,
            'temperature_K': result['temperature'],
            'undercooling_K': result['undercooling'],
            'has_critical_point': result['has_critical_point'],
            'r_critical_nm': result['r_critical'],
            'delta_g_critical_J': result['delta_g_critical']
        }
        report['curve_analysis'].append(curve_info)
    
    filepath = './mid_result/materials/nucleation_analysis_report.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"FILE_GENERATED: json | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'json',
            'num_curves': len(labels),
            'description': '成核分析报告'
        }
    }


# ==================== 主函数：三个场景演示 ====================

def main():
    """
    演示三个场景：
    1. 解决原始问题：识别图中三条曲线的温度高低
    2. 定量分析：计算不同温度下的临界参数
    3. 参数敏感性分析：研究界面张力对成核的影响
    """
    
    # ========== 场景1：解决原始问题 ==========
    print("=" * 60)
    print("场景1：识别成核曲线的温度高低顺序")
    print("=" * 60)
    print("问题描述：根据ΔG-r曲线特征，判断T1、T2、T3哪个温度最高、哪个最低")
    print("-" * 60)
    
    # 步骤1：根据图片描述提取曲线特征
    print("\n步骤1：提取曲线特征")
    curve_features = [
        {
            'label': 'T1',
            'has_critical_point': False,  # 单调上升，无极大值
            'r_critical': None,
            'delta_g_critical': None
        },
        {
            'label': 'T2',
            'has_critical_point': True,  # 有极大值点
            'r_critical': 7.5,  # 较大的临界半径
            'delta_g_critical': 5.0e-19  # 较高的能垒
        },
        {
            'label': 'T3',
            'has_critical_point': True,  # 有极大值点
            'r_critical': 4.5,  # 较小的临界半径
            'delta_g_critical': 2.0e-19  # 较低的能垒
        }
    ]
    print(f"提取的曲线特征: {json.dumps(curve_features, indent=2)}")
    
    # 步骤2：识别温度顺序
    # 调用函数：identify_temperature_order()
    print("\n步骤2：根据成核理论识别温度顺序")
    temp_order = identify_temperature_order(curve_features)
    print(f"FUNCTION_CALL: identify_temperature_order | PARAMS: {curve_features} | RESULT: {temp_order}")
    
    highest_temp = temp_order['result']['highest']
    lowest_temp = temp_order['result']['lowest']
    
    print(f"\n分析结果：")
    print(f"  - 最高温度: {highest_temp}")
    print(f"  - 最低温度: {lowest_temp}")
    print(f"  - 理论依据: {temp_order['metadata']['reasoning']}")
    
    print(f"\nFINAL_ANSWER: 最高温度={highest_temp}, 最低温度={lowest_temp}")
    
    
    # ========== 场景2：定量计算不同温度的临界参数 ==========
    print("\n" + "=" * 60)
    print("场景2：定量计算不同温度下的临界成核参数")
    print("=" * 60)
    print("问题描述：对于某金属材料，计算不同过冷度下的临界半径和成核能垒")
    print("-" * 60)
    
    # 材料参数（假设为铝）
    melting_point = 933.0  # K
    latent_heat = 1.07e9  # J/m³
    surface_tension = 0.15  # J/m²
    
    # 三个温度：高于熔点、略低于熔点、远低于熔点
    temperatures = [950.0, 920.0, 880.0]  # K
    labels = ['T_high', 'T_medium', 'T_low']
    
    print(f"\n材料参数：")
    print(f"  - 熔点: {melting_point} K")
    print(f"  - 潜热: {latent_heat:.2e} J/m³")
    print(f"  - 界面张力: {surface_tension} J/m²")
    print(f"  - 分析温度: {temperatures} K")
    
    # 步骤1：计算各温度的ΔG_v
    print("\n步骤1：计算各温度的单位体积自由能变化")
    for temp, label in zip(temperatures, labels):
        # 调用函数：calculate_delta_gv_from_temperature()
        delta_gv_result = calculate_delta_gv_from_temperature(temp, melting_point, latent_heat)
        print(f"FUNCTION_CALL: calculate_delta_gv_from_temperature | PARAMS: {{temp={temp}, Tm={melting_point}, Lf={latent_heat}}} | RESULT: {delta_gv_result}")
        print(f"  {label}: ΔG_v = {delta_gv_result['result']:.2e} J/m³, 过冷度 = {delta_gv_result['metadata']['undercooling_K']} K")
    
    # 步骤2：生成并比较成核曲线（直接使用原子函数）
    print("\n步骤2：生成并比较成核曲线（原子函数逐点计算）")
    radius_range = [0.1, 15.0]  # nm
    radii = np.linspace(radius_range[0], radius_range[1], 200)
    
    comparison_results = []
    
    for temp, label in zip(temperatures, labels):
        delta_gv_result = calculate_delta_gv_from_temperature(temp, melting_point, latent_heat)
        delta_gv = delta_gv_result['result']
        
        delta_g_values = []
        for r in radii:
            total_g = calculate_total_gibbs_free_energy(r, delta_gv, surface_tension)
            delta_g_values.append(total_g['result'])
        
        if delta_gv < 0:
            r_crit_result = calculate_critical_radius(delta_gv, surface_tension)
            dg_crit_result = calculate_critical_free_energy(delta_gv, surface_tension)
            r_critical_value = r_crit_result['result']
            dg_critical_value = dg_crit_result['result']
            has_critical_point = True
        else:
            r_critical_value = None
            dg_critical_value = None
            has_critical_point = False
        
        comparison_results.append({
            'temperature': temp,
            'undercooling': melting_point - temp,
            'r_critical': r_critical_value,
            'delta_g_critical': dg_critical_value,
            'has_critical_point': has_critical_point,
            'curve_data': {
                'radii': radii.tolist(),
                'delta_g_values': delta_g_values
            }
        })
    
    comparison_metadata = {
        'num_temperatures': len(temperatures),
        'temperature_range_K': [min(temperatures), max(temperatures)],
        'melting_point_K': melting_point,
        'description': '多温度成核曲线比较（原子函数计算）'
    }
    
    print(f"已生成 {len(comparison_results)} 条曲线数据。")
    
    # 步骤3：提取临界参数
    print("\n步骤3：提取临界成核参数")
    for result, label in zip(comparison_results, labels):
        if result['has_critical_point']:
            print(f"  {label} (T={result['temperature']} K):")
            print(f"    - 临界半径 r* = {result['r_critical']:.3f} nm")
            print(f"    - 临界能垒 ΔG* = {result['delta_g_critical']:.3e} J")
        else:
            print(f"  {label} (T={result['temperature']} K): 无临界点（温度高于熔点）")
    
    # 步骤4：可视化
    print("\n步骤4：绘制成核曲线对比图")
    # 调用函数：plot_nucleation_curves()
    plot_result = plot_nucleation_curves(labels=labels, 
                                        title="Nucleation Curves at Different Temperatures",
                                        precomputed_results=comparison_results)
    print(f"FUNCTION_CALL: plot_nucleation_curves | PARAMS: {{'precomputed_results': 'provided', 'labels': {labels}}} | RESULT: {plot_result}")
    
    # 步骤5：绘制临界参数随温度变化
    print("\n步骤5：绘制临界参数随温度变化")
    temps_with_critical = [r['temperature'] for r in comparison_results if r['has_critical_point']]
    r_crits = [r['r_critical'] for r in comparison_results if r['has_critical_point']]
    dg_crits = [r['delta_g_critical'] for r in comparison_results if r['has_critical_point']]
    
    # 调用函数：plot_critical_parameters_vs_temperature()
    param_plot = plot_critical_parameters_vs_temperature(temps_with_critical, r_crits, dg_crits)
    print(f"FUNCTION_CALL: plot_critical_parameters_vs_temperature | PARAMS: {{temps={temps_with_critical}, r*={r_crits}, ΔG*={dg_crits}}} | RESULT: {param_plot}")
    
    print(f"\nFINAL_ANSWER: 温度越低，临界半径越小，成核能垒越低，成核越容易")
    
    
    # ========== 场景3：界面张力对成核的影响 ==========
    print("\n" + "=" * 60)
    print("场景3：研究界面张力对成核行为的影响")
    print("=" * 60)
    print("问题描述：固定温度，改变界面张力，分析对临界半径和成核能垒的影响")
    print("-" * 60)
    
    # 固定参数
    fixed_temp = 900.0  # K
    surface_tensions = [0.10, 0.15, 0.20]  # J/m²
    sigma_labels = ['σ_low', 'σ_medium', 'σ_high']
    
    print(f"\n固定参数：")
    print(f"  - 温度: {fixed_temp} K")
    print(f"  - 熔点: {melting_point} K")
    print(f"  - 过冷度: {melting_point - fixed_temp} K")
    print(f"  - 界面张力范围: {surface_tensions} J/m²")
    
    # 步骤1：计算固定温度的ΔG_v
    print("\n步骤1：计算固定温度下的ΔG_v")
    # 调用函数：calculate_delta_gv_from_temperature()
    delta_gv_fixed = calculate_delta_gv_from_temperature(fixed_temp, melting_point, latent_heat)
    print(f"FUNCTION_CALL: calculate_delta_gv_from_temperature | PARAMS: {{T={fixed_temp}, Tm={melting_point}}} | RESULT: {delta_gv_fixed}")
    print(f"  ΔG_v = {delta_gv_fixed['result']:.2e} J/m³")
    
    # 步骤2：计算不同界面张力的临界参数
    print("\n步骤2：计算不同界面张力下的临界参数")
    sigma_results = []
    for sigma, label in zip(surface_tensions, sigma_labels):
        # 调用函数：calculate_critical_radius()
        r_crit = calculate_critical_radius(delta_gv_fixed['result'], sigma)
        print(f"FUNCTION_CALL: calculate_critical_radius | PARAMS: {{ΔGv={delta_gv_fixed['result']:.2e}, σ={sigma}}} | RESULT: {r_crit}")
        
        # 调用函数：calculate_critical_free_energy()
        dg_crit = calculate_critical_free_energy(delta_gv_fixed['result'], sigma)
        print(f"FUNCTION_CALL: calculate_critical_free_energy | PARAMS: {{ΔGv={delta_gv_fixed['result']:.2e}, σ={sigma}}} | RESULT: {dg_crit}")
        
        sigma_results.append({
            'label': label,
            'sigma': sigma,
            'r_critical': r_crit['result'],
            'delta_g_critical': dg_crit['result']
        })
        
        print(f"  {label} (σ={sigma} J/m²):")
        print(f"    - r* = {r_crit['result']:.3f} nm")
        print(f"    - ΔG* = {dg_crit['result']:.3e} J")
    
    # 步骤3：生成不同界面张力的成核曲线
    print("\n步骤3：生成不同界面张力的成核曲线")
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'green', 'red']
    
    for i, (sigma, label, result) in enumerate(zip(surface_tensions, sigma_labels, sigma_results)):
        radii = np.linspace(0.1, 15.0, 200)
        delta_g_values = []
        
        for r in radii:
            # 调用函数：calculate_total_gibbs_free_energy()
            dg = calculate_total_gibbs_free_energy(r, delta_gv_fixed['result'], sigma)
            delta_g_values.append(dg['result'])
        
        plt.plot(radii, delta_g_values, label=f'{label} (σ={sigma} J/m²)', 
                color=colors[i], linewidth=2)
        plt.plot(result['r_critical'], result['delta_g_critical'], 'o', 
                color=colors[i], markersize=8)
    
    plt.xlabel('Nucleus Radius r (nm)', fontsize=12)
    plt.ylabel('Gibbs Free Energy Change ΔG (J)', fontsize=12)
    plt.title(f'Effect of Surface Tension on Nucleation (T={fixed_temp} K)', 
             fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    filepath_sigma = './tool_images/surface_tension_effect.png'
    plt.savefig(filepath_sigma, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {filepath_sigma}")
    
    # 步骤4：分析界面张力的影响规律
    print("\n步骤4：分析界面张力影响规律")
    print("  理论关系：")
    print("    - r* ∝ σ (临界半径与界面张力成正比)")
    print("    - ΔG* ∝ σ³ (成核能垒与界面张力的三次方成正比)")
    print("  实际计算验证：")
    
    for i in range(len(sigma_results) - 1):
        sigma_ratio = surface_tensions[i+1] / surface_tensions[i]
        r_ratio = sigma_results[i+1]['r_critical'] / sigma_results[i]['r_critical']
        dg_ratio = sigma_results[i+1]['delta_g_critical'] / sigma_results[i]['delta_g_critical']
        
        print(f"    σ比值 = {sigma_ratio:.2f}, r*比值 = {r_ratio:.2f}, ΔG*比值 = {dg_ratio:.2f} (理论值≈{sigma_ratio**3:.2f})")
    
    print(f"\nFINAL_ANSWER: 界面张力越大，临界半径和成核能垒越大，成核越困难")
    
    print("\n" + "=" * 60)
    print("所有场景执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()