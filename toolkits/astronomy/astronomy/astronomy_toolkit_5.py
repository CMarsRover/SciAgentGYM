# Filename: astronomy_toolkit.py
"""
天文学计算工具包

主要功能：
1. 恒星距离计算：基于距离模数公式和星际消光修正
2. 星际消光分析：计算不同波段的消光效应
3. 恒星参数批量处理：支持多恒星数据的批量分析和排序

依赖库：
pip install numpy scipy astropy matplotlib plotly pandas
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import os
from datetime import datetime
import json

# 导入天文学专属库
from astropy import units as u
from astropy.coordinates import Distance
from astropy.table import Table
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# 全局常量
RV_STANDARD = 3.1  # 标准消光比值
DISTANCE_MODULUS_CONSTANT = 5.0  # 距离模数常数
PC_TO_LY = 3.26156  # 秒差距到光年的转换因子

# 创建中间结果保存目录
os.makedirs('./mid_result/astronomy', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)


# ============ 第一层：原子工具函数（Atomic Tools） ============

def calculate_extinction(e_bv: float, rv: float = RV_STANDARD) -> dict:
    """
    计算V波段的星际消光量
    
    物理原理：星际尘埃导致星光被吸收和散射，消光量与色余成正比。
    公式：A_V = R_V × E(B-V)
    
    Args:
        e_bv: B-V色余（单位：mag），范围通常0-2
        rv: 消光比值（无量纲），默认3.1（银河系标准值）
    
    Returns:
        dict: {
            'result': V波段消光量（mag）,
            'metadata': {
                'e_bv': 输入色余,
                'rv': 使用的消光比值,
                'formula': 使用的公式
            }
        }
    
    Example:
        >>> result = calculate_extinction(0.2, 3.1)
        >>> print(result['result'])
        0.62
    """
    # === 完整边界检查 ===
    if not isinstance(e_bv, (int, float)):
        raise TypeError(f"e_bv必须是数值类型，当前类型：{type(e_bv)}")
    if not isinstance(rv, (int, float)):
        raise TypeError(f"rv必须是数值类型，当前类型：{type(rv)}")
    if e_bv < 0:
        raise ValueError(f"色余不能为负值：e_bv={e_bv}")
    if e_bv > 5:
        raise Warning(f"色余值异常大：e_bv={e_bv}，请检查数据")
    if rv <= 0:
        raise ValueError(f"消光比值必须为正：rv={rv}")
    
    # === 核心计算 ===
    a_v = rv * e_bv
    
    return {
        'result': a_v,
        'metadata': {
            'e_bv': e_bv,
            'rv': rv,
            'formula': 'A_V = R_V × E(B-V)'
        }
    }


def calculate_distance_modulus(m_obs: float, M_abs: float, a_v: float = 0.0) -> dict:
    """
    计算考虑消光修正的距离模数
    
    物理原理：视星等、绝对星等和距离的关系，需扣除星际消光影响。
    公式：m - M = 5×log10(d) - 5 + A_V  =>  μ = m - M - A_V
    
    Args:
        m_obs: 观测视星等（mag），范围通常-2到30
        M_abs: 绝对星等（mag），范围通常-10到20
        a_v: V波段消光量（mag），默认0（无消光）
    
    Returns:
        dict: {
            'result': 距离模数（mag）,
            'metadata': {
                'm_obs': 观测星等,
                'M_abs': 绝对星等,
                'a_v': 消光量,
                'formula': 使用的公式
            }
        }
    
    Example:
        >>> result = calculate_distance_modulus(7.0, 8.0, 0.62)
        >>> print(result['result'])
        -1.62
    """
    # === 完整边界检查 ===
    if not isinstance(m_obs, (int, float)):
        raise TypeError(f"m_obs必须是数值类型，当前类型：{type(m_obs)}")
    if not isinstance(M_abs, (int, float)):
        raise TypeError(f"M_abs必须是数值类型，当前类型：{type(M_abs)}")
    if not isinstance(a_v, (int, float)):
        raise TypeError(f"a_v必须是数值类型，当前类型：{type(a_v)}")
    if a_v < 0:
        raise ValueError(f"消光量不能为负：a_v={a_v}")
    
    # === 核心计算 ===
    mu = m_obs - M_abs - a_v
    
    return {
        'result': mu,
        'metadata': {
            'm_obs': m_obs,
            'M_abs': M_abs,
            'a_v': a_v,
            'formula': 'μ = m - M - A_V'
        }
    }


def distance_from_modulus(mu: float) -> dict:
    """
    从距离模数计算距离（秒差距）
    
    物理原理：距离模数与距离的对数关系。
    公式：d = 10^((μ + 5) / 5) pc
    
    Args:
        mu: 距离模数（mag），范围通常-5到20
    
    Returns:
        dict: {
            'result': 距离（pc）,
            'metadata': {
                'mu': 距离模数,
                'distance_ly': 距离（光年）,
                'formula': 使用的公式
            }
        }
    
    Example:
        >>> result = distance_from_modulus(0.0)
        >>> print(result['result'])
        10.0
    """
    # === 完整边界检查 ===
    if not isinstance(mu, (int, float)):
        raise TypeError(f"mu必须是数值类型，当前类型：{type(mu)}")
    if mu < -10:
        raise Warning(f"距离模数异常小：mu={mu}，对应距离<1pc")
    if mu > 30:
        raise Warning(f"距离模数异常大：mu={mu}，对应距离>100kpc")
    
    # === 核心计算 ===
    distance_pc = 10 ** ((mu + DISTANCE_MODULUS_CONSTANT) / DISTANCE_MODULUS_CONSTANT)
    distance_ly = distance_pc * PC_TO_LY
    
    return {
        'result': distance_pc,
        'metadata': {
            'mu': mu,
            'distance_pc': distance_pc,
            'distance_ly': distance_ly,
            'formula': 'd = 10^((μ + 5) / 5) pc'
        }
    }


def validate_solar_neighborhood(distance_pc: float, threshold: float = 500.0) -> dict:
    """
    验证恒星是否在太阳邻域内
    
    定义：太阳邻域通常指距离太阳<500pc的区域。
    
    Args:
        distance_pc: 恒星距离（pc）
        threshold: 太阳邻域阈值（pc），默认500
    
    Returns:
        dict: {
            'result': 是否在太阳邻域（bool）,
            'metadata': {
                'distance_pc': 距离,
                'threshold': 阈值,
                'distance_ratio': 距离与阈值的比值
            }
        }
    
    Example:
        >>> result = validate_solar_neighborhood(100.0)
        >>> print(result['result'])
        True
    """
    # === 完整边界检查 ===
    if not isinstance(distance_pc, (int, float)):
        raise TypeError(f"distance_pc必须是数值类型")
    if distance_pc < 0:
        raise ValueError(f"距离不能为负：{distance_pc}")
    if threshold <= 0:
        raise ValueError(f"阈值必须为正：{threshold}")
    
    # === 核心计算 ===
    is_in_neighborhood = distance_pc < threshold
    ratio = distance_pc / threshold
    
    return {
        'result': is_in_neighborhood,
        'metadata': {
            'distance_pc': distance_pc,
            'threshold': threshold,
            'distance_ratio': ratio,
            'status': 'in_neighborhood' if is_in_neighborhood else 'beyond_neighborhood'
        }
    }


# ============ 第二层：组合工具函数（Composite Tools） ============

def calculate_stellar_distance(m_obs: float, M_abs: float, e_bv: float = 0.0, 
                               rv: float = RV_STANDARD) -> dict:
    """
    计算恒星距离的完整流程（考虑星际消光）
    
    物理流程：
    1. 计算V波段消光量 A_V = R_V × E(B-V)
    2. 计算距离模数 μ = m - M - A_V
    3. 计算距离 d = 10^((μ + 5) / 5) pc
    
    Args:
        m_obs: 观测视星等（mag）
        M_abs: 绝对星等（mag）
        e_bv: B-V色余（mag），默认0
        rv: 消光比值，默认3.1
    
    Returns:
        dict: {
            'result': 距离（pc）,
            'metadata': {
                'extinction': 消光计算结果,
                'modulus': 距离模数计算结果,
                'distance_details': 距离详细信息
            }
        }
    
    Example:
        >>> result = calculate_stellar_distance(7.0, 8.0, 0.2, 3.1)
        >>> print(result['result'])
        4.74
    """
    # === 参数完全可序列化检查 ===
    if not all(isinstance(x, (int, float)) for x in [m_obs, M_abs, e_bv, rv]):
        raise TypeError("所有参数必须是数值类型")
    
    # === 步骤1：计算消光量 ===
    ## using calculate_extinction, and get extinction result
    extinction_result = calculate_extinction(e_bv, rv)
    a_v = extinction_result['result']
    
    # === 步骤2：计算距离模数 ===
    ## using calculate_distance_modulus, and get modulus result
    modulus_result = calculate_distance_modulus(m_obs, M_abs, a_v)
    mu = modulus_result['result']
    
    # === 步骤3：计算距离 ===
    ## using distance_from_modulus, and get distance result
    distance_result = distance_from_modulus(mu)
    distance_pc = distance_result['result']
    
    return {
        'result': distance_pc,
        'metadata': {
            'extinction': extinction_result,
            'modulus': modulus_result,
            'distance_details': distance_result['metadata'],
            'calculation_steps': [
                f"Step 1: A_V = {a_v:.3f} mag",
                f"Step 2: μ = {mu:.3f} mag",
                f"Step 3: d = {distance_pc:.3f} pc"
            ]
        }
    }


def analyze_star_sample(stars_data: List[Dict]) -> dict:
    """
    批量分析多颗恒星的距离并排序
    
    功能：
    1. 批量计算每颗恒星的距离
    2. 按距离排序
    3. 生成统计摘要
    4. 保存详细结果到文件
    
    Args:
        stars_data: 恒星数据列表，每个元素为字典，包含键：
                   'id': 恒星标识（str）
                   'm_obs': 观测星等（float）
                   'M_abs': 绝对星等（float）
                   'e_bv': 色余（float，可选，默认0）
                   'rv': 消光比值（float，可选，默认3.1）
    
    Returns:
        dict: {
            'result': 排序后的恒星列表（按距离从近到远）,
            'metadata': {
                'total_stars': 恒星总数,
                'distance_range': 距离范围,
                'file_path': 保存的详细结果文件路径
            }
        }
    
    Example:
        >>> stars = [
        ...     {'id': 'a', 'm_obs': 8, 'M_abs': 8, 'e_bv': 0},
        ...     {'id': 'b', 'm_obs': 7, 'M_abs': 8, 'e_bv': 0}
        ... ]
        >>> result = analyze_star_sample(stars)
        >>> print(result['result'][0]['id'])
        'b'
    """
    # === 参数完全可序列化检查 ===
    if not isinstance(stars_data, list):
        raise TypeError("stars_data必须是列表")
    if len(stars_data) == 0:
        raise ValueError("stars_data不能为空")
    
    required_keys = {'id', 'm_obs', 'M_abs'}
    for i, star in enumerate(stars_data):
        if not isinstance(star, dict):
            raise TypeError(f"第{i}个元素必须是字典")
        if not required_keys.issubset(star.keys()):
            raise ValueError(f"第{i}个元素缺少必需键：{required_keys - star.keys()}")
    
    # === 批量计算距离 ===
    results = []
    for star in stars_data:
        star_id = star['id']
        m_obs = star['m_obs']
        M_abs = star['M_abs']
        e_bv = star.get('e_bv', 0.0)
        rv = star.get('rv', RV_STANDARD)
        
        ## using calculate_stellar_distance, and get distance for each star
        distance_result = calculate_stellar_distance(m_obs, M_abs, e_bv, rv)
        
        results.append({
            'id': star_id,
            'distance_pc': distance_result['result'],
            'm_obs': m_obs,
            'M_abs': M_abs,
            'e_bv': e_bv,
            'a_v': distance_result['metadata']['extinction']['result'],
            'mu': distance_result['metadata']['modulus']['result']
        })
    
    # === 按距离排序 ===
    sorted_results = sorted(results, key=lambda x: x['distance_pc'])
    
    # === 统计分析 ===
    distances = [r['distance_pc'] for r in sorted_results]
    distance_range = {
        'min': min(distances),
        'max': max(distances),
        'mean': np.mean(distances),
        'median': np.median(distances)
    }
    
    # === 保存详细结果 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'./mid_result/astronomy/star_analysis_{timestamp}.json'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'sorted_stars': sorted_results,
            'statistics': distance_range,
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False)
    
    return {
        'result': sorted_results,
        'metadata': {
            'total_stars': len(sorted_results),
            'distance_range': distance_range,
            'file_path': filepath,
            'sorted_order': [s['id'] for s in sorted_results]
        }
    }


def compare_extinction_effects(m_obs: float, M_abs: float, 
                               e_bv_values: List[float]) -> dict:
    """
    比较不同消光条件下的距离差异
    
    物理意义：展示星际消光对距离测量的影响程度
    
    Args:
        m_obs: 观测星等（float）
        M_abs: 绝对星等（float）
        e_bv_values: 色余值列表（list of float）
    
    Returns:
        dict: {
            'result': 比较结果列表,
            'metadata': {
                'distance_variation': 距离变化统计,
                'file_path': 保存的结果文件
            }
        }
    
    Example:
        >>> result = compare_extinction_effects(7.0, 8.0, [0.0, 0.1, 0.2])
        >>> print(len(result['result']))
        3
    """
    # === 参数检查 ===
    if not isinstance(e_bv_values, list):
        raise TypeError("e_bv_values必须是列表")
    if not all(isinstance(x, (int, float)) for x in e_bv_values):
        raise TypeError("e_bv_values中所有元素必须是数值")
    
    # === 批量计算 ===
    comparison_results = []
    for e_bv in e_bv_values:
        ## using calculate_stellar_distance for each E(B-V) value
        dist_result = calculate_stellar_distance(m_obs, M_abs, e_bv)
        
        comparison_results.append({
            'e_bv': e_bv,
            'a_v': dist_result['metadata']['extinction']['result'],
            'distance_pc': dist_result['result'],
            'distance_ly': dist_result['result'] * PC_TO_LY
        })
    
    # === 统计分析 ===
    distances = [r['distance_pc'] for r in comparison_results]
    variation = {
        'min_distance': min(distances),
        'max_distance': max(distances),
        'distance_range': max(distances) - min(distances),
        'relative_change': (max(distances) - min(distances)) / min(distances) * 100
    }
    
    # === 保存结果 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'./mid_result/astronomy/extinction_comparison_{timestamp}.json'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'input_parameters': {'m_obs': m_obs, 'M_abs': M_abs},
            'comparison_results': comparison_results,
            'statistics': variation
        }, f, indent=2)
    
    return {
        'result': comparison_results,
        'metadata': {
            'distance_variation': variation,
            'file_path': filepath
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def visualize_distance_distribution(stars_data: List[Dict], 
                                    save_dir: str = './tool_images/',
                                    filename: str = None) -> dict:
    """
    可视化恒星距离分布
    
    生成图表：
    1. 距离柱状图（按恒星ID排序）
    2. 距离-星等关系散点图
    
    Args:
        stars_data: 恒星数据列表（来自analyze_star_sample的result）
        save_dir: 保存目录
        filename: 文件名（不含扩展名），默认自动生成
    
    Returns:
        dict: {
            'result': 保存的图片路径,
            'metadata': {
                'plot_type': 图表类型,
                'num_stars': 恒星数量
            }
        }
    """
    # === 参数检查 ===
    if not isinstance(stars_data, list) or len(stars_data) == 0:
        raise ValueError("stars_data必须是非空列表")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'distance_distribution_{timestamp}'
    
    # === 数据准备 ===
    ids = [s['id'] for s in stars_data]
    distances = [s['distance_pc'] for s in stars_data]
    m_obs_values = [s['m_obs'] for s in stars_data]
    
    # === 创建图表 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 子图1：距离柱状图
    colors = plt.cm.viridis(np.linspace(0, 1, len(ids)))
    ax1.bar(ids, distances, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Star ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance (pc)', fontsize=12, fontweight='bold')
    ax1.set_title('Stellar Distance Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (id_val, dist) in enumerate(zip(ids, distances)):
        ax1.text(i, dist + max(distances)*0.02, f'{dist:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 子图2：距离-星等关系
    scatter = ax2.scatter(m_obs_values, distances, c=distances, 
                         cmap='plasma', s=150, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Observed Magnitude (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance (pc)', fontsize=12, fontweight='bold')
    ax2.set_title('Distance vs Observed Magnitude', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Distance (pc)', fontsize=10)
    
    # 添加恒星ID标签
    for id_val, m, d in zip(ids, m_obs_values, distances):
        ax2.annotate(id_val, (m, d), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    
    # === 保存图片 ===
    save_path = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Plot | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'plot_type': 'distance_distribution',
            'num_stars': len(stars_data),
            'file_format': 'png',
            'dpi': 300
        }
    }


def visualize_extinction_impact(comparison_data: List[Dict],
                                save_dir: str = './tool_images/',
                                filename: str = None) -> dict:
    """
    可视化消光对距离测量的影响
    
    生成图表：消光量-距离关系曲线
    
    Args:
        comparison_data: 消光比较数据（来自compare_extinction_effects的result）
        save_dir: 保存目录
        filename: 文件名
    
    Returns:
        dict: {
            'result': 保存的图片路径,
            'metadata': {
                'plot_type': 图表类型
            }
        }
    """
    # === 参数检查 ===
    if not isinstance(comparison_data, list) or len(comparison_data) == 0:
        raise ValueError("comparison_data必须是非空列表")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'extinction_impact_{timestamp}'
    
    # === 数据准备 ===
    e_bv_values = [d['e_bv'] for d in comparison_data]
    a_v_values = [d['a_v'] for d in comparison_data]
    distances = [d['distance_pc'] for d in comparison_data]
    
    # === 创建图表 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 子图1：E(B-V) vs Distance
    ax1.plot(e_bv_values, distances, marker='o', linewidth=2.5, 
            markersize=10, color='#2E86AB', markerfacecolor='#A23B72')
    ax1.set_xlabel('E(B-V) (mag)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance (pc)', fontsize=12, fontweight='bold')
    ax1.set_title('Impact of Color Excess on Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 子图2：A_V vs Distance
    ax2.plot(a_v_values, distances, marker='s', linewidth=2.5,
            markersize=10, color='#F18F01', markerfacecolor='#C73E1D')
    ax2.set_xlabel('A_V (mag)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance (pc)', fontsize=12, fontweight='bold')
    ax2.set_title('Impact of V-band Extinction on Distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # === 保存图片 ===
    save_path = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Plot | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'plot_type': 'extinction_impact',
            'num_points': len(comparison_data),
            'file_format': 'png'
        }
    }


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【恒星距离排序问题】+【至少2个相关场景】
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - 恒星距离排序")
    print("=" * 60)
    print("问题描述：给定6颗恒星的观测星等、绝对星等和色余，计算它们到太阳的距离并排序")
    print("-" * 60)
    
    # 定义6颗恒星的数据
    stars_original = [
        {'id': 'a', 'm_obs': 8, 'M_abs': 8, 'e_bv': 0.0},
        {'id': 'b', 'm_obs': 7, 'M_abs': 8, 'e_bv': 0.0},
        {'id': 'c', 'm_obs': 9, 'M_abs': 8, 'e_bv': 0.0},
        {'id': 'd', 'm_obs': 7, 'M_abs': 8, 'e_bv': 0.2},
        {'id': 'e', 'm_obs': 7, 'M_abs': 7, 'e_bv': 0.2},
        {'id': 'f', 'm_obs': 7, 'M_abs': 7, 'e_bv': 0.0}
    ]
    
    # 步骤1：批量计算恒星距离并排序
    # 调用函数：analyze_star_sample()，该函数内部调用了 calculate_stellar_distance()
    print("\n步骤1：批量计算恒星距离...")
    result1 = analyze_star_sample(stars_original)
    sorted_stars = result1['result']
    
    print(f"FUNCTION_CALL: analyze_star_sample | PARAMS: 6 stars | RESULT: sorted by distance")
    print(f"\n距离排序结果（从近到远）：")
    for i, star in enumerate(sorted_stars, 1):
        print(f"  {i}. Star {star['id']}: {star['distance_pc']:.2f} pc "
              f"(m={star['m_obs']}, M={star['M_abs']}, E(B-V)={star['e_bv']}, A_V={star['a_v']:.2f})")
    
    # 步骤2：验证答案
    # 调用函数：提取排序结果
    print("\n步骤2：验证排序顺序...")
    sorted_order = ' < '.join([s['id'] for s in sorted_stars])
    expected_order = "d < b < f < c"
    
    print(f"FUNCTION_CALL: extract_order | PARAMS: sorted_stars | RESULT: {sorted_order}")
    print(f"计算得到的排序：{sorted_order}")
    print(f"预期答案：{expected_order}")
    
    # 提取题目要求的4颗星
    required_stars = ['d', 'b', 'f', 'c']
    filtered_order = ' < '.join([s['id'] for s in sorted_stars if s['id'] in required_stars])
    
    print(f"✓ 场景1完成：题目要求的4颗星排序为 {filtered_order}")
    print(f"FINAL_ANSWER: {filtered_order}")
    
    # 步骤3：可视化结果
    # 调用函数：visualize_distance_distribution()
    print("\n步骤3：生成可视化图表...")
    vis_result1 = visualize_distance_distribution(sorted_stars, filename='scenario1_distance_dist')
    print(f"FUNCTION_CALL: visualize_distance_distribution | PARAMS: 6 stars | RESULT: {vis_result1['result']}")
    
    print("\n" + "=" * 60)
    print("场景2：参数扫描 - 消光效应分析")
    print("=" * 60)
    print("问题描述：固定观测星等和绝对星等，研究不同色余值对距离测量的影响")
    print("-" * 60)
    
    # 步骤1：设置参数扫描范围
    # 调用函数：compare_extinction_effects()
    print("\n步骤1：扫描色余参数范围...")
    m_test = 7.0
    M_test = 8.0
    e_bv_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    result2 = compare_extinction_effects(m_test, M_test, e_bv_range)
    comparison_data = result2['result']
    
    print(f"FUNCTION_CALL: compare_extinction_effects | PARAMS: m={m_test}, M={M_test}, E(B-V)={e_bv_range} | RESULT: {len(comparison_data)} points")
    print(f"\n消光效应分析结果：")
    for data in comparison_data:
        print(f"  E(B-V)={data['e_bv']:.2f} → A_V={data['a_v']:.2f} mag → d={data['distance_pc']:.2f} pc")
    
    # 步骤2：统计分析
    variation = result2['metadata']['distance_variation']
    print(f"\n距离变化统计：")
    print(f"  最小距离：{variation['min_distance']:.2f} pc (E(B-V)={e_bv_range[-1]})")
    print(f"  最大距离：{variation['max_distance']:.2f} pc (E(B-V)={e_bv_range[0]})")
    print(f"  距离范围：{variation['distance_range']:.2f} pc")
    print(f"  相对变化：{variation['relative_change']:.1f}%")
    
    # 步骤3：可视化消光影响
    # 调用函数：visualize_extinction_impact()
    print("\n步骤2：生成消光影响可视化...")
    vis_result2 = visualize_extinction_impact(comparison_data, filename='scenario2_extinction_impact')
    print(f"FUNCTION_CALL: visualize_extinction_impact | PARAMS: {len(comparison_data)} points | RESULT: {vis_result2['result']}")
    
    print(f"✓ 场景2完成：消光使距离测量产生 {variation['relative_change']:.1f}% 的系统误差")
    
    print("\n" + "=" * 60)
    print("场景3：太阳邻域恒星筛选")
    print("=" * 60)
    print("问题描述：从恒星样本中筛选出太阳邻域（<500pc）内的恒星")
    print("-" * 60)
    
    # 步骤1：验证每颗星是否在太阳邻域
    # 调用函数：validate_solar_neighborhood()
    print("\n步骤1：验证恒星是否在太阳邻域...")
    neighborhood_results = []
    for star in sorted_stars:
        # 调用函数：validate_solar_neighborhood()
        validation = validate_solar_neighborhood(star['distance_pc'])
        neighborhood_results.append({
            'id': star['id'],
            'distance_pc': star['distance_pc'],
            'in_neighborhood': validation['result'],
            'distance_ratio': validation['metadata']['distance_ratio']
        })
        
        status = "✓ 在太阳邻域内" if validation['result'] else "✗ 超出太阳邻域"
        print(f"FUNCTION_CALL: validate_solar_neighborhood | PARAMS: d={star['distance_pc']:.2f} pc | RESULT: {status}")
        print(f"  Star {star['id']}: {star['distance_pc']:.2f} pc ({star['distance_pc']/500*100:.1f}% of threshold) - {status}")
    
    # 步骤2：统计分析
    in_neighborhood = [r for r in neighborhood_results if r['in_neighborhood']]
    print(f"\n步骤2：统计分析...")
    print(f"FUNCTION_CALL: filter_neighborhood | PARAMS: 6 stars | RESULT: {len(in_neighborhood)} in neighborhood")
    print(f"太阳邻域内恒星：{len(in_neighborhood)}/{len(neighborhood_results)}")
    print(f"邻域内恒星ID：{[r['id'] for r in in_neighborhood]}")
    
    print(f"✓ 场景3完成：所有6颗恒星均在太阳邻域内（<500pc）")
    
    print("\n" + "=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始问题的完整流程（恒星距离排序）")
    print("- 场景2展示了工具的参数泛化能力（消光效应参数扫描）")
    print("- 场景3展示了工具的条件筛选能力（太阳邻域验证）")
    print(f"\n最终答案验证：{filtered_order} {'✓ 正确' if filtered_order == expected_order else '✗ 错误'}")
    print(f"FINAL_ANSWER: {filtered_order}")


if __name__ == "__main__":
    main()