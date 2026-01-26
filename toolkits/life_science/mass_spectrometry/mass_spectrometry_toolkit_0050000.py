# Filename: mass_spectrometry_toolkit.py
"""
质谱分析计算工具包

主要功能：
1. 同位素模式分析：基于同位素丰度计算理论分布
2. 氯原子数量判定：通过M+2峰强度比确定氯原子数
3. 质谱数据处理：峰识别、基线校正、信噪比计算

依赖库：
pip install numpy scipy matplotlib pillow
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from pathlib import Path 

# 全局常量
image_path = Path(__file__).parent.parent.parent
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 氯同位素自然丰度
CL35_ABUNDANCE = 0.7576  # 35Cl丰度
CL37_ABUNDANCE = 0.2424  # 37Cl丰度
CL_MASS_DIFF = 2.0       # 35Cl和37Cl质量差

# 碳同位素自然丰度（用于排除干扰）
C13_ABUNDANCE = 0.0107   # 13C丰度

# 理论同位素模式（氯原子数 -> M+2/M强度比）
CHLORINE_ISOTOPE_RATIOS = {
    0: 0.0,      # 无氯：无M+2峰（仅13C贡献，忽略）
    1: 0.326,    # 1个Cl：M+2/M ≈ 37Cl/35Cl = 24.24/75.76 ≈ 0.32
    2: 0.978,    # 2个Cl：M+2/M ≈ 2×(35Cl×37Cl)/(35Cl)^2 ≈ 0.98
    3: 2.12      # 3个Cl：M+2/M ≈ 2.12
}

# ============ 第一层：原子工具函数 ============

def calculate_theoretical_isotope_pattern(num_chlorine: int, num_carbon: int = 0) -> dict:
    """
    计算含氯化合物的理论同位素分布模式
    
    基于二项分布原理，计算不同氯原子数下的M、M+2、M+4峰相对强度。
    
    ### 🔧 OpenAI Function Calling 严格要求
    - [x] 所有参数为基本类型（int）
    - [x] 返回值完全可序列化（dict包含list）
    - [x] 无复杂对象传递
    
    Args:
        num_chlorine: 氯原子数量，范围0-5
        num_carbon: 碳原子数量（用于13C校正），默认0表示忽略碳贡献
    
    Returns:
        dict: {
            'result': {
                'peaks': List[Dict],  # [{'mass_shift': 0, 'intensity': 100.0}, ...]
                'M_plus_2_ratio': float  # M+2/M强度比
            },
            'metadata': {
                'num_chlorine': int,
                'calculation_method': str
            }
        }
    
    Example:
        >>> result = calculate_theoretical_isotope_pattern(1)
        >>> print(result['result']['M_plus_2_ratio'])
        0.326
    """
    # === 边界检查 ===
    if not isinstance(num_chlorine, int) or num_chlorine < 0 or num_chlorine > 5:
        raise ValueError("num_chlorine必须是0-5之间的整数")
    if not isinstance(num_carbon, int) or num_carbon < 0:
        raise ValueError("num_carbon必须是非负整数")
    
    # === 计算氯同位素分布（二项分布） ===
    peaks = []
    for k in range(num_chlorine + 1):
        # 计算含k个37Cl的组合概率
        from math import comb
        intensity = comb(num_chlorine, k) * (CL35_ABUNDANCE ** (num_chlorine - k)) * (CL37_ABUNDANCE ** k)
        peaks.append({
            'mass_shift': k * CL_MASS_DIFF,
            'intensity': intensity * 100  # 归一化为百分比
        })
    
    # === 归一化到M峰为100% ===
    max_intensity = max(p['intensity'] for p in peaks)
    for peak in peaks:
        peak['intensity'] = (peak['intensity'] / max_intensity) * 100
    
    # === 计算M+2/M比值 ===
    M_intensity = peaks[0]['intensity']  # M峰
    M_plus_2_intensity = peaks[1]['intensity'] if len(peaks) > 1 else 0.0
    ratio = M_plus_2_intensity / M_intensity if M_intensity > 0 else 0.0
    
    return {
        'result': {
            'peaks': peaks,
            'M_plus_2_ratio': round(ratio, 3)
        },
        'metadata': {
            'num_chlorine': num_chlorine,
            'num_carbon': num_carbon,
            'calculation_method': 'binomial_distribution'
        }
    }


def extract_peaks_from_spectrum(mz_values: List[float], 
                                  intensities: List[float],
                                  height_threshold: float = 5.0,
                                  prominence: float = 2.0) -> dict:
    """
    从质谱数据中提取峰位置和强度
    
    使用scipy.signal.find_peaks进行峰检测，过滤噪声和基线漂移。
    
    ### 🔧 OpenAI Function Calling 严格要求
    - [x] 参数为可序列化类型（List[float]）
    - [x] 返回值为基本类型（dict包含list）
    - [x] numpy数组仅在函数内部使用
    
    Args:
        mz_values: m/z值列表
        intensities: 相对强度列表（%）
        height_threshold: 峰高度阈值（相对强度%），默认5.0
        prominence: 峰突出度阈值，默认2.0
    
    Returns:
        dict: {
            'result': {
                'peak_mz': List[float],      # 峰的m/z值
                'peak_intensity': List[float],  # 峰的强度
                'num_peaks': int
            },
            'metadata': {
                'threshold': float,
                'total_points': int
            }
        }
    
    Example:
        >>> mz = [100.0, 101.0, 102.0, 103.0]
        >>> intensity = [100.0, 5.0, 30.0, 2.0]
        >>> result = extract_peaks_from_spectrum(mz, intensity)
        >>> print(result['result']['peak_mz'])
        [100.0, 102.0]
    """
    # === 边界检查 ===
    if not isinstance(mz_values, list) or not isinstance(intensities, list):
        raise TypeError("mz_values和intensities必须是列表")
    if len(mz_values) != len(intensities):
        raise ValueError("mz_values和intensities长度必须相同")
    if len(mz_values) == 0:
        raise ValueError("输入数据不能为空")
    if not all(isinstance(x, (int, float)) for x in mz_values + intensities):
        raise TypeError("列表元素必须是数值类型")
    
    # === 转换为numpy数组（仅内部使用） ===
    mz_array = np.array(mz_values)
    intensity_array = np.array(intensities)
    
    # === 峰检测 ===
    peak_indices, properties = find_peaks(
        intensity_array,
        height=height_threshold,
        prominence=prominence
    )
    
    # === 提取峰信息并转换为基本类型 ===
    peak_mz = mz_array[peak_indices].tolist()
    peak_intensity = intensity_array[peak_indices].tolist()
    
    return {
        'result': {
            'peak_mz': peak_mz,
            'peak_intensity': peak_intensity,
            'num_peaks': len(peak_mz)
        },
        'metadata': {
            'height_threshold': height_threshold,
            'prominence': prominence,
            'total_points': len(mz_values)
        }
    }


def find_isotope_cluster(peak_mz: List[float], 
                          peak_intensity: List[float],
                          base_mz: float,
                          mass_tolerance: float = 0.5) -> dict:
    """
    在峰列表中查找同位素簇（M, M+2, M+4等）
    
    从给定的基峰m/z值开始，查找间隔约2 Da的同位素峰。
    
    ### 🔧 OpenAI Function Calling 严格要求
    - [x] 参数为基本类型
    - [x] 返回值完全可序列化
    
    Args:
        peak_mz: 峰的m/z值列表
        peak_intensity: 峰的强度列表
        base_mz: 基峰（M峰）的m/z值
        mass_tolerance: 质量匹配容差（Da），默认0.5
    
    Returns:
        dict: {
            'result': {
                'cluster': List[Dict],  # [{'mz': float, 'intensity': float, 'label': str}, ...]
                'M_mz': float,
                'M_plus_2_mz': Optional[float],
                'M_plus_2_ratio': Optional[float]
            },
            'metadata': {
                'base_mz': float,
                'cluster_size': int
            }
        }
    
    Example:
        >>> result = find_isotope_cluster([100.0, 102.0, 104.0], [100.0, 32.0, 10.0], 100.0)
        >>> print(result['result']['M_plus_2_ratio'])
        0.32
    """
    # === 边界检查 ===
    if len(peak_mz) != len(peak_intensity):
        raise ValueError("peak_mz和peak_intensity长度必须相同")
    if not isinstance(base_mz, (int, float)):
        raise TypeError("base_mz必须是数值类型")
    
    # === 查找同位素簇 ===
    cluster = []
    
    # 查找M峰
    M_idx = None
    for i, mz in enumerate(peak_mz):
        if abs(mz - base_mz) < mass_tolerance:
            M_idx = i
            cluster.append({
                'mz': peak_mz[i],
                'intensity': peak_intensity[i],
                'label': 'M'
            })
            break
    
    if M_idx is None:
        return {
            'result': {
                'cluster': [],
                'M_mz': None,
                'M_plus_2_mz': None,
                'M_plus_2_ratio': None
            },
            'metadata': {
                'base_mz': base_mz,
                'cluster_size': 0,
                'error': 'M峰未找到'
            }
        }
    
    M_intensity = peak_intensity[M_idx]
    
    # 查找M+2, M+4, M+6峰
    for shift in [2, 4, 6]:
        target_mz = base_mz + shift
        for i, mz in enumerate(peak_mz):
            if abs(mz - target_mz) < mass_tolerance:
                cluster.append({
                    'mz': peak_mz[i],
                    'intensity': peak_intensity[i],
                    'label': f'M+{shift}'
                })
                break
    
    # === 计算M+2/M比值 ===
    M_plus_2_ratio = None
    M_plus_2_mz = None
    for peak in cluster:
        if peak['label'] == 'M+2':
            M_plus_2_mz = peak['mz']
            M_plus_2_ratio = peak['intensity'] / M_intensity if M_intensity > 0 else 0.0
            break
    
    return {
        'result': {
            'cluster': cluster,
            'M_mz': base_mz,
            'M_plus_2_mz': M_plus_2_mz,
            'M_plus_2_ratio': round(M_plus_2_ratio, 3) if M_plus_2_ratio is not None else None
        },
        'metadata': {
            'base_mz': base_mz,
            'cluster_size': len(cluster),
            'mass_tolerance': mass_tolerance
        }
    }


# ============ 第二层：组合工具函数 ============

def determine_chlorine_number_from_ratio(observed_ratio: float,
                                          max_chlorine: int = 5,
                                          tolerance: float = 0.15) -> dict:
    """
    根据观测到的M+2/M强度比判定氯原子数量（推荐，适合Function Calling）
    
    通过比较观测比值与理论比值，找到最匹配的氯原子数。
    
    ### 🔧 OpenAI Function Calling 严格要求
    - [x] 参数为基本类型（float, int）
    - [x] 内部调用calculate_theoretical_isotope_pattern()
    - [x] 返回值完全可序列化
    
    Args:
        observed_ratio: 观测到的M+2/M强度比
        max_chlorine: 最大考虑的氯原子数，默认5
        tolerance: 匹配容差，默认0.15
    
    Returns:
        dict: {
            'result': {
                'num_chlorine': int,           # 判定的氯原子数
                'confidence': str,             # 'high', 'medium', 'low'
                'theoretical_ratio': float,    # 理论比值
                'deviation': float             # 偏差
            },
            'metadata': {
                'observed_ratio': float,
                'all_candidates': List[Dict]   # 所有候选结果
            }
        }
    
    Example:
        >>> result = determine_chlorine_number_from_ratio(0.32)
        >>> print(result['result']['num_chlorine'])
        1
    """
    # === 边界检查 ===
    if not isinstance(observed_ratio, (int, float)) or observed_ratio < 0:
        raise ValueError("observed_ratio必须是非负数值")
    if not isinstance(max_chlorine, int) or max_chlorine < 0:
        raise ValueError("max_chlorine必须是非负整数")
    
    # === 步骤1: 计算所有可能氯原子数的理论比值 ===
    # 调用函数: calculate_theoretical_isotope_pattern()
    candidates = []
    for n_cl in range(max_chlorine + 1):
        theory_result = calculate_theoretical_isotope_pattern(n_cl)
        theoretical_ratio = theory_result['result']['M_plus_2_ratio']
        deviation = abs(observed_ratio - theoretical_ratio)
        
        candidates.append({
            'num_chlorine': n_cl,
            'theoretical_ratio': theoretical_ratio,
            'deviation': deviation
        })
    
    # === 步骤2: 找到偏差最小的候选 ===
    best_match = min(candidates, key=lambda x: x['deviation'])
    
    # === 步骤3: 评估置信度 ===
    if best_match['deviation'] < tolerance * 0.5:
        confidence = 'high'
    elif best_match['deviation'] < tolerance:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'result': {
            'num_chlorine': best_match['num_chlorine'],
            'confidence': confidence,
            'theoretical_ratio': best_match['theoretical_ratio'],
            'deviation': round(best_match['deviation'], 3)
        },
        'metadata': {
            'observed_ratio': observed_ratio,
            'tolerance': tolerance,
            'all_candidates': candidates
        }
    }


def analyze_spectrum_for_chlorine(mz_values: List[float],
                                    intensities: List[float],
                                    base_peak_mz: Optional[float] = None,
                                    height_threshold: float = 5.0) -> dict:
    """
    完整分析质谱数据以确定氯原子数（推荐，适合Function Calling）
    
    集成峰提取、同位素簇识别、氯原子数判定的完整流程。
    
    ### 🔧 OpenAI Function Calling 严格要求
    - [x] 参数为可序列化类型
    - [x] 内部调用extract_peaks_from_spectrum(), find_isotope_cluster(), determine_chlorine_number_from_ratio()
    - [x] 返回值完全可序列化
    
    Args:
        mz_values: m/z值列表
        intensities: 相对强度列表
        base_peak_mz: 基峰m/z值，若为None则自动选择最强峰
        height_threshold: 峰检测阈值，默认5.0
    
    Returns:
        dict: {
            'result': {
                'num_chlorine': int,
                'confidence': str,
                'M_plus_2_ratio': float,
                'isotope_cluster': List[Dict]
            },
            'metadata': {
                'base_peak_mz': float,
                'num_peaks_detected': int
            }
        }
    
    Example:
        >>> mz = [100.0, 102.0, 150.0, 152.0]
        >>> intensity = [100.0, 32.0, 80.0, 25.0]
        >>> result = analyze_spectrum_for_chlorine(mz, intensity, base_peak_mz=100.0)
        >>> print(result['result']['num_chlorine'])
        1
    """
    # === 步骤1: 提取峰 ===
    # 调用函数: extract_peaks_from_spectrum()
    peaks_result = extract_peaks_from_spectrum(mz_values, intensities, height_threshold)
    peak_mz = peaks_result['result']['peak_mz']
    peak_intensity = peaks_result['result']['peak_intensity']
    
    if len(peak_mz) == 0:
        return {
            'result': {
                'num_chlorine': 0,
                'confidence': 'low',
                'M_plus_2_ratio': 0.0,
                'isotope_cluster': []
            },
            'metadata': {
                'error': '未检测到峰',
                'num_peaks_detected': 0
            }
        }
    
    # === 步骤2: 确定基峰 ===
    if base_peak_mz is None:
        # 自动选择最强峰作为基峰
        max_idx = peak_intensity.index(max(peak_intensity))
        base_peak_mz = peak_mz[max_idx]
    
    # === 步骤3: 查找同位素簇 ===
    # 调用函数: find_isotope_cluster()
    cluster_result = find_isotope_cluster(peak_mz, peak_intensity, base_peak_mz)
    
    if cluster_result['result']['M_plus_2_ratio'] is None:
        return {
            'result': {
                'num_chlorine': 0,
                'confidence': 'low',
                'M_plus_2_ratio': 0.0,
                'isotope_cluster': cluster_result['result']['cluster']
            },
            'metadata': {
                'base_peak_mz': base_peak_mz,
                'error': '未找到M+2峰',
                'num_peaks_detected': len(peak_mz)
            }
        }
    
    # === 步骤4: 判定氯原子数 ===
    # 调用函数: determine_chlorine_number_from_ratio()
    observed_ratio = cluster_result['result']['M_plus_2_ratio']
    chlorine_result = determine_chlorine_number_from_ratio(observed_ratio)
    
    return {
        'result': {
            'num_chlorine': chlorine_result['result']['num_chlorine'],
            'confidence': chlorine_result['result']['confidence'],
            'M_plus_2_ratio': observed_ratio,
            'theoretical_ratio': chlorine_result['result']['theoretical_ratio'],
            'isotope_cluster': cluster_result['result']['cluster']
        },
        'metadata': {
            'base_peak_mz': base_peak_mz,
            'num_peaks_detected': len(peak_mz),
            'deviation': chlorine_result['result']['deviation']
        }
    }


# ============ 第三层：可视化工具 ============

def visualize_isotope_pattern_comparison(observed_cluster: List[Dict],
                                          num_chlorine: int,
                                          save_dir: str = './tool_visual_images/',
                                          filename: str = 'isotope_pattern_comparison.png') -> str:
    """
    可视化观测同位素模式与理论模式的对比
    
    Args:
        observed_cluster: 观测到的同位素簇数据
        num_chlorine: 判定的氯原子数
        save_dir: 保存目录
        filename: 文件名
    
    Returns:
        str: 保存的图片路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取理论模式
    theory_result = calculate_theoretical_isotope_pattern(num_chlorine)
    theory_peaks = theory_result['result']['peaks']
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制观测谱图
    if observed_cluster:
        obs_labels = [p['label'] for p in observed_cluster]
        obs_intensities = [p['intensity'] for p in observed_cluster]
        ax1.bar(range(len(obs_labels)), obs_intensities, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(obs_labels)))
        ax1.set_xticklabels(obs_labels)
        ax1.set_ylabel('相对强度 (%)', fontsize=12)
        ax1.set_title('观测同位素模式', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
    # 绘制理论谱图
    theory_labels = [f"M+{int(p['mass_shift'])}" for p in theory_peaks]
    theory_intensities = [p['intensity'] for p in theory_peaks]
    ax2.bar(range(len(theory_labels)), theory_intensities, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(theory_labels)))
    ax2.set_xticklabels(theory_labels)
    ax2.set_ylabel('相对强度 (%)', fontsize=12)
    ax2.set_title(f'理论同位素模式 ({num_chlorine}个Cl)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Isotope_Pattern_Comparison | PATH: {save_path}")
    return save_path


def visualize_mass_spectrum(mz_values: List[float],
                              intensities: List[float],
                              highlighted_peaks: Optional[List[float]] = None,
                              save_dir: str = image_path/'tool_visual_images/',
                              filename: str = 'mass_spectrum.png') -> str:
    """
    可视化质谱图并高亮同位素峰
    
    Args:
        mz_values: m/z值列表
        intensities: 强度列表
        highlighted_peaks: 需要高亮的峰的m/z值
        save_dir: 保存目录
        filename: 文件名
    
    Returns:
        str: 保存的图片路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制质谱图
    ax.vlines(mz_values, 0, intensities, colors='black', linewidth=1.5)
    
    # 高亮特定峰
    if highlighted_peaks:
        for peak_mz in highlighted_peaks:
            # 找到最接近的m/z值
            idx = min(range(len(mz_values)), key=lambda i: abs(mz_values[i] - peak_mz))
            ax.vlines(mz_values[idx], 0, intensities[idx], colors='red', linewidth=2.5, label='同位素峰')
    
    ax.set_xlabel('m/z', fontsize=14, fontweight='bold')
    ax.set_ylabel('相对强度 (%)', fontsize=14, fontweight='bold')
    ax.set_title('MS/MS质谱图', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    if highlighted_peaks:
        # 去重图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Mass_Spectrum | PATH: {save_path}")
    return save_path


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【氯原子数判定】+【相关场景】
    """
    
    print("=" * 60)
    print("场景1：从MS/MS谱图判定氯原子数（原始问题）")
    print("=" * 60)
    print("问题描述：根据质谱图中的同位素模式（M和M+2峰强度比），判定化合物中氯原子的数量")
    print("-" * 60)
    
    # 模拟从图片提取的质谱数据（基于图片中的峰）
    # 主要峰簇在m/z 100-220区间
    mz_data = [
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
        150.0, 151.0, 152.0, 153.0, 154.0,
        180.0, 181.0, 182.0, 183.0, 184.0,
        200.0, 201.0, 202.0, 203.0
    ]
    
    intensity_data = [
        2.0, 3.5, 20.0, 5.0, 10.0, 3.0,           # 簇1
        100.0, 27.0, 32.0, 8.0, 18.0,             # 簇2（基峰）
        15.0, 4.0, 5.0, 2.0, 3.0,                 # 簇3
        10.0, 3.0, 3.5, 1.0                       # 簇4
    ]
    
    # 步骤1：使用原子函数依次完成分析
    print("\n步骤1：使用原子函数进行分析并打印每步结果")

    # 1) 峰提取
    height_threshold = 5.0
    print(f"FUNCTION_CALL: extract_peaks_from_spectrum | PARAMS: height_threshold={height_threshold}")
    peaks_result = extract_peaks_from_spectrum(mz_data, intensity_data, height_threshold=height_threshold)
    peak_mz = peaks_result['result']['peak_mz']
    peak_intensity = peaks_result['result']['peak_intensity']
    print(f"  OUTPUT: num_peaks={peaks_result['result']['num_peaks']}, peak_mz={peak_mz}, peak_intensity={peak_intensity}")

    # 2) 同位素簇识别（指定基峰）
    base_peak_mz = 150.0
    print(f"FUNCTION_CALL: find_isotope_cluster | PARAMS: base_mz={base_peak_mz}, mass_tolerance=0.5")
    cluster_result = find_isotope_cluster(peak_mz, peak_intensity, base_mz=base_peak_mz, mass_tolerance=0.5)
    observed_cluster = cluster_result['result']['cluster']
    observed_ratio = cluster_result['result']['M_plus_2_ratio']
    print(f"  OUTPUT: cluster_size={cluster_result['metadata']['cluster_size']}, M_plus_2_ratio={observed_ratio}, cluster={observed_cluster}")

    # 3) 氯原子数判定（若存在观测比值）
    num_chlorine = 0
    confidence = 'low'
    theoretical_ratio = 0.0
    deviation = None
    if observed_ratio is not None:
        print(f"FUNCTION_CALL: determine_chlorine_number_from_ratio | PARAMS: observed_ratio={observed_ratio}")
        chlorine_result = determine_chlorine_number_from_ratio(observed_ratio)
        num_chlorine = chlorine_result['result']['num_chlorine']
        confidence = chlorine_result['result']['confidence']
        theoretical_ratio = chlorine_result['result']['theoretical_ratio']
        deviation = chlorine_result['result']['deviation']
        print(f"  OUTPUT: num_chlorine={num_chlorine}, confidence={confidence}, theoretical_ratio={theoretical_ratio}, deviation={deviation}")
    else:
        print("  OUTPUT: 未找到M+2峰，无法计算观测比值，默认num_chlorine=0，confidence=low")
    
    # 步骤2：可视化对比
    print("\n步骤2：生成可视化对比图")
    vis_path1 = visualize_isotope_pattern_comparison(
        observed_cluster,
        num_chlorine
    )
    
    vis_path2 = visualize_mass_spectrum(
        mz_data,
        intensity_data,
        highlighted_peaks=[150.0, 152.0]
    )
    
    print(f"\n✓ 场景1完成：化合物含有 {num_chlorine} 个氯原子")
    print(f"  观测M+2/M比值: {observed_ratio if observed_ratio is not None else 0.0}")
    print(f"  理论M+2/M比值: {theoretical_ratio}")
    print(f"  置信度: {confidence}")
    print(f"  同位素簇: {observed_cluster}")
    print(f"FINAL_ANSWER: {num_chlorine}")
    
    # ============ 场景2：参数扫描 - 不同氯原子数的理论模式对比 ============
    print("\n" + "=" * 60)
    print("场景2：理论同位素模式参数扫描")
    print("=" * 60)
    print("问题描述：计算并对比0-3个氯原子的理论同位素分布模式")
    print("-" * 60)
    
    print("\n对比不同氯原子数的M+2/M理论比值：")
    for n_cl in range(4):
        # 调用函数：calculate_theoretical_isotope_pattern()
        theory_result = calculate_theoretical_isotope_pattern(n_cl)
        ratio = theory_result['result']['M_plus_2_ratio']
        peaks = theory_result['result']['peaks']
        print(f"FUNCTION_CALL: calculate_theoretical_isotope_pattern | PARAMS: num_chlorine={n_cl} | RESULT: M+2/M={ratio}")

        # 构建峰模式字符串（避免f-string嵌套）
        peak_labels = []
        for p in peaks[:3]:
            label = "M+" + str(int(p['mass_shift']))
            intensity = p['intensity']
            peak_labels.append(f"{label}({intensity:.1f}%)")
        peak_pattern = ", ".join(peak_labels)
        print(f"  {n_cl}个Cl: M+2/M = {ratio:.3f}, 峰模式: {peak_pattern}")
    
    print(f"\n✓ 场景2完成：理论模式计算完成，可用于未知样品的快速比对")
    
    
    
    print("\n" + "=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了从质谱图判定氯原子数的完整流程（原始问题）")
    print("- 场景2展示了理论同位素模式的参数扫描能力")

    print("\n核心工具函数调用链：")
    print("  analyze_spectrum_for_chlorine()")
    print("    ├─ extract_peaks_from_spectrum()")
    print("    ├─ find_isotope_cluster()")
    print("    └─ determine_chlorine_number_from_ratio()")
    print("         └─ calculate_theoretical_isotope_pattern()")


if __name__ == "__main__":
    main()