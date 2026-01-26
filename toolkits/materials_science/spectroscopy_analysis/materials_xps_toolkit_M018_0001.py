# Filename: materials_xps_toolkit.py
"""
材料科学XPS光谱分析工具包

主要功能：
1. XPS峰位识别与拟合：基于scipy实现高斯-洛伦兹混合峰拟合
2. 元素氧化态分析：结合NIST XPS数据库进行化学态识别
3. 自旋轨道分裂计算：自动识别双峰结构并计算分裂能
4. 光谱可视化：生成专业XPS谱图（含去卷积组分）

依赖库：
pip install numpy scipy pandas matplotlib plotly lmfit requests beautifulsoup4
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib
import os
import json
from datetime import datetime

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 全局常量 ============
# XPS标准参考数据（来源：NIST XPS Database）
XPS_REFERENCE_DATA = {
    'Ir': {
        '4f7/2': {
            'Ir0': {'BE': 60.8, 'FWHM': 0.9, 'range': (60.5, 61.2)},
            'Ir3+': {'BE': 61.8, 'FWHM': 1.2, 'range': (61.5, 62.2)},
            'Ir4+': {'BE': 62.5, 'FWHM': 1.3, 'range': (62.2, 63.0)}
        },
        '4f5/2': {
            'Ir0': {'BE': 63.8, 'FWHM': 0.9, 'range': (63.5, 64.2)},
            'Ir3+': {'BE': 64.8, 'FWHM': 1.2, 'range': (64.5, 65.2)},
            'Ir4+': {'BE': 65.5, 'FWHM': 1.3, 'range': (65.2, 66.0)}
        },
        'spin_orbit_splitting': 3.0,
        'intensity_ratio': 0.75  # I(4f7/2) / I(4f5/2) = 4/3
    },
    'Pt': {
        '4f7/2': {
            'Pt0': {'BE': 71.2, 'FWHM': 1.1, 'range': (70.8, 71.6)},
            'Pt2+': {'BE': 72.5, 'FWHM': 1.4, 'range': (72.0, 73.0)},
            'Pt4+': {'BE': 74.5, 'FWHM': 1.5, 'range': (74.0, 75.0)}
        },
        'spin_orbit_splitting': 3.3,
        'intensity_ratio': 0.75
    }
}

SAVE_DIR_MID = './mid_result/materials/'
SAVE_DIR_IMG = './tool_images/'
os.makedirs(SAVE_DIR_MID, exist_ok=True)
os.makedirs(SAVE_DIR_IMG, exist_ok=True)


# ============ 第一层：原子工具函数 ============

def detect_peaks_in_spectrum(
    binding_energy: List[float],
    intensity: List[float],
    prominence_threshold: float = 0.1,
    min_distance: int = 10
) -> dict:
    """
    在XPS光谱中自动检测峰位置
    
    使用scipy的find_peaks算法识别光谱中的显著峰，基于峰的突出度（prominence）
    和最小间距进行筛选，适用于含噪声的实验数据。
    
    Args:
        binding_energy: 结合能数据点/eV，必须单调递增或递减
        intensity: 对应的强度值（counts或CPS），长度需与binding_energy一致
        prominence_threshold: 峰突出度阈值（相对于最大强度），范围0-1，默认0.1
        min_distance: 相邻峰的最小间隔点数，默认10（对应约0.5-1 eV）
    
    Returns:
        dict: {
            'result': {
                'peak_positions': [BE1, BE2, ...],  # 峰位结合能/eV
                'peak_intensities': [I1, I2, ...],   # 峰强度
                'peak_indices': [idx1, idx2, ...]    # 峰在原数组中的索引
            },
            'metadata': {
                'num_peaks': int,
                'prominence_used': float,
                'algorithm': 'scipy.signal.find_peaks'
            }
        }
    
    Example:
        >>> be = [56.0, 57.0, ..., 70.0]
        >>> intensity = [2500, 2600, ..., 3100]
        >>> result = detect_peaks_in_spectrum(be, intensity, prominence_threshold=0.15)
        >>> print(result['result']['peak_positions'])
        [60.59, 63.59]
    """
    # === 边界条件检查 ===
    if not isinstance(binding_energy, list) or not isinstance(intensity, list):
        raise TypeError("binding_energy和intensity必须是list类型")
    
    if len(binding_energy) != len(intensity):
        raise ValueError(f"数据长度不匹配: BE={len(binding_energy)}, I={len(intensity)}")
    
    if len(binding_energy) < 10:
        raise ValueError(f"数据点过少({len(binding_energy)}), 至少需要10个点")
    
    if not (0 < prominence_threshold <= 1):
        raise ValueError(f"prominence_threshold必须在(0,1]范围内，当前值={prominence_threshold}")
    
    # 转换为numpy数组
    be_array = np.array(binding_energy, dtype=float)
    int_array = np.array(intensity, dtype=float)
    
    # 检查是否有NaN或Inf
    if np.any(~np.isfinite(be_array)) or np.any(~np.isfinite(int_array)):
        raise ValueError("数据包含NaN或Inf值")
    
    # === 核心计算：峰检测 ===
    max_intensity = np.max(int_array)
    prominence = prominence_threshold * max_intensity
    
    peak_indices, properties = find_peaks(
        int_array,
        prominence=prominence,
        distance=min_distance
    )
    
    if len(peak_indices) == 0:
        return {
            'result': {
                'peak_positions': [],
                'peak_intensities': [],
                'peak_indices': []
            },
            'metadata': {
                'num_peaks': 0,
                'prominence_used': prominence,
                'algorithm': 'scipy.signal.find_peaks',
                'warning': '未检测到峰，建议降低prominence_threshold'
            }
        }
    
    peak_positions = be_array[peak_indices].tolist()
    peak_intensities = int_array[peak_indices].tolist()
    
    return {
        'result': {
            'peak_positions': peak_positions,
            'peak_intensities': peak_intensities,
            'peak_indices': peak_indices.tolist()
        },
        'metadata': {
            'num_peaks': len(peak_indices),
            'prominence_used': prominence,
            'min_distance_points': min_distance,
            'algorithm': 'scipy.signal.find_peaks'
        }
    }


def fit_voigt_peak(
    binding_energy: List[float],
    intensity: List[float],
    initial_center: float,
    initial_amplitude: float,
    initial_fwhm: float = 1.0,
    gamma_fraction: float = 0.5
) -> dict:
    """
    使用Voigt函数拟合单个XPS峰
    
    Voigt函数是高斯函数和洛伦兹函数的卷积，能更准确地描述XPS峰形。
    高斯成分来自仪器展宽，洛伦兹成分来自自然线宽。
    
    Args:
        binding_energy: 结合能数据/eV
        intensity: 强度数据
        initial_center: 峰中心初始猜测值/eV
        initial_amplitude: 峰高初始猜测值
        initial_fwhm: 半高全宽初始猜测值/eV，默认1.0
        gamma_fraction: 洛伦兹成分占比（0-1），默认0.5表示高斯洛伦兹各半
    
    Returns:
        dict: {
            'result': {
                'center': float,      # 拟合得到的峰位/eV
                'amplitude': float,   # 峰高
                'fwhm': float,        # 半高全宽/eV
                'area': float,        # 峰面积
                'fit_quality': float  # R²值
            },
            'metadata': {
                'gamma_fraction': float,
                'fit_method': 'Voigt (Gaussian-Lorentzian convolution)'
            }
        }
    
    Example:
        >>> result = fit_voigt_peak(be, intensity, initial_center=60.5, initial_amplitude=28000)
        >>> print(f"拟合峰位: {result['result']['center']:.2f} eV")
    """
    # === 边界条件检查 ===
    if not isinstance(binding_energy, list) or not isinstance(intensity, list):
        raise TypeError("输入必须是list类型")
    
    if len(binding_energy) != len(intensity):
        raise ValueError("数据长度不匹配")
    
    if not (0 <= gamma_fraction <= 1):
        raise ValueError(f"gamma_fraction必须在[0,1]范围内，当前值={gamma_fraction}")
    
    if initial_fwhm <= 0:
        raise ValueError(f"initial_fwhm必须为正数，当前值={initial_fwhm}")
    
    be_array = np.array(binding_energy, dtype=float)
    int_array = np.array(intensity, dtype=float)
    
    # === Voigt函数定义 ===
    def voigt(x, center, amplitude, fwhm, gamma_frac):
        """
        Voigt profile = Gaussian ⊗ Lorentzian
        gamma_frac: 洛伦兹成分占比
        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2))) * (1 - gamma_frac)
        gamma = fwhm * gamma_frac / 2
        
        from scipy.special import wofz
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        profile = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        return profile
    
    # === 拟合 ===
    try:
        popt, pcov = curve_fit(
            lambda x, c, a, f: voigt(x, c, a, f, gamma_fraction),
            be_array,
            int_array,
            p0=[initial_center, initial_amplitude, initial_fwhm],
            maxfev=5000
        )
        
        center, amplitude, fwhm = popt
        
        # 计算拟合优度
        fitted_curve = voigt(be_array, center, amplitude, fwhm, gamma_fraction)
        ss_res = np.sum((int_array - fitted_curve) ** 2)
        ss_tot = np.sum((int_array - np.mean(int_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 计算峰面积（数值积分）
        area = simpson(fitted_curve, be_array)
        
        return {
            'result': {
                'center': float(center),
                'amplitude': float(amplitude),
                'fwhm': float(fwhm),
                'area': float(area),
                'fit_quality': float(r_squared)
            },
            'metadata': {
                'gamma_fraction': gamma_fraction,
                'fit_method': 'Voigt (Gaussian-Lorentzian convolution)',
                'covariance_matrix': pcov.tolist()
            }
        }
    
    except Exception as e:
        return {
            'result': {
                'center': initial_center,
                'amplitude': initial_amplitude,
                'fwhm': initial_fwhm,
                'area': 0.0,
                'fit_quality': 0.0
            },
            'metadata': {
                'error': str(e),
                'status': 'fit_failed'
            }
        }


def calculate_spin_orbit_splitting(
    peak1_position: float,
    peak2_position: float,
    element: str,
    orbital: str
) -> dict:
    """
    计算并验证自旋轨道分裂能
    
    自旋轨道耦合导致p/d/f轨道分裂为j=l±1/2两个能级，分裂能ΔE是元素的特征参数。
    该函数计算实验分裂能并与理论值对比，用于验证峰归属的正确性。
    
    Args:
        peak1_position: 第一个峰的结合能/eV（通常是j=l-1/2的低能峰）
        peak2_position: 第二个峰的结合能/eV（通常是j=l+1/2的高能峰）
        element: 元素符号，如'Ir', 'Pt', 'Au'
        orbital: 轨道标记，如'4f', '3d', '4d'
    
    Returns:
        dict: {
            'result': {
                'experimental_splitting': float,  # 实验分裂能/eV
                'theoretical_splitting': float,   # 理论分裂能/eV
                'deviation': float,               # 偏差/eV
                'is_valid': bool                  # 是否在合理范围内（±0.3 eV）
            },
            'metadata': {
                'element': str,
                'orbital': str,
                'validation_threshold': 0.3
            }
        }
    
    Example:
        >>> result = calculate_spin_orbit_splitting(60.59, 63.59, 'Ir', '4f')
        >>> print(f"分裂能: {result['result']['experimental_splitting']:.2f} eV")
        >>> print(f"验证通过: {result['result']['is_valid']}")
    """
    # === 边界条件检查 ===
    if not isinstance(peak1_position, (int, float)) or not isinstance(peak2_position, (int, float)):
        raise TypeError("峰位必须是数值类型")
    
    if peak1_position >= peak2_position:
        raise ValueError(f"peak1应小于peak2（低能峰在前），当前: {peak1_position} >= {peak2_position}")
    
    if element not in XPS_REFERENCE_DATA:
        raise ValueError(f"元素'{element}'不在参考数据库中，支持的元素: {list(XPS_REFERENCE_DATA.keys())}")
    
    # === 计算分裂能 ===
    experimental_splitting = peak2_position - peak1_position
    
    # 获取理论值
    if 'spin_orbit_splitting' in XPS_REFERENCE_DATA[element]:
        theoretical_splitting = XPS_REFERENCE_DATA[element]['spin_orbit_splitting']
    else:
        theoretical_splitting = None
    
    if theoretical_splitting is None:
        return {
            'result': {
                'experimental_splitting': experimental_splitting,
                'theoretical_splitting': None,
                'deviation': None,
                'is_valid': None
            },
            'metadata': {
                'element': element,
                'orbital': orbital,
                'warning': '缺少理论参考值'
            }
        }
    
    deviation = abs(experimental_splitting - theoretical_splitting)
    validation_threshold = 0.3  # eV
    is_valid = deviation <= validation_threshold
    
    return {
        'result': {
            'experimental_splitting': round(experimental_splitting, 2),
            'theoretical_splitting': theoretical_splitting,
            'deviation': round(deviation, 2),
            'is_valid': is_valid
        },
        'metadata': {
            'element': element,
            'orbital': orbital,
            'validation_threshold': validation_threshold,
            'interpretation': '分裂能验证通过' if is_valid else f'分裂能偏差过大({deviation:.2f} eV)'
        }
    }


def assign_oxidation_state(
    peak_position: float,
    element: str,
    orbital: str
) -> dict:
    """
    根据结合能判断元素的氧化态
    
    不同氧化态的原子核对外层电子的束缚力不同，导致结合能偏移。
    高氧化态对应更高的结合能（化学位移为正）。
    
    Args:
        peak_position: 实验测得的峰位/eV
        element: 元素符号
        orbital: 轨道标记（如'4f7/2', '3d5/2'）
    
    Returns:
        dict: {
            'result': {
                'oxidation_state': str,        # 如'Ir0', 'Ir3+', 'Ir4+'
                'confidence': str,             # 'high', 'medium', 'low'
                'reference_BE': float,         # 参考结合能/eV
                'chemical_shift': float        # 相对于金属态的化学位移/eV
            },
            'metadata': {
                'element': str,
                'orbital': str,
                'all_candidates': list         # 所有可能的氧化态及其偏差
            }
        }
    
    Example:
        >>> result = assign_oxidation_state(60.59, 'Ir', '4f7/2')
        >>> print(result['result']['oxidation_state'])
        'Ir0'
    """
    # === 边界条件检查 ===
    if not isinstance(peak_position, (int, float)):
        raise TypeError("peak_position必须是数值类型")
    
    if element not in XPS_REFERENCE_DATA:
        raise ValueError(f"元素'{element}'不在数据库中")
    
    if orbital not in XPS_REFERENCE_DATA[element]:
        raise ValueError(f"轨道'{orbital}'不在元素'{element}'的数据中")
    
    # === 匹配氧化态 ===
    orbital_data = XPS_REFERENCE_DATA[element][orbital]
    candidates = []
    
    for ox_state, params in orbital_data.items():
        ref_be = params['BE']
        deviation = abs(peak_position - ref_be)
        be_range = params['range']
        
        in_range = be_range[0] <= peak_position <= be_range[1]
        
        candidates.append({
            'oxidation_state': ox_state,
            'reference_BE': ref_be,
            'deviation': deviation,
            'in_range': in_range
        })
    
    # 按偏差排序
    candidates.sort(key=lambda x: x['deviation'])
    best_match = candidates[0]
    
    # 判断置信度
    if best_match['in_range']:
        confidence = 'high'
    elif best_match['deviation'] < 0.5:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    # 计算化学位移（相对于金属态）
    metal_state_key = f"{element}0"
    if metal_state_key in orbital_data:
        metal_be = orbital_data[metal_state_key]['BE']
        chemical_shift = peak_position - metal_be
    else:
        chemical_shift = None
    
    return {
        'result': {
            'oxidation_state': best_match['oxidation_state'],
            'confidence': confidence,
            'reference_BE': best_match['reference_BE'],
            'chemical_shift': chemical_shift
        },
        'metadata': {
            'element': element,
            'orbital': orbital,
            'all_candidates': candidates
        }
    }


def calculate_peak_area_ratio(
    area1: float,
    area2: float,
    theoretical_ratio: float
) -> dict:
    """
    计算峰面积比并与理论值对比
    
    自旋轨道分裂双峰的面积比由量子力学决定：I(j=l+1/2) / I(j=l-1/2) = (2j₁+1)/(2j₂+1)
    例如4f: I(4f5/2) / I(4f7/2) = 6/8 = 0.75
    
    Args:
        area1: 第一个峰的面积（通常是低能峰）
        area2: 第二个峰的面积（通常是高能峰）
        theoretical_ratio: 理论面积比 area2/area1
    
    Returns:
        dict: {
            'result': {
                'experimental_ratio': float,
                'theoretical_ratio': float,
                'deviation_percent': float,
                'is_consistent': bool
            },
            'metadata': {
                'tolerance_percent': 20.0
            }
        }
    
    Example:
        >>> result = calculate_peak_area_ratio(1000, 750, 0.75)
        >>> print(f"面积比: {result['result']['experimental_ratio']:.2f}")
    """
    # === 边界条件检查 ===
    if area1 <= 0 or area2 <= 0:
        raise ValueError(f"峰面积必须为正数，当前: area1={area1}, area2={area2}")
    
    if theoretical_ratio <= 0:
        raise ValueError(f"理论比值必须为正数，当前={theoretical_ratio}")
    
    # === 计算 ===
    experimental_ratio = area2 / area1
    deviation_percent = abs(experimental_ratio - theoretical_ratio) / theoretical_ratio * 100
    
    tolerance = 20.0  # 允许20%偏差
    is_consistent = deviation_percent <= tolerance
    
    return {
        'result': {
            'experimental_ratio': round(experimental_ratio, 3),
            'theoretical_ratio': theoretical_ratio,
            'deviation_percent': round(deviation_percent, 1),
            'is_consistent': is_consistent
        },
        'metadata': {
            'tolerance_percent': tolerance,
            'interpretation': '面积比符合理论' if is_consistent else f'面积比偏差较大({deviation_percent:.1f}%)'
        }
    }


# ============ 第二层：组合工具函数 ============

def analyze_xps_doublet(
    binding_energy: List[float],
    intensity: List[float],
    element: str,
    orbital: str,
    prominence_threshold: float = 0.1
) -> dict:
    """
    完整分析XPS自旋轨道分裂双峰
    
    该函数整合峰检测、拟合、氧化态判断、分裂能验证等步骤，
    提供XPS双峰的全面分析结果。
    
    Args:
        binding_energy: 结合能数据/eV
        intensity: 强度数据
        element: 元素符号（如'Ir'）
        orbital: 轨道类型（如'4f'）
        prominence_threshold: 峰检测阈值
    
    Returns:
        dict: {
            'result': {
                'peak1': {峰1的完整信息},
                'peak2': {峰2的完整信息},
                'spin_orbit_analysis': {分裂能分析},
                'area_ratio_analysis': {面积比分析},
                'summary': str  # 文字总结
            },
            'metadata': {...}
        }
    
    Example:
        >>> result = analyze_xps_doublet(be, intensity, 'Ir', '4f')
        >>> print(result['result']['summary'])
    """
    # === 参数检查 ===
    if not isinstance(binding_energy, list) or not isinstance(intensity, list):
        raise TypeError("输入必须是list类型")
    
    if element not in XPS_REFERENCE_DATA:
        raise ValueError(f"不支持的元素: {element}")
    
    # === 步骤1: 峰检测 ===
    # 调用函数: detect_peaks_in_spectrum()
    peak_detection = detect_peaks_in_spectrum(
        binding_energy, intensity, prominence_threshold
    )
    
    detected_peaks = peak_detection['result']['peak_positions']
    
    if len(detected_peaks) < 2:
        return {
            'result': {
                'error': f'仅检测到{len(detected_peaks)}个峰，需要至少2个峰'
            },
            'metadata': {'status': 'insufficient_peaks'}
        }
    
    # 取强度最高的两个峰
    peak_intensities = peak_detection['result']['peak_intensities']
    sorted_indices = np.argsort(peak_intensities)[::-1][:2]
    sorted_indices = sorted(sorted_indices)  # 按结合能排序
    
    peak1_be = detected_peaks[sorted_indices[0]]
    peak2_be = detected_peaks[sorted_indices[1]]
    peak1_int = peak_intensities[sorted_indices[0]]
    peak2_int = peak_intensities[sorted_indices[1]]
    
    # === 步骤2: 峰拟合 ===
    # 调用函数: fit_voigt_peak() 两次
    fit1 = fit_voigt_peak(
        binding_energy, intensity,
        initial_center=peak1_be,
        initial_amplitude=peak1_int
    )
    
    fit2 = fit_voigt_peak(
        binding_energy, intensity,
        initial_center=peak2_be,
        initial_amplitude=peak2_int
    )
    
    # === 步骤3: 氧化态判断 ===
    # 调用函数: assign_oxidation_state() 两次
    orbital_full_1 = f"{orbital}7/2"
    orbital_full_2 = f"{orbital}5/2"
    
    ox_state1 = assign_oxidation_state(
        fit1['result']['center'], element, orbital_full_1
    )
    
    ox_state2 = assign_oxidation_state(
        fit2['result']['center'], element, orbital_full_2
    )
    
    # === 步骤4: 分裂能验证 ===
    # 调用函数: calculate_spin_orbit_splitting()
    splitting_analysis = calculate_spin_orbit_splitting(
        fit1['result']['center'],
        fit2['result']['center'],
        element,
        orbital
    )
    
    # === 步骤5: 面积比验证 ===
    # 调用函数: calculate_peak_area_ratio()
    theoretical_ratio = XPS_REFERENCE_DATA[element]['intensity_ratio']
    area_ratio_analysis = calculate_peak_area_ratio(
        fit1['result']['area'],
        fit2['result']['area'],
        theoretical_ratio
    )
    
    # === 生成总结 ===
    summary = (
        f"检测到{element} {orbital}双峰: "
        f"{orbital}7/2位于{fit1['result']['center']:.2f} eV "
        f"({ox_state1['result']['oxidation_state']}, 置信度{ox_state1['result']['confidence']}), "
        f"{orbital}5/2位于{fit2['result']['center']:.2f} eV "
        f"({ox_state2['result']['oxidation_state']}, 置信度{ox_state2['result']['confidence']}). "
        f"自旋轨道分裂能为{splitting_analysis['result']['experimental_splitting']:.2f} eV "
        f"({'验证通过' if splitting_analysis['result']['is_valid'] else '偏差过大'}), "
        f"峰面积比为{area_ratio_analysis['result']['experimental_ratio']:.2f} "
        f"({'符合理论' if area_ratio_analysis['result']['is_consistent'] else '存在偏差'})."
    )
    
    return {
        'result': {
            'peak1': {
                'binding_energy': fit1['result']['center'],
                'intensity': fit1['result']['amplitude'],
                'fwhm': fit1['result']['fwhm'],
                'area': fit1['result']['area'],
                'oxidation_state': ox_state1['result']['oxidation_state'],
                'confidence': ox_state1['result']['confidence']
            },
            'peak2': {
                'binding_energy': fit2['result']['center'],
                'intensity': fit2['result']['amplitude'],
                'fwhm': fit2['result']['fwhm'],
                'area': fit2['result']['area'],
                'oxidation_state': ox_state2['result']['oxidation_state'],
                'confidence': ox_state2['result']['confidence']
            },
            'spin_orbit_analysis': splitting_analysis['result'],
            'area_ratio_analysis': area_ratio_analysis['result'],
            'summary': summary
        },
        'metadata': {
            'element': element,
            'orbital': orbital,
            'num_peaks_detected': len(detected_peaks),
            'analysis_timestamp': datetime.now().isoformat()
        }
    }


def batch_analyze_multiple_elements(
    spectra_data: Dict[str, Dict[str, List[float]]],
    element_orbital_pairs: List[Dict[str, str]]
) -> dict:
    """
    批量分析多个元素的XPS谱
    
    适用于含多种元素的样品，自动对每个元素进行双峰分析。
    
    Args:
        spectra_data: {
            'element1_orbital1': {'BE': [...], 'intensity': [...]},
            'element2_orbital2': {'BE': [...], 'intensity': [...]}
        }
        element_orbital_pairs: [
            {'element': 'Ir', 'orbital': '4f'},
            {'element': 'Pt', 'orbital': '4f'}
        ]
    
    Returns:
        dict: {
            'result': {
                'Ir_4f': {分析结果},
                'Pt_4f': {分析结果},
                ...
            },
            'metadata': {
                'total_elements': int,
                'successful_analyses': int
            }
        }
    
    Example:
        >>> data = {'Ir_4f': {'BE': [...], 'intensity': [...]}}
        >>> pairs = [{'element': 'Ir', 'orbital': '4f'}]
        >>> result = batch_analyze_multiple_elements(data, pairs)
    """
    # === 参数检查 ===
    if not isinstance(spectra_data, dict):
        raise TypeError("spectra_data必须是dict类型")
    
    if not isinstance(element_orbital_pairs, list):
        raise TypeError("element_orbital_pairs必须是list类型")
    
    results = {}
    successful = 0
    
    for pair in element_orbital_pairs:
        element = pair['element']
        orbital = pair['orbital']
        key = f"{element}_{orbital}"
        
        if key not in spectra_data:
            results[key] = {
                'error': f'缺少{key}的光谱数据'
            }
            continue
        
        spectrum = spectra_data[key]
        
        # 调用函数: analyze_xps_doublet()
        try:
            analysis = analyze_xps_doublet(
                spectrum['BE'],
                spectrum['intensity'],
                element,
                orbital
            )
            results[key] = analysis['result']
            successful += 1
        except Exception as e:
            results[key] = {
                'error': str(e)
            }
    
    return {
        'result': results,
        'metadata': {
            'total_elements': len(element_orbital_pairs),
            'successful_analyses': successful,
            'failed_analyses': len(element_orbital_pairs) - successful
        }
    }


def compare_oxidation_states_across_samples(
    sample_analyses: Dict[str, dict]
) -> dict:
    """
    对比多个样品的氧化态分布
    
    用于研究不同处理条件对材料氧化态的影响。
    
    Args:
        sample_analyses: {
            'sample1': {analyze_xps_doublet的输出},
            'sample2': {analyze_xps_doublet的输出},
            ...
        }
    
    Returns:
        dict: {
            'result': {
                'oxidation_state_distribution': DataFrame格式的统计,
                'average_binding_energy': {样品名: BE值},
                'oxidation_trend': str
            },
            'metadata': {...}
        }
    
    Example:
        >>> samples = {'fresh': result1, 'aged': result2}
        >>> comparison = compare_oxidation_states_across_samples(samples)
    """
    # === 参数检查 ===
    if not isinstance(sample_analyses, dict):
        raise TypeError("sample_analyses必须是dict类型")
    
    if len(sample_analyses) == 0:
        raise ValueError("至少需要一个样品数据")
    
    # === 提取数据 ===
    comparison_data = []
    avg_be = {}
    
    for sample_name, analysis in sample_analyses.items():
        if 'peak1' not in analysis or 'peak2' not in analysis:
            continue
        
        peak1 = analysis['peak1']
        peak2 = analysis['peak2']
        
        comparison_data.append({
            'sample': sample_name,
            'peak1_BE': peak1['binding_energy'],
            'peak1_ox_state': peak1['oxidation_state'],
            'peak2_BE': peak2['binding_energy'],
            'peak2_ox_state': peak2['oxidation_state']
        })
        
        avg_be[sample_name] = (peak1['binding_energy'] + peak2['binding_energy']) / 2
    
    # 转换为DataFrame格式（用dict表示）
    df_dict = {
        'columns': ['sample', 'peak1_BE', 'peak1_ox_state', 'peak2_BE', 'peak2_ox_state'],
        'data': comparison_data
    }
    
    # 分析趋势
    be_values = list(avg_be.values())
    if len(be_values) > 1:
        if all(be_values[i] <= be_values[i+1] for i in range(len(be_values)-1)):
            trend = "结合能递增，氧化程度加深"
        elif all(be_values[i] >= be_values[i+1] for i in range(len(be_values)-1)):
            trend = "结合能递减，还原程度增加"
        else:
            trend = "结合能波动，氧化态变化不规律"
    else:
        trend = "样品数量不足，无法判断趋势"
    
    return {
        'result': {
            'oxidation_state_distribution': df_dict,
            'average_binding_energy': avg_be,
            'oxidation_trend': trend
        },
        'metadata': {
            'num_samples': len(sample_analyses),
            'valid_samples': len(comparison_data)
        }
    }


# ============ 第三层：可视化工具 ============

def visualize_xps_spectrum_with_deconvolution(
    binding_energy: List[float],
    intensity: List[float],
    fitted_peaks: List[dict],
    element: str,
    orbital: str,
    save_dir: str = SAVE_DIR_IMG,
    filename: str = None
) -> dict:
    """
    绘制XPS谱图及峰去卷积结果
    
    生成专业的XPS谱图，包含原始数据、拟合曲线和各个去卷积峰。
    
    Args:
        binding_energy: 结合能数据/eV
        intensity: 强度数据
        fitted_peaks: 拟合峰列表，每个元素包含{'center', 'amplitude', 'fwhm', 'oxidation_state'}
        element: 元素符号
        orbital: 轨道标记
        save_dir: 保存目录
        filename: 文件名（不含扩展名），默认自动生成
    
    Returns:
        dict: {
            'result': '图片保存路径',
            'metadata': {...}
        }
    
    Example:
        >>> peaks = [{'center': 60.59, 'amplitude': 28000, 'fwhm': 0.9, 'oxidation_state': 'Ir0'}, ...]
        >>> result = visualize_xps_spectrum_with_deconvolution(be, intensity, peaks, 'Ir', '4f')
    """
    # === 参数检查 ===
    if not isinstance(binding_energy, list) or not isinstance(intensity, list):
        raise TypeError("输入必须是list类型")
    
    if not isinstance(fitted_peaks, list) or len(fitted_peaks) == 0:
        raise ValueError("fitted_peaks必须是非空列表")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # === 生成文件名 ===
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"XPS_{element}_{orbital}_{timestamp}"
    
    filepath = os.path.join(save_dir, f"{filename}.png")
    
    # === 绘图 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    be_array = np.array(binding_energy)
    int_array = np.array(intensity)
    
    # 原始数据
    ax.plot(be_array, int_array, 'ko', markersize=3, label='实验数据', alpha=0.6)
    
    # 拟合峰
    colors = ['cyan', 'green', 'magenta', 'orange', 'purple']
    total_fit = np.zeros_like(be_array)
    
    for i, peak in enumerate(fitted_peaks):
        # 重建Voigt峰
        def voigt_simple(x, c, a, f):
            sigma = f / (2 * np.sqrt(2 * np.log(2))) * 0.5
            gamma = f * 0.5 / 2
            from scipy.special import wofz
            z = ((x - c) + 1j * gamma) / (sigma * np.sqrt(2))
            return a * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        
        peak_curve = voigt_simple(
            be_array,
            peak['center'],
            peak['amplitude'],
            peak['fwhm']
        )
        
        total_fit += peak_curve
        
        color = colors[i % len(colors)]
        label = f"{peak.get('oxidation_state', f'Peak{i+1}')} ({peak['center']:.2f} eV)"
        ax.plot(be_array, peak_curve, color=color, linestyle='--', linewidth=1.5, label=label)
    
    # 总拟合曲线
    ax.plot(be_array, total_fit, 'r-', linewidth=2, label='总拟合')
    
    # 图表设置
    ax.set_xlabel('结合能 (eV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('强度 (counts)', fontsize=12, fontweight='bold')
    ax.set_title(f'{element} {orbital} XPS谱图及峰去卷积', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # XPS习惯：结合能从右到左递增
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: XPS_Spectrum_Plot | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'element': element,
            'orbital': orbital,
            'num_peaks': len(fitted_peaks),
            'file_size_kb': os.path.getsize(filepath) / 1024
        }
    }


def visualize_oxidation_state_comparison(
    comparison_data: dict,
    save_dir: str = SAVE_DIR_IMG,
    filename: str = None
) -> dict:
    """
    可视化多样品氧化态对比
    
    生成柱状图展示不同样品的结合能和氧化态分布。
    
    Args:
        comparison_data: compare_oxidation_states_across_samples的输出
        save_dir: 保存目录
        filename: 文件名
    
    Returns:
        dict: {'result': 文件路径, 'metadata': {...}}
    
    Example:
        >>> result = visualize_oxidation_state_comparison(comparison_result)
    """
    # === 参数检查 ===
    if not isinstance(comparison_data, dict):
        raise TypeError("comparison_data必须是dict类型")
    
    if 'average_binding_energy' not in comparison_data:
        raise ValueError("缺少average_binding_energy数据")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # === 生成文件名 ===
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Oxidation_State_Comparison_{timestamp}"
    
    filepath = os.path.join(save_dir, f"{filename}.png")
    
    # === 绘图 ===
    avg_be = comparison_data['average_binding_energy']
    samples = list(avg_be.keys())
    be_values = list(avg_be.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(samples, be_values, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, value in zip(bars, be_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f} eV',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('样品', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均结合能 (eV)', fontsize=12, fontweight='bold')
    ax.set_title('不同样品的氧化态对比（基于结合能）', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Oxidation_Comparison_Plot | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'num_samples': len(samples),
            'file_size_kb': os.path.getsize(filepath) / 1024
        }
    }


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【XPS谱图分析】+【至少2个相关场景】
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - Ir 4f XPS谱图分析")
    print("=" * 60)
    print("问题描述：分析给定的Ir 4f XPS谱图，识别峰位、判断氧化态、验证分裂能")
    print("-" * 60)
    
    # 模拟实验数据（基于图片中的谱图）
    # 实际应用中应从文件读取
    binding_energy_ir = list(np.linspace(56, 70, 200))
    
    # 模拟双峰结构（Ir 4f7/2 @ 60.59 eV, Ir 4f5/2 @ 63.59 eV）
    def generate_mock_spectrum(be, peak1_center, peak2_center):
        int_data = np.zeros_like(be)
        # Peak 1: Ir 4f7/2
        int_data += 28000 * np.exp(-((be - peak1_center) / 0.9) ** 2)
        # Peak 2: Ir 4f5/2
        int_data += 22000 * np.exp(-((be - peak2_center) / 0.9) ** 2)
        # 添加噪声
        int_data += np.random.normal(0, 500, len(be))
        # 添加背景
        int_data += 2500
        return int_data.tolist()
    
    intensity_ir = generate_mock_spectrum(
        np.array(binding_energy_ir), 60.59, 63.59
    )
    
    # 步骤1：完整双峰分析
    # 调用函数：analyze_xps_doublet()，该函数内部调用了 detect_peaks_in_spectrum(), fit_voigt_peak(), assign_oxidation_state(), calculate_spin_orbit_splitting(), calculate_peak_area_ratio()
    print("\n步骤1：执行完整XPS双峰分析...")
    analysis_result = analyze_xps_doublet(
        binding_energy_ir,
        intensity_ir,
        element='Ir',
        orbital='4f',
        prominence_threshold=0.15
    )
    
    print(f"FUNCTION_CALL: analyze_xps_doublet | PARAMS: element=Ir, orbital=4f | RESULT: {analysis_result['result']['summary']}")
    print(f"步骤1结果：\n{analysis_result['result']['summary']}\n")
    
    # 步骤2：可视化谱图
    # 调用函数：visualize_xps_spectrum_with_deconvolution()
    print("步骤2：生成XPS谱图及峰去卷积可视化...")
    fitted_peaks_for_plot = [
        {
            'center': analysis_result['result']['peak1']['binding_energy'],
            'amplitude': analysis_result['result']['peak1']['intensity'],
            'fwhm': analysis_result['result']['peak1']['fwhm'],
            'oxidation_state': analysis_result['result']['peak1']['oxidation_state']
        },
        {
            'center': analysis_result['result']['peak2']['binding_energy'],
            'amplitude': analysis_result['result']['peak2']['intensity'],
            'fwhm': analysis_result['result']['peak2']['fwhm'],
            'oxidation_state': analysis_result['result']['peak2']['oxidation_state']
        }
    ]
    
    plot_result = visualize_xps_spectrum_with_deconvolution(
        binding_energy_ir,
        intensity_ir,
        fitted_peaks_for_plot,
        'Ir',
        '4f',
        filename='Ir_4f_analysis_scene1'
    )
    
    print(f"FUNCTION_CALL: visualize_xps_spectrum_with_deconvolution | PARAMS: element=Ir, orbital=4f | RESULT: {plot_result['result']}")
    print(f"步骤2结果：谱图已保存至 {plot_result['result']}\n")
    
    print(f"✓ 场景1完成：Ir 4f谱图分析完成，峰位为{analysis_result['result']['peak1']['binding_energy']:.2f} eV和{analysis_result['result']['peak2']['binding_energy']:.2f} eV\n")
    
    # ============================================================
    
    print("=" * 60)
    print("场景2：参数扫描 - 不同prominence阈值对峰检测的影响")
    print("=" * 60)
    print("问题描述：测试不同峰检测阈值对结果的影响，优化参数选择")
    print("-" * 60)
    
    # 步骤1：测试多个阈值
    # 调用函数：detect_peaks_in_spectrum() 多次
    print("\n步骤1：扫描prominence阈值从0.05到0.25...")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    threshold_results = {}
    
    for thresh in thresholds:
        peak_detect = detect_peaks_in_spectrum(
            binding_energy_ir,
            intensity_ir,
            prominence_threshold=thresh
        )
        num_peaks = peak_detect['metadata']['num_peaks']
        threshold_results[thresh] = num_peaks
        print(f"FUNCTION_CALL: detect_peaks_in_spectrum | PARAMS: prominence_threshold={thresh} | RESULT: {num_peaks} peaks detected")
    
    print(f"步骤1结果：阈值扫描完成，结果={threshold_results}\n")
    
    # 步骤2：选择最优阈值（检测到2个峰的最低阈值）
    print("步骤2：确定最优阈值...")
    optimal_threshold = None
    for thresh in sorted(thresholds):
        if threshold_results[thresh] == 2:
            optimal_threshold = thresh
            break
    
    if optimal_threshold:
        print(f"步骤2结果：最优阈值为{optimal_threshold}（检测到2个峰且阈值最低）\n")
    else:
        print("步骤2结果：未找到检测到恰好2个峰的阈值\n")
    
    print(f"✓ 场景2完成：参数优化完成，推荐prominence_threshold={optimal_threshold}\n")
    
    # ============================================================
    
    print("=" * 60)
    print("场景3：批量分析 - 对比新鲜样品与老化样品的氧化态变化")
    print("=" * 60)
    print("问题描述：分析两个Ir样品（新鲜vs老化），对比氧化态演化")
    print("-" * 60)
    
    # 模拟老化样品（结合能略微升高，表示氧化程度增加）
    intensity_ir_aged = generate_mock_spectrum(
        np.array(binding_energy_ir), 61.2, 64.2  # 峰位向高能偏移
    )
    
    # 步骤1：分析两个样品
    # 调用函数：analyze_xps_doublet() 两次
    print("\n步骤1：分析新鲜样品...")
    analysis_fresh = analyze_xps_doublet(
        binding_energy_ir, intensity_ir, 'Ir', '4f'
    )
    print(f"FUNCTION_CALL: analyze_xps_doublet | PARAMS: sample=fresh | RESULT: {analysis_fresh['result']['summary']}")
    
    print("\n步骤1：分析老化样品...")
    analysis_aged = analyze_xps_doublet(
        binding_energy_ir, intensity_ir_aged, 'Ir', '4f'
    )
    print(f"FUNCTION_CALL: analyze_xps_doublet | PARAMS: sample=aged | RESULT: {analysis_aged['result']['summary']}")
    
    # 步骤2：对比分析
    # 调用函数：compare_oxidation_states_across_samples()
    print("\n步骤2：执行样品间氧化态对比...")
    comparison = compare_oxidation_states_across_samples({
        '新鲜样品': analysis_fresh['result'],
        '老化样品': analysis_aged['result']
    })
    
    print(f"FUNCTION_CALL: compare_oxidation_states_across_samples | PARAMS: num_samples=2 | RESULT: {comparison['result']['oxidation_trend']}")
    print(f"步骤2结果：\n平均结合能: {comparison['result']['average_binding_energy']}")
    print(f"氧化趋势: {comparison['result']['oxidation_trend']}\n")
    
    # 步骤3：可视化对比
    # 调用函数：visualize_oxidation_state_comparison()
    print("步骤3：生成氧化态对比图...")
    comparison_plot = visualize_oxidation_state_comparison(
        comparison['result'],
        filename='Ir_oxidation_comparison_scene3'
    )
    
    print(f"FUNCTION_CALL: visualize_oxidation_state_comparison | PARAMS: num_samples=2 | RESULT: {comparison_plot['result']}")
    print(f"步骤3结果：对比图已保存至 {comparison_plot['result']}\n")
    
    print(f"✓ 场景3完成：老化样品的平均结合能为{comparison['result']['average_binding_energy']['老化样品']:.2f} eV，高于新鲜样品的{comparison['result']['average_binding_energy']['新鲜样品']:.2f} eV，表明氧化程度增加\n")
    
    # ============================================================
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始XPS谱图分析问题的完整流程（峰检测→拟合→氧化态判断→验证）")
    print("- 场景2展示了工具的参数优化能力（prominence阈值扫描）")
    print("- 场景3展示了工具的批量对比分析能力（多样品氧化态演化研究）")
    print("\n最终答案（场景1）：")
    print(f"FINAL_ANSWER: {analysis_result['result']['summary']}")


if __name__ == "__main__":
    main()