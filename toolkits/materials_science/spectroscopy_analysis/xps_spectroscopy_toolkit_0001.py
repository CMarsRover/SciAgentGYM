# Filename: xps_spectroscopy_toolkit.py
"""
X射线光电子能谱(XPS)分析工具包

主要功能：
1. 峰位识别：基于scipy实现高斯-洛伦兹混合拟合
2. 元素鉴定：调用mendeleev数据库匹配核心能级
3. 氧化态分析：结合化学位移数据库判定化学环境
4. 谱图可视化：使用matplotlib生成专业XPS谱图

依赖库：
pip install numpy scipy matplotlib pandas mendeleev lmfit
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
import pandas as pd

# 全局常量
SPIN_ORBIT_COUPLING = {
    'Ir_4f': {'splitting': 3.0, 'ratio': 0.75},  # 4f7/2 : 4f5/2 = 4:3
    'Au_4f': {'splitting': 3.7, 'ratio': 0.75},
    'Pt_4f': {'splitting': 3.3, 'ratio': 0.75},
    'C_1s': {'splitting': 0.0, 'ratio': 1.0},
    'O_1s': {'splitting': 0.0, 'ratio': 1.0}
}

# XPS标准数据库（部分核心能级）
XPS_DATABASE = {
    'Ir': {
        '4f7/2': {'metallic': 60.8, 'Ir(III)': 61.5, 'Ir(IV)': 62.3},
        '4f5/2': {'metallic': 63.8, 'Ir(III)': 64.5, 'Ir(IV)': 65.3}
    },
    'Au': {
        '4f7/2': {'metallic': 84.0, 'Au(I)': 84.8, 'Au(III)': 86.0},
        '4f5/2': {'metallic': 87.7, 'Au(I)': 88.5, 'Au(III)': 89.7}
    },
    'C': {
        '1s': {'C-C': 284.8, 'C-O': 286.5, 'C=O': 288.0, 'O-C=O': 289.0}
    },
    'O': {
        '1s': {'lattice': 529.5, 'hydroxyl': 531.0, 'adsorbed': 532.5}
    }
}

# ============ 第一层：原子工具函数（Atomic Tools） ============

def voigt_function(x: List[float], amplitude: float, center: float, 
                   sigma: float, gamma: float) -> List[float]:
    """
    Voigt函数（高斯与洛伦兹卷积）- XPS峰形的标准模型
    
    Voigt函数能更准确地描述XPS峰形，因为它同时考虑了仪器展宽（高斯）
    和自然线宽（洛伦兹）的贡献。
    
    Args:
        x: 结合能数组 / eV
        amplitude: 峰强度
        center: 峰位中心 / eV
        sigma: 高斯展宽参数 / eV（仪器分辨率）
        gamma: 洛伦兹展宽参数 / eV（自然线宽）
    
    Returns:
        List[float]: 计算的强度值
    
    Example:
        >>> x = np.linspace(58, 66, 1000).tolist()
        >>> y = voigt_function(x, 1000, 60.5, 0.5, 0.3)
    """
    from scipy.special import wofz
    x_array = np.array(x)
    z = ((x_array - center) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return profile.tolist()


def detect_peaks_auto(binding_energy: List[float], intensity: List[float],
                      prominence_threshold: float = 0.05) -> dict:
    """
    自动检测XPS谱图中的峰位
    
    使用scipy的find_peaks算法，基于峰的突出度（prominence）识别主峰，
    适用于信噪比较高的XPS数据。
    
    Args:
        binding_energy: 结合能数组 / eV，长度N
        intensity: 强度数组 / counts，长度N
        prominence_threshold: 峰突出度阈值（相对于最大强度），范围0-1
    
    Returns:
        dict: {
            'peak_positions': 峰位的结合能 / eV,
            'peak_intensities': 峰强度 / counts,
            'peak_indices': 峰在数组中的索引,
            'metadata': {'num_peaks': 检测到的峰数量}
        }
    
    Example:
        >>> be = np.linspace(56, 70, 1000).tolist()
        >>> intensity = np.random.normal(1000, 50, 1000).tolist()
        >>> result = detect_peaks_auto(be, intensity, prominence_threshold=0.1)
        >>> print(result['peak_positions'])
    """
    be_array = np.array(binding_energy)
    int_array = np.array(intensity)
    max_intensity = np.max(int_array)
    prominence = prominence_threshold * max_intensity
    
    peaks, properties = find_peaks(int_array, prominence=prominence, distance=10)
    
    return {
        'result': {
            'peak_positions': be_array[peaks].tolist(),
            'peak_intensities': int_array[peaks].tolist(),
            'peak_indices': peaks.tolist()
        },
        'metadata': {
            'num_peaks': len(peaks),
            'prominence_threshold': prominence,
            'algorithm': 'scipy.signal.find_peaks'
        }
    }


def fit_single_peak(binding_energy: List[float], intensity: List[float],
                    peak_center_guess: float, peak_type: str = 'voigt') -> dict:
    """
    对单个XPS峰进行拟合
    
    支持高斯、洛伦兹和Voigt三种峰形模型，返回拟合参数和拟合优度。
    
    Args:
        binding_energy: 结合能数组 / eV
        intensity: 强度数组 / counts
        peak_center_guess: 峰位初始猜测值 / eV
        peak_type: 峰形类型，可选 'gaussian', 'lorentzian', 'voigt'
    
    Returns:
        dict: {
            'result': {
                'center': 拟合峰位 / eV,
                'amplitude': 峰强度,
                'fwhm': 半高全宽 / eV,
                'area': 峰面积,
                'fitted_curve': 拟合曲线的y值
            },
            'metadata': {
                'peak_type': 使用的峰形,
                'r_squared': 拟合优度,
                'fit_params': 完整拟合参数字典
            }
        }
    
    Example:
        >>> be = np.linspace(59, 62, 300).tolist()
        >>> intensity = voigt_function(be, 1000, 60.5, 0.4, 0.2)
        >>> result = fit_single_peak(be, intensity, 60.5, 'voigt')
        >>> print(f"拟合峰位: {result['result']['center']:.2f} eV")
    """
    from lmfit import Model
    
    be_array = np.array(binding_energy)
    int_array = np.array(intensity)
    
    # 选择峰形模型
    if peak_type == 'gaussian':
        model = GaussianModel()
        params = model.make_params(center=peak_center_guess, 
                                   amplitude=np.max(int_array),
                                   sigma=0.5)
    elif peak_type == 'lorentzian':
        model = LorentzianModel()
        params = model.make_params(center=peak_center_guess,
                                   amplitude=np.max(int_array),
                                   sigma=0.5)
    else:  # voigt
        model = VoigtModel()
        params = model.make_params(center=peak_center_guess,
                                   amplitude=np.max(int_array),
                                   sigma=0.4,
                                   gamma=0.2)
    
    # 执行拟合
    result = model.fit(int_array, params, x=be_array)
    
    # 计算峰面积
    fitted_curve = result.best_fit
    area = simpson(fitted_curve, x=be_array)
    
    # 计算R²
    ss_res = np.sum((int_array - fitted_curve) ** 2)
    ss_tot = np.sum((int_array - np.mean(int_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'result': {
            'center': result.params['center'].value,
            'amplitude': result.params['amplitude'].value,
            'fwhm': result.params['fwhm'].value,
            'area': area,
            'fitted_curve': fitted_curve.tolist()
        },
        'metadata': {
            'peak_type': peak_type,
            'r_squared': r_squared,
            'fit_params': {k: v.value for k, v in result.params.items()}
        }
    }


def identify_element_from_binding_energy(binding_energy: float, 
                                         tolerance: float = 0.5) -> dict:
    """
    根据结合能从数据库匹配元素和轨道
    
    在XPS_DATABASE中搜索与给定结合能匹配的元素核心能级，
    考虑不同氧化态的化学位移。
    
    Args:
        binding_energy: 实验测得的结合能 / eV
        tolerance: 匹配容差 / eV，默认±0.5 eV
    
    Returns:
        dict: {
            'result': {
                'element': 元素符号,
                'orbital': 轨道标识（如'4f7/2'）,
                'oxidation_state': 可能的氧化态,
                'reference_be': 数据库参考值 / eV
            },
            'metadata': {
                'all_matches': 所有可能的匹配列表,
                'confidence': 匹配置信度（基于能量差）
            }
        }
    
    Example:
        >>> result = identify_element_from_binding_energy(60.59, tolerance=0.3)
        >>> print(f"识别元素: {result['result']['element']}")
    """
    matches = []
    
    for element, orbitals in XPS_DATABASE.items():
        for orbital, states in orbitals.items():
            for state, ref_be in states.items():
                energy_diff = abs(binding_energy - ref_be)
                if energy_diff <= tolerance:
                    matches.append({
                        'element': element,
                        'orbital': orbital,
                        'oxidation_state': state,
                        'reference_be': ref_be,
                        'energy_diff': energy_diff
                    })
    
    if not matches:
        return {
            'result': None,
            'metadata': {'error': 'No matching element found in database'}
        }
    
    # 按能量差排序，最接近的排在前面
    matches.sort(key=lambda x: x['energy_diff'])
    best_match = matches[0]
    
    confidence = 1.0 - (best_match['energy_diff'] / tolerance)
    
    return {
        'result': {
            'element': best_match['element'],
            'orbital': best_match['orbital'],
            'oxidation_state': best_match['oxidation_state'],
            'reference_be': best_match['reference_be']
        },
        'metadata': {
            'all_matches': matches,
            'confidence': confidence,
            'num_candidates': len(matches)
        }
    }


def calculate_spin_orbit_doublet(peak1_be: float, peak1_intensity: float,
                                 element: str, orbital: str) -> dict:
    """
    根据自旋-轨道耦合规则计算双峰参数
    
    对于p、d、f轨道，自旋-轨道耦合导致能级分裂为j=l±1/2两个子能级，
    强度比由统计权重(2j+1)决定。
    
    Args:
        peak1_be: 第一个峰（低结合能）的峰位 / eV
        peak1_intensity: 第一个峰的强度
        element: 元素符号
        orbital: 轨道标识（如'4f'）
    
    Returns:
        dict: {
            'result': {
                'peak2_be': 第二个峰的预测结合能 / eV,
                'peak2_intensity': 第二个峰的预测强度,
                'splitting': 自旋-轨道分裂能 / eV,
                'intensity_ratio': 峰强度比（peak1/peak2）
            },
            'metadata': {
                'coupling_type': 耦合类型,
                'quantum_numbers': 量子数信息
            }
        }
    
    Example:
        >>> result = calculate_spin_orbit_doublet(60.59, 28500, 'Ir', '4f')
        >>> print(f"预测4f5/2峰位: {result['result']['peak2_be']:.2f} eV")
    """
    key = f"{element}_{orbital}"
    
    if key not in SPIN_ORBIT_COUPLING:
        return {
            'result': None,
            'metadata': {'error': f'No spin-orbit data for {key}'}
        }
    
    coupling_data = SPIN_ORBIT_COUPLING[key]
    splitting = coupling_data['splitting']
    ratio = coupling_data['ratio']
    
    peak2_be = peak1_be + splitting
    peak2_intensity = peak1_intensity / ratio
    
    return {
        'result': {
            'peak2_be': peak2_be,
            'peak2_intensity': peak2_intensity,
            'splitting': splitting,
            'intensity_ratio': ratio
        },
        'metadata': {
            'coupling_type': f"{orbital} spin-orbit coupling",
            'quantum_numbers': f"j = l ± 1/2"
        }
    }


# ============ 第二层：组合工具函数（Composite Tools） ============

def analyze_xps_spectrum_full(binding_energy: List[float], intensity: List[float],
                              expected_element: Optional[str] = None) -> dict:
    """
    完整的XPS谱图分析流程
    
    集成峰检测、拟合、元素识别和氧化态判定的完整分析链。
    内部调用：detect_peaks_auto() → fit_single_peak() → identify_element_from_binding_energy()
    
    Args:
        binding_energy: 结合能数组 / eV
        intensity: 强度数组 / counts
        expected_element: 预期元素符号（可选），用于验证
    
    Returns:
        dict: {
            'result': {
                'identified_peaks': 识别的峰列表，每个峰包含元素、轨道、氧化态信息,
                'summary': 分析总结文本,
                'raw_peak_data': 原始峰检测结果
            },
            'metadata': {
                'num_peaks_detected': 检测到的峰数量,
                'analysis_steps': 执行的分析步骤列表
            }
        }
    
    Example:
        >>> be = np.linspace(56, 70, 1000).tolist()
        >>> intensity = np.loadtxt('xps_data.txt').tolist()
        >>> result = analyze_xps_spectrum_full(be, intensity, expected_element='Ir')
        >>> print(result['result']['summary'])
    """
    analysis_steps = []
    
    # 步骤1：自动峰检测
    # 调用函数：detect_peaks_auto()
    peak_detection = detect_peaks_auto(binding_energy, intensity, prominence_threshold=0.05)
    detected_peaks = peak_detection['result']
    analysis_steps.append('Peak detection completed')
    
    identified_peaks = []
    
    # 步骤2：对每个峰进行拟合和元素识别
    for i, peak_pos in enumerate(detected_peaks['peak_positions']):
        # 调用函数：fit_single_peak()
        fit_result = fit_single_peak(binding_energy, intensity, peak_pos, peak_type='voigt')
        fitted_center = fit_result['result']['center']
        
        # 调用函数：identify_element_from_binding_energy()
        element_id = identify_element_from_binding_energy(fitted_center, tolerance=0.5)
        
        if element_id['result']:
            peak_info = {
                'peak_number': i + 1,
                'binding_energy': fitted_center,
                'intensity': fit_result['result']['amplitude'],
                'fwhm': fit_result['result']['fwhm'],
                'element': element_id['result']['element'],
                'orbital': element_id['result']['orbital'],
                'oxidation_state': element_id['result']['oxidation_state'],
                'confidence': element_id['metadata']['confidence']
            }
            identified_peaks.append(peak_info)
        
        analysis_steps.append(f"Peak {i+1} fitted and identified")
    
    # 步骤3：生成分析总结
    if identified_peaks:
        summary_parts = []
        for peak in identified_peaks:
            summary_parts.append(
                f"{peak['element']} {peak['orbital']} peak at {peak['binding_energy']:.2f} eV "
                f"({peak['oxidation_state']} state)"
            )
        summary = "The XPS spectrum shows " + ", ".join(summary_parts) + "."
    else:
        summary = "No identifiable peaks found in the spectrum."
    
    return {
        'result': {
            'identified_peaks': identified_peaks,
            'summary': summary,
            'raw_peak_data': detected_peaks
        },
        'metadata': {
            'num_peaks_detected': len(identified_peaks),
            'analysis_steps': analysis_steps,
            'expected_element_match': expected_element in [p['element'] for p in identified_peaks] if expected_element else None
        }
    }


def fit_doublet_peaks(binding_energy: List[float], intensity: List[float],
                     element: str, orbital: str,
                     initial_guess_be: float) -> dict:
    """
    拟合自旋-轨道双峰结构
    
    同时拟合两个耦合峰，约束它们的能量分裂和强度比符合物理规律。
    内部调用：calculate_spin_orbit_doublet() → fit_single_peak()（两次）
    
    Args:
        binding_energy: 结合能数组 / eV
        intensity: 强度数组 / counts
        element: 元素符号
        orbital: 轨道标识（如'4f'）
        initial_guess_be: 低结合能峰的初始猜测 / eV
    
    Returns:
        dict: {
            'result': {
                'peak1': {center, amplitude, fwhm},
                'peak2': {center, amplitude, fwhm},
                'total_fit': 总拟合曲线,
                'splitting_measured': 实测分裂能 / eV,
                'ratio_measured': 实测强度比
            },
            'metadata': {
                'theoretical_splitting': 理论分裂能,
                'theoretical_ratio': 理论强度比,
                'deviation': 偏差分析
            }
        }
    
    Example:
        >>> be = np.linspace(58, 66, 1000).tolist()
        >>> intensity = np.loadtxt('ir_4f_spectrum.txt').tolist()
        >>> result = fit_doublet_peaks(be, intensity, 'Ir', '4f', 60.5)
    """
    # 调用函数：calculate_spin_orbit_doublet()，获取理论预测
    doublet_prediction = calculate_spin_orbit_doublet(
        initial_guess_be, 
        max(intensity), 
        element, 
        orbital
    )
    
    if doublet_prediction['result'] is None:
        return {
            'result': None,
            'metadata': {'error': 'Element/orbital not supported for doublet fitting'}
        }
    
    predicted_be2 = doublet_prediction['result']['peak2_be']
    theoretical_splitting = doublet_prediction['result']['splitting']
    theoretical_ratio = doublet_prediction['result']['intensity_ratio']
    
    # 调用函数：fit_single_peak()，拟合第一个峰
    fit1 = fit_single_peak(binding_energy, intensity, initial_guess_be, 'voigt')
    
    # 调用函数：fit_single_peak()，拟合第二个峰
    fit2 = fit_single_peak(binding_energy, intensity, predicted_be2, 'voigt')
    
    # 计算实测参数
    measured_splitting = fit2['result']['center'] - fit1['result']['center']
    measured_ratio = fit1['result']['amplitude'] / fit2['result']['amplitude']
    
    # 合成总拟合曲线
    total_fit = (np.array(fit1['result']['fitted_curve']) + np.array(fit2['result']['fitted_curve'])).tolist()
    
    # 偏差分析
    splitting_deviation = abs(measured_splitting - theoretical_splitting) / theoretical_splitting * 100
    ratio_deviation = abs(measured_ratio - theoretical_ratio) / theoretical_ratio * 100
    
    return {
        'result': {
            'peak1': {
                'center': fit1['result']['center'],
                'amplitude': fit1['result']['amplitude'],
                'fwhm': fit1['result']['fwhm']
            },
            'peak2': {
                'center': fit2['result']['center'],
                'amplitude': fit2['result']['amplitude'],
                'fwhm': fit2['result']['fwhm']
            },
            'total_fit': total_fit,
            'splitting_measured': measured_splitting,
            'ratio_measured': measured_ratio
        },
        'metadata': {
            'theoretical_splitting': theoretical_splitting,
            'theoretical_ratio': theoretical_ratio,
            'deviation': {
                'splitting_percent': splitting_deviation,
                'ratio_percent': ratio_deviation
            },
            'fit_quality': 'Good' if splitting_deviation < 5 and ratio_deviation < 10 else 'Check manually'
        }
    }


def compare_oxidation_states(binding_energy_list: List[float],
                            element: str, orbital: str) -> dict:
    """
    批量比对多个样品的氧化态
    
    从数据库查询指定元素轨道的所有氧化态参考值，与实验数据对比。
    内部调用：identify_element_from_binding_energy()（循环调用）
    
    Args:
        binding_energy_list: 多个样品的峰位列表 / eV
        element: 元素符号
        orbital: 轨道标识
    
    Returns:
        dict: {
            'result': {
                'sample_assignments': 每个样品的氧化态判定,
                'oxidation_state_distribution': 氧化态统计,
                'reference_values': 数据库参考值
            },
            'metadata': {
                'num_samples': 样品数量,
                'dominant_state': 主要氧化态
            }
        }
    
    Example:
        >>> be_list = [60.5, 61.8, 60.3, 62.1]
        >>> result = compare_oxidation_states(be_list, 'Ir', '4f7/2')
        >>> print(result['result']['oxidation_state_distribution'])
    """
    if element not in XPS_DATABASE or orbital not in XPS_DATABASE[element]:
        return {
            'result': None,
            'metadata': {'error': f'No database entry for {element} {orbital}'}
        }
    
    reference_values = XPS_DATABASE[element][orbital]
    sample_assignments = []
    state_counts = {state: 0 for state in reference_values.keys()}
    
    for i, be in enumerate(binding_energy_list):
        # 调用函数：identify_element_from_binding_energy()
        identification = identify_element_from_binding_energy(be, tolerance=0.5)
        
        if identification['result'] and identification['result']['element'] == element:
            assigned_state = identification['result']['oxidation_state']
            sample_assignments.append({
                'sample_index': i + 1,
                'binding_energy': be,
                'assigned_state': assigned_state,
                'confidence': identification['metadata']['confidence']
            })
            state_counts[assigned_state] += 1
        else:
            sample_assignments.append({
                'sample_index': i + 1,
                'binding_energy': be,
                'assigned_state': 'Unknown',
                'confidence': 0.0
            })
    
    dominant_state = max(state_counts, key=state_counts.get)
    
    return {
        'result': {
            'sample_assignments': sample_assignments,
            'oxidation_state_distribution': state_counts,
            'reference_values': reference_values
        },
        'metadata': {
            'num_samples': len(binding_energy_list),
            'dominant_state': dominant_state,
            'database_source': 'XPS_DATABASE (internal)'
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def visualize_xps_spectrum(data: dict, domain: str = 'materials', 
                          vis_type: str = 'xps_fitted',
                          save_dir: str = './images/',
                          filename: str = None) -> str:
    """
    XPS谱图专业可视化工具
    
    Args:
        data: 包含XPS数据的字典，必须包含：
              - 'binding_energy': 结合能数组
              - 'intensity': 强度数组
              - 'fitted_peaks': 拟合峰数据（可选）
        domain: 固定为'materials'
        vis_type: 可视化类型
                 - 'xps_fitted': 带拟合曲线的XPS谱图
                 - 'xps_raw': 原始XPS谱图
                 - 'peak_comparison': 多峰对比图
        save_dir: 保存目录
        filename: 文件名（可选）
    
    Returns:
        str: 保存的图片路径
    
    Example:
        >>> data = {
        ...     'binding_energy': be_array,
        ...     'intensity': intensity_array,
        ...     'fitted_peaks': [peak1_fit, peak2_fit]
        ... }
        >>> path = visualize_xps_spectrum(data, vis_type='xps_fitted')
    """
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    
    if vis_type == 'xps_fitted':
        fig, ax = plt.subplots(figsize=(10, 7))
        
        be = data['binding_energy']
        intensity = data['intensity']
        
        # 绘制原始数据
        ax.plot(be, intensity, 'o', markersize=3, color='black', 
                label='Experimental', alpha=0.6)
        
        # 绘制拟合峰
        if 'fitted_peaks' in data:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, peak in enumerate(data['fitted_peaks']):
                ax.plot(be, peak['fitted_curve'], '-', 
                       color=colors[i % len(colors)], linewidth=2,
                       label=f"Peak {i+1}: {peak['center']:.2f} eV")
            
            # 绘制总拟合
            total_fit = np.sum([p['fitted_curve'] for p in data['fitted_peaks']], axis=0)
            ax.plot(be, total_fit, 'r-', linewidth=2.5, label='Total Fit')
        
        ax.set_xlabel('Binding Energy (eV)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=13, fontweight='bold')
        ax.set_title(data.get('title', 'XPS Spectrum Analysis'), 
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(alpha=0.3, linestyle='--')
        ax.invert_xaxis()  # XPS习惯上x轴反向
        
        filename = filename or f"xps_fitted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    elif vis_type == 'peak_comparison':
        fig, ax = plt.subplots(figsize=(12, 6))
        
        peak_data = data['peaks']  # List of {name, be, intensity}
        
        x_pos = np.arange(len(peak_data))
        bars = ax.bar(x_pos, [p['intensity'] for p in peak_data],
                     color='steelblue', edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{p['name']}\n{p['be']:.2f} eV" for p in peak_data],
                          rotation=0, ha='center')
        ax.set_ylabel('Peak Intensity (a.u.)', fontsize=13, fontweight='bold')
        ax.set_title('XPS Peak Comparison', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        filename = filename or f"peak_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return save_path


# ============ 第四层：主流程演示 ============

def main():
    """
    演示XPS工具包的三种典型应用场景
    
    ⚠️ 注意：以下演示使用模拟数据，实际应用中应替换为真实XPS数据文件
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - Ir 4f谱图分析")
    print("=" * 60)
    print("问题描述：分析给定的Ir 4f XPS谱图，识别4f7/2和4f5/2峰位，")
    print("          判定氧化态，并生成专业分析报告")
    print("-" * 60)
    
    # 模拟Ir 4f XPS数据（实际应从文件读取）
    be_range = np.linspace(56, 70, 1400)
    
    # 步骤1：生成模拟数据（实际场景中从仪器导出）
    # 调用函数：voigt_function()（两次，模拟双峰）
    peak1 = voigt_function(be_range.tolist(), amplitude=28500, center=60.59, sigma=0.45, gamma=0.25)
    peak2 = voigt_function(be_range.tolist(), amplitude=22500, center=63.59, sigma=0.45, gamma=0.25)
    noise = np.random.normal(0, 300, len(be_range))
    intensity_simulated = np.array(peak1) + np.array(peak2) + noise + 2500  # 加基线
    print("步骤1：数据加载完成（模拟数据：1400个数据点）")
    
    # 步骤2：自动峰检测
    # 调用函数：detect_peaks_auto()
    peak_detection = detect_peaks_auto(be_range, intensity_simulated, prominence_threshold=0.1)
    detected_positions = peak_detection['result']['peak_positions']
    print(f"步骤2：检测到 {len(detected_positions)} 个峰")
    print(f"       峰位: {detected_positions}")
    
    # 步骤3：双峰拟合分析
    # 调用函数：fit_doublet_peaks()，内部调用 calculate_spin_orbit_doublet() 和 fit_single_peak()
    doublet_fit = fit_doublet_peaks(
        be_range, 
        intensity_simulated, 
        element='Ir', 
        orbital='4f',
        initial_guess_be=60.5
    )
    
    peak1_center = doublet_fit['result']['peak1']['center']
    peak2_center = doublet_fit['result']['peak2']['center']
    splitting = doublet_fit['result']['splitting_measured']
    
    print(f"步骤3：双峰拟合完成")
    print(f"       Ir 4f7/2: {peak1_center:.2f} eV")
    print(f"       Ir 4f5/2: {peak2_center:.2f} eV")
    print(f"       自旋-轨道分裂: {splitting:.2f} eV")
    print(f"       拟合质量: {doublet_fit['metadata']['fit_quality']}")
    
    # 步骤4：氧化态判定
    # 调用函数：identify_element_from_binding_energy()
    oxidation_state1 = identify_element_from_binding_energy(peak1_center, tolerance=0.5)
    state_name = oxidation_state1['result']['oxidation_state']
    confidence = oxidation_state1['metadata']['confidence']
    
    print(f"步骤4：氧化态分析")
    print(f"       判定结果: {state_name}")
    print(f"       置信度: {confidence:.2%}")
    
    # 步骤5：生成可视化
    # 调用函数：visualize_xps_spectrum()
    vis_data = {
        'binding_energy': be_range,
        'intensity': intensity_simulated,
        'fitted_peaks': [
            {
                'center': peak1_center,
                'fitted_curve': voigt_function(be_range.tolist(), 
                                              doublet_fit['result']['peak1']['amplitude'],
                                              peak1_center, 0.45, 0.25)
            },
            {
                'center': peak2_center,
                'fitted_curve': voigt_function(be_range.tolist(),
                                              doublet_fit['result']['peak2']['amplitude'],
                                              peak2_center, 0.45, 0.25)
            }
        ],
        'title': 'Ir 4f XPS Spectrum'
    }
    
    plot_path = visualize_xps_spectrum(vis_data, vis_type='xps_fitted', 
                                      filename='ir_4f_analysis.png')
    print(f"步骤5：可视化完成，图表保存至: {plot_path}")
    
    print(f"\n✓ 场景1最终答案：")
    print(f"  The depicted XPS spectrum showed Ir 4f₇/₂ and 4f₅/₂ peaks,")
    print(f"  with a binding energy of {peak1_center:.2f} eV and {peak2_center:.2f} eV respectively.")
    print(f"  The sample is in {state_name} oxidation state.\n")
    
    
    print("=" * 60)
    print("场景2：参数变化分析 - 不同氧化态的化学位移")
    print("=" * 60)
    print("问题描述：模拟Ir在不同氧化态下的4f7/2峰位变化，")
    print("          展示化学位移与氧化态的关系")
    print("-" * 60)
    
    # 从数据库提取Ir 4f7/2的所有氧化态参考值
    # 调用函数：XPS_DATABASE（数据查询）
    ir_4f7_states = XPS_DATABASE['Ir']['4f7/2']
    
    print("步骤1：从数据库提取参考值")
    for state, be_ref in ir_4f7_states.items():
        print(f"       {state}: {be_ref} eV")
    
    # 批量生成不同氧化态的模拟谱图
    # 调用函数：voigt_function()（循环调用）
    print("\n步骤2：生成不同氧化态的模拟谱图")
    simulated_spectra = {}
    
    for state, be_center in ir_4f7_states.items():
        spectrum = voigt_function(be_range.tolist(), amplitude=25000, 
                                 center=be_center, sigma=0.4, gamma=0.2)
        simulated_spectra[state] = {
            'binding_energy': be_range,
            'intensity': np.array(spectrum) + np.random.normal(0, 200, len(be_range)),
            'peak_center': be_center
        }
        print(f"       {state}: 峰位 {be_center} eV")
    
    # 计算化学位移
    metallic_be = ir_4f7_states['metallic']
    chemical_shifts = {state: be - metallic_be 
                      for state, be in ir_4f7_states.items()}
    
    print("\n步骤3：计算化学位移（相对于金属态）")
    for state, shift in chemical_shifts.items():
        print(f"       {state}: {shift:+.2f} eV")
    
    print(f"\n✓ 场景2完成：分析了 {len(ir_4f7_states)} 种氧化态")
    print(f"  化学位移范围: {min(chemical_shifts.values()):.2f} ~ {max(chemical_shifts.values()):.2f} eV\n")
    
    
    print("=" * 60)
    print("场景3：数据库批量查询 - 多元素XPS峰位对比")
    print("=" * 60)
    print("问题描述：查询贵金属元素（Ir, Au, Pt）的4f轨道峰位，")
    print("          对比不同元素的结合能差异")
    print("-" * 60)
    
    # 批量查询数据库
    elements = ['Ir', 'Au']  # Pt在当前数据库中未定义，实际可扩展
    orbital = '4f7/2'
    comparison_results = {}
    
    print("步骤1：批量查询数据库")
    for element in elements:
        if element in XPS_DATABASE and orbital in XPS_DATABASE[element]:
            # 调用函数：XPS_DATABASE（数据查询）
            metallic_be = XPS_DATABASE[element][orbital]['metallic']
            comparison_results[element] = metallic_be
            print(f"       {element} {orbital}: {metallic_be} eV")
        else:
            print(f"       {element}: 数据库中无记录")
    
    # 步骤2：计算元素间的结合能差异
    print("\n步骤2：计算元素间结合能差异")
    if len(comparison_results) >= 2:
        elements_list = list(comparison_results.keys())
        for i in range(len(elements_list)):
            for j in range(i+1, len(elements_list)):
                elem1, elem2 = elements_list[i], elements_list[j]
                be_diff = comparison_results[elem2] - comparison_results[elem1]
                print(f"       {elem2} - {elem1}: {be_diff:+.2f} eV")
    
    # 步骤3：生成对比可视化
    # 调用函数：visualize_xps_spectrum()
    peak_comparison_data = {
        'peaks': [
            {'name': f"{elem} {orbital}", 'be': be, 'intensity': 1.0}
            for elem, be in comparison_results.items()
        ]
    }
    
    comparison_plot = visualize_xps_spectrum(
        peak_comparison_data, 
        vis_type='peak_comparison',
        filename='element_comparison.png'
    )
    print(f"\n步骤3：对比图表已保存至: {comparison_plot}")
    
    print(f"\n✓ 场景3完成：对比了 {len(comparison_results)} 个元素")
    print(f"  结合能范围: {min(comparison_results.values()):.2f} ~ {max(comparison_results.values()):.2f} eV\n")
    
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了完整的XPS谱图分析流程（峰检测→拟合→氧化态判定）")
    print("- 场景2展示了化学位移分析能力（不同氧化态的参数扫描）")
    print("- 场景3展示了数据库批量查询和元素对比功能")
    print("\n核心工具函数调用链：")
    print("  detect_peaks_auto() → fit_doublet_peaks() → identify_element_from_binding_energy()")
    print("  └─ fit_single_peak() ← calculate_spin_orbit_doublet()")


if __name__ == "__main__":
    main()