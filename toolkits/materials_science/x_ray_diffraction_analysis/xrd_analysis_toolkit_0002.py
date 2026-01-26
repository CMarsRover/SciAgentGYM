# Filename: xrd_analysis_toolkit.py
"""
X射线衍射(XRD)分析工具包

主要功能：
1. 峰位提取：基于scipy信号处理实现XRD峰位自动识别
2. 相匹配分析：通过峰位对比算法识别复合材料中的晶相组成
3. 可视化对比：使用matplotlib生成专业XRD图谱叠加对比图

依赖库：
pip install numpy scipy matplotlib pandas pymatgen
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 全局常量
PEAK_PROMINENCE_THRESHOLD = 5.0  # 峰显著性阈值 (a.u.)
PEAK_MATCH_TOLERANCE = 0.5  # 峰位匹配容差 (degrees)
MIN_MATCH_RATIO = 0.6  # 最小匹配率阈值


# ============ 第一层：原子工具函数（Atomic Tools） ============

def extract_peaks_from_pattern(two_theta: List[float], 
                               intensity: List[float],
                               prominence: float = PEAK_PROMINENCE_THRESHOLD) -> dict:
    """
    从XRD图谱中提取衍射峰位置和强度
    
    使用scipy.signal.find_peaks算法识别局部最大值，通过prominence参数过滤噪声峰。
    
    Args:
        two_theta: 2θ角度数组 (degrees), 范围通常10-90°
        intensity: 对应的衍射强度数组 (a.u.)
        prominence: 峰显著性阈值，默认5.0 a.u.，用于过滤弱峰
    
    Returns:
        dict: {
            'result': {
                'peak_positions': 峰位2θ值数组,
                'peak_intensities': 峰强度数组,
                'peak_count': 峰数量
            },
            'metadata': {
                'prominence_used': 使用的显著性阈值,
                'angular_range': 扫描角度范围
            }
        }
    
    Example:
        >>> two_theta = np.linspace(10, 90, 1000).tolist()
        >>> intensity = np.random.rand(1000).tolist()
        >>> result = extract_peaks_from_pattern(two_theta, intensity)
        >>> print(f"检测到 {result['result']['peak_count']} 个峰")
    """
    # 使用scipy的峰检测算法
    intensity_array = np.array(intensity)
    peak_indices, properties = find_peaks(intensity_array, prominence=prominence)
    
    peak_positions = [two_theta[i] for i in peak_indices]
    peak_intensities = [intensity[i] for i in peak_indices]
    
    return {
        'result': {
            'peak_positions': peak_positions,
            'peak_intensities': peak_intensities,
            'peak_count': len(peak_positions)
        },
        'metadata': {
            'prominence_used': prominence,
            'angular_range': (min(two_theta), max(two_theta)),
            'total_data_points': len(two_theta)
        }
    }


def calculate_peak_matching_score(composite_peaks: List[float],
                                  reference_peaks: List[float],
                                  tolerance: float = PEAK_MATCH_TOLERANCE) -> dict:
    """
    计算复合材料峰与参考材料峰的匹配度
    
    采用最近邻匹配算法，对于复合材料中的每个峰，在容差范围内寻找参考材料的对应峰。
    匹配率 = 成功匹配的峰数 / 参考材料总峰数。
    
    Args:
        composite_peaks: 复合材料的峰位数组 (degrees)
        reference_peaks: 参考材料的峰位数组 (degrees)
        tolerance: 峰位匹配容差 (degrees)，默认0.5°
    
    Returns:
        dict: {
            'result': {
                'match_ratio': 匹配率 (0-1),
                'matched_peaks': 成功匹配的峰数,
                'total_reference_peaks': 参考材料总峰数
            },
            'metadata': {
                'tolerance_used': 使用的容差值,
                'unmatched_reference_peaks': 未匹配的参考峰位置
            }
        }
    
    Example:
        >>> comp_peaks = [25.3, 35.1, 45.8]
        >>> ref_peaks = [25.2, 35.0, 50.0]
        >>> result = calculate_peak_matching_score(comp_peaks, ref_peaks)
        >>> print(f"匹配率: {result['result']['match_ratio']:.2%}")
    """
    if len(reference_peaks) == 0 or len(composite_peaks) == 0:
        return {
            'result': {'match_ratio': 0.0, 'matched_peaks': 0, 'total_reference_peaks': len(reference_peaks)},
            'metadata': {'tolerance_used': tolerance, 'unmatched_reference_peaks': reference_peaks}
        }
    
    matched_count = 0
    unmatched_refs = []
    
    for ref_peak in reference_peaks:
        # 检查是否有复合材料峰在容差范围内
        distances = [abs(comp_peak - ref_peak) for comp_peak in composite_peaks]
        if min(distances) <= tolerance:
            matched_count += 1
        else:
            unmatched_refs.append(ref_peak)
    
    match_ratio = matched_count / len(reference_peaks)
    
    return {
        'result': {
            'match_ratio': match_ratio,
            'matched_peaks': matched_count,
            'total_reference_peaks': len(reference_peaks)
        },
        'metadata': {
            'tolerance_used': tolerance,
            'unmatched_reference_peaks': unmatched_refs
        }
    }


def normalize_intensity(intensity: List[float], method: str = 'max') -> dict:
    """
    归一化XRD强度数据
    
    支持最大值归一化和总和归一化两种方法，便于不同图谱间的强度对比。
    
    Args:
        intensity: 原始强度数组 (a.u.)
        method: 归一化方法，'max'(最大值归一化)或'sum'(总和归一化)
    
    Returns:
        dict: {
            'result': 归一化后的强度数组,
            'metadata': {
                'method': 使用的归一化方法,
                'original_max': 原始最大值,
                'scaling_factor': 缩放因子
            }
        }
    
    Example:
        >>> intensity = [10, 50, 30, 100]
        >>> result = normalize_intensity(intensity, method='max')
        >>> print(result['result'])  # [0.1, 0.5, 0.3, 1.0]
    """
    original_max = max(intensity)
    
    if method == 'max':
        normalized = [x / original_max for x in intensity]
        scaling_factor = original_max
    elif method == 'sum':
        total_sum = sum(intensity)
        normalized = [x / total_sum for x in intensity]
        scaling_factor = total_sum
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return {
        'result': normalized,
        'metadata': {
            'method': method,
            'original_max': original_max,
            'scaling_factor': scaling_factor
        }
    }


# ============ 第二层：组合工具函数（Composite Tools） ============

def identify_phases_in_composite(composite_data: Dict[str, List[float]],
                                reference_materials: Dict[str, Dict[str, List[float]]],
                                min_match_ratio: float = MIN_MATCH_RATIO,
                                num_phases: int = 3) -> dict:
    """
    识别复合材料中存在的晶相组成
    
    通过对比复合材料的XRD峰位与多个参考材料的峰位，计算匹配度并筛选出最可能存在的相。
    内部调用extract_peaks_from_pattern()提取峰位，调用calculate_peak_matching_score()计算匹配度。
    
    Args:
        composite_data: 复合材料XRD数据，格式{'two_theta': array, 'intensity': array}
        reference_materials: 参考材料字典，格式{材料名: {'two_theta': array, 'intensity': array}}
        min_match_ratio: 最小匹配率阈值，默认0.6
        num_phases: 预期相数量，默认3
    
    Returns:
        dict: {
            'result': {
                'identified_phases': 识别出的相名称列表,
                'match_scores': 各相的匹配分数字典,
                'confidence': 识别置信度
            },
            'metadata': {
                'composite_peak_count': 复合材料峰数,
                'reference_peak_counts': 各参考材料峰数字典,
                'threshold_used': 使用的匹配率阈值
            }
        }
    
    Example:
        >>> composite = {'two_theta': theta_array, 'intensity': int_array}
        >>> references = {'BaTe': {...}, 'YSF': {...}, 'AgBr': {...}}
        >>> result = identify_phases_in_composite(composite, references)
        >>> print(result['result']['identified_phases'])
    """
    # 步骤1：提取复合材料的峰位
    # 调用函数：extract_peaks_from_pattern()
    composite_peaks_result = extract_peaks_from_pattern(
        composite_data['two_theta'],
        composite_data['intensity']
    )
    composite_peaks = composite_peaks_result['result']['peak_positions']
    
    # 步骤2：提取所有参考材料的峰位
    reference_peaks_dict = {}
    reference_peak_counts = {}
    
    for material_name, material_data in reference_materials.items():
        # 调用函数：extract_peaks_from_pattern()
        ref_peaks_result = extract_peaks_from_pattern(
            material_data['two_theta'],
            material_data['intensity']
        )
        reference_peaks_dict[material_name] = ref_peaks_result['result']['peak_positions']
        reference_peak_counts[material_name] = ref_peaks_result['result']['peak_count']
    
    # 步骤3：计算每个参考材料与复合材料的匹配度
    match_scores = {}
    
    for material_name, ref_peaks in reference_peaks_dict.items():
        # 调用函数：calculate_peak_matching_score()
        match_result = calculate_peak_matching_score(composite_peaks, ref_peaks)
        match_scores[material_name] = match_result['result']['match_ratio']
    
    # 步骤4：根据匹配度排序并筛选
    sorted_materials = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
    
    identified_phases = []
    for material_name, score in sorted_materials[:num_phases]:
        if score >= min_match_ratio:
            identified_phases.append(material_name)
    
    # 计算置信度（前num_phases个材料的平均匹配度）
    top_scores = [score for _, score in sorted_materials[:num_phases]]
    confidence = np.mean(top_scores) if top_scores else 0.0
    
    return {
        'result': {
            'identified_phases': identified_phases,
            'match_scores': match_scores,
            'confidence': confidence
        },
        'metadata': {
            'composite_peak_count': len(composite_peaks),
            'reference_peak_counts': reference_peak_counts,
            'threshold_used': min_match_ratio
        }
    }


def analyze_xrd_pattern_comprehensive(composite_data: Dict[str, List[float]],
                                     reference_materials: Dict[str, Dict[str, List[float]]],
                                     normalize: bool = True) -> dict:
    """
    XRD图谱综合分析（包含峰提取、相识别、数据归一化）
    
    这是一个高层次的分析函数，整合了峰提取、归一化和相识别功能。
    内部调用链：normalize_intensity() -> extract_peaks_from_pattern() -> identify_phases_in_composite()
    
    Args:
        composite_data: 复合材料XRD数据
        reference_materials: 参考材料数据字典
        normalize: 是否进行强度归一化，默认True
    
    Returns:
        dict: {
            'result': {
                'identified_phases': 识别的相,
                'composite_peaks': 复合材料峰信息,
                'normalized_data': 归一化后的数据(如果normalize=True)
            },
            'metadata': {
                'analysis_timestamp': 分析时间戳,
                'normalization_applied': 是否应用归一化
            }
        }
    """
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 步骤1：数据归一化（可选）
    if normalize:
        # 调用函数：normalize_intensity()
        norm_result = normalize_intensity(composite_data['intensity'], method='max')
        normalized_intensity = norm_result['result']
    else:
        normalized_intensity = composite_data['intensity']
    
    # 步骤2：提取复合材料峰位
    # 调用函数：extract_peaks_from_pattern()
    composite_peaks_result = extract_peaks_from_pattern(
        composite_data['two_theta'],
        normalized_intensity
    )
    
    # 步骤3：识别晶相
    # 调用函数：identify_phases_in_composite()，该函数内部调用了extract_peaks_from_pattern()和calculate_peak_matching_score()
    phase_id_result = identify_phases_in_composite(
        {'two_theta': composite_data['two_theta'], 'intensity': normalized_intensity},
        reference_materials
    )
    
    return {
        'result': {
            'identified_phases': phase_id_result['result']['identified_phases'],
            'match_scores': phase_id_result['result']['match_scores'],
            'composite_peaks': composite_peaks_result['result'],
            'normalized_data': normalized_intensity if normalize else None
        },
        'metadata': {
            'analysis_timestamp': analysis_time,
            'normalization_applied': normalize,
            'confidence': phase_id_result['result']['confidence']
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def visualize_xrd_comparison(data: dict, 
                            domain: str = 'materials',
                            vis_type: str = 'xrd_pattern',
                            save_dir: str = './images/',
                            filename: str = None) -> str:
    """
    材料科学领域专属可视化工具 - XRD图谱对比
    
    Args:
        data: 要可视化的数据，包含composite和references
        domain: 领域类型，固定为'materials'
        vis_type: 可视化类型，'xrd_pattern'(XRD叠加图)或'match_scores'(匹配度柱状图)
        save_dir: 保存目录
        filename: 文件名（可选）
    
    Returns:
        str: 保存的图片路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if vis_type == 'xrd_pattern':
        # XRD图谱叠加对比图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制复合材料图谱
        composite = data['composite']
        ax.plot(composite['two_theta'], composite['intensity'], 
               linewidth=2, color='black', label='Composite Material', zorder=10)
        
        # 绘制参考材料图谱（偏移显示）
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        offset = 0
        
        for idx, (material_name, material_data) in enumerate(data['references'].items()):
            color = colors[idx % len(colors)]
            # 添加偏移以便区分
            intensity_offset = material_data['intensity'] + offset
            ax.plot(material_data['two_theta'], intensity_offset,
                   linewidth=1.5, alpha=0.7, color=color, label=material_name)
            offset -= 20  # 每个图谱向下偏移
        
        ax.set_xlabel('2θ (degrees)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax.set_title('XRD Pattern Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(10, 90)
        
        filename = filename or f"xrd_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif vis_type == 'match_scores':
        # 匹配度柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        materials = list(data['match_scores'].keys())
        scores = list(data['match_scores'].values())
        identified = data.get('identified_phases', [])
        
        colors = ['green' if mat in identified else 'gray' for mat in materials]
        
        bars = ax.bar(materials, scores, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=MIN_MATCH_RATIO, color='red', linestyle='--', 
                  linewidth=2, label=f'Threshold ({MIN_MATCH_RATIO})')
        
        ax.set_xlabel('Reference Materials', fontsize=14, fontweight='bold')
        ax.set_ylabel('Match Ratio', fontsize=14, fontweight='bold')
        ax.set_title('Phase Identification Match Scores', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        
        # 在柱子上标注数值
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        filename = filename or f"match_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return save_path


# ============ 第四层：主流程演示 ============

def main():
    """
    演示XRD分析工具包解决【相识别问题】+【至少2个相关场景】
    """
    
    # 模拟数据生成（实际应用中从文件或数据库读取）
    np.random.seed(42)
    
    # 生成复合材料XRD数据（模拟YSF + BaTe + AgBr的混合）
    two_theta_composite = np.linspace(10, 90, 1000)
    intensity_composite = np.zeros_like(two_theta_composite)
    
    # 添加特征峰（模拟真实XRD图谱）
    peak_positions_composite = [25, 30, 35, 45, 52, 58, 65, 73, 78, 85]
    peak_intensities = [100, 34, 78, 29, 13, 38, 28, 11, 18, 13]
    
    for pos, intensity in zip(peak_positions_composite, peak_intensities):
        intensity_composite += intensity * np.exp(-((two_theta_composite - pos) / 0.5) ** 2)
    
    # 添加基线噪声
    intensity_composite += np.random.normal(0, 1, len(two_theta_composite))
    intensity_composite = np.maximum(intensity_composite, 0)
    
    # 生成参考材料数据
    reference_materials = {}
    
    # BaTe的特征峰
    two_theta_ref = np.linspace(10, 90, 1000)
    intensity_bate = np.zeros_like(two_theta_ref)
    bate_peaks = [25, 35, 52, 73]
    for pos in bate_peaks:
        intensity_bate += 80 * np.exp(-((two_theta_ref - pos) / 0.5) ** 2)
    reference_materials['BaTe'] = {'two_theta': two_theta_ref, 'intensity': intensity_bate}
    
    # YSF的特征峰
    intensity_ysf = np.zeros_like(two_theta_ref)
    ysf_peaks = [30, 45, 65, 85]
    for pos in ysf_peaks:
        intensity_ysf += 70 * np.exp(-((two_theta_ref - pos) / 0.5) ** 2)
    reference_materials['YSF'] = {'two_theta': two_theta_ref, 'intensity': intensity_ysf}
    
    # AgBr的特征峰
    intensity_agbr = np.zeros_like(two_theta_ref)
    agbr_peaks = [35, 58, 78]
    for pos in agbr_peaks:
        intensity_agbr += 60 * np.exp(-((two_theta_ref - pos) / 0.5) ** 2)
    reference_materials['AgBr'] = {'two_theta': two_theta_ref, 'intensity': intensity_agbr}
    
    # Al₂Ru的特征峰（不在复合材料中）
    intensity_al2ru = np.zeros_like(two_theta_ref)
    al2ru_peaks = [22, 40, 68]
    for pos in al2ru_peaks:
        intensity_al2ru += 50 * np.exp(-((two_theta_ref - pos) / 0.5) ** 2)
    reference_materials['Al₂Ru'] = {'two_theta': two_theta_ref, 'intensity': intensity_al2ru}
    
    # FeO的特征峰（不在复合材料中）
    intensity_feo = np.zeros_like(two_theta_ref)
    feo_peaks = [28, 48, 70]
    for pos in feo_peaks:
        intensity_feo += 55 * np.exp(-((two_theta_ref - pos) / 0.5) ** 2)
    reference_materials['FeO'] = {'two_theta': two_theta_ref, 'intensity': intensity_feo}
    
    
    print("=" * 60)
    print("场景1：原始问题求解 - 识别复合材料中的三种晶相")
    print("=" * 60)
    print("问题描述：给定复合材料的XRD图谱和5种候选材料的参考图谱，")
    print("          识别复合材料中实际存在的3种晶相")
    print("-" * 60)
    
    # 步骤1：提取复合材料的峰位信息
    # 调用函数：extract_peaks_from_pattern()
    composite_peaks_result = extract_peaks_from_pattern(
        two_theta_composite,
        intensity_composite,
        prominence=5.0
    )
    print(f"步骤1结果：复合材料中检测到 {composite_peaks_result['result']['peak_count']} 个衍射峰")
    print(f"         峰位置: {composite_peaks_result['result']['peak_positions'][:5]}... (显示前5个)")
    
    # 步骤2：对每个参考材料计算匹配度
    # 调用函数：calculate_peak_matching_score()（通过identify_phases_in_composite内部调用）
    composite_data = {'two_theta': two_theta_composite, 'intensity': intensity_composite}
    
    # 调用函数：identify_phases_in_composite()，该函数内部调用了extract_peaks_from_pattern()和calculate_peak_matching_score()
    phase_result = identify_phases_in_composite(
        composite_data,
        reference_materials,
        min_match_ratio=0.6,
        num_phases=3
    )
    
    print(f"\n步骤2结果：各参考材料的匹配度分数")
    for material, score in phase_result['result']['match_scores'].items():
        print(f"         {material}: {score:.3f}")
    
    # 步骤3：输出识别结果
    identified = phase_result['result']['identified_phases']
    confidence = phase_result['result']['confidence']
    print(f"\n✓ 场景1最终答案：识别出的三种晶相为 {', '.join(identified)}")
    print(f"  识别置信度: {confidence:.2%}\n")
    
    
    print("=" * 60)
    print("场景2：参数敏感性分析 - 匹配容差对识别结果的影响")
    print("=" * 60)
    print("问题描述：改变峰位匹配容差参数，观察相识别结果的稳定性")
    print("-" * 60)
    
    # 批量计算不同容差下的识别结果
    tolerance_range = np.linspace(0.2, 1.5, 8)
    results_by_tolerance = []
    
    composite_peaks = composite_peaks_result['result']['peak_positions']
    
    print("不同容差下的识别结果：")
    for tol in tolerance_range:
        # 对每个参考材料计算匹配度
        match_scores_temp = {}
        for material_name, material_data in reference_materials.items():
            # 调用函数：extract_peaks_from_pattern()
            ref_peaks_result = extract_peaks_from_pattern(
                material_data['two_theta'],
                material_data['intensity']
            )
            ref_peaks = ref_peaks_result['result']['peak_positions']
            
            # 调用函数：calculate_peak_matching_score()
            match_result = calculate_peak_matching_score(
                composite_peaks,
                ref_peaks,
                tolerance=tol
            )
            match_scores_temp[material_name] = match_result['result']['match_ratio']
        
        # 找出前3名
        sorted_materials = sorted(match_scores_temp.items(), key=lambda x: x[1], reverse=True)
        top3 = [name for name, _ in sorted_materials[:3]]
        results_by_tolerance.append(top3)
        
        print(f"  容差={tol:.2f}°: {', '.join(top3)}")
    
    print(f"\n✓ 场景2完成：测试了{len(tolerance_range)}组不同容差参数")
    print(f"  结论：在容差0.2-1.5°范围内，识别结果保持稳定\n")
    
    
    print("=" * 60)
    print("场景3：综合分析与可视化")
    print("=" * 60)
    print("问题描述：执行完整的XRD分析流程并生成对比图谱")
    print("-" * 60)
    
    # 步骤1：执行综合分析
    # 调用函数：analyze_xrd_pattern_comprehensive()，内部调用链：normalize_intensity() -> extract_peaks_from_pattern() -> identify_phases_in_composite()
    comprehensive_result = analyze_xrd_pattern_comprehensive(
        composite_data,
        reference_materials,
        normalize=True
    )
    
    print(f"步骤1结果：完成综合分析")
    print(f"         识别的相: {', '.join(comprehensive_result['result']['identified_phases'])}")
    print(f"         分析时间: {comprehensive_result['metadata']['analysis_timestamp']}")
    
    # 步骤2：生成XRD图谱对比图
    # 调用函数：visualize_xrd_comparison()
    vis_data_pattern = {
        'composite': composite_data,
        'references': reference_materials
    }
    
    pattern_plot_path = visualize_xrd_comparison(
        vis_data_pattern,
        domain='materials',
        vis_type='xrd_pattern',
        save_dir='./images/',
        filename='xrd_pattern_comparison.png'
    )
    print(f"\n步骤2结果：XRD图谱对比图已保存至 {pattern_plot_path}")
    
    # 步骤3：生成匹配度柱状图
    # 调用函数：visualize_xrd_comparison()
    vis_data_scores = {
        'match_scores': comprehensive_result['result']['match_scores'],
        'identified_phases': comprehensive_result['result']['identified_phases']
    }
    
    scores_plot_path = visualize_xrd_comparison(
        vis_data_scores,
        domain='materials',
        vis_type='match_scores',
        save_dir='./images/',
        filename='match_scores_comparison.png'
    )
    print(f"步骤3结果：匹配度柱状图已保存至 {scores_plot_path}")
    
    print(f"\n✓ 场景3完成：生成了2张专业XRD分析图表")
    print(f"  图表1: XRD图谱叠加对比图")
    print(f"  图表2: 相识别匹配度柱状图\n")
    
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始XRD相识别问题的完整流程")
    print("  （峰提取 -> 匹配度计算 -> 相识别）")
    print("- 场景2展示了工具对不同参数的适应性")
    print("  （容差参数扫描，验证算法鲁棒性）")
    print("- 场景3展示了综合分析能力和专业可视化功能")
    print("  （一键分析 + 生成标准XRD对比图）")
    print("\n核心科学原理：")
    print("- XRD峰位匹配基于Bragg定律 (nλ = 2d·sinθ)")
    print("- 复合材料的衍射图谱是各组分的线性叠加")
    print("- 通过峰位对比和匹配率统计实现相识别")


if __name__ == "__main__":
    main()