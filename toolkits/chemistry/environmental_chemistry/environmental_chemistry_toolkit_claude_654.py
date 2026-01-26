# Filename: environmental_chemistry_toolkit.py

"""
Environmental Chemistry Toolkit for Water Quality Analysis
专业环境化学水质分析工具包

主要功能：
1. BOD5（五日生化需氧量）计算
2. 溶解氧（DO）测定与转换
3. 水质参数分析与可视化

依赖库：
- numpy: 数值计算
- matplotlib: 数据可视化
- scipy: 统计分析
- json: 数据序列化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import json
import os
from datetime import datetime

# 配置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 全局常量
OXYGEN_MOLAR_MASS = 32.0  # g/mol - O2摩尔质量
IODINE_TO_OXYGEN_RATIO = 0.25  # 碘量法中I2与O2的摩尔比 (1:4)
STANDARD_TEMPERATURE = 20.0  # °C - BOD5标准培养温度
INCUBATION_DAYS = 5  # BOD5培养天数

# 创建结果保存目录
os.makedirs('./mid_result/environmental_chemistry', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)


# ============================================================================
# 第一层：原子函数 - 基础计算单元
# ============================================================================

def calculate_dissolved_oxygen(na2s2o3_volume: float, 
                               na2s2o3_concentration: float,
                               sample_volume: float) -> Dict[str, Union[float, dict]]:
    """
    根据碘量法滴定结果计算溶解氧浓度
    
    原理：Winkler碘量法
    - 水样中的溶解氧氧化Mn(OH)2生成MnO(OH)2
    - 酸性条件下MnO(OH)2氧化I-生成I2
    - 用Na2S2O3标准溶液滴定I2
    - 反应关系：1 mol O2 ≡ 4 mol Na2S2O3
    
    Parameters:
    -----------
    na2s2o3_volume : float
        Na2S2O3标准溶液消耗体积 (mL)
    na2s2o3_concentration : float
        Na2S2O3标准溶液浓度 (mol/L)
    sample_volume : float
        水样体积 (mL)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        result: 溶解氧浓度 (mg/L)
        metadata: 计算过程的详细信息
    """
    # 参数验证
    if na2s2o3_volume < 0:
        raise ValueError(f"Na2S2O3体积不能为负: {na2s2o3_volume} mL")
    if na2s2o3_concentration <= 0:
        raise ValueError(f"Na2S2O3浓度必须为正: {na2s2o3_concentration} mol/L")
    if sample_volume <= 0:
        raise ValueError(f"样品体积必须为正: {sample_volume} mL")
    
    # 计算Na2S2O3的物质的量 (mol)
    n_na2s2o3 = na2s2o3_volume * na2s2o3_concentration / 1000.0
    
    # 根据化学计量关系计算O2的物质的量
    # 2Na2S2O3 + I2 → Na2S4O6 + 2NaI
    # O2 + 4Mn(OH)2 + 2H2O → 4Mn(OH)3
    # 2Mn(OH)3 + 6H+ + 2I- → 2Mn2+ + I2 + 6H2O
    # 总反应：1 O2 ≡ 4 Na2S2O3
    n_oxygen = n_na2s2o3 * IODINE_TO_OXYGEN_RATIO
    
    # 计算氧气质量 (mg)
    mass_oxygen = n_oxygen * OXYGEN_MOLAR_MASS * 1000.0
    
    # 计算溶解氧浓度 (mg/L)
    do_concentration = mass_oxygen / (sample_volume / 1000.0)
    
    metadata = {
        'na2s2o3_moles': n_na2s2o3,
        'oxygen_moles': n_oxygen,
        'oxygen_mass_mg': mass_oxygen,
        'calculation_method': 'Winkler iodometric method',
        'stoichiometry': '1 O2 ≡ 4 Na2S2O3'
    }
    
    return {
        'result': round(do_concentration, 4),
        'metadata': metadata
    }


def calculate_oxygen_depletion(do_initial: float, 
                               do_final: float,
                               blank_do_initial: float,
                               blank_do_final: float) -> Dict[str, Union[float, dict]]:
    """
    计算样品的净氧消耗量（扣除空白对照）
    
    Parameters:
    -----------
    do_initial : float
        样品初始溶解氧 (mg/L)
    do_final : float
        样品第5天溶解氧 (mg/L)
    blank_do_initial : float
        空白对照初始溶解氧 (mg/L)
    blank_do_final : float
        空白对照第5天溶解氧 (mg/L)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        result: 净氧消耗量 (mg/L)
    """
    # 参数验证
    if do_initial < 0 or do_final < 0:
        raise ValueError(f"溶解氧浓度不能为负: initial={do_initial}, final={do_final}")
    if blank_do_initial < 0 or blank_do_final < 0:
        raise ValueError(f"空白对照溶解氧不能为负: initial={blank_do_initial}, final={blank_do_final}")
    
    # 样品氧消耗
    sample_depletion = do_initial - do_final
    
    # 空白对照氧消耗（用于校正）
    blank_depletion = blank_do_initial - blank_do_final
    
    # 净氧消耗 = 样品消耗 - 空白消耗
    net_depletion = sample_depletion - blank_depletion
    
    metadata = {
        'sample_depletion': round(sample_depletion, 4),
        'blank_depletion': round(blank_depletion, 4),
        'correction_applied': True,
        'note': 'Blank correction removes background oxygen consumption'
    }
    
    return {
        'result': round(net_depletion, 4),
        'metadata': metadata
    }


def apply_dilution_factor(oxygen_depletion: float, 
                          dilution_factor: float) -> Dict[str, Union[float, dict]]:
    """
    应用稀释倍数校正，得到原水样的BOD5值
    
    Parameters:
    -----------
    oxygen_depletion : float
        稀释水样的净氧消耗量 (mg/L)
    dilution_factor : float
        稀释倍数（原水样体积/稀释后总体积）
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        result: 原水样BOD5值 (mg/L)
    """
    # 参数验证
    if oxygen_depletion < 0:
        raise ValueError(f"氧消耗量不能为负: {oxygen_depletion} mg/L")
    if dilution_factor <= 0:
        raise ValueError(f"稀释倍数必须为正: {dilution_factor}")
    
    # BOD5 = 稀释水样氧消耗 × 稀释倍数
    bod5 = oxygen_depletion * dilution_factor
    
    metadata = {
        'dilution_factor': dilution_factor,
        'diluted_sample_depletion': oxygen_depletion,
        'formula': 'BOD5 = ΔDO × Dilution_Factor',
        'unit': 'mg/L'
    }
    
    return {
        'result': round(bod5, 2),
        'metadata': metadata
    }


def validate_bod5_measurement(do_initial: float,
                              do_final: float,
                              min_do_threshold: float = 2.0,
                              min_depletion_threshold: float = 2.0) -> Dict[str, Union[bool, dict]]:
    """
    验证BOD5测定是否符合质量控制标准
    
    根据HJ 505-2009标准：
    - 培养后溶解氧应≥2 mg/L
    - 氧消耗量应≥2 mg/L
    - 氧消耗量不应超过初始DO的70%
    
    Parameters:
    -----------
    do_initial : float
        初始溶解氧 (mg/L)
    do_final : float
        第5天溶解氧 (mg/L)
    min_do_threshold : float
        最小剩余DO阈值 (mg/L)，默认2.0
    min_depletion_threshold : float
        最小氧消耗阈值 (mg/L)，默认2.0
    
    Returns:
    --------
    dict : {'result': bool, 'metadata': dict}
        result: 是否通过验证
        metadata: 详细的验证信息
    """
    depletion = do_initial - do_final
    depletion_ratio = depletion / do_initial if do_initial > 0 else 0
    
    checks = {
        'final_do_sufficient': do_final >= min_do_threshold,
        'depletion_sufficient': depletion >= min_depletion_threshold,
        'depletion_not_excessive': depletion_ratio <= 0.7
    }
    
    all_passed = all(checks.values())
    
    metadata = {
        'initial_do': do_initial,
        'final_do': do_final,
        'oxygen_depletion': round(depletion, 2),
        'depletion_ratio': round(depletion_ratio, 3),
        'checks': checks,
        'standard': 'HJ 505-2009',
        'recommendations': []
    }
    
    if not checks['final_do_sufficient']:
        metadata['recommendations'].append(f"Final DO ({do_final:.2f} mg/L) < {min_do_threshold} mg/L. Increase dilution factor.")
    if not checks['depletion_sufficient']:
        metadata['recommendations'].append(f"Depletion ({depletion:.2f} mg/L) < {min_depletion_threshold} mg/L. Decrease dilution factor.")
    if not checks['depletion_not_excessive']:
        metadata['recommendations'].append(f"Depletion ratio ({depletion_ratio:.1%}) > 70%. Increase dilution factor.")
    
    return {
        'result': all_passed,
        'metadata': metadata
    }


# ============================================================================
# 第二层：组合函数 - 完整分析流程
# ============================================================================

def calculate_bod5_from_titration(sample_data: Dict[str, Union[float, str]],
                                  blank_data: Dict[str, Union[float, str]]) -> Dict[str, Union[float, dict]]:
    """
    从滴定数据计算BOD5值（完整流程）
    
    Parameters:
    -----------
    sample_data : dict
        样品数据，包含：
        - 'dilution_factor': 稀释倍数
        - 'sample_volume': 样品体积 (mL)
        - 'na2s2o3_conc': Na2S2O3浓度 (mol/L)
        - 'na2s2o3_vol_day0': 第0天Na2S2O3用量 (mL)
        - 'na2s2o3_vol_day5': 第5天Na2S2O3用量 (mL)
    
    blank_data : dict
        空白对照数据，结构同sample_data（dilution_factor为0）
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        result: BOD5值 (mg/L)
        metadata: 完整的计算过程
    """
    # 步骤1：计算样品的初始和最终DO
    sample_do_day0 = calculate_dissolved_oxygen(
        sample_data['na2s2o3_vol_day0'],
        sample_data['na2s2o3_conc'],
        sample_data['sample_volume']
    )
    
    sample_do_day5 = calculate_dissolved_oxygen(
        sample_data['na2s2o3_vol_day5'],
        sample_data['na2s2o3_conc'],
        sample_data['sample_volume']
    )
    
    # 步骤2：计算空白对照的初始和最终DO
    blank_do_day0 = calculate_dissolved_oxygen(
        blank_data['na2s2o3_vol_day0'],
        blank_data['na2s2o3_conc'],
        blank_data['sample_volume']
    )
    
    blank_do_day5 = calculate_dissolved_oxygen(
        blank_data['na2s2o3_vol_day5'],
        blank_data['na2s2o3_conc'],
        blank_data['sample_volume']
    )
    
    # 步骤3：计算净氧消耗
    net_depletion = calculate_oxygen_depletion(
        sample_do_day0['result'],
        sample_do_day5['result'],
        blank_do_day0['result'],
        blank_do_day5['result']
    )
    
    # 步骤4：应用稀释倍数
    bod5_result = apply_dilution_factor(
        net_depletion['result'],
        sample_data['dilution_factor']
    )
    
    # 步骤5：质量控制验证
    validation = validate_bod5_measurement(
        sample_do_day0['result'],
        sample_do_day5['result']
    )
    
    # 汇总元数据
    metadata = {
        'sample_do_day0': sample_do_day0['result'],
        'sample_do_day5': sample_do_day5['result'],
        'blank_do_day0': blank_do_day0['result'],
        'blank_do_day5': blank_do_day5['result'],
        'net_oxygen_depletion': net_depletion['result'],
        'dilution_factor': sample_data['dilution_factor'],
        'quality_control': validation['metadata'],
        'calculation_steps': [
            '1. Calculate DO from Na2S2O3 titration (Winkler method)',
            '2. Calculate oxygen depletion with blank correction',
            '3. Apply dilution factor to get original sample BOD5',
            '4. Validate measurement against HJ 505-2009 standards'
        ]
    }
    
    # 保存中间结果
    result_file = './mid_result/environmental_chemistry/bod5_calculation.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'bod5_value': bod5_result['result'],
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    return {
        'result': bod5_result['result'],
        'metadata': metadata
    }


def batch_calculate_bod5(samples: List[Dict], 
                         blank: Dict) -> Dict[str, Union[list, dict]]:
    """
    批量计算多个样品的BOD5值
    
    Parameters:
    -----------
    samples : list of dict
        多个样品数据列表
    blank : dict
        空白对照数据
    
    Returns:
    --------
    dict : {'result': list, 'metadata': dict}
        result: BOD5值列表
        metadata: 统计信息
    """
    results = []
    
    for i, sample in enumerate(samples):
        try:
            bod5 = calculate_bod5_from_titration(sample, blank)
            results.append({
                'sample_id': sample.get('id', f'Sample_{i+1}'),
                'bod5': bod5['result'],
                'valid': bod5['metadata']['quality_control']['checks']
            })
        except Exception as e:
            results.append({
                'sample_id': sample.get('id', f'Sample_{i+1}'),
                'bod5': None,
                'error': str(e)
            })
    
    # 统计分析
    valid_bod5 = [r['bod5'] for r in results if r['bod5'] is not None]
    
    metadata = {
        'total_samples': len(samples),
        'successful_calculations': len(valid_bod5),
        'mean_bod5': round(np.mean(valid_bod5), 2) if valid_bod5 else None,
        'std_bod5': round(np.std(valid_bod5), 2) if valid_bod5 else None,
        'min_bod5': round(min(valid_bod5), 2) if valid_bod5 else None,
        'max_bod5': round(max(valid_bod5), 2) if valid_bod5 else None
    }
    
    return {
        'result': results,
        'metadata': metadata
    }


def estimate_optimal_dilution(estimated_cod: float,
                              target_do_depletion_range: Tuple[float, float] = (2.0, 6.0)) -> Dict[str, Union[float, dict]]:
    """
    根据COD估算最佳稀释倍数
    
    经验关系：BOD5 ≈ 0.5-0.7 × COD（对于生活污水）
    目标：使氧消耗量在2-6 mg/L之间
    
    Parameters:
    -----------
    estimated_cod : float
        估算的COD值 (mg/L)
    target_do_depletion_range : tuple
        目标氧消耗范围 (mg/L)
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
        result: 推荐的稀释倍数
    """
    if estimated_cod <= 0:
        raise ValueError(f"COD值必须为正: {estimated_cod} mg/L")
    
    # 估算BOD5（取中间值0.6）
    estimated_bod5 = estimated_cod * 0.6
    
    # 计算稀释倍数使氧消耗在目标范围中点
    target_depletion = np.mean(target_do_depletion_range)
    recommended_dilution = estimated_bod5 / target_depletion
    
    # 圆整到标准稀释倍数
    standard_dilutions = [2, 5, 10, 20, 30, 40, 50, 100, 200, 300]
    recommended_dilution = min(standard_dilutions, 
                              key=lambda x: abs(x - recommended_dilution))
    
    metadata = {
        'estimated_cod': estimated_cod,
        'estimated_bod5': round(estimated_bod5, 2),
        'bod5_cod_ratio': 0.6,
        'target_depletion_range': target_do_depletion_range,
        'calculated_dilution': round(estimated_bod5 / target_depletion, 1),
        'standard_dilutions': standard_dilutions,
        'note': 'Rounded to nearest standard dilution factor'
    }
    
    return {
        'result': recommended_dilution,
        'metadata': metadata
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def plot_do_time_series(sample_do_day0: float,
                        sample_do_day5: float,
                        blank_do_day0: float,
                        blank_do_day5: float,
                        sample_id: str = "Sample A") -> Dict[str, Union[str, dict]]:
    """
    绘制溶解氧随时间变化曲线
    
    Parameters:
    -----------
    sample_do_day0 : float
        样品第0天DO (mg/L)
    sample_do_day5 : float
        样品第5天DO (mg/L)
    blank_do_day0 : float
        空白第0天DO (mg/L)
    blank_do_day5 : float
        空白第5天DO (mg/L)
    sample_id : str
        样品标识
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        result: 图像文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = [0, 5]
    sample_do = [sample_do_day0, sample_do_day5]
    blank_do = [blank_do_day0, blank_do_day5]
    
    # 绘制曲线
    ax.plot(days, sample_do, 'o-', linewidth=2, markersize=8, 
            label=f'{sample_id}', color='#2E86AB')
    ax.plot(days, blank_do, 's--', linewidth=2, markersize=8,
            label='Blank Control', color='#A23B72')
    
    # 标注数值
    for i, (d, s, b) in enumerate(zip(days, sample_do, blank_do)):
        ax.text(d, s, f'{s:.2f}', ha='center', va='bottom', fontsize=10)
        ax.text(d, b, f'{b:.2f}', ha='center', va='top', fontsize=10)
    
    # 设置坐标轴
    ax.set_xlabel('Incubation Time (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dissolved Oxygen (mg/L)', fontsize=12, fontweight='bold')
    ax.set_title('BOD5 Measurement: DO Time Series', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # 添加参考线
    ax.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, 
               label='Min DO Threshold (2 mg/L)')
    
    plt.tight_layout()
    
    # 保存图像
    filepath = f'./tool_images/bod5_do_timeseries_{sample_id.replace(" ", "_")}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    metadata = {
        'sample_id': sample_id,
        'sample_do_depletion': round(sample_do_day0 - sample_do_day5, 2),
        'blank_do_depletion': round(blank_do_day0 - blank_do_day5, 2),
        'figure_size': '10x6 inches',
        'dpi': 300
    }
    
    return {
        'result': filepath,
        'metadata': metadata
    }


def plot_bod5_comparison(bod5_values: List[float],
                         sample_ids: List[str],
                         water_quality_standards: Dict[str, float] = None) -> Dict[str, Union[str, dict]]:
    """
    绘制多个样品的BOD5对比柱状图
    
    Parameters:
    -----------
    bod5_values : list of float
        BOD5值列表 (mg/L)
    sample_ids : list of str
        样品标识列表
    water_quality_standards : dict, optional
        水质标准参考值，如 {'Class I': 3, 'Class II': 6, 'Class III': 10}
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        result: 图像文件路径
    """
    if len(bod5_values) != len(sample_ids):
        raise ValueError("BOD5值数量与样品ID数量不匹配")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(sample_ids))
    bars = ax.bar(x, bod5_values, width=0.6, color='#06A77D', 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, bod5_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加水质标准参考线
    if water_quality_standards:
        colors = ['green', 'yellow', 'orange', 'red']
        for i, (grade, value) in enumerate(water_quality_standards.items()):
            ax.axhline(y=value, color=colors[i % len(colors)], 
                      linestyle='--', linewidth=2, alpha=0.7,
                      label=f'{grade}: {value} mg/L')
    
    ax.set_xlabel('Sample ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('BOD5 (mg/L)', fontsize=13, fontweight='bold')
    ax.set_title('BOD5 Comparison Across Samples', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if water_quality_standards:
        ax.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    filepath = './tool_images/bod5_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    metadata = {
        'num_samples': len(sample_ids),
        'mean_bod5': round(np.mean(bod5_values), 2),
        'max_bod5': round(max(bod5_values), 2),
        'min_bod5': round(min(bod5_values), 2),
        'standards_included': water_quality_standards is not None
    }
    
    return {
        'result': filepath,
        'metadata': metadata
    }


def plot_dilution_optimization(cod_range: Tuple[float, float],
                               num_points: int = 50) -> Dict[str, Union[str, dict]]:
    """
    绘制稀释倍数优化曲线
    
    展示不同COD值对应的推荐稀释倍数
    
    Parameters:
    -----------
    cod_range : tuple
        COD范围 (mg/L)
    num_points : int
        曲线采样点数
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
        result: 图像文件路径
    """
    cod_values = np.linspace(cod_range[0], cod_range[1], num_points)
    dilution_factors = []
    
    for cod in cod_values:
        result = estimate_optimal_dilution(cod)
        dilution_factors.append(result['result'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：稀释倍数 vs COD
    ax1.plot(cod_values, dilution_factors, linewidth=2.5, color='#D62828')
    ax1.set_xlabel('Estimated COD (mg/L)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recommended Dilution Factor', fontsize=12, fontweight='bold')
    ax1.set_title('Optimal Dilution Factor vs COD', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图：预期氧消耗 vs COD
    expected_depletion = cod_values * 0.6 / np.array(dilution_factors)
    ax2.plot(cod_values, expected_depletion, linewidth=2.5, color='#F77F00')
    ax2.axhspan(2, 6, alpha=0.2, color='green', label='Target Range (2-6 mg/L)')
    ax2.set_xlabel('Estimated COD (mg/L)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Expected DO Depletion (mg/L)', fontsize=12, fontweight='bold')
    ax2.set_title('Expected Oxygen Depletion', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    filepath = './tool_images/dilution_optimization.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    metadata = {
        'cod_range': cod_range,
        'num_points': num_points,
        'target_depletion_range': (2.0, 6.0),
        'bod5_cod_ratio': 0.6
    }
    
    return {
        'result': filepath,
        'metadata': metadata
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示环境化学工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 计算给定水样的BOD5值")
    print("=" * 80)
    print("问题描述：根据表格中的滴定数据，计算水样A的BOD5值")
    print("数据来源：样品A（稀释倍数40）和空白对照的Na2S2O3滴定结果")
    print("-" * 80)
    
    # 步骤1：准备样品数据
    sample_a_data = {
        'id': 'Sample A',
        'dilution_factor': 40.0,
        'sample_volume': 100.0,  # mL
        'na2s2o3_conc': 0.0125,  # mol/L
        'na2s2o3_vol_day0': 9.50,  # mL
        'na2s2o3_vol_day5': 5.45   # mL
    }
    
    blank_data = {
        'id': 'Blank',
        'dilution_factor': 0.0,
        'sample_volume': 100.0,
        'na2s2o3_conc': 0.0125,
        'na2s2o3_vol_day0': 9.45,
        'na2s2o3_vol_day5': 9.20
    }
    
    print("\n步骤1：计算样品A的初始溶解氧（第0天）")
    print("调用函数：calculate_dissolved_oxygen()")
    sample_do_day0 = calculate_dissolved_oxygen(
        sample_a_data['na2s2o3_vol_day0'],
        sample_a_data['na2s2o3_conc'],
        sample_a_data['sample_volume']
    )
    print(f"FUNCTION_CALL: calculate_dissolved_oxygen | PARAMS: {{na2s2o3_volume: {sample_a_data['na2s2o3_vol_day0']}, na2s2o3_conc: {sample_a_data['na2s2o3_conc']}, sample_volume: {sample_a_data['sample_volume']}}} | RESULT: {sample_do_day0}")
    
    print("\n步骤2：计算样品A的最终溶解氧（第5天）")
    print("调用函数：calculate_dissolved_oxygen()")
    sample_do_day5 = calculate_dissolved_oxygen(
        sample_a_data['na2s2o3_vol_day5'],
        sample_a_data['na2s2o3_conc'],
        sample_a_data['sample_volume']
    )
    print(f"FUNCTION_CALL: calculate_dissolved_oxygen | PARAMS: {{na2s2o3_volume: {sample_a_data['na2s2o3_vol_day5']}, na2s2o3_conc: {sample_a_data['na2s2o3_conc']}, sample_volume: {sample_a_data['sample_volume']}}} | RESULT: {sample_do_day5}")
    
    print("\n步骤3：计算空白对照的初始和最终溶解氧")
    print("调用函数：calculate_dissolved_oxygen()")
    blank_do_day0 = calculate_dissolved_oxygen(
        blank_data['na2s2o3_vol_day0'],
        blank_data['na2s2o3_conc'],
        blank_data['sample_volume']
    )
    blank_do_day5 = calculate_dissolved_oxygen(
        blank_data['na2s2o3_vol_day5'],
        blank_data['na2s2o3_conc'],
        blank_data['sample_volume']
    )
    print(f"FUNCTION_CALL: calculate_dissolved_oxygen (Blank Day0) | PARAMS: {{na2s2o3_volume: {blank_data['na2s2o3_vol_day0']}, na2s2o3_conc: {blank_data['na2s2o3_conc']}, sample_volume: {blank_data['sample_volume']}}} | RESULT: {blank_do_day0}")
    print(f"FUNCTION_CALL: calculate_dissolved_oxygen (Blank Day5) | PARAMS: {{na2s2o3_volume: {blank_data['na2s2o3_vol_day5']}, na2s2o3_conc: {blank_data['na2s2o3_conc']}, sample_volume: {blank_data['sample_volume']}}} | RESULT: {blank_do_day5}")
    
    print("\n步骤4：计算净氧消耗量（扣除空白对照）")
    print("调用函数：calculate_oxygen_depletion()")
    net_depletion = calculate_oxygen_depletion(
        sample_do_day0['result'],
        sample_do_day5['result'],
        blank_do_day0['result'],
        blank_do_day5['result']
    )
    print(f"FUNCTION_CALL: calculate_oxygen_depletion | PARAMS: {{sample_do_day0: {sample_do_day0['result']}, sample_do_day5: {sample_do_day5['result']}, blank_do_day0: {blank_do_day0['result']}, blank_do_day5: {blank_do_day5['result']}}} | RESULT: {net_depletion}")
    
    print("\n步骤5：应用稀释倍数得到原水样BOD5")
    print("调用函数：apply_dilution_factor()")
    bod5_result = apply_dilution_factor(
        net_depletion['result'],
        sample_a_data['dilution_factor']
    )
    print(f"FUNCTION_CALL: apply_dilution_factor | PARAMS: {{oxygen_depletion: {net_depletion['result']}, dilution_factor: {sample_a_data['dilution_factor']}}} | RESULT: {bod5_result}")
    
    print("\n步骤6：质量控制验证")
    print("调用函数：validate_bod5_measurement()")
    validation = validate_bod5_measurement(
        sample_do_day0['result'],
        sample_do_day5['result']
    )
    print(f"FUNCTION_CALL: validate_bod5_measurement | PARAMS: {{do_initial: {sample_do_day0['result']}, do_final: {sample_do_day5['result']}}} | RESULT: {validation}")
    
    print("\n步骤7：绘制溶解氧时间序列图")
    print("调用函数：plot_do_time_series()")
    plot_result = plot_do_time_series(
        sample_do_day0['result'],
        sample_do_day5['result'],
        blank_do_day0['result'],
        blank_do_day5['result'],
        sample_a_data['id']
    )
    print(f"FUNCTION_CALL: plot_do_time_series | PARAMS: {{sample_do_day0: {sample_do_day0['result']}, sample_do_day5: {sample_do_day5['result']}, blank_do_day0: {blank_do_day0['result']}, blank_do_day5: {blank_do_day5['result']}, sample_id: '{sample_a_data['id']}'}} | RESULT: {plot_result}")
    
    print("\n" + "=" * 80)
    print(f"场景1最终答案：样品A的BOD5值 = {bod5_result['result']} mg/L")
    print(f"质量控制：{'通过' if validation['result'] else '未通过'}")
    print(f"FINAL_ANSWER: {bod5_result['result']}")
    print("=" * 80)
    
    
    print("\n\n" + "=" * 80)
    print("场景2：批量样品分析 - 多个水样的BOD5测定与对比")
    print("=" * 80)
    print("问题描述：同时测定3个不同污染程度水样的BOD5，并进行对比分析")
    print("应用场景：污水处理厂进水、出水和河流水质监测")
    print("-" * 80)
    
    # 步骤1：准备多个样品数据
    samples = [
        {
            'id': 'Influent',  # 进水
            'dilution_factor': 100.0,
            'sample_volume': 100.0,
            'na2s2o3_conc': 0.0125,
            'na2s2o3_vol_day0': 9.60,
            'na2s2o3_vol_day5': 4.80
        },
        {
            'id': 'Effluent',  # 出水
            'dilution_factor': 5.0,
            'sample_volume': 100.0,
            'na2s2o3_conc': 0.0125,
            'na2s2o3_vol_day0': 9.50,
            'na2s2o3_vol_day5': 8.70
        },
        {
            'id': 'River Water',  # 河水
            'dilution_factor': 2.0,
            'sample_volume': 100.0,
            'na2s2o3_conc': 0.0125,
            'na2s2o3_vol_day0': 9.45,
            'na2s2o3_vol_day5': 8.95
        }
    ]
    
    print("\n步骤1：批量计算所有样品的BOD5值")
    print("调用函数：batch_calculate_bod5()")
    batch_results = batch_calculate_bod5(samples, blank_data)
    print(f"FUNCTION_CALL: batch_calculate_bod5 | PARAMS: {{num_samples: {len(samples)}, blank_id: '{blank_data['id']}'}} | RESULT: {batch_results}")
    
    print("\n步骤2：提取BOD5值用于可视化")
    bod5_values = [r['bod5'] for r in batch_results['result']]
    sample_ids = [r['sample_id'] for r in batch_results['result']]
    print(f"提取的BOD5值: {bod5_values}")
    print(f"样品标识: {sample_ids}")
    
    print("\n步骤3：绘制BOD5对比图（含水质标准参考线）")
    print("调用函数：plot_bod5_comparison()")
    water_standards = {
        'Class I': 3.0,
        'Class II': 6.0,
        'Class III': 10.0,
        'Class IV': 20.0
    }
    comparison_plot = plot_bod5_comparison(bod5_values, sample_ids, water_standards)
    print(f"FUNCTION_CALL: plot_bod5_comparison | PARAMS: {{bod5_values: {bod5_values}, sample_ids: {sample_ids}, standards: {water_standards}}} | RESULT: {comparison_plot}")
    
    print("\n步骤4：统计分析")
    print(f"样品数量: {batch_results['metadata']['total_samples']}")
    print(f"平均BOD5: {batch_results['metadata']['mean_bod5']} mg/L")
    print(f"BOD5范围: {batch_results['metadata']['min_bod5']} - {batch_results['metadata']['max_bod5']} mg/L")
    
    print("\n" + "=" * 80)
    print(f"场景2最终答案：成功分析{batch_results['metadata']['successful_calculations']}个样品")
    print(f"平均BOD5 = {batch_results['metadata']['mean_bod5']} mg/L")
    print(f"FINAL_ANSWER: {batch_results['metadata']['mean_bod5']}")
    print("=" * 80)
    
    
    print("\n\n" + "=" * 80)
    print("场景3：稀释倍数优化 - 基于COD估算最佳稀释方案")
    print("=" * 80)
    print("问题描述：对于未知BOD5的水样，根据COD值预测最佳稀释倍数")
    print("应用场景：实验前的方案设计，避免重复测定")
    print("-" * 80)
    
    # 步骤1：模拟不同COD水平的水样
    test_samples = [
        {'name': 'Low Pollution', 'cod': 50.0},
        {'name': 'Medium Pollution', 'cod': 200.0},
        {'name': 'High Pollution', 'cod': 500.0},
        {'name': 'Very High Pollution', 'cod': 1000.0}
    ]
    
    print("\n步骤1：为不同污染程度的水样估算最佳稀释倍数")
    print("调用函数：estimate_optimal_dilution()")
    
    optimization_results = []
    for sample in test_samples:
        result = estimate_optimal_dilution(sample['cod'])
        optimization_results.append({
            'name': sample['name'],
            'cod': sample['cod'],
            'recommended_dilution': result['result'],
            'estimated_bod5': result['metadata']['estimated_bod5']
        })
        print(f"FUNCTION_CALL: estimate_optimal_dilution | PARAMS: {{estimated_cod: {sample['cod']}}} | RESULT: {result}")
    
    print("\n步骤2：生成稀释优化曲线")
    print("调用函数：plot_dilution_optimization()")
    optimization_plot = plot_dilution_optimization(
        cod_range=(20.0, 1000.0),
        num_points=100
    )
    print(f"FUNCTION_CALL: plot_dilution_optimization | PARAMS: {{cod_range: (20.0, 1000.0), num_points: 100}} | RESULT: {optimization_plot}")
    
    print("\n步骤3：输出优化建议表")
    print("\n" + "-" * 80)
    print(f"{'水样类型':<20} {'COD (mg/L)':<15} {'推荐稀释倍数':<15} {'预期BOD5 (mg/L)':<20}")
    print("-" * 80)
    for res in optimization_results:
        print(f"{res['name']:<20} {res['cod']:<15.1f} {res['recommended_dilution']:<15.0f} {res['estimated_bod5']:<20.2f}")
    print("-" * 80)
    
    print("\n步骤4：验证优化方案 - 使用推荐稀释倍数计算场景1样品")
    print("假设场景1样品的COD为254.2 mg/L（根据BOD5反推）")
    estimated_cod_sample_a = bod5_result['result'] / 0.6  # BOD5/COD ≈ 0.6
    optimized_dilution = estimate_optimal_dilution(estimated_cod_sample_a)
    print(f"FUNCTION_CALL: estimate_optimal_dilution | PARAMS: {{estimated_cod: {estimated_cod_sample_a:.2f}}} | RESULT: {optimized_dilution}")
    print(f"推荐稀释倍数: {optimized_dilution['result']}")
    print(f"实际使用稀释倍数: {sample_a_data['dilution_factor']}")
    print(f"倍数匹配度: {'良好' if abs(optimized_dilution['result'] - sample_a_data['dilution_factor']) <= 10 else '需调整'}")
    
    print("\n" + "=" * 80)
    print(f"场景3最终答案：成功为{len(test_samples)}种污染水平提供稀释方案")
    print(f"COD范围: 50-1000 mg/L，推荐稀释倍数: 5-200倍")
    print(f"FINAL_ANSWER: {len(test_samples)}")
    print("=" * 80)
    
    print("\n\n" + "=" * 80)
    print("工具包演示完成！")
    print("=" * 80)
    print("生成的文件：")
    print("1. 中间结果: ./mid_result/environmental_chemistry/bod5_calculation.json")
    print("2. 图像文件: ./tool_images/bod5_do_timeseries_Sample_A.png")
    print("3. 图像文件: ./tool_images/bod5_comparison.png")
    print("4. 图像文件: ./tool_images/dilution_optimization.png")
    print("=" * 80)


if __name__ == "__main__":
    main()