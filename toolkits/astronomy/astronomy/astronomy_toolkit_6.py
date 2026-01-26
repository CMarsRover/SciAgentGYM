# Filename: astronomy_toolkit.py
"""
天文学计算工具包

主要功能：
1. 恒星光度计算：基于Stefan-Boltzmann定律计算恒星辐射功率
2. 食双星系统分析：计算掩食事件中的光度变化
3. 凌星事件模拟：分析行星凌星对系统亮度的影响
4. 光变曲线生成：可视化多体系统的亮度变化

依赖库：
pip install numpy scipy astropy matplotlib astroquery
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import sigma_sb, R_sun
import os
from datetime import datetime

# 全局常量
STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # W m^-2 K^-4
SOLAR_RADIUS = 6.957e8  # meters
SOLAR_LUMINOSITY = 3.828e26  # watts

# ============ 第一层：原子工具函数（Atomic Tools） ============

def calculate_stellar_luminosity(radius: float, temperature: float, 
                                 radius_unit: str = 'solar') -> dict:
    """
    计算恒星光度（基于Stefan-Boltzmann定律）
    
    物理原理：L = 4πR²σT⁴，其中σ为Stefan-Boltzmann常数。
    光度与恒星半径的平方和有效温度的四次方成正比。
    
    Args:
        radius: 恒星半径（默认单位：太阳半径）
        temperature: 恒星有效温度/K，范围2000-50000K
        radius_unit: 半径单位，'solar'(太阳半径)或'meters'
        
    Returns:
        dict: {
            'result': 光度值（单位：太阳光度）,
            'metadata': {
                'luminosity_watts': 光度（瓦特）,
                'radius_meters': 半径（米）,
                'temperature_K': 温度（开尔文）
            }
        }
    
    Example:
        >>> result = calculate_stellar_luminosity(1.0, 6000)
        >>> print(result['result'])
        1.0
    """
    # === 完整边界检查 ===
    if not isinstance(radius, (int, float)):
        raise TypeError(f"radius必须是数值类型，当前类型：{type(radius)}")
    if not isinstance(temperature, (int, float)):
        raise TypeError(f"temperature必须是数值类型，当前类型：{type(temperature)}")
    
    if radius <= 0:
        raise ValueError(f"恒星半径必须为正数，当前值：{radius}")
    if not (2000 <= temperature <= 50000):
        raise ValueError(f"温度超出合理范围(2000-50000K)，当前值：{temperature}K")
    
    # 单位转换
    if radius_unit == 'solar':
        radius_m = radius * SOLAR_RADIUS
    elif radius_unit == 'meters':
        radius_m = radius
    else:
        raise ValueError(f"不支持的单位：{radius_unit}，仅支持'solar'或'meters'")
    
    # 核心计算：Stefan-Boltzmann定律
    luminosity_watts = 4 * np.pi * (radius_m ** 2) * STEFAN_BOLTZMANN_CONSTANT * (temperature ** 4)
    luminosity_solar = luminosity_watts / SOLAR_LUMINOSITY
    
    return {
        'result': luminosity_solar,
        'metadata': {
            'luminosity_watts': luminosity_watts,
            'radius_meters': radius_m,
            'temperature_K': temperature,
            'formula': 'L = 4πR²σT⁴'
        }
    }


def calculate_eclipse_depth(blocking_radius: float, blocked_radius: float,
                           blocking_luminosity: float, blocked_luminosity: float,
                           eclipse_type: str = 'partial') -> dict:
    """
    计算掩食事件的光度损失
    
    物理原理：当一个天体遮挡另一个天体时，系统总光度减少。
    减少量取决于被遮挡面积和该区域的光度贡献。
    
    Args:
        blocking_radius: 遮挡天体半径（太阳半径）
        blocked_radius: 被遮挡天体半径（太阳半径）
        blocking_luminosity: 遮挡天体光度（太阳光度）
        blocked_luminosity: 被遮挡天体光度（太阳光度）
        eclipse_type: 掩食类型，'partial'(部分食)或'total'(全食)
        
    Returns:
        dict: {
            'result': 剩余光度（太阳光度）,
            'metadata': {
                'blocked_fraction': 被遮挡的面积比例,
                'light_loss': 损失的光度,
                'eclipse_type': 掩食类型
            }
        }
    
    Example:
        >>> result = calculate_eclipse_depth(0.5, 1.0, 1.0, 20.0)
        >>> print(result['result'])
        16.0
    """
    # === 完整边界检查 ===
    if not all(isinstance(x, (int, float)) for x in [blocking_radius, blocked_radius, 
                                                       blocking_luminosity, blocked_luminosity]):
        raise TypeError("所有半径和光度参数必须是数值类型")
    
    if any(x <= 0 for x in [blocking_radius, blocked_radius, blocking_luminosity, blocked_luminosity]):
        raise ValueError("所有半径和光度必须为正数")
    
    if eclipse_type not in ['partial', 'total']:
        raise ValueError(f"不支持的掩食类型：{eclipse_type}")
    
    # 计算遮挡面积比例
    if blocking_radius >= blocked_radius:
        # 完全遮挡
        blocked_fraction = 1.0
        eclipse_type_actual = 'total'
    else:
        # 部分遮挡（假设中心对齐）
        blocked_fraction = (blocking_radius / blocked_radius) ** 2
        eclipse_type_actual = 'partial'
    
    # 计算光度损失
    light_loss = blocked_luminosity * blocked_fraction
    remaining_luminosity = blocking_luminosity + blocked_luminosity - light_loss
    
    return {
        'result': remaining_luminosity,
        'metadata': {
            'blocked_fraction': blocked_fraction,
            'light_loss': light_loss,
            'eclipse_type': eclipse_type_actual,
            'total_luminosity_before': blocking_luminosity + blocked_luminosity
        }
    }


def calculate_transit_depth(planet_radius: float, star_radius: float,
                           star_luminosity: float, companion_luminosity: float = 0.0) -> dict:
    """
    计算行星凌星事件的光度变化
    
    物理原理：行星凌星时遮挡恒星表面，导致系统亮度下降。
    下降幅度与行星和恒星的面积比成正比。
    
    Args:
        planet_radius: 行星半径（太阳半径）
        star_radius: 被凌恒星半径（太阳半径）
        star_luminosity: 被凌恒星光度（太阳光度）
        companion_luminosity: 伴星光度（太阳光度），默认0
        
    Returns:
        dict: {
            'result': 凌星期间的系统光度（太阳光度）,
            'metadata': {
                'transit_depth': 相对光度下降（百分比）,
                'blocked_area_fraction': 遮挡面积比例
            }
        }
    
    Example:
        >>> result = calculate_transit_depth(0.1, 1.0, 20.0, 1.0)
        >>> print(result['result'])
        20.8
    """
    # === 完整边界检查 ===
    if not all(isinstance(x, (int, float)) for x in [planet_radius, star_radius, 
                                                       star_luminosity, companion_luminosity]):
        raise TypeError("所有参数必须是数值类型")
    
    if planet_radius <= 0 or star_radius <= 0 or star_luminosity <= 0:
        raise ValueError("行星半径、恒星半径和光度必须为正数")
    
    if companion_luminosity < 0:
        raise ValueError("伴星光度不能为负数")
    
    if planet_radius > star_radius:
        raise ValueError(f"行星半径({planet_radius})不能大于恒星半径({star_radius})")
    
    # 计算遮挡面积比例
    blocked_area_fraction = (planet_radius / star_radius) ** 2
    
    # 计算光度损失
    light_loss = star_luminosity * blocked_area_fraction
    remaining_luminosity = star_luminosity - light_loss + companion_luminosity
    
    # 计算相对深度（百分比）
    total_luminosity = star_luminosity + companion_luminosity
    transit_depth_percent = (light_loss / total_luminosity) * 100 if total_luminosity > 0 else 0
    
    return {
        'result': remaining_luminosity,
        'metadata': {
            'transit_depth_percent': transit_depth_percent,
            'blocked_area_fraction': blocked_area_fraction,
            'light_loss': light_loss,
            'total_luminosity_before': total_luminosity
        }
    }


def find_brightness_extrema(luminosity_states: List[float]) -> dict:
    """
    找出系统亮度的最大值和最小值
    
    用于分析多种天文事件（掩食、凌星等）下的亮度变化范围。
    
    Args:
        luminosity_states: 不同状态下的光度列表（太阳光度）
        
    Returns:
        dict: {
            'result': {
                'max_luminosity': 最大光度,
                'min_luminosity': 最小光度,
                'brightness_ratio': 最大/最小亮度比
            },
            'metadata': {
                'num_states': 状态数量,
                'luminosity_range': 光度范围
            }
        }
    
    Example:
        >>> result = find_brightness_extrema([21.25, 20.0, 16.0])
        >>> print(result['result']['brightness_ratio'])
        1.328
    """
    # === 完整边界检查 ===
    if not isinstance(luminosity_states, list):
        raise TypeError(f"luminosity_states必须是列表，当前类型：{type(luminosity_states)}")
    
    if len(luminosity_states) == 0:
        raise ValueError("光度状态列表不能为空")
    
    if not all(isinstance(x, (int, float)) for x in luminosity_states):
        raise TypeError("光度列表中所有元素必须是数值类型")
    
    if any(x <= 0 for x in luminosity_states):
        raise ValueError("所有光度值必须为正数")
    
    # 核心计算
    max_luminosity = max(luminosity_states)
    min_luminosity = min(luminosity_states)
    brightness_ratio = max_luminosity / min_luminosity
    
    return {
        'result': {
            'max_luminosity': max_luminosity,
            'min_luminosity': min_luminosity,
            'brightness_ratio': brightness_ratio
        },
        'metadata': {
            'num_states': len(luminosity_states),
            'luminosity_range': max_luminosity - min_luminosity,
            'all_states': luminosity_states
        }
    }


# ============ 第二层：组合工具函数（Composite Tools） ============

def analyze_eclipsing_binary_system(star_a_radius: float, star_a_temp: float,
                                   star_b_radius: float, star_b_temp: float,
                                   planet_radius: Optional[float] = None) -> dict:
    """
    分析食双星系统的完整光变特性
    
    综合计算恒星光度、掩食深度和凌星效应，确定系统亮度变化范围。
    适用于研究Algol型、β Lyrae型等食双星系统。
    
    Args:
        star_a_radius: 主星半径（太阳半径）
        star_a_temp: 主星温度（开尔文）
        star_b_radius: 伴星半径（太阳半径）
        star_b_temp: 伴星温度（开尔文）
        planet_radius: 行星半径（太阳半径），可选
        
    Returns:
        dict: {
            'result': {
                'brightness_ratio': 最大/最小亮度比,
                'max_brightness_state': 最亮状态描述,
                'min_brightness_state': 最暗状态描述
            },
            'metadata': {
                'star_a_luminosity': 主星光度,
                'star_b_luminosity': 伴星光度,
                'all_scenarios': 所有计算场景的光度
            }
        }
    
    Example:
        >>> result = analyze_eclipsing_binary_system(1.0, 6000, 0.5, 4000, 0.1)
        >>> print(result['result']['brightness_ratio'])
        1.33
    """
    # === 参数完全可序列化检查 ===
    if not all(isinstance(x, (int, float)) for x in [star_a_radius, star_a_temp, 
                                                       star_b_radius, star_b_temp]):
        raise TypeError("所有恒星参数必须是数值类型")
    
    if planet_radius is not None and not isinstance(planet_radius, (int, float)):
        raise TypeError("planet_radius必须是数值类型或None")
    
    # === 步骤1：计算恒星光度 ===
    # 调用函数：calculate_stellar_luminosity()
    print("# 步骤1：计算主星光度")
    star_a_result = calculate_stellar_luminosity(star_a_radius, star_a_temp)
    l_a = star_a_result['result']
    print(f"FUNCTION_CALL: calculate_stellar_luminosity | PARAMS: radius={star_a_radius}, temp={star_a_temp} | RESULT: {l_a:.4f} L_sun")
    
    print("# 步骤2：计算伴星光度")
    star_b_result = calculate_stellar_luminosity(star_b_radius, star_b_temp)
    l_b = star_b_result['result']
    print(f"FUNCTION_CALL: calculate_stellar_luminosity | PARAMS: radius={star_b_radius}, temp={star_b_temp} | RESULT: {l_b:.4f} L_sun")
    
    # === 步骤3：计算所有可能的光度状态 ===
    luminosity_states = []
    scenario_descriptions = []
    
    # 状态1：无遮挡（最大亮度）
    max_luminosity = l_a + l_b
    luminosity_states.append(max_luminosity)
    scenario_descriptions.append("无遮挡状态")
    print(f"# 状态1：无遮挡 - 光度 = {max_luminosity:.4f} L_sun")
    
    # 状态2：伴星遮挡主星
    # 调用函数：calculate_eclipse_depth()
    print("# 步骤3：计算伴星遮挡主星的光度")
    eclipse_b_blocks_a = calculate_eclipse_depth(star_b_radius, star_a_radius, l_b, l_a)
    luminosity_states.append(eclipse_b_blocks_a['result'])
    scenario_descriptions.append(f"伴星遮挡主星（遮挡比例：{eclipse_b_blocks_a['metadata']['blocked_fraction']:.2%}）")
    print(f"FUNCTION_CALL: calculate_eclipse_depth | PARAMS: blocking_r={star_b_radius}, blocked_r={star_a_radius} | RESULT: {eclipse_b_blocks_a['result']:.4f} L_sun")
    
    # 状态3：主星遮挡伴星
    print("# 步骤4：计算主星遮挡伴星的光度")
    eclipse_a_blocks_b = calculate_eclipse_depth(star_a_radius, star_b_radius, l_a, l_b)
    luminosity_states.append(eclipse_a_blocks_b['result'])
    scenario_descriptions.append(f"主星遮挡伴星（遮挡比例：{eclipse_a_blocks_b['metadata']['blocked_fraction']:.2%}）")
    print(f"FUNCTION_CALL: calculate_eclipse_depth | PARAMS: blocking_r={star_a_radius}, blocked_r={star_b_radius} | RESULT: {eclipse_a_blocks_b['result']:.4f} L_sun")
    
    # 状态4-5：行星凌星（如果存在）
    if planet_radius is not None:
        print("# 步骤5：计算行星凌主星的光度")
        # 调用函数：calculate_transit_depth()
        transit_a = calculate_transit_depth(planet_radius, star_a_radius, l_a, l_b)
        luminosity_states.append(transit_a['result'])
        scenario_descriptions.append(f"行星凌主星（深度：{transit_a['metadata']['transit_depth_percent']:.3f}%）")
        print(f"FUNCTION_CALL: calculate_transit_depth | PARAMS: planet_r={planet_radius}, star_r={star_a_radius} | RESULT: {transit_a['result']:.4f} L_sun")
        
        print("# 步骤6：计算行星凌伴星的光度")
        transit_b = calculate_transit_depth(planet_radius, star_b_radius, l_b, l_a)
        luminosity_states.append(transit_b['result'])
        scenario_descriptions.append(f"行星凌伴星（深度：{transit_b['metadata']['transit_depth_percent']:.3f}%）")
        print(f"FUNCTION_CALL: calculate_transit_depth | PARAMS: planet_r={planet_radius}, star_r={star_b_radius} | RESULT: {transit_b['result']:.4f} L_sun")
    
    # === 步骤7：找出亮度极值 ===
    # 调用函数：find_brightness_extrema()，该函数内部调用了 max() 和 min()
    print("# 步骤7：分析亮度极值")
    extrema_result = find_brightness_extrema(luminosity_states)
    brightness_ratio = extrema_result['result']['brightness_ratio']
    print(f"FUNCTION_CALL: find_brightness_extrema | PARAMS: states={len(luminosity_states)} | RESULT: ratio={brightness_ratio:.4f}")
    
    # 找出最亮和最暗状态的描述
    max_idx = luminosity_states.index(extrema_result['result']['max_luminosity'])
    min_idx = luminosity_states.index(extrema_result['result']['min_luminosity'])
    
    return {
        'result': {
            'brightness_ratio': brightness_ratio,
            'max_brightness_state': scenario_descriptions[max_idx],
            'min_brightness_state': scenario_descriptions[min_idx],
            'max_luminosity': extrema_result['result']['max_luminosity'],
            'min_luminosity': extrema_result['result']['min_luminosity']
        },
        'metadata': {
            'star_a_luminosity': l_a,
            'star_b_luminosity': l_b,
            'luminosity_ratio': l_a / l_b,
            'all_scenarios': list(zip(scenario_descriptions, luminosity_states)),
            'num_scenarios': len(luminosity_states)
        }
    }


def scan_parameter_space(base_params: dict, scan_param: str, 
                        scan_range: List[float]) -> dict:
    """
    扫描参数空间以研究系统行为
    
    通过改变单一参数（如恒星温度、行星半径等），分析系统亮度变化的敏感性。
    用于优化观测策略或验证理论模型。
    
    Args:
        base_params: 基准参数字典，包含 'star_a_radius', 'star_a_temp', 'star_b_radius', 
                    'star_b_temp', 'planet_radius'
        scan_param: 要扫描的参数名称
        scan_range: 参数扫描范围（列表）
        
    Returns:
        dict: {
            'result': {
                'scan_values': 扫描的参数值,
                'brightness_ratios': 对应的亮度比
            },
            'metadata': {
                'optimal_value': 产生最大亮度变化的参数值,
                'sensitivity': 参数敏感性分析
            }
        }
    
    Example:
        >>> params = {'star_a_radius': 1.0, 'star_a_temp': 6000, 
        ...           'star_b_radius': 0.5, 'star_b_temp': 4000, 'planet_radius': 0.1}
        >>> result = scan_parameter_space(params, 'planet_radius', [0.05, 0.1, 0.15])
    """
    # === 参数检查 ===
    if not isinstance(base_params, dict):
        raise TypeError("base_params必须是字典")
    
    required_keys = ['star_a_radius', 'star_a_temp', 'star_b_radius', 'star_b_temp']
    if not all(key in base_params for key in required_keys):
        raise ValueError(f"base_params必须包含：{required_keys}")
    
    if not isinstance(scan_range, list):
        raise TypeError("scan_range必须是列表")
    
    if scan_param not in ['star_a_radius', 'star_a_temp', 'star_b_radius', 
                          'star_b_temp', 'planet_radius']:
        raise ValueError(f"不支持扫描参数：{scan_param}")
    
    # === 参数扫描 ===
    brightness_ratios = []
    
    for value in scan_range:
        # 更新参数
        current_params = base_params.copy()
        current_params[scan_param] = value
        
        # 调用函数：analyze_eclipsing_binary_system()
        result = analyze_eclipsing_binary_system(
            current_params['star_a_radius'],
            current_params['star_a_temp'],
            current_params['star_b_radius'],
            current_params['star_b_temp'],
            current_params.get('planet_radius')
        )
        
        brightness_ratios.append(result['result']['brightness_ratio'])
    
    # 找出最优值
    max_ratio_idx = np.argmax(brightness_ratios)
    optimal_value = scan_range[max_ratio_idx]
    
    # 计算敏感性（变化率）
    sensitivity = np.std(brightness_ratios) / np.mean(brightness_ratios) if len(brightness_ratios) > 1 else 0
    
    return {
        'result': {
            'scan_values': scan_range,
            'brightness_ratios': brightness_ratios
        },
        'metadata': {
            'optimal_value': optimal_value,
            'max_brightness_ratio': brightness_ratios[max_ratio_idx],
            'sensitivity': sensitivity,
            'scan_parameter': scan_param
        }
    }


def batch_analyze_systems_from_catalog(systems_data: List[dict]) -> dict:
    """
    批量分析多个食双星系统
    
    从观测目录或数据库中读取多个系统的参数，批量计算亮度变化特性。
    支持统计分析和系统间对比。
    
    Args:
        systems_data: 系统参数列表，每个元素为包含恒星参数的字典
        
    Returns:
        dict: {
            'result': {
                'system_names': 系统名称列表,
                'brightness_ratios': 亮度比列表,
                'statistics': 统计信息
            },
            'metadata': {
                'num_systems': 系统数量,
                'detailed_results': 每个系统的详细结果
            }
        }
    
    Example:
        >>> systems = [
        ...     {'name': 'System1', 'star_a_radius': 1.0, 'star_a_temp': 6000, 
        ...      'star_b_radius': 0.5, 'star_b_temp': 4000},
        ...     {'name': 'System2', 'star_a_radius': 1.2, 'star_a_temp': 5500, 
        ...      'star_b_radius': 0.8, 'star_b_temp': 4500}
        ... ]
        >>> result = batch_analyze_systems_from_catalog(systems)
    """
    # === 参数检查 ===
    if not isinstance(systems_data, list):
        raise TypeError("systems_data必须是列表")
    
    if len(systems_data) == 0:
        raise ValueError("系统数据列表不能为空")
    
    # === 批量分析 ===
    results = []
    system_names = []
    brightness_ratios = []
    
    for i, system in enumerate(systems_data):
        # 调用函数：analyze_eclipsing_binary_system()
        try:
            result = analyze_eclipsing_binary_system(
                system['star_a_radius'],
                system['star_a_temp'],
                system['star_b_radius'],
                system['star_b_temp'],
                system.get('planet_radius')
            )
            
            system_name = system.get('name', f'System_{i+1}')
            system_names.append(system_name)
            brightness_ratios.append(result['result']['brightness_ratio'])
            results.append({
                'name': system_name,
                'result': result
            })
            
        except Exception as e:
            print(f"警告：系统 {system.get('name', i)} 分析失败：{str(e)}")
            continue
    
    # 统计分析
    if len(brightness_ratios) > 0:
        statistics = {
            'mean': np.mean(brightness_ratios),
            'std': np.std(brightness_ratios),
            'min': np.min(brightness_ratios),
            'max': np.max(brightness_ratios),
            'median': np.median(brightness_ratios)
        }
    else:
        statistics = {}
    
    return {
        'result': {
            'system_names': system_names,
            'brightness_ratios': brightness_ratios,
            'statistics': statistics
        },
        'metadata': {
            'num_systems': len(results),
            'detailed_results': results
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def visualize_light_curve(time_array: List[float], luminosity_array: List[float],
                         events: Optional[List[dict]] = None,
                         title: str = "Light Curve",
                         save_dir: str = './tool_images/',
                         filename: str = None) -> dict:
    """
    可视化光变曲线
    
    绘制系统亮度随时间的变化，标注掩食和凌星事件。
    
    Args:
        time_array: 时间数组（天）
        luminosity_array: 光度数组（太阳光度）
        events: 事件标注列表，格式 [{'time': t, 'label': '事件名'}]
        title: 图表标题
        save_dir: 保存目录
        filename: 文件名（不含扩展名）
        
    Returns:
        dict: {
            'result': '图片保存路径',
            'metadata': {'plot_type': 'light_curve'}
        }
    """
    # === 参数检查 ===
    if not isinstance(time_array, list) or not isinstance(luminosity_array, list):
        raise TypeError("time_array和luminosity_array必须是列表")
    
    if len(time_array) != len(luminosity_array):
        raise ValueError("时间和光度数组长度必须相同")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"light_curve_{timestamp}"
    
    save_path = os.path.join(save_dir, f"{filename}.png")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.plot(time_array, luminosity_array, 'b-', linewidth=2, label='系统光度')
    
    # 标注事件
    if events:
        for event in events:
            plt.axvline(x=event['time'], color='r', linestyle='--', alpha=0.5)
            plt.text(event['time'], max(luminosity_array) * 0.95, 
                    event['label'], rotation=90, verticalalignment='top')
    
    plt.xlabel('时间 (天)', fontsize=12)
    plt.ylabel('光度 (太阳光度)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Light Curve Plot | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'plot_type': 'light_curve',
            'num_points': len(time_array),
            'time_range': [min(time_array), max(time_array)]
        }
    }


def visualize_parameter_sensitivity(scan_values: List[float], 
                                   brightness_ratios: List[float],
                                   param_name: str,
                                   save_dir: str = './tool_images/',
                                   filename: str = None) -> dict:
    """
    可视化参数敏感性分析
    
    绘制亮度变化比随参数变化的曲线，用于优化观测策略。
    
    Args:
        scan_values: 扫描的参数值列表
        brightness_ratios: 对应的亮度比列表
        param_name: 参数名称（用于标签）
        save_dir: 保存目录
        filename: 文件名
        
    Returns:
        dict: {
            'result': '图片保存路径',
            'metadata': {'plot_type': 'sensitivity_analysis'}
        }
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensitivity_{param_name}_{timestamp}"
    
    save_path = os.path.join(save_dir, f"{filename}.png")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    
    plt.plot(scan_values, brightness_ratios, 'o-', linewidth=2, markersize=8)
    plt.xlabel(f'{param_name}', fontsize=12)
    plt.ylabel('亮度变化比 (最大/最小)', fontsize=12)
    plt.title(f'{param_name} 敏感性分析', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Sensitivity Plot | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'plot_type': 'sensitivity_analysis',
            'parameter': param_name
        }
    }


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【当前问题】+【至少2个相关场景】
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - 食双星系统亮度变化分析")
    print("=" * 60)
    print("问题描述：计算一个包含凌星行星的食双星系统的最大亮度变化比")
    print("系统参数：主星(R=1.0 Rsun, T=6000K), 伴星(R=0.5 Rsun, T=4000K), 行星(R=0.1 Rsun)")
    print("-" * 60)
    
    # 调用函数：analyze_eclipsing_binary_system()
    # 该函数内部调用了 calculate_stellar_luminosity(), calculate_eclipse_depth(), 
    # calculate_transit_depth(), find_brightness_extrema()
    result1 = analyze_eclipsing_binary_system(
        star_a_radius=1.0,
        star_a_temp=6000,
        star_b_radius=0.5,
        star_b_temp=4000,
        planet_radius=0.1
    )
    
    print(f"\n✓ 场景1最终答案：")
    print(f"  最大亮度变化比 = {result1['result']['brightness_ratio']:.4f}")
    print(f"  最亮状态：{result1['result']['max_brightness_state']}")
    print(f"  最暗状态：{result1['result']['min_brightness_state']}")
    print(f"  主星光度：{result1['metadata']['star_a_luminosity']:.4f} L_sun")
    print(f"  伴星光度：{result1['metadata']['star_b_luminosity']:.4f} L_sun")
    print(f"FINAL_ANSWER: {result1['result']['brightness_ratio']:.2f}")
    
    print("\n" + "=" * 60)
    print("场景2：参数敏感性分析 - 行星半径对亮度变化的影响")
    print("=" * 60)
    print("问题描述：研究不同行星半径下系统亮度变化比的变化趋势")
    print("-" * 60)
    
    # 调用函数：scan_parameter_space()
    # 该函数内部多次调用 analyze_eclipsing_binary_system()
    base_params = {
        'star_a_radius': 1.0,
        'star_a_temp': 6000,
        'star_b_radius': 0.5,
        'star_b_temp': 4000,
        'planet_radius': 0.1
    }
    
    planet_radii = [0.05, 0.08, 0.1, 0.12, 0.15]
    result2 = scan_parameter_space(base_params, 'planet_radius', planet_radii)
    
    print(f"\n参数扫描结果：")
    for r, ratio in zip(result2['result']['scan_values'], result2['result']['brightness_ratios']):
        print(f"  行星半径 = {r:.2f} Rsun → 亮度比 = {ratio:.4f}")
    
    print(f"\n✓ 场景2完成：")
    print(f"  最优行星半径：{result2['metadata']['optimal_value']:.2f} Rsun")
    print(f"  对应最大亮度比：{result2['metadata']['max_brightness_ratio']:.4f}")
    print(f"  参数敏感性：{result2['metadata']['sensitivity']:.4f}")
    print(f"FUNCTION_CALL: scan_parameter_space | PARAMS: param=planet_radius, range={len(planet_radii)} | RESULT: sensitivity={result2['metadata']['sensitivity']:.4f}")
    
    # 可视化敏感性分析
    # 调用函数：visualize_parameter_sensitivity()
    vis_result2 = visualize_parameter_sensitivity(
        result2['result']['scan_values'],
        result2['result']['brightness_ratios'],
        '行星半径 (Rsun)'
    )
    
    print("\n" + "=" * 60)
    print("场景3：批量系统对比分析")
    print("=" * 60)
    print("问题描述：对比分析多个不同类型的食双星系统")
    print("-" * 60)
    
    # 调用函数：batch_analyze_systems_from_catalog()
    # 该函数内部对每个系统调用 analyze_eclipsing_binary_system()
    systems_catalog = [
        {
            'name': 'Algol型系统',
            'star_a_radius': 1.0,
            'star_a_temp': 6000,
            'star_b_radius': 0.5,
            'star_b_temp': 4000,
            'planet_radius': 0.1
        },
        {
            'name': '高温双星',
            'star_a_radius': 1.5,
            'star_a_temp': 8000,
            'star_b_radius': 1.2,
            'star_b_temp': 7000
        },
        {
            'name': '低温双星',
            'star_a_radius': 0.8,
            'star_a_temp': 4500,
            'star_b_radius': 0.6,
            'star_b_temp': 3500
        },
        {
            'name': '极端质量比系统',
            'star_a_radius': 2.0,
            'star_a_temp': 10000,
            'star_b_radius': 0.3,
            'star_b_temp': 3000
        }
    ]
    
    result3 = batch_analyze_systems_from_catalog(systems_catalog)
    
    print(f"\n批量分析结果：")
    for name, ratio in zip(result3['result']['system_names'], result3['result']['brightness_ratios']):
        print(f"  {name}: 亮度比 = {ratio:.4f}")
    
    print(f"\n统计信息：")
    stats = result3['result']['statistics']
    print(f"  平均亮度比：{stats['mean']:.4f}")
    print(f"  标准差：{stats['std']:.4f}")
    print(f"  范围：{stats['min']:.4f} - {stats['max']:.4f}")
    
    print(f"\n✓ 场景3完成：分析了 {result3['metadata']['num_systems']} 个系统")
    print(f"FUNCTION_CALL: batch_analyze_systems_from_catalog | PARAMS: num_systems={len(systems_catalog)} | RESULT: mean_ratio={stats['mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始问题的完整流程（亮度比 ≈ 1.33）")
    print("- 场景2展示了工具的参数敏感性分析能力")
    print("- 场景3展示了工具的批量处理和统计分析能力")
    print(f"\n所有生成的图片已保存到：./tool_images/")


if __name__ == "__main__":
    main()