# Filename: astrophysics_toolkit.py
"""
天体物理计算工具包

主要功能：
1. 双星系统分析：基于开普勒定律和径向速度数据计算轨道参数和质量
2. 恒星参数查询：从SIMBAD等免费数据库获取恒星物理参数
3. 轨道动力学：计算双星系统的轨道演化和稳定性

依赖库：
pip install numpy scipy astropy astroquery matplotlib plotly
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import os
from datetime import datetime
import json

# 导入天体物理专属库
from astropy import units as u
from astropy import constants as const
from astropy.time import Time
from astroquery.simbad import Simbad
from scipy.optimize import fsolve, minimize
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 全局常量
G_CONST = const.G.to(u.km**3 / (u.kg * u.s**2)).value  # 引力常数
SOLAR_MASS = const.M_sun.to(u.kg).value  # 太阳质量 (kg)
AU_TO_KM = const.au.to(u.km).value  # 天文单位转千米
YEAR_TO_SEC = (1 * u.year).to(u.s).value  # 年转秒

# 创建中间结果保存目录
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============ 第一层：原子工具函数（Atomic Tools） ============

def calculate_mass_ratio_from_rv(k1: float, k2: float) -> dict:
    """
    根据径向速度振幅计算双星质量比
    
    物理原理：由于双星系统质心守恒，m1*v1 = m2*v2，因此质量比等于速度比的倒数。
    
    Args:
        k1: 第一颗星的径向速度振幅 (km/s)
        k2: 第二颗星的径向速度振幅 (km/s)
    
    Returns:
        dict: {
            'result': 质量比 m1/m2,
            'metadata': {
                'k1': 输入的k1值,
                'k2': 输入的k2值,
                'principle': '动量守恒定律'
            }
        }
    
    Example:
        >>> result = calculate_mass_ratio_from_rv(10.0, 5.0)
        >>> print(result['result'])
        0.5
    """
    # === 完整边界检查 ===
    if not isinstance(k1, (int, float)):
        raise TypeError(f"k1必须是数值类型，当前类型: {type(k1)}")
    if not isinstance(k2, (int, float)):
        raise TypeError(f"k2必须是数值类型，当前类型: {type(k2)}")
    
    if k1 <= 0:
        raise ValueError(f"k1必须为正值，当前值: {k1}")
    if k2 <= 0:
        raise ValueError(f"k2必须为正值，当前值: {k2}")
    
    # 计算质量比
    mass_ratio = k2 / k1
    
    return {
        'result': mass_ratio,
        'metadata': {
            'k1_km_s': k1,
            'k2_km_s': k2,
            'principle': 'Conservation of momentum: m1*v1 = m2*v2',
            'formula': 'm1/m2 = k2/k1'
        }
    }


def calculate_orbital_separation(period_years: float, k1: float, k2: float) -> dict:
    """
    根据轨道周期和径向速度计算双星轨道半长轴
    
    物理原理：对于圆轨道，a = (P/2π) * (K1 + K2)，其中P是周期，K是径向速度振幅。
    
    Args:
        period_years: 轨道周期 (年)
        k1: 第一颗星的径向速度振幅 (km/s)
        k2: 第二颗星的径向速度振幅 (km/s)
    
    Returns:
        dict: {
            'result': 轨道半长轴 (km),
            'metadata': {
                'period_years': 周期,
                'separation_au': 半长轴(天文单位),
                'formula': 'a = (P/2π)(K1+K2)'
            }
        }
    
    Example:
        >>> result = calculate_orbital_separation(2.0, 10.0, 5.0)
        >>> print(f"{result['result']:.2e} km")
    """
    # === 完整边界检查 ===
    if not isinstance(period_years, (int, float)):
        raise TypeError(f"period_years必须是数值类型，当前类型: {type(period_years)}")
    if not isinstance(k1, (int, float)):
        raise TypeError(f"k1必须是数值类型，当前类型: {type(k1)}")
    if not isinstance(k2, (int, float)):
        raise TypeError(f"k2必须是数值类型，当前类型: {type(k2)}")
    
    if period_years <= 0:
        raise ValueError(f"period_years必须为正值，当前值: {period_years}")
    if k1 <= 0:
        raise ValueError(f"k1必须为正值，当前值: {k1}")
    if k2 <= 0:
        raise ValueError(f"k2必须为正值，当前值: {k2}")
    
    # 计算轨道半长轴
    # a = (P/2π) * (K1 + K2)
    # P需要转换为秒，K已经是km/s
    period_seconds = period_years * YEAR_TO_SEC
    separation_km = (period_seconds / (2 * np.pi)) * (k1 + k2)
    separation_au = separation_km / AU_TO_KM
    
    return {
        'result': separation_km,
        'metadata': {
            'period_years': period_years,
            'period_seconds': period_seconds,
            'k1_km_s': k1,
            'k2_km_s': k2,
            'separation_km': separation_km,
            'separation_au': separation_au,
            'formula': 'a = (P/2π)(K1+K2)'
        }
    }


def apply_keplers_third_law(period1: float, period2: float, 
                            separation1: float, separation2: float) -> dict:
    """
    应用开普勒第三定律计算两个系统的质量比
    
    物理原理：P²/a³ = 4π²/G(M1+M2)，对于两个系统有 (P1²/P2²) = (a1³/a2³) * (M2/M1)
    
    Args:
        period1: 系统1的轨道周期 (年)
        period2: 系统2的轨道周期 (年)
        separation1: 系统1的轨道半长轴 (km)
        separation2: 系统2的轨道半长轴 (km)
    
    Returns:
        dict: {
            'result': 系统1与系统2的质量比 M1/M2,
            'metadata': {
                'period_ratio': P1/P2,
                'separation_ratio': a1/a2,
                'formula': '开普勒第三定律'
            }
        }
    
    Example:
        >>> result = apply_keplers_third_law(2.0, 1.0, 1e9, 8e8)
        >>> print(result['result'])
    """
    # === 完整边界检查 ===
    if not all(isinstance(x, (int, float)) for x in [period1, period2, separation1, separation2]):
        raise TypeError("所有参数必须是数值类型")
    
    if any(x <= 0 for x in [period1, period2, separation1, separation2]):
        raise ValueError("所有参数必须为正值")
    
    # 应用开普勒第三定律
    # (P1/P2)² = (a1/a2)³ * (M2/M1)
    # M1/M2 = (a1/a2)³ / (P1/P2)²
    
    period_ratio = period1 / period2
    separation_ratio = separation1 / separation2
    
    mass_ratio = (separation_ratio ** 3) / (period_ratio ** 2)
    
    return {
        'result': mass_ratio,
        'metadata': {
            'period1_years': period1,
            'period2_years': period2,
            'separation1_km': separation1,
            'separation2_km': separation2,
            'period_ratio': period_ratio,
            'separation_ratio': separation_ratio,
            'formula': 'M1/M2 = (a1/a2)³ / (P1/P2)²',
            'principle': 'Kepler\'s Third Law'
        }
    }


def query_star_parameters(star_name: str) -> dict:
    """
    从SIMBAD数据库查询恒星基本参数
    
    通过Astroquery访问SIMBAD天文数据库，获取恒星的基本物理参数。
    
    Args:
        star_name: 恒星名称或标识符 (如 'Sirius', 'HD 48915', 'HIP 32349')
    
    Returns:
        dict: {
            'result': {
                'name': 恒星名称,
                'ra': 赤经,
                'dec': 赤纬,
                'spectral_type': 光谱型,
                'magnitude': 视星等
            },
            'metadata': {
                'database': 'SIMBAD',
                'query_time': 查询时间戳
            }
        }
    
    Example:
        >>> result = query_star_parameters('Sirius')
        >>> print(result['result']['spectral_type'])
    """
    # === 完整边界检查 ===
    if not isinstance(star_name, str):
        raise TypeError(f"star_name必须是字符串类型，当前类型: {type(star_name)}")
    
    if not star_name.strip():
        raise ValueError("star_name不能为空字符串")
    
    try:
        # 配置SIMBAD查询字段
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('sptype', 'flux(V)')
        
        # 查询恒星
        result_table = custom_simbad.query_object(star_name)
        
        if result_table is None:
            return {
                'result': None,
                'metadata': {
                    'database': 'SIMBAD',
                    'query_time': datetime.now().isoformat(),
                    'status': 'not_found',
                    'message': f'未找到恒星: {star_name}'
                }
            }
        
        # 提取数据
        star_data = {
            'name': result_table['MAIN_ID'][0],
            'ra': float(result_table['RA'][0]) if result_table['RA'][0] else None,
            'dec': float(result_table['DEC'][0]) if result_table['DEC'][0] else None,
            'spectral_type': result_table['SP_TYPE'][0] if 'SP_TYPE' in result_table.colnames else None,
            'magnitude_v': float(result_table['FLUX_V'][0]) if 'FLUX_V' in result_table.colnames and result_table['FLUX_V'][0] else None
        }
        
        return {
            'result': star_data,
            'metadata': {
                'database': 'SIMBAD',
                'query_time': datetime.now().isoformat(),
                'status': 'success'
            }
        }
        
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'database': 'SIMBAD',
                'query_time': datetime.now().isoformat(),
                'status': 'error',
                'error_message': str(e)
            }
        }


def calculate_orbital_velocity(semi_major_axis: float, period: float, 
                               mass_ratio: float) -> dict:
    """
    计算双星系统中各星的轨道速度
    
    物理原理：v = 2πa/P，其中a是该星到质心的距离，由质量比确定。
    
    Args:
        semi_major_axis: 系统轨道半长轴 (km)
        period: 轨道周期 (年)
        mass_ratio: 质量比 m1/m2
    
    Returns:
        dict: {
            'result': {
                'v1': 第一颗星的轨道速度 (km/s),
                'v2': 第二颗星的轨道速度 (km/s)
            },
            'metadata': {
                'a1': 第一颗星到质心距离 (km),
                'a2': 第二颗星到质心距离 (km)
            }
        }
    
    Example:
        >>> result = calculate_orbital_velocity(1e9, 2.0, 0.5)
        >>> print(result['result'])
    """
    # === 完整边界检查 ===
    if not all(isinstance(x, (int, float)) for x in [semi_major_axis, period, mass_ratio]):
        raise TypeError("所有参数必须是数值类型")
    
    if semi_major_axis <= 0:
        raise ValueError(f"semi_major_axis必须为正值，当前值: {semi_major_axis}")
    if period <= 0:
        raise ValueError(f"period必须为正值，当前值: {period}")
    if mass_ratio <= 0:
        raise ValueError(f"mass_ratio必须为正值，当前值: {mass_ratio}")
    
    # 计算各星到质心的距离
    # a1 = a * m2/(m1+m2) = a * 1/(q+1), where q = m1/m2
    # a2 = a * m1/(m1+m2) = a * q/(q+1)
    
    a1 = semi_major_axis / (mass_ratio + 1)
    a2 = semi_major_axis * mass_ratio / (mass_ratio + 1)
    
    # 计算轨道速度
    period_seconds = period * YEAR_TO_SEC
    v1 = 2 * np.pi * a1 / period_seconds
    v2 = 2 * np.pi * a2 / period_seconds
    
    return {
        'result': {
            'v1_km_s': v1,
            'v2_km_s': v2
        },
        'metadata': {
            'a1_km': a1,
            'a2_km': a2,
            'mass_ratio': mass_ratio,
            'period_years': period,
            'formula': 'v = 2πa/P'
        }
    }


# ============ 第二层：组合工具函数（Composite Tools） ============

def analyze_binary_system_mass(period: float, k1: float, k2: float) -> dict:
    """
    综合分析双星系统，计算质量比、轨道参数和系统总质量相对值
    
    这是一个组合函数，整合了质量比计算、轨道分离计算和速度验证。
    
    Args:
        period: 轨道周期 (年)
        k1: 第一颗星的径向速度振幅 (km/s)
        k2: 第二颗星的径向速度振幅 (km/s)
    
    Returns:
        dict: {
            'result': {
                'mass_ratio': 质量比 m1/m2,
                'separation_km': 轨道半长轴 (km),
                'separation_au': 轨道半长轴 (AU),
                'total_mass_factor': 相对质量因子 (用于比较)
            },
            'metadata': {
                'period_years': 周期,
                'rv_amplitudes': [k1, k2],
                'analysis_steps': 分析步骤列表
            }
        }
    
    Example:
        >>> result = analyze_binary_system_mass(2.0, 10.0, 5.0)
        >>> print(result['result']['mass_ratio'])
    """
    # === 参数完全可序列化检查 ===
    if not all(isinstance(x, (int, float)) for x in [period, k1, k2]):
        raise TypeError("所有参数必须是数值类型")
    
    analysis_steps = []
    
    # 步骤1：计算质量比
    ## using calculate_mass_ratio_from_rv, and get mass_ratio returns
    mass_ratio_result = calculate_mass_ratio_from_rv(k1, k2)
    mass_ratio = mass_ratio_result['result']
    analysis_steps.append(f"Step 1: Calculated mass ratio m1/m2 = {mass_ratio:.4f}")
    
    # 步骤2：计算轨道分离
    ## using calculate_orbital_separation, and get separation returns
    separation_result = calculate_orbital_separation(period, k1, k2)
    separation_km = separation_result['result']
    separation_au = separation_result['metadata']['separation_au']
    analysis_steps.append(f"Step 2: Calculated orbital separation a = {separation_au:.4f} AU")
    
    # 步骤3：验证轨道速度
    ## using calculate_orbital_velocity, and get velocity returns
    velocity_result = calculate_orbital_velocity(separation_km, period, mass_ratio)
    v1_calc = velocity_result['result']['v1_km_s']
    v2_calc = velocity_result['result']['v2_km_s']
    analysis_steps.append(f"Step 3: Verified velocities v1={v1_calc:.2f} km/s, v2={v2_calc:.2f} km/s")
    
    # 计算相对质量因子 (用于系统间比较)
    # 从开普勒第三定律: M ∝ a³/P²
    total_mass_factor = (separation_km ** 3) / ((period * YEAR_TO_SEC) ** 2)
    
    return {
        'result': {
            'mass_ratio_m1_m2': mass_ratio,
            'separation_km': separation_km,
            'separation_au': separation_au,
            'total_mass_factor': total_mass_factor,
            'verified_v1_km_s': v1_calc,
            'verified_v2_km_s': v2_calc
        },
        'metadata': {
            'period_years': period,
            'rv_amplitudes_km_s': [k1, k2],
            'analysis_steps': analysis_steps,
            'functions_called': [
                'calculate_mass_ratio_from_rv',
                'calculate_orbital_separation',
                'calculate_orbital_velocity'
            ]
        }
    }


def compare_binary_systems(system1_params: dict, system2_params: dict) -> dict:
    """
    比较两个双星系统的质量
    
    综合应用开普勒第三定律和径向速度分析，计算两个系统的质量比。
    
    Args:
        system1_params: 系统1参数 {'period': 周期(年), 'k1': RV1(km/s), 'k2': RV2(km/s)}
        system2_params: 系统2参数 {'period': 周期(年), 'k1': RV1(km/s), 'k2': RV2(km/s)}
    
    Returns:
        dict: {
            'result': {
                'mass_ratio_system1_to_system2': M1/M2,
                'system1_analysis': 系统1详细分析,
                'system2_analysis': 系统2详细分析
            },
            'metadata': {
                'comparison_method': '开普勒第三定律',
                'calculation_steps': 计算步骤
            }
        }
    
    Example:
        >>> s1 = {'period': 2.0, 'k1': 10.0, 'k2': 5.0}
        >>> s2 = {'period': 1.0, 'k1': 15.0, 'k2': 10.0}
        >>> result = compare_binary_systems(s1, s2)
        >>> print(result['result']['mass_ratio_system1_to_system2'])
    """
    # === 参数完全可序列化检查 ===
    if not isinstance(system1_params, dict):
        raise TypeError("system1_params必须是dict")
    if not isinstance(system2_params, dict):
        raise TypeError("system2_params必须是dict")
    
    required_keys = ['period', 'k1', 'k2']
    for key in required_keys:
        if key not in system1_params:
            raise ValueError(f"system1_params缺少必需键: {key}")
        if key not in system2_params:
            raise ValueError(f"system2_params缺少必需键: {key}")
    
    calculation_steps = []
    
    # 步骤1：分析系统1
    ## using analyze_binary_system_mass, and get system1 complete analysis returns
    system1_analysis = analyze_binary_system_mass(
        system1_params['period'],
        system1_params['k1'],
        system1_params['k2']
    )
    calculation_steps.append("Step 1: Analyzed system 1 parameters")
    
    # 步骤2：分析系统2
    ## using analyze_binary_system_mass, and get system2 complete analysis returns
    system2_analysis = analyze_binary_system_mass(
        system2_params['period'],
        system2_params['k1'],
        system2_params['k2']
    )
    calculation_steps.append("Step 2: Analyzed system 2 parameters")
    
    # 步骤3：应用开普勒第三定律比较质量
    ## using apply_keplers_third_law, and get mass ratio returns
    mass_ratio_result = apply_keplers_third_law(
        system1_params['period'],
        system2_params['period'],
        system1_analysis['result']['separation_km'],
        system2_analysis['result']['separation_km']
    )
    mass_ratio = mass_ratio_result['result']
    calculation_steps.append(f"Step 3: Applied Kepler's 3rd law, M1/M2 = {mass_ratio:.4f}")
    
    # 保存详细结果到文件
    detailed_results = {
        'system1': system1_analysis,
        'system2': system2_analysis,
        'mass_ratio': mass_ratio,
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = './mid_result/physics/binary_comparison.json'
    with open(filepath, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    return {
        'result': {
            'mass_ratio_system1_to_system2': mass_ratio,
            'system1_total_mass_factor': system1_analysis['result']['total_mass_factor'],
            'system2_total_mass_factor': system2_analysis['result']['total_mass_factor'],
            'system1_separation_au': system1_analysis['result']['separation_au'],
            'system2_separation_au': system2_analysis['result']['separation_au']
        },
        'metadata': {
            'comparison_method': 'Kepler\'s Third Law',
            'calculation_steps': calculation_steps,
            'detailed_results_file': filepath,
            'functions_called': [
                'analyze_binary_system_mass (x2)',
                'apply_keplers_third_law'
            ]
        }
    }


def simulate_binary_orbit(period: float, k1: float, k2: float, 
                          num_points: int = 100) -> dict:
    """
    模拟双星系统的轨道运动
    
    基于开普勒定律生成双星系统的轨道轨迹数据，用于可视化和分析。
    
    Args:
        period: 轨道周期 (年)
        k1: 第一颗星的径向速度振幅 (km/s)
        k2: 第二颗星的径向速度振幅 (km/s)
        num_points: 轨道上的采样点数，默认100
    
    Returns:
        dict: {
            'result': {
                'time_array': 时间数组 (年),
                'star1_x': 星1的x坐标数组 (AU),
                'star1_y': 星1的y坐标数组 (AU),
                'star2_x': 星2的x坐标数组 (AU),
                'star2_y': 星2的y坐标数组 (AU),
                'rv1': 星1的径向速度数组 (km/s),
                'rv2': 星2的径向速度数组 (km/s)
            },
            'metadata': {
                'num_points': 采样点数,
                'period_years': 周期
            }
        }
    
    Example:
        >>> result = simulate_binary_orbit(2.0, 10.0, 5.0, 50)
        >>> print(len(result['result']['time_array']))
        50
    """
    # === 完整边界检查 ===
    if not all(isinstance(x, (int, float)) for x in [period, k1, k2]):
        raise TypeError("period, k1, k2必须是数值类型")
    if not isinstance(num_points, int):
        raise TypeError(f"num_points必须是整数，当前类型: {type(num_points)}")
    
    if period <= 0:
        raise ValueError(f"period必须为正值，当前值: {period}")
    if k1 <= 0 or k2 <= 0:
        raise ValueError("k1和k2必须为正值")
    if num_points < 10:
        raise ValueError(f"num_points必须>=10，当前值: {num_points}")
    
    # 步骤1：获取系统参数
    ## using analyze_binary_system_mass, and get orbital parameters returns
    system_analysis = analyze_binary_system_mass(period, k1, k2)
    mass_ratio = system_analysis['result']['mass_ratio_m1_m2']
    separation_km = system_analysis['result']['separation_km']
    
    # 计算各星到质心的距离
    a1_km = separation_km / (mass_ratio + 1)
    a2_km = separation_km * mass_ratio / (mass_ratio + 1)
    
    # 转换为AU
    a1_au = a1_km / AU_TO_KM
    a2_au = a2_km / AU_TO_KM
    
    # 生成时间数组
    time_array = np.linspace(0, period, num_points)
    
    # 计算轨道位置（假设圆轨道，从x轴正方向开始）
    theta = 2 * np.pi * time_array / period
    
    star1_x = a1_au * np.cos(theta)
    star1_y = a1_au * np.sin(theta)
    
    # 星2在相反方向
    star2_x = -a2_au * np.cos(theta)
    star2_y = -a2_au * np.sin(theta)
    
    # 计算径向速度（假设观测者在y轴方向）
    rv1 = -k1 * np.sin(theta)
    rv2 = -k2 * np.sin(theta)
    
    return {
        'result': {
            'time_array_years': time_array.tolist(),
            'star1_x_au': star1_x.tolist(),
            'star1_y_au': star1_y.tolist(),
            'star2_x_au': star2_x.tolist(),
            'star2_y_au': star2_y.tolist(),
            'rv1_km_s': rv1.tolist(),
            'rv2_km_s': rv2.tolist()
        },
        'metadata': {
            'num_points': num_points,
            'period_years': period,
            'mass_ratio': mass_ratio,
            'separation_au': separation_km / AU_TO_KM,
            'a1_au': a1_au,
            'a2_au': a2_au,
            'orbit_type': 'circular',
            'functions_called': ['analyze_binary_system_mass']
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def visualize_binary_orbit(orbit_data: dict, system_name: str = "Binary System",
                          save_dir: str = './tool_images/') -> dict:
    """
    可视化双星系统的轨道运动
    
    生成双星轨道的2D轨迹图和径向速度曲线图。
    
    Args:
        orbit_data: 轨道数据字典，包含time_array, star1_x等键
        system_name: 系统名称，用于图表标题
        save_dir: 保存目录
    
    Returns:
        dict: {
            'result': '可视化完成',
            'metadata': {
                'orbit_plot_path': 轨道图路径,
                'rv_plot_path': 径向速度图路径
            }
        }
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取数据
    time = np.array(orbit_data['time_array_years'])
    star1_x = np.array(orbit_data['star1_x_au'])
    star1_y = np.array(orbit_data['star1_y_au'])
    star2_x = np.array(orbit_data['star2_x_au'])
    star2_y = np.array(orbit_data['star2_y_au'])
    rv1 = np.array(orbit_data['rv1_km_s'])
    rv2 = np.array(orbit_data['rv2_km_s'])
    
    # 图1：轨道轨迹
    plt.figure(figsize=(10, 10))
    plt.plot(star1_x, star1_y, 'b-', linewidth=2, label='Star 1')
    plt.plot(star2_x, star2_y, 'r-', linewidth=2, label='Star 2')
    plt.plot(0, 0, 'ko', markersize=10, label='Center of Mass')
    plt.plot(star1_x[0], star1_y[0], 'bs', markersize=12, label='Start Position')
    plt.plot(star2_x[0], star2_y[0], 'rs', markersize=12)
    
    plt.xlabel('X Position (AU)', fontsize=14)
    plt.ylabel('Y Position (AU)', fontsize=14)
    plt.title(f'{system_name} - Orbital Trajectories', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    orbit_plot_path = os.path.join(save_dir, f'{system_name.replace(" ", "_")}_orbit.png')
    plt.savefig(orbit_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: Orbit Plot | PATH: {orbit_plot_path}")
    
    # 图2：径向速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(time, rv1, 'b-', linewidth=2, label='Star 1 RV')
    plt.plot(time, rv2, 'r-', linewidth=2, label='Star 2 RV')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Radial Velocity (km/s)', fontsize=14)
    plt.title(f'{system_name} - Radial Velocity Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    rv_plot_path = os.path.join(save_dir, f'{system_name.replace(" ", "_")}_rv.png')
    plt.savefig(rv_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: RV Plot | PATH: {rv_plot_path}")
    
    return {
        'result': f'Visualization completed for {system_name}',
        'metadata': {
            'orbit_plot_path': orbit_plot_path,
            'rv_plot_path': rv_plot_path,
            'num_data_points': len(time)
        }
    }


def visualize_system_comparison(system1_data: dict, system2_data: dict,
                                save_dir: str = './tool_images/') -> dict:
    """
    对比可视化两个双星系统
    
    生成并排对比图，展示两个系统的轨道尺度和径向速度差异。
    
    Args:
        system1_data: 系统1的轨道数据
        system2_data: 系统2的轨道数据
        save_dir: 保存目录
    
    Returns:
        dict: {
            'result': '对比可视化完成',
            'metadata': {
                'comparison_plot_path': 对比图路径
            }
        }
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 系统1轨道
    ax = axes[0, 0]
    s1_x1 = np.array(system1_data['star1_x_au'])
    s1_y1 = np.array(system1_data['star1_y_au'])
    s1_x2 = np.array(system1_data['star2_x_au'])
    s1_y2 = np.array(system1_data['star2_y_au'])
    
    ax.plot(s1_x1, s1_y1, 'b-', linewidth=2, label='Star 1')
    ax.plot(s1_x2, s1_y2, 'r-', linewidth=2, label='Star 2')
    ax.plot(0, 0, 'ko', markersize=8)
    ax.set_xlabel('X (AU)', fontsize=12)
    ax.set_ylabel('Y (AU)', fontsize=12)
    ax.set_title('System 1 Orbit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 系统2轨道
    ax = axes[0, 1]
    s2_x1 = np.array(system2_data['star1_x_au'])
    s2_y1 = np.array(system2_data['star1_y_au'])
    s2_x2 = np.array(system2_data['star2_x_au'])
    s2_y2 = np.array(system2_data['star2_y_au'])
    
    ax.plot(s2_x1, s2_y1, 'b-', linewidth=2, label='Star 1')
    ax.plot(s2_x2, s2_y2, 'r-', linewidth=2, label='Star 2')
    ax.plot(0, 0, 'ko', markersize=8)
    ax.set_xlabel('X (AU)', fontsize=12)
    ax.set_ylabel('Y (AU)', fontsize=12)
    ax.set_title('System 2 Orbit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 系统1径向速度
    ax = axes[1, 0]
    time1 = np.array(system1_data['time_array_years'])
    rv1_1 = np.array(system1_data['rv1_km_s'])
    rv1_2 = np.array(system1_data['rv2_km_s'])
    
    ax.plot(time1, rv1_1, 'b-', linewidth=2, label='Star 1')
    ax.plot(time1, rv1_2, 'r-', linewidth=2, label='Star 2')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('RV (km/s)', fontsize=12)
    ax.set_title('System 1 Radial Velocities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 系统2径向速度
    ax = axes[1, 1]
    time2 = np.array(system2_data['time_array_years'])
    rv2_1 = np.array(system2_data['rv1_km_s'])
    rv2_2 = np.array(system2_data['rv2_km_s'])
    
    ax.plot(time2, rv2_1, 'b-', linewidth=2, label='Star 1')
    ax.plot(time2, rv2_2, 'r-', linewidth=2, label='Star 2')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('RV (km/s)', fontsize=12)
    ax.set_title('System 2 Radial Velocities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_path = os.path.join(save_dir, 'binary_systems_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: Comparison Plot | PATH: {comparison_path}")
    
    return {
        'result': 'System comparison visualization completed',
        'metadata': {
            'comparison_plot_path': comparison_path
        }
    }


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【双星系统质量比较】+【参数扫描】+【数据库查询】
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - 比较两个双星系统的质量")
    print("=" * 60)
    print("问题描述：给定两个双星系统的轨道周期和径向速度振幅，")
    print("计算系统1相对于系统2的质量比。")
    print("系统1: P=2年, K1=10 km/s, K2=5 km/s")
    print("系统2: P=1年, K1=15 km/s, K2=10 km/s")
    print("-" * 60)
    
    # 步骤1：定义系统参数
    # 调用函数：无（直接定义参数）
    system1_params = {'period': 2.0, 'k1': 10.0, 'k2': 5.0}
    system2_params = {'period': 1.0, 'k1': 15.0, 'k2': 10.0}
    print(f"FUNCTION_CALL: Parameter Definition | PARAMS: system1={system1_params}, system2={system2_params} | RESULT: Parameters defined")
    
    # 步骤2：比较两个系统
    # 调用函数：compare_binary_systems()，该函数内部调用了 analyze_binary_system_mass() 和 apply_keplers_third_law()
    comparison_result = compare_binary_systems(system1_params, system2_params)
    mass_ratio = comparison_result['result']['mass_ratio_system1_to_system2']
    print(f"FUNCTION_CALL: compare_binary_systems | PARAMS: system1={system1_params}, system2={system2_params} | RESULT: mass_ratio={mass_ratio:.4f}")
    print(f"步骤2结果：系统1/系统2的质量比 = {mass_ratio:.4f}")
    print(f"系统1轨道半长轴 = {comparison_result['result']['system1_separation_au']:.4f} AU")
    print(f"系统2轨道半长轴 = {comparison_result['result']['system2_separation_au']:.4f} AU")
    
    # 步骤3：模拟轨道用于可视化
    # 调用函数：simulate_binary_orbit()，该函数内部调用了 analyze_binary_system_mass()
    orbit1_data = simulate_binary_orbit(
        system1_params['period'],
        system1_params['k1'],
        system1_params['k2'],
        num_points=100
    )
    print(f"FUNCTION_CALL: simulate_binary_orbit | PARAMS: period=2.0, k1=10.0, k2=5.0, num_points=100 | RESULT: Generated 100 orbital points")
    
    orbit2_data = simulate_binary_orbit(
        system2_params['period'],
        system2_params['k1'],
        system2_params['k2'],
        num_points=100
    )
    print(f"FUNCTION_CALL: simulate_binary_orbit | PARAMS: period=1.0, k1=15.0, k2=10.0, num_points=100 | RESULT: Generated 100 orbital points")
    
    # 步骤4：可视化对比
    # 调用函数：visualize_system_comparison()
    vis_result = visualize_system_comparison(
        orbit1_data['result'],
        orbit2_data['result']
    )
    print(f"FUNCTION_CALL: visualize_system_comparison | PARAMS: system1_orbit_data, system2_orbit_data | RESULT: {vis_result['result']}")
    
    print(f"✓ 场景1完成：系统1的质量是系统2的 {mass_ratio:.4f} 倍 (约0.4倍)")
    print(f"FINAL_ANSWER: {mass_ratio:.4f}")
    print()
    
    print("=" * 60)
    print("场景2：参数扫描 - 分析周期对质量比的影响")
    print("=" * 60)
    print("问题描述：固定径向速度振幅，扫描不同周期组合，")
    print("分析周期比对系统质量比的影响。")
    print("-" * 60)
    
    # 步骤1：定义扫描参数
    # 调用函数：无（直接定义）
    period_ratios = [1.5, 2.0, 2.5, 3.0]
    k1_fixed, k2_fixed = 10.0, 5.0
    reference_period = 1.0
    print(f"FUNCTION_CALL: Scan Setup | PARAMS: period_ratios={period_ratios}, k1={k1_fixed}, k2={k2_fixed} | RESULT: Scan parameters defined")
    
    # 步骤2：执行参数扫描
    # 调用函数：compare_binary_systems() (多次调用)
    scan_results = []
    for p_ratio in period_ratios:
        sys1 = {'period': p_ratio * reference_period, 'k1': k1_fixed, 'k2': k2_fixed}
        sys2 = {'period': reference_period, 'k1': k1_fixed, 'k2': k2_fixed}
        
        result = compare_binary_systems(sys1, sys2)
        mass_ratio_scan = result['result']['mass_ratio_system1_to_system2']
        scan_results.append({
            'period_ratio': p_ratio,
            'mass_ratio': mass_ratio_scan
        })
        print(f"FUNCTION_CALL: compare_binary_systems | PARAMS: P1/P2={p_ratio:.1f} | RESULT: M1/M2={mass_ratio_scan:.4f}")
    
    # 步骤3：可视化扫描结果
    # 调用函数：matplotlib绘图
    plt.figure(figsize=(10, 6))
    p_ratios = [r['period_ratio'] for r in scan_results]
    m_ratios = [r['mass_ratio'] for r in scan_results]
    plt.plot(p_ratios, m_ratios, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Period Ratio (P1/P2)', fontsize=14)
    plt.ylabel('Mass Ratio (M1/M2)', fontsize=14)
    plt.title('Effect of Period Ratio on Mass Ratio\n(Fixed RV Amplitudes)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    scan_plot_path = './tool_images/period_scan.png'
    plt.savefig(scan_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: Period Scan Plot | PATH: {scan_plot_path}")
    
    print(f"✓ 场景2完成：周期比从1.5到3.0时，质量比从{m_ratios[0]:.3f}变化到{m_ratios[-1]:.3f}")
    print()
    
    print("=" * 60)
    print("场景3：数据库集成 - 查询真实双星系统")
    print("=" * 60)
    print("问题描述：从SIMBAD数据库查询著名双星系统的参数，")
    print("演示工具包与天文数据库的集成能力。")
    print("-" * 60)
    
    # 步骤1：查询著名双星系统
    # 调用函数：query_star_parameters()
    famous_binaries = ['Sirius', 'Algol', 'Mizar']
    
    for star_name in famous_binaries:
        star_info = query_star_parameters(star_name)
        
        if star_info['result'] is not None:
            info = star_info['result']
            print(f"FUNCTION_CALL: query_star_parameters | PARAMS: star_name='{star_name}' | RESULT: Found")
            print(f"  恒星: {info['name']}")
            print(f"  光谱型: {info['spectral_type']}")
            print(f"  视星等: {info['magnitude_v']}")
            print(f"  赤经/赤纬: {info['ra']}, {info['dec']}")
        else:
            print(f"FUNCTION_CALL: query_star_parameters | PARAMS: star_name='{star_name}' | RESULT: Not found")
            print(f"  未找到 {star_name} 的数据")
        print()
    
    # 步骤2：演示假设参数分析
    # 调用函数：analyze_binary_system_mass()
    print("假设Sirius双星系统参数（示例）：")
    sirius_params = {'period': 50.0, 'k1': 5.5, 'k2': 8.2}
    sirius_analysis = analyze_binary_system_mass(
        sirius_params['period'],
        sirius_params['k1'],
        sirius_params['k2']
    )
    print(f"FUNCTION_CALL: analyze_binary_system_mass | PARAMS: period=50.0, k1=5.5, k2=8.2 | RESULT: mass_ratio={sirius_analysis['result']['mass_ratio_m1_m2']:.4f}")
    print(f"  质量比 m1/m2 = {sirius_analysis['result']['mass_ratio_m1_m2']:.4f}")
    print(f"  轨道半长轴 = {sirius_analysis['result']['separation_au']:.2f} AU")
    
    print(f"✓ 场景3完成：成功查询{len(famous_binaries)}个双星系统并演示参数分析")
    print()
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始问题的完整流程（质量比≈0.4）")
    print("- 场景2展示了工具的参数扫描能力（周期比影响分析）")
    print("- 场景3展示了工具与SIMBAD数据库的集成能力")
    print(f"- 生成的可视化文件保存在: ./tool_images/")
    print(f"- 中间结果保存在: ./mid_result/physics/")


if __name__ == "__main__":
    main()