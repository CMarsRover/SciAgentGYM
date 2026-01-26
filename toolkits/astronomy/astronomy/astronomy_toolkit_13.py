# Filename: astronomy_toolkit.py
"""
天文观测计算工具包

主要功能：
1. 恒星视星等计算：基于距离模数和消光效应计算观测星等
2. 天文台可见性分析：判断恒星在不同纬度天文台的可观测性
3. 光谱仪适配性评估：根据仪器灵敏度限制筛选可观测目标
4. 坐标系统转换：支持赤道坐标系统的时角/角度转换

依赖库：
pip install numpy scipy astropy astroquery matplotlib
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import os
from datetime import datetime

# 导入天文专属库
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# 全局常量
EXTINCTION_COEFFICIENT_V = 3.1  # A_V = 3.1 * E(B-V)
ESPRESSO_LIMIT = 17.0  # ESPRESSO视星等限制 (mag)
HIRES_LIMIT = 16.0     # HIRES视星等限制 (mag)
PARANAL_LAT = -24.6    # Paranal天文台纬度 (度)
KECK_LAT = 19.8        # Keck天文台纬度 (度)

# 创建中间结果目录
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)


# ============ 第一层：原子工具函数（Atomic Tools） ============

def convert_ra_to_degrees(ra: Union[float, str]) -> dict:
    """
    将赤经从时角格式转换为角度格式
    
    天文学中赤经可用两种格式表示：
    - 角度格式：0-360度
    - 时角格式：0h-24h (1h = 15度)
    
    Args:
        ra: 赤经值，可以是角度(float)或时角字符串(如'11h'或'11 h')
        
    Returns:
        dict: {
            'result': 赤经角度值(度),
            'metadata': {'input_format': 'degrees'或'hours', 'input_value': 原始输入}
        }
        
    Example:
        >>> result = convert_ra_to_degrees('11h')
        >>> print(result['result'])
        165.0
    """
    # === 完整边界检查 ===
    if ra is None:
        raise ValueError("RA不能为None")
    
    # 检查输入类型
    if isinstance(ra, (int, float)):
        # 已经是角度格式
        ra_deg = float(ra)
        if not (0 <= ra_deg <= 360):
            raise ValueError(f"RA角度必须在0-360度之间，当前值: {ra_deg}")
        input_format = 'degrees'
        input_value = ra_deg
        
    elif isinstance(ra, str):
        # 时角格式，需要转换
        ra_str = ra.strip().lower()
        if 'h' not in ra_str:
            raise ValueError(f"时角格式必须包含'h'，如'11h'，当前值: {ra}")
        
        try:
            # 提取数值部分
            hours = float(ra_str.replace('h', '').strip())
            if not (0 <= hours <= 24):
                raise ValueError(f"时角必须在0-24h之间，当前值: {hours}h")
            
            # 转换为角度 (1h = 15度)
            ra_deg = hours * 15.0
            input_format = 'hours'
            input_value = ra_str
            
        except ValueError as e:
            raise ValueError(f"无法解析时角格式'{ra}': {e}")
    else:
        raise TypeError(f"RA必须是float或str类型，当前类型: {type(ra)}")
    
    return {
        'result': ra_deg,
        'metadata': {
            'input_format': input_format,
            'input_value': input_value,
            'output_degrees': ra_deg
        }
    }


def calculate_distance_modulus(distance_pc: float) -> dict:
    """
    计算距离模数 (m - M)
    
    距离模数公式：μ = 5 * log10(d) - 5
    其中 d 为距离(秒差距)，该公式将绝对星等转换为视星等的距离项
    
    Args:
        distance_pc: 距离/秒差距(pc)，范围 > 0
        
    Returns:
        dict: {
            'result': 距离模数值(mag),
            'metadata': {'distance_pc': 输入距离, 'formula': '5*log10(d)-5'}
        }
        
    Example:
        >>> result = calculate_distance_modulus(10.0)
        >>> print(result['result'])
        5.0
    """
    # === 完整边界检查 ===
    if not isinstance(distance_pc, (int, float)):
        raise TypeError(f"距离必须是数值类型，当前类型: {type(distance_pc)}")
    
    if distance_pc <= 0:
        raise ValueError(f"距离必须大于0，当前值: {distance_pc} pc")
    
    if distance_pc < 1e-6:
        raise ValueError(f"距离过小，可能导致数值不稳定: {distance_pc} pc")
    
    if distance_pc > 1e9:
        raise ValueError(f"距离过大，超出合理范围: {distance_pc} pc")
    
    # === 核心计算 ===
    distance_modulus = 5.0 * np.log10(distance_pc) - 5.0
    
    return {
        'result': float(distance_modulus),
        'metadata': {
            'distance_pc': distance_pc,
            'formula': '5*log10(d)-5',
            'log10_distance': np.log10(distance_pc)
        }
    }


def calculate_extinction(ebv: float, coefficient: float = EXTINCTION_COEFFICIENT_V) -> dict:
    """
    计算V波段总消光 A_V
    
    消光效应：星际尘埃导致星光减弱
    公式：A_V = R_V * E(B-V)，其中 R_V ≈ 3.1 (银河系典型值)
    
    Args:
        ebv: 色余 E(B-V)/mag，范围 >= 0
        coefficient: 消光系数 R_V，默认3.1
        
    Returns:
        dict: {
            'result': V波段总消光 A_V/mag,
            'metadata': {'ebv': 输入色余, 'coefficient': 使用的系数}
        }
        
    Example:
        >>> result = calculate_extinction(0.6)
        >>> print(result['result'])
        1.86
    """
    # === 完整边界检查 ===
    if not isinstance(ebv, (int, float)):
        raise TypeError(f"E(B-V)必须是数值类型，当前类型: {type(ebv)}")
    
    if ebv < 0:
        raise ValueError(f"E(B-V)不能为负值，当前值: {ebv}")
    
    if ebv > 10:
        raise ValueError(f"E(B-V)过大，超出合理范围: {ebv}")
    
    if not isinstance(coefficient, (int, float)):
        raise TypeError(f"消光系数必须是数值类型，当前类型: {type(coefficient)}")
    
    if coefficient <= 0:
        raise ValueError(f"消光系数必须大于0，当前值: {coefficient}")
    
    # === 核心计算 ===
    extinction_v = coefficient * ebv
    
    return {
        'result': float(extinction_v),
        'metadata': {
            'ebv': ebv,
            'coefficient': coefficient,
            'formula': f'A_V = {coefficient} * E(B-V)'
        }
    }


def calculate_apparent_magnitude(
    absolute_mag: float,
    distance_pc: float,
    extinction: float = 0.0
) -> dict:
    """
    计算视星等 m
    
    视星等公式：m = M + 5*log10(d) - 5 + A_V
    - M: 绝对星等（恒星在10pc处的亮度）
    - d: 距离(pc)
    - A_V: V波段消光
    
    Args:
        absolute_mag: 绝对星等 M/mag
        distance_pc: 距离/pc，范围 > 0
        extinction: V波段消光 A_V/mag，默认0（无消光）
        
    Returns:
        dict: {
            'result': 视星等 m/mag,
            'metadata': {
                'absolute_mag': 绝对星等,
                'distance_modulus': 距离模数,
                'extinction': 消光值,
                'components': 各项分量
            }
        }
        
    Example:
        >>> result = calculate_apparent_magnitude(15.5, 10.0, 1.24)
        >>> print(result['result'])
        16.74
    """
    # === 完整边界检查 ===
    if not isinstance(absolute_mag, (int, float)):
        raise TypeError(f"绝对星等必须是数值类型，当前类型: {type(absolute_mag)}")
    
    if not isinstance(distance_pc, (int, float)):
        raise TypeError(f"距离必须是数值类型，当前类型: {type(distance_pc)}")
    
    if distance_pc <= 0:
        raise ValueError(f"距离必须大于0，当前值: {distance_pc} pc")
    
    if not isinstance(extinction, (int, float)):
        raise TypeError(f"消光必须是数值类型，当前类型: {type(extinction)}")
    
    if extinction < 0:
        raise ValueError(f"消光不能为负值，当前值: {extinction}")
    
    # === 核心计算 ===
    # 调用原子函数计算距离模数
    dm_result = calculate_distance_modulus(distance_pc)
    distance_modulus = dm_result['result']
    
    # 计算视星等
    apparent_mag = absolute_mag + distance_modulus + extinction
    
    return {
        'result': float(apparent_mag),
        'metadata': {
            'absolute_mag': absolute_mag,
            'distance_modulus': distance_modulus,
            'extinction': extinction,
            'components': {
                'M': absolute_mag,
                'mu': distance_modulus,
                'A_V': extinction
            },
            'formula': 'm = M + μ + A_V'
        }
    }


def check_observatory_visibility(
    dec_deg: float,
    observatory_lat: float,
    min_elevation: float = 0.0
) -> dict:
    """
    检查恒星在给定纬度天文台的理论可见性
    
    可见性判据（忽略望远镜指向限制）：
    - 北半球天文台：DEC > -(90° - lat)
    - 南半球天文台：DEC < (90° + lat)
    
    Args:
        dec_deg: 赤纬/度，范围 -90 到 +90
        observatory_lat: 天文台纬度/度，范围 -90 到 +90
        min_elevation: 最小仰角/度，默认0（地平线）
        
    Returns:
        dict: {
            'result': True/False (是否可见),
            'metadata': {
                'dec': 赤纬,
                'observatory_lat': 天文台纬度,
                'min_dec': 最小可见赤纬,
                'max_dec': 最大可见赤纬
            }
        }
        
    Example:
        >>> result = check_observatory_visibility(48.0, 19.8)
        >>> print(result['result'])
        True
    """
    # === 完整边界检查 ===
    if not isinstance(dec_deg, (int, float)):
        raise TypeError(f"赤纬必须是数值类型，当前类型: {type(dec_deg)}")
    
    if not (-90 <= dec_deg <= 90):
        raise ValueError(f"赤纬必须在-90到+90度之间，当前值: {dec_deg}")
    
    if not isinstance(observatory_lat, (int, float)):
        raise TypeError(f"天文台纬度必须是数值类型，当前类型: {type(observatory_lat)}")
    
    if not (-90 <= observatory_lat <= 90):
        raise ValueError(f"天文台纬度必须在-90到+90度之间，当前值: {observatory_lat}")
    
    if not isinstance(min_elevation, (int, float)):
        raise TypeError(f"最小仰角必须是数值类型，当前类型: {type(min_elevation)}")
    
    # === 核心计算 ===
    # 计算可见赤纬范围（简化模型：忽略大气折射和望远镜限制）
    # 最小可见赤纬：-(90° - |lat|) 对于北半球，(90° - |lat|) - 180° 对于南半球
    # 最大可见赤纬：+(90° - |lat|) 对于南半球，90° 对于北半球
    
    if observatory_lat >= 0:
        # 北半球天文台
        min_visible_dec = -(90.0 - observatory_lat) + min_elevation
        max_visible_dec = 90.0
    else:
        # 南半球天文台
        min_visible_dec = -90.0
        max_visible_dec = (90.0 + observatory_lat) - min_elevation
    
    is_visible = min_visible_dec <= dec_deg <= max_visible_dec
    
    return {
        'result': bool(is_visible),
        'metadata': {
            'dec': dec_deg,
            'observatory_lat': observatory_lat,
            'min_visible_dec': min_visible_dec,
            'max_visible_dec': max_visible_dec,
            'hemisphere': 'Northern' if observatory_lat >= 0 else 'Southern'
        }
    }


def check_spectrograph_compatibility(
    apparent_mag: float,
    espresso_limit: float = ESPRESSO_LIMIT,
    hires_limit: float = HIRES_LIMIT
) -> dict:
    """
    检查恒星与光谱仪的兼容性
    
    光谱仪灵敏度限制：
    - ESPRESSO (VLT): V < 17 mag
    - HIRES (Keck): V < 16 mag
    
    Args:
        apparent_mag: 视星等/mag
        espresso_limit: ESPRESSO限制/mag，默认17
        hires_limit: HIRES限制/mag，默认16
        
    Returns:
        dict: {
            'result': {
                'espresso': True/False,
                'hires': True/False,
                'both': True/False
            },
            'metadata': {
                'apparent_mag': 视星等,
                'limits': 各仪器限制,
                'margins': 与限制的差值
            }
        }
        
    Example:
        >>> result = check_spectrograph_compatibility(15.5)
        >>> print(result['result']['both'])
        True
    """
    # === 完整边界检查 ===
    if not isinstance(apparent_mag, (int, float)):
        raise TypeError(f"视星等必须是数值类型，当前类型: {type(apparent_mag)}")
    
    if apparent_mag < -30 or apparent_mag > 30:
        raise ValueError(f"视星等超出合理范围，当前值: {apparent_mag}")
    
    if not isinstance(espresso_limit, (int, float)):
        raise TypeError(f"ESPRESSO限制必须是数值类型，当前类型: {type(espresso_limit)}")
    
    if not isinstance(hires_limit, (int, float)):
        raise TypeError(f"HIRES限制必须是数值类型，当前类型: {type(hires_limit)}")
    
    # === 核心计算 ===
    # 星等越小越亮，所以用 < 判断
    espresso_ok = apparent_mag < espresso_limit
    hires_ok = apparent_mag < hires_limit
    both_ok = espresso_ok and hires_ok
    
    return {
        'result': {
            'espresso': bool(espresso_ok),
            'hires': bool(hires_ok),
            'both': bool(both_ok)
        },
        'metadata': {
            'apparent_mag': apparent_mag,
            'limits': {
                'espresso': espresso_limit,
                'hires': hires_limit
            },
            'margins': {
                'espresso': espresso_limit - apparent_mag,
                'hires': hires_limit - apparent_mag
            }
        }
    }


def fetch_star_data_from_simbad(identifier: str) -> dict:
    """
    从SIMBAD数据库获取恒星基础数据
    
    SIMBAD是免费的天文数据库，包含恒星的位置、星等等信息
    
    Args:
        identifier: 恒星标识符，如'Sirius'、'HD 48915'、'HIP 32349'
        
    Returns:
        dict: {
            'result': {
                'ra': 赤经/度,
                'dec': 赤纬/度,
                'vmag': V波段视星等,
                'name': 主要名称
            },
            'metadata': {'source': 'SIMBAD', 'query_time': 查询时间}
        }
        
    Example:
        >>> result = fetch_star_data_from_simbad('Sirius')
        >>> print(result['result']['vmag'])
    """
    # === 完整边界检查 ===
    if not isinstance(identifier, str):
        raise TypeError(f"标识符必须是字符串类型，当前类型: {type(identifier)}")
    
    if not identifier.strip():
        raise ValueError("标识符不能为空")
    
    try:
        # === 核心计算：查询SIMBAD ===
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('flux(V)')
        
        result_table = custom_simbad.query_object(identifier)
        
        if result_table is None or len(result_table) == 0:
            raise ValueError(f"在SIMBAD中未找到恒星'{identifier}'")
        
        # 提取数据
        row = result_table[0]
        coord = SkyCoord(row['RA'], row['DEC'], unit=(u.hourangle, u.deg))
        
        star_data = {
            'ra': float(coord.ra.degree),
            'dec': float(coord.dec.degree),
            'vmag': float(row['FLUX_V']) if row['FLUX_V'] else None,
            'name': str(row['MAIN_ID'])
        }
        
        return {
            'result': star_data,
            'metadata': {
                'source': 'SIMBAD',
                'query_time': datetime.now().isoformat(),
                'identifier': identifier
            }
        }
        
    except Exception as e:
        # 如果查询失败，返回错误信息
        return {
            'result': None,
            'metadata': {
                'source': 'SIMBAD',
                'error': str(e),
                'identifier': identifier
            }
        }


# ============ 第二层：组合工具函数（Composite Tools） ============

def analyze_star_observability(
    ra: Union[float, str],
    dec: float,
    absolute_mag: Optional[float] = None,
    apparent_mag: Optional[float] = None,
    distance_pc: Optional[float] = None,
    ebv: float = 0.0,
    star_name: str = "Unknown"
) -> dict:
    """
    综合分析恒星的可观测性（视星等计算 + 天文台可见性 + 光谱仪兼容性）
    
    该函数整合多个原子函数，完成完整的观测可行性评估：
    1. 坐标转换（如需要）
    2. 视星等计算（如未提供）
    3. 两个天文台的可见性检查
    4. 两个光谱仪的兼容性检查
    
    Args:
        ra: 赤经，可以是角度(float)或时角(str如'11h')
        dec: 赤纬/度，范围 -90 到 +90
        absolute_mag: 绝对星等/mag（如提供apparent_mag则可选）
        apparent_mag: 视星等/mag（如提供则跳过计算）
        distance_pc: 距离/pc（计算视星等时必需）
        ebv: 色余 E(B-V)/mag，默认0
        star_name: 恒星名称，用于输出
        
    Returns:
        dict: {
            'result': {
                'star_name': 恒星名称,
                'coordinates': {'ra_deg': 赤经, 'dec_deg': 赤纬},
                'apparent_mag': 视星等,
                'observable_both': True/False (两台天文台都可观测),
                'spectrograph_both': True/False (两个光谱仪都兼容),
                'final_verdict': True/False (最终可观测性)
            },
            'metadata': {详细的中间计算结果}
        }
        
    Example:
        >>> result = analyze_star_observability(
        ...     ra='11h', dec=48.0, absolute_mag=15.5, 
        ...     distance_pc=15.0, ebv=0.6, star_name='Star3'
        ... )
        >>> print(result['result']['final_verdict'])
    """
    # === 参数完全可序列化检查 ===
    if not isinstance(ra, (int, float, str)):
        raise TypeError("ra必须是int、float或str类型")
    if not isinstance(dec, (int, float)):
        raise TypeError("dec必须是数值类型")
    if absolute_mag is not None and not isinstance(absolute_mag, (int, float)):
        raise TypeError("absolute_mag必须是数值类型或None")
    if apparent_mag is not None and not isinstance(apparent_mag, (int, float)):
        raise TypeError("apparent_mag必须是数值类型或None")
    if distance_pc is not None and not isinstance(distance_pc, (int, float)):
        raise TypeError("distance_pc必须是数值类型或None")
    if not isinstance(ebv, (int, float)):
        raise TypeError("ebv必须是数值类型")
    if not isinstance(star_name, str):
        raise TypeError("star_name必须是字符串类型")
    
    # 检查必要参数
    if apparent_mag is None:
        if absolute_mag is None or distance_pc is None:
            raise ValueError("必须提供apparent_mag，或同时提供absolute_mag和distance_pc")
    
    metadata = {}
    
    # === 步骤1：坐标转换 ===
    ## using convert_ra_to_degrees, and get ra_deg
    ra_result = convert_ra_to_degrees(ra)
    ra_deg = ra_result['result']
    metadata['ra_conversion'] = ra_result['metadata']
    
    # === 步骤2：计算视星等（如未提供）===
    if apparent_mag is None:
        ## using calculate_extinction, and get extinction_v
        extinction_result = calculate_extinction(ebv)
        extinction_v = extinction_result['result']
        metadata['extinction'] = extinction_result['metadata']
        
        ## using calculate_apparent_magnitude, and get apparent_mag
        ## 该函数内部调用了 calculate_distance_modulus
        mag_result = calculate_apparent_magnitude(
            absolute_mag, distance_pc, extinction_v
        )
        apparent_mag = mag_result['result']
        metadata['magnitude_calculation'] = mag_result['metadata']
    else:
        metadata['magnitude_calculation'] = {'note': 'apparent_mag provided directly'}
    
    # === 步骤3：检查天文台可见性 ===
    ## using check_observatory_visibility for Paranal
    paranal_result = check_observatory_visibility(dec, PARANAL_LAT)
    paranal_visible = paranal_result['result']
    metadata['paranal_visibility'] = paranal_result['metadata']
    
    ## using check_observatory_visibility for Keck
    keck_result = check_observatory_visibility(dec, KECK_LAT)
    keck_visible = keck_result['result']
    metadata['keck_visibility'] = keck_result['metadata']
    
    observable_both = paranal_visible and keck_visible
    
    # === 步骤4：检查光谱仪兼容性 ===
    ## using check_spectrograph_compatibility
    spec_result = check_spectrograph_compatibility(apparent_mag)
    spectrograph_both = spec_result['result']['both']
    metadata['spectrograph_compatibility'] = spec_result['metadata']
    
    # === 最终判断 ===
    final_verdict = observable_both and spectrograph_both
    
    return {
        'result': {
            'star_name': star_name,
            'coordinates': {
                'ra_deg': ra_deg,
                'dec_deg': dec
            },
            'apparent_mag': apparent_mag,
            'observable_both': observable_both,
            'spectrograph_both': spectrograph_both,
            'final_verdict': final_verdict,
            'details': {
                'paranal_visible': paranal_visible,
                'keck_visible': keck_visible,
                'espresso_compatible': spec_result['result']['espresso'],
                'hires_compatible': spec_result['result']['hires']
            }
        },
        'metadata': metadata
    }


def batch_analyze_stars(stars_data: List[dict]) -> dict:
    """
    批量分析多颗恒星的可观测性
    
    该函数对多颗恒星执行完整的观测可行性评估，并汇总结果
    
    Args:
        stars_data: 恒星数据列表，每个元素是包含恒星参数的字典
            必需字段：'ra', 'dec'
            可选字段：'absolute_mag', 'apparent_mag', 'distance_pc', 'ebv', 'name'
            
    Returns:
        dict: {
            'result': {
                'total_stars': 总恒星数,
                'observable_count': 可观测恒星数,
                'observable_stars': [可观测恒星名称列表],
                'summary_table': 汇总表格数据
            },
            'metadata': {
                'analysis_time': 分析时间,
                'individual_results': [每颗恒星的详细结果]
            }
        }
        
    Example:
        >>> stars = [
        ...     {'ra': 15, 'dec': -75, 'absolute_mag': 15.5, 
        ...      'distance_pc': 10, 'name': 'Star1'},
        ...     {'ra': '11h', 'dec': 48, 'apparent_mag': 15.5, 'name': 'Star3'}
        ... ]
        >>> result = batch_analyze_stars(stars)
        >>> print(result['result']['observable_stars'])
    """
    # === 参数完全可序列化检查 ===
    if not isinstance(stars_data, list):
        raise TypeError("stars_data必须是列表类型")
    
    if len(stars_data) == 0:
        raise ValueError("stars_data不能为空")
    
    for i, star in enumerate(stars_data):
        if not isinstance(star, dict):
            raise TypeError(f"stars_data[{i}]必须是字典类型")
        if 'ra' not in star or 'dec' not in star:
            raise ValueError(f"stars_data[{i}]必须包含'ra'和'dec'字段")
    
    # === 核心计算：批量分析 ===
    individual_results = []
    observable_stars = []
    summary_table = []
    
    for i, star in enumerate(stars_data):
        star_name = star.get('name', f'Star{i+1}')
        
        try:
            ## using analyze_star_observability for each star
            ## 该函数内部调用了多个原子函数
            result = analyze_star_observability(
                ra=star['ra'],
                dec=star['dec'],
                absolute_mag=star.get('absolute_mag'),
                apparent_mag=star.get('apparent_mag'),
                distance_pc=star.get('distance_pc'),
                ebv=star.get('ebv', 0.0),
                star_name=star_name
            )
            
            individual_results.append(result)
            
            # 汇总可观测恒星
            if result['result']['final_verdict']:
                observable_stars.append(star_name)
            
            # 构建汇总表格
            summary_table.append({
                'name': star_name,
                'ra': result['result']['coordinates']['ra_deg'],
                'dec': result['result']['coordinates']['dec_deg'],
                'apparent_mag': result['result']['apparent_mag'],
                'observable': result['result']['final_verdict']
            })
            
        except Exception as e:
            # 记录错误但继续处理其他恒星
            individual_results.append({
                'result': None,
                'metadata': {'error': str(e), 'star_name': star_name}
            })
            summary_table.append({
                'name': star_name,
                'error': str(e)
            })
    
    return {
        'result': {
            'total_stars': len(stars_data),
            'observable_count': len(observable_stars),
            'observable_stars': observable_stars,
            'summary_table': summary_table
        },
        'metadata': {
            'analysis_time': datetime.now().isoformat(),
            'individual_results': individual_results
        }
    }


def compare_magnitude_systems(
    stars_data: List[dict],
    magnitude_bands: List[str] = ['V', 'B', 'R']
) -> dict:
    """
    比较不同星等系统下的恒星亮度
    
    该函数展示如何处理多波段星等数据，虽然当前问题只涉及V波段，
    但该工具可扩展到其他波段的分析
    
    Args:
        stars_data: 恒星数据列表，包含不同波段的星等
        magnitude_bands: 要比较的波段列表，默认['V', 'B', 'R']
        
    Returns:
        dict: {
            'result': {
                'comparison_table': 比较表格,
                'color_indices': 色指数计算结果
            },
            'metadata': {'bands': 使用的波段}
        }
    """
    # === 参数完全可序列化检查 ===
    if not isinstance(stars_data, list):
        raise TypeError("stars_data必须是列表类型")
    if not isinstance(magnitude_bands, list):
        raise TypeError("magnitude_bands必须是列表类型")
    
    comparison_table = []
    color_indices = []
    
    for star in stars_data:
        star_name = star.get('name', 'Unknown')
        row = {'name': star_name}
        
        # 提取各波段星等
        for band in magnitude_bands:
            mag_key = f'{band.lower()}_mag'
            row[band] = star.get(mag_key, None)
        
        comparison_table.append(row)
        
        # 计算色指数（如B-V）
        if 'B' in magnitude_bands and 'V' in magnitude_bands:
            b_mag = star.get('b_mag')
            v_mag = star.get('v_mag')
            if b_mag is not None and v_mag is not None:
                color_indices.append({
                    'name': star_name,
                    'B-V': b_mag - v_mag
                })
    
    return {
        'result': {
            'comparison_table': comparison_table,
            'color_indices': color_indices
        },
        'metadata': {
            'bands': magnitude_bands,
            'note': 'Color indices calculated where data available'
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def visualize_star_distribution(
    stars_data: List[dict],
    save_dir: str = './tool_images/',
    filename: str = None
) -> dict:
    """
    可视化恒星在天球上的分布
    
    生成赤道坐标系下的天球投影图，标注恒星位置和可观测性
    
    Args:
        stars_data: 恒星数据列表，包含'ra', 'dec', 'name', 'observable'等字段
        save_dir: 保存目录，默认'./tool_images/'
        filename: 文件名，默认自动生成
        
    Returns:
        dict: {
            'result': 保存的图片路径,
            'metadata': {'plot_type': 'sky_distribution', 'star_count': 恒星数}
        }
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    # === 参数检查 ===
    if not isinstance(stars_data, list):
        raise TypeError("stars_data必须是列表类型")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'star_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
    filepath = os.path.join(save_dir, filename)
    
    # === 创建图表 ===
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})
    
    # 提取数据
    observable_ra = []
    observable_dec = []
    unobservable_ra = []
    unobservable_dec = []
    
    for star in stars_data:
        ra = star.get('ra', 0)
        dec = star.get('dec', 0)
        observable = star.get('observable', False)
        
        # 转换为弧度（Mollweide投影需要）
        ra_rad = np.radians(ra - 180)  # 中心在180度
        dec_rad = np.radians(dec)
        
        if observable:
            observable_ra.append(ra_rad)
            observable_dec.append(dec_rad)
        else:
            unobservable_ra.append(ra_rad)
            unobservable_dec.append(dec_rad)
    
    # 绘制恒星
    if observable_ra:
        ax.scatter(observable_ra, observable_dec, c='green', s=100, 
                  marker='*', label='可观测', edgecolors='black', linewidths=0.5)
    if unobservable_ra:
        ax.scatter(unobservable_ra, unobservable_dec, c='red', s=100,
                  marker='x', label='不可观测', linewidths=2)
    
    # 标注恒星名称
    for star in stars_data:
        ra = star.get('ra', 0)
        dec = star.get('dec', 0)
        name = star.get('name', '')
        
        ra_rad = np.radians(ra - 180)
        dec_rad = np.radians(dec)
        
        ax.text(ra_rad, dec_rad, f'  {name}', fontsize=9, 
               verticalalignment='bottom')
    
    # 设置标签和标题
    ax.set_xlabel('赤经 (RA)', fontsize=12)
    ax.set_ylabel('赤纬 (DEC)', fontsize=12)
    ax.set_title('恒星天球分布图\n(ESPRESSO + HIRES 可观测性分析)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Sky Distribution Plot | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'plot_type': 'sky_distribution',
            'star_count': len(stars_data),
            'observable_count': len(observable_ra),
            'projection': 'mollweide'
        }
    }


def visualize_magnitude_comparison(
    stars_data: List[dict],
    save_dir: str = './tool_images/',
    filename: str = None
) -> dict:
    """
    可视化恒星视星等与观测限制的比较
    
    生成柱状图，展示各恒星的视星等与ESPRESSO/HIRES限制的关系
    
    Args:
        stars_data: 恒星数据列表，包含'name', 'apparent_mag', 'observable'等字段
        save_dir: 保存目录
        filename: 文件名
        
    Returns:
        dict: {
            'result': 保存的图片路径,
            'metadata': {'plot_type': 'magnitude_comparison'}
        }
    """
    import matplotlib.pyplot as plt
    
    # === 参数检查 ===
    if not isinstance(stars_data, list):
        raise TypeError("stars_data必须是列表类型")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'magnitude_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
    filepath = os.path.join(save_dir, filename)
    
    # === 创建图表 ===
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据
    names = [star.get('name', f'Star{i+1}') for i, star in enumerate(stars_data)]
    mags = [star.get('apparent_mag', 0) for star in stars_data]
    observable = [star.get('observable', False) for star in stars_data]
    
    # 设置颜色
    colors = ['green' if obs else 'red' for obs in observable]
    
    # 绘制柱状图
    bars = ax.bar(names, mags, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加限制线
    ax.axhline(y=HIRES_LIMIT, color='blue', linestyle='--', linewidth=2, 
              label=f'HIRES限制 (V < {HIRES_LIMIT} mag)')
    ax.axhline(y=ESPRESSO_LIMIT, color='purple', linestyle='--', linewidth=2,
              label=f'ESPRESSO限制 (V < {ESPRESSO_LIMIT} mag)')
    
    # 在柱子上标注数值
    for i, (bar, mag) in enumerate(zip(bars, mags)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mag:.2f}',
               ha='center', va='bottom', fontsize=9)
    
    # 设置标签和标题
    ax.set_xlabel('恒星', fontsize=12)
    ax.set_ylabel('视星等 (mag)', fontsize=12)
    ax.set_title('恒星视星等与光谱仪限制比较\n(星等越小越亮)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 反转y轴（星等小的在上方）
    ax.invert_yaxis()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Magnitude Comparison Plot | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'plot_type': 'magnitude_comparison',
            'star_count': len(stars_data)
        }
    }


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【当前问题】+【至少2个相关场景】
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - 筛选ESPRESSO和HIRES都能观测的恒星")
    print("=" * 60)
    print("问题描述：给定5颗恒星的参数，判断哪些恒星可以被Paranal的ESPRESSO")
    print("          和Keck的HIRES光谱仪同时观测")
    print("-" * 60)
    
    # 定义5颗恒星的数据
    stars_problem = [
        {
            'name': 'Star1',
            'ra': 15,  # 度
            'dec': -75,  # 度
            'absolute_mag': 15.5,
            'distance_pc': 10,
            'ebv': 0.0
        },
        {
            'name': 'Star2',
            'ra': 30,  # 度
            'dec': 55,  # 度
            'apparent_mag': 16.5,  # 直接给定视星等
            'distance_pc': 5,
            'ebv': 0.0
        },
        {
            'name': 'Star3',
            'ra': '11h',  # 时角格式
            'dec': 48,  # 度
            'apparent_mag': 15.5,  # 直接给定视星等
            'distance_pc': 15,
            'ebv': 0.6  # 有消光
        },
        {
            'name': 'Star4',
            'ra': 85,  # 度
            'dec': -48,  # 度
            'absolute_mag': 15.5,
            'distance_pc': 10,
            'ebv': 0.4  # 有消光
        },
        {
            'name': 'Star5',
            'ra': '10h',  # 时角格式
            'dec': 60,  # 度
            'absolute_mag': 16.5,
            'distance_pc': 5,
            'ebv': 0.0
        }
    ]
    
    # 步骤1：批量分析所有恒星
    # 调用函数：batch_analyze_stars()，该函数内部调用了 analyze_star_observability()
    # analyze_star_observability() 又调用了多个原子函数
    print("\n步骤1：批量分析5颗恒星的可观测性...")
    result1 = batch_analyze_stars(stars_problem)
    print(f"FUNCTION_CALL: batch_analyze_stars | PARAMS: 5 stars | RESULT: {result1['result']['observable_count']} observable")
    
    print(f"\n分析结果：")
    print(f"  总恒星数: {result1['result']['total_stars']}")
    print(f"  可观测恒星数: {result1['result']['observable_count']}")
    print(f"  可观测恒星: {', '.join(result1['result']['observable_stars'])}")
    
    # 步骤2：详细输出每颗恒星的分析结果
    print("\n步骤2：详细分析每颗恒星...")
    for i, star_result in enumerate(result1['metadata']['individual_results']):
        if star_result['result'] is not None:
            res = star_result['result']
            print(f"\n{res['star_name']}:")
            print(f"  坐标: RA={res['coordinates']['ra_deg']:.2f}°, DEC={res['coordinates']['dec_deg']:.2f}°")
            print(f"  视星等: {res['apparent_mag']:.2f} mag")
            print(f"  Paranal可见: {res['details']['paranal_visible']}")
            print(f"  Keck可见: {res['details']['keck_visible']}")
            print(f"  ESPRESSO兼容: {res['details']['espresso_compatible']}")
            print(f"  HIRES兼容: {res['details']['hires_compatible']}")
            print(f"  最终判定: {'可观测' if res['final_verdict'] else '不可观测'}")
    
    # 步骤3：可视化结果
    # 调用函数：visualize_star_distribution()
    print("\n步骤3：生成天球分布图...")
    vis_data = []
    for star_result in result1['metadata']['individual_results']:
        if star_result['result'] is not None:
            res = star_result['result']
            vis_data.append({
                'name': res['star_name'],
                'ra': res['coordinates']['ra_deg'],
                'dec': res['coordinates']['dec_deg'],
                'observable': res['final_verdict']
            })
    
    vis_result1 = visualize_star_distribution(vis_data, filename='scenario1_sky_distribution.png')
    print(f"FUNCTION_CALL: visualize_star_distribution | PARAMS: {len(vis_data)} stars | RESULT: {vis_result1['result']}")
    
    # 调用函数：visualize_magnitude_comparison()
    print("\n步骤4：生成星等比较图...")
    mag_data = []
    for star_result in result1['metadata']['individual_results']:
        if star_result['result'] is not None:
            res = star_result['result']
            mag_data.append({
                'name': res['star_name'],
                'apparent_mag': res['apparent_mag'],
                'observable': res['final_verdict']
            })
    
    vis_result2 = visualize_magnitude_comparison(mag_data, filename='scenario1_magnitude_comparison.png')
    print(f"FUNCTION_CALL: visualize_magnitude_comparison | PARAMS: {len(mag_data)} stars | RESULT: {vis_result2['result']}")
    
    print(f"\n场景1完成：找到 {len(result1['result']['observable_stars'])} 颗可观测恒星")
    print(f"FINAL_ANSWER: {', '.join(result1['result']['observable_stars'])}")
    
    
    print("\n" + "=" * 60)
    print("场景2：参数扫描 - 分析不同距离下的可观测性")
    print("=" * 60)
    print("问题描述：固定恒星的绝对星等和位置，扫描不同距离，")
    print("          分析在什么距离范围内恒星仍可被观测")
    print("-" * 60)
    
    # 步骤1：定义参考恒星
    print("\n步骤1：定义参考恒星（类似Star3的参数）...")
    reference_star = {
        'ra': '11h',
        'dec': 48,
        'absolute_mag': 15.5,
        'ebv': 0.0,  # 简化：不考虑消光
        'name': 'Reference Star'
    }
    
    # 步骤2：扫描距离范围
    # 调用函数：analyze_star_observability() 在循环中
    print("\n步骤2：扫描距离范围 1-50 pc...")
    distances = np.linspace(1, 50, 20)
    scan_results = []
    
    for dist in distances:
        # 调用函数：analyze_star_observability()
        result = analyze_star_observability(
            ra=reference_star['ra'],
            dec=reference_star['dec'],
            absolute_mag=reference_star['absolute_mag'],
            distance_pc=dist,
            ebv=reference_star['ebv'],
            star_name=f'@{dist:.1f}pc'
        )
        scan_results.append({
            'distance': dist,
            'apparent_mag': result['result']['apparent_mag'],
            'observable': result['result']['final_verdict']
        })
    
    print(f"FUNCTION_CALL: analyze_star_observability (loop) | PARAMS: 20 distances | RESULT: scan completed")
    
    # 步骤3：找到临界距离
    print("\n步骤3：分析临界距离...")
    observable_distances = [r['distance'] for r in scan_results if r['observable']]
    if observable_distances:
        max_distance = max(observable_distances)
        print(f"  最大可观测距离: {max_distance:.1f} pc")
        print(f"  对应视星等: {[r['apparent_mag'] for r in scan_results if r['distance'] == max_distance][0]:.2f} mag")
    else:
        print("  在扫描范围内无可观测距离")
    
    # 步骤4：可视化距离-星等关系
    print("\n步骤4：生成距离-星等关系图...")
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dists = [r['distance'] for r in scan_results]
    mags = [r['apparent_mag'] for r in scan_results]
    obs = [r['observable'] for r in scan_results]
    
    colors = ['green' if o else 'red' for o in obs]
    ax.scatter(dists, mags, c=colors, s=50, alpha=0.7)
    ax.plot(dists, mags, 'k--', alpha=0.3)
    
    ax.axhline(y=HIRES_LIMIT, color='blue', linestyle='--', linewidth=2,
              label=f'HIRES限制 ({HIRES_LIMIT} mag)')
    ax.axhline(y=ESPRESSO_LIMIT, color='purple', linestyle='--', linewidth=2,
              label=f'ESPRESSO限制 ({ESPRESSO_LIMIT} mag)')
    
    ax.set_xlabel('距离 (pc)', fontsize=12)
    ax.set_ylabel('视星等 (mag)', fontsize=12)
    ax.set_title('恒星距离与可观测性关系\n(绝对星等 M=15.5 mag, DEC=48°)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    filepath2 = './tool_images/scenario2_distance_scan.png'
    plt.tight_layout()
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Distance Scan Plot | PATH: {filepath2}")
    print(f"FUNCTION_CALL: matplotlib.pyplot.savefig | PARAMS: distance_scan | RESULT: {filepath2}")
    
    print(f"\n场景2完成：临界距离为 {max_distance:.1f} pc")
    
    
    print("\n" + "=" * 60)
    print("场景3：数据库集成 - 查询真实恒星数据")
    print("=" * 60)
    print("问题描述：从SIMBAD数据库查询著名恒星的数据，")
    print("          分析它们的可观测性")
    print("-" * 60)
    
    # 步骤1：定义要查询的恒星
    print("\n步骤1：定义要查询的著名恒星...")
    famous_stars = ['Sirius', 'Vega', 'Betelgeuse', 'Rigel', 'Altair']
    print(f"  目标恒星: {', '.join(famous_stars)}")
    
    # 步骤2：从SIMBAD查询数据
    # 调用函数：fetch_star_data_from_simbad()
    print("\n步骤2：从SIMBAD数据库查询...")
    simbad_results = []
    for star_name in famous_stars:
        print(f"  查询 {star_name}...")
        # 调用函数：fetch_star_data_from_simbad()
        result = fetch_star_data_from_simbad(star_name)
        if result['result'] is not None:
            simbad_results.append({
                'name': star_name,
                'ra': result['result']['ra'],
                'dec': result['result']['dec'],
                'apparent_mag': result['result']['vmag']
            })
            print(f"   成功: RA={result['result']['ra']:.2f}°, DEC={result['result']['dec']:.2f}°, V={result['result']['vmag']:.2f}")
        else:
            print(f"   查询失败: {result['metadata'].get('error', 'Unknown error')}")
    
    print(f"FUNCTION_CALL: fetch_star_data_from_simbad (loop) | PARAMS: {len(famous_stars)} stars | RESULT: {len(simbad_results)} successful")
    
    # 步骤3：分析查询到的恒星
    # 调用函数：batch_analyze_stars()
    if simbad_results:
        print("\n步骤3：分析查询到的恒星...")
        result3 = batch_analyze_stars(simbad_results)
        print(f"FUNCTION_CALL: batch_analyze_stars | PARAMS: {len(simbad_results)} stars | RESULT: {result3['result']['observable_count']} observable")
        
        print(f"\n分析结果：")
        print(f"  查询成功: {len(simbad_results)} 颗")
        print(f"  可观测: {result3['result']['observable_count']} 颗")
        if result3['result']['observable_stars']:
            print(f"  可观测恒星: {', '.join(result3['result']['observable_stars'])}")
        
        # 步骤4：可视化
        print("\n步骤4：生成可视化图表...")
        vis_data3 = []
        for star_result in result3['metadata']['individual_results']:
            if star_result['result'] is not None:
                res = star_result['result']
                vis_data3.append({
                    'name': res['star_name'],
                    'ra': res['coordinates']['ra_deg'],
                    'dec': res['coordinates']['dec_deg'],
                    'apparent_mag': res['apparent_mag'],
                    'observable': res['final_verdict']
                })
        
        vis_result3 = visualize_star_distribution(vis_data3, filename='scenario3_famous_stars.png')
        print(f"FUNCTION_CALL: visualize_star_distribution | PARAMS: {len(vis_data3)} stars | RESULT: {vis_result3['result']}")
        
        print(f"\n场景3完成：从SIMBAD查询并分析了 {len(simbad_results)} 颗著名恒星")
    else:
        print("\n场景3失败：未能从SIMBAD查询到任何恒星数据")
        print("  （可能是网络问题或SIMBAD服务不可用）")
    
    
    print("\n" + "=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始问题的完整流程（5颗恒星筛选）")
    print("- 场景2展示了工具的参数泛化能力（距离扫描分析）")
    print("- 场景3展示了工具与数据库的集成能力（SIMBAD查询）")
    print("\n生成的文件：")
    print("  - ./tool_images/scenario1_sky_distribution.png")
    print("  - ./tool_images/scenario1_magnitude_comparison.png")
    print("  - ./tool_images/scenario2_distance_scan.png")
    if simbad_results:
        print("  - ./tool_images/scenario3_famous_stars.png")


if __name__ == "__main__":
    main()