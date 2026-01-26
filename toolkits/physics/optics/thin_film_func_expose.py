#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄膜干涉计算器函数暴露模块
将 EnhancedThinFilmInterferenceCalculator 类的所有方法暴露为顶层函数
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
try:
    # 作为脚本直接运行当前目录测试时
    from thin_film_interference import EnhancedThinFilmInterferenceCalculator
except ImportError:
    try:
        # 包内相对导入（作为包使用时）
        from .thin_film_interference import EnhancedThinFilmInterferenceCalculator
    except ImportError:
        # 绝对导入（从项目根目录）
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from tools.ThinFilmInterference.thin_film_interference import EnhancedThinFilmInterferenceCalculator

# 创建一个默认的计算器实例（单例模式）
_default_calculator = None

def _get_calculator(n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> EnhancedThinFilmInterferenceCalculator:
    """获取或创建计算器实例"""
    global _default_calculator
    if _default_calculator is None or \
       _default_calculator.n_air != n_air or \
       _default_calculator.n_film != n_film or \
       _default_calculator.n_substrate != n_substrate:
        _default_calculator = EnhancedThinFilmInterferenceCalculator(n_air, n_film, n_substrate)
    return _default_calculator

def _deg_to_rad(angle_deg: float) -> float:
    """角度转换：度转弧度"""
    return np.radians(angle_deg)

def _nm_to_m(wavelength_nm: float) -> float:
    """单位转换：纳米转米"""
    return wavelength_nm * 1e-9

def _m_to_nm(length_m: float) -> float:
    """单位转换：米转纳米"""
    return length_m * 1e9

# ========================
# 1. 核心物理计算模块
# ========================

def optical_path_difference(thickness: float, wavelength: float, angle: float = 0, 
                              n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    计算光程差（支持斜入射）
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率，通常为空气=1.0
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 光程差，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = _nm_to_m(thickness)
    opd_m = calculator.optical_path_difference(thickness_m, theta_rad, wavelength_m)
    return _m_to_nm(opd_m)

def interference_condition(thickness: float, wavelength: float, angle: float = 0, 
                           n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> Dict[str, Union[str, float]]:
    """
    计算干涉条件（增强版）
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        Dict[str, Union[str, float]]: 包含以下键的字典：
            - "opd" (float): 光程差，单位：纳米 (nm)
            - "phase_diff" (float): 相位差，单位：弧度
            - "interference" (str): 干涉类型 ("constructive"/"destructive"/"partial"/"total_reflection")
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = _nm_to_m(thickness)
    result = calculator.interference_condition(thickness_m, theta_rad, wavelength_m)
    
    # 转换光程差单位为nm
    if not np.isnan(result["opd"]):
        result["opd"] = _m_to_nm(result["opd"])
    
    return result

# ========================
# 2. 厚度计算模块
# ========================

def min_thickness_for_constructive_reflection(wavelength: float, angle: float = 0, 
                                               n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    计算反射光相长干涉的最小厚度（考虑半波损失）
    
    Args:
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 最小厚度，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = calculator.min_thickness_for_constructive_reflection(wavelength_m, theta_rad)
    return _m_to_nm(thickness_m)

def min_thickness_for_constructive_transmission(wavelength: float, angle: float = 0, 
                                                n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    计算透射光相长干涉的最小厚度
    
    Args:
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 最小厚度，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = calculator.min_thickness_for_constructive_transmission(wavelength_m, theta_rad)
    return _m_to_nm(thickness_m)

def min_antireflection_thickness(wavelength: float, n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    计算最小增透厚度（λ/4n）
    
    Args:
        wavelength (float): 波长，单位：纳米 (nm)
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 最小增透厚度，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = calculator.min_antireflection_thickness(wavelength_m)
    return _m_to_nm(thickness_m)

def wavelength_for_constructive(thickness: float, order: int = 1, angle: float = 0, 
                               n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    计算相长干涉的波长
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        order (int, optional): 干涉级次，默认为1
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 相长干涉波长，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    thickness_m = _nm_to_m(thickness)
    wavelength_m = calculator.wavelength_for_constructive(thickness_m, order, theta_rad)
    return _m_to_nm(wavelength_m)

# ========================
# 3. 干涉判断模块
# ========================

def is_constructive(thickness: float, wavelength: float, angle: float = 0, 
                  n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> bool:
    """
    判断是否为相长干涉
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        bool: 是否为相长干涉
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = _nm_to_m(thickness)
    return calculator.is_constructive(thickness_m, theta_rad, wavelength_m)

def is_destructive(thickness: float, wavelength: float, angle: float = 0, 
                 n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> bool:
    """
    判断是否为相消干涉
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        bool: 是否为相消干涉
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = _nm_to_m(thickness)
    return calculator.is_destructive(thickness_m, theta_rad, wavelength_m)

# ========================
# 4. 光谱分析模块
# ========================

def reflectance_spectrum(thickness: float, wavelength_range: Tuple[float, float] = (400, 700), 
                        angle: float = 0, n_air: float = 1.0, n_film: float = 1.5, 
                        n_substrate: float = 1.0, num_points: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算反射光谱
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        wavelength_range (Tuple[float, float], optional): 波长范围，单位：纳米 (nm)，默认(400, 700)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
        num_points (int, optional): 计算点数，默认300
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (wavelengths, reflectance) 波长数组和反射率数组
            - wavelengths: 波长数组，单位：纳米 (nm)
            - reflectance: 反射率数组，范围[0,1]
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    thickness_m = _nm_to_m(thickness)
    
    wavelengths_nm = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    wavelengths_m = _nm_to_m(wavelengths_nm)
    
    R = calculator.reflectance_spectrum(thickness_m, theta_rad, wavelengths_m)
    return wavelengths_nm, R

def predict_color(thickness: float, angle: float = 0, n_air: float = 1.0, 
                n_film: float = 1.5, n_substrate: float = 1.0) -> Dict[str, Union[float, List[int], np.ndarray]]:
    """
    预测薄膜颜色
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        Dict[str, Union[float, List[int], np.ndarray]]: 包含以下键的字典：
            - "peak_wavelength_nm" (float): 峰值波长，单位：纳米 (nm)
            - "rgb" (List[int]): RGB颜色值，范围[0,255]
            - "reflectance" (np.ndarray): 反射光谱数组
            - "wavelengths_nm" (np.ndarray): 对应的波长数组，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    thickness_m = _nm_to_m(thickness)
    result = calculator.predict_color(thickness_m, theta_rad)
    
    # 转换反射光谱的波长单位为nm
    if 'reflectance' in result:
        wavelengths_nm = np.linspace(400, 700, len(result['reflectance']))
        result['wavelengths_nm'] = wavelengths_nm
    
    return result

# ========================
# 5. 符号计算模块
# ========================

def solve_thickness_symbolically(wavelength: float, interference: str = "destructive", 
                                 n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    使用符号计算求解厚度
    
    Args:
        wavelength (float): 波长，单位：纳米 (nm)
        interference (str, optional): 干涉类型，可选"constructive"或"destructive"，默认"destructive"
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 厚度值，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    wavelength_m = _nm_to_m(wavelength)
    thickness_m = calculator.solve_thickness_symbolically(wavelength_m, interference)
    return _m_to_nm(thickness_m)

def find_optimal_thickness(target_wavelength: float, angle: float = 0, 
                        n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> float:
    """
    数值优化求解最优厚度
    
    Args:
        target_wavelength (float): 目标波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        float: 最优厚度，单位：纳米 (nm)
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(target_wavelength)
    thickness_m = calculator.find_optimal_thickness(wavelength_m, theta_rad)
    return _m_to_nm(thickness_m)

# ========================
# 6. 综合求解模块
# ========================

def solve_complete_problem(wavelength: float, angle: float = 0, n_air: float = 1.0, 
                        n_film: float = 1.5, n_substrate: float = 1.0) -> Dict[str, Any]:
    """
    综合求解薄膜干涉问题
    
    Args:
        wavelength (float): 波长，单位：纳米 (nm)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        Dict[str, Any]: 完整的分析结果，包含以下键：
            - "wavelength" (float): 输入波长，单位：纳米 (nm)
            - "theta_degrees" (float): 入射角，单位：度
            - "n_air", "n_film", "n_substrate" (float): 各层折射率
            - "thicknesses" (Dict[str, float]): 各种厚度值，单位：纳米 (nm)
            - "interference_condition" (Dict): 干涉条件信息
            - "color_prediction" (Dict): 颜色预测信息
            - "status" (str): 计算状态 ("success"/"error")
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    result = calculator.solve_complete_problem(wavelength_m, theta_rad)
    
    # 转换厚度单位为nm
    if result["status"] == "success" and "thicknesses" in result:
        for key in result["thicknesses"]:
            result["thicknesses"][key] = _m_to_nm(result["thicknesses"][key])
    
    return result

# ========================
# 7. 可视化模块
# ========================

def plot_intensity_vs_thickness(wavelength: float, max_thickness: float = 1000, angle: float = 0, 
                                n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0, 
                                num_points: int = 500) -> None:
    """
    绘制强度随厚度的变化
    
    Args:
        wavelength (float): 波长，单位：纳米 (nm)
        max_thickness (float, optional): 最大厚度，单位：纳米 (nm)，默认1000
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
        num_points (int, optional): 计算点数，默认500
    
    Returns:
        None: 直接显示图形
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    wavelength_m = _nm_to_m(wavelength)
    max_thickness_m = _nm_to_m(max_thickness)
    
    calculator.plot_intensity_vs_thickness(wavelength_m, theta_rad, max_thickness_m, num_points)

def plot_reflectance_spectrum(thickness: float, wavelength_range: Tuple[float, float] = (400, 700), 
                             angle: float = 0, n_air: float = 1.0, n_film: float = 1.5, 
                             n_substrate: float = 1.0, title: str = "反射光谱") -> None:
    """
    绘制反射光谱
    
    Args:
        thickness (float): 薄膜厚度，单位：纳米 (nm)
        wavelength_range (Tuple[float, float], optional): 波长范围，单位：纳米 (nm)，默认(400, 700)
        angle (float, optional): 入射角，单位：度，默认0为垂直入射
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
        title (str, optional): 图表标题，默认"反射光谱"
    
    Returns:
        None: 直接显示图形
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    theta_rad = _deg_to_rad(angle)
    thickness_m = _nm_to_m(thickness)
    
    calculator.plot_reflectance_spectrum(thickness_m, theta_rad, title)

def plot_thickness_wavelength_heatmap(thickness_range: Tuple[float, float] = (0, 1000), 
                                     wavelength_range: Tuple[float, float] = (400, 700),
                                     n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0) -> None:
    """
    绘制厚度-波长热图
    
    Args:
        thickness_range (Tuple[float, float], optional): 厚度范围，单位：纳米 (nm)，默认(0, 1000)
        wavelength_range (Tuple[float, float], optional): 波长范围，单位：纳米 (nm)，默认(400, 700)
        n_air (float, optional): 入射介质折射率
        n_film (float, optional): 薄膜折射率
        n_substrate (float, optional): 基底折射率
    
    Returns:
        None: 直接显示图形
    """
    calculator = _get_calculator(n_air, n_film, n_substrate)
    
    # 转换单位为米
    d_range_m = (_nm_to_m(thickness_range[0]), _nm_to_m(thickness_range[1]))
    wl_range_m = (_nm_to_m(wavelength_range[0]), _nm_to_m(wavelength_range[1]))
    
    calculator.plot_thickness_wavelength_heatmap(d_range_m, wl_range_m)

