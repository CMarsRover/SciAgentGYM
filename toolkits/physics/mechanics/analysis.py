#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析工具模块
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.optimize import curve_fit

def analyze_solution(solution: Dict[str, np.ndarray],
                    analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    分析变分问题的解
    
    Args:
        solution: 解的数据
        analysis_type: 分析类型
        
    Returns:
        分析结果
    """
    analysis = {}
    
    if analysis_type in ["comprehensive", "basic"]:
        analysis.update(_basic_analysis(solution))
    
    if analysis_type in ["comprehensive", "energy"]:
        analysis.update(_energy_analysis(solution))
    
    if analysis_type in ["comprehensive", "stability"]:
        analysis.update(_stability_analysis(solution))
    
    if analysis_type in ["comprehensive", "statistical"]:
        analysis.update(_statistical_analysis(solution))
    
    return analysis

def _basic_analysis(solution: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """基本分析"""
    t = solution['t']
    
    analysis = {
        'time_span': (t[0], t[-1]),
        'num_points': len(t),
        'time_step': t[1] - t[0] if len(t) > 1 else 0
    }
    
    # 分析位置
    if 'x' in solution:
        x = solution['x']
        analysis['position'] = {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.mean(x),
            'std': np.std(x),
            'amplitude': np.max(x) - np.min(x)
        }
    
    # 分析速度
    if 'x_dot' in solution:
        x_dot = solution['x_dot']
        analysis['velocity'] = {
            'min': np.min(x_dot),
            'max': np.max(x_dot),
            'mean': np.mean(x_dot),
            'std': np.std(x_dot),
            'max_speed': np.max(np.abs(x_dot))
        }
    
    return analysis

def _energy_analysis(solution: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """能量分析"""
    analysis = {}
    
    if 'energy' in solution:
        energy = solution['energy']
        analysis['energy'] = {
            'initial': energy[0],
            'final': energy[-1],
            'mean': np.mean(energy),
            'std': np.std(energy),
            'max': np.max(energy),
            'min': np.min(energy),
            'variation': np.max(energy) - np.min(energy),
            'conservation_error': np.std(energy) / np.mean(energy) if np.mean(energy) != 0 else 0
        }
    
    return analysis

def _stability_analysis(solution: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """稳定性分析"""
    analysis = {}
    
    if 'x' in solution and 'x_dot' in solution:
        x = solution['x']
        x_dot = solution['x_dot']
        
        # 计算李雅普诺夫指数（简化版本）
        # 对于线性系统，可以通过相图分析稳定性
        if len(x) > 1:
            # 计算位移的标准差
            x_std = np.std(x)
            x_dot_std = np.std(x_dot)
            
            # 判断是否发散
            is_bounded = x_std < 10 * np.mean(np.abs(x)) and x_dot_std < 10 * np.mean(np.abs(x_dot))
            
            analysis['stability'] = {
                'is_bounded': is_bounded,
                'position_std': x_std,
                'velocity_std': x_dot_std,
                'stability_indicator': x_std + x_dot_std
            }
    
    return analysis

def _statistical_analysis(solution: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """统计分析"""
    analysis = {}
    
    if 'x' in solution:
        x = solution['x']
        
        # 基本统计量
        analysis['statistics'] = {
            'skewness': stats.skew(x),
            'kurtosis': stats.kurtosis(x),
            'percentiles': {
                '25': np.percentile(x, 25),
                '50': np.percentile(x, 50),
                '75': np.percentile(x, 75)
            }
        }
        
        # 周期性检测
        if len(x) > 10:
            # 使用自相关函数检测周期性
            autocorr = np.correlate(x, x, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 寻找第一个峰值（除了零延迟）
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            
            if peaks:
                estimated_period = peaks[0]
                analysis['periodicity'] = {
                    'estimated_period': estimated_period,
                    'has_periodicity': len(peaks) > 1
                }
    
    return analysis

def compute_energy(trajectory: Dict[str, np.ndarray],
                  lagrangian_func: Optional[callable] = None) -> np.ndarray:
    """
    计算能量
    
    Args:
        trajectory: 轨迹数据
        lagrangian_func: 拉格朗日函数
        
    Returns:
        能量时间序列
    """
    t = trajectory['t']
    
    if lagrangian_func is None:
        # 默认能量函数（谐振子）
        def lagrangian_func(t, x, x_dot):
            return 0.5 * x_dot**2 - 0.5 * x**2
    
    energy = []
    for i in range(len(t)):
        if 'x' in trajectory and 'x_dot' in trajectory:
            e = lagrangian_func(t[i], trajectory['x'][i], trajectory['x_dot'][i])
            energy.append(e)
        else:
            energy.append(0.0)
    
    return np.array(energy)

def fit_oscillation(trajectory: Dict[str, np.ndarray],
                   coord: str = 'x') -> Dict[str, Any]:
    """
    拟合振荡
    
    Args:
        trajectory: 轨迹数据
        coord: 坐标名称
        
    Returns:
        拟合结果
    """
    if coord not in trajectory:
        return {}
    
    t = trajectory['t']
    x = trajectory[coord]
    
    # 定义拟合函数：A * cos(ωt + φ) + C
    def oscillation_func(t, A, omega, phi, C):
        return A * np.cos(omega * t + phi) + C
    
    try:
        # 初始猜测
        A_guess = (np.max(x) - np.min(x)) / 2
        omega_guess = 2 * np.pi / (t[-1] - t[0])  # 假设一个周期
        phi_guess = 0
        C_guess = np.mean(x)
        
        # 拟合
        popt, pcov = curve_fit(oscillation_func, t, x, 
                              p0=[A_guess, omega_guess, phi_guess, C_guess])
        
        # 计算拟合质量
        fitted_values = oscillation_func(t, *popt)
        residuals = x - fitted_values
        r_squared = 1 - np.sum(residuals**2) / np.sum((x - np.mean(x))**2)
        
        return {
            'amplitude': popt[0],
            'frequency': popt[1] / (2 * np.pi),
            'phase': popt[2],
            'offset': popt[3],
            'r_squared': r_squared,
            'fitted_values': fitted_values,
            'residuals': residuals
        }
    
    except:
        return {}

def detect_equilibrium_points(trajectory: Dict[str, np.ndarray],
                            tolerance: float = 1e-3) -> List[float]:
    """
    检测平衡点
    
    Args:
        trajectory: 轨迹数据
        tolerance: 容差
        
    Returns:
        平衡点列表
    """
    equilibrium_points = []
    
    if 'x' in trajectory and 'x_dot' in trajectory:
        x = trajectory['x']
        x_dot = trajectory['x_dot']
        
        # 寻找速度接近零的点
        zero_velocity_indices = np.where(np.abs(x_dot) < tolerance)[0]
        
        for idx in zero_velocity_indices:
            if 0 < idx < len(x) - 1:
                # 检查是否为局部极值
                if (x[idx] > x[idx-1] and x[idx] > x[idx+1]) or \
                   (x[idx] < x[idx-1] and x[idx] < x[idx+1]):
                    equilibrium_points.append(x[idx])
    
    return list(set(equilibrium_points))  # 去重

def compute_phase_space_volume(trajectory: Dict[str, np.ndarray],
                              time_window: Optional[Tuple[int, int]] = None) -> float:
    """
    计算相空间体积
    
    Args:
        trajectory: 轨迹数据
        time_window: 时间窗口
        
    Returns:
        相空间体积
    """
    if 'x' not in trajectory or 'x_dot' not in trajectory:
        return 0.0
    
    x = trajectory['x']
    x_dot = trajectory['x_dot']
    
    if time_window is not None:
        start, end = time_window
        x = x[start:end]
        x_dot = x_dot[start:end]
    
    # 计算相空间中的范围
    x_range = np.max(x) - np.min(x)
    x_dot_range = np.max(x_dot) - np.min(x_dot)
    
    # 相空间体积（2D情况下的面积）
    volume = x_range * x_dot_range
    
    return volume
