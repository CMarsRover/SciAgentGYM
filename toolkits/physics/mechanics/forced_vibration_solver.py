#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
强迫振动系统求解工具

该模块提供用于分析和求解机械振动系统（如示功计划针系统）的函数：
- 计算自然频率
- 计算受迫振动稳态振幅
- 数值求解二阶系统的时域响应
- 绘制振动响应与示功图
- 计算蒸汽机示功计划针振幅的便捷接口

依赖：numpy, matplotlib, scipy
"""

import os
import sys
import argparse
from typing import Callable, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 确保运行路径下可导入项目模块（避免相对/绝对路径问题）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def solve_forced_vibration(k: float, m: float, force_amplitude: float, 
                          driving_frequency: float, damping: float = 0.0,
                          initial_conditions: Tuple[float, float] = (0.0, 0.0),
                          time_span: Tuple[float, float] = (0.0, 10.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解受迫振动系统
    
    Parameters:
    -----------
    k : float
        弹簧刚度系数
    m : float
        质量
    force_amplitude : float
        外力幅值
    driving_frequency : float
        驱动频率
    damping : float, optional
        阻尼系数，默认为0.0
    initial_conditions : tuple, optional
        初始条件 (位移, 速度)，默认为(0.0, 0.0)
    time_span : tuple, optional
        时间范围，默认为(0.0, 10.0)
        
    Returns:
    --------
    tuple
        (时间数组, 位移数组)
    """
    return solve_vibration_equation(k, m, force_amplitude, driving_frequency, 
                                  damping, initial_conditions, time_span)


def calculate_resonance_frequency(k: float, m: float, damping: float = 0.0) -> float:
    """
    计算共振频率
    
    Parameters:
    -----------
    k : float
        弹簧刚度系数
    m : float
        质量
    damping : float, optional
        阻尼系数，默认为0.0
        
    Returns:
    --------
    float
        共振频率
    """
    natural_freq = calc_natural_frequency(k, m)
    if damping == 0.0:
        return natural_freq
    else:
        # 有阻尼时的共振频率
        return natural_freq * np.sqrt(1 - (damping / (2 * np.sqrt(k * m)))**2)


def analyze_amplitude_response(k: float, m: float, force_amplitude: float,
                             frequency_range: Tuple[float, float] = (0.1, 10.0),
                             num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    分析振幅响应
    
    Parameters:
    -----------
    k : float
        弹簧刚度系数
    m : float
        质量
    force_amplitude : float
        外力幅值
    frequency_range : tuple, optional
        频率范围，默认为(0.1, 10.0)
    num_points : int, optional
        计算点数，默认为100
        
    Returns:
    --------
    tuple
        (频率数组, 振幅数组)
    """
    frequencies = np.linspace(frequency_range[0], frequency_range[1], num_points)
    amplitudes = []
    
    for freq in frequencies:
        amplitude = calc_forced_vibration_amplitude(force_amplitude, k, m, freq)
        amplitudes.append(amplitude)
    
    return frequencies, np.array(amplitudes)


def calc_natural_frequency(k: float, m: float) -> float:
    """
    计算系统的自然角频率。
    
    Parameters:
    -----------
    k : float
        弹簧刚度系数，单位与质量保持一致（常用N/m）。
    m : float
        质量，单位kg。
    
    Returns:
    --------
    float
        自然角频率 ω0，单位 rad/s。
    """
    return np.sqrt(k / m)


def calc_forced_vibration_amplitude(force_amplitude: float, k: float, m: float, driving_frequency: float) -> float:
    """
    计算无阻尼受迫振动（稳态）振幅。
    
    公式：A = F0 / (m (ω0^2 - ω^2))
    
    Parameters:
    -----------
    force_amplitude : float
        驱动力振幅 F0，单位N。
    k : float
        弹簧刚度系数。
    m : float
        质量，kg。
    driving_frequency : float
        外力角频率 ω，rad/s。
    
    Returns:
    --------
    float
        振幅 A（与位移单位一致，若力单位N且k单位N/长度，则A单位为长度）。
    """
    natural_freq = calc_natural_frequency(k, m)
    return force_amplitude / (m * (natural_freq**2 - driving_frequency**2))


def solve_vibration_equation(
    m: float,
    k: float,
    force_func: Callable[[float], float],
    t_span: Tuple[float, float],
    initial_conditions: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解无阻尼二阶受迫系统：m x¨ + k x = F(t)。
    
    Parameters:
    -----------
    m : float
        质量 kg。
    k : float
        刚度 N/长度。
    force_func : callable
        外力函数 F(t) -> N。
    t_span : tuple
        (t0, tf) 时间区间（秒）。
    initial_conditions : tuple
        (x0, v0) 初始位移与速度。
    t_eval : array_like, optional
        解的采样时间点；None 则由求解器自适应。
    
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        (t, y)，t为时间；y.shape=(2, len(t))，y[0]=x(t), y[1]=v(t)。
    """
    def system(t, y):
        x, v = y
        dxdt = v
        dvdt = (force_func(t) - k * x) / m
        return [dxdt, dvdt]

    sol = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, method='RK45')
    return sol.t, sol.y


def plot_vibration(t: np.ndarray, displacement: np.ndarray, title: str = "Vibration Response", save_path: Optional[str] = None) -> None:
    """
    绘制位移-时间响应曲线。
    
    Parameters:
    -----------
    t : array_like
        时间数组（秒）。
    displacement : array_like
        位移数组，与 t 一一对应。
    title : str, optional
        图标题。
    save_path : str, optional
        若提供则保存到该路径。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, displacement)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_indicator_diagram(pressure: np.ndarray, volume: np.ndarray, title: str = "Indicator Diagram", save_path: Optional[str] = None) -> float:
    """
    绘制示功图（p-V）并计算包络面积（循环功，梯形积分）。
    
    Parameters:
    -----------
    pressure : array_like
        压力数组。
    volume : array_like
        体积数组，需与 pressure 等长。
    title : str, optional
        图标题。
    save_path : str, optional
        若提供则保存到该路径。
    
    Returns:
    --------
    float
        示功图面积（功）。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(volume, pressure)
    plt.grid(True)
    plt.xlabel('Volume')
    plt.ylabel('Pressure')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    work = 0.0
    for i in range(len(volume) - 1):
        work += (pressure[i] + pressure[i + 1]) * (volume[i + 1] - volume[i]) / 2.0
    return abs(work)


def steam_engine_indicator_amplitude(
    pressure_func: Callable[[float], float],
    piston_area: float,
    mass: float,
    spring_constant: float,
    rotation_period: float,
) -> Tuple[float, dict]:
    """
    计算蒸汽机示功计划针的稳态振幅（无阻尼简化模型）。
    
    Parameters:
    -----------
    pressure_func : callable
        压力随时间变化函数 p(t)。
    piston_area : float
        活塞面积（与压力单位匹配，保证 F = p·S 的单位为 N）。
    mass : float
        质量 kg。
    spring_constant : float
        刚度系数，与长度单位一致（N/长度）。
    rotation_period : float
        轴单次转动周期 T（秒），外力角频率 ω = 2π/T。
    
    Returns:
    --------
    (float, dict)
        (振幅A, 中间结果字典)。
    """
    driving_frequency = 2.0 * np.pi / rotation_period

    # 基于一个周期估计平均与振幅
    t_values = np.linspace(0.0, rotation_period, 1000)
    p_values = np.array([pressure_func(t) for t in t_values])
    p_mean = float(np.mean(p_values))
    p_amplitude = float((np.max(p_values) - np.min(p_values)) / 2.0)

    force_amplitude = p_amplitude * piston_area
    natural_frequency = calc_natural_frequency(spring_constant, mass)
    amplitude = abs(calc_forced_vibration_amplitude(force_amplitude, spring_constant, mass, driving_frequency))

    results = {
        "natural_frequency": natural_frequency,
        "driving_frequency": driving_frequency,
        "force_amplitude": force_amplitude,
        "p_mean": p_mean,
        "p_amplitude": p_amplitude,
    }
    return amplitude, results


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数以运行示功计分析。
    
    Returns:
    --------
    argparse.Namespace
        解析后的参数集合。
    """
    parser = argparse.ArgumentParser(description="强迫振动系统分析工具")
    parser.add_argument("--mass", type=float, required=True, help="质量(kg)")
    parser.add_argument("--spring_constant", type=float, required=True, help="弹簧刚度系数(N/长度)")
    parser.add_argument("--piston_area", type=float, required=True, help="活塞面积(与压力单位匹配)")
    parser.add_argument("--rotation_period", type=float, required=True, help="旋转周期T(秒)")
    parser.add_argument("--pressure_mean", type=float, required=True, help="平均压力(与面积匹配)")
    parser.add_argument("--pressure_amplitude", type=float, required=True, help="压力振幅")
    parser.add_argument("--plot", action="store_true", help="是否绘制振动图")
    return parser.parse_args()


def _demo_args() -> argparse.Namespace:
    """
    当未提供命令行参数时，返回一组示例参数，便于直接运行。
    """
    class _A:
        pass
    a = _A()
    a.mass = 1.0
    a.spring_constant = 30.0
    a.piston_area = 4.0
    a.rotation_period = 1.0 / 3.0
    a.pressure_mean = 40.0
    a.pressure_amplitude = 30.0
    a.plot = False
    return a  # 类型上等同于具有属性的简单对象


def main() -> None:
    """命令行入口：支持无参数直接运行（使用内置示例），或提供参数运行。"""
    if len(sys.argv) <= 1:
        args = _demo_args()
    else:
        args = parse_args()

    def pressure_func(t: float) -> float:
        return args.pressure_mean + args.pressure_amplitude * np.sin(2.0 * np.pi * t / args.rotation_period)

    amplitude, results = steam_engine_indicator_amplitude(
        pressure_func, args.piston_area, args.mass, args.spring_constant, args.rotation_period
    )

    print(f"自然频率: {results['natural_frequency']:.4f} rad/s")
    print(f"驱动频率: {results['driving_frequency']:.4f} rad/s")
    print(f"振幅: {amplitude:.6f} (位移单位)")

    if args.plot:
        t = np.linspace(0.0, 2.0 * args.rotation_period, 1000)

        def force_func(t_: float) -> float:
            return pressure_func(t_) * args.piston_area

        tt, yy = solve_vibration_equation(args.mass, args.spring_constant, force_func, (0.0, float(2.0 * args.rotation_period)), (0.0, 0.0), t)
        plot_vibration(tt, yy[0], "示功计划针的振动响应")


if __name__ == "__main__":
    main()
