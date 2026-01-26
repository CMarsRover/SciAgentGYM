#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
气体动力学分析工具

这个模块提供了用于分析气体样本中粒子动能和温度关系的函数。
基于统计力学和气体动理论，可以计算气体样本的平均动能和相对温度。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Tuple, Union, Optional

# 添加项目根目录到Python路径，避免相对路径问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def calculate_particle_kinetic_energy(mass: float, velocity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    计算粒子的动能。
    
    Parameters:
    -----------
    mass : float
        粒子质量，单位为kg
    velocity : float or numpy.ndarray
        粒子速度，可以是标量(速率)或向量(速度)，单位为m/s
        
    Returns:
    --------
    float or numpy.ndarray
        粒子动能，单位为J
    """
    # 如果速度是向量，先计算速率
    if isinstance(velocity, np.ndarray) and velocity.ndim > 0:
        v_squared = np.sum(velocity**2, axis=-1)
    else:
        v_squared = velocity**2
        
    # 计算动能: E_k = 1/2 * m * v^2
    kinetic_energy = 0.5 * mass * v_squared
    return kinetic_energy


def calculate_average_kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> float:
    """
    计算一组粒子的平均动能。
    
    Parameters:
    -----------
    masses : numpy.ndarray
        粒子质量数组，单位为kg
    velocities : numpy.ndarray
        粒子速度数组，每行表示一个粒子的速度向量，单位为m/s
        
    Returns:
    --------
    float
        平均动能，单位为J
    """
    # 确保输入数组形状兼容
    if masses.shape[0] != velocities.shape[0]:
        raise ValueError("质量数组和速度数组的粒子数量必须相同")
    
    # 计算每个粒子的动能
    if velocities.ndim == 1:  # 如果只提供了速率
        kinetic_energies = np.array([calculate_particle_kinetic_energy(m, v) 
                                     for m, v in zip(masses, velocities)])
    else:  # 如果提供了速度向量
        kinetic_energies = np.array([calculate_particle_kinetic_energy(m, v) 
                                     for m, v in zip(masses, velocities)])
    
    # 计算平均值
    average_kinetic_energy = np.mean(kinetic_energies)
    return average_kinetic_energy


def compare_temperatures(avg_ke_1: float, avg_ke_2: float) -> Tuple[str, float]:
    """
    比较两个气体样本的温度，基于它们的平均动能。
    
    Parameters:
    -----------
    avg_ke_1 : float
        第一个样本的平均动能，单位为J
    avg_ke_2 : float
        第二个样本的平均动能，单位为J
        
    Returns:
    --------
    tuple
        包含两个元素：
        - 具有更高温度的样本标识("sample1", "sample2"或"equal")
        - 平均动能之比(avg_ke_1 / avg_ke_2)
    """
    ratio = avg_ke_1 / avg_ke_2
    
    if avg_ke_1 > avg_ke_2:
        return "sample1", ratio
    elif avg_ke_2 > avg_ke_1:
        return "sample2", ratio
    else:
        return "equal", 1.0


def temperature_from_kinetic_energy(avg_kinetic_energy: float, 
                                   boltzmann_constant: float = 1.380649e-23) -> float:
    """
    根据平均动能计算温度，使用气体分子运动论关系。
    
    Parameters:
    -----------
    avg_kinetic_energy : float
        平均动能，单位为J
    boltzmann_constant : float, optional
        玻尔兹曼常数，默认为1.380649e-23 J/K
        
    Returns:
    --------
    float
        温度，单位为K
    """
    # 使用关系式 T = 2<E_k>/(3k_B)
    temperature = (2 * avg_kinetic_energy) / (3 * boltzmann_constant)
    return temperature


def compare_gas_samples(sample1_velocities: np.ndarray, sample2_velocities: np.ndarray,
                       mass1: float = 1.0, mass2: float = 1.0) -> Dict:
    """
    比较两个气体样本的平均动能和温度。
    
    Parameters:
    -----------
    sample1_velocities : numpy.ndarray
        样本1中粒子的速度数组，单位为m/s
    sample2_velocities : numpy.ndarray
        样本2中粒子的速度数组，单位为m/s
    mass1 : float, optional
        样本1中粒子的质量，默认为1.0（相对单位）
    mass2 : float, optional
        样本2中粒子的质量，默认为1.0（相对单位）
        
    Returns:
    --------
    dict
        包含比较结果的字典
    """
    # 为每个粒子创建质量数组
    masses1 = np.ones_like(sample1_velocities) * mass1
    masses2 = np.ones_like(sample2_velocities) * mass2
    
    # 计算平均动能
    avg_ke1 = calculate_average_kinetic_energy(masses1, sample1_velocities)
    avg_ke2 = calculate_average_kinetic_energy(masses2, sample2_velocities)
    
    # 计算温度
    temp1 = temperature_from_kinetic_energy(avg_ke1)
    temp2 = temperature_from_kinetic_energy(avg_ke2)
    
    # 确定哪个样本温度更高
    higher_temp = "Sample 1" if temp1 > temp2 else "Sample 2"
    
    return {
        'avg_kinetic_energy': {
            'sample1': avg_ke1,
            'sample2': avg_ke2
        },
        'temperature': {
            'sample1': temp1,
            'sample2': temp2
        },
        'higher_temperature': higher_temp
    }


def visualize_velocity_distribution(sample1_velocities: np.ndarray, sample2_velocities: np.ndarray,
                                  bins: int = 10, save_path: Optional[str] = None) -> None:
    """
    可视化两个样本的速度分布。
    
    Parameters:
    -----------
    sample1_velocities : numpy.ndarray
        样本1中粒子的速度数组
    sample2_velocities : numpy.ndarray
        样本2中粒子的速度数组
    bins : int, optional
        直方图的箱数，默认为10
    save_path : str, optional
        图表保存路径，如果为None则显示图表
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(sample1_velocities, bins=bins, alpha=0.7, label='Sample 1', color='blue')
    plt.hist(sample2_velocities, bins=bins, alpha=0.7, label='Sample 2', color='red')
    
    # 添加平均值线
    plt.axvline(np.mean(sample1_velocities), color='blue', linestyle='--', 
                label=f'Sample 1 Mean: {np.mean(sample1_velocities):.2f}')
    plt.axvline(np.mean(sample2_velocities), color='red', linestyle='--', 
                label=f'Sample 2 Mean: {np.mean(sample2_velocities):.2f}')
    
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Frequency')
    plt.title('Velocity Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def estimate_temperature_from_ke(avg_ke: float, reference_ke: Optional[float] = None, 
                                reference_temp: Optional[float] = None) -> float:
    """
    基于平均动能估计气体的温度。
    
    Parameters:
    -----------
    avg_ke : float
        气体样本的平均动能，单位为J
    reference_ke : float, optional
        参考气体的平均动能，用于相对温度计算
    reference_temp : float, optional
        参考气体的温度，单位为K
        
    Returns:
    --------
    float
        估计的温度，如果提供了参考值则为绝对温度(K)，否则为相对温度
    """
    # 玻尔兹曼常数
    k_B = 1.380649e-23  # J/K
    
    if reference_ke is None:
        # 使用统计力学关系: <E_k> = (3/2)k_B*T
        temperature = (2/3) * avg_ke / k_B
    else:
        # 计算相对温度
        temperature = avg_ke / reference_ke
        if reference_temp is not None:
            temperature *= reference_temp
            
    return temperature


def visualize_kinetic_energy_distribution(kinetic_energies: np.ndarray, 
                                          sample_name: str = "Gas Sample") -> plt.Figure:
    """
    可视化气体样本中粒子动能的分布。
    
    Parameters:
    -----------
    kinetic_energies : numpy.ndarray
        粒子动能数组，单位为J
    sample_name : str, optional
        样本名称，用于图表标题
        
    Returns:
    --------
    matplotlib.figure.Figure
        包含动能分布直方图的图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建直方图
    ax.hist(kinetic_energies, bins='auto', alpha=0.7, color='skyblue', 
            edgecolor='black', density=True)
    
    # 添加标题和标签
    ax.set_title(f'Kinetic Energy Distribution - {sample_name}')
    ax.set_xlabel('Kinetic Energy (J)')
    ax.set_ylabel('Probability Density')
    
    # 添加平均值线
    mean_ke = np.mean(kinetic_energies)
    ax.axvline(mean_ke, color='red', linestyle='dashed', linewidth=2)
    ax.text(mean_ke*1.1, ax.get_ylim()[1]*0.9, f'Mean: {mean_ke:.2e} J', 
            color='red', fontsize=12)
    
    # 美化图表
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


def parse_args():
    """
    解析命令行参数。
    
    Returns:
    --------
    argparse.Namespace
        包含解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='气体动力学分析工具')
    
    parser.add_argument('--masses1', type=float, nargs='+', 
                        help='第一个样本中粒子的质量列表(kg)')
    parser.add_argument('--velocities1', type=float, nargs='+', 
                        help='第一个样本中粒子的速率列表(m/s)')
    parser.add_argument('--masses2', type=float, nargs='+', 
                        help='第二个样本中粒子的质量列表(kg)')
    parser.add_argument('--velocities2', type=float, nargs='+', 
                        help='第二个样本中粒子的速率列表(m/s)')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否生成动能分布可视化')
    parser.add_argument('--output', type=str, default='ke_distribution.png', 
                        help='可视化输出文件名')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 将输入转换为numpy数组
    if args.masses1 and args.velocities1:
        masses1 = np.array(args.masses1)
        velocities1 = np.array(args.velocities1)
        
        avg_ke1 = calculate_average_kinetic_energy(masses1, velocities1)
        print(f"样本1的平均动能: {avg_ke1:.2e} J")
        
        if args.masses2 and args.velocities2:
            masses2 = np.array(args.masses2)
            velocities2 = np.array(args.velocities2)
            
            avg_ke2 = calculate_average_kinetic_energy(masses2, velocities2)
            print(f"样本2的平均动能: {avg_ke2:.2e} J")
            
            higher_temp, ratio = compare_temperatures(avg_ke1, avg_ke2)
            print(f"温度较高的样本: {higher_temp}")
            print(f"平均动能比率: {ratio:.4f}")
            
            if args.visualize:
                ke1 = calculate_particle_kinetic_energy(masses1, velocities1)
                ke2 = calculate_particle_kinetic_energy(masses2, velocities2)
                
                fig1 = visualize_kinetic_energy_distribution(ke1, "Sample 1")
                fig1.savefig("sample1_" + args.output)
                
                fig2 = visualize_kinetic_energy_distribution(ke2, "Sample 2")
                fig2.savefig("sample2_" + args.output)
