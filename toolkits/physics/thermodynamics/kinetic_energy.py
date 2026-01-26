#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
气体粒子动能和温度分析工具

该模块提供分析气体粒子动能和温度关系的函数，基于气体动理论。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse


def calculate_average_kinetic_energy(velocities, masses=None):
    """
    计算粒子系统的平均动能。
    
    Parameters:
    -----------
    velocities : ndarray
        粒子速度数组，可以是以下形式之一：
        - 1D数组：表示粒子的速度大小
        - 2D数组：形状为(n, 3)，表示n个粒子在3D空间的速度向量
    masses : List[float]
        粒子质量，默认为1（单位质量）。与velocities长度相同的数组。
    
    Returns:
    --------
    float
        系统的平均动能
    """
    velocities = np.array(velocities)
    # 处理质量参数
    if masses is None:
        n_particles = len(velocities) if velocities.ndim == 1 else velocities.shape[0]
        masses = np.ones(n_particles)
    else:
        masses = np.array(masses)
        # 检查质量列表长度是否与速度匹配
        expected_length = len(velocities) if velocities.ndim == 1 else velocities.shape[0]
        if len(masses) != expected_length:
            raise ValueError("Length of masses must match number of particles")
    
    # 使用numpy进行计算
    if velocities.ndim == 1:
        # 如果输入是速度大小
        kinetic_energies = 0.5 * masses * velocities**2
    else:
        # 如果输入是速度向量
        v_squared = np.sum(velocities**2, axis=1)
        kinetic_energies = 0.5 * masses * v_squared
    
    return np.mean(kinetic_energies)


def temperature_from_kinetic_energy(avg_kinetic_energy, degrees_of_freedom=3):
    """
    基于平均动能计算气体温度。
    
    Parameters:
    -----------
    avg_kinetic_energy : float
        粒子的平均动能，单位为焦耳(J)
    degrees_of_freedom : int
        自由度，默认为3（单原子气体的平移自由度）
    
    Returns:
    --------
    float
        气体温度，单位为开尔文(K)
    """
    # 玻尔兹曼常数 (J/K)
    k_B = 1.380649e-23
    
    # 使用能量均分定理: E_k = (f/2) * k_B * T
    temperature = (2 * avg_kinetic_energy) / (degrees_of_freedom * k_B)
    return temperature


def compare_temperatures(kinetic_energy1, kinetic_energy2, degrees_of_freedom=3):
    """
    比较两个气体样本的温度。
    
    Parameters:
    -----------
    kinetic_energy1 : float
        第一个样本的平均动能
    kinetic_energy2 : float
        第二个样本的平均动能
    degrees_of_freedom : int
        自由度，默认为3
    
    Returns:
    --------
    dict
        包含比较结果的字典，键为'higher_temperature'，值为'sample1'或'sample2'
    """
    temp1 = temperature_from_kinetic_energy(kinetic_energy1, degrees_of_freedom)
    temp2 = temperature_from_kinetic_energy(kinetic_energy2, degrees_of_freedom)
    
    if temp1 > temp2:
        return {'higher_temperature': 'sample1', 'temp1': temp1, 'temp2': temp2}
    elif temp2 > temp1:
        return {'higher_temperature': 'sample2', 'temp1': temp1, 'temp2': temp2}
    else:
        return {'higher_temperature': 'equal', 'temp1': temp1, 'temp2': temp2}


def visualize_velocity_distribution(velocities1, velocities2, labels=None):
    """
    可视化两个气体样本的速度分布。
    
    Parameters:
    -----------
    velocities1 : ndarray
        第一个样本的速度数组
    velocities2 : ndarray
        第二个样本的速度数组
    labels : List[str]
        两个样本的标签，默认为['Sample 1', 'Sample 2']
    
    Returns:
    --------
    matplotlib.figure.Figure
        生成的图形对象
    """
    if labels is None:
        labels = ['Sample 1', 'Sample 2']
    
    # 计算速度大小
    if velocities1.ndim > 1:
        v1_mag = np.sqrt(np.sum(velocities1**2, axis=1))
    else:
        v1_mag = velocities1
        
    if velocities2.ndim > 1:
        v2_mag = np.sqrt(np.sum(velocities2**2, axis=1))
    else:
        v2_mag = velocities2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制速度分布直方图
    bins = np.linspace(0, max(np.max(v1_mag), np.max(v2_mag)) * 1.1, 50)
    ax.hist(v1_mag, bins=bins, alpha=0.5, label=labels[0])
    ax.hist(v2_mag, bins=bins, alpha=0.5, label=labels[1])
    
    # 拟合麦克斯韦分布
    x = np.linspace(0, max(np.max(v1_mag), np.max(v2_mag)) * 1.2, 1000)
    
    # 样本1的拟合
    params1 = stats.maxwell.fit(v1_mag, loc=0)
    pdf1 = stats.maxwell.pdf(x, *params1)
    ax.plot(x, pdf1 * len(v1_mag) * (bins[1] - bins[0]), 'r-', linewidth=2, 
            label=f'{labels[0]} Maxwell Fit')
    
    # 样本2的拟合
    params2 = stats.maxwell.fit(v2_mag, loc=0)
    pdf2 = stats.maxwell.pdf(x, *params2)
    ax.plot(x, pdf2 * len(v2_mag) * (bins[1] - bins[0]), 'b-', linewidth=2, 
            label=f'{labels[1]} Maxwell Fit')
    
    ax.set_xlabel('Velocity Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Velocity Distribution Comparison')
    ax.legend()
    
    return fig


def parse_args():
    """
    解析命令行参数。
    
    Returns:
    --------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='Analyze gas particle kinetic energy and temperature')
    
    parser.add_argument('--velocities1', type=str, help='Comma-separated velocities for sample 1')
    parser.add_argument('--velocities2', type=str, help='Comma-separated velocities for sample 2')
    parser.add_argument('--masses1', type=str, default=None, help='Comma-separated masses for sample 1')
    parser.add_argument('--masses2', type=str, default=None, help='Comma-separated masses for sample 2')
    parser.add_argument('--visualize', action='store_true', help='Generate velocity distribution visualization')
    parser.add_argument('--output', type=str, default='velocity_distribution.png', 
                        help='Output file for visualization')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 解析输入数据
    # 样品1：较低速度分布
    velocities1 = np.random.normal(0, 1, (100, 2)).tolist()

    # 样品2：较高速度分布
    velocities2 = np.random.normal(0, 2, (100, 2)).tolist()
    
    masses1 = None if args.masses1 is None else np.array([float(m) for m in args.masses1.split(',')]).tolist()
    masses2 = None if args.masses2 is None else np.array([float(m) for m in args.masses2.split(',')]).tolist()
    
    # 计算平均动能
    ke1 = calculate_average_kinetic_energy(velocities1, masses1)
    ke2 = calculate_average_kinetic_energy(velocities2, masses2)
    
    # 比较温度
    result = compare_temperatures(ke1, ke2)
    
    print(f"Average kinetic energy of sample 1: {ke1:.4e} J")
    print(f"Average kinetic energy of sample 2: {ke2:.4e} J")
    print(f"Temperature of sample 1: {result['temp1']:.2f} K")
    print(f"Temperature of sample 2: {result['temp2']:.2f} K")
    
    if result['higher_temperature'] == 'sample1':
        print("Sample 1 has the higher temperature.")
    elif result['higher_temperature'] == 'sample2':
        print("Sample 2 has the higher temperature.")
    else:
        print("Both samples have the same temperature.")
    
    # 可视化
    if args.visualize:
        fig = visualize_velocity_distribution(velocities1, velocities2)
        plt.savefig(args.output)
        plt.close(fig)
        print(f"Visualization saved to {args.output}")