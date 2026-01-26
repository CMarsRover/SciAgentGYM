#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
气体动力学分析工具使用示例

这个文件展示了如何使用gas_kinetics模块来解决实际的物理问题。
"""

import sys
import os
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gas_kinetics import (
    calculate_particle_kinetic_energy,
    calculate_average_kinetic_energy,
    compare_temperatures,
    estimate_temperature_from_ke,
    visualize_kinetic_energy_distribution
)


def example_1_basic_kinetic_energy():
    """示例1: 基本动能计算"""
    print("=" * 50)
    print("示例1: 基本动能计算")
    print("=" * 50)
    
    # 计算单个粒子的动能
    mass = 1.67e-27  # kg (质子质量)
    velocity = 1000  # m/s
    
    ke = calculate_particle_kinetic_energy(mass, velocity)
    print(f"质量为 {mass:.2e} kg 的粒子，速度为 {velocity} m/s")
    print(f"动能为: {ke:.2e} J")
    print()


def example_2_gas_sample_comparison():
    """示例2: 气体样本比较"""
    print("=" * 50)
    print("示例2: 气体样本比较")
    print("=" * 50)
    
    # 创建两个气体样本
    n_particles = 50
    
    # 氢分子质量
    mass_h2 = 3.35e-27  # kg
    masses = np.full(n_particles, mass_h2)
    
    # 样本1: 较高温度 (粒子运动较快)
    np.random.seed(42)
    velocities1 = np.random.normal(1500, 150, n_particles)
    
    # 样本2: 较低温度 (粒子运动较慢)
    velocities2 = np.random.normal(800, 80, n_particles)
    
    # 计算平均动能
    avg_ke1 = calculate_average_kinetic_energy(masses, velocities1)
    avg_ke2 = calculate_average_kinetic_energy(masses, velocities2)
    
    # 比较温度
    higher_temp, ratio = compare_temperatures(avg_ke1, avg_ke2)
    
    # 估算温度
    temp1 = estimate_temperature_from_ke(avg_ke1)
    temp2 = estimate_temperature_from_ke(avg_ke2)
    
    print(f"样本1平均动能: {avg_ke1:.2e} J")
    print(f"样本2平均动能: {avg_ke2:.2e} J")
    print(f"温度较高的样本: {higher_temp}")
    print(f"动能比率: {ratio:.2f}")
    print(f"样本1估算温度: {temp1:.1f} K")
    print(f"样本2估算温度: {temp2:.1f} K")
    print()


def example_3_visualization():
    """示例3: 动能分布可视化"""
    print("=" * 50)
    print("示例3: 动能分布可视化")
    print("=" * 50)
    
    # 创建不同温度的气体样本
    n_particles = 1000
    mass = 3.35e-27  # kg (氢分子)
    masses = np.full(n_particles, mass)
    
    # 三个不同温度的样本
    temperatures = [1000, 3000, 5000]  # K
    sample_names = ["低温样本", "中温样本", "高温样本"]
    
    for i, (temp, name) in enumerate(zip(temperatures, sample_names)):
        # 根据温度生成速度分布 (使用麦克斯韦-玻尔兹曼分布)
        k_B = 1.380649e-23  # J/K
        v_rms = np.sqrt(3 * k_B * temp / mass)  # 均方根速度
        
        np.random.seed(42 + i)
        velocities = np.random.normal(0, v_rms/np.sqrt(3), n_particles)
        
        # 计算动能
        kinetic_energies = calculate_particle_kinetic_energy(masses, velocities)
        
        # 创建可视化
        fig = visualize_kinetic_energy_distribution(kinetic_energies, name)
        
        # 保存图片
        output_file = f"kinetic_energy_distribution_{temp}K.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"已保存 {name} 的动能分布图: {output_file}")
        
        # 计算统计信息
        avg_ke = np.mean(kinetic_energies)
        std_ke = np.std(kinetic_energies)
        print(f"{name} - 平均动能: {avg_ke:.2e} J, 标准差: {std_ke:.2e} J")
    
    print()


def example_4_relative_temperature():
    """示例4: 相对温度计算"""
    print("=" * 50)
    print("示例4: 相对温度计算")
    print("=" * 50)
    
    # 已知参考样本的温度和动能
    reference_temp = 300  # K (室温)
    reference_ke = 6.21e-21  # J (300K时氢分子的平均动能)
    
    # 未知样本的动能
    unknown_ke = 1.24e-20  # J
    
    # 计算相对温度
    relative_temp = estimate_temperature_from_ke(
        unknown_ke, 
        reference_ke=reference_ke, 
        reference_temp=reference_temp
    )
    
    print(f"参考样本温度: {reference_temp} K")
    print(f"参考样本平均动能: {reference_ke:.2e} J")
    print(f"未知样本平均动能: {unknown_ke:.2e} J")
    print(f"未知样本相对温度: {relative_temp:.1f} K")
    print(f"温度比: {relative_temp/reference_temp:.2f}")
    print()


def main():
    """主函数"""
    print("气体动力学分析工具使用示例")
    print("=" * 60)
    
    try:
        example_1_basic_kinetic_energy()
        example_2_gas_sample_comparison()
        example_3_visualization()
        example_4_relative_temperature()
        
        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行示例时出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
