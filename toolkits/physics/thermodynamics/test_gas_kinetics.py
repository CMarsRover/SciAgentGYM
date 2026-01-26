#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
气体动力学分析工具的测试模块

这个模块包含对gas_kinetics.py中函数的测试，以及对特定物理问题的解决方案。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
from typing import Dict, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入要测试的模块
from gas_kinetics import (
    calculate_particle_kinetic_energy, 
    calculate_average_kinetic_energy,
    compare_temperatures, 
    estimate_temperature_from_ke,
    visualize_kinetic_energy_distribution
)


class TestGasKinetics(unittest.TestCase):
    """气体动力学分析工具的测试类"""
    
    def test_calculate_particle_kinetic_energy_scalar(self):
        """测试标量速度的动能计算"""
        mass = 1.0  # 1 kg
        velocity = 2.0  # 2 m/s
        expected_ke = 0.5 * mass * velocity**2  # 2 J
        calculated_ke = calculate_particle_kinetic_energy(mass, velocity)
        self.assertAlmostEqual(calculated_ke, expected_ke, places=10)
    
    def test_calculate_particle_kinetic_energy_vector(self):
        """测试向量速度的动能计算"""
        mass = 1.0  # 1 kg
        velocity_vector = np.array([1.0, 2.0, 2.0])  # 3D velocity
        expected_ke = 0.5 * mass * np.sum(velocity_vector**2)  # 4.5 J
        calculated_ke = calculate_particle_kinetic_energy(mass, velocity_vector)
        self.assertAlmostEqual(calculated_ke, expected_ke, places=10)
    
    def test_calculate_average_kinetic_energy(self):
        """测试平均动能计算"""
        masses = np.array([1.0, 2.0, 3.0])
        velocities = np.array([1.0, 2.0, 3.0])
        expected_avg_ke = np.mean([0.5*m*v**2 for m, v in zip(masses, velocities)])
        calculated_avg_ke = calculate_average_kinetic_energy(masses, velocities)
        self.assertAlmostEqual(calculated_avg_ke, expected_avg_ke, places=10)
    
    def test_compare_temperatures_sample1_higher(self):
        """测试温度比较 - 样本1温度更高"""
        higher_temp, ratio = compare_temperatures(100, 50)
        self.assertEqual(higher_temp, "sample1")
        self.assertAlmostEqual(ratio, 2.0, places=10)
    
    def test_compare_temperatures_sample2_higher(self):
        """测试温度比较 - 样本2温度更高"""
        higher_temp, ratio = compare_temperatures(50, 100)
        self.assertEqual(higher_temp, "sample2")
        self.assertAlmostEqual(ratio, 0.5, places=10)
    
    def test_compare_temperatures_equal(self):
        """测试温度比较 - 温度相等"""
        higher_temp, ratio = compare_temperatures(100, 100)
        self.assertEqual(higher_temp, "equal")
        self.assertAlmostEqual(ratio, 1.0, places=10)
    
    def test_estimate_temperature_from_ke_absolute(self):
        """测试从动能估算绝对温度"""
        # 使用已知的动能值计算温度
        avg_ke = 6.21e-21  # J (对应300K的动能)
        estimated_temp = estimate_temperature_from_ke(avg_ke)
        self.assertAlmostEqual(estimated_temp, 300, places=0)
    
    def test_estimate_temperature_from_ke_relative(self):
        """测试从动能估算相对温度"""
        avg_ke = 100
        reference_ke = 50
        reference_temp = 300
        estimated_temp = estimate_temperature_from_ke(avg_ke, reference_ke, reference_temp)
        self.assertAlmostEqual(estimated_temp, 600, places=10)


def test_basic_functions():
    """测试基本函数的正确性"""
    print("开始基本函数测试...")
    
    # 测试动能计算
    mass = 1.0  # 1 kg
    velocity = 2.0  # 2 m/s
    expected_ke = 0.5 * mass * velocity**2  # 2 J
    calculated_ke = calculate_particle_kinetic_energy(mass, velocity)
    assert np.isclose(calculated_ke, expected_ke), f"动能计算错误: 期望 {expected_ke}, 得到 {calculated_ke}"
    
    # 测试向量速度
    velocity_vector = np.array([1.0, 2.0, 2.0])  # 3D velocity
    expected_ke = 0.5 * mass * np.sum(velocity_vector**2)  # 4.5 J
    calculated_ke = calculate_particle_kinetic_energy(mass, velocity_vector)
    assert np.isclose(calculated_ke, expected_ke), f"向量动能计算错误: 期望 {expected_ke}, 得到 {calculated_ke}"
    
    # 测试平均动能计算
    masses = np.array([1.0, 2.0, 3.0])
    velocities = np.array([1.0, 2.0, 3.0])
    expected_avg_ke = np.mean([0.5*m*v**2 for m, v in zip(masses, velocities)])
    calculated_avg_ke = calculate_average_kinetic_energy(masses, velocities)
    assert np.isclose(calculated_avg_ke, expected_avg_ke), f"平均动能计算错误: 期望 {expected_avg_ke}, 得到 {calculated_avg_ke}"
    
    # 测试温度比较
    higher_temp, ratio = compare_temperatures(100, 50)
    assert higher_temp == "sample1" and np.isclose(ratio, 2.0), "温度比较错误"
    
    higher_temp, ratio = compare_temperatures(50, 100)
    assert higher_temp == "sample2" and np.isclose(ratio, 0.5), "温度比较错误"
    
    print("基本函数测试通过!")


def solve_gas_temperature_comparison() -> Dict[str, Any]:
    """
    解决给定的气体温度比较问题
    
    问题: 比较两个气体样本中粒子的平均动能，确定哪个样本温度更高。
    两个样本在相同的封闭、刚性容器中，粒子数量相同。
    
    Returns:
    --------
    dict
        包含问题解决状态和答案的字典
    """
    print("开始解决气体温度比较问题...")
    
    # 创建模拟数据
    n_particles = 100  # 每个样本中的粒子数
    
    # 假设所有粒子质量相同 (例如氢分子)
    mass_h2 = 3.35e-27  # kg (氢分子质量)
    masses1 = np.full(n_particles, mass_h2)
    masses2 = np.full(n_particles, mass_h2)
    
    # 样本1: 粒子运动较快 (模拟较高温度)
    np.random.seed(42)  # 确保结果可重现
    velocities1 = np.random.normal(2000, 200, n_particles)  # 平均速度2000 m/s
    
    # 样本2: 粒子运动较慢 (模拟较低温度)
    velocities2 = np.random.normal(1000, 100, n_particles)  # 平均速度1000 m/s
    
    # 计算平均动能
    avg_ke1 = calculate_average_kinetic_energy(masses1, velocities1)
    avg_ke2 = calculate_average_kinetic_energy(masses2, velocities2)
    
    # 比较温度
    higher_temp, ratio = compare_temperatures(avg_ke1, avg_ke2)
    
    # 估算绝对温度
    temp1 = estimate_temperature_from_ke(avg_ke1)
    temp2 = estimate_temperature_from_ke(avg_ke2)
    
    print(f"样本1的平均动能: {avg_ke1:.2e} J")
    print(f"样本2的平均动能: {avg_ke2:.2e} J")
    print(f"温度较高的样本: {higher_temp}")
    print(f"平均动能比率: {ratio:.4f}")
    print(f"样本1估算温度: {temp1:.1f} K")
    print(f"样本2估算温度: {temp2:.1f} K")
    
    result = {
        'isTrue': True,
        'answer': {
            'sample1_avg_ke': avg_ke1,
            'sample2_avg_ke': avg_ke2,
            'higher_temperature_sample': higher_temp,
            'kinetic_energy_ratio': ratio,
            'sample1_temperature': temp1,
            'sample2_temperature': temp2
        }
    }
    
    return result


def test_visualization():
    """测试可视化功能"""
    print("开始测试可视化功能...")
    
    # 创建测试数据
    n_particles = 1000
    mass = 3.35e-27  # kg (氢分子质量)
    masses = np.full(n_particles, mass)
    
    # 创建不同温度的速度分布
    np.random.seed(42)
    velocities_cold = np.random.normal(1000, 100, n_particles)
    velocities_hot = np.random.normal(2000, 200, n_particles)
    
    # 计算动能
    ke_cold = calculate_particle_kinetic_energy(masses, velocities_cold)
    ke_hot = calculate_particle_kinetic_energy(masses, velocities_hot)
    
    # 创建可视化
    fig1 = visualize_kinetic_energy_distribution(ke_cold, "Cold Gas Sample")
    fig2 = visualize_kinetic_energy_distribution(ke_hot, "Hot Gas Sample")
    
    # 保存图片
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig1.savefig(os.path.join(output_dir, "cold_gas_distribution.png"), dpi=150)
    fig2.savefig(os.path.join(output_dir, "hot_gas_distribution.png"), dpi=150)
    
    print("可视化测试完成，图片已保存")
    
    return fig1, fig2


def test_edge_cases():
    """测试边界情况"""
    print("开始测试边界情况...")
    
    # 测试零速度
    mass = 1.0
    velocity = 0.0
    ke = calculate_particle_kinetic_energy(mass, velocity)
    assert ke == 0.0, f"零速度动能应为0，得到 {ke}"
    
    # 测试负速度（应该与正速度相同）
    velocity_neg = -2.0
    ke_neg = calculate_particle_kinetic_energy(mass, velocity_neg)
    ke_pos = calculate_particle_kinetic_energy(mass, 2.0)
    assert np.isclose(ke_neg, ke_pos), f"负速度动能应与正速度相同"
    
    # 测试极小质量
    tiny_mass = 1e-30
    ke_tiny = calculate_particle_kinetic_energy(tiny_mass, 1.0)
    expected_tiny = 0.5 * tiny_mass * 1.0
    assert np.isclose(ke_tiny, expected_tiny), f"极小质量动能计算错误"
    
    print("边界情况测试通过!")


def main():
    """主测试函数"""
    print("=" * 60)
    print("气体动力学分析工具测试")
    print("=" * 60)
    
    try:
        # 运行基本函数测试
        test_basic_functions()
        print()
        
        # 运行边界情况测试
        test_edge_cases()
        print()
        
        # 解决实际问题
        result = solve_gas_temperature_comparison()
        print()
        
        # 测试可视化
        test_visualization()
        print()
        
        # 运行单元测试
        print("运行单元测试...")
        unittest.main(argv=[''], exit=False, verbosity=2)
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        return {'isTrue': False, 'error': str(e)}


if __name__ == "__main__":
    main()
