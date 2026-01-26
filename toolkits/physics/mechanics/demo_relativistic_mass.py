#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相对论质量演示脚本
展示纵向质量和横向质量的计算
"""

from RelativisticPhysicsCalculator import RelativisticPhysicsCalculator
from astropy import units as u
from astropy.constants import c
import numpy as np

def demo_relativistic_mass():
    """演示相对论质量计算"""
    print("🎯 相对论质量计算演示")
    print("=" * 60)
    
    # 创建计算器
    calc = RelativisticPhysicsCalculator()
    
    # 测试粒子：电子
    electron_mass = 9.109e-31 * u.kg
    
    print("📊 理论背景:")
    print("   纵向质量: m_long = γ³m (力平行于运动方向)")
    print("   横向质量: m_trans = γm (力垂直于运动方向)")
    print("   其中 γ = 1/√(1 - v²/c²) 是洛伦兹因子")
    print()
    
    # 测试不同速度
    velocities = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]
    
    print("📈 不同速度下的质量比较:")
    print("-" * 60)
    print(f"{'速度 v/c':<10} {'γ':<8} {'m_trans/m':<12} {'m_long/m':<12} {'m_long/m_trans':<12}")
    print("-" * 60)
    
    for v_ratio in velocities:
        v = v_ratio * c
        gamma = calc.lorentz_factor_coding(v)
        m_trans = calc.transverse_mass_coding(electron_mass, v)
        m_long = calc.longitudinal_mass_coding(electron_mass, v)
        
        print(f"{v_ratio:<10.2f} {gamma:<8.3f} {m_trans/electron_mass:<12.3f} {m_long/electron_mass:<12.3f} {m_long/m_trans:<12.3f}")
    
    print()
    
    # 详细分析一个特定速度
    v_test = 0.9 * c
    print(f"🔍 详细分析：v = {v_test/c:.1f}c")
    print("-" * 40)
    
    mass_data = calc.relativistic_mass_comparison_coding(electron_mass, v_test)
    
    print(f"静质量 (m):           {mass_data['rest_mass']:.3e} kg")
    print(f"横向质量 (m_trans):   {mass_data['transverse_mass']:.3e} kg")
    print(f"纵向质量 (m_long):    {mass_data['longitudinal_mass']:.3e} kg")
    print(f"洛伦兹因子 (γ):       {mass_data['lorentz_factor']:.3f}")
    print()
    print(f"横向质量比 (m_trans/m): {mass_data['mass_ratio_transverse']:.3f}")
    print(f"纵向质量比 (m_long/m):  {mass_data['mass_ratio_longitudinal']:.3f}")
    print(f"纵向/横向质量比:       {mass_data['mass_ratio_longitudinal']/mass_data['mass_ratio_transverse']:.3f}")
    
    print()
    print("💡 物理意义:")
    print("   • 横向质量 = γm：垂直于运动方向的力产生的加速度")
    print("   • 纵向质量 = γ³m：平行于运动方向的力产生的加速度")
    print("   • 纵向质量比横向质量大 γ² 倍")
    print("   • 这解释了为什么加速高能粒子越来越困难")
    
    return calc, electron_mass

def demo_force_acceleration_ratio():
    """演示力与加速度比值"""
    print("\n" + "=" * 60)
    print("🔧 力与加速度比值演示")
    print("=" * 60)
    
    calc = RelativisticPhysicsCalculator()
    electron_mass = 9.109e-31 * u.kg
    
    print("📊 理论公式:")
    print("   纵向力: F_parallel = m_long × a = γ³m × a")
    print("   横向力: F_perpendicular = m_trans × a = γm × a")
    print()
    
    # 测试不同速度下的力/加速度比值
    velocities = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]
    
    print("📈 力与加速度比值 (F/a):")
    print("-" * 60)
    print(f"{'速度 v/c':<10} {'横向 F/a':<15} {'纵向 F/a':<15} {'比值':<10}")
    print("-" * 60)
    
    for v_ratio in velocities:
        v = v_ratio * c
        m_trans_ratio = calc.calculate_force_acceleration_ratio_coding(electron_mass, v, 'transverse')
        m_long_ratio = calc.calculate_force_acceleration_ratio_coding(electron_mass, v, 'longitudinal')
        
        print(f"{v_ratio:<10.2f} {m_trans_ratio:<15.3e} {m_long_ratio:<15.3e} {m_long_ratio/m_trans_ratio:<10.3f}")
    
    print()
    print("💡 应用意义:")
    print("   • 在粒子加速器中，纵向力需要克服 γ³ 倍的惯性")
    print("   • 这就是为什么高能粒子加速器需要越来越大的功率")
    print("   • 横向力（如磁场偏转）只需要克服 γ 倍的惯性")

def demo_visualization():
    """演示可视化"""
    print("\n" + "=" * 60)
    print("📊 可视化演示")
    print("=" * 60)
    
    calc = RelativisticPhysicsCalculator()
    electron_mass = 9.109e-31 * u.kg
    
    print("🎨 生成相对论质量随速度变化的图表...")
    
    # 生成图表
    calc.plot_relativistic_mass_visual(electron_mass)
    
    print("✅ 图表已生成！")
    print("   • 红线：纵向质量 m_long = γ³m")
    print("   • 蓝线：横向质量 m_trans = γm") 
    print("   • 黑线：静质量 m")
    print("   • 注意纵向质量在高速度下急剧增长")

if __name__ == "__main__":
    # 运行所有演示
    calc, electron_mass = demo_relativistic_mass()
    demo_force_acceleration_ratio()
    
    # 询问是否显示图表
    try:
        response = input("\n是否显示可视化图表？(y/n): ")
        if response.lower() in ['y', 'yes', '是']:
            demo_visualization()
    except KeyboardInterrupt:
        print("\n演示结束。")
    
    print("\n🎉 相对论质量计算演示完成！")
