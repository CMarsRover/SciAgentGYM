#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例：展示如何使用生成的MCP格式和tool格式
"""

import json
from RelativisticPhysicsCalculator import RelativisticPhysicsCalculator
from astropy import units as u
from astropy.constants import c

def demonstrate_mcp_usage():
    """演示MCP格式的使用"""
    print("🎯 MCP格式使用演示")
    print("=" * 60)
    
    # 加载MCP格式
    with open('relativistic_physics_mcp_format.json', 'r', encoding='utf-8') as f:
        mcp_config = json.load(f)
    
    print(f"工具名称: {mcp_config['name']}")
    print(f"工具描述: {mcp_config['description']}")
    print(f"作者: {mcp_config['author']}")
    print(f"类别: {mcp_config['category']}")
    print(f"可用工具数量: {len(mcp_config['tools'])}")
    
    print("\n📋 可用工具列表:")
    for i, tool in enumerate(mcp_config['tools'], 1):
        print(f"  {i:2d}. {tool}")
    
    return mcp_config

def demonstrate_tool_usage():
    """演示tool格式的使用"""
    print("\n🔧 Tool格式使用演示")
    print("=" * 60)
    
    # 加载tool格式
    with open('relativistic_physics_tool_format.json', 'r', encoding='utf-8') as f:
        tool_config = json.load(f)
    
    print(f"工具函数总数: {len(tool_config['tools'])}")
    
    # 创建计算器实例
    calc = RelativisticPhysicsCalculator()
    
    # 演示几个关键工具的使用
    print("\n📊 工具使用示例:")
    
    # 示例1: 洛伦兹因子计算
    print("\n1. 洛伦兹因子计算:")
    v = 0.8 * c.value  # 0.8倍光速
    gamma = calc.lorentz_factor(v)
    print(f"   速度: {v/c.value:.2f}c")
    print(f"   洛伦兹因子: {gamma:.3f}")
    
    # 示例2: 纵向质量和横向质量
    print("\n2. 纵向质量和横向质量:")
    electron_mass = 9.109e-31  # 电子质量
    m_long = calc.longitudinal_mass_coding(electron_mass, v)
    m_trans = calc.transverse_mass_coding(electron_mass, v)
    print(f"   静质量: {electron_mass:.3e} kg")
    print(f"   横向质量: {m_trans:.3e} kg")
    print(f"   纵向质量: {m_long:.3e} kg")
    print(f"   纵向/横向质量比: {m_long/m_trans:.3f}")
    
    # 示例3: 能量动量关系
    print("\n3. 能量动量关系:")
    energy_momentum = calc.energy_momentum_coding(electron_mass, v)
    print(f"   总能量: {energy_momentum['total_energy']:.3f}")
    print(f"   静能: {energy_momentum['rest_energy']:.3f}")
    print(f"   动能: {energy_momentum['kinetic_energy']:.3f}")
    
    return tool_config

def demonstrate_api_integration():
    """演示API集成"""
    print("\n🔌 API集成演示")
    print("=" * 60)
    
    # 模拟API调用
    api_calls = [
        {
            "function": "lorentz_factor",
            "parameters": {"v": 0.9 * c.value},
            "description": "计算0.9c速度下的洛伦兹因子"
        },
        {
            "function": "longitudinal_mass_coding",
            "parameters": {"m": 9.109e-31, "v": 0.9 * c.value},
            "description": "计算电子在0.9c下的纵向质量"
        },
        {
            "function": "transverse_mass_coding", 
            "parameters": {"m": 9.109e-31, "v": 0.9 * c.value},
            "description": "计算电子在0.9c下的横向质量"
        },
        {
            "function": "calculate_force_acceleration_ratio_coding",
            "parameters": {"m": 9.109e-31, "v": 0.9 * c.value, "force_direction": "longitudinal"},
            "description": "计算纵向力与加速度比值"
        }
    ]
    
    calc = RelativisticPhysicsCalculator()
    
    print("📡 模拟API调用:")
    for i, call in enumerate(api_calls, 1):
        print(f"\n{i}. {call['description']}")
        print(f"   函数: {call['function']}")
        print(f"   参数: {call['parameters']}")
        
        # 执行函数调用
        try:
            func = getattr(calc, call['function'])
            result = func(**call['parameters'])
            print(f"   结果: {result}")
        except Exception as e:
            print(f"   错误: {e}")

def create_integration_example():
    """创建集成示例"""
    print("\n📝 创建集成示例")
    print("=" * 60)
    
    # 创建一个完整的集成示例
    integration_example = {
        "tool_name": "RelativisticPhysicsCalculator",
        "version": "1.0.0",
        "description": "相对论物理计算工具集成示例",
        "mcp_format": "relativistic_physics_mcp_format.json",
        "tool_format": "relativistic_physics_tool_format.json",
        "usage_example": {
            "import_statement": "from RelativisticPhysicsCalculator import RelativisticPhysicsCalculator",
            "instantiation": "calc = RelativisticPhysicsCalculator()",
            "key_functions": [
                "calc.lorentz_factor(v)",
                "calc.longitudinal_mass_coding(m, v)", 
                "calc.transverse_mass_coding(m, v)",
                "calc.energy_momentum_coding(m, v)",
                "calc.calculate_force_acceleration_ratio_coding(m, v, direction)"
            ]
        }
    }
    
    # 保存集成示例
    with open('integration_example.json', 'w', encoding='utf-8') as f:
        json.dump(integration_example, f, ensure_ascii=False, indent=2)
    
    print("✅ 集成示例已保存到: integration_example.json")
    
    return integration_example

def main():
    """主函数"""
    print("🚀 RelativisticPhysicsCalculator MCP格式使用演示")
    print("=" * 80)
    
    # 演示MCP格式
    mcp_config = demonstrate_mcp_usage()
    
    # 演示tool格式
    tool_config = demonstrate_tool_usage()
    
    # 演示API集成
    demonstrate_api_integration()
    
    # 创建集成示例
    integration_example = create_integration_example()
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("\n📁 生成的文件:")
    print("   - relativistic_physics_mcp_format.json (MCP协议格式)")
    print("   - relativistic_physics_tool_format.json (Tool格式)")
    print("   - integration_example.json (集成示例)")
    
    print("\n🎯 关键特性:")
    print("   - 支持22个相对论物理计算函数")
    print("   - 包含纵向质量和横向质量计算")
    print("   - 提供完整的参数类型和描述")
    print("   - 支持MCP协议和标准tool格式")

if __name__ == "__main__":
    main()
