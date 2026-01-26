#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将RelativisticPhysicsCalculator中的工具函数转换为MCP协议格式和tool格式
"""

import json
import inspect
from typing import Dict, List, Any

def extract_function_info(func) -> Dict[str, Any]:
    """提取函数信息"""
    # 获取函数签名
    sig = inspect.signature(func)
    parameters = sig.parameters
    
    # 跳过self参数
    param_info = {}
    required_params = []
    
    for name, param in parameters.items():
        if name == 'self':
            continue
            
        param_type = "number"  # 默认类型
        description = f"参数 {name}"
        
        # 根据参数名和函数名推断类型和描述
        if name in ['v', 'u_prime', 'velocity']:
            param_type = "number"
            description = f"速度，单位为米每秒(m/s)"
        elif name in ['m', 'mass']:
            param_type = "number"
            description = f"质量，单位为千克(kg)"
        elif name in ['delta_t_0', 'proper_time']:
            param_type = "number"
            description = f"固有时，单位为秒(s)"
        elif name in ['L_0', 'proper_length']:
            param_type = "number"
            description = f"固有长度，单位为米(m)"
        elif name in ['E', 'energy']:
            param_type = "number"
            description = f"能量，单位为焦耳(J)"
        elif name in ['p', 'momentum']:
            param_type = "number"
            description = f"动量，单位为千克米每秒(kg·m/s)"
        elif name in ['x', 'position']:
            param_type = "number"
            description = f"位置，单位为米(m)"
        elif name in ['t', 'time']:
            param_type = "number"
            description = f"时间，单位为秒(s)"
        elif name in ['force_direction']:
            param_type = "string"
            description = f"力的方向，可选值：'longitudinal'（纵向）或 'transverse'（横向）"
        elif name in ['v_range']:
            param_type = "array"
            description = f"速度范围，格式：[最小值, 最大值]，单位为光速的倍数"
        elif name in ['num_points']:
            param_type = "number"
            description = f"计算点数，用于数值计算"
        elif name in ['m', 'M']:
            param_type = "number"
            description = f"质量，单位为千克(kg)"
        elif name in ['r0']:
            param_type = "number"
            description = f"初始径向距离，单位为米(m)"
        elif name in ['v0_radial', 'v0_angular']:
            param_type = "number"
            description = f"初始径向/角向速度，单位为米每秒(m/s)"
        elif name in ['lambda_max']:
            param_type = "number"
            description = f"最大仿射参数值"
        elif name in ['steps']:
            param_type = "number"
            description = f"计算步数"
        
        param_info[name] = {
            "type": param_type,
            "description": description
        }
        
        # 如果参数没有默认值，则为必需参数
        if param.default == inspect.Parameter.empty:
            required_params.append(name)
    
    return {
        "parameters": param_info,
        "required": required_params
    }

def get_function_description(func_name: str) -> str:
    """根据函数名获取描述"""
    descriptions = {
        "lorentz_factor": "计算洛伦兹因子 γ = 1/√(1 - v²/c²)，用于相对论效应计算",
        "lorentz_factor_coding": "数值计算洛伦兹因子 γ = 1/√(1 - v²/c²)",
        "lorentz_factor_math": "符号推导洛伦兹因子公式",
        "plot_lorentz_factor_visual": "绘制洛伦兹因子随速度变化曲线",
        
        "time_dilation": "计算时间膨胀：坐标时 = γ × 固有时",
        "time_dilation_coding": "时间膨胀：坐标时 = γ × 固有时",
        "length_contraction": "计算长度收缩：观测长度 = 固有长度 / γ",
        "length_contraction_coding": "长度收缩：观测长度 = 固有长度 / γ",
        "relativistic_effects_math": "符号化表达时间膨胀与长度收缩",
        "plot_time_length_effects_visual": "绘制时间膨胀与长度收缩对比图",
        
        "relativistic_velocity_addition": "计算相对论速度叠加：w = (u + v) / (1 + u*v/c²)",
        "velocity_addition_coding": "相对论速度叠加：w = (u + v) / (1 + u*v/c²)",
        "velocity_addition_math": "符号推导速度叠加公式",
        "plot_velocity_addition_visual": "经典 vs 相对论速度叠加对比",
        
        "relativistic_energy": "计算相对论性总能量 E = γmc²",
        "relativistic_momentum": "计算相对论性动量 p = γmv",
        "energy_momentum_coding": "计算相对论总能量与动量",
        "energy_momentum_math": "符号化能量-动量关系",
        "plot_energy_components_visual": "绘制能量随速度变化：静能、动能、总能",
        
        "longitudinal_mass_coding": "计算纵向质量：m_long = γ³m（力平行于运动方向）",
        "transverse_mass_coding": "计算横向质量：m_trans = γm（力垂直于运动方向）",
        "relativistic_mass_math": "符号化表达纵向质量和横向质量",
        "plot_relativistic_mass_visual": "绘制纵向质量和横向质量随速度变化",
        "calculate_force_acceleration_ratio_coding": "计算力与加速度的比值（纵向或横向）",
        "relativistic_mass_comparison_coding": "比较不同质量定义",
        
        "schwarzschild_geodesic_rhs": "史瓦西度规下赤道面测地线微分方程右侧",
        "solve_black_hole_orbit_coding": "数值求解黑洞附近粒子轨道",
        "plot_black_hole_orbit_visual": "绘制黑洞周围粒子轨道",
        
        "energy_momentum_relation": "计算能量-动量关系 (E² = (pc)² + (mc²)²)，求静止质量",
        "lorentz_transformation": "执行一维洛伦兹变换 (从 S 系到 S' 系)"
    }
    
    return descriptions.get(func_name, f"执行{func_name}函数")

def generate_mcp_format() -> Dict[str, Any]:
    """生成MCP协议格式"""
    return {
        "name": "RelativisticPhysicsCalculator",
        "description": "相对论物理计算工具，提供狭义相对论和广义相对论的各种计算功能，包括洛伦兹因子、时间膨胀、长度收缩、速度叠加、能量动量关系、纵向横向质量计算等",
        "author": "@yangyajie",
        "category": "Physics",
        "tools": [
            "lorentz_factor",
            "lorentz_factor_coding", 
            "lorentz_factor_math",
            "plot_lorentz_factor_visual",
            "time_dilation",
            "time_dilation_coding",
            "length_contraction",
            "length_contraction_coding",
            "relativistic_effects_math",
            "plot_time_length_effects_visual",
            "relativistic_velocity_addition",
            "velocity_addition_coding",
            "velocity_addition_math",
            "plot_velocity_addition_visual",
            "relativistic_energy",
            "relativistic_momentum",
            "energy_momentum_coding",
            "energy_momentum_math",
            "plot_energy_components_visual",
            "longitudinal_mass_coding",
            "transverse_mass_coding",
            "relativistic_mass_math",
            "plot_relativistic_mass_visual",
            "calculate_force_acceleration_ratio_coding",
            "relativistic_mass_comparison_coding",
            "schwarzschild_geodesic_rhs",
            "solve_black_hole_orbit_coding",
            "plot_black_hole_orbit_visual",
            "energy_momentum_relation",
            "lorentz_transformation"
        ]
    }

def generate_tool_format() -> List[Dict[str, Any]]:
    """生成tool格式"""
    from RelativisticPhysicsCalculator import RelativisticPhysicsCalculator
    
    calc = RelativisticPhysicsCalculator()
    tools = []
    
    # 获取所有方法
    methods = inspect.getmembers(calc, predicate=inspect.ismethod)
    
    for method_name, method in methods:
        if method_name.startswith('_'):
            continue
            
        # 提取函数信息
        func_info = extract_function_info(method)
        description = get_function_description(method_name)
        
        tool = {
            "type": "function",
            "function": {
                "name": method_name,
                "description": description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": func_info["parameters"],
                    "required": func_info["required"],
                    "additionalProperties": False
                }
            }
        }
        
        tools.append(tool)
    
    return tools

def save_formats():
    """保存两种格式到文件"""
    # 生成MCP格式
    mcp_format = generate_mcp_format()
    
    # 生成tool格式
    tool_format = generate_tool_format()
    
    # 保存MCP格式
    with open('relativistic_physics_mcp_format.json', 'w', encoding='utf-8') as f:
        json.dump(mcp_format, f, ensure_ascii=False, indent=2)
    
    # 保存tool格式
    with open('relativistic_physics_tool_format.json', 'w', encoding='utf-8') as f:
        json.dump({"tools": tool_format}, f, ensure_ascii=False, indent=2)
    
    print("✅ MCP格式已保存到: relativistic_physics_mcp_format.json")
    print("✅ Tool格式已保存到: relativistic_physics_tool_format.json")
    
    return mcp_format, tool_format

def print_formats():
    """打印两种格式"""
    mcp_format, tool_format = save_formats()
    
    print("\n" + "="*60)
    print("MCP协议格式:")
    print("="*60)
    print(json.dumps(mcp_format, ensure_ascii=False, indent=2))
    
    print("\n" + "="*60)
    print("Tool格式 (前3个工具作为示例):")
    print("="*60)
    print(json.dumps({"tools": tool_format[:3]}, ensure_ascii=False, indent=2))
    
    print(f"\n总共生成了 {len(tool_format)} 个工具函数")

if __name__ == "__main__":
    print("🔧 正在转换RelativisticPhysicsCalculator为MCP格式...")
    print_formats()
