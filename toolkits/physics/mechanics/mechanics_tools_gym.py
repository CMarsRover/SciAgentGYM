#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mechanics 工具注册模块
使用 gym.tool.EnvironmentTool 为 mechanics 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool, convert_to_json_serializable
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.physics.mechanics.circular_motion_solver_107 import *  # 动态导入
# from toolkits.physics.mechanics.circular_motion_solver_6415 import *  # 动态导入
# from toolkits.physics.mechanics.friction_dynamics_solver_8486 import *  # 动态导入
# from toolkits.physics.mechanics.mechanical_energy_solver_7368 import *  # 动态导入
# from toolkits.physics.mechanics.pulley_friction_system_solver_7229 import *  # 动态导入
# from toolkits.physics.mechanics.pulley_system_solver_6252 import *  # 动态导入
# from toolkits.physics.mechanics.pulley_system_solver_6557 import *  # 动态导入
# from toolkits.physics.mechanics.pulley_system_solver_6851 import *  # 动态导入
# from toolkits.physics.mechanics.pulley_system_solver_7004 import *  # 动态导入
# from toolkits.physics.mechanics.static_equilibrium_solver_6466 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="calculate_tension_circular_motion")
class CalculateTensionCircularMotionTool(EnvironmentTool):
    """计算两个质点在水平面上做匀速圆周运动时绳索上的张力。该函数基于牛顿第二定律和向心力原理，计算连接两个质点的绳索上的张力。在匀速圆周运动中，向心力由绳索的张力提供。"""
    
    name = "calculate_tension_circular_motion"
    description = "计算两个质点在水平面上做匀速圆周运动时绳索上的张力。该函数基于牛顿第二定律和向心力原理，计算连接两个质点的绳索上的张力。在匀速圆周运动中，向心力由绳索的张力提供。"
    arguments = {
        "m1": {"type": "number", "description": "第一个质点的质量，单位：kg"},
        "m2": {"type": "number", "description": "第二个质点的质量，单位：kg"},
        "L1": {"type": "number", "description": "第一段绳索的长度（从固定点O到质点m1），单位：m"},
        "L2": {"type": "number", "description": "第二段绳索的长度（从质点m1到质点m2），单位：m"},
        "omega": {"type": "number", "description": "角速度，单位：rad/s"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_tension_circular_motion 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            L1 = args.get("L1")
            if L1 is None:
                return Observation(self.name, "错误: 缺少必需参数 L1")
            L2 = args.get("L2")
            if L2 is None:
                return Observation(self.name, "错误: 缺少必需参数 L2")
            omega = args.get("omega")
            if omega is None:
                return Observation(self.name, "错误: 缺少必需参数 omega")
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_107 import calculate_tension_circular_motion
            
            # 调用函数
            result = calculate_tension_circular_motion(m1, m2, L1, L2, omega)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="simulate_circular_motion")
class SimulateCircularMotionTool(EnvironmentTool):
    """模拟两个质点在水平面上做匀速圆周运动的运动轨迹。"""
    
    name = "simulate_circular_motion"
    description = "模拟两个质点在水平面上做匀速圆周运动的运动轨迹。"
    arguments = {
        "m1": {"type": "number", "description": "第一个质点的质量，单位：kg"},
        "m2": {"type": "number", "description": "第二个质点的质量，单位：kg"},
        "L1": {"type": "number", "description": "第一段绳索的长度，单位：m"},
        "L2": {"type": "number", "description": "第二段绳索的长度，单位：m"},
        "omega": {"type": "number", "description": "角速度，单位：rad/s"},
        "duration": {"type": "number", "description": "模拟持续时间，单位：s，默认为10秒"},
        "fps": {"type": "integer", "description": "每秒帧数，默认为30"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 simulate_circular_motion 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            L1 = args.get("L1")
            if L1 is None:
                return Observation(self.name, "错误: 缺少必需参数 L1")
            L2 = args.get("L2")
            if L2 is None:
                return Observation(self.name, "错误: 缺少必需参数 L2")
            omega = args.get("omega")
            if omega is None:
                return Observation(self.name, "错误: 缺少必需参数 omega")
            duration = args.get("duration", None)
            fps = args.get("fps", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_107 import simulate_circular_motion
            
            # 调用函数
            result = simulate_circular_motion(m1, m2, L1, L2, omega, duration, fps)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="analyze_forces")
class AnalyzeForcesTool(EnvironmentTool):
    """分析圆周运动中的力学关系，计算张力并验证力平衡。"""
    
    name = "analyze_forces"
    description = "分析圆周运动中的力学关系，计算张力并验证力平衡。"
    arguments = {
        "m1": {"type": "number", "description": "第一个质点的质量，单位：kg"},
        "m2": {"type": "number", "description": "第二个质点的质量，单位：kg"},
        "L1": {"type": "number", "description": "第一段绳索的长度，单位：m"},
        "L2": {"type": "number", "description": "第二段绳索的长度，单位：m"},
        "omega": {"type": "number", "description": "角速度，单位：rad/s"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_forces 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            L1 = args.get("L1")
            if L1 is None:
                return Observation(self.name, "错误: 缺少必需参数 L1")
            L2 = args.get("L2")
            if L2 is None:
                return Observation(self.name, "错误: 缺少必需参数 L2")
            omega = args.get("omega")
            if omega is None:
                return Observation(self.name, "错误: 缺少必需参数 omega")
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_107 import analyze_forces
            
            # 调用函数
            result = analyze_forces(m1, m2, L1, L2, omega)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_motion")
class VisualizeMotionTool(EnvironmentTool):
    """可视化两个质点的圆周运动。"""
    
    name = "visualize_motion"
    description = "可视化两个质点的圆周运动。"
    arguments = {
        "m1": {"type": "number", "description": "第一个质点的质量，单位：kg"},
        "m2": {"type": "number", "description": "第二个质点的质量，单位：kg"},
        "L1": {"type": "number", "description": "第一段绳索的长度，单位：m"},
        "L2": {"type": "number", "description": "第二段绳索的长度，单位：m"},
        "omega": {"type": "number", "description": "角速度，单位：rad/s"},
        "duration": {"type": "number", "description": "模拟持续时间，单位：s，默认为5秒"},
        "fps": {"type": "integer", "description": "每秒帧数，默认为30"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_motion 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            L1 = args.get("L1")
            if L1 is None:
                return Observation(self.name, "错误: 缺少必需参数 L1")
            L2 = args.get("L2")
            if L2 is None:
                return Observation(self.name, "错误: 缺少必需参数 L2")
            omega = args.get("omega")
            if omega is None:
                return Observation(self.name, "错误: 缺少必需参数 omega")
            duration = args.get("duration", None)
            fps = args.get("fps", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_107 import visualize_motion
            
            # 调用函数
            result = visualize_motion(m1, m2, L1, L2, omega, duration, fps)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_pulley_system_acceleration")
class CalculatePulleySystemAccelerationTool(EnvironmentTool):
    """计算由不可伸长绳索连接的两个质量物体组成的滑轮系统的加速度。"""
    
    name = "calculate_pulley_system_acceleration"
    description = "计算由不可伸长绳索连接的两个质量物体组成的滑轮系统的加速度。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "external_force": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""},
        "angle": {"type": "number", "description": ""},
        "friction_coef": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_pulley_system_acceleration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            external_force = args.get("external_force")
            if external_force is None:
                return Observation(self.name, "错误: 缺少必需参数 external_force")
            g = args.get("g", None)
            angle = args.get("angle", None)
            friction_coef = args.get("friction_coef", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6252 import calculate_pulley_system_acceleration
            
            # 调用函数
            result = calculate_pulley_system_acceleration(m1, m2, external_force, g, angle, friction_coef)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_tension")
class CalculateTensionTool(EnvironmentTool):
    """计算滑轮系统中连接两个质量的绳索的张力。"""
    
    name = "calculate_tension"
    description = "计算滑轮系统中连接两个质量的绳索的张力。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "acceleration": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_tension 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            acceleration = args.get("acceleration")
            if acceleration is None:
                return Observation(self.name, "错误: 缺少必需参数 acceleration")
            g = args.get("g", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6252 import calculate_tension
            
            # 调用函数
            result = calculate_tension(m1, m2, acceleration, g)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_pulley_system")
class VisualizePulleySystemTool(EnvironmentTool):
    """可视化滑轮系统，包括力和加速度。"""
    
    name = "visualize_pulley_system"
    description = "可视化滑轮系统，包括力和加速度。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "external_force": {"type": "number", "description": ""},
        "acceleration": {"type": "number", "description": ""},
        "tension": {"type": "number", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            external_force = args.get("external_force")
            if external_force is None:
                return Observation(self.name, "错误: 缺少必需参数 external_force")
            acceleration = args.get("acceleration")
            if acceleration is None:
                return Observation(self.name, "错误: 缺少必需参数 acceleration")
            tension = args.get("tension")
            if tension is None:
                return Observation(self.name, "错误: 缺少必需参数 tension")
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6252 import visualize_pulley_system
            
            # 调用函数
            result = visualize_pulley_system(m1, m2, external_force, acceleration, tension, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="analyze_pulley_system")
class AnalyzePulleySystemTool(EnvironmentTool):
    """对滑轮系统进行完整分析。"""
    
    name = "analyze_pulley_system"
    description = "对滑轮系统进行完整分析。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "external_force": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""},
        "angle": {"type": "number", "description": ""},
        "friction_coef": {"type": "number", "description": ""},
        "visualize": {"type": "boolean", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            external_force = args.get("external_force")
            if external_force is None:
                return Observation(self.name, "错误: 缺少必需参数 external_force")
            g = args.get("g", None)
            angle = args.get("angle", None)
            friction_coef = args.get("friction_coef", None)
            visualize = args.get("visualize", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6252 import analyze_pulley_system
            
            # 调用函数
            result = analyze_pulley_system(m1, m2, external_force, g, angle, friction_coef, visualize)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_conical_pendulum_velocity")
class CalculateConicalPendulumVelocityTool(EnvironmentTool):
    """计算圆锥摆（在水平圆周上运动的物体）的速度。"""
    
    name = "calculate_conical_pendulum_velocity"
    description = "计算圆锥摆（在水平圆周上运动的物体）的速度。"
    arguments = {
        "length": {"type": "number", "description": "绳子或杆的长度（米）"},
        "angle": {"type": "number", "description": "绳子与垂直方向的夹角（弧度）"},
        "gravity": {"type": "number", "description": "重力加速度（米/秒²）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_conical_pendulum_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            length = args.get("length")
            if length is None:
                return Observation(self.name, "错误: 缺少必需参数 length")
            angle = args.get("angle")
            if angle is None:
                return Observation(self.name, "错误: 缺少必需参数 angle")
            gravity = args.get("gravity", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_6415 import calculate_conical_pendulum_velocity
            
            # 调用函数
            result = calculate_conical_pendulum_velocity(length, angle, gravity)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_circular_motion_parameters")
class CalculateCircularMotionParametersTool(EnvironmentTool):
    """计算匀速圆周运动物体的各种参数。"""
    
    name = "calculate_circular_motion_parameters"
    description = "计算匀速圆周运动物体的各种参数。"
    arguments = {
        "velocity": {"type": "number", "description": "物体的切向速度（米/秒）"},
        "radius": {"type": "number", "description": "圆周路径的半径（米）"},
        "mass": {"type": "number", "description": "物体的质量（千克），默认为1.0千克"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_circular_motion_parameters 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            velocity = args.get("velocity")
            if velocity is None:
                return Observation(self.name, "错误: 缺少必需参数 velocity")
            radius = args.get("radius")
            if radius is None:
                return Observation(self.name, "错误: 缺少必需参数 radius")
            mass = args.get("mass", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_6415 import calculate_circular_motion_parameters
            
            # 调用函数
            result = calculate_circular_motion_parameters(velocity, radius, mass)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_tension_force")
class CalculateTensionForceTool(EnvironmentTool):
    """计算圆锥摆中绳子或杆的张力。"""
    
    name = "calculate_tension_force"
    description = "计算圆锥摆中绳子或杆的张力。"
    arguments = {
        "mass": {"type": "number", "description": "物体的质量（千克）"},
        "gravity": {"type": "number", "description": "重力加速度（米/秒²）"},
        "angle": {"type": "number", "description": "绳子与垂直方向的夹角（弧度）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_tension_force 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            gravity = args.get("gravity")
            if gravity is None:
                return Observation(self.name, "错误: 缺少必需参数 gravity")
            angle = args.get("angle")
            if angle is None:
                return Observation(self.name, "错误: 缺少必需参数 angle")
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_6415 import calculate_tension_force
            
            # 调用函数
            result = calculate_tension_force(mass, gravity, angle)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_conical_pendulum")
class VisualizeConicalPendulumTool(EnvironmentTool):
    """可视化圆锥摆系统及其关键参数。"""
    
    name = "visualize_conical_pendulum"
    description = "可视化圆锥摆系统及其关键参数。"
    arguments = {
        "length": {"type": "number", "description": "绳子或杆的长度（米）"},
        "angle": {"type": "number", "description": "绳子与垂直方向的夹角（弧度）"},
        "mass": {"type": "number", "description": "物体的质量（千克），默认为1.0千克"},
        "gravity": {"type": "number", "description": "重力加速度（米/秒²），默认为9.8米/秒²"},
        "save_path": {"type": "string", "description": "保存图像的路径，如果为None，则不保存图像"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_conical_pendulum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            length = args.get("length")
            if length is None:
                return Observation(self.name, "错误: 缺少必需参数 length")
            angle = args.get("angle")
            if angle is None:
                return Observation(self.name, "错误: 缺少必需参数 angle")
            mass = args.get("mass", None)
            gravity = args.get("gravity", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.circular_motion_solver_6415 import visualize_conical_pendulum
            
            # 调用函数
            result = visualize_conical_pendulum(length, angle, mass, gravity, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_force_from_torque")
class CalculateForceFromTorqueTool(EnvironmentTool):
    """基于力矩平衡计算静力系统中未知力。"""
    
    name = "calculate_force_from_torque"
    description = "基于力矩平衡计算静力系统中未知力。"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""},
        "pivot_point": {"type": "array", "description": ""},
        "force_points": {"type": "array", "description": ""},
        "weight_point": {"type": "array", "description": ""},
        "precision": {"type": "integer", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_force_from_torque 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            gravity = args.get("gravity", None)
            pivot_point = args.get("pivot_point")
            if pivot_point is None:
                return Observation(self.name, "错误: 缺少必需参数 pivot_point")
            force_points = args.get("force_points")
            if force_points is None:
                return Observation(self.name, "错误: 缺少必需参数 force_points")
            weight_point = args.get("weight_point")
            if weight_point is None:
                return Observation(self.name, "错误: 缺少必需参数 weight_point")
            precision = args.get("precision", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.static_equilibrium_solver_6466 import calculate_force_from_torque
            
            # 调用函数
            result = calculate_force_from_torque(mass, gravity, pivot_point, force_points, weight_point, precision)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_pushup")
class VisualizePushupTool(EnvironmentTool):
    """可视化俯卧撑受力与距离示意图。"""
    
    name = "visualize_pushup"
    description = "可视化俯卧撑受力与距离示意图。"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "cg_distance": {"type": "number", "description": ""},
        "hand_distance": {"type": "number", "description": ""},
        "force": {"type": "number", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_pushup 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            cg_distance = args.get("cg_distance")
            if cg_distance is None:
                return Observation(self.name, "错误: 缺少必需参数 cg_distance")
            hand_distance = args.get("hand_distance")
            if hand_distance is None:
                return Observation(self.name, "错误: 缺少必需参数 hand_distance")
            force = args.get("force", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.static_equilibrium_solver_6466 import visualize_pushup
            
            # 调用函数
            result = visualize_pushup(mass, cg_distance, hand_distance, force, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_static_equilibrium")
class VisualizeStaticEquilibriumTool(EnvironmentTool):
    """可视化静力平衡系统，包含支点、力点与重心。"""
    
    name = "visualize_static_equilibrium"
    description = "可视化静力平衡系统，包含支点、力点与重心。"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "pivot_point": {"type": "array", "description": ""},
        "force_points": {"type": "array", "description": ""},
        "weight_point": {"type": "array", "description": ""},
        "forces": {"type": "object", "description": ""},
        "title": {"type": "string", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_static_equilibrium 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            pivot_point = args.get("pivot_point")
            if pivot_point is None:
                return Observation(self.name, "错误: 缺少必需参数 pivot_point")
            force_points = args.get("force_points")
            if force_points is None:
                return Observation(self.name, "错误: 缺少必需参数 force_points")
            weight_point = args.get("weight_point")
            if weight_point is None:
                return Observation(self.name, "错误: 缺少必需参数 weight_point")
            forces = args.get("forces", None)
            title = args.get("title", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.static_equilibrium_solver_6466 import visualize_static_equilibrium
            
            # 调用函数
            result = visualize_static_equilibrium(mass, pivot_point, force_points, weight_point, forces, title, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_pulley_system_acceleration")
class CalculatePulleySystemAccelerationTool(EnvironmentTool):
    """计算含外力与角度、两质量、可含摩擦的滑轮系统的加速度、张力、法向力与摩擦力。"""
    
    name = "calculate_pulley_system_acceleration"
    description = "计算含外力与角度、两质量、可含摩擦的滑轮系统的加速度、张力、法向力与摩擦力。"
    arguments = {
        "applied_force": {"type": "number", "description": ""},
        "applied_force_angle": {"type": "number", "description": ""},
        "mass_1": {"type": "number", "description": ""},
        "mass_2": {"type": "number", "description": ""},
        "friction_coefficient": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_pulley_system_acceleration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            applied_force = args.get("applied_force")
            if applied_force is None:
                return Observation(self.name, "错误: 缺少必需参数 applied_force")
            applied_force_angle = args.get("applied_force_angle")
            if applied_force_angle is None:
                return Observation(self.name, "错误: 缺少必需参数 applied_force_angle")
            mass_1 = args.get("mass_1")
            if mass_1 is None:
                return Observation(self.name, "错误: 缺少必需参数 mass_1")
            mass_2 = args.get("mass_2")
            if mass_2 is None:
                return Observation(self.name, "错误: 缺少必需参数 mass_2")
            friction_coefficient = args.get("friction_coefficient", None)
            gravity = args.get("gravity", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6557 import calculate_pulley_system_acceleration
            
            # 调用函数
            result = calculate_pulley_system_acceleration(applied_force, applied_force_angle, mass_1, mass_2, friction_coefficient, gravity)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_pulley_system")
class VisualizePulleySystemTool(EnvironmentTool):
    """可视化滑轮系统及计算结果。"""
    
    name = "visualize_pulley_system"
    description = "可视化滑轮系统及计算结果。"
    arguments = {
        "applied_force": {"type": "number", "description": ""},
        "applied_force_angle": {"type": "number", "description": ""},
        "mass_1": {"type": "number", "description": ""},
        "mass_2": {"type": "number", "description": ""},
        "results": {"type": "object", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            applied_force = args.get("applied_force")
            if applied_force is None:
                return Observation(self.name, "错误: 缺少必需参数 applied_force")
            applied_force_angle = args.get("applied_force_angle")
            if applied_force_angle is None:
                return Observation(self.name, "错误: 缺少必需参数 applied_force_angle")
            mass_1 = args.get("mass_1")
            if mass_1 is None:
                return Observation(self.name, "错误: 缺少必需参数 mass_1")
            mass_2 = args.get("mass_2")
            if mass_2 is None:
                return Observation(self.name, "错误: 缺少必需参数 mass_2")
            results = args.get("results")
            if results is None:
                return Observation(self.name, "错误: 缺少必需参数 results")
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6557 import visualize_pulley_system
            
            # 调用函数
            result = visualize_pulley_system(applied_force, applied_force_angle, mass_1, mass_2, results, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="solve_connected_masses_problem")
class SolveConnectedMassesProblemTool(EnvironmentTool):
    """通用求解相连质量的滑轮系统，支持质量比或直接给出质量。"""
    
    name = "solve_connected_masses_problem"
    description = "通用求解相连质量的滑轮系统，支持质量比或直接给出质量。"
    arguments = {
        "force": {"type": "number", "description": ""},
        "force_angle": {"type": "number", "description": ""},
        "mass_ratio": {"type": "string", "description": ""},
        "mass_value": {"type": "number", "description": ""},
        "friction_coefficient": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""},
        "visualize": {"type": "boolean", "description": ""},
        "save_path": {"type": "string", "description": ""},
        "round_digits": {"type": "integer", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_connected_masses_problem 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            force = args.get("force")
            if force is None:
                return Observation(self.name, "错误: 缺少必需参数 force")
            force_angle = args.get("force_angle")
            if force_angle is None:
                return Observation(self.name, "错误: 缺少必需参数 force_angle")
            mass_ratio = args.get("mass_ratio")
            if mass_ratio is None:
                return Observation(self.name, "错误: 缺少必需参数 mass_ratio")
            mass_value = args.get("mass_value", None)
            friction_coefficient = args.get("friction_coefficient", None)
            gravity = args.get("gravity", None)
            visualize = args.get("visualize", None)
            save_path = args.get("save_path", None)
            round_digits = args.get("round_digits", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6557 import solve_connected_masses_problem
            
            # 调用函数
            result = solve_connected_masses_problem(force, force_angle, mass_ratio, mass_value, friction_coefficient, gravity, visualize, save_path, round_digits)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_forces_on_inclined_block")
class CalculateForcesOnInclinedBlockTool(EnvironmentTool):
    """计算斜面上物体的受力分量、法向力与摩擦力。"""
    
    name = "calculate_forces_on_inclined_block"
    description = "计算斜面上物体的受力分量、法向力与摩擦力。"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "mu_k": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_forces_on_inclined_block 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            mu_k = args.get("mu_k", None)
            g = args.get("g", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6851 import calculate_forces_on_inclined_block
            
            # 调用函数
            result = calculate_forces_on_inclined_block(mass, angle_deg, mu_k, g)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="solve_pulley_system_with_incline")
class SolvePulleySystemWithInclineTool(EnvironmentTool):
    """求解含斜面的恒速滑轮系统的未知质量。"""
    
    name = "solve_pulley_system_with_incline"
    description = "求解含斜面的恒速滑轮系统的未知质量。"
    arguments = {
        "m2": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "mu_k": {"type": "number", "description": ""},
        "constant_speed": {"type": "boolean", "description": ""},
        "g": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_pulley_system_with_incline 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            mu_k = args.get("mu_k")
            if mu_k is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_k")
            constant_speed = args.get("constant_speed", None)
            g = args.get("g", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6851 import solve_pulley_system_with_incline
            
            # 调用函数
            result = solve_pulley_system_with_incline(m2, angle_deg, mu_k, constant_speed, g)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="analyze_pulley_system")
class AnalyzePulleySystemTool(EnvironmentTool):
    """分析滑轮-斜面系统动力学并给出受力与运动状态。"""
    
    name = "analyze_pulley_system"
    description = "分析滑轮-斜面系统动力学并给出受力与运动状态。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "mu_k": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            mu_k = args.get("mu_k")
            if mu_k is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_k")
            g = args.get("g", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6851 import analyze_pulley_system
            
            # 调用函数
            result = analyze_pulley_system(m1, m2, angle_deg, mu_k, g)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_pulley_system")
class VisualizePulleySystemTool(EnvironmentTool):
    """可视化滑轮-斜面系统与力矢量图。"""
    
    name = "visualize_pulley_system"
    description = "可视化滑轮-斜面系统与力矢量图。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "mu_k": {"type": "number", "description": ""},
        "show_forces": {"type": "boolean", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            mu_k = args.get("mu_k")
            if mu_k is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_k")
            show_forces = args.get("show_forces", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_6851 import visualize_pulley_system
            
            # 调用函数
            result = visualize_pulley_system(m1, m2, angle_deg, mu_k, show_forces, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_tension")
class CalculateTensionTool(EnvironmentTool):
    """计算滑轮系统中绳索/绳子的张力。"""
    
    name = "calculate_tension"
    description = "计算滑轮系统中绳索/绳子的张力。"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "acceleration": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""},
        "angle": {"type": "number", "description": ""},
        "friction_coef": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_tension 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            acceleration = args.get("acceleration")
            if acceleration is None:
                return Observation(self.name, "错误: 缺少必需参数 acceleration")
            gravity = args.get("gravity", None)
            angle = args.get("angle", None)
            friction_coef = args.get("friction_coef", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_7004 import calculate_tension
            
            # 调用函数
            result = calculate_tension(mass, acceleration, gravity, angle, friction_coef)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_acceleration_in_pulley_system")
class CalculateAccelerationInPulleySystemTool(EnvironmentTool):
    """计算桌面物体与悬挂质量的滑轮系统加速度(含μ、效率)。"""
    
    name = "calculate_acceleration_in_pulley_system"
    description = "计算桌面物体与悬挂质量的滑轮系统加速度(含μ、效率)。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""},
        "friction_coef": {"type": "number", "description": ""},
        "pulley_efficiency": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_acceleration_in_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            gravity = args.get("gravity", None)
            friction_coef = args.get("friction_coef", None)
            pulley_efficiency = args.get("pulley_efficiency", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_7004 import calculate_acceleration_in_pulley_system
            
            # 调用函数
            result = calculate_acceleration_in_pulley_system(m1, m2, gravity, friction_coef, pulley_efficiency)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_mass_from_acceleration")
class CalculateMassFromAccelerationTool(EnvironmentTool):
    """根据已知质量及其加速度计算未知质量(阿特伍德机推导)。"""
    
    name = "calculate_mass_from_acceleration"
    description = "根据已知质量及其加速度计算未知质量(阿特伍德机推导)。"
    arguments = {
        "known_mass": {"type": "number", "description": ""},
        "known_acceleration": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""},
        "pulley_efficiency": {"type": "number", "description": ""},
        "friction_coef": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_mass_from_acceleration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            known_mass = args.get("known_mass")
            if known_mass is None:
                return Observation(self.name, "错误: 缺少必需参数 known_mass")
            known_acceleration = args.get("known_acceleration")
            if known_acceleration is None:
                return Observation(self.name, "错误: 缺少必需参数 known_acceleration")
            gravity = args.get("gravity", None)
            pulley_efficiency = args.get("pulley_efficiency", None)
            friction_coef = args.get("friction_coef", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_7004 import calculate_mass_from_acceleration
            
            # 调用函数
            result = calculate_mass_from_acceleration(known_mass, known_acceleration, gravity, pulley_efficiency, friction_coef)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_pulley_system")
class VisualizePulleySystemTool(EnvironmentTool):
    """可视化两质量与加速度的滑轮系统。"""
    
    name = "visualize_pulley_system"
    description = "可视化两质量与加速度的滑轮系统。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "a1": {"type": "number", "description": ""},
        "a2": {"type": "number", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            a1 = args.get("a1")
            if a1 is None:
                return Observation(self.name, "错误: 缺少必需参数 a1")
            a2 = args.get("a2", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_system_solver_7004 import visualize_pulley_system
            
            # 调用函数
            result = visualize_pulley_system(m1, m2, a1, a2, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_forces_in_pulley_system")
class CalculateForcesInPulleySystemTool(EnvironmentTool):
    """计算带两质量的滑轮系统受力，含张力/法向/摩擦与状态。"""
    
    name = "calculate_forces_in_pulley_system"
    description = "计算带两质量的滑轮系统受力，含张力/法向/摩擦与状态。"
    arguments = {
        "m1": {"type": "number", "description": ""},
        "m2": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "mu_s": {"type": "number", "description": ""},
        "mu_k": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""},
        "is_moving": {"type": "boolean", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_forces_in_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m1 = args.get("m1")
            if m1 is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            m2 = args.get("m2")
            if m2 is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            mu_s = args.get("mu_s", None)
            mu_k = args.get("mu_k", None)
            g = args.get("g", None)
            is_moving = args.get("is_moving", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_friction_system_solver_7229 import calculate_forces_in_pulley_system
            
            # 调用函数
            result = calculate_forces_in_pulley_system(m1, m2, angle_deg, mu_s, mu_k, g, is_moving)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="analyze_pulley_system_equilibrium")
class AnalyzePulleySystemEquilibriumTool(EnvironmentTool):
    """分析M=因子*m的斜面-滑轮系统的静力平衡与所需μ_s。"""
    
    name = "analyze_pulley_system_equilibrium"
    description = "分析M=因子*m的斜面-滑轮系统的静力平衡与所需μ_s。"
    arguments = {
        "m": {"type": "number", "description": ""},
        "M_factor": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "mu_s": {"type": "number", "description": ""},
        "mu_k": {"type": "number", "description": ""},
        "g": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_pulley_system_equilibrium 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m = args.get("m")
            if m is None:
                return Observation(self.name, "错误: 缺少必需参数 m")
            M_factor = args.get("M_factor")
            if M_factor is None:
                return Observation(self.name, "错误: 缺少必需参数 M_factor")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            mu_s = args.get("mu_s")
            if mu_s is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_s")
            mu_k = args.get("mu_k", None)
            g = args.get("g", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_friction_system_solver_7229 import analyze_pulley_system_equilibrium
            
            # 调用函数
            result = analyze_pulley_system_equilibrium(m, M_factor, angle_deg, mu_s, mu_k, g)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_pulley_system")
class VisualizePulleySystemTool(EnvironmentTool):
    """可视化滑轮-斜面系统及受力。"""
    
    name = "visualize_pulley_system"
    description = "可视化滑轮-斜面系统及受力。"
    arguments = {
        "m": {"type": "number", "description": ""},
        "M": {"type": "number", "description": ""},
        "angle_deg": {"type": "number", "description": ""},
        "results": {"type": "object", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_pulley_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            m = args.get("m")
            if m is None:
                return Observation(self.name, "错误: 缺少必需参数 m")
            M = args.get("M")
            if M is None:
                return Observation(self.name, "错误: 缺少必需参数 M")
            angle_deg = args.get("angle_deg")
            if angle_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_deg")
            results = args.get("results")
            if results is None:
                return Observation(self.name, "错误: 缺少必需参数 results")
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.pulley_friction_system_solver_7229 import visualize_pulley_system
            
            # 调用函数
            result = visualize_pulley_system(m, M, angle_deg, results, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_kinetic_energy")
class CalculateKineticEnergyTool(EnvironmentTool):
    """计算动能"""
    
    name = "calculate_kinetic_energy"
    description = "计算动能"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "velocity": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_kinetic_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            velocity = args.get("velocity")
            if velocity is None:
                return Observation(self.name, "错误: 缺少必需参数 velocity")
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.mechanical_energy_solver_7368 import calculate_kinetic_energy
            
            # 调用函数
            result = calculate_kinetic_energy(mass, velocity)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_gravitational_potential_energy")
class CalculateGravitationalPotentialEnergyTool(EnvironmentTool):
    """计算重力势能"""
    
    name = "calculate_gravitational_potential_energy"
    description = "计算重力势能"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "height": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_gravitational_potential_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            height = args.get("height")
            if height is None:
                return Observation(self.name, "错误: 缺少必需参数 height")
            gravity = args.get("gravity", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.mechanical_energy_solver_7368 import calculate_gravitational_potential_energy
            
            # 调用函数
            result = calculate_gravitational_potential_energy(mass, height, gravity)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_total_mechanical_energy")
class CalculateTotalMechanicalEnergyTool(EnvironmentTool):
    """计算总机械能"""
    
    name = "calculate_total_mechanical_energy"
    description = "计算总机械能"
    arguments = {
        "kinetic_energy": {"type": "number", "description": ""},
        "potential_energies": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_total_mechanical_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            kinetic_energy = args.get("kinetic_energy")
            if kinetic_energy is None:
                return Observation(self.name, "错误: 缺少必需参数 kinetic_energy")
            potential_energies = args.get("potential_energies")
            if potential_energies is None:
                return Observation(self.name, "错误: 缺少必需参数 potential_energies")
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.mechanical_energy_solver_7368 import calculate_total_mechanical_energy
            
            # 调用函数
            result = calculate_total_mechanical_energy(kinetic_energy, potential_energies)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="solve_final_velocity_conservation")
class SolveFinalVelocityConservationTool(EnvironmentTool):
    """用机械能守恒求末速度"""
    
    name = "solve_final_velocity_conservation"
    description = "用机械能守恒求末速度"
    arguments = {
        "mass": {"type": "number", "description": ""},
        "initial_velocity": {"type": "string", "description": ""},
        "initial_height": {"type": "number", "description": ""},
        "final_height": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_final_velocity_conservation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            initial_velocity = args.get("initial_velocity")
            if initial_velocity is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_velocity")
            initial_height = args.get("initial_height")
            if initial_height is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_height")
            final_height = args.get("final_height", None)
            gravity = args.get("gravity", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.mechanical_energy_solver_7368 import solve_final_velocity_conservation
            
            # 调用函数
            result = solve_final_velocity_conservation(mass, initial_velocity, initial_height, final_height, gravity)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_energy_conservation")
class VisualizeEnergyConservationTool(EnvironmentTool):
    """可视化初末态与能量守恒"""
    
    name = "visualize_energy_conservation"
    description = "可视化初末态与能量守恒"
    arguments = {
        "initial_state": {"type": "object", "description": ""},
        "final_state": {"type": "object", "description": ""},
        "save_path": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_energy_conservation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            initial_state = args.get("initial_state")
            if initial_state is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_state")
            final_state = args.get("final_state")
            if final_state is None:
                return Observation(self.name, "错误: 缺少必需参数 final_state")
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.mechanical_energy_solver_7368 import visualize_energy_conservation
            
            # 调用函数
            result = visualize_energy_conservation(initial_state, final_state, save_path)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_acceleration")
class CalculateAccelerationTool(EnvironmentTool):
    """Calculate constant acceleration based on initial velocity, final velocity, and time."""
    
    name = "calculate_acceleration"
    description = "Calculate constant acceleration based on initial velocity, final velocity, and time."
    arguments = {
        "initial_velocity": {"type": "number", "description": ""},
        "final_velocity": {"type": "number", "description": ""},
        "time": {"type": "number", "description": ""},
        "units": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_acceleration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            initial_velocity = args.get("initial_velocity")
            if initial_velocity is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_velocity")
            final_velocity = args.get("final_velocity")
            if final_velocity is None:
                return Observation(self.name, "错误: 缺少必需参数 final_velocity")
            time = args.get("time")
            if time is None:
                return Observation(self.name, "错误: 缺少必需参数 time")
            units = args.get("units", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.friction_dynamics_solver_8486 import calculate_acceleration
            
            # 调用函数
            result = calculate_acceleration(initial_velocity, final_velocity, time, units)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_friction_coefficient")
class CalculateFrictionCoefficientTool(EnvironmentTool):
    """Calculate the minimum coefficient of friction needed to prevent an object from sliding."""
    
    name = "calculate_friction_coefficient"
    description = "Calculate the minimum coefficient of friction needed to prevent an object from sliding."
    arguments = {
        "mass": {"type": "number", "description": ""},
        "acceleration": {"type": "number", "description": ""},
        "angle": {"type": "number", "description": ""},
        "gravity": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_friction_coefficient 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mass = args.get("mass")
            if mass is None:
                return Observation(self.name, "错误: 缺少必需参数 mass")
            acceleration = args.get("acceleration")
            if acceleration is None:
                return Observation(self.name, "错误: 缺少必需参数 acceleration")
            angle = args.get("angle", None)
            gravity = args.get("gravity", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.friction_dynamics_solver_8486 import calculate_friction_coefficient
            
            # 调用函数
            result = calculate_friction_coefficient(mass, acceleration, angle, gravity)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="analyze_cargo_stability")
class AnalyzeCargoStabilityTool(EnvironmentTool):
    """Analyze the stability of cargo on a moving vehicle."""
    
    name = "analyze_cargo_stability"
    description = "Analyze the stability of cargo on a moving vehicle."
    arguments = {
        "cargo_mass": {"type": "number", "description": ""},
        "vehicle_mass": {"type": "number", "description": ""},
        "acceleration_profile": {"type": "object", "description": ""},
        "surface_properties": {"type": "object", "description": ""},
        "visualize": {"type": "boolean", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_cargo_stability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            cargo_mass = args.get("cargo_mass")
            if cargo_mass is None:
                return Observation(self.name, "错误: 缺少必需参数 cargo_mass")
            vehicle_mass = args.get("vehicle_mass")
            if vehicle_mass is None:
                return Observation(self.name, "错误: 缺少必需参数 vehicle_mass")
            acceleration_profile = args.get("acceleration_profile")
            if acceleration_profile is None:
                return Observation(self.name, "错误: 缺少必需参数 acceleration_profile")
            surface_properties = args.get("surface_properties", None)
            visualize = args.get("visualize", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.friction_dynamics_solver_8486 import analyze_cargo_stability
            
            # 调用函数
            result = analyze_cargo_stability(cargo_mass, vehicle_mass, acceleration_profile, surface_properties, visualize)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="create_stability_visualization")
class CreateStabilityVisualizationTool(EnvironmentTool):
    """Create a visualization of cargo stability analysis."""
    
    name = "create_stability_visualization"
    description = "Create a visualization of cargo stability analysis."
    arguments = {
        "cargo_mass": {"type": "number", "description": ""},
        "vehicle_mass": {"type": "number", "description": ""},
        "acceleration": {"type": "number", "description": ""},
        "required_friction": {"type": "number", "description": ""},
        "available_friction": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_stability_visualization 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            cargo_mass = args.get("cargo_mass")
            if cargo_mass is None:
                return Observation(self.name, "错误: 缺少必需参数 cargo_mass")
            vehicle_mass = args.get("vehicle_mass")
            if vehicle_mass is None:
                return Observation(self.name, "错误: 缺少必需参数 vehicle_mass")
            acceleration = args.get("acceleration")
            if acceleration is None:
                return Observation(self.name, "错误: 缺少必需参数 acceleration")
            required_friction = args.get("required_friction")
            if required_friction is None:
                return Observation(self.name, "错误: 缺少必需参数 required_friction")
            available_friction = args.get("available_friction", None)
            
            # 导入并调用原始函数
            from toolkits.physics.mechanics.friction_dynamics_solver_8486 import create_stability_visualization
            
            # 调用函数
            result = create_stability_visualization(cargo_mass, vehicle_mass, acceleration, required_friction, available_friction)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64, bool_）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


# ==================== 工具注册函数 ====================

def register_mechanics_tools(environment):
    """
    将所有 mechanics 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

