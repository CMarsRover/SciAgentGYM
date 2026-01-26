#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plasma_physics 工具注册模块
使用 gym.tool.EnvironmentTool 为 plasma_physics 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.physics.plasma_physics.fluid_plasma_solver_218 import *  # 动态导入
# from toolkits.physics.plasma_physics.fluid_plasma_solver_239 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="pipe_flow_velocity")
class PipeFlowVelocityTool(EnvironmentTool):
    """计算圆管内层流(泊肃叶流动)的速度分布。基于Hagen-Poiseuille方程，计算圆管内任意位置的流速。对于不可压缩流体在圆管内的层流，速度分布呈抛物线形。"""
    
    name = "pipe_flow_velocity"
    description = "计算圆管内层流(泊肃叶流动)的速度分布。基于Hagen-Poiseuille方程，计算圆管内任意位置的流速。对于不可压缩流体在圆管内的层流，速度分布呈抛物线形。"
    arguments = {
        "radius": {"type": "number", "description": "管道半径，单位：米"},
        "position": {"type": "number", "description": "距离管道中心的径向距离，单位：米"},
        "flow_rate": {"type": "number", "description": "体积流量，单位：立方米/秒"},
        "viscosity": {"type": "number", "description": "流体动力粘度，单位：Pa·s，默认为1.0"},
        "pressure_gradient": {"type": "number", "description": "沿管道轴向的压力梯度，单位：Pa/m，默认为1.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 pipe_flow_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            radius = args.get("radius")
            if radius is None:
                return Observation(self.name, "错误: 缺少必需参数 radius")
            position = args.get("position")
            if position is None:
                return Observation(self.name, "错误: 缺少必需参数 position")
            flow_rate = args.get("flow_rate")
            if flow_rate is None:
                return Observation(self.name, "错误: 缺少必需参数 flow_rate")
            viscosity = args.get("viscosity", None)
            pressure_gradient = args.get("pressure_gradient", None)
            
            # 导入并调用原始函数
            from toolkits.physics.plasma_physics.fluid_plasma_solver_218 import pipe_flow_velocity
            
            # 调用函数
            result = pipe_flow_velocity(radius, position, flow_rate, viscosity, pressure_gradient)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="solve_river_crossing")
class SolveRiverCrossingTool(EnvironmentTool):
    """计算河流中船只往返时的水流速度比值。"""
    
    name = "solve_river_crossing"
    description = "计算河流中船只往返时的水流速度比值。"
    arguments = {
        "angle_outbound": {"type": "number", "description": "去程夹角(度)"},
        "angle_return": {"type": "number", "description": "返程夹角(度)"},
        "v_boat": {"type": "number", "description": "静水船速"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_river_crossing 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            angle_outbound = args.get("angle_outbound")
            if angle_outbound is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_outbound")
            angle_return = args.get("angle_return")
            if angle_return is None:
                return Observation(self.name, "错误: 缺少必需参数 angle_return")
            v_boat = args.get("v_boat", None)
            
            # 导入并调用原始函数
            from toolkits.physics.plasma_physics.fluid_plasma_solver_239 import solve_river_crossing
            
            # 调用函数
            result = solve_river_crossing(angle_outbound, angle_return, v_boat)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


# ==================== 工具注册函数 ====================

def register_plasma_physics_tools(environment):
    """
    将所有 plasma_physics 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

