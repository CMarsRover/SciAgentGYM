#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
physical_chemistry 工具注册模块
使用 gym.tool.EnvironmentTool 为 physical_chemistry 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.chemistry.physical_chemistry.physical_chemistry_toolkit_4720 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="ideal_gas_calculation")
class IdealGasCalculationTool(EnvironmentTool):
    """理想气体状态方程计算工具，可以计算压力、体积、温度、物质的量中的任意一个，给定其他三个参数。"""
    
    name = "ideal_gas_calculation"
    description = "理想气体状态方程计算工具，可以计算压力、体积、温度、物质的量中的任意一个，给定其他三个参数。"
    arguments = {
        "pressure": {"type": "number", "description": "气体压力，单位由units['P']指定，默认为Pa"},
        "volume": {"type": "number", "description": "气体体积，单位由units['V']指定，默认为m³"},
        "temperature": {"type": "number", "description": "气体温度，单位由units['T']指定，默认为K"},
        "gas_constant": {"type": "number", "description": "气体常数，默认为8.314 J/(mol·K)"},
        "quantity": {"type": "number", "description": "物质的量，单位由units['n']指定，默认为mol"},
        "units": {"type": "object", "description": "各参数的单位，支持常见单位转换"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 ideal_gas_calculation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pressure = args.get("pressure")
            if pressure is None:
                return Observation(self.name, "错误: 缺少必需参数 pressure")
            volume = args.get("volume")
            if volume is None:
                return Observation(self.name, "错误: 缺少必需参数 volume")
            temperature = args.get("temperature")
            if temperature is None:
                return Observation(self.name, "错误: 缺少必需参数 temperature")
            gas_constant = args.get("gas_constant", None)
            quantity = args.get("quantity", None)
            units = args.get("units", None)
            
            # 导入并调用原始函数
            from toolkits.chemistry.physical_chemistry.physical_chemistry_toolkit_4720 import ideal_gas_calculation
            
            # 调用函数
            result = ideal_gas_calculation(pressure, volume, temperature, gas_constant, quantity, units)
            
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

def register_physical_chemistry_tools(environment):
    """
    将所有 physical_chemistry 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

