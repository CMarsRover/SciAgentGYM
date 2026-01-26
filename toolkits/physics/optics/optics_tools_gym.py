#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optics 工具注册模块
使用 gym.tool.EnvironmentTool 为 optics 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.physics.optics.optical_interference_solver_204 import *  # 动态导入
# from toolkits.physics.optics.optical_wave_propagation_65 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="calculate_thin_film_interference")
class CalculateThinFilmInterferenceTool(EnvironmentTool):
    """计算薄膜干涉中反射光的增强和减弱波长。"""
    
    name = "calculate_thin_film_interference"
    description = "计算薄膜干涉中反射光的增强和减弱波长。"
    arguments = {
        "n1": {"type": "number", "description": "入射介质的折射率"},
        "n2": {"type": "number", "description": "薄膜介质的折射率"},
        "n3": {"type": "number", "description": "基底介质的折射率"},
        "d": {"type": "number", "description": "薄膜厚度，单位nm"},
        "wavelength_range": {"type": "array", "description": "波长范围(nm)"},
        "incidence_angle": {"type": "number", "description": "入射角(度)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_thin_film_interference 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            n1 = args.get("n1")
            if n1 is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            n2 = args.get("n2")
            if n2 is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            n3 = args.get("n3")
            if n3 is None:
                return Observation(self.name, "错误: 缺少必需参数 n3")
            d = args.get("d")
            if d is None:
                return Observation(self.name, "错误: 缺少必需参数 d")
            wavelength_range = args.get("wavelength_range", None)
            incidence_angle = args.get("incidence_angle", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_interference_solver_204 import calculate_thin_film_interference
            
            # 调用函数
            result = calculate_thin_film_interference(n1, n2, n3, d, wavelength_range, incidence_angle)
            
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


@Toolbox.register(name="find_extrema_wavelengths")
class FindExtremaWavelengthsTool(EnvironmentTool):
    """通过数值方法找出反射系数的极值点，确定反射增强和减弱的精确波长。"""
    
    name = "find_extrema_wavelengths"
    description = "通过数值方法找出反射系数的极值点，确定反射增强和减弱的精确波长。"
    arguments = {
        "n1": {"type": "number", "description": ""},
        "n2": {"type": "number", "description": ""},
        "n3": {"type": "number", "description": ""},
        "d": {"type": "number", "description": ""},
        "wavelength_range": {"type": "array", "description": ""},
        "num_points": {"type": "integer", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_extrema_wavelengths 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            n1 = args.get("n1")
            if n1 is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            n2 = args.get("n2")
            if n2 is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            n3 = args.get("n3")
            if n3 is None:
                return Observation(self.name, "错误: 缺少必需参数 n3")
            d = args.get("d")
            if d is None:
                return Observation(self.name, "错误: 缺少必需参数 d")
            wavelength_range = args.get("wavelength_range", None)
            num_points = args.get("num_points", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_interference_solver_204 import find_extrema_wavelengths
            
            # 调用函数
            result = find_extrema_wavelengths(n1, n2, n3, d, wavelength_range, num_points)
            
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


@Toolbox.register(name="calculate_reflection_coefficient")
class CalculateReflectionCoefficientTool(EnvironmentTool):
    """计算光波在两种介质界面上的反射系数（振幅反射率）。基于菲涅耳方程计算光波在两种不同折射率介质界面上的反射系数。对于垂直入射(θ=0)，TE和TM偏振的反射系数相同。"""
    
    name = "calculate_reflection_coefficient"
    description = "计算光波在两种介质界面上的反射系数（振幅反射率）。基于菲涅耳方程计算光波在两种不同折射率介质界面上的反射系数。对于垂直入射(θ=0)，TE和TM偏振的反射系数相同。"
    arguments = {
        "n1": {"type": "number", "description": "入射介质的折射率"},
        "n2": {"type": "number", "description": "透射介质的折射率"},
        "theta_i": {"type": "number", "description": "入射角（弧度），默认为0（垂直入射）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_reflection_coefficient 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            n1 = args.get("n1")
            if n1 is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            n2 = args.get("n2")
            if n2 is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            theta_i = args.get("theta_i", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_wave_propagation_65 import calculate_reflection_coefficient
            
            # 调用函数
            result = calculate_reflection_coefficient(n1, n2, theta_i)
            
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


@Toolbox.register(name="calculate_power_reflection")
class CalculatePowerReflectionTool(EnvironmentTool):
    """计算光波在两种介质界面上的功率反射率。功率反射率等于反射系数的平方，表示反射功率与入射功率之比。"""
    
    name = "calculate_power_reflection"
    description = "计算光波在两种介质界面上的功率反射率。功率反射率等于反射系数的平方，表示反射功率与入射功率之比。"
    arguments = {
        "n1": {"type": "number", "description": "入射介质的折射率"},
        "n2": {"type": "number", "description": "透射介质的折射率"},
        "theta_i": {"type": "number", "description": "入射角（弧度），默认为0（垂直入射）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_power_reflection 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            n1 = args.get("n1")
            if n1 is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            n2 = args.get("n2")
            if n2 is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            theta_i = args.get("theta_i", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_wave_propagation_65 import calculate_power_reflection
            
            # 调用函数
            result = calculate_power_reflection(n1, n2, theta_i)
            
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


@Toolbox.register(name="angular_spectrum_propagation")
class AngularSpectrumPropagationTool(EnvironmentTool):
    """使用角谱方法计算光场在自由空间中的传播。基于傅里叶光学原理，通过频域中的传递函数计算光场传播。"""
    
    name = "angular_spectrum_propagation"
    description = "使用角谱方法计算光场在自由空间中的传播。基于傅里叶光学原理，通过频域中的传递函数计算光场传播。"
    arguments = {
        "field": {"type": "array", "description": "输入光场的复振幅分布"},
        "dx": {"type": "number", "description": "x方向的空间采样间隔（米）"},
        "dy": {"type": "number", "description": "y方向的空间采样间隔（米）"},
        "wavelength": {"type": "number", "description": "光的波长（米）"},
        "z": {"type": "number", "description": "传播距离（米）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 angular_spectrum_propagation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            field = args.get("field")
            if field is None:
                return Observation(self.name, "错误: 缺少必需参数 field")
            dx = args.get("dx")
            if dx is None:
                return Observation(self.name, "错误: 缺少必需参数 dx")
            dy = args.get("dy")
            if dy is None:
                return Observation(self.name, "错误: 缺少必需参数 dy")
            wavelength = args.get("wavelength")
            if wavelength is None:
                return Observation(self.name, "错误: 缺少必需参数 wavelength")
            z = args.get("z")
            if z is None:
                return Observation(self.name, "错误: 缺少必需参数 z")
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_wave_propagation_65 import angular_spectrum_propagation
            
            # 调用函数
            result = angular_spectrum_propagation(field, dx, dy, wavelength, z)
            
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


@Toolbox.register(name="optimize_optical_system")
class OptimizeOpticalSystemTool(EnvironmentTool):
    """优化光学系统参数以最小化给定的评价函数。使用数值优化方法找到光学系统的最佳参数配置。"""
    
    name = "optimize_optical_system"
    description = "优化光学系统参数以最小化给定的评价函数。使用数值优化方法找到光学系统的最佳参数配置。"
    arguments = {
        "initial_params": {"type": "array", "description": "初始参数值"},
        "merit_function": {"type": "object", "description": "评价函数，接受参数并返回需要最小化的标量值"},
        "bounds": {"type": "array", "description": "参数的边界约束，格式为[(min1, max1), (min2, max2), ...]"},
        "method": {"type": "string", "description": "优化算法，默认为'Nelder-Mead'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 optimize_optical_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            initial_params = args.get("initial_params")
            if initial_params is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_params")
            merit_function = args.get("merit_function")
            if merit_function is None:
                return Observation(self.name, "错误: 缺少必需参数 merit_function")
            bounds = args.get("bounds", None)
            method = args.get("method", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_wave_propagation_65 import optimize_optical_system
            
            # 调用函数
            result = optimize_optical_system(initial_params, merit_function, bounds, method)
            
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


@Toolbox.register(name="analyze_spectrum")
class AnalyzeSpectrumTool(EnvironmentTool):
    """分析光谱数据，识别峰值并提取特征。对光谱数据进行峰值检测和特征提取，用于光谱分析。"""
    
    name = "analyze_spectrum"
    description = "分析光谱数据，识别峰值并提取特征。对光谱数据进行峰值检测和特征提取，用于光谱分析。"
    arguments = {
        "wavelengths": {"type": "array", "description": "波长数据（纳米）"},
        "intensities": {"type": "array", "description": "对应的强度数据"},
        "prominence": {"type": "number", "description": "峰值检测的突出度阈值"},
        "width": {"type": "number", "description": "峰值检测的宽度约束"},
        "height": {"type": "number", "description": "峰值检测的高度阈值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            wavelengths = args.get("wavelengths")
            if wavelengths is None:
                return Observation(self.name, "错误: 缺少必需参数 wavelengths")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            prominence = args.get("prominence", None)
            width = args.get("width", None)
            height = args.get("height", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_wave_propagation_65 import analyze_spectrum
            
            # 调用函数
            result = analyze_spectrum(wavelengths, intensities, prominence, width, height)
            
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


@Toolbox.register(name="calculate_prism_reflection")
class CalculatePrismReflectionTool(EnvironmentTool):
    """计算棱镜表面的反射功率比。计算光波在棱镜表面的反射功率与入射功率之比，适用于各种棱镜几何形状。"""
    
    name = "calculate_prism_reflection"
    description = "计算棱镜表面的反射功率比。计算光波在棱镜表面的反射功率与入射功率之比，适用于各种棱镜几何形状。"
    arguments = {
        "epsilon_r": {"type": "number", "description": "棱镜材料的相对介电常数"},
        "incident_angle": {"type": "number", "description": "入射角（弧度），默认为0（垂直入射）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_prism_reflection 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            epsilon_r = args.get("epsilon_r")
            if epsilon_r is None:
                return Observation(self.name, "错误: 缺少必需参数 epsilon_r")
            incident_angle = args.get("incident_angle", None)
            
            # 导入并调用原始函数
            from toolkits.physics.optics.optical_wave_propagation_65 import calculate_prism_reflection
            
            # 调用函数
            result = calculate_prism_reflection(epsilon_r, incident_angle)
            
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

def register_optics_tools(environment):
    """
    将所有 optics 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

