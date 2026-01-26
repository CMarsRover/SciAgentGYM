#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
astronomy 工具注册模块
使用 gym.tool.EnvironmentTool 为 astronomy 目录中的工具提供统一的注册与调用接口

本文件由 register_tools_for_directory.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# ==================== 工具类定义 ====================


# ==================== 新增工具类（由 register_tools_for_directory.py 自动添加） ====================


@Toolbox.register(name="calculate_stellar_luminosity")
class CalculateStellarLuminosityTool(EnvironmentTool):
    """计算恒星光度（基于Stefan-Boltzmann定律）"""
    
    name = "calculate_stellar_luminosity"
    description = "计算恒星光度（基于Stefan-Boltzmann定律）"
    arguments = {
        "radius": {"type": "number", "description": "恒星半径（默认单位：太阳半径）"},
        "temperature": {"type": "number", "description": "恒星有效温度/K，范围2000-50000K"},
        "radius_unit": {"type": "string", "description": "半径单位，'solar'(太阳半径)或'meters'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_stellar_luminosity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "radius" not in args or args["radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 radius")
            if "temperature" not in args or args["temperature"] is None:
                return Observation(self.name, "错误: 缺少必需参数 temperature")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import calculate_stellar_luminosity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["radius", "temperature", "radius_unit"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_stellar_luminosity(**func_kwargs)
            
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


@Toolbox.register(name="calculate_eclipse_depth")
class CalculateEclipseDepthTool(EnvironmentTool):
    """计算掩食事件的光度损失"""
    
    name = "calculate_eclipse_depth"
    description = "计算掩食事件的光度损失"
    arguments = {
        "blocking_radius": {"type": "number", "description": "遮挡天体半径（太阳半径）"},
        "blocked_radius": {"type": "number", "description": "被遮挡天体半径（太阳半径）"},
        "blocking_luminosity": {"type": "number", "description": "遮挡天体光度（太阳光度）"},
        "blocked_luminosity": {"type": "number", "description": "被遮挡天体光度（太阳光度）"},
        "eclipse_type": {"type": "string", "description": "掩食类型，'partial'(部分食)或'total'(全食)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_eclipse_depth 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "blocking_radius" not in args or args["blocking_radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blocking_radius")
            if "blocked_radius" not in args or args["blocked_radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blocked_radius")
            if "blocking_luminosity" not in args or args["blocking_luminosity"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blocking_luminosity")
            if "blocked_luminosity" not in args or args["blocked_luminosity"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blocked_luminosity")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import calculate_eclipse_depth
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["blocking_radius", "blocked_radius", "blocking_luminosity", "blocked_luminosity", "eclipse_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_eclipse_depth(**func_kwargs)
            
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


@Toolbox.register(name="calculate_transit_depth")
class CalculateTransitDepthTool(EnvironmentTool):
    """计算行星凌星事件的光度变化"""
    
    name = "calculate_transit_depth"
    description = "计算行星凌星事件的光度变化"
    arguments = {
        "planet_radius": {"type": "number", "description": "行星半径（太阳半径）"},
        "star_radius": {"type": "number", "description": "被凌恒星半径（太阳半径）"},
        "star_luminosity": {"type": "number", "description": "被凌恒星光度（太阳光度）"},
        "companion_luminosity": {"type": "number", "description": "伴星光度（太阳光度），默认0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_transit_depth 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "planet_radius" not in args or args["planet_radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 planet_radius")
            if "star_radius" not in args or args["star_radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_radius")
            if "star_luminosity" not in args or args["star_luminosity"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_luminosity")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import calculate_transit_depth
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["planet_radius", "star_radius", "star_luminosity", "companion_luminosity"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_transit_depth(**func_kwargs)
            
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


@Toolbox.register(name="find_brightness_extrema")
class FindBrightnessExtremaTool(EnvironmentTool):
    """找出系统亮度的最大值和最小值"""
    
    name = "find_brightness_extrema"
    description = "找出系统亮度的最大值和最小值"
    arguments = {
        "luminosity_states": {"type": "array", "description": "不同状态下的光度列表（太阳光度）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_brightness_extrema 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "luminosity_states" not in args or args["luminosity_states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 luminosity_states")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import find_brightness_extrema
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["luminosity_states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = find_brightness_extrema(**func_kwargs)
            
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


@Toolbox.register(name="analyze_eclipsing_binary_system")
class AnalyzeEclipsingBinarySystemTool(EnvironmentTool):
    """分析食双星系统的完整光变特性。综合计算恒星光度、掩食深度和凌星效应，确定系统亮度变化范围。适用于研究Algol型、β Lyrae型等食双星系统。"""
    
    name = "analyze_eclipsing_binary_system"
    description = "分析食双星系统的完整光变特性。综合计算恒星光度、掩食深度和凌星效应，确定系统亮度变化范围。适用于研究Algol型、β Lyrae型等食双星系统。"
    arguments = {
        "star_a_radius": {"type": "number", "description": "主星半径（太阳半径）"},
        "star_a_temp": {"type": "number", "description": "主星温度（开尔文）"},
        "star_b_radius": {"type": "number", "description": "伴星半径（太阳半径）"},
        "star_b_temp": {"type": "number", "description": "伴星温度（开尔文）"},
        "planet_radius": {"type": "number", "description": "行星半径（太阳半径），可选"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_eclipsing_binary_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "star_a_radius" not in args or args["star_a_radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_a_radius")
            if "star_a_temp" not in args or args["star_a_temp"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_a_temp")
            if "star_b_radius" not in args or args["star_b_radius"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_b_radius")
            if "star_b_temp" not in args or args["star_b_temp"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_b_temp")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import analyze_eclipsing_binary_system
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["star_a_radius", "star_a_temp", "star_b_radius", "star_b_temp", "planet_radius"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_eclipsing_binary_system(**func_kwargs)
            
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


@Toolbox.register(name="scan_parameter_space")
class ScanParameterSpaceTool(EnvironmentTool):
    """扫描参数空间以研究系统行为"""
    
    name = "scan_parameter_space"
    description = "扫描参数空间以研究系统行为"
    arguments = {
        "base_params": {"type": "object", "description": "基准参数字典，包含 'star_a_radius', 'star_a_temp', 'star_b_radius', 'star_b_temp', 'planet_radius'"},
        "scan_param": {"type": "string", "description": "要扫描的参数名称"},
        "scan_range": {"type": "array", "description": "参数扫描范围（列表）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 scan_parameter_space 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "base_params" not in args or args["base_params"] is None:
                return Observation(self.name, "错误: 缺少必需参数 base_params")
            if "scan_param" not in args or args["scan_param"] is None:
                return Observation(self.name, "错误: 缺少必需参数 scan_param")
            if "scan_range" not in args or args["scan_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 scan_range")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import scan_parameter_space
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["base_params", "scan_param", "scan_range"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = scan_parameter_space(**func_kwargs)
            
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


@Toolbox.register(name="batch_analyze_systems_from_catalog")
class BatchAnalyzeSystemsFromCatalogTool(EnvironmentTool):
    """批量分析多个食双星系统"""
    
    name = "batch_analyze_systems_from_catalog"
    description = "批量分析多个食双星系统"
    arguments = {
        "systems_data": {"type": "array", "description": "系统参数列表，每个元素为包含恒星参数的字典"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_analyze_systems_from_catalog 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "systems_data" not in args or args["systems_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 systems_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import batch_analyze_systems_from_catalog
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["systems_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = batch_analyze_systems_from_catalog(**func_kwargs)
            
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


@Toolbox.register(name="visualize_light_curve")
class VisualizeLightCurveTool(EnvironmentTool):
    """可视化光变曲线，绘制系统亮度随时间的变化，标注掩食和凌星事件。"""
    
    name = "visualize_light_curve"
    description = "可视化光变曲线，绘制系统亮度随时间的变化，标注掩食和凌星事件。"
    arguments = {
        "time_array": {"type": "array", "description": "时间数组（天）"},
        "luminosity_array": {"type": "array", "description": "光度数组（太阳光度）"},
        "events": {"type": "array", "description": "事件标注列表，每个事件为字典对象，包含time和label字段，格式 [{'time': t, 'label': '事件名'}]"},
        "title": {"type": "string", "description": "图表标题，默认为'Light Curve'"},
        "save_dir": {"type": "string", "description": "保存目录，默认为'./tool_images/'"},
        "filename": {"type": "string", "description": "文件名（不含扩展名），若未提供则自动生成时间戳文件名"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_light_curve 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "time_array" not in args or args["time_array"] is None:
                return Observation(self.name, "错误: 缺少必需参数 time_array")
            if "luminosity_array" not in args or args["luminosity_array"] is None:
                return Observation(self.name, "错误: 缺少必需参数 luminosity_array")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import visualize_light_curve
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["time_array", "luminosity_array", "events", "title", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_light_curve(**func_kwargs)
            
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


@Toolbox.register(name="visualize_parameter_sensitivity")
class VisualizeParameterSensitivityTool(EnvironmentTool):
    """可视化参数敏感性分析。绘制亮度变化比随参数变化的曲线，用于优化观测策略。"""
    
    name = "visualize_parameter_sensitivity"
    description = "可视化参数敏感性分析。绘制亮度变化比随参数变化的曲线，用于优化观测策略。"
    arguments = {
        "scan_values": {"type": "array", "description": "扫描的参数值列表"},
        "brightness_ratios": {"type": "array", "description": "对应的亮度比列表"},
        "param_name": {"type": "string", "description": "参数名称，用于标签"},
        "save_dir": {"type": "string", "description": "保存目录，默认为'./tool_images/'"},
        "filename": {"type": "string", "description": "文件名，若未提供则自动生成时间戳文件名"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_parameter_sensitivity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "scan_values" not in args or args["scan_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 scan_values")
            if "brightness_ratios" not in args or args["brightness_ratios"] is None:
                return Observation(self.name, "错误: 缺少必需参数 brightness_ratios")
            if "param_name" not in args or args["param_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 param_name")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_6 import visualize_parameter_sensitivity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["scan_values", "brightness_ratios", "param_name", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_parameter_sensitivity(**func_kwargs)
            
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


@Toolbox.register(name="main")
class MainTool(EnvironmentTool):
    """演示工具包解决【恒星距离排序问题】+【至少2个相关场景】"""
    
    name = "main"
    description = "演示工具包解决【恒星距离排序问题】+【至少2个相关场景】"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 main 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import main
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = main(**func_kwargs)
            
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


@Toolbox.register(name="wavelength_shift_to_velocity")
class WavelengthShiftToVelocityTool(EnvironmentTool):
    """将光谱线波长偏移转换为径向速度（多普勒效应）"""
    
    name = "wavelength_shift_to_velocity"
    description = "将光谱线波长偏移转换为径向速度（多普勒效应）"
    arguments = {
        "delta_lambda": {"type": "number", "description": "波长偏移量/Å（埃），可为正负值"},
        "lambda_0": {"type": "number", "description": "参考波长/Å，必须为正值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 wavelength_shift_to_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "delta_lambda" not in args or args["delta_lambda"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_lambda")
            if "lambda_0" not in args or args["lambda_0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lambda_0")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import wavelength_shift_to_velocity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["delta_lambda", "lambda_0"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = wavelength_shift_to_velocity(**func_kwargs)
            
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


@Toolbox.register(name="rv_amplitude_to_semimajor_axis")
class RvAmplitudeToSemimajorAxisTool(EnvironmentTool):
    """从径向速度振幅反推轨道半长轴。物理原理：K = (2πa/P) * (M_p sin i)/(M_* + M_p)，对于圆轨道且 M_p << M_*，简化为：a ≈ (K * P * M_*) / (2π * M_p * sin i)"""
    
    name = "rv_amplitude_to_semimajor_axis"
    description = "从径向速度振幅反推轨道半长轴。物理原理：K = (2πa/P) * (M_p sin i)/(M_* + M_p)，对于圆轨道且 M_p << M_*，简化为：a ≈ (K * P * M_*) / (2π * M_p * sin i)"
    arguments = {
        "K": {"type": "number", "description": "径向速度半振幅，单位：m/s"},
        "P": {"type": "number", "description": "轨道周期，单位：s"},
        "M_star": {"type": "number", "description": "恒星质量，单位：kg"},
        "M_planet": {"type": "number", "description": "行星质量，单位：kg"},
        "inclination": {"type": "number", "description": "轨道倾角，单位：度，默认90°（边缘观测）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 rv_amplitude_to_semimajor_axis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "K" not in args or args["K"] is None:
                return Observation(self.name, "错误: 缺少必需参数 K")
            if "P" not in args or args["P"] is None:
                return Observation(self.name, "错误: 缺少必需参数 P")
            if "M_star" not in args or args["M_star"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_star")
            if "M_planet" not in args or args["M_planet"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_planet")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import rv_amplitude_to_semimajor_axis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["K", "P", "M_star", "M_planet", "inclination"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = rv_amplitude_to_semimajor_axis(**func_kwargs)
            
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


@Toolbox.register(name="kepler_third_law")
class KeplerThirdLawTool(EnvironmentTool):
    """开普勒第三定律：计算轨道周期。物理原理：P² = (4π²/GM) * a³"""
    
    name = "kepler_third_law"
    description = "开普勒第三定律：计算轨道周期。物理原理：P² = (4π²/GM) * a³"
    arguments = {
        "a": {"type": "number", "description": "轨道半长轴，单位：米"},
        "M_star": {"type": "number", "description": "中心天体质量，单位：千克"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 kepler_third_law 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "a" not in args or args["a"] is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            if "M_star" not in args or args["M_star"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_star")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import kepler_third_law
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["a", "M_star"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = kepler_third_law(**func_kwargs)
            
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


@Toolbox.register(name="rv_amplitude_from_orbit")
class RvAmplitudeFromOrbitTool(EnvironmentTool):
    """从轨道参数计算径向速度振幅（正向计算）。物理原理：K = (2πa/P) * (M_p sin i)/(M_* + M_p)"""
    
    name = "rv_amplitude_from_orbit"
    description = "从轨道参数计算径向速度振幅（正向计算）。物理原理：K = (2πa/P) * (M_p sin i)/(M_* + M_p)"
    arguments = {
        "a": {"type": "number", "description": "轨道半长轴，单位：米"},
        "P": {"type": "number", "description": "轨道周期，单位：秒"},
        "M_star": {"type": "number", "description": "恒星质量，单位：千克"},
        "M_planet": {"type": "number", "description": "行星质量，单位：千克"},
        "inclination": {"type": "number", "description": "轨道倾角，单位：度，默认90°"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 rv_amplitude_from_orbit 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "a" not in args or args["a"] is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            if "P" not in args or args["P"] is None:
                return Observation(self.name, "错误: 缺少必需参数 P")
            if "M_star" not in args or args["M_star"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_star")
            if "M_planet" not in args or args["M_planet"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_planet")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import rv_amplitude_from_orbit
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["a", "P", "M_star", "M_planet", "inclination"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = rv_amplitude_from_orbit(**func_kwargs)
            
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


@Toolbox.register(name="fetch_exoplanet_data")
class FetchExoplanetDataTool(EnvironmentTool):
    """从NASA Exoplanet Archive获取系外行星数据"""
    
    name = "fetch_exoplanet_data"
    description = "从NASA Exoplanet Archive获取系外行星数据"
    arguments = {
        "planet_name": {"type": "string", "description": "行星名称，如'Kepler-186 f', '51 Peg b'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_exoplanet_data 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "planet_name" not in args or args["planet_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 planet_name")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import fetch_exoplanet_data
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["planet_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = fetch_exoplanet_data(**func_kwargs)
            
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


@Toolbox.register(name="calculate_period_ratio_from_rv")
class CalculatePeriodRatioFromRvTool(EnvironmentTool):
    """从两颗行星的径向速度数据计算轨道周期比"""
    
    name = "calculate_period_ratio_from_rv"
    description = "从两颗行星的径向速度数据计算轨道周期比"
    arguments = {
        "delta_lambda_1": {"type": "number", "description": "行星1的波长偏移/mÅ (毫埃)"},
        "delta_lambda_2": {"type": "number", "description": "行星2的波长偏移/mÅ"},
        "lambda_0": {"type": "number", "description": "参考波长/Å"},
        "M_star": {"type": "number", "description": "恒星质量/kg"},
        "M_planet_1": {"type": "number", "description": "行星1质量/kg"},
        "M_planet_2": {"type": "number", "description": "行星2质量/kg"},
        "inclination": {"type": "number", "description": "轨道倾角/度"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_period_ratio_from_rv 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "delta_lambda_1" not in args or args["delta_lambda_1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_lambda_1")
            if "delta_lambda_2" not in args or args["delta_lambda_2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_lambda_2")
            if "lambda_0" not in args or args["lambda_0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lambda_0")
            if "M_star" not in args or args["M_star"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_star")
            if "M_planet_1" not in args or args["M_planet_1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_planet_1")
            if "M_planet_2" not in args or args["M_planet_2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_planet_2")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import calculate_period_ratio_from_rv
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["delta_lambda_1", "delta_lambda_2", "lambda_0", "M_star", "M_planet_1", "M_planet_2", "inclination"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_period_ratio_from_rv(**func_kwargs)
            
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


@Toolbox.register(name="analyze_rv_system")
class AnalyzeRvSystemTool(EnvironmentTool):
    """完整分析单个径向速度系统的轨道参数。求解流程：1. 波长偏移 -> 径向速度振幅 K；2. 假设周期 P -> 计算半长轴 a；3. 验证开普勒第三定律 -> 迭代修正；4. 输出完整轨道参数。"""
    
    name = "analyze_rv_system"
    description = "完整分析单个径向速度系统的轨道参数。求解流程：1. 波长偏移 -> 径向速度振幅 K；2. 假设周期 P -> 计算半长轴 a；3. 验证开普勒第三定律 -> 迭代修正；4. 输出完整轨道参数。"
    arguments = {
        "delta_lambda": {"type": "number", "description": "波长偏移，单位：毫埃(mÅ)"},
        "lambda_0": {"type": "number", "description": "参考波长，单位：埃(Å)"},
        "M_star": {"type": "number", "description": "恒星质量，单位：千克(kg)"},
        "M_planet": {"type": "number", "description": "行星质量，单位：千克(kg)"},
        "inclination": {"type": "number", "description": "轨道倾角，单位：度，默认为90.0"},
        "initial_period_guess": {"type": "number", "description": "初始周期猜测，单位：天，默认为365.25"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_rv_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "delta_lambda" not in args or args["delta_lambda"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_lambda")
            if "lambda_0" not in args or args["lambda_0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lambda_0")
            if "M_star" not in args or args["M_star"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_star")
            if "M_planet" not in args or args["M_planet"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_planet")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import analyze_rv_system
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["delta_lambda", "lambda_0", "M_star", "M_planet", "inclination", "initial_period_guess"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_rv_system(**func_kwargs)
            
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


@Toolbox.register(name="compare_exoplanet_systems")
class CompareExoplanetSystemsTool(EnvironmentTool):
    """比较两个系外行星系统的轨道参数"""
    
    name = "compare_exoplanet_systems"
    description = "比较两个系外行星系统的轨道参数"
    arguments = {
        "system1_params": {"type": "object", "description": "系统1参数，包含 delta_lambda, lambda_0, M_star, M_planet 字段"},
        "system2_params": {"type": "object", "description": "系统2参数，包含 delta_lambda, lambda_0, M_star, M_planet 字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_exoplanet_systems 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "system1_params" not in args or args["system1_params"] is None:
                return Observation(self.name, "错误: 缺少必需参数 system1_params")
            if "system2_params" not in args or args["system2_params"] is None:
                return Observation(self.name, "错误: 缺少必需参数 system2_params")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import compare_exoplanet_systems
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["system1_params", "system2_params"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_exoplanet_systems(**func_kwargs)
            
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


@Toolbox.register(name="visualize_rv_curve")
class VisualizeRvCurveTool(EnvironmentTool):
    """可视化径向速度曲线"""
    
    name = "visualize_rv_curve"
    description = "可视化径向速度曲线"
    arguments = {
        "time_array": {"type": "array", "description": "时间数组/天"},
        "rv_array": {"type": "array", "description": "径向速度数组/m/s"},
        "period": {"type": "number", "description": "轨道周期/天"},
        "K": {"type": "number", "description": "RV振幅/m/s"},
        "save_dir": {"type": "string", "description": "保存目录，默认为'./tool_images/'"},
        "filename": {"type": "string", "description": "文件名（不含扩展名）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_rv_curve 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "time_array" not in args or args["time_array"] is None:
                return Observation(self.name, "错误: 缺少必需参数 time_array")
            if "rv_array" not in args or args["rv_array"] is None:
                return Observation(self.name, "错误: 缺少必需参数 rv_array")
            if "period" not in args or args["period"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period")
            if "K" not in args or args["K"] is None:
                return Observation(self.name, "错误: 缺少必需参数 K")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import visualize_rv_curve
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["time_array", "rv_array", "period", "K", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_rv_curve(**func_kwargs)
            
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


@Toolbox.register(name="visualize_orbital_comparison")
class VisualizeOrbitalComparisonTool(EnvironmentTool):
    """可视化多个行星系统的轨道对比"""
    
    name = "visualize_orbital_comparison"
    description = "可视化多个行星系统的轨道对比"
    arguments = {
        "systems_data": {"type": "array", "description": "系统数据列表，每个元素为包含 name, a_au, period_days 字段的字典"},
        "save_dir": {"type": "string", "description": "保存目录"},
        "filename": {"type": "string", "description": "文件名"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_orbital_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "systems_data" not in args or args["systems_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 systems_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import visualize_orbital_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["systems_data", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_orbital_comparison(**func_kwargs)
            
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


@Toolbox.register(name="consistency_equation")
class ConsistencyEquationTool(EnvironmentTool):
    """自洽性方程：从 K 和 P 计算 a，然后验证开普勒第三定律。返回计算的周期与输入周期的差值（应为0）。"""
    
    name = "consistency_equation"
    description = "自洽性方程：从 K 和 P 计算 a，然后验证开普勒第三定律。返回计算的周期与输入周期的差值（应为0）。"
    arguments = {
        "P_seconds": {"type": "number", "description": "输入的轨道周期，单位：秒"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 consistency_equation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "P_seconds" not in args or args["P_seconds"] is None:
                return Observation(self.name, "错误: 缺少必需参数 P_seconds")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_12 import consistency_equation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["P_seconds"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = consistency_equation(**func_kwargs)
            
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


@Toolbox.register(name="convert_ra_to_degrees")
class ConvertRaToDegreesTool(EnvironmentTool):
    """将赤经从时角格式转换为角度格式。天文学中赤经可用两种格式表示：角度格式（0-360度）或时角格式（0h-24h，1h=15度）。"""
    
    name = "convert_ra_to_degrees"
    description = "将赤经从时角格式转换为角度格式。天文学中赤经可用两种格式表示：角度格式（0-360度）或时角格式（0h-24h，1h=15度）。"
    arguments = {
        "ra": {"type": "string", "description": "赤经值，可以是角度或时角字符串（如'11h'或'11 h'）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 convert_ra_to_degrees 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "ra" not in args or args["ra"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ra")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import convert_ra_to_degrees
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["ra"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = convert_ra_to_degrees(**func_kwargs)
            
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


@Toolbox.register(name="calculate_distance_modulus")
class CalculateDistanceModulusTool(EnvironmentTool):
    """计算考虑消光修正的距离模数。物理原理：视星等、绝对星等和距离的关系，需扣除星际消光影响。"""
    
    name = "calculate_distance_modulus"
    description = "计算考虑消光修正的距离模数。物理原理：视星等、绝对星等和距离的关系，需扣除星际消光影响。"
    arguments = {
        "m_obs": {"type": "number", "description": "观测视星等（mag），范围通常-2到30"},
        "M_abs": {"type": "number", "description": "绝对星等（mag），范围通常-10到20"},
        "a_v": {"type": "number", "description": "V波段消光量（mag），默认0（无消光）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_distance_modulus 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "m_obs" not in args or args["m_obs"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_obs")
            if "M_abs" not in args or args["M_abs"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_abs")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import calculate_distance_modulus
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["m_obs", "M_abs", "a_v"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_distance_modulus(**func_kwargs)
            
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


@Toolbox.register(name="calculate_extinction")
class CalculateExtinctionTool(EnvironmentTool):
    """计算V波段的星际消光量。物理原理：星际尘埃导致星光被吸收和散射，消光量与色余成正比。公式：A_V = R_V × E(B-V)"""
    
    name = "calculate_extinction"
    description = "计算V波段的星际消光量。物理原理：星际尘埃导致星光被吸收和散射，消光量与色余成正比。公式：A_V = R_V × E(B-V)"
    arguments = {
        "e_bv": {"type": "number", "description": "B-V色余（单位：mag），范围通常0-2"},
        "rv": {"type": "number", "description": "消光比值（无量纲），默认3.1（银河系标准值）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_extinction 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "e_bv" not in args or args["e_bv"] is None:
                return Observation(self.name, "错误: 缺少必需参数 e_bv")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import calculate_extinction
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["e_bv", "rv"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_extinction(**func_kwargs)
            
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


@Toolbox.register(name="calculate_apparent_magnitude")
class CalculateApparentMagnitudeTool(EnvironmentTool):
    """计算视星等 m。视星等公式：m = M + 5*log10(d) - 5 + A_V，其中 M 为绝对星等，d 为距离(pc)，A_V 为 V波段消光。"""
    
    name = "calculate_apparent_magnitude"
    description = "计算视星等 m。视星等公式：m = M + 5*log10(d) - 5 + A_V，其中 M 为绝对星等，d 为距离(pc)，A_V 为 V波段消光。"
    arguments = {
        "absolute_mag": {"type": "number", "description": "绝对星等 M/mag（恒星在10pc处的亮度）"},
        "distance_pc": {"type": "number", "description": "距离/pc，范围必须大于0"},
        "extinction": {"type": "number", "description": "V波段消光 A_V/mag，默认0（无消光）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_apparent_magnitude 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "absolute_mag" not in args or args["absolute_mag"] is None:
                return Observation(self.name, "错误: 缺少必需参数 absolute_mag")
            if "distance_pc" not in args or args["distance_pc"] is None:
                return Observation(self.name, "错误: 缺少必需参数 distance_pc")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import calculate_apparent_magnitude
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["absolute_mag", "distance_pc", "extinction"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_apparent_magnitude(**func_kwargs)
            
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


@Toolbox.register(name="check_observatory_visibility")
class CheckObservatoryVisibilityTool(EnvironmentTool):
    """检查恒星在给定纬度天文台的理论可见性"""
    
    name = "check_observatory_visibility"
    description = "检查恒星在给定纬度天文台的理论可见性"
    arguments = {
        "dec_deg": {"type": "number", "description": "赤纬/度，范围 -90 到 +90"},
        "observatory_lat": {"type": "number", "description": "天文台纬度/度，范围 -90 到 +90"},
        "min_elevation": {"type": "number", "description": "最小仰角/度，默认0（地平线）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 check_observatory_visibility 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "dec_deg" not in args or args["dec_deg"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dec_deg")
            if "observatory_lat" not in args or args["observatory_lat"] is None:
                return Observation(self.name, "错误: 缺少必需参数 observatory_lat")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import check_observatory_visibility
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["dec_deg", "observatory_lat", "min_elevation"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = check_observatory_visibility(**func_kwargs)
            
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


@Toolbox.register(name="check_spectrograph_compatibility")
class CheckSpectrographCompatibilityTool(EnvironmentTool):
    """检查恒星与光谱仪的兼容性。光谱仪灵敏度限制：ESPRESSO (VLT): V < 17 mag；HIRES (Keck): V < 16 mag。"""
    
    name = "check_spectrograph_compatibility"
    description = "检查恒星与光谱仪的兼容性。光谱仪灵敏度限制：ESPRESSO (VLT): V < 17 mag；HIRES (Keck): V < 16 mag。"
    arguments = {
        "apparent_mag": {"type": "number", "description": "视星等，单位：mag"},
        "espresso_limit": {"type": "number", "description": "ESPRESSO限制，单位：mag，默认17"},
        "hires_limit": {"type": "number", "description": "HIRES限制，单位：mag，默认16"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 check_spectrograph_compatibility 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "apparent_mag" not in args or args["apparent_mag"] is None:
                return Observation(self.name, "错误: 缺少必需参数 apparent_mag")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import check_spectrograph_compatibility
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["apparent_mag", "espresso_limit", "hires_limit"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = check_spectrograph_compatibility(**func_kwargs)
            
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


@Toolbox.register(name="fetch_star_data_from_simbad")
class FetchStarDataFromSimbadTool(EnvironmentTool):
    """从SIMBAD数据库获取恒星基础数据。SIMBAD是免费的天文数据库，包含恒星的位置、星等等信息。"""
    
    name = "fetch_star_data_from_simbad"
    description = "从SIMBAD数据库获取恒星基础数据。SIMBAD是免费的天文数据库，包含恒星的位置、星等等信息。"
    arguments = {
        "identifier": {"type": "string", "description": "恒星标识符，如'Sirius'、'HD 48915'、'HIP 32349'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_star_data_from_simbad 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "identifier" not in args or args["identifier"] is None:
                return Observation(self.name, "错误: 缺少必需参数 identifier")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import fetch_star_data_from_simbad
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["identifier"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = fetch_star_data_from_simbad(**func_kwargs)
            
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


@Toolbox.register(name="analyze_star_observability")
class AnalyzeStarObservabilityTool(EnvironmentTool):
    """综合分析恒星的可观测性（视星等计算 + 天文台可见性 + 光谱仪兼容性）"""
    
    name = "analyze_star_observability"
    description = "综合分析恒星的可观测性（视星等计算 + 天文台可见性 + 光谱仪兼容性）"
    arguments = {
        "ra": {"type": "string", "description": "赤经，可以是角度(float)或时角(str如'11h')"},
        "dec": {"type": "number", "description": "赤纬/度，范围 -90 到 +90"},
        "absolute_mag": {"type": "number", "description": "绝对星等/mag（如提供apparent_mag则可选）"},
        "apparent_mag": {"type": "number", "description": "视星等/mag（如提供则跳过计算）"},
        "distance_pc": {"type": "number", "description": "距离/pc（计算视星等时必需）"},
        "ebv": {"type": "number", "description": "色余 E(B-V)/mag，默认0"},
        "star_name": {"type": "string", "description": "恒星名称，用于输出"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_star_observability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "ra" not in args or args["ra"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ra")
            if "dec" not in args or args["dec"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dec")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import analyze_star_observability
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["ra", "dec", "absolute_mag", "apparent_mag", "distance_pc", "ebv", "star_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_star_observability(**func_kwargs)
            
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


@Toolbox.register(name="batch_analyze_stars")
class BatchAnalyzeStarsTool(EnvironmentTool):
    """批量分析多颗恒星的可观测性"""
    
    name = "batch_analyze_stars"
    description = "批量分析多颗恒星的可观测性"
    arguments = {
        "stars_data": {"type": "array", "description": "恒星数据列表，每个元素是包含恒星参数的字典，必需字段：'ra', 'dec'，可选字段：'absolute_mag', 'apparent_mag', 'distance_pc', 'ebv', 'name'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_analyze_stars 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "stars_data" not in args or args["stars_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stars_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import batch_analyze_stars
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["stars_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = batch_analyze_stars(**func_kwargs)
            
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


@Toolbox.register(name="compare_magnitude_systems")
class CompareMagnitudeSystemsTool(EnvironmentTool):
    """比较不同星等系统下的恒星亮度。该函数展示如何处理多波段星等数据，虽然当前问题只涉及V波段，但该工具可扩展到其他波段的分析。"""
    
    name = "compare_magnitude_systems"
    description = "比较不同星等系统下的恒星亮度。该函数展示如何处理多波段星等数据，虽然当前问题只涉及V波段，但该工具可扩展到其他波段的分析。"
    arguments = {
        "stars_data": {"type": "array", "description": "恒星数据列表，包含不同波段的星等；每个恒星为字典对象，包含name、v_mag、b_mag、r_mag等字段"},
        "magnitude_bands": {"type": "array", "description": "要比较的波段列表，默认['V', 'B', 'R']"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_magnitude_systems 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "stars_data" not in args or args["stars_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stars_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import compare_magnitude_systems
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["stars_data", "magnitude_bands"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_magnitude_systems(**func_kwargs)
            
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


@Toolbox.register(name="visualize_star_distribution")
class VisualizeStarDistributionTool(EnvironmentTool):
    """可视化恒星在天球上的分布"""
    
    name = "visualize_star_distribution"
    description = "可视化恒星在天球上的分布"
    arguments = {
        "stars_data": {"type": "array", "description": "恒星数据列表，包含'ra', 'dec', 'name', 'observable'等字段"},
        "save_dir": {"type": "string", "description": "保存目录，默认'./tool_images/'"},
        "filename": {"type": "string", "description": "文件名，默认自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_star_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "stars_data" not in args or args["stars_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stars_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import visualize_star_distribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["stars_data", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_star_distribution(**func_kwargs)
            
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


@Toolbox.register(name="visualize_magnitude_comparison")
class VisualizeMagnitudeComparisonTool(EnvironmentTool):
    """可视化恒星视星等与观测限制的比较"""
    
    name = "visualize_magnitude_comparison"
    description = "可视化恒星视星等与观测限制的比较"
    arguments = {
        "stars_data": {"type": "array", "description": "恒星数据列表，包含'name', 'apparent_mag', 'observable'等字段"},
        "save_dir": {"type": "string", "description": "保存目录"},
        "filename": {"type": "string", "description": "文件名"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_magnitude_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "stars_data" not in args or args["stars_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stars_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_13 import visualize_magnitude_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["stars_data", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_magnitude_comparison(**func_kwargs)
            
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


@Toolbox.register(name="calculate_mass_ratio_from_rv")
class CalculateMassRatioFromRvTool(EnvironmentTool):
    """根据径向速度振幅计算双星质量比。物理原理：由于双星系统质心守恒，m1*v1 = m2*v2，因此质量比等于速度比的倒数。"""
    
    name = "calculate_mass_ratio_from_rv"
    description = "根据径向速度振幅计算双星质量比。物理原理：由于双星系统质心守恒，m1*v1 = m2*v2，因此质量比等于速度比的倒数。"
    arguments = {
        "k1": {"type": "number", "description": "第一颗星的径向速度振幅 (km/s)"},
        "k2": {"type": "number", "description": "第二颗星的径向速度振幅 (km/s)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_mass_ratio_from_rv 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "k1" not in args or args["k1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k1")
            if "k2" not in args or args["k2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k2")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import calculate_mass_ratio_from_rv
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["k1", "k2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_mass_ratio_from_rv(**func_kwargs)
            
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


@Toolbox.register(name="calculate_orbital_separation")
class CalculateOrbitalSeparationTool(EnvironmentTool):
    """根据轨道周期和径向速度计算双星轨道半长轴。"""
    
    name = "calculate_orbital_separation"
    description = "根据轨道周期和径向速度计算双星轨道半长轴。"
    arguments = {
        "period_years": {"type": "number", "description": "轨道周期，单位：年"},
        "k1": {"type": "number", "description": "第一颗星的径向速度振幅，单位：km/s"},
        "k2": {"type": "number", "description": "第二颗星的径向速度振幅，单位：km/s"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_orbital_separation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "period_years" not in args or args["period_years"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period_years")
            if "k1" not in args or args["k1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k1")
            if "k2" not in args or args["k2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k2")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import calculate_orbital_separation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["period_years", "k1", "k2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_orbital_separation(**func_kwargs)
            
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


@Toolbox.register(name="apply_keplers_third_law")
class ApplyKeplersThirdLawTool(EnvironmentTool):
    """应用开普勒第三定律计算两个系统的质量比。"""
    
    name = "apply_keplers_third_law"
    description = "应用开普勒第三定律计算两个系统的质量比。"
    arguments = {
        "period1": {"type": "number", "description": "系统1的轨道周期，单位：年"},
        "period2": {"type": "number", "description": "系统2的轨道周期，单位：年"},
        "separation1": {"type": "number", "description": "系统1的轨道半长轴，单位：km"},
        "separation2": {"type": "number", "description": "系统2的轨道半长轴，单位：km"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 apply_keplers_third_law 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "period1" not in args or args["period1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period1")
            if "period2" not in args or args["period2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period2")
            if "separation1" not in args or args["separation1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 separation1")
            if "separation2" not in args or args["separation2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 separation2")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import apply_keplers_third_law
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["period1", "period2", "separation1", "separation2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = apply_keplers_third_law(**func_kwargs)
            
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


@Toolbox.register(name="query_star_parameters")
class QueryStarParametersTool(EnvironmentTool):
    """从SIMBAD数据库查询恒星基本参数。通过Astroquery访问SIMBAD天文数据库，获取恒星的基本物理参数。"""
    
    name = "query_star_parameters"
    description = "从SIMBAD数据库查询恒星基本参数。通过Astroquery访问SIMBAD天文数据库，获取恒星的基本物理参数。"
    arguments = {
        "star_name": {"type": "string", "description": "恒星名称或标识符 (如 'Sirius', 'HD 48915', 'HIP 32349')"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 query_star_parameters 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "star_name" not in args or args["star_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 star_name")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import query_star_parameters
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["star_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = query_star_parameters(**func_kwargs)
            
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


@Toolbox.register(name="calculate_orbital_velocity")
class CalculateOrbitalVelocityTool(EnvironmentTool):
    """计算双星系统中各星的轨道速度。物理原理：v = 2πa/P，其中a是该星到质心的距离，由质量比确定。"""
    
    name = "calculate_orbital_velocity"
    description = "计算双星系统中各星的轨道速度。物理原理：v = 2πa/P，其中a是该星到质心的距离，由质量比确定。"
    arguments = {
        "semi_major_axis": {"type": "number", "description": "系统轨道半长轴，单位：km"},
        "period": {"type": "number", "description": "轨道周期，单位：年"},
        "mass_ratio": {"type": "number", "description": "质量比 m1/m2"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_orbital_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "semi_major_axis" not in args or args["semi_major_axis"] is None:
                return Observation(self.name, "错误: 缺少必需参数 semi_major_axis")
            if "period" not in args or args["period"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period")
            if "mass_ratio" not in args or args["mass_ratio"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mass_ratio")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import calculate_orbital_velocity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["semi_major_axis", "period", "mass_ratio"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_orbital_velocity(**func_kwargs)
            
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


@Toolbox.register(name="analyze_binary_system_mass")
class AnalyzeBinarySystemMassTool(EnvironmentTool):
    """综合分析双星系统，计算质量比、轨道参数和系统总质量相对值"""
    
    name = "analyze_binary_system_mass"
    description = "综合分析双星系统，计算质量比、轨道参数和系统总质量相对值"
    arguments = {
        "period": {"type": "number", "description": "轨道周期 (年)"},
        "k1": {"type": "number", "description": "第一颗星的径向速度振幅 (km/s)"},
        "k2": {"type": "number", "description": "第二颗星的径向速度振幅 (km/s)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_binary_system_mass 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "period" not in args or args["period"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period")
            if "k1" not in args or args["k1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k1")
            if "k2" not in args or args["k2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k2")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import analyze_binary_system_mass
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["period", "k1", "k2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_binary_system_mass(**func_kwargs)
            
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


@Toolbox.register(name="compare_binary_systems")
class CompareBinarySystemsTool(EnvironmentTool):
    """比较两个双星系统的质量"""
    
    name = "compare_binary_systems"
    description = "比较两个双星系统的质量"
    arguments = {
        "system1_params": {"type": "object", "description": "系统1参数 {'period': 周期(年), 'k1': RV1(km/s), 'k2': RV2(km/s)}"},
        "system2_params": {"type": "object", "description": "系统2参数 {'period': 周期(年), 'k1': RV1(km/s), 'k2': RV2(km/s)}"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_binary_systems 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "system1_params" not in args or args["system1_params"] is None:
                return Observation(self.name, "错误: 缺少必需参数 system1_params")
            if "system2_params" not in args or args["system2_params"] is None:
                return Observation(self.name, "错误: 缺少必需参数 system2_params")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import compare_binary_systems
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["system1_params", "system2_params"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_binary_systems(**func_kwargs)
            
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


@Toolbox.register(name="simulate_binary_orbit")
class SimulateBinaryOrbitTool(EnvironmentTool):
    """模拟双星系统的轨道运动"""
    
    name = "simulate_binary_orbit"
    description = "模拟双星系统的轨道运动"
    arguments = {
        "period": {"type": "number", "description": "轨道周期 (年)"},
        "k1": {"type": "number", "description": "第一颗星的径向速度振幅 (km/s)"},
        "k2": {"type": "number", "description": "第二颗星的径向速度振幅 (km/s)"},
        "num_points": {"type": "integer", "description": "轨道上的采样点数，默认100"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 simulate_binary_orbit 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "period" not in args or args["period"] is None:
                return Observation(self.name, "错误: 缺少必需参数 period")
            if "k1" not in args or args["k1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k1")
            if "k2" not in args or args["k2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k2")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import simulate_binary_orbit
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["period", "k1", "k2", "num_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = simulate_binary_orbit(**func_kwargs)
            
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


@Toolbox.register(name="visualize_binary_orbit")
class VisualizeBinaryOrbitTool(EnvironmentTool):
    """可视化双星系统的轨道运动，生成双星轨道的2D轨迹图和径向速度曲线图。"""
    
    name = "visualize_binary_orbit"
    description = "可视化双星系统的轨道运动，生成双星轨道的2D轨迹图和径向速度曲线图。"
    arguments = {
        "orbit_data": {"type": "object", "description": "轨道数据字典，包含time_array_years、star1_x_au、star1_y_au、star2_x_au、star2_y_au、rv1_km_s、rv2_km_s等键"},
        "system_name": {"type": "string", "description": "系统名称，用于图表标题，默认为'Binary System'"},
        "save_dir": {"type": "string", "description": "保存目录，默认为'./tool_images/'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_binary_orbit 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "orbit_data" not in args or args["orbit_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 orbit_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import visualize_binary_orbit
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["orbit_data", "system_name", "save_dir"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_binary_orbit(**func_kwargs)
            
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


@Toolbox.register(name="visualize_system_comparison")
class VisualizeSystemComparisonTool(EnvironmentTool):
    """对比可视化两个双星系统"""
    
    name = "visualize_system_comparison"
    description = "对比可视化两个双星系统"
    arguments = {
        "system1_data": {"type": "object", "description": "系统1的轨道数据"},
        "system2_data": {"type": "object", "description": "系统2的轨道数据"},
        "save_dir": {"type": "string", "description": "保存目录"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_system_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "system1_data" not in args or args["system1_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 system1_data")
            if "system2_data" not in args or args["system2_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 system2_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astrophysics_toolkit_1 import visualize_system_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["system1_data", "system2_data", "save_dir"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_system_comparison(**func_kwargs)
            
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


@Toolbox.register(name="distance_from_modulus")
class DistanceFromModulusTool(EnvironmentTool):
    """从距离模数计算距离（秒差距）。物理原理：距离模数与距离的对数关系。公式：d = 10^((μ + 5) / 5) pc"""
    
    name = "distance_from_modulus"
    description = "从距离模数计算距离（秒差距）。物理原理：距离模数与距离的对数关系。公式：d = 10^((μ + 5) / 5) pc"
    arguments = {
        "mu": {"type": "number", "description": "距离模数（mag），范围通常-5到20"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 distance_from_modulus 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "mu" not in args or args["mu"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import distance_from_modulus
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["mu"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = distance_from_modulus(**func_kwargs)
            
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


@Toolbox.register(name="validate_solar_neighborhood")
class ValidateSolarNeighborhoodTool(EnvironmentTool):
    """验证恒星是否在太阳邻域内"""
    
    name = "validate_solar_neighborhood"
    description = "验证恒星是否在太阳邻域内"
    arguments = {
        "distance_pc": {"type": "number", "description": "恒星距离（pc）"},
        "threshold": {"type": "number", "description": "太阳邻域阈值（pc），默认500"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 validate_solar_neighborhood 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "distance_pc" not in args or args["distance_pc"] is None:
                return Observation(self.name, "错误: 缺少必需参数 distance_pc")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import validate_solar_neighborhood
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["distance_pc", "threshold"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = validate_solar_neighborhood(**func_kwargs)
            
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


@Toolbox.register(name="calculate_stellar_distance")
class CalculateStellarDistanceTool(EnvironmentTool):
    """计算恒星距离的完整流程（考虑星际消光）"""
    
    name = "calculate_stellar_distance"
    description = "计算恒星距离的完整流程（考虑星际消光）"
    arguments = {
        "m_obs": {"type": "number", "description": "观测视星等（mag）"},
        "M_abs": {"type": "number", "description": "绝对星等（mag）"},
        "e_bv": {"type": "number", "description": "B-V色余（mag），默认0"},
        "rv": {"type": "number", "description": "消光比值，默认3.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_stellar_distance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "m_obs" not in args or args["m_obs"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_obs")
            if "M_abs" not in args or args["M_abs"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_abs")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import calculate_stellar_distance
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["m_obs", "M_abs", "e_bv", "rv"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_stellar_distance(**func_kwargs)
            
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


@Toolbox.register(name="analyze_star_sample")
class AnalyzeStarSampleTool(EnvironmentTool):
    """批量分析多颗恒星的距离并排序"""
    
    name = "analyze_star_sample"
    description = "批量分析多颗恒星的距离并排序"
    arguments = {
        "stars_data": {"type": "array", "description": "恒星数据列表，每个元素为字典，包含键：'id'（恒星标识）、'm_obs'（观测星等）、'M_abs'（绝对星等）、'e_bv'（色余，可选）、'rv'（消光比值，可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_star_sample 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "stars_data" not in args or args["stars_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stars_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import analyze_star_sample
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["stars_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_star_sample(**func_kwargs)
            
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


@Toolbox.register(name="compare_extinction_effects")
class CompareExtinctionEffectsTool(EnvironmentTool):
    """比较不同消光条件下的距离差异"""
    
    name = "compare_extinction_effects"
    description = "比较不同消光条件下的距离差异"
    arguments = {
        "m_obs": {"type": "number", "description": "观测星等"},
        "M_abs": {"type": "number", "description": "绝对星等"},
        "e_bv_values": {"type": "array", "description": "色余值列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_extinction_effects 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "m_obs" not in args or args["m_obs"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_obs")
            if "M_abs" not in args or args["M_abs"] is None:
                return Observation(self.name, "错误: 缺少必需参数 M_abs")
            if "e_bv_values" not in args or args["e_bv_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 e_bv_values")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import compare_extinction_effects
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["m_obs", "M_abs", "e_bv_values"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_extinction_effects(**func_kwargs)
            
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


@Toolbox.register(name="visualize_distance_distribution")
class VisualizeDistanceDistributionTool(EnvironmentTool):
    """可视化恒星距离分布"""
    
    name = "visualize_distance_distribution"
    description = "可视化恒星距离分布"
    arguments = {
        "stars_data": {"type": "array", "description": "恒星数据列表（来自analyze_star_sample的result）"},
        "save_dir": {"type": "string", "description": "保存目录，默认'./tool_images/'"},
        "filename": {"type": "string", "description": "文件名（不含扩展名），默认自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_distance_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "stars_data" not in args or args["stars_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stars_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import visualize_distance_distribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["stars_data", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_distance_distribution(**func_kwargs)
            
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


@Toolbox.register(name="visualize_extinction_impact")
class VisualizeExtinctionImpactTool(EnvironmentTool):
    """可视化消光对距离测量的影响"""
    
    name = "visualize_extinction_impact"
    description = "可视化消光对距离测量的影响"
    arguments = {
        "comparison_data": {"type": "array", "description": "消光比较数据（来自compare_extinction_effects的result）"},
        "save_dir": {"type": "string", "description": "保存目录，默认'./tool_images/'"},
        "filename": {"type": "string", "description": "文件名，默认自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_extinction_impact 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "comparison_data" not in args or args["comparison_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 comparison_data")
            
            # 导入并调用原始函数
            from toolkits.astronomy.astronomy.astronomy_toolkit_5 import visualize_extinction_impact
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["comparison_data", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_extinction_impact(**func_kwargs)
            
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

