#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectroscopy_analysis 工具注册模块
使用 gym.tool.EnvironmentTool 为 spectroscopy_analysis 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="voigt_function")
class VoigtFunctionTool(EnvironmentTool):
    """Voigt函数（高斯与洛伦兹卷积）- XPS峰形的标准模型。能更准确地描述XPS峰形，因为它同时考虑了仪器展宽（高斯）和自然线宽（洛伦兹）的贡献。"""
    
    name = "voigt_function"
    description = "Voigt函数（高斯与洛伦兹卷积）- XPS峰形的标准模型。能更准确地描述XPS峰形，因为它同时考虑了仪器展宽（高斯）和自然线宽（洛伦兹）的贡献。"
    arguments = {
        "x": {"type": "array", "description": "结合能数组 / eV"},
        "amplitude": {"type": "number", "description": "峰强度"},
        "center": {"type": "number", "description": "峰位中心 / eV"},
        "sigma": {"type": "number", "description": "高斯展宽参数 / eV（仪器分辨率）"},
        "gamma": {"type": "number", "description": "洛伦兹展宽参数 / eV（自然线宽）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 voigt_function 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            x = args.get("x")
            if x is None:
                return Observation(self.name, "错误: 缺少必需参数 x")
            amplitude = args.get("amplitude")
            if amplitude is None:
                return Observation(self.name, "错误: 缺少必需参数 amplitude")
            center = args.get("center")
            if center is None:
                return Observation(self.name, "错误: 缺少必需参数 center")
            sigma = args.get("sigma")
            if sigma is None:
                return Observation(self.name, "错误: 缺少必需参数 sigma")
            gamma = args.get("gamma")
            if gamma is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma")
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import voigt_function
            
            # 调用函数
            result = voigt_function(x, amplitude, center, sigma, gamma)
            
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


@Toolbox.register(name="detect_peaks_auto")
class DetectPeaksAutoTool(EnvironmentTool):
    """自动检测XPS谱图中的峰位。使用scipy的find_peaks算法，基于峰的突出度（prominence）识别主峰，适用于信噪比较高的XPS数据。"""
    
    name = "detect_peaks_auto"
    description = "自动检测XPS谱图中的峰位。使用scipy的find_peaks算法，基于峰的突出度（prominence）识别主峰，适用于信噪比较高的XPS数据。"
    arguments = {
        "binding_energy": {"type": "array", "description": "结合能数组 / eV，长度N"},
        "intensity": {"type": "array", "description": "强度数组 / counts，长度N"},
        "prominence_threshold": {"type": "number", "description": "峰突出度阈值（相对于最大强度），范围0-1，默认0.05"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_peaks_auto 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            binding_energy = args.get("binding_energy")
            if binding_energy is None:
                return Observation(self.name, "错误: 缺少必需参数 binding_energy")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            prominence_threshold = args.get("prominence_threshold", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import detect_peaks_auto
            
            # 调用函数
            result = detect_peaks_auto(binding_energy, intensity, prominence_threshold)
            
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


@Toolbox.register(name="fit_single_peak")
class FitSinglePeakTool(EnvironmentTool):
    """对单个XPS峰进行拟合。支持高斯、洛伦兹和Voigt三种峰形模型，返回拟合参数和拟合优度。"""
    
    name = "fit_single_peak"
    description = "对单个XPS峰进行拟合。支持高斯、洛伦兹和Voigt三种峰形模型，返回拟合参数和拟合优度。"
    arguments = {
        "binding_energy": {"type": "array", "description": "结合能数组 / eV"},
        "intensity": {"type": "array", "description": "强度数组 / counts"},
        "peak_center_guess": {"type": "number", "description": "峰位初始猜测值 / eV"},
        "peak_type": {"type": "string", "description": "峰形类型", "enum": ["gaussian", "lorentzian", "voigt"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fit_single_peak 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            binding_energy = args.get("binding_energy")
            if binding_energy is None:
                return Observation(self.name, "错误: 缺少必需参数 binding_energy")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            peak_center_guess = args.get("peak_center_guess")
            if peak_center_guess is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_center_guess")
            peak_type = args.get("peak_type", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import fit_single_peak
            
            # 调用函数
            result = fit_single_peak(binding_energy, intensity, peak_center_guess, peak_type)
            
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


@Toolbox.register(name="identify_element_from_binding_energy")
class IdentifyElementFromBindingEnergyTool(EnvironmentTool):
    """根据结合能从数据库匹配元素和轨道。在XPS_DATABASE中搜索与给定结合能匹配的元素核心能级，考虑不同氧化态的化学位移。"""
    
    name = "identify_element_from_binding_energy"
    description = "根据结合能从数据库匹配元素和轨道。在XPS_DATABASE中搜索与给定结合能匹配的元素核心能级，考虑不同氧化态的化学位移。"
    arguments = {
        "binding_energy": {"type": "number", "description": "实验测得的结合能 / eV"},
        "tolerance": {"type": "number", "description": "匹配容差 / eV，默认±0.5 eV"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_element_from_binding_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            binding_energy = args.get("binding_energy")
            if binding_energy is None:
                return Observation(self.name, "错误: 缺少必需参数 binding_energy")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import identify_element_from_binding_energy
            
            # 调用函数
            result = identify_element_from_binding_energy(binding_energy, tolerance)
            
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


@Toolbox.register(name="calculate_spin_orbit_doublet")
class CalculateSpinOrbitDoubletTool(EnvironmentTool):
    """根据自旋-轨道耦合规则计算双峰参数。对于p、d、f轨道，自旋-轨道耦合导致能级分裂为j=l±1/2两个子能级，强度比由统计权重(2j+1)决定。"""
    
    name = "calculate_spin_orbit_doublet"
    description = "根据自旋-轨道耦合规则计算双峰参数。对于p、d、f轨道，自旋-轨道耦合导致能级分裂为j=l±1/2两个子能级，强度比由统计权重(2j+1)决定。"
    arguments = {
        "peak1_be": {"type": "number", "description": "第一个峰（低结合能）的峰位 / eV"},
        "peak1_intensity": {"type": "number", "description": "第一个峰的强度"},
        "element": {"type": "string", "description": "元素符号"},
        "orbital": {"type": "string", "description": "轨道标识（如'4f'）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_spin_orbit_doublet 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak1_be = args.get("peak1_be")
            if peak1_be is None:
                return Observation(self.name, "错误: 缺少必需参数 peak1_be")
            peak1_intensity = args.get("peak1_intensity")
            if peak1_intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 peak1_intensity")
            element = args.get("element")
            if element is None:
                return Observation(self.name, "错误: 缺少必需参数 element")
            orbital = args.get("orbital")
            if orbital is None:
                return Observation(self.name, "错误: 缺少必需参数 orbital")
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import calculate_spin_orbit_doublet
            
            # 调用函数
            result = calculate_spin_orbit_doublet(peak1_be, peak1_intensity, element, orbital)
            
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


@Toolbox.register(name="fit_doublet_peaks")
class FitDoubletPeaksTool(EnvironmentTool):
    """拟合自旋-轨道双峰结构。同时拟合两个耦合峰，约束它们的能量分裂和强度比符合物理规律。"""
    
    name = "fit_doublet_peaks"
    description = "拟合自旋-轨道双峰结构。同时拟合两个耦合峰，约束它们的能量分裂和强度比符合物理规律。"
    arguments = {
        "binding_energy": {"type": "array", "description": "结合能数组 / eV"},
        "intensity": {"type": "array", "description": "强度数组 / counts"},
        "element": {"type": "string", "description": "元素符号"},
        "orbital": {"type": "string", "description": "轨道标识（如'4f'）"},
        "initial_guess_be": {"type": "number", "description": "低结合能峰的初始猜测 / eV"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fit_doublet_peaks 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            binding_energy = args.get("binding_energy")
            if binding_energy is None:
                return Observation(self.name, "错误: 缺少必需参数 binding_energy")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            element = args.get("element")
            if element is None:
                return Observation(self.name, "错误: 缺少必需参数 element")
            orbital = args.get("orbital")
            if orbital is None:
                return Observation(self.name, "错误: 缺少必需参数 orbital")
            initial_guess_be = args.get("initial_guess_be")
            if initial_guess_be is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_guess_be")
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import fit_doublet_peaks
            
            # 调用函数
            result = fit_doublet_peaks(binding_energy, intensity, element, orbital, initial_guess_be)
            
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


@Toolbox.register(name="visualize_xps_spectrum")
class VisualizeXpsSpectrumTool(EnvironmentTool):
    """XPS谱图专业可视化工具。生成带拟合曲线的XPS谱图、原始XPS谱图或多峰对比图。"""
    
    name = "visualize_xps_spectrum"
    description = "XPS谱图专业可视化工具。生成带拟合曲线的XPS谱图、原始XPS谱图或多峰对比图。"
    arguments = {
        "data": {"type": "object", "description": "包含XPS数据的字典，必须包含：'binding_energy'（结合能数组）、'intensity'（强度数组）、'fitted_peaks'（拟合峰数据，可选）"},
        "domain": {"type": "string", "description": "固定为'materials'"},
        "vis_type": {"type": "string", "description": "可视化类型：'xps_fitted'（带拟合曲线的XPS谱图）、'xps_raw'（原始XPS谱图）、'peak_comparison'（多峰对比图）", "enum": ["xps_fitted", "xps_raw", "peak_comparison"]},
        "save_dir": {"type": "string", "description": "保存目录，默认'./images/'"},
        "filename": {"type": "string", "description": "文件名（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_xps_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            data = args.get("data")
            if data is None:
                return Observation(self.name, "错误: 缺少必需参数 data")
            domain = args.get("domain", None)
            vis_type = args.get("vis_type", None)
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.spectroscopy_analysis.xps_spectroscopy_toolkit_0001 import visualize_xps_spectrum
            
            # 调用函数
            result = visualize_xps_spectrum(data, domain, vis_type, save_dir, filename)
            
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

def register_spectroscopy_analysis_tools(environment):
    """
    将所有 spectroscopy_analysis 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

