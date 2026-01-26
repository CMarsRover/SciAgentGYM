#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
x-ray_diffraction_analysis 工具注册模块
使用 gym.tool.EnvironmentTool 为 x-ray_diffraction_analysis 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M020_0004 import *  # 动态导入
# from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M022_0000 import *  # 动态导入
# from toolkits.materials_science.x-ray_diffraction_analysis.materials_xrd_toolkit_M021_0001 import *  # 动态导入
# from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import *  # 动态导入
# from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import *  # 动态导入
# from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="detect_peaks_from_pattern")
class DetectPeaksFromPatternTool(EnvironmentTool):
    """从XRD图谱中自动检测衍射峰位置。使用scipy的find_peaks算法识别局部最大值，通过prominence参数过滤噪声峰。"""
    
    name = "detect_peaks_from_pattern"
    description = "从XRD图谱中自动检测衍射峰位置。使用scipy的find_peaks算法识别局部最大值，通过prominence参数过滤噪声峰。"
    arguments = {
        "two_theta": {"type": "array", "description": "2θ角度数组 (degrees)，范围通常10-90°"},
        "intensity": {"type": "array", "description": "对应的衍射强度数组 (a.u.)"},
        "prominence": {"type": "number", "description": "峰的显著性阈值，默认5.0（相对强度单位）"},
        "min_distance": {"type": "integer", "description": "相邻峰的最小间隔（数据点数），默认5"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_peaks_from_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            prominence = args.get("prominence", None)
            min_distance = args.get("min_distance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import detect_peaks_from_pattern
            
            # 调用函数
            result = detect_peaks_from_pattern(two_theta, intensity, prominence, min_distance)
            
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


@Toolbox.register(name="calculate_d_spacing")
class CalculateDSpacingTool(EnvironmentTool):
    """根据Bragg定律计算晶面间距d。应用Bragg定律: nλ = 2d·sinθ，其中n=1（一级衍射）。"""
    
    name = "calculate_d_spacing"
    description = "根据Bragg定律计算晶面间距d。应用Bragg定律: nλ = 2d·sinθ，其中n=1（一级衍射）。"
    arguments = {
        "two_theta": {"type": "string", "description": "2θ衍射角 (degrees)，可以是单值或数组"},
        "wavelength": {"type": "number", "description": "X射线波长 (Å)，默认Cu Kα = 1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_d_spacing 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import calculate_d_spacing
            
            # 调用函数
            result = calculate_d_spacing(two_theta, wavelength)
            
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


@Toolbox.register(name="assign_miller_indices_tetragonal")
class AssignMillerIndicesTetragonalTool(EnvironmentTool):
    """为四方晶系的d间距分配Miller指数(hkl)。使用四方晶系公式: 1/d² = (h²+k²)/a² + l²/c²，遍历可能的hkl组合。"""
    
    name = "assign_miller_indices_tetragonal"
    description = "为四方晶系的d间距分配Miller指数(hkl)。使用四方晶系公式: 1/d² = (h²+k²)/a² + l²/c²，遍历可能的hkl组合。"
    arguments = {
        "d_spacing": {"type": "number", "description": "实验测得的晶面间距 (Å)"},
        "a": {"type": "number", "description": "四方晶系的a晶格参数 (Å)"},
        "c": {"type": "number", "description": "四方晶系的c晶格参数 (Å)"},
        "tolerance": {"type": "number", "description": "匹配容差，默认0.02 (Å)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 assign_miller_indices_tetragonal 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            d_spacing = args.get("d_spacing")
            if d_spacing is None:
                return Observation(self.name, "错误: 缺少必需参数 d_spacing")
            a = args.get("a")
            if a is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            c = args.get("c")
            if c is None:
                return Observation(self.name, "错误: 缺少必需参数 c")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import assign_miller_indices_tetragonal
            
            # 调用函数
            result = assign_miller_indices_tetragonal(d_spacing, a, c, tolerance)
            
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


@Toolbox.register(name="refine_lattice_parameters_tetragonal")
class RefineLatticeParametersTetragonalTool(EnvironmentTool):
    """通过最小二乘法精修四方晶系的晶格参数。使用scipy.optimize.least_squares最小化理论与实验d值的差异。"""
    
    name = "refine_lattice_parameters_tetragonal"
    description = "通过最小二乘法精修四方晶系的晶格参数。使用scipy.optimize.least_squares最小化理论与实验d值的差异。"
    arguments = {
        "peak_data": {"type": "array", "description": "峰数据列表，每项包含 {'two_theta': float, 'hkl': (h,k,l)}"},
        "initial_a": {"type": "number", "description": "a参数初始值 (Å)"},
        "initial_c": {"type": "number", "description": "c参数初始值 (Å)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 refine_lattice_parameters_tetragonal 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_data = args.get("peak_data")
            if peak_data is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_data")
            initial_a = args.get("initial_a")
            if initial_a is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_a")
            initial_c = args.get("initial_c")
            if initial_c is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_c")
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import refine_lattice_parameters_tetragonal
            
            # 调用函数
            result = refine_lattice_parameters_tetragonal(peak_data, initial_a, initial_c)
            
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


@Toolbox.register(name="fetch_structure_from_mp")
class FetchStructureFromMpTool(EnvironmentTool):
    """从Materials Project数据库获取晶体结构信息。通过mp-api访问Materials Project免费数据库，获取化学式对应的晶体结构。"""
    
    name = "fetch_structure_from_mp"
    description = "从Materials Project数据库获取晶体结构信息。通过mp-api访问Materials Project免费数据库，获取化学式对应的晶体结构。"
    arguments = {
        "formula": {"type": "string", "description": "化学式，如 'SnO2', 'TiO2'"},
        "api_key": {"type": "string", "description": "Materials Project API密钥（可选，使用环境变量MP_API_KEY）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_structure_from_mp 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            formula = args.get("formula")
            if formula is None:
                return Observation(self.name, "错误: 缺少必需参数 formula")
            api_key = args.get("api_key", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import fetch_structure_from_mp
            
            # 调用函数
            result = fetch_structure_from_mp(formula, api_key)
            
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


@Toolbox.register(name="analyze_xrd_pattern")
class AnalyzeXrdPatternTool(EnvironmentTool):
    """完整分析XRD图谱：峰检测、相鉴定、Miller指数标定、晶格参数计算。组合调用多个原子工具函数，实现从原始数据到晶格参数的全流程分析。"""
    
    name = "analyze_xrd_pattern"
    description = "完整分析XRD图谱：峰检测、相鉴定、Miller指数标定、晶格参数计算。组合调用多个原子工具函数，实现从原始数据到晶格参数的全流程分析。"
    arguments = {
        "two_theta": {"type": "array", "description": "2θ角度数组 (degrees)"},
        "intensity": {"type": "array", "description": "衍射强度数组 (a.u.)"},
        "expected_phase": {"type": "string", "description": "预期相的化学式，默认'SnO2'"},
        "prominence": {"type": "number", "description": "峰检测显著性阈值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_xrd_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            expected_phase = args.get("expected_phase", None)
            prominence = args.get("prominence", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import analyze_xrd_pattern
            
            # 调用函数
            result = analyze_xrd_pattern(two_theta, intensity, expected_phase, prominence)
            
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


@Toolbox.register(name="compare_experimental_theoretical_xrd")
class CompareExperimentalTheoreticalXrdTool(EnvironmentTool):
    """对比实验XRD图谱与理论计算图谱。使用pymatgen的XRDCalculator生成理论图谱，与实验数据对比。"""
    
    name = "compare_experimental_theoretical_xrd"
    description = "对比实验XRD图谱与理论计算图谱。使用pymatgen的XRDCalculator生成理论图谱，与实验数据对比。"
    arguments = {
        "experimental_2theta": {"type": "array", "description": "实验2θ数组 (degrees)"},
        "experimental_intensity": {"type": "array", "description": "实验强度数组 (a.u.)"},
        "structure": {"type": "object", "description": "pymatgen Structure对象"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_experimental_theoretical_xrd 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            experimental_2theta = args.get("experimental_2theta")
            if experimental_2theta is None:
                return Observation(self.name, "错误: 缺少必需参数 experimental_2theta")
            experimental_intensity = args.get("experimental_intensity")
            if experimental_intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 experimental_intensity")
            structure = args.get("structure")
            if structure is None:
                return Observation(self.name, "错误: 缺少必需参数 structure")
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import compare_experimental_theoretical_xrd
            
            # 调用函数
            result = compare_experimental_theoretical_xrd(experimental_2theta, experimental_intensity, structure)
            
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


@Toolbox.register(name="plot_xrd_analysis")
class PlotXrdAnalysisTool(EnvironmentTool):
    """生成XRD分析的交互式可视化图表。使用plotly创建包含实验数据、峰标注和理论对比的多子图。"""
    
    name = "plot_xrd_analysis"
    description = "生成XRD分析的交互式可视化图表。使用plotly创建包含实验数据、峰标注和理论对比的多子图。"
    arguments = {
        "two_theta": {"type": "array", "description": "实验2θ数组"},
        "intensity": {"type": "array", "description": "实验强度数组"},
        "peak_assignments": {"type": "array", "description": "峰标注列表 [{'2theta': float, 'hkl': tuple}, ...]"},
        "theoretical_pattern": {"type": "object", "description": "理论图谱数据（可选）"},
        "save_path": {"type": "string", "description": "图片保存路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_xrd_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            peak_assignments = args.get("peak_assignments")
            if peak_assignments is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_assignments")
            theoretical_pattern = args.get("theoretical_pattern", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0001 import plot_xrd_analysis
            
            # 调用函数
            result = plot_xrd_analysis(two_theta, intensity, peak_assignments, theoretical_pattern, save_path)
            
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


@Toolbox.register(name="extract_peaks_from_pattern")
class ExtractPeaksFromPatternTool(EnvironmentTool):
    """从XRD图谱中提取衍射峰位置和强度。使用scipy.signal.find_peaks算法识别局部最大值，通过prominence参数过滤噪声峰。"""
    
    name = "extract_peaks_from_pattern"
    description = "从XRD图谱中提取衍射峰位置和强度。使用scipy.signal.find_peaks算法识别局部最大值，通过prominence参数过滤噪声峰。"
    arguments = {
        "two_theta": {"type": "array", "description": "2θ角度数组 (degrees), 范围通常10-90°"},
        "intensity": {"type": "array", "description": "对应的衍射强度数组 (a.u.)"},
        "prominence": {"type": "number", "description": "峰显著性阈值，默认5.0 a.u.，用于过滤弱峰"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 extract_peaks_from_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            prominence = args.get("prominence", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import extract_peaks_from_pattern
            
            # 调用函数
            result = extract_peaks_from_pattern(two_theta, intensity, prominence)
            
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


@Toolbox.register(name="calculate_peak_matching_score")
class CalculatePeakMatchingScoreTool(EnvironmentTool):
    """计算复合材料峰与参考材料峰的匹配度。采用最近邻匹配算法，对于复合材料中的每个峰，在容差范围内寻找参考材料的对应峰。"""
    
    name = "calculate_peak_matching_score"
    description = "计算复合材料峰与参考材料峰的匹配度。采用最近邻匹配算法，对于复合材料中的每个峰，在容差范围内寻找参考材料的对应峰。"
    arguments = {
        "composite_peaks": {"type": "array", "description": "复合材料的峰位数组 (degrees)"},
        "reference_peaks": {"type": "array", "description": "参考材料的峰位数组 (degrees)"},
        "tolerance": {"type": "number", "description": "峰位匹配容差 (degrees)，默认0.5°"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_peak_matching_score 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            composite_peaks = args.get("composite_peaks")
            if composite_peaks is None:
                return Observation(self.name, "错误: 缺少必需参数 composite_peaks")
            reference_peaks = args.get("reference_peaks")
            if reference_peaks is None:
                return Observation(self.name, "错误: 缺少必需参数 reference_peaks")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import calculate_peak_matching_score
            
            # 调用函数
            result = calculate_peak_matching_score(composite_peaks, reference_peaks, tolerance)
            
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


@Toolbox.register(name="normalize_intensity")
class NormalizeIntensityTool(EnvironmentTool):
    """归一化XRD强度数据。支持最大值归一化和总和归一化两种方法，便于不同图谱间的强度对比。"""
    
    name = "normalize_intensity"
    description = "归一化XRD强度数据。支持最大值归一化和总和归一化两种方法，便于不同图谱间的强度对比。"
    arguments = {
        "intensity": {"type": "array", "description": "原始强度数组 (a.u.)"},
        "method": {"type": "string", "description": "归一化方法", "enum": ["max", "sum"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 normalize_intensity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            method = args.get("method", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import normalize_intensity
            
            # 调用函数
            result = normalize_intensity(intensity, method)
            
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


@Toolbox.register(name="identify_phases_in_composite")
class IdentifyPhasesInCompositeTool(EnvironmentTool):
    """识别复合材料中存在的晶相组成。通过对比复合材料的XRD峰位与多个参考材料的峰位，计算匹配度并筛选出最可能存在的相。"""
    
    name = "identify_phases_in_composite"
    description = "识别复合材料中存在的晶相组成。通过对比复合材料的XRD峰位与多个参考材料的峰位，计算匹配度并筛选出最可能存在的相。"
    arguments = {
        "composite_data": {"type": "object", "description": "复合材料XRD数据，格式{'two_theta': array, 'intensity': array}"},
        "reference_materials": {"type": "object", "description": "参考材料字典，格式{材料名: {'two_theta': array, 'intensity': array}}"},
        "min_match_ratio": {"type": "number", "description": "最小匹配率阈值，默认0.6"},
        "num_phases": {"type": "integer", "description": "预期相数量，默认3"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_phases_in_composite 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            composite_data = args.get("composite_data")
            if composite_data is None:
                return Observation(self.name, "错误: 缺少必需参数 composite_data")
            reference_materials = args.get("reference_materials")
            if reference_materials is None:
                return Observation(self.name, "错误: 缺少必需参数 reference_materials")
            min_match_ratio = args.get("min_match_ratio", None)
            num_phases = args.get("num_phases", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import identify_phases_in_composite
            
            # 调用函数
            result = identify_phases_in_composite(composite_data, reference_materials, min_match_ratio, num_phases)
            
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


@Toolbox.register(name="analyze_xrd_pattern_comprehensive")
class AnalyzeXrdPatternComprehensiveTool(EnvironmentTool):
    """XRD图谱综合分析（包含峰提取、相识别、数据归一化）。这是一个高层次的分析函数，整合了峰提取、归一化和相识别功能。"""
    
    name = "analyze_xrd_pattern_comprehensive"
    description = "XRD图谱综合分析（包含峰提取、相识别、数据归一化）。这是一个高层次的分析函数，整合了峰提取、归一化和相识别功能。"
    arguments = {
        "composite_data": {"type": "object", "description": "复合材料XRD数据，格式{'two_theta': array, 'intensity': array}"},
        "reference_materials": {"type": "object", "description": "参考材料数据字典，格式{材料名: {'two_theta': array, 'intensity': array}}"},
        "normalize": {"type": "boolean", "description": "是否进行强度归一化，默认true"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_xrd_pattern_comprehensive 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            composite_data = args.get("composite_data")
            if composite_data is None:
                return Observation(self.name, "错误: 缺少必需参数 composite_data")
            reference_materials = args.get("reference_materials")
            if reference_materials is None:
                return Observation(self.name, "错误: 缺少必需参数 reference_materials")
            normalize = args.get("normalize", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import analyze_xrd_pattern_comprehensive
            
            # 调用函数
            result = analyze_xrd_pattern_comprehensive(composite_data, reference_materials, normalize)
            
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


@Toolbox.register(name="visualize_xrd_comparison")
class VisualizeXrdComparisonTool(EnvironmentTool):
    """材料科学领域专属可视化工具 - XRD图谱对比。生成XRD叠加图或匹配度柱状图。"""
    
    name = "visualize_xrd_comparison"
    description = "材料科学领域专属可视化工具 - XRD图谱对比。生成XRD叠加图或匹配度柱状图。"
    arguments = {
        "data": {"type": "object", "description": "要可视化的数据，包含composite和references或match_scores"},
        "domain": {"type": "string", "description": "领域类型，固定为'materials'"},
        "vis_type": {"type": "string", "description": "可视化类型", "enum": ["xrd_pattern", "match_scores"]},
        "save_dir": {"type": "string", "description": "保存目录，默认'./images/'"},
        "filename": {"type": "string", "description": "文件名（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_xrd_comparison 操作"""
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
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0002 import visualize_xrd_comparison
            
            # 调用函数
            result = visualize_xrd_comparison(data, domain, vis_type, save_dir, filename)
            
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


@Toolbox.register(name="bragg_law_d_spacing")
class BraggLawDSpacingTool(EnvironmentTool):
    """根据Bragg定律计算晶面间距。使用Bragg定律 nλ = 2d·sinθ 计算晶面间距，假设一级衍射(n=1)。"""
    
    name = "bragg_law_d_spacing"
    description = "根据Bragg定律计算晶面间距。使用Bragg定律 nλ = 2d·sinθ 计算晶面间距，假设一级衍射(n=1)。"
    arguments = {
        "two_theta": {"type": "number", "description": "衍射角2θ，单位度(°)，范围10-90"},
        "wavelength": {"type": "number", "description": "X射线波长，单位埃(Å)，默认Cu Kα = 1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 bragg_law_d_spacing 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import bragg_law_d_spacing
            
            # 调用函数
            result = bragg_law_d_spacing(two_theta, wavelength)
            
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


@Toolbox.register(name="cubic_lattice_parameter")
class CubicLatticeParameterTool(EnvironmentTool):
    """从晶面间距和Miller指数计算立方晶系晶格参数。对于立方晶系，晶格参数 a = d·√(h²+k²+l²)。"""
    
    name = "cubic_lattice_parameter"
    description = "从晶面间距和Miller指数计算立方晶系晶格参数。对于立方晶系，晶格参数 a = d·√(h²+k²+l²)。"
    arguments = {
        "d_spacing": {"type": "number", "description": "晶面间距，单位埃(Å)"},
        "hkl": {"type": "array", "description": "Miller指数元组 (h, k, l)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 cubic_lattice_parameter 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            d_spacing = args.get("d_spacing")
            if d_spacing is None:
                return Observation(self.name, "错误: 缺少必需参数 d_spacing")
            hkl = args.get("hkl")
            if hkl is None:
                return Observation(self.name, "错误: 缺少必需参数 hkl")
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import cubic_lattice_parameter
            
            # 调用函数
            result = cubic_lattice_parameter(d_spacing, hkl)
            
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


@Toolbox.register(name="assign_fcc_miller_indices")
class AssignFccMillerIndicesTool(EnvironmentTool):
    """为FCC结构的衍射峰自动分配Miller指数。基于FCC晶格系统消光规则(h,k,l全奇或全偶)，按h²+k²+l²递增顺序分配指数。"""
    
    name = "assign_fcc_miller_indices"
    description = "为FCC结构的衍射峰自动分配Miller指数。基于FCC晶格系统消光规则(h,k,l全奇或全偶)，按h²+k²+l²递增顺序分配指数。"
    arguments = {
        "peak_positions": {"type": "array", "description": "衍射峰2θ位置列表，单位度(°)"},
        "wavelength": {"type": "number", "description": "X射线波长，单位埃(Å)，默认Cu Kα = 1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 assign_fcc_miller_indices 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_positions = args.get("peak_positions")
            if peak_positions is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_positions")
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import assign_fcc_miller_indices
            
            # 调用函数
            result = assign_fcc_miller_indices(peak_positions, wavelength)
            
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


@Toolbox.register(name="refine_lattice_parameter_lsq")
class RefineLatticeParameterLsqTool(EnvironmentTool):
    """使用最小二乘法精修立方晶格参数。通过最小化理论2θ与实验2θ的残差平方和，优化晶格参数a。"""
    
    name = "refine_lattice_parameter_lsq"
    description = "使用最小二乘法精修立方晶格参数。通过最小化理论2θ与实验2θ的残差平方和，优化晶格参数a。"
    arguments = {
        "peak_data": {"type": "array", "description": "峰位置与Miller指数对列表，格式[(2θ, (h,k,l))...]"},
        "initial_guess": {"type": "number", "description": "晶格参数初始猜测值(Å)，默认4.5"},
        "wavelength": {"type": "number", "description": "X射线波长(Å)，默认Cu Kα = 1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 refine_lattice_parameter_lsq 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_data = args.get("peak_data")
            if peak_data is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_data")
            initial_guess = args.get("initial_guess", None)
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import refine_lattice_parameter_lsq
            
            # 调用函数
            result = refine_lattice_parameter_lsq(peak_data, initial_guess, wavelength)
            
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


@Toolbox.register(name="fetch_structure_from_mp")
class FetchStructureFromMpTool(EnvironmentTool):
    """从Materials Project数据库获取晶体结构。使用mp-api访问Materials Project数据库，获取标准晶体结构数据。"""
    
    name = "fetch_structure_from_mp"
    description = "从Materials Project数据库获取晶体结构。使用mp-api访问Materials Project数据库，获取标准晶体结构数据。"
    arguments = {
        "material_id": {"type": "string", "description": "Materials Project ID (如 'mp-1234') 或化学式 (如 'BP')"},
        "api_key": {"type": "string", "description": "MP API密钥，若为None则尝试使用环境变量"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_structure_from_mp 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            material_id = args.get("material_id")
            if material_id is None:
                return Observation(self.name, "错误: 缺少必需参数 material_id")
            api_key = args.get("api_key", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import fetch_structure_from_mp
            
            # 调用函数
            result = fetch_structure_from_mp(material_id, api_key)
            
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


@Toolbox.register(name="analyze_xrd_pattern")
class AnalyzeXrdPatternTool(EnvironmentTool):
    """完整分析XRD图谱：峰标定、晶格参数计算、精修。组合调用峰位标定、晶格参数计算和最小二乘精修功能。"""
    
    name = "analyze_xrd_pattern"
    description = "完整分析XRD图谱：峰标定、晶格参数计算、精修。组合调用峰位标定、晶格参数计算和最小二乘精修功能。"
    arguments = {
        "peak_positions": {"type": "array", "description": "实验测得的衍射峰2θ位置列表(°)"},
        "structure_type": {"type": "string", "description": "晶体结构类型，目前支持'FCC'", "enum": ["FCC"]},
        "wavelength": {"type": "number", "description": "X射线波长(Å)，默认Cu Kα = 1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_xrd_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_positions = args.get("peak_positions")
            if peak_positions is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_positions")
            structure_type = args.get("structure_type", None)
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import analyze_xrd_pattern
            
            # 调用函数
            result = analyze_xrd_pattern(peak_positions, structure_type, wavelength)
            
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


@Toolbox.register(name="compare_with_database")
class CompareWithDatabaseTool(EnvironmentTool):
    """将实验XRD与数据库标准谱图对比。从Materials Project获取标准结构，计算理论XRD，与实验数据对比。"""
    
    name = "compare_with_database"
    description = "将实验XRD与数据库标准谱图对比。从Materials Project获取标准结构，计算理论XRD，与实验数据对比。"
    arguments = {
        "experimental_peaks": {"type": "array", "description": "实验衍射峰位置(°)"},
        "material_formula": {"type": "string", "description": "材料化学式"},
        "api_key": {"type": "string", "description": "Materials Project API密钥"},
        "wavelength": {"type": "number", "description": "X射线波长(Å)，默认Cu Kα = 1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_with_database 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            experimental_peaks = args.get("experimental_peaks")
            if experimental_peaks is None:
                return Observation(self.name, "错误: 缺少必需参数 experimental_peaks")
            material_formula = args.get("material_formula")
            if material_formula is None:
                return Observation(self.name, "错误: 缺少必需参数 material_formula")
            api_key = args.get("api_key", None)
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import compare_with_database
            
            # 调用函数
            result = compare_with_database(experimental_peaks, material_formula, api_key, wavelength)
            
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


@Toolbox.register(name="plot_xrd_analysis")
class PlotXrdAnalysisTool(EnvironmentTool):
    """生成XRD图谱分析可视化图表。绘制衍射峰并标注Miller指数，支持交互式和静态两种模式。"""
    
    name = "plot_xrd_analysis"
    description = "生成XRD图谱分析可视化图表。绘制衍射峰并标注Miller指数，支持交互式和静态两种模式。"
    arguments = {
        "peak_positions": {"type": "array", "description": "峰位置2θ(°)"},
        "intensities": {"type": "array", "description": "峰强度（归一化）"},
        "assignments": {"type": "array", "description": "标定结果列表，格式[(2θ, hkl, d)...]"},
        "plot_type": {"type": "string", "description": "绘图类型", "enum": ["interactive", "static"]},
        "save_path": {"type": "string", "description": "图片保存路径，默认'./images/'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_xrd_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_positions = args.get("peak_positions")
            if peak_positions is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_positions")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            assignments = args.get("assignments")
            if assignments is None:
                return Observation(self.name, "错误: 缺少必需参数 assignments")
            plot_type = args.get("plot_type", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.xrd_analysis_toolkit_0003 import plot_xrd_analysis
            
            # 调用函数
            result = plot_xrd_analysis(peak_positions, intensities, assignments, plot_type, save_path)
            
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


@Toolbox.register(name="save_json_data")
class SaveJsonDataTool(EnvironmentTool):
    """将字典数据保存为JSON文件，统一保存在./mid_result/materials目录。"""
    
    name = "save_json_data"
    description = "将字典数据保存为JSON文件，统一保存在./mid_result/materials目录。"
    arguments = {
        "filename": {"type": "string", "description": "文件名（不含路径），如 'peaks.json'"},
        "data": {"type": "object", "description": "需要保存的数据字典，如XRD图谱数据（包含peaks_2theta和intensity数组）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 save_json_data 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            filename = args.get("filename")
            if filename is None:
                return Observation(self.name, "错误: 缺少必需参数 filename")
            data = args.get("data")
            if data is None:
                return Observation(self.name, "错误: 缺少必需参数 data")
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M020_0004 import save_json_data
            
            # 调用函数
            result = save_json_data(filename, data)
            
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


@Toolbox.register(name="identify_phases_by_matching")
class IdentifyPhasesByMatchingTool(EnvironmentTool):
    """从候选参考中识别最可能的晶相，基于峰匹配与可选的强度加权和漂移校准。"""
    
    name = "identify_phases_by_matching"
    description = "从候选参考中识别最可能的晶相，基于峰匹配与可选的强度加权和漂移校准。"
    arguments = {
        "observed_2theta": {"type": "array", "description": "观测峰位（度）"},
        "observed_intensity": {"type": "array", "description": "观测相对强度（可选，用于强度加权）"},
        "candidate_refs": {"type": "array", "description": "候选参考列表，每个包含{name, peaks_2theta, intensity}"},
        "tolerance": {"type": "number", "description": "峰位匹配容差（度），默认0.25"},
        "allow_shift": {"type": "boolean", "description": "是否允许整体角度漂移优化，默认True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_phases_by_matching 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            observed_2theta = args.get("observed_2theta")
            if observed_2theta is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_2theta")
            observed_intensity = args.get("observed_intensity", None)
            candidate_refs = args.get("candidate_refs")
            if candidate_refs is None:
                return Observation(self.name, "错误: 缺少必需参数 candidate_refs")
            tolerance = args.get("tolerance", None)
            allow_shift = args.get("allow_shift", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M020_0004 import identify_phases_by_matching
            
            # 调用函数
            result = identify_phases_by_matching(observed_2theta, observed_intensity, candidate_refs, tolerance, allow_shift)
            
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


@Toolbox.register(name="plot_xrd_pattern")
class PlotXrdPatternTool(EnvironmentTool):
    """绘制并保存XRD图谱（可叠加多条曲线），自动保存到./tool_images目录。"""
    
    name = "plot_xrd_pattern"
    description = "绘制并保存XRD图谱（可叠加多条曲线），自动保存到./tool_images目录。"
    arguments = {
        "patterns": {"type": "object", "description": "曲线字典，形如 {'Composite': {'two_theta': [...], 'intensity': [...]}, 'Ag2O': {...}}"},
        "title": {"type": "string", "description": "图标题"},
        "filename": {"type": "string", "description": "自定义文件名（不含路径和扩展名），默认根据标题自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_xrd_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            patterns = args.get("patterns")
            if patterns is None:
                return Observation(self.name, "错误: 缺少必需参数 patterns")
            title = args.get("title")
            if title is None:
                return Observation(self.name, "错误: 缺少必需参数 title")
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M020_0004 import plot_xrd_pattern
            
            # 调用函数
            result = plot_xrd_pattern(patterns, title, filename)
            
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


@Toolbox.register(name="bragg_law_d_spacing")
class BraggLawDSpacingTool(EnvironmentTool):
    """根据Bragg定律计算晶面间距 d = λ / (2sinθ)。"""
    
    name = "bragg_law_d_spacing"
    description = "根据Bragg定律计算晶面间距 d = λ / (2sinθ)。"
    arguments = {
        "two_theta": {"type": "number", "description": "衍射角2θ（度），范围10-120°"},
        "wavelength": {"type": "number", "description": "X射线波长（Å），默认Cu Kα=1.5406 Å"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 bragg_law_d_spacing 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            wavelength = args.get("wavelength", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_xrd_toolkit_M021_0001 import bragg_law_d_spacing
            
            # 调用函数
            result = bragg_law_d_spacing(two_theta, wavelength)
            
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


@Toolbox.register(name="visualize_xrd_pattern")
class VisualizeXrdPatternTool(EnvironmentTool):
    """绘制XRD衍射图谱并标注峰位，可保存为图片文件。"""
    
    name = "visualize_xrd_pattern"
    description = "绘制XRD衍射图谱并标注峰位，可保存为图片文件。"
    arguments = {
        "two_theta": {"type": "array", "description": "2θ角度列表（度）"},
        "intensity": {"type": "array", "description": "强度列表"},
        "peak_labels": {"type": "array", "description": "峰标注列表，格式[{'position': 2θ, 'label': '(hkl)'}]"},
        "title": {"type": "string", "description": "图表标题"},
        "save_dir": {"type": "string", "description": "图片保存目录"},
        "filename": {"type": "string", "description": "文件名（不含扩展名），默认自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_xrd_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta = args.get("two_theta")
            if two_theta is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            peak_labels = args.get("peak_labels", None)
            title = args.get("title", None)
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_xrd_toolkit_M021_0001 import visualize_xrd_pattern
            
            # 调用函数
            result = visualize_xrd_pattern(two_theta, intensity, peak_labels, title, save_dir, filename)
            
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


@Toolbox.register(name="calculate_scherrer_grain_size")
class CalculateScherrerGrainSizeTool(EnvironmentTool):
    """用Scherrer方程计算平均晶粒尺寸。"""
    
    name = "calculate_scherrer_grain_size"
    description = "用Scherrer方程计算平均晶粒尺寸。"
    arguments = {
        "peak_2theta_deg": {"type": "number", "description": "峰位2θ（度），范围0-180"},
        "fwhm_deg": {"type": "number", "description": "测得FWHM（度），>0"},
        "wavelength_nm": {"type": "number", "description": "X射线波长（nm），如Cu Kα=0.15406"},
        "shape_factor": {"type": "number", "description": "形状因子K，通常0.89-1.0"},
        "instrument_fwhm_deg": {"type": "number", "description": "仪器展宽FWHM（度），默认0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_scherrer_grain_size 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_2theta_deg = args.get("peak_2theta_deg")
            if peak_2theta_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_2theta_deg")
            fwhm_deg = args.get("fwhm_deg")
            if fwhm_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 fwhm_deg")
            wavelength_nm = args.get("wavelength_nm", None)
            shape_factor = args.get("shape_factor", None)
            instrument_fwhm_deg = args.get("instrument_fwhm_deg", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M022_0000 import calculate_scherrer_grain_size
            
            # 调用函数
            result = calculate_scherrer_grain_size(peak_2theta_deg, fwhm_deg, wavelength_nm, shape_factor, instrument_fwhm_deg)
            
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


@Toolbox.register(name="detect_strongest_peak")
class DetectStrongestPeakTool(EnvironmentTool):
    """从XRD数据中识别最强峰（返回峰位与强度）。"""
    
    name = "detect_strongest_peak"
    description = "从XRD数据中识别最强峰（返回峰位与强度）。"
    arguments = {
        "two_theta_deg": {"type": "array", "description": "2θ（度）列表"},
        "intensity": {"type": "array", "description": "对应强度（a.u.）列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_strongest_peak 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta_deg = args.get("two_theta_deg")
            if two_theta_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta_deg")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M022_0000 import detect_strongest_peak
            
            # 调用函数
            result = detect_strongest_peak(two_theta_deg, intensity)
            
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


@Toolbox.register(name="visualize_xrd_pattern")
class VisualizeXrdPatternTool(EnvironmentTool):
    """可视化XRD衍射图谱（Plotly图）。"""
    
    name = "visualize_xrd_pattern"
    description = "可视化XRD衍射图谱（Plotly图）。"
    arguments = {
        "two_theta_deg": {"type": "array", "description": "2θ列表（度）"},
        "intensity": {"type": "array", "description": "强度列表（a.u.）"},
        "title": {"type": "string", "description": "标题"},
        "filename": {"type": "string", "description": "保存文件名，默认自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_xrd_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            two_theta_deg = args.get("two_theta_deg")
            if two_theta_deg is None:
                return Observation(self.name, "错误: 缺少必需参数 two_theta_deg")
            intensity = args.get("intensity")
            if intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 intensity")
            title = args.get("title", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.x-ray_diffraction_analysis.materials_toolkit_M022_0000 import visualize_xrd_pattern
            
            # 调用函数
            result = visualize_xrd_pattern(two_theta_deg, intensity, title, filename)
            
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

def register_x-ray_diffraction_analysis_tools(environment):
    """
    将所有 x-ray_diffraction_analysis 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

