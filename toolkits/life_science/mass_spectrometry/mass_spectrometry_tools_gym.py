#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mass_spectrometry 工具注册模块
使用 gym.tool.EnvironmentTool 为 mass_spectrometry 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0050000 import *  # 动态导入
# from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0070000 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="extract_peaks_from_spectrum")
class ExtractPeaksFromSpectrumTool(EnvironmentTool):
    """从质谱数据中提取峰位置和强度"""
    
    name = "extract_peaks_from_spectrum"
    description = "从质谱数据中提取峰位置和强度"
    arguments = {
        "mz_values": {"type": "array", "description": "m/z值列表"},
        "intensities": {"type": "array", "description": "相对强度列表（%）"},
        "height_threshold": {"type": "number", "description": "峰高度阈值（相对强度%），默认5.0"},
        "prominence": {"type": "number", "description": "峰突出度阈值，默认2.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 extract_peaks_from_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mz_values = args.get("mz_values")
            if mz_values is None:
                return Observation(self.name, "错误: 缺少必需参数 mz_values")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            height_threshold = args.get("height_threshold", None)
            prominence = args.get("prominence", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0050000 import extract_peaks_from_spectrum
            
            # 调用函数
            result = extract_peaks_from_spectrum(mz_values, intensities, height_threshold, prominence)
            
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


@Toolbox.register(name="find_isotope_cluster")
class FindIsotopeClusterTool(EnvironmentTool):
    """在峰列表中查找同位素簇（M, M+2, M+4等）"""
    
    name = "find_isotope_cluster"
    description = "在峰列表中查找同位素簇（M, M+2, M+4等）"
    arguments = {
        "peak_mz": {"type": "array", "description": "峰的m/z值列表"},
        "peak_intensity": {"type": "array", "description": "峰的强度列表"},
        "base_mz": {"type": "number", "description": "基峰（M峰）的m/z值"},
        "mass_tolerance": {"type": "number", "description": "质量匹配容差（Da），默认0.5"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_isotope_cluster 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            peak_mz = args.get("peak_mz")
            if peak_mz is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_mz")
            peak_intensity = args.get("peak_intensity")
            if peak_intensity is None:
                return Observation(self.name, "错误: 缺少必需参数 peak_intensity")
            base_mz = args.get("base_mz")
            if base_mz is None:
                return Observation(self.name, "错误: 缺少必需参数 base_mz")
            mass_tolerance = args.get("mass_tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0050000 import find_isotope_cluster
            
            # 调用函数
            result = find_isotope_cluster(peak_mz, peak_intensity, base_mz, mass_tolerance)
            
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


@Toolbox.register(name="determine_chlorine_number_from_ratio")
class DetermineChlorineNumberFromRatioTool(EnvironmentTool):
    """根据观测到的M+2/M强度比判定氯原子数量"""
    
    name = "determine_chlorine_number_from_ratio"
    description = "根据观测到的M+2/M强度比判定氯原子数量"
    arguments = {
        "observed_ratio": {"type": "number", "description": "观测到的M+2/M强度比"},
        "max_chlorine": {"type": "integer", "description": "最大考虑的氯原子数，默认5"},
        "tolerance": {"type": "number", "description": "匹配容差，默认0.15"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 determine_chlorine_number_from_ratio 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            observed_ratio = args.get("observed_ratio")
            if observed_ratio is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_ratio")
            max_chlorine = args.get("max_chlorine", None)
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0050000 import determine_chlorine_number_from_ratio
            
            # 调用函数
            result = determine_chlorine_number_from_ratio(observed_ratio, max_chlorine, tolerance)
            
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


@Toolbox.register(name="visualize_isotope_pattern_comparison")
class VisualizeIsotopePatternComparisonTool(EnvironmentTool):
    """可视化观测同位素模式与理论模式的对比"""
    
    name = "visualize_isotope_pattern_comparison"
    description = "可视化观测同位素模式与理论模式的对比"
    arguments = {
        "observed_cluster": {"type": "array", "description": "观测到的同位素簇数据"},
        "num_chlorine": {"type": "integer", "description": "判定的氯原子数"},
        "save_dir": {"type": "string", "description": "保存目录，默认为'./images/'"},
        "filename": {"type": "string", "description": "文件名，默认为'isotope_pattern_comparison.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_isotope_pattern_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            observed_cluster = args.get("observed_cluster")
            if observed_cluster is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_cluster")
            num_chlorine = args.get("num_chlorine")
            if num_chlorine is None:
                return Observation(self.name, "错误: 缺少必需参数 num_chlorine")
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0050000 import visualize_isotope_pattern_comparison
            
            # 调用函数
            result = visualize_isotope_pattern_comparison(observed_cluster, num_chlorine, save_dir, filename)
            
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


@Toolbox.register(name="visualize_mass_spectrum")
class VisualizeMassSpectrumTool(EnvironmentTool):
    """可视化质谱图并高亮同位素峰"""
    
    name = "visualize_mass_spectrum"
    description = "可视化质谱图并高亮同位素峰"
    arguments = {
        "mz_values": {"type": "array", "description": "m/z值列表"},
        "intensities": {"type": "array", "description": "强度列表"},
        "highlighted_peaks": {"type": "array", "description": "需要高亮的峰的m/z值"},
        "save_dir": {"type": "string", "description": "保存目录，默认为'./images/'"},
        "filename": {"type": "string", "description": "文件名，默认为'mass_spectrum.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_mass_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mz_values = args.get("mz_values")
            if mz_values is None:
                return Observation(self.name, "错误: 缺少必需参数 mz_values")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            highlighted_peaks = args.get("highlighted_peaks", None)
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0050000 import visualize_mass_spectrum
            
            # 调用函数
            result = visualize_mass_spectrum(mz_values, intensities, highlighted_peaks, save_dir, filename)
            
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


@Toolbox.register(name="parse_mass_spectrum_peaks")
class ParseMassSpectrumPeaksTool(EnvironmentTool):
    """解析质谱峰数据，提取显著峰。从原始m/z和强度数据中筛选出高于阈值的峰，用于后续分析。"""
    
    name = "parse_mass_spectrum_peaks"
    description = "解析质谱峰数据，提取显著峰。从原始m/z和强度数据中筛选出高于阈值的峰，用于后续分析。"
    arguments = {
        "mz_values": {"type": "array", "description": "m/z值列表，范围通常0-2000"},
        "intensities": {"type": "array", "description": "相对强度列表（%），范围0-100"},
        "intensity_threshold": {"type": "number", "description": "强度阈值（%），默认1.0，低于此值的峰被过滤"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_mass_spectrum_peaks 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mz_values = args.get("mz_values")
            if mz_values is None:
                return Observation(self.name, "错误: 缺少必需参数 mz_values")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            intensity_threshold = args.get("intensity_threshold", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0070000 import parse_mass_spectrum_peaks
            
            # 调用函数
            result = parse_mass_spectrum_peaks(mz_values, intensities, intensity_threshold)
            
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


@Toolbox.register(name="calculate_molecular_properties")
class CalculateMolecularPropertiesTool(EnvironmentTool):
    """计算分子的质谱相关性质。基于SMILES字符串计算分子量、不饱和度、杂原子数等关键参数。"""
    
    name = "calculate_molecular_properties"
    description = "计算分子的质谱相关性质。基于SMILES字符串计算分子量、不饱和度、杂原子数等关键参数。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示，如'CCO'表示乙醇"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_molecular_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0070000 import calculate_molecular_properties
            
            # 调用函数
            result = calculate_molecular_properties(smiles)
            
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


@Toolbox.register(name="predict_fragmentation_pattern")
class PredictFragmentationPatternTool(EnvironmentTool):
    """预测分子的碎片化模式。基于分子结构预测可能的碎片离子m/z值（简化模型）。"""
    
    name = "predict_fragmentation_pattern"
    description = "预测分子的碎片化模式。基于分子结构预测可能的碎片离子m/z值（简化模型）。"
    arguments = {
        "smiles": {"type": "string", "description": "分子SMILES表示"},
        "ionization_mode": {"type": "string", "description": "电离模式，'EI'(电子轰击)或'CI'(化学电离)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 predict_fragmentation_pattern 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            ionization_mode = args.get("ionization_mode", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0070000 import predict_fragmentation_pattern
            
            # 调用函数
            result = predict_fragmentation_pattern(smiles, ionization_mode)
            
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


@Toolbox.register(name="match_spectrum_to_structure")
class MatchSpectrumToStructureTool(EnvironmentTool):
    """将实验质谱与候选分子结构进行匹配。综合分析实验谱图和理论碎片，计算匹配度分数。"""
    
    name = "match_spectrum_to_structure"
    description = "将实验质谱与候选分子结构进行匹配。综合分析实验谱图和理论碎片，计算匹配度分数。"
    arguments = {
        "mz_values": {"type": "array", "description": "实验m/z值列表"},
        "intensities": {"type": "array", "description": "实验强度列表"},
        "candidate_smiles": {"type": "string", "description": "候选分子SMILES"},
        "tolerance": {"type": "number", "description": "m/z匹配容差（Da），默认0.5"},
        "intensity_threshold": {"type": "number", "description": "峰强度阈值（%），默认1.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 match_spectrum_to_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mz_values = args.get("mz_values")
            if mz_values is None:
                return Observation(self.name, "错误: 缺少必需参数 mz_values")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            candidate_smiles = args.get("candidate_smiles")
            if candidate_smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 candidate_smiles")
            tolerance = args.get("tolerance", None)
            intensity_threshold = args.get("intensity_threshold", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0070000 import match_spectrum_to_structure
            
            # 调用函数
            result = match_spectrum_to_structure(mz_values, intensities, candidate_smiles, tolerance, intensity_threshold)
            
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


@Toolbox.register(name="batch_structure_screening")
class BatchStructureScreeningTool(EnvironmentTool):
    """批量筛选候选分子结构。对多个候选结构进行匹配，返回排序后的结果。"""
    
    name = "batch_structure_screening"
    description = "批量筛选候选分子结构。对多个候选结构进行匹配，返回排序后的结果。"
    arguments = {
        "mz_values": {"type": "array", "description": "实验m/z值列表"},
        "intensities": {"type": "array", "description": "实验强度列表"},
        "candidate_smiles_list": {"type": "array", "description": "候选分子SMILES列表"},
        "tolerance": {"type": "number", "description": "m/z匹配容差（Da）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_structure_screening 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            mz_values = args.get("mz_values")
            if mz_values is None:
                return Observation(self.name, "错误: 缺少必需参数 mz_values")
            intensities = args.get("intensities")
            if intensities is None:
                return Observation(self.name, "错误: 缺少必需参数 intensities")
            candidate_smiles_list = args.get("candidate_smiles_list")
            if candidate_smiles_list is None:
                return Observation(self.name, "错误: 缺少必需参数 candidate_smiles_list")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.mass_spectrometry.mass_spectrometry_toolkit_0070000 import batch_structure_screening
            
            # 调用函数
            result = batch_structure_screening(mz_values, intensities, candidate_smiles_list, tolerance)
            
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

def register_mass_spectrometry_tools(environment):
    """
    将所有 mass_spectrometry 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

