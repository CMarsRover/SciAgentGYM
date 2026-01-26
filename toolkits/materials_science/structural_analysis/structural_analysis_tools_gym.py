#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structural_analysis 工具注册模块
使用 gym.tool.EnvironmentTool 为 structural_analysis 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="parse_band_data_from_file")
class ParseBandDataFromFileTool(EnvironmentTool):
    """从文件中解析能带结构数据，支持VASP/Quantum ESPRESSO/CASTEP等常见格式。"""
    
    name = "parse_band_data_from_file"
    description = "从文件中解析能带结构数据，支持VASP/Quantum ESPRESSO/CASTEP等常见格式。"
    arguments = {
        "filepath": {"type": "string", "description": "能带数据文件路径"},
        "file_format": {"type": "string", "description": "文件格式（'vasp'，'qe'，'castep'，'auto'），默认为'auto'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_band_data_from_file 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            filepath = args.get("filepath")
            if filepath is None:
                return Observation(self.name, "错误: 缺少必需参数 filepath")
            file_format = args.get("file_format", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import parse_band_data_from_file
            
            # 调用函数
            result = parse_band_data_from_file(filepath, file_format)
            
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


@Toolbox.register(name="identify_vbm_cbm")
class IdentifyVbmCbmTool(EnvironmentTool):
    """识别价带顶(VBM)和导带底(CBM)的位置，并计算带隙。"""
    
    name = "identify_vbm_cbm"
    description = "识别价带顶(VBM)和导带底(CBM)的位置，并计算带隙。"
    arguments = {
        "energies": {"type": "array", "description": "能带能量数据，形状为 (n_bands × n_kpoints)"},
        "kpath": {"type": "array", "description": "k点路径坐标，长度与每条能带的点数相同"},
        "fermi_energy": {"type": "number", "description": "费米能级(eV)，默认0.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_vbm_cbm 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            energies = args.get("energies")
            if energies is None:
                return Observation(self.name, "错误: 缺少必需参数 energies")
            kpath = args.get("kpath")
            if kpath is None:
                return Observation(self.name, "错误: 缺少必需参数 kpath")
            fermi_energy = args.get("fermi_energy", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import identify_vbm_cbm
            
            # 调用函数
            result = identify_vbm_cbm(energies, kpath, fermi_energy)
            
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


@Toolbox.register(name="determine_bandgap_type")
class DetermineBandgapTypeTool(EnvironmentTool):
    """判断带隙类型（直接/间接），并给出对应的高对称k点说明与k空间距离。"""
    
    name = "determine_bandgap_type"
    description = "判断带隙类型（直接/间接），并给出对应的高对称k点说明与k空间距离。"
    arguments = {
        "vbm_info": {"type": "object", "description": "VBM信息字典，包含{'energy','kpoint_index','kpoint_value'}等键"},
        "cbm_info": {"type": "object", "description": "CBM信息字典，包含{'energy','kpoint_index','kpoint_value'}等键"},
        "labels": {"type": "array", "description": "高对称点标记列表；每个元素为二元组(label: str, k_index: int)"},
        "k_tolerance": {"type": "number", "description": "k点位置判断容差，默认0.05"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 determine_bandgap_type 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            vbm_info = args.get("vbm_info")
            if vbm_info is None:
                return Observation(self.name, "错误: 缺少必需参数 vbm_info")
            cbm_info = args.get("cbm_info")
            if cbm_info is None:
                return Observation(self.name, "错误: 缺少必需参数 cbm_info")
            labels = args.get("labels")
            if labels is None:
                return Observation(self.name, "错误: 缺少必需参数 labels")
            k_tolerance = args.get("k_tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import determine_bandgap_type
            
            # 调用函数
            result = determine_bandgap_type(vbm_info, cbm_info, labels, k_tolerance)
            
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


@Toolbox.register(name="fetch_material_bandstructure_from_mp")
class FetchMaterialBandstructureFromMpTool(EnvironmentTool):
    """从Materials Project数据库获取材料的能带结构基础信息（带隙、是否直接带隙等）。"""
    
    name = "fetch_material_bandstructure_from_mp"
    description = "从Materials Project数据库获取材料的能带结构基础信息（带隙、是否直接带隙等）。"
    arguments = {
        "material_id": {"type": "string", "description": "Materials Project材料ID（如'mp-149'，'mp-66'）"},
        "api_key": {"type": "string", "description": "MP API密钥（可选，若未提供则尝试从环境变量MP_API_KEY读取）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_material_bandstructure_from_mp 操作"""
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
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import fetch_material_bandstructure_from_mp
            
            # 调用函数
            result = fetch_material_bandstructure_from_mp(material_id, api_key)
            
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


@Toolbox.register(name="compare_bandgaps_batch")
class CompareBandgapsBatchTool(EnvironmentTool):
    """批量对比多个材料的带隙特性，从Materials Project获取数据并统计分析。"""
    
    name = "compare_bandgaps_batch"
    description = "批量对比多个材料的带隙特性，从Materials Project获取数据并统计分析。"
    arguments = {
        "material_ids": {"type": "array", "description": "材料ID列表，例如['mp-149','mp-66',...]"},
        "api_key": {"type": "string", "description": "MP API密钥（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_bandgaps_batch 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            material_ids = args.get("material_ids")
            if material_ids is None:
                return Observation(self.name, "错误: 缺少必需参数 material_ids")
            api_key = args.get("api_key", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import compare_bandgaps_batch
            
            # 调用函数
            result = compare_bandgaps_batch(material_ids, api_key)
            
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


@Toolbox.register(name="calculate_optical_properties")
class CalculateOpticalPropertiesTool(EnvironmentTool):
    """基于带隙特性估算光学性质，包括吸收边波长与激子结合能等。"""
    
    name = "calculate_optical_properties"
    description = "基于带隙特性估算光学性质，包括吸收边波长与激子结合能等。"
    arguments = {
        "bandgap": {"type": "number", "description": "带隙值(eV)"},
        "is_direct": {"type": "boolean", "description": "是否为直接带隙"},
        "effective_mass_e": {"type": "number", "description": "电子有效质量（以自由电子质量m0为单位），默认0.5"},
        "effective_mass_h": {"type": "number", "description": "空穴有效质量（以自由电子质量m0为单位），默认0.5"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_optical_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            bandgap = args.get("bandgap")
            if bandgap is None:
                return Observation(self.name, "错误: 缺少必需参数 bandgap")
            is_direct = args.get("is_direct")
            if is_direct is None:
                return Observation(self.name, "错误: 缺少必需参数 is_direct")
            effective_mass_e = args.get("effective_mass_e", None)
            effective_mass_h = args.get("effective_mass_h", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import calculate_optical_properties
            
            # 调用函数
            result = calculate_optical_properties(bandgap, is_direct, effective_mass_e, effective_mass_h)
            
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


@Toolbox.register(name="visualize_bandstructure")
class VisualizeBandstructureTool(EnvironmentTool):
    """可视化能带结构图，支持标记VBM/CBM并保存为图片文件。"""
    
    name = "visualize_bandstructure"
    description = "可视化能带结构图，支持标记VBM/CBM并保存为图片文件。"
    arguments = {
        "band_data": {"type": "object", "description": "能带数据字典，包含'kpath'、'energies'、'labels'等键"},
        "vbm_cbm_info": {"type": "object", "description": "VBM/CBM信息（可选），用于图中标记"},
        "save_dir": {"type": "string", "description": "图片保存目录，默认'./tool_images/'"},
        "filename": {"type": "string", "description": "保存文件名（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_bandstructure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            band_data = args.get("band_data")
            if band_data is None:
                return Observation(self.name, "错误: 缺少必需参数 band_data")
            vbm_cbm_info = args.get("vbm_cbm_info", None)
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import visualize_bandstructure
            
            # 调用函数
            result = visualize_bandstructure(band_data, vbm_cbm_info, save_dir, filename)
            
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


@Toolbox.register(name="visualize_bandgap_comparison")
class VisualizeBandgapComparisonTool(EnvironmentTool):
    """可视化多个材料的带隙对比柱状图并保存为图片文件。"""
    
    name = "visualize_bandgap_comparison"
    description = "可视化多个材料的带隙对比柱状图并保存为图片文件。"
    arguments = {
        "materials_data": {"type": "array", "description": "材料数据列表；每个元素为字典对象，包含formula、bandgap、is_direct等字段"},
        "save_dir": {"type": "string", "description": "图片保存目录，默认'./tool_images/'"},
        "filename": {"type": "string", "description": "保存文件名（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_bandgap_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            materials_data = args.get("materials_data")
            if materials_data is None:
                return Observation(self.name, "错误: 缺少必需参数 materials_data")
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.structural_analysis.materials_bandstructure_toolkit_M010_0001 import visualize_bandgap_comparison
            
            # 调用函数
            result = visualize_bandgap_comparison(materials_data, save_dir, filename)
            
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

def register_structural_analysis_tools(environment):
    """
    将所有 structural_analysis 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

