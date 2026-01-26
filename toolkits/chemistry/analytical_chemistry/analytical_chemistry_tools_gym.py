#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analytical_chemistry 工具注册模块
使用 gym.tool.EnvironmentTool 为 analytical_chemistry 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool, convert_to_json_serializable
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import *  # 动态导入
# from toolkits.chemistry.analytical_chemistry.molecule_analyzer import *  # 动态导入
# from toolkits.chemistry.analytical_chemistry.molecule_visualize import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="horwitz_trumpet")
class HorwitzTrumpetTool(EnvironmentTool):
    """计算实验室间分析测量的Horwitz喇叭相对标准偏差。"""
    
    name = "horwitz_trumpet"
    description = "计算实验室间分析测量的Horwitz喇叭相对标准偏差。"
    arguments = {
        "concentration": {"type": "number", "description": "分析物浓度(g/g)"},
        "plateau_level": {"type": "number", "description": "低浓度平台阈值，默认1e-7 g/g"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 horwitz_trumpet 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            concentration = args.get("concentration")
            if concentration is None:
                return Observation(self.name, "错误: 缺少必需参数 concentration")
            plateau_level = args.get("plateau_level", None)
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import horwitz_trumpet
            
            # 调用函数
            result = horwitz_trumpet(concentration, plateau_level)
            
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


@Toolbox.register(name="intra_laboratory_rsd")
class IntraLaboratoryRsdTool(EnvironmentTool):
    """基于实验室间RSD估计实验室内RSD，默认因子0.6，可取0.5-0.7。"""
    
    name = "intra_laboratory_rsd"
    description = "基于实验室间RSD估计实验室内RSD，默认因子0.6，可取0.5-0.7。"
    arguments = {
        "interlaboratory_rsd": {"type": "number", "description": "实验室间RSD(%)"},
        "factor": {"type": "number", "description": "转换因子，默认0.6"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 intra_laboratory_rsd 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            interlaboratory_rsd = args.get("interlaboratory_rsd")
            if interlaboratory_rsd is None:
                return Observation(self.name, "错误: 缺少必需参数 interlaboratory_rsd")
            factor = args.get("factor", None)
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import intra_laboratory_rsd
            
            # 调用函数
            result = intra_laboratory_rsd(interlaboratory_rsd, factor)
            
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






@Toolbox.register(name="chem_visualizer")
class ChemVisualizerTool(EnvironmentTool):
    """分子可视化工具，支持批量生成分子的2D结构图和3D结构图。直接生成图片文件到当前目录。"""
    
    name = "chem_visualizer"
    description = "分子可视化工具，支持批量生成分子的2D结构图和3D结构图。直接生成图片文件到当前目录。"
    arguments = {
        "molecules": {"type": "array", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 chem_visualizer 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            molecules = args.get("molecules")
            if molecules is None:
                return Observation(self.name, "错误: 缺少必需参数 molecules")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_visualize import chem_visualizer
            
            # 调用函数
            result = chem_visualizer(molecules)
            
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


@Toolbox.register(name="mol_analyzer_generate_3d")
class MolAnalyzerGenerate3dTool(EnvironmentTool):
    """从SMILES字符串生成分子的3D构象。基于RDKit的嵌入算法生成分子的三维坐标。"""
    
    name = "mol_analyzer_generate_3d"
    description = "从SMILES字符串生成分子的3D构象。基于RDKit的嵌入算法生成分子的三维坐标。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示字符串，或化学名称（会自动转换），例如: 'CCO', 'c1ccccc1', 'aspirin'"},
        "method": {"type": "string", "description": "3D坐标生成方法：'ETKDG'（标准ETKDG方法）、'ETKDGv3'（改进的ETKDGv3方法，推荐）、'basic'（基本嵌入方法）", "enum": ["ETKDG", "ETKDGv3", "basic"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 mol_analyzer_generate_3d 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            method = args.get("method")
            if method is None:
                return Observation(self.name, "错误: 缺少必需参数 method")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import mol_analyzer_generate_3d
            
            # 调用函数
            result = mol_analyzer_generate_3d(smiles, method)
            
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






@Toolbox.register(name="generate_multiple_conformers_with_optimization")
class GenerateMultipleConformersWithOptimizationTool(EnvironmentTool):
    """为分子生成多个构象并进行能量优化。集成工具函数，用于处理需要多构象分析的场景。"""
    
    name = "generate_multiple_conformers_with_optimization"
    description = "为分子生成多个构象并进行能量优化。集成工具函数，用于处理需要多构象分析的场景。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示字符串，例如: 'C1CCCCC1'（环己烷）、'CCCCCCCC'（辛烷）"},
        "num_confs": {"type": "integer", "description": "要生成的构象数量，默认为10"},
        "method": {"type": "string", "description": "3D坐标生成方法，默认'ETKDGv3'：'ETKDG'（标准ETKDG方法）、'ETKDGv3'（改进版本，推荐）、'basic'（基本嵌入方法）", "enum": ["ETKDG", "ETKDGv3", "basic"]},
        "force_field": {"type": "string", "description": "力场类型，默认'MMFF'。'MMFF'：MMFF94力场，适用于大多数有机分子；'UFF'：通用力场，适用范围更广", "enum": ["MMFF", "UFF"]},
        "max_iters": {"type": "integer", "description": "能量优化的最大迭代次数，默认200"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 generate_multiple_conformers_with_optimization 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            num_confs = args.get("num_confs", None)
            method = args.get("method", None)
            force_field = args.get("force_field", None)
            max_iters = args.get("max_iters", None)
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import generate_multiple_conformers_with_optimization
            
            # 调用函数
            result = generate_multiple_conformers_with_optimization(smiles, num_confs, method, force_field, max_iters)
            
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






@Toolbox.register(name="get_3d_properties")
class Get3dPropertiesTool(EnvironmentTool):
    """计算分子的3D几何性质和形状描述符。基于主惯性矩分析分子的三维形状特征。"""
    
    name = "get_3d_properties"
    description = "计算分子的3D几何性质和形状描述符。基于主惯性矩分析分子的三维形状特征。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示字符串，或化学名称（会自动转换），例如: 'CCO', 'c1ccccc1', 'aspirin'"},
        "method": {"type": "string", "description": "3D坐标生成方法：'ETKDG'（标准ETKDG方法）、'ETKDGv3'（改进的ETKDGv3方法，推荐）、'basic'（基本嵌入方法）", "enum": ["ETKDG", "ETKDGv3", "basic"]},
        "conf_id": {"type": "integer", "description": "构象ID，默认为0（第一个构象）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_3d_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            method = args.get("method")
            if method is None:
                return Observation(self.name, "错误: 缺少必需参数 method")
            conf_id = args.get("conf_id", 0)
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import get_3d_properties
            
            # 调用函数
            result = get_3d_properties(smiles, method, conf_id)
            
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


@Toolbox.register(name="optimize_geometry")
class OptimizeGeometryTool(EnvironmentTool):
    """分子几何优化工具（能量最小化）。基于分子力场进行能量最小化，优化分子的三维几何构型。"""
    
    name = "optimize_geometry"
    description = "分子几何优化工具（能量最小化）。基于分子力场进行能量最小化，优化分子的三维几何构型。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示字符串，或化学名称（会自动转换），例如: 'CCO', 'c1ccccc1', 'aspirin'"},
        "method": {"type": "string", "description": "3D坐标生成方法：'ETKDG'（标准ETKDG方法）、'ETKDGv3'（改进的ETKDGv3方法，推荐）、'basic'（基本嵌入方法）", "enum": ["ETKDG", "ETKDGv3", "basic"]},
        "force_field": {"type": "string", "description": "力场类型，默认为'MMFF'。'MMFF'：MMFF94力场，适用于大多数有机分子，精度较高；'UFF'：通用力场，适用范围更广但精度稍低", "enum": ["MMFF", "UFF"]},
        "max_iters": {"type": "integer", "description": "能量最小化的最大迭代次数，默认200次"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 optimize_geometry 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            method = args.get("method")
            if method is None:
                return Observation(self.name, "错误: 缺少必需参数 method")
            force_field = args.get("force_field", "MMFF")
            max_iters = args.get("max_iters", None)
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import optimize_geometry
            
            # 调用函数
            result = optimize_geometry(smiles, method, force_field, max_iters)
            
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


@Toolbox.register(name="mol_basic_physicochemical_info")
class MolBasicPhysicochemicalInfoTool(EnvironmentTool):
    """计算分子的基本理化性质和类药性评估。提供分子的基本信息、物化性质和Lipinski五规则评估。"""
    
    name = "mol_basic_physicochemical_info"
    description = "计算分子的基本理化性质和类药性评估。提供分子的基本信息、物化性质和Lipinski五规则评估。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示字符串"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 mol_basic_physicochemical_info 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            if smiles is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import mol_basic_physicochemical_info
            
            # 调用函数
            result = mol_basic_physicochemical_info(smiles)
            
            # 处理返回值：转换为 JSON 可序列化格式
            # 注意：mol_basic_physicochemical_info 返回 (MoleculeAnalyzer对象, info字符串)
            # 我们只返回可序列化的 info 字符串，忽略 MoleculeAnalyzer 对象
            if isinstance(result, tuple) and len(result) == 2:
                # 通常第二个元素是 info 字符串
                mol_obj, info_str = result
                # 只返回 info 字符串
                result_dict = {"result": info_str, "info": info_str}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 对于其他 tuple，尝试转换每个元素
                serializable_items = [convert_to_json_serializable(item) for item in result]
                result_dict = {"result": serializable_items}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            elif isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            else:
                # 转换其他类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


# ==================== 工具注册函数 ====================

def register_analytical_chemistry_tools(environment):
    """
    将所有 analytical_chemistry 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass


# ==================== 新增工具类（由 register_tools_for_directory.py 自动添加） ====================


@Toolbox.register(name="visualize")
class VisualizeTool(EnvironmentTool):
    """综合可视化分子结构，支持2D结构、3D结构的2D投影和Matplotlib 3D图三种方式。"""
    
    name = "visualize"
    description = "综合可视化分子结构，支持2D结构、3D结构的2D投影和Matplotlib 3D图三种方式。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES字符串表示"},
        "output_prefix": {"type": "string", "description": "输出文件的前缀路径"},
        "methods": {"type": "array", "description": "可视化方法列表，可选值：'2d'（标准2D结构）、'3d_projection'（3D结构的2D投影）、'3d_matplotlib'（Matplotlib 3D图），默认为全部三种方法"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "smiles" not in args or args["smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            if "output_prefix" not in args or args["output_prefix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 output_prefix")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_visualize import visualize
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["smiles", "output_prefix", "methods"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize(**func_kwargs)
            
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
    """主函数：演示如何使用工具函数解决分析化学中的实际问题"""
    
    name = "main"
    description = "主函数：演示如何使用工具函数解决分析化学中的实际问题"
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
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import main
            
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


@Toolbox.register(name="add_compound")
class AddCompoundTool(EnvironmentTool):
    """动态添加新化合物"""
    
    name = "add_compound"
    description = "动态添加新化合物"
    arguments = {
        "name": {"type": "string", "description": "化合物名称"},
        "compound_data": {"type": "object", "description": "化合物数据字典"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 add_compound 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            if "compound_data" not in args or args["compound_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 compound_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import add_compound
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name", "compound_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = add_compound(**func_kwargs)
            
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


@Toolbox.register(name="remove_compound")
class RemoveCompoundTool(EnvironmentTool):
    """移除化合物"""
    
    name = "remove_compound"
    description = "移除化合物"
    arguments = {
        "name": {"type": "string", "description": "化合物名称"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 remove_compound 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import remove_compound
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = remove_compound(**func_kwargs)
            
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


@Toolbox.register(name="update_compound")
class UpdateCompoundTool(EnvironmentTool):
    """更新化合物信息"""
    
    name = "update_compound"
    description = "更新化合物信息"
    arguments = {
        "name": {"type": "string", "description": "化合物名称"},
        "compound_data": {"type": "object", "description": "新的化合物数据"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 update_compound 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            if "compound_data" not in args or args["compound_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 compound_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import update_compound
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name", "compound_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = update_compound(**func_kwargs)
            
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


@Toolbox.register(name="search_compounds")
class SearchCompoundsTool(EnvironmentTool):
    """根据条件搜索化合物"""
    
    name = "search_compounds"
    description = "根据条件搜索化合物"
    arguments = {
        "criteria": {"type": "object", "description": "搜索条件字典，可包含 color_solid（固体颜色）、gas（气体）、solution_color（溶液颜色）、ion（离子）等字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 search_compounds 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "criteria" not in args or args["criteria"] is None:
                return Observation(self.name, "错误: 缺少必需参数 criteria")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import search_compounds
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["criteria"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = search_compounds(**func_kwargs)
            
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


@Toolbox.register(name="get_compound_info")
class GetCompoundInfoTool(EnvironmentTool):
    """获取化合物详细信息。"""
    
    name = "get_compound_info"
    description = "获取化合物详细信息。"
    arguments = {
        "name": {"type": "string", "description": "化合物名称"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_compound_info 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import get_compound_info
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_compound_info(**func_kwargs)
            
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


@Toolbox.register(name="list_all_compounds")
class ListAllCompoundsTool(EnvironmentTool):
    """列出所有化合物名称。"""
    
    name = "list_all_compounds"
    description = "列出所有化合物名称。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 list_all_compounds 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import list_all_compounds
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = list_all_compounds(**func_kwargs)
            
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


@Toolbox.register(name="get_statistics")
class GetStatisticsTool(EnvironmentTool):
    """获取数据库统计信息"""
    
    name = "get_statistics"
    description = "获取数据库统计信息"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_statistics 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import get_statistics
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_statistics(**func_kwargs)
            
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


@Toolbox.register(name="save_compounds_to_file")
class SaveCompoundsToFileTool(EnvironmentTool):
    """保存化合物数据库到文件"""
    
    name = "save_compounds_to_file"
    description = "保存化合物数据库到文件"
    arguments = {
        "file_path": {"type": "string", "description": "文件路径（支持JSON和CSV格式）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 save_compounds_to_file 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "file_path" not in args or args["file_path"] is None:
                return Observation(self.name, "错误: 缺少必需参数 file_path")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import save_compounds_to_file
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["file_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = save_compounds_to_file(**func_kwargs)
            
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


@Toolbox.register(name="load_compounds_from_file")
class LoadCompoundsFromFileTool(EnvironmentTool):
    """从文件加载化合物数据库"""
    
    name = "load_compounds_from_file"
    description = "从文件加载化合物数据库"
    arguments = {
        "file_path": {"type": "string", "description": "文件路径（支持JSON和CSV格式）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 load_compounds_from_file 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "file_path" not in args or args["file_path"] is None:
                return Observation(self.name, "错误: 缺少必需参数 file_path")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import load_compounds_from_file
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["file_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = load_compounds_from_file(**func_kwargs)
            
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


@Toolbox.register(name="solve_chemical_puzzle")
class SolveChemicalPuzzleTool(EnvironmentTool):
    """根据观察到的化学现象预测化合物，返回预测结果、化学式、置信度和解释。"""
    
    name = "solve_chemical_puzzle"
    description = "根据观察到的化学现象预测化合物，返回预测结果、化学式、置信度和解释。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_chemical_puzzle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import solve_chemical_puzzle
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve_chemical_puzzle(**func_kwargs)
            
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


@Toolbox.register(name="visualize_analysis")
class VisualizeAnalysisTool(EnvironmentTool):
    """生成化学反应性分析的可视化图表，包括匹配得分柱状图、颜色变化路径、候选化合物信息对比和反应性特征热力图。"""
    
    name = "visualize_analysis"
    description = "生成化学反应性分析的可视化图表，包括匹配得分柱状图、颜色变化路径、候选化合物信息对比和反应性特征热力图。"
    arguments = {
        "result": {"type": "object", "description": "分析结果字典，包含预测化合物、化学式、置信度、解释、所有候选得分等字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "result" not in args or args["result"] is None:
                return Observation(self.name, "错误: 缺少必需参数 result")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import visualize_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["result"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_analysis(**func_kwargs)
            
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


@Toolbox.register(name="match_phenomenon")
class MatchPhenomenonTool(EnvironmentTool):
    """根据观察到的化学现象匹配化合物，计算每个化合物与观察现象的匹配度评分。"""
    
    name = "match_phenomenon"
    description = "根据观察到的化学现象匹配化合物，计算每个化合物与观察现象的匹配度评分。"
    arguments = {
        "observed_phenomena": {"type": "object", "description": "观察到的化学现象字典，包含固体颜色、气体产物、溶液颜色（浓/稀）、KSCN+丙酮反应等字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 match_phenomenon 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "observed_phenomena" not in args or args["observed_phenomena"] is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_phenomena")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import match_phenomenon
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["observed_phenomena"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = match_phenomenon(**func_kwargs)
            
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


@Toolbox.register(name="predict_compound")
class PredictCompoundTool(EnvironmentTool):
    """根据观察到的现象预测化合物，返回最佳匹配的化合物及其得分。"""
    
    name = "predict_compound"
    description = "根据观察到的现象预测化合物，返回最佳匹配的化合物及其得分。"
    arguments = {
        "observed_phenomena": {"type": "object", "description": "观察到的现象数据，用于匹配和预测化合物"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 predict_compound 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "observed_phenomena" not in args or args["observed_phenomena"] is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_phenomena")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import predict_compound
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["observed_phenomena"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = predict_compound(**func_kwargs)
            
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


@Toolbox.register(name="chemical_name_to_molecular_formula")
class ChemicalNameToMolecularFormulaTool(EnvironmentTool):
    """化学名称转SMILES字符串工具。基于CIR（Chemical Identifier Resolver）在线服务，将化学物质的常用名称、IUPAC名称或其他标识符转换为SMILES分子表示形式。"""
    
    name = "chemical_name_to_molecular_formula"
    description = "化学名称转SMILES字符串工具。基于CIR（Chemical Identifier Resolver）在线服务，将化学物质的常用名称、IUPAC名称或其他标识符转换为SMILES分子表示形式。"
    arguments = {
        "chemical_name": {"type": "string", "description": "化学物质的名称，支持常用名（如'aspirin'、'ethanol'、'caffeine'）、IUPAC名称（如'2-acetoxybenzoic acid'）、CAS号等其他标识符"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 chemical_name_to_molecular_formula 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "chemical_name" not in args or args["chemical_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 chemical_name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.py_smiles import chemical_name_to_molecular_formula
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["chemical_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = chemical_name_to_molecular_formula(**func_kwargs)
            
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


@Toolbox.register(name="smiles_to_formula")
class SmilesToFormulaTool(EnvironmentTool):
    """从SMILES获取化学式"""
    
    name = "smiles_to_formula"
    description = "从SMILES获取化学式"
    arguments = {
        "smiles": {"type": "string", "description": "SMILES字符串"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 smiles_to_formula 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "smiles" not in args or args["smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.py_smiles import smiles_to_formula
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["smiles"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = smiles_to_formula(**func_kwargs)
            
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


@Toolbox.register(name="parse_molecular_formula")
class ParseMolecularFormulaTool(EnvironmentTool):
    """解析分子式，提取各元素的个数。将化学分子式（如C9H11FN2O5）解析为各元素及其数量的字典。"""
    
    name = "parse_molecular_formula"
    description = "解析分子式，提取各元素的个数。将化学分子式（如C9H11FN2O5）解析为各元素及其数量的字典。"
    arguments = {
        "formula": {"type": "string", "description": "分子式字符串，例如: 'C9H11FN2O5', 'C6H12O6', 'H2O', 'CH4'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_molecular_formula 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "formula" not in args or args["formula"] is None:
                return Observation(self.name, "错误: 缺少必需参数 formula")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.py_smiles import parse_molecular_formula
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["formula"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_molecular_formula(**func_kwargs)
            
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


@Toolbox.register(name="smiles_to_element_composition")
class SmilesToElementCompositionTool(EnvironmentTool):
    """从SMILES获取元素组成信息。将SMILES字符串转换为分子式，然后解析出各元素的个数。"""
    
    name = "smiles_to_element_composition"
    description = "从SMILES获取元素组成信息。将SMILES字符串转换为分子式，然后解析出各元素的个数。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES字符串，例如: 'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 smiles_to_element_composition 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "smiles" not in args or args["smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.py_smiles import smiles_to_element_composition
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["smiles"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = smiles_to_element_composition(**func_kwargs)
            
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


@Toolbox.register(name="Reactivi_get_compound_info")
class ReactiviGetCompoundInfoTool(EnvironmentTool):
    """根据化合物名称获取信息。先查本地数据库，若不存在则调用本地名称解析工具获取SMILES和分子式等信息。"""
    
    name = "Reactivi_get_compound_info"
    description = "根据化合物名称获取信息。先查本地数据库，若不存在则调用本地名称解析工具获取SMILES和分子式等信息。"
    arguments = {
        "name": {"type": "string", "description": "化合物名称"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 Reactivi_get_compound_info 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import Reactivi_get_compound_info
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = Reactivi_get_compound_info(**func_kwargs)
            
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


@Toolbox.register(name="Reactivi_match_phenomenon")
class ReactiviMatchPhenomenonTool(EnvironmentTool):
    """协议封装：匹配观测到的现象与候选化合物的行为。"""
    
    name = "Reactivi_match_phenomenon"
    description = "协议封装：匹配观测到的现象与候选化合物的行为。"
    arguments = {
        "observed_phenomena": {"type": "object", "description": "观测到的现象，可为JSON字符串或字典对象"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 Reactivi_match_phenomenon 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "observed_phenomena" not in args or args["observed_phenomena"] is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_phenomena")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import Reactivi_match_phenomenon
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["observed_phenomena"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = Reactivi_match_phenomenon(**func_kwargs)
            
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


@Toolbox.register(name="Reactivi_predict_compound")
class ReactiviPredictCompoundTool(EnvironmentTool):
    """根据现象预测最可能的化合物，返回预测的化合物、置信度和所有评分。"""
    
    name = "Reactivi_predict_compound"
    description = "根据现象预测最可能的化合物，返回预测的化合物、置信度和所有评分。"
    arguments = {
        "observed_phenomena": {"type": "object", "description": "观察到的现象字典对象，包含各种化学反应现象的描述"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 Reactivi_predict_compound 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "observed_phenomena" not in args or args["observed_phenomena"] is None:
                return Observation(self.name, "错误: 缺少必需参数 observed_phenomena")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import Reactivi_predict_compound
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["observed_phenomena"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = Reactivi_predict_compound(**func_kwargs)
            
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


@Toolbox.register(name="Reactivi_solve_chemical_puzzle")
class ReactiviSolveChemicalPuzzleTool(EnvironmentTool):
    """协议封装：调用原有的 solve_chemical_puzzle 工作流。"""
    
    name = "Reactivi_solve_chemical_puzzle"
    description = "协议封装：调用原有的 solve_chemical_puzzle 工作流。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 Reactivi_solve_chemical_puzzle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import Reactivi_solve_chemical_puzzle
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = Reactivi_solve_chemical_puzzle(**func_kwargs)
            
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


@Toolbox.register(name="Reactivi_visualize_analysis")
class ReactiviVisualizeAnalysisTool(EnvironmentTool):
    """协议封装：可视化分析结果。返回一个占位结果（图像显示由 matplotlib 负责）。"""
    
    name = "Reactivi_visualize_analysis"
    description = "协议封装：可视化分析结果。返回一个占位结果（图像显示由 matplotlib 负责）。"
    arguments = {
        "result": {"type": "object", "description": "分析结果对象，包含需要可视化的数据"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 Reactivi_visualize_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "result" not in args or args["result"] is None:
                return Observation(self.name, "错误: 缺少必需参数 result")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.ReactivityAnalyzer import Reactivi_visualize_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["result"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = Reactivi_visualize_analysis(**func_kwargs)
            
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


@Toolbox.register(name="save_to_file")
class SaveToFileTool(EnvironmentTool):
    """保存3D分子到文件"""
    
    name = "save_to_file"
    description = "保存3D分子到文件"
    arguments = {
        "filename": {"type": "string", "description": "文件名"},
        "format": {"type": "string", "description": "文件格式，可选值：'mol', 'sdf', 'pdb', 'xyz'，默认为'mol'"},
        "conf_id": {"type": "integer", "description": "构象ID，默认为0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 save_to_file 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "filename" not in args or args["filename"] is None:
                return Observation(self.name, "错误: 缺少必需参数 filename")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import save_to_file
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["filename", "format", "conf_id"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = save_to_file(**func_kwargs)
            
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


@Toolbox.register(name="calculate_synthetic_accessibility")
class CalculateSyntheticAccessibilityTool(EnvironmentTool):
    """计算分子的合成可及性分数（Synthetic Accessibility Score, SAS）。基于Ertl-Schuffenhauer算法，评估有机分子的合成难度。"""
    
    name = "calculate_synthetic_accessibility"
    description = "计算分子的合成可及性分数（Synthetic Accessibility Score, SAS）。基于Ertl-Schuffenhauer算法，评估有机分子的合成难度。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示字符串，或化学名称（会自动转换）。例如: 'CCO'（乙醇）, 'CC(=O)Oc1ccccc1C(=O)O'（阿司匹林）, 'aspirin'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_synthetic_accessibility 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "smiles" not in args or args["smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import calculate_synthetic_accessibility
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["smiles"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_synthetic_accessibility(**func_kwargs)
            
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


@Toolbox.register(name="parse_smiles")
class ParseSmilesTool(EnvironmentTool):
    """解析SMILES字符串并返回分子对象。"""
    
    name = "parse_smiles"
    description = "解析SMILES字符串并返回分子对象。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_smiles 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import parse_smiles
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_smiles(**func_kwargs)
            
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


@Toolbox.register(name="generate_3d")
class Generate3dTool(EnvironmentTool):
    """生成3D坐标"""
    
    name = "generate_3d"
    description = "生成3D坐标"
    arguments = {
        "method": {"type": "string", "description": "生成方法，可选值：'ETKDG'（推荐）、'ETKDGv3'、'basic'"},
        "num_confs": {"type": "integer", "description": "生成构象数量，默认为1"},
        "random_seed": {"type": "integer", "description": "随机种子，默认为42"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 generate_3d 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import generate_3d
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["method", "num_confs", "random_seed"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = generate_3d(**func_kwargs)
            
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


@Toolbox.register(name="get_basic_info")
class GetBasicInfoTool(EnvironmentTool):
    """获取基本信息"""
    
    name = "get_basic_info"
    description = "获取基本信息"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_basic_info 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import get_basic_info
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_basic_info(**func_kwargs)
            
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


@Toolbox.register(name="get_properties")
class GetPropertiesTool(EnvironmentTool):
    """获取物化性质"""
    
    name = "get_properties"
    description = "获取物化性质"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import get_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_properties(**func_kwargs)
            
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


@Toolbox.register(name="get_lipinski_rule")
class GetLipinskiRuleTool(EnvironmentTool):
    """计算分子的Lipinski五规则，评估其类药性。"""
    
    name = "get_lipinski_rule"
    description = "计算分子的Lipinski五规则，评估其类药性。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_lipinski_rule 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import get_lipinski_rule
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_lipinski_rule(**func_kwargs)
            
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


@Toolbox.register(name="example_basic")
class ExampleBasicTool(EnvironmentTool):
    """示例1：基本3D生成和优化"""
    
    name = "example_basic"
    description = "示例1：基本3D生成和优化"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 example_basic 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import example_basic
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = example_basic(**func_kwargs)
            
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


@Toolbox.register(name="example_multiple_conformers")
class ExampleMultipleConformersTool(EnvironmentTool):
    """示例2：多构象生成和能量比较"""
    
    name = "example_multiple_conformers"
    description = "示例2：多构象生成和能量比较"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 example_multiple_conformers 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import example_multiple_conformers
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = example_multiple_conformers(**func_kwargs)
            
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


@Toolbox.register(name="example_comparison")
class ExampleComparisonTool(EnvironmentTool):
    """示例3：比较不同3D生成方法"""
    
    name = "example_comparison"
    description = "示例3：比较不同3D生成方法"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 example_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.molecule_analyzer import example_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = example_comparison(**func_kwargs)
            
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


@Toolbox.register(name="baseline_correction")
class BaselineCorrectionTool(EnvironmentTool):
    """对光谱数据进行基线校正。"""
    
    name = "baseline_correction"
    description = "对光谱数据进行基线校正。"
    arguments = {
        "spectrum": {"type": "array", "description": "光谱数据数组"},
        "method": {"type": "string", "description": "基线校正方法，可选值：'als'（非对称最小二乘法）等"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 baseline_correction 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spectrum" not in args or args["spectrum"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spectrum")
            if "method" not in args or args["method"] is None:
                return Observation(self.name, "错误: 缺少必需参数 method")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import baseline_correction
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spectrum", "method"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = baseline_correction(**func_kwargs)
            
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


@Toolbox.register(name="peak_detection")
class PeakDetectionTool(EnvironmentTool):
    """检测光谱数据中的峰。"""
    
    name = "peak_detection"
    description = "检测光谱数据中的峰。"
    arguments = {
        "spectrum": {"type": "array", "description": "光谱数据数组"},
        "x_values": {"type": "array", "description": "光谱对应的x轴坐标值（波长或波数）"},
        "threshold": {"type": "number", "description": "峰检测阈值"},
        "min_distance": {"type": "integer", "description": "相邻峰之间的最小距离（数据点数）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 peak_detection 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spectrum" not in args or args["spectrum"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spectrum")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import peak_detection
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spectrum", "x_values", "threshold", "min_distance"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = peak_detection(**func_kwargs)
            
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


@Toolbox.register(name="multivariate_analysis")
class MultivariateAnalysisTool(EnvironmentTool):
    """执行多变量分析（如PCA主成分分析）。"""
    
    name = "multivariate_analysis"
    description = "执行多变量分析（如PCA主成分分析）。"
    arguments = {
        "data": {"type": "object", "description": "输入数据，DataFrame格式，包含多个特征变量"},
        "method": {"type": "string", "description": "分析方法，可选值：'pca'（主成分分析）等"},
        "n_components": {"type": "integer", "description": "主成分数量"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 multivariate_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "data" not in args or args["data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 data")
            if "method" not in args or args["method"] is None:
                return Observation(self.name, "错误: 缺少必需参数 method")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import multivariate_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["data", "method", "n_components"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = multivariate_analysis(**func_kwargs)
            
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


@Toolbox.register(name="fit_calibration_curve")
class FitCalibrationCurveTool(EnvironmentTool):
    """拟合分析校准曲线。"""
    
    name = "fit_calibration_curve"
    description = "拟合分析校准曲线。"
    arguments = {
        "concentrations": {"type": "array", "description": "标准溶液浓度数组"},
        "responses": {"type": "array", "description": "对应的仪器响应值数组"},
        "model": {"type": "string", "description": "拟合模型类型，可选值：'linear'（线性）等"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fit_calibration_curve 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "concentrations" not in args or args["concentrations"] is None:
                return Observation(self.name, "错误: 缺少必需参数 concentrations")
            if "responses" not in args or args["responses"] is None:
                return Observation(self.name, "错误: 缺少必需参数 responses")
            if "model" not in args or args["model"] is None:
                return Observation(self.name, "错误: 缺少必需参数 model")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import fit_calibration_curve
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["concentrations", "responses", "model"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = fit_calibration_curve(**func_kwargs)
            
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


@Toolbox.register(name="plot_horwitz_trumpet")
class PlotHorwitzTrumpetTool(EnvironmentTool):
    """绘制Horwitz Trumpet曲线图。"""
    
    name = "plot_horwitz_trumpet"
    description = "绘制Horwitz Trumpet曲线图。"
    arguments = {
        "conc_range": {"type": "string", "description": "参数 conc_range"},
        "show_plateau": {"type": "string", "description": "参数 show_plateau"},
        "annotate": {"type": "string", "description": "参数 annotate"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_horwitz_trumpet 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import plot_horwitz_trumpet
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["conc_range", "show_plateau", "annotate"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_horwitz_trumpet(**func_kwargs)
            
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


@Toolbox.register(name="predict")
class PredictTool(EnvironmentTool):
    """使用拟合参数对输入数据进行预测。"""
    
    name = "predict"
    description = "使用拟合参数对输入数据进行预测。"
    arguments = {
        "x": {"type": "array", "description": "输入数据，用于预测的自变量值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 predict 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "x" not in args or args["x"] is None:
                return Observation(self.name, "错误: 缺少必需参数 x")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import predict
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["x"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = predict(**func_kwargs)
            
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


@Toolbox.register(name="fit_func")
class FitFuncTool(EnvironmentTool):
    """二次函数拟合模型，计算 a*x^2 + b*x + c。"""
    
    name = "fit_func"
    description = "二次函数拟合模型，计算 a*x^2 + b*x + c。"
    arguments = {
        "x": {"type": "number", "description": "自变量x的值"},
        "a": {"type": "number", "description": "二次项系数"},
        "b": {"type": "number", "description": "一次项系数"},
        "c": {"type": "number", "description": "常数项"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fit_func 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "x" not in args or args["x"] is None:
                return Observation(self.name, "错误: 缺少必需参数 x")
            if "a" not in args or args["a"] is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            if "b" not in args or args["b"] is None:
                return Observation(self.name, "错误: 缺少必需参数 b")
            if "c" not in args or args["c"] is None:
                return Observation(self.name, "错误: 缺少必需参数 c")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_15080 import fit_func
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["x", "a", "b", "c"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = fit_func(**func_kwargs)
            
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


@Toolbox.register(name="calculate_precipitation_pH")
class CalculatePrecipitationPhTool(EnvironmentTool):
    """计算金属离子开始沉淀和完全沉淀的pH值。基于溶度积平衡原理，计算指定金属离子在不同pH条件下的沉淀行为。"""
    
    name = "calculate_precipitation_pH"
    description = "计算金属离子开始沉淀和完全沉淀的pH值。基于溶度积平衡原理，计算指定金属离子在不同pH条件下的沉淀行为。"
    arguments = {
        "concentration": {"type": "number", "description": "金属离子浓度，单位为mol/L"},
        "ksp": {"type": "number", "description": "金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)"},
        "hydroxide_stoich": {"type": "integer", "description": "氢氧根离子的化学计量数（如Mg(OH)₂为2，Fe(OH)₃为3）"},
        "complete_threshold": {"type": "number", "description": "完全沉淀的浓度阈值，默认为1e-5 mol/L"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_precipitation_pH 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "concentration" not in args or args["concentration"] is None:
                return Observation(self.name, "错误: 缺少必需参数 concentration")
            if "ksp" not in args or args["ksp"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ksp")
            if "hydroxide_stoich" not in args or args["hydroxide_stoich"] is None:
                return Observation(self.name, "错误: 缺少必需参数 hydroxide_stoich")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.calculate_precipitation_pH import calculate_precipitation_pH
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["concentration", "ksp", "hydroxide_stoich", "complete_threshold"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_precipitation_pH(**func_kwargs)
            
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


@Toolbox.register(name="can_separate_ions")
class CanSeparateIonsTool(EnvironmentTool):
    """判断两种金属离子是否可以通过pH控制实现选择性沉淀分离。"""
    
    name = "can_separate_ions"
    description = "判断两种金属离子是否可以通过pH控制实现选择性沉淀分离。"
    arguments = {
        "conc1": {"type": "number", "description": "第一种金属离子浓度，单位为mol/L"},
        "conc2": {"type": "number", "description": "第二种金属离子浓度，单位为mol/L"},
        "ksp1": {"type": "number", "description": "第一种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)"},
        "ksp2": {"type": "number", "description": "第二种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)"},
        "stoich1": {"type": "integer", "description": "第一种金属氢氧化物中氢氧根离子的化学计量数"},
        "stoich2": {"type": "integer", "description": "第二种金属氢氧化物中氢氧根离子的化学计量数"},
        "complete_threshold": {"type": "number", "description": "完全沉淀的浓度阈值，默认为1e-5 mol/L"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 can_separate_ions 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "conc1" not in args or args["conc1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 conc1")
            if "conc2" not in args or args["conc2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 conc2")
            if "ksp1" not in args or args["ksp1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ksp1")
            if "ksp2" not in args or args["ksp2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ksp2")
            if "stoich1" not in args or args["stoich1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stoich1")
            if "stoich2" not in args or args["stoich2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stoich2")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.calculate_precipitation_pH import can_separate_ions
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["conc1", "conc2", "ksp1", "ksp2", "stoich1", "stoich2", "complete_threshold"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = can_separate_ions(**func_kwargs)
            
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


@Toolbox.register(name="plot_separation_window")
class PlotSeparationWindowTool(EnvironmentTool):
    """绘制选择性沉淀分离的pH窗口可视化图表，创建离子积与pH的关系图，显示两种金属离子的沉淀行为，高亮显示选择性分离的pH区间，帮助理解分离原理。"""
    
    name = "plot_separation_window"
    description = "绘制选择性沉淀分离的pH窗口可视化图表，创建离子积与pH的关系图，显示两种金属离子的沉淀行为，高亮显示选择性分离的pH区间，帮助理解分离原理。"
    arguments = {
        "conc1": {"type": "number", "description": "第一种金属离子浓度，单位为mol/L"},
        "conc2": {"type": "number", "description": "第二种金属离子浓度，单位为mol/L"},
        "ksp1": {"type": "number", "description": "第一种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)"},
        "ksp2": {"type": "number", "description": "第二种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)"},
        "stoich1": {"type": "integer", "description": "第一种金属氢氧化物中氢氧根离子的化学计量数"},
        "stoich2": {"type": "integer", "description": "第二种金属氢氧化物中氢氧根离子的化学计量数"},
        "label1": {"type": "string", "description": "第一种离子的标签，默认为Fe³⁺"},
        "label2": {"type": "string", "description": "第二种离子的标签，默认为Mg²⁺"},
        "title": {"type": "string", "description": "图表标题，默认为Selective Precipitation by pH"},
        "complete_threshold": {"type": "number", "description": "完全沉淀的浓度阈值，默认为1e-5 mol/L"},
        "figsize": {"type": "array", "description": "图表尺寸，默认为(10, 6)"},
        "n_points": {"type": "integer", "description": "绘图点数，默认为500"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_separation_window 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "conc1" not in args or args["conc1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 conc1")
            if "conc2" not in args or args["conc2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 conc2")
            if "ksp1" not in args or args["ksp1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ksp1")
            if "ksp2" not in args or args["ksp2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 ksp2")
            if "stoich1" not in args or args["stoich1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stoich1")
            if "stoich2" not in args or args["stoich2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 stoich2")
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.calculate_precipitation_pH import plot_separation_window
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["conc1", "conc2", "ksp1", "ksp2", "stoich1", "stoich2", "label1", "label2", "title", "complete_threshold", "figsize", "n_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_separation_window(**func_kwargs)
            
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


@Toolbox.register(name="ChemicalEquilibriumCalculator_separate_metals")
class ChemicalequilibriumcalculatorSeparateMetalsTool(EnvironmentTool):
    """化学平衡计算器：金属离子选择性沉淀分离分析。分析Fe³⁺和Mg²⁺两种金属离子在氢氧化物沉淀中的选择性分离行为。"""
    
    name = "ChemicalEquilibriumCalculator_separate_metals"
    description = "化学平衡计算器：金属离子选择性沉淀分离分析。分析Fe³⁺和Mg²⁺两种金属离子在氢氧化物沉淀中的选择性分离行为。"
    arguments = {
        "conc_fe": {"type": "number", "description": "Fe³⁺离子浓度，单位为mol/L，默认为0.01"},
        "conc_mg": {"type": "number", "description": "Mg²⁺离子浓度，单位为mol/L，默认为0.01"},
        "ksp_fe": {"type": "number", "description": "Fe(OH)₃的溶度积常数，单位为(mol/L)⁴，默认为2.79e-39"},
        "ksp_mg": {"type": "number", "description": "Mg(OH)₂的溶度积常数，单位为(mol/L)³，默认为5.61e-12"},
        "stoich_fe": {"type": "integer", "description": "Fe(OH)₃中氢氧根离子的化学计量数，默认为3"},
        "stoich_mg": {"type": "integer", "description": "Mg(OH)₂中氢氧根离子的化学计量数，默认为2"},
        "visualize": {"type": "boolean", "description": "是否生成可视化图表，默认为True"},
        "complete_threshold": {"type": "number", "description": "完全沉淀的浓度阈值，默认为1e-5 mol/L"},
        "verbose": {"type": "boolean", "description": "是否显示详细输出，默认为True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 ChemicalEquilibriumCalculator_separate_metals 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.analytical_chemistry.calculate_precipitation_pH import ChemicalEquilibriumCalculator_separate_metals
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["conc_fe", "conc_mg", "ksp_fe", "ksp_mg", "stoich_fe", "stoich_mg", "visualize", "complete_threshold", "verbose"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = ChemicalEquilibriumCalculator_separate_metals(**func_kwargs)
            
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

