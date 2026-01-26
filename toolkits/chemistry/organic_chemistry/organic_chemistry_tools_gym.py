#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
organic_chemistry 工具注册模块
使用 gym.tool.EnvironmentTool 为 organic_chemistry 目录中的工具提供统一的注册与调用接口

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


@Toolbox.register(name="generate_product_via_smarts")
class GenerateProductViaSmartsTool(EnvironmentTool):
    """根据亲核试剂和底物的SMILES字符串，通过SMARTS模式匹配生成有机反应产物"""
    
    name = "generate_product_via_smarts"
    description = "根据亲核试剂和底物的SMILES字符串，通过SMARTS模式匹配生成有机反应产物"
    arguments = {
        "nuc_smiles": {"type": "string", "description": "亲核试剂的SMILES字符串表示"},
        "sub_smiles": {"type": "string", "description": "底物的SMILES字符串表示"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 generate_product_via_smarts 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "nuc_smiles" not in args or args["nuc_smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 nuc_smiles")
            if "sub_smiles" not in args or args["sub_smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 sub_smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import generate_product_via_smarts
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["nuc_smiles", "sub_smiles"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = generate_product_via_smarts(**func_kwargs)
            
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


@Toolbox.register(name="simulate_nucleophilic_substitution")
class SimulateNucleophilicSubstitutionTool(EnvironmentTool):
    """模拟有机化学中的亲核取代反应，预测产物和反应能量。"""
    
    name = "simulate_nucleophilic_substitution"
    description = "模拟有机化学中的亲核取代反应，预测产物和反应能量。"
    arguments = {
        "nucleophile": {"type": "string", "description": "亲核试剂的SMILES表示，例如 "CH3O-" 表示甲氧基负离子"},
        "substrate": {"type": "string", "description": "底物的SMILES表示，例如 "CH3Br" 表示溴甲烷"},
        "solvent": {"type": "string", "description": "溶剂类型，可选值为 "polar", "nonpolar", "neutral"，默认为 "neutral""},
        "temperature": {"type": "number", "description": "反应温度，单位为K，默认为298.15K (25°C)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 simulate_nucleophilic_substitution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "nucleophile" not in args or args["nucleophile"] is None:
                return Observation(self.name, "错误: 缺少必需参数 nucleophile")
            if "substrate" not in args or args["substrate"] is None:
                return Observation(self.name, "错误: 缺少必需参数 substrate")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import simulate_nucleophilic_substitution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["nucleophile", "substrate", "solvent", "temperature"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = simulate_nucleophilic_substitution(**func_kwargs)
            
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


@Toolbox.register(name="predict_hydrolysis_product")
class PredictHydrolysisProductTool(EnvironmentTool):
    """预测有机化合物在水解条件下的产物。基于底物结构和反应条件，模拟水解反应并预测主要产物。"""
    
    name = "predict_hydrolysis_product"
    description = "预测有机化合物在水解条件下的产物。基于底物结构和反应条件，模拟水解反应并预测主要产物。"
    arguments = {
        "substrate": {"type": "string", "description": "底物的SMILES表示，例如 "CH3C(OCH3)2+" 表示缩醛正离子"},
        "pH": {"type": "number", "description": "反应溶液的pH值，默认为7.0（中性）"},
        "temperature": {"type": "number", "description": "反应温度，单位为K，默认为298.15K (25°C)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 predict_hydrolysis_product 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "substrate" not in args or args["substrate"] is None:
                return Observation(self.name, "错误: 缺少必需参数 substrate")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import predict_hydrolysis_product
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["substrate", "pH", "temperature"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = predict_hydrolysis_product(**func_kwargs)
            
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


@Toolbox.register(name="calculate_reaction_energy")
class CalculateReactionEnergyTool(EnvironmentTool):
    """计算有机反应的能量变化。使用量子化学方法计算反应物和产物之间的能量差异，评估反应的热力学可行性。"""
    
    name = "calculate_reaction_energy"
    description = "计算有机反应的能量变化。使用量子化学方法计算反应物和产物之间的能量差异，评估反应的热力学可行性。"
    arguments = {
        "reactants": {"type": "array", "description": "反应物的SMILES表示列表"},
        "products": {"type": "array", "description": "产物的SMILES表示列表"},
        "method": {"type": "string", "description": "计算方法，可选值为 "semi-empirical", "DFT", "force-field"，默认为 "semi-empirical""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_reaction_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "reactants" not in args or args["reactants"] is None:
                return Observation(self.name, "错误: 缺少必需参数 reactants")
            if "products" not in args or args["products"] is None:
                return Observation(self.name, "错误: 缺少必需参数 products")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import calculate_reaction_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["reactants", "products", "method"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_reaction_energy(**func_kwargs)
            
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


@Toolbox.register(name="visualize_reaction_pathway")
class VisualizeReactionPathwayTool(EnvironmentTool):
    """可视化有机反应的能量路径。创建反应坐标与能量关系图，展示反应的能量变化过程。"""
    
    name = "visualize_reaction_pathway"
    description = "可视化有机反应的能量路径。创建反应坐标与能量关系图，展示反应的能量变化过程。"
    arguments = {
        "reaction_steps": {"type": "array", "description": "反应步骤的描述列表"},
        "energies": {"type": "array", "description": "对应每个反应步骤的能量值 (kJ/mol)"},
        "title": {"type": "string", "description": "图表标题，默认为 "反应能量图""},
        "out_path": {"type": "string", "description": "输出图像的保存路径，默认为 "./images/reaction_pathway.png""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_reaction_pathway 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "reaction_steps" not in args or args["reaction_steps"] is None:
                return Observation(self.name, "错误: 缺少必需参数 reaction_steps")
            if "energies" not in args or args["energies"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energies")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import visualize_reaction_pathway
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["reaction_steps", "energies", "title", "out_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_reaction_pathway(**func_kwargs)
            
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


@Toolbox.register(name="optimize_molecular_geometry")
class OptimizeMolecularGeometryTool(EnvironmentTool):
    """优化分子几何构型，寻找能量最低构象。使用分子力场或量子化学方法优化分子的三维结构。"""
    
    name = "optimize_molecular_geometry"
    description = "优化分子几何构型，寻找能量最低构象。使用分子力场或量子化学方法优化分子的三维结构。"
    arguments = {
        "smiles": {"type": "string", "description": "分子的SMILES表示"},
        "force_field": {"type": "string", "description": "使用的力场类型，可选值为 "MMFF94", "UFF"，默认为 "MMFF94""},
        "max_iterations": {"type": "integer", "description": "优化的最大迭代次数，默认为1000"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 optimize_molecular_geometry 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "smiles" not in args or args["smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import optimize_molecular_geometry
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["smiles", "force_field", "max_iterations"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = optimize_molecular_geometry(**func_kwargs)
            
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


@Toolbox.register(name="predict_organic_reaction_products")
class PredictOrganicReactionProductsTool(EnvironmentTool):
    """预测有机反应的产物。基于反应物和反应类型，预测可能的产物及其分布。"""
    
    name = "predict_organic_reaction_products"
    description = "预测有机反应的产物。基于反应物和反应类型，预测可能的产物及其分布。"
    arguments = {
        "reactants": {"type": "array", "description": "反应物的SMILES表示列表"},
        "reaction_type": {"type": "string", "description": "反应类型，如 "SN2", "E2", "aldol", "Diels-Alder" 等"},
        "conditions": {"type": "object", "description": "反应条件，包含温度、溶剂、催化剂等信息"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 predict_organic_reaction_products 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "reactants" not in args or args["reactants"] is None:
                return Observation(self.name, "错误: 缺少必需参数 reactants")
            if "reaction_type" not in args or args["reaction_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 reaction_type")
            
            # 导入并调用原始函数
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import predict_organic_reaction_products
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["reactants", "reaction_type", "conditions"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = predict_organic_reaction_products(**func_kwargs)
            
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
    """主函数：演示如何使用工具函数求解有机化学反应问题"""
    
    name = "main"
    description = "主函数：演示如何使用工具函数求解有机化学反应问题"
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
            from toolkits.chemistry.organic_chemistry.organic_reaction_simulator_15923 import main
            
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

