#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
condensed_matter_physics 工具注册模块
使用 gym.tool.EnvironmentTool 为 condensed_matter_physics 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool, convert_to_json_serializable
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.physics.condensed_matter_physics.composite_material_solver_17911 import *  # 动态导入
# from toolkits.physics.condensed_matter_physics.condensed_matter_tools_17363 import *  # 动态导入
# from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="calculate_diamond_bond_angles")
class CalculateDiamondBondAnglesTool(EnvironmentTool):
    """计算金刚石晶格中键之间的夹角"""
    
    name = "calculate_diamond_bond_angles"
    description = "计算金刚石晶格中键之间的夹角"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_diamond_bond_angles 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            

            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_17363 import calculate_diamond_bond_angles
            
            # 调用函数
            result = calculate_diamond_bond_angles()
            
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


@Toolbox.register(name="calculate_load_distribution")
class CalculateLoadDistributionTool(EnvironmentTool):
    """计算复合材料系统中的载荷分布和最大安全载荷。该函数基于材料的弹性模量、横截面积、长度和最大允许应力，计算复合材料系统中各组件的载荷分布，并确定整个系统的最大安全载荷。"""
    
    name = "calculate_load_distribution"
    description = "计算复合材料系统中的载荷分布和最大安全载荷。该函数基于材料的弹性模量、横截面积、长度和最大允许应力，计算复合材料系统中各组件的载荷分布，并确定整个系统的最大安全载荷。"
    arguments = {
        "areas": {"type": "array", "description": "各组件的横截面积，单位为 mm²"},
        "elastic_moduli": {"type": "array", "description": "各组件的弹性模量，单位为 N/mm²"},
        "lengths": {"type": "array", "description": "各组件的长度，单位为 mm"},
        "max_stresses": {"type": "array", "description": "各组件的最大允许应力，单位为 N/mm²"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_load_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            areas = args.get("areas")
            if areas is None:
                return Observation(self.name, "错误: 缺少必需参数 areas")
            elastic_moduli = args.get("elastic_moduli")
            if elastic_moduli is None:
                return Observation(self.name, "错误: 缺少必需参数 elastic_moduli")
            lengths = args.get("lengths")
            if lengths is None:
                return Observation(self.name, "错误: 缺少必需参数 lengths")
            max_stresses = args.get("max_stresses")
            if max_stresses is None:
                return Observation(self.name, "错误: 缺少必需参数 max_stresses")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.composite_material_solver_17911 import calculate_load_distribution
            
            # 调用函数
            result = calculate_load_distribution(areas, elastic_moduli, lengths, max_stresses)
            
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


@Toolbox.register(name="construct_hamiltonian")
class ConstructHamiltonianTool(EnvironmentTool):
    """构建量子多体系统的哈密顿量矩阵（稀疏格式）"""
    
    name = "construct_hamiltonian"
    description = "构建量子多体系统的哈密顿量矩阵（稀疏格式）"
    arguments = {
        "system_size": {"type": "string", "description": ""},
        "interaction_matrix": {"type": "string", "description": ""},
        "potential": {"type": "object", "description": ""},
        "periodic": {"type": "boolean", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_hamiltonian 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            system_size = args.get("system_size")
            if system_size is None:
                return Observation(self.name, "错误: 缺少必需参数 system_size")
            interaction_matrix = args.get("interaction_matrix")
            if interaction_matrix is None:
                return Observation(self.name, "错误: 缺少必需参数 interaction_matrix")
            potential = args.get("potential", None)
            periodic = args.get("periodic", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import construct_hamiltonian
            
            # 调用函数
            result = construct_hamiltonian(system_size, interaction_matrix, potential, periodic)
            
            # 处理返回值：将 scipy 稀疏矩阵和 numpy 数组转换为 JSON 可序列化格式
            import numpy as np
            try:
                import scipy.sparse as sp
                HAS_SCIPY = True
            except ImportError:
                HAS_SCIPY = False
            
            def convert_types(obj):
                """递归地将 numpy/scipy 类型转换为 Python 原生类型"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif HAS_SCIPY and isinstance(obj, sp.spmatrix):
                    # 稀疏矩阵：转换为密集格式（如果不太大）
                    try:
                        if obj.shape[0] * obj.shape[1] <= 10000:
                            return obj.toarray().tolist()
                        else:
                            return {
                                "_type": "sparse_matrix",
                                "shape": list(obj.shape),
                                "format": obj.format,
                                "nnz": obj.nnz,
                                "note": "矩阵太大，未序列化完整数据"
                            }
                    except Exception:
                        return {
                            "_type": "sparse_matrix",
                            "shape": list(obj.shape),
                            "format": obj.format,
                            "error": "无法转换为密集格式"
                        }
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            result = convert_types(result)
            
            # 处理返回值格式
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


@Toolbox.register(name="solve_eigensystem")
class SolveEigensystemTool(EnvironmentTool):
    """求解稀疏哈密顿量的本征值和本征态"""
    
    name = "solve_eigensystem"
    description = "求解稀疏哈密顿量的本征值和本征态"
    arguments = {
        "hamiltonian": {"type": "object", "description": ""},
        "k": {"type": "integer", "description": ""},
        "which": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_eigensystem 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            hamiltonian = args.get("hamiltonian")
            if hamiltonian is None:
                return Observation(self.name, "错误: 缺少必需参数 hamiltonian")
            k = args.get("k", None)
            which = args.get("which", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import solve_eigensystem
            
            # 调用函数
            result = solve_eigensystem(hamiltonian, k, which)
            
            # 处理返回值：转换为 JSON 可序列化格式
            result = convert_to_json_serializable(result)
            
            # 处理返回值格式
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为字典，包含 eigenvalues 和 eigenvectors
                result_dict = {
                    "eigenvalues": result[0] if len(result) > 0 else None,
                    "eigenvectors": result[1] if len(result) > 1 else None
                }
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_density_matrix")
class CalculateDensityMatrixTool(EnvironmentTool):
    """计算量子态的密度矩阵或约化密度矩阵"""
    
    name = "calculate_density_matrix"
    description = "计算量子态的密度矩阵或约化密度矩阵"
    arguments = {
        "eigenvector": {"type": "array", "description": ""},
        "trace_subsystem": {"type": "array", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_density_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            eigenvector = args.get("eigenvector")
            if eigenvector is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenvector")
            trace_subsystem = args.get("trace_subsystem", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import calculate_density_matrix
            
            # 调用函数
            result = calculate_density_matrix(eigenvector, trace_subsystem)
            
            # 处理返回值：转换为 JSON 可序列化格式
            result = convert_to_json_serializable(result)
            
            # 处理返回值格式
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


@Toolbox.register(name="monte_carlo_ising")
class MonteCarloIsingTool(EnvironmentTool):
    """二维Ising模型的蒙特卡洛模拟"""
    
    name = "monte_carlo_ising"
    description = "二维Ising模型的蒙特卡洛模拟"
    arguments = {
        "lattice_size": {"type": "array", "description": ""},
        "temperature": {"type": "number", "description": ""},
        "num_steps": {"type": "integer", "description": ""},
        "J": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 monte_carlo_ising 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice_size = args.get("lattice_size")
            if lattice_size is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_size")
            temperature = args.get("temperature")
            if temperature is None:
                return Observation(self.name, "错误: 缺少必需参数 temperature")
            num_steps = args.get("num_steps")
            if num_steps is None:
                return Observation(self.name, "错误: 缺少必需参数 num_steps")
            J = args.get("J", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import monte_carlo_ising
            
            # 调用函数
            result = monte_carlo_ising(lattice_size, temperature, num_steps, J)
            
            # 处理返回值：转换为 JSON 可序列化格式
            result = convert_to_json_serializable(result)
            
            # 处理返回值格式
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


@Toolbox.register(name="plot_phase_diagram")
class PlotPhaseDiagramTool(EnvironmentTool):
    """绘制相变图，展示物理量随温度变化"""
    
    name = "plot_phase_diagram"
    description = "绘制相变图，展示物理量随温度变化"
    arguments = {
        "temp_range": {"type": "array", "description": ""},
        "results": {"type": "array", "description": ""},
        "observable": {"type": "string", "description": ""},
        "title": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_phase_diagram 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            temp_range = args.get("temp_range")
            if temp_range is None:
                return Observation(self.name, "错误: 缺少必需参数 temp_range")
            results = args.get("results")
            if results is None:
                return Observation(self.name, "错误: 缺少必需参数 results")
            observable = args.get("observable", None)
            title = args.get("title", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import plot_phase_diagram
            
            # 调用函数
            result = plot_phase_diagram(temp_range, results, observable, title)
            
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


@Toolbox.register(name="visualize_lattice")
class VisualizeLatticeTool(EnvironmentTool):
    """可视化二维Ising晶格构型"""
    
    name = "visualize_lattice"
    description = "可视化二维Ising晶格构型"
    arguments = {
        "lattice": {"type": "array", "description": ""},
        "title": {"type": "string", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_lattice 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice = args.get("lattice")
            if lattice is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice")
            title = args.get("title", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import visualize_lattice
            
            # 调用函数
            result = visualize_lattice(lattice, title)
            
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


@Toolbox.register(name="calculate_correlation_function")
class CalculateCorrelationFunctionTool(EnvironmentTool):
    """计算二维Ising模型的自旋关联函数"""
    
    name = "calculate_correlation_function"
    description = "计算二维Ising模型的自旋关联函数"
    arguments = {
        "lattice": {"type": "array", "description": ""},
        "max_distance": {"type": "integer", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_correlation_function 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice = args.get("lattice")
            if lattice is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice")
            max_distance = args.get("max_distance", None)
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_tools_4179 import calculate_correlation_function
            
            # 调用函数
            result = calculate_correlation_function(lattice, max_distance)
            
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

def register_condensed_matter_physics_tools(environment):
    """
    将所有 condensed_matter_physics 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass


# ==================== 新增工具类（由 register_tools_for_directory.py 自动添加） ====================


@Toolbox.register(name="main")
class MainTool(EnvironmentTool):
    """主函数：演示如何使用工具函数求解凝聚态物理问题"""
    
    name = "main"
    description = "主函数：演示如何使用工具函数求解凝聚态物理问题"
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
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_18998 import main
            
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


@Toolbox.register(name="solve")
class SolveTool(EnvironmentTool):
    """求解本征值问题"""
    
    name = "solve"
    description = "求解本征值问题"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.eigenvalue import solve
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve(**func_kwargs)
            
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


@Toolbox.register(name="create_heisenberg_hamiltonian")
class CreateHeisenbergHamiltonianTool(EnvironmentTool):
    """创建二维正方晶格上的量子海森堡模型哈密顿量。

Parameters:
-----------
L : int
    晶格的线性尺寸
J : float
    交换耦合强度
periodic : bool
    是否使用周期性边界条件
Returns:
H : scipy.sparse.csr_matrix
    哈密顿量
--------"""
    
    name = "create_heisenberg_hamiltonian"
    description = "创建二维正方晶格上的量子海森堡模型哈密顿量。

Parameters:
-----------
L : int
    晶格的线性尺寸
J : float
    交换耦合强度
periodic : bool
    是否使用周期性边界条件
Returns:
H : scipy.sparse.csr_matrix
    哈密顿量
--------"
    arguments = {
        "L": {"type": "string", "description": "参数 L"},
        "J": {"type": "string", "description": "参数 J"},
        "periodic": {"type": "string", "description": "参数 periodic"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_heisenberg_hamiltonian 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import create_heisenberg_hamiltonian
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["L", "J", "periodic"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_heisenberg_hamiltonian(**func_kwargs)
            
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


@Toolbox.register(name="calculate_susceptibility")
class CalculateSusceptibilityTool(EnvironmentTool):
    """计算不同温度下的磁化率

Parameters:
-----------
n_sites : int
    链中的格点数量
T_range : List[float]
    温度范围
J : float
    交换相互作用强度
h_small : float
    用于数值微分的小磁场
    
Returns:
--------
array_like
    不同温度下的磁化率"""
    
    name = "calculate_susceptibility"
    description = "计算不同温度下的磁化率

Parameters:
-----------
n_sites : int
    链中的格点数量
T_range : List[float]
    温度范围
J : float
    交换相互作用强度
h_small : float
    用于数值微分的小磁场
    
Returns:
--------
array_like
    不同温度下的磁化率"
    arguments = {
        "n_sites": {"type": "string", "description": "参数 n_sites"},
        "T_range": {"type": "string", "description": "参数 T_range"},
        "J": {"type": "string", "description": "参数 J"},
        "h_small": {"type": "string", "description": "参数 h_small"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_susceptibility 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.code_block_1 import calculate_susceptibility
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_sites", "T_range", "J", "h_small"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_susceptibility(**func_kwargs)
            
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


@Toolbox.register(name="calculate_magnetization")
class CalculateMagnetizationTool(EnvironmentTool):
    """计算给定量子态的磁化强度。

Parameters:
-----------
state : List[float]
    量子态向量
L : int
    晶格的线性尺寸
    
Returns:
--------
float
    平均每个格点的磁化强度
numpy.ndarray
    每个格点的磁化强度分布"""
    
    name = "calculate_magnetization"
    description = "计算给定量子态的磁化强度。

Parameters:
-----------
state : List[float]
    量子态向量
L : int
    晶格的线性尺寸
    
Returns:
--------
float
    平均每个格点的磁化强度
numpy.ndarray
    每个格点的磁化强度分布"
    arguments = {
        "state": {"type": "string", "description": "参数 state"},
        "L": {"type": "string", "description": "参数 L"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_magnetization 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import calculate_magnetization
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state", "L"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_magnetization(**func_kwargs)
            
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


@Toolbox.register(name="calculate_thermal_expectation")
class CalculateThermalExpectationTool(EnvironmentTool):
    """计算给定温度下量子算符的热力学期望值。

Parameters:
-----------
hamiltonian : qt.Qobj
    系统的哈密顿量
operator : qt.Qobj
    要计算期望值的量子算符
temperature : float
    系统温度，单位为能量单位（假设kB=1）
    
Returns:
--------
float
    算符的热力学期望值"""
    
    name = "calculate_thermal_expectation"
    description = "计算给定温度下量子算符的热力学期望值。

Parameters:
-----------
hamiltonian : qt.Qobj
    系统的哈密顿量
operator : qt.Qobj
    要计算期望值的量子算符
temperature : float
    系统温度，单位为能量单位（假设kB=1）
    
Returns:
--------
float
    算符的热力学期望值"
    arguments = {
        "hamiltonian": {"type": "string", "description": "参数 hamiltonian"},
        "operator": {"type": "string", "description": "参数 operator"},
        "temperature": {"type": "string", "description": "参数 temperature"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_thermal_expectation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.magnetic_force_analysis import calculate_thermal_expectation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["hamiltonian", "operator", "temperature"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_thermal_expectation(**func_kwargs)
            
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


@Toolbox.register(name="simulate_phase_transition")
class SimulatePhaseTransitionTool(EnvironmentTool):
    """模拟不同磁场和温度下的磁化行为，研究相变现象。

Parameters:
-----------
n_sites : int
    自旋链中的格点数量
j_coupling : float
    相邻自旋之间的交换耦合强度
h_fields : np.ndarray
    要扫描的外部磁场强度数组
temperatures : np.ndarray
    要扫描的温度数组
boundary_condition : str, optional
    边界条件
    
Returns:
--------
Dict[str, np.ndarray]
    包含模拟结果的字典，键包括：
    - 'h_fields': 磁场强度数组
    - 'temperatures': 温度数组
    - 'magnetization': 磁化强度数组（形状为[len(temperatures), len(h_fields)]）
    - 'susceptibility': 磁化率数组"""
    
    name = "simulate_phase_transition"
    description = "模拟不同磁场和温度下的磁化行为，研究相变现象。

Parameters:
-----------
n_sites : int
    自旋链中的格点数量
j_coupling : float
    相邻自旋之间的交换耦合强度
h_fields : np.ndarray
    要扫描的外部磁场强度数组
temperatures : np.ndarray
    要扫描的温度数组
boundary_condition : str, optional
    边界条件
    
Returns:
--------
Dict[str, np.ndarray]
    包含模拟结果的字典，键包括：
    - 'h_fields': 磁场强度数组
    - 'temperatures': 温度数组
    - 'magnetization': 磁化强度数组（形状为[len(temperatures), len(h_fields)]）
    - 'susceptibility': 磁化率数组"
    arguments = {
        "n_sites": {"type": "string", "description": "参数 n_sites"},
        "j_coupling": {"type": "string", "description": "参数 j_coupling"},
        "h_fields": {"type": "string", "description": "参数 h_fields"},
        "temperatures": {"type": "string", "description": "参数 temperatures"},
        "boundary_condition": {"type": "string", "description": "参数 boundary_condition"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 simulate_phase_transition 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.magnetic_force_analysis import simulate_phase_transition
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_sites", "j_coupling", "h_fields", "temperatures", "boundary_condition"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = simulate_phase_transition(**func_kwargs)
            
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


@Toolbox.register(name="find_critical_point")
class FindCriticalPointTool(EnvironmentTool):
    """通过寻找磁化率峰值来确定临界点。

Parameters:
-----------
h_fields : np.ndarray
    磁场强度数组
susceptibility : np.ndarray
    对应的磁化率数组
    
Returns:
--------
float
    估计的临界磁场强度"""
    
    name = "find_critical_point"
    description = "通过寻找磁化率峰值来确定临界点。

Parameters:
-----------
h_fields : np.ndarray
    磁场强度数组
susceptibility : np.ndarray
    对应的磁化率数组
    
Returns:
--------
float
    估计的临界磁场强度"
    arguments = {
        "h_fields": {"type": "string", "description": "参数 h_fields"},
        "susceptibility": {"type": "string", "description": "参数 susceptibility"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_critical_point 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.magnetic_force_analysis import find_critical_point
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["h_fields", "susceptibility"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = find_critical_point(**func_kwargs)
            
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


@Toolbox.register(name="plot_results")
class PlotResultsTool(EnvironmentTool):
    """绘制模拟结果。

Parameters:
-----------
results : Dict[str, np.ndarray]
    模拟结果字典
save_path : str, optional
    保存图像的路径，如果为None则显示图像"""
    
    name = "plot_results"
    description = "绘制模拟结果。

Parameters:
-----------
results : Dict[str, np.ndarray]
    模拟结果字典
save_path : str, optional
    保存图像的路径，如果为None则显示图像"
    arguments = {
        "results": {"type": "string", "description": "参数 results"},
        "save_path": {"type": "string", "description": "参数 save_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_results 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.magnetic_force_analysis import plot_results
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["results", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_results(**func_kwargs)
            
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


@Toolbox.register(name="parse_args")
class ParseArgsTool(EnvironmentTool):
    """解析命令行参数"""
    
    name = "parse_args"
    description = "解析命令行参数"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_args 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import parse_args
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_args(**func_kwargs)
            
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


@Toolbox.register(name="calculate_ground_state")
class CalculateGroundStateTool(EnvironmentTool):
    """计算哈密顿量的基态和基态能量。

Parameters:
-----------
L : int
    晶格的线性尺寸
J : float
    交换耦合强度
periodic : bool
    是否使用周期性边界条件
Returns:
--------
tuple
    (基态能量, 基态波函数)"""
    
    name = "calculate_ground_state"
    description = "计算哈密顿量的基态和基态能量。

Parameters:
-----------
L : int
    晶格的线性尺寸
J : float
    交换耦合强度
periodic : bool
    是否使用周期性边界条件
Returns:
--------
tuple
    (基态能量, 基态波函数)"
    arguments = {
        "L": {"type": "string", "description": "参数 L"},
        "J": {"type": "string", "description": "参数 J"},
        "periodic": {"type": "string", "description": "参数 periodic"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_ground_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import calculate_ground_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["L", "J", "periodic"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_ground_state(**func_kwargs)
            
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


@Toolbox.register(name="calculate_entanglement_entropy")
class CalculateEntanglementEntropyTool(EnvironmentTool):
    """计算量子态的纠缠熵

Parameters:
-----------
state : qutip.Qobj
    量子态向量
site : int
    计算纠缠熵的切分点
N : int
    系统总大小

Returns:
--------
float
    纠缠熵值"""
    
    name = "calculate_entanglement_entropy"
    description = "计算量子态的纠缠熵

Parameters:
-----------
state : qutip.Qobj
    量子态向量
site : int
    计算纠缠熵的切分点
N : int
    系统总大小

Returns:
--------
float
    纠缠熵值"
    arguments = {
        "state": {"type": "string", "description": "参数 state"},
        "site": {"type": "string", "description": "参数 site"},
        "N": {"type": "string", "description": "参数 N"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_entanglement_entropy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_spin_chain import calculate_entanglement_entropy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state", "site", "N"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_entanglement_entropy(**func_kwargs)
            
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


@Toolbox.register(name="simulate_time_evolution")
class SimulateTimeEvolutionTool(EnvironmentTool):
    """模拟量子系统的时间演化。

在量子多体系统模拟中，时间演化是研究动力学行为的基本工具。
此函数基于薛定谔方程计算量子态随时间的演化。

Parameters:
-----------
hamiltonian : np.ndarray
    系统的哈密顿量，形状为(n, n)的厄米矩阵
initial_state : np.ndarray
    初始量子态，形状为(n,)的向量
time_points : np.ndarray
    要计算的时间点数组
operators : Dict[str, np.ndarray], optional
    要计算期望值的算符字典，键为算符名称，值为算符矩阵
    
Returns:
--------
Dict[str, np.ndarray]
    包含时间演化结果的字典:
    - 'states': 每个时间点的量子态
    - 算符名称: 对应算符在每个时间点的期望值"""
    
    name = "simulate_time_evolution"
    description = "模拟量子系统的时间演化。

在量子多体系统模拟中，时间演化是研究动力学行为的基本工具。
此函数基于薛定谔方程计算量子态随时间的演化。

Parameters:
-----------
hamiltonian : np.ndarray
    系统的哈密顿量，形状为(n, n)的厄米矩阵
initial_state : np.ndarray
    初始量子态，形状为(n,)的向量
time_points : np.ndarray
    要计算的时间点数组
operators : Dict[str, np.ndarray], optional
    要计算期望值的算符字典，键为算符名称，值为算符矩阵
    
Returns:
--------
Dict[str, np.ndarray]
    包含时间演化结果的字典:
    - 'states': 每个时间点的量子态
    - 算符名称: 对应算符在每个时间点的期望值"
    arguments = {
        "hamiltonian": {"type": "string", "description": "参数 hamiltonian"},
        "initial_state": {"type": "string", "description": "参数 initial_state"},
        "time_points": {"type": "string", "description": "参数 time_points"},
        "operators": {"type": "string", "description": "参数 operators"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 simulate_time_evolution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import simulate_time_evolution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["hamiltonian", "initial_state", "time_points", "operators"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = simulate_time_evolution(**func_kwargs)
            
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


@Toolbox.register(name="plot_magnetization_dynamics")
class PlotMagnetizationDynamicsTool(EnvironmentTool):
    """绘制磁化强度随时间的演化

Parameters:
-----------
result : qutip.solver.Result
    时间演化结果
N : int
    系统大小
times : numpy.ndarray
    时间点数组

Returns:
--------
matplotlib.figure.Figure
    绘制的图形"""
    
    name = "plot_magnetization_dynamics"
    description = "绘制磁化强度随时间的演化

Parameters:
-----------
result : qutip.solver.Result
    时间演化结果
N : int
    系统大小
times : numpy.ndarray
    时间点数组

Returns:
--------
matplotlib.figure.Figure
    绘制的图形"
    arguments = {
        "result": {"type": "string", "description": "参数 result"},
        "N": {"type": "string", "description": "参数 N"},
        "times": {"type": "string", "description": "参数 times"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_magnetization_dynamics 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_spin_chain import plot_magnetization_dynamics
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["result", "N", "times"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_magnetization_dynamics(**func_kwargs)
            
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


@Toolbox.register(name="plot_entanglement_entropy")
class PlotEntanglementEntropyTool(EnvironmentTool):
    """绘制不同耦合强度下的纠缠熵

Parameters:
-----------
J_values : numpy.ndarray
    耦合强度值数组
N : int
    系统大小

Returns:
--------
matplotlib.figure.Figure
    绘制的图形"""
    
    name = "plot_entanglement_entropy"
    description = "绘制不同耦合强度下的纠缠熵

Parameters:
-----------
J_values : numpy.ndarray
    耦合强度值数组
N : int
    系统大小

Returns:
--------
matplotlib.figure.Figure
    绘制的图形"
    arguments = {
        "J_values": {"type": "string", "description": "参数 J_values"},
        "N": {"type": "string", "description": "参数 N"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_entanglement_entropy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_spin_chain import plot_entanglement_entropy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["J_values", "N"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_entanglement_entropy(**func_kwargs)
            
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


@Toolbox.register(name="simulate_temperature_dependence")
class SimulateTemperatureDependenceTool(EnvironmentTool):
    """模拟不同温度下的磁化行为。

Parameters:
-----------
L : int
    晶格的线性尺寸
J : float
    交换耦合强度
T_range : List[float]
    温度范围
periodic : bool
    是否使用周期性边界条件
    
Returns:
--------
tuple
    (温度数组, 磁化强度数组)"""
    
    name = "simulate_temperature_dependence"
    description = "模拟不同温度下的磁化行为。

Parameters:
-----------
L : int
    晶格的线性尺寸
J : float
    交换耦合强度
T_range : List[float]
    温度范围
periodic : bool
    是否使用周期性边界条件
    
Returns:
--------
tuple
    (温度数组, 磁化强度数组)"
    arguments = {
        "L": {"type": "string", "description": "参数 L"},
        "J": {"type": "string", "description": "参数 J"},
        "T_range": {"type": "string", "description": "参数 T_range"},
        "periodic": {"type": "string", "description": "参数 periodic"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 simulate_temperature_dependence 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import simulate_temperature_dependence
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["L", "J", "T_range", "periodic"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = simulate_temperature_dependence(**func_kwargs)
            
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


@Toolbox.register(name="visualize_magnetization")
class VisualizeMagnetizationTool(EnvironmentTool):
    """可视化磁化强度分布。

Parameters:
-----------
mag_2d : ndarray
    二维磁化强度分布
title : str
    图表标题
save_path : str
    保存路径，如果为None则显示图形"""
    
    name = "visualize_magnetization"
    description = "可视化磁化强度分布。

Parameters:
-----------
mag_2d : ndarray
    二维磁化强度分布
title : str
    图表标题
save_path : str
    保存路径，如果为None则显示图形"
    arguments = {
        "mag_2d": {"type": "string", "description": "参数 mag_2d"},
        "title": {"type": "string", "description": "参数 title"},
        "save_path": {"type": "string", "description": "参数 save_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_magnetization 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import visualize_magnetization
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["mag_2d", "title", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_magnetization(**func_kwargs)
            
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


@Toolbox.register(name="visualize_temperature_dependence")
class VisualizeTemperatureDependenceTool(EnvironmentTool):
    """可视化磁化强度随温度的变化。

Parameters:
-----------
T_range : List[float]
    温度数组
magnetizations : List[float]
    磁化强度数组
J : float
    交换耦合强度
save_path : str
    保存路径，如果为None则显示图形"""
    
    name = "visualize_temperature_dependence"
    description = "可视化磁化强度随温度的变化。

Parameters:
-----------
T_range : List[float]
    温度数组
magnetizations : List[float]
    磁化强度数组
J : float
    交换耦合强度
save_path : str
    保存路径，如果为None则显示图形"
    arguments = {
        "T_range": {"type": "string", "description": "参数 T_range"},
        "magnetizations": {"type": "string", "description": "参数 magnetizations"},
        "J": {"type": "string", "description": "参数 J"},
        "save_path": {"type": "string", "description": "参数 save_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_temperature_dependence 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.quantum_heisenberg import visualize_temperature_dependence
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["T_range", "magnetizations", "J", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_temperature_dependence(**func_kwargs)
            
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


@Toolbox.register(name="construct_hubbard_hamiltonian")
class ConstructHubbardHamiltonianTool(EnvironmentTool):
    """构建Hubbard模型的哈密顿量。用于描述强关联电子系统。"""
    
    name = "construct_hubbard_hamiltonian"
    description = "构建Hubbard模型的哈密顿量。用于描述强关联电子系统。"
    arguments = {
        "L": {"type": "integer", "description": "格点数（系统大小）"},
        "N_up": {"type": "integer", "description": "自旋向上电子数"},
        "N_down": {"type": "integer", "description": "自旋向下电子数"},
        "t": {"type": "number", "description": "跃迁积分（动能项），默认1.0"},
        "U": {"type": "number", "description": "库仑相互作用强度，默认1.0"},
        "boundary": {"type": "string", "description": "边界条件：'periodic'（周期性）或'open'（开边界），默认'periodic'", "enum": ["periodic", "open"]},
        "method": {"type": "string", "description": "使用的方法，'quspin'或'tenpy'，默认'quspin'", "enum": ["quspin", "tenpy"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_hubbard_hamiltonian 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "L" not in args or args["L"] is None:
                return Observation(self.name, "错误: 缺少必需参数 L")
            if "N_up" not in args or args["N_up"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N_up")
            if "N_down" not in args or args["N_down"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N_down")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.strongly_correlated_systems import construct_hubbard_hamiltonian
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["L", "N_up", "N_down", "t", "U", "boundary", "method"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_hubbard_hamiltonian(**func_kwargs)
            
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


@Toolbox.register(name="solve_hubbard_ground_state")
class SolveHubbardGroundStateTool(EnvironmentTool):
    """求解Hubbard模型的基态或低能激发态。可直接传入模型参数（L,N_up,N_down,t,U,boundary,method）由函数内部重建哈密顿量；亦兼容传入 H_dict（由构造函数返回）。"""
    
    name = "solve_hubbard_ground_state"
    description = "求解Hubbard模型的基态或低能激发态。可直接传入模型参数（L,N_up,N_down,t,U,boundary,method）由函数内部重建哈密顿量；亦兼容传入 H_dict（由构造函数返回）。"
    arguments = {
        "H_dict": {"type": "object", "description": "可选。由 construct_hubbard_hamiltonian 返回的对象。若提供则忽略 L/N_up/... 参数。"},
        "L": {"type": "integer", "description": "格点数（系统大小）"},
        "N_up": {"type": "integer", "description": "自旋向上电子数"},
        "N_down": {"type": "integer", "description": "自旋向下电子数"},
        "t": {"type": "number", "description": "跃迁积分（动能项），默认1.0"},
        "U": {"type": "number", "description": "库仑相互作用强度，默认1.0"},
        "boundary": {"type": "string", "description": "边界条件，默认'periodic'", "enum": ["periodic", "open"]},
        "method": {"type": "string", "description": "计算方法，默认'quspin'", "enum": ["quspin", "tenpy"]},
        "k": {"type": "integer", "description": "求解前k个本征态，默认1（只求基态）"},
        "which": {"type": "string", "description": "'SA'（最小代数值）或'LA'（最大代数值）", "enum": ["SA", "LA"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_hubbard_ground_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.strongly_correlated_systems import solve_hubbard_ground_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["H_dict", "L", "N_up", "N_down", "t", "U", "boundary", "method", "k", "which"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve_hubbard_ground_state(**func_kwargs)
            
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


@Toolbox.register(name="calculate_double_occupancy")
class CalculateDoubleOccupancyTool(EnvironmentTool):
    """计算Hubbard模型的双占据（double occupancy）。"""
    
    name = "calculate_double_occupancy"
    description = "计算Hubbard模型的双占据（double occupancy）。"
    arguments = {
        "L": {"type": "integer", "description": "格点数（系统大小）"},
        "N_up": {"type": "integer", "description": "自旋向上电子数"},
        "N_down": {"type": "integer", "description": "自旋向下电子数"},
        "t": {"type": "number", "description": "跃迁积分（动能项），默认1.0"},
        "U": {"type": "number", "description": "库仑相互作用强度，默认1.0"},
        "boundary": {"type": "string", "description": "边界条件，默认'periodic'", "enum": ["periodic", "open"]},
        "method": {"type": "string", "description": "计算方法，默认'quspin'", "enum": ["quspin", "tenpy"]},
        "return_total": {"type": "boolean", "description": "True返回总双占据数，False返回包含总数与每格点平均"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_double_occupancy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "L" not in args or args["L"] is None:
                return Observation(self.name, "错误: 缺少必需参数 L")
            if "N_up" not in args or args["N_up"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N_up")
            if "N_down" not in args or args["N_down"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N_down")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.strongly_correlated_systems import calculate_double_occupancy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["L", "N_up", "N_down", "t", "U", "boundary", "method", "return_total"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_double_occupancy(**func_kwargs)
            
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


@Toolbox.register(name="calculate_charge_gap")
class CalculateChargeGapTool(EnvironmentTool):
    """计算电荷能隙。"""
    
    name = "calculate_charge_gap"
    description = "计算电荷能隙。"
    arguments = {
        "L": {"type": "integer", "description": "系统大小（格点数）"},
        "N_up": {"type": "integer", "description": "自旋向上电子数"},
        "N_down": {"type": "integer", "description": "自旋向下电子数"},
        "t": {"type": "number", "description": "跃迁积分"},
        "U": {"type": "number", "description": "相互作用强度"},
        "boundary": {"type": "string", "description": "边界条件，默认'periodic'", "enum": ["periodic", "open"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_charge_gap 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "L" not in args or args["L"] is None:
                return Observation(self.name, "错误: 缺少必需参数 L")
            if "N_up" not in args or args["N_up"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N_up")
            if "N_down" not in args or args["N_down"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N_down")
            if "t" not in args or args["t"] is None:
                return Observation(self.name, "错误: 缺少必需参数 t")
            if "U" not in args or args["U"] is None:
                return Observation(self.name, "错误: 缺少必需参数 U")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.strongly_correlated_systems import calculate_charge_gap
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["L", "N_up", "N_down", "t", "U", "boundary"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_charge_gap(**func_kwargs)
            
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


@Toolbox.register(name="construct_ssh_hamiltonian")
class ConstructSshHamiltonianTool(EnvironmentTool):
    """构建SSH模型（Su-Schrieffer-Heeger）哈密顿量。最简单的1D拓扑绝缘体。"""
    
    name = "construct_ssh_hamiltonian"
    description = "构建SSH模型（Su-Schrieffer-Heeger）哈密顿量。最简单的1D拓扑绝缘体。"
    arguments = {
        "N": {"type": "integer", "description": "单胞数（总格点数为2N）"},
        "t1": {"type": "number", "description": "单胞内跃迁积分（intra-cell hopping）"},
        "t2": {"type": "number", "description": "单胞间跃迁积分（inter-cell hopping）"},
        "periodic": {"type": "boolean", "description": "是否使用周期性边界条件，False表示开边界（可观察边缘态），默认False"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_ssh_hamiltonian 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "N" not in args or args["N"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N")
            if "t1" not in args or args["t1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 t1")
            if "t2" not in args or args["t2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 t2")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.topological_systems import construct_ssh_hamiltonian
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["N", "t1", "t2", "periodic"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_ssh_hamiltonian(**func_kwargs)
            
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


@Toolbox.register(name="calculate_ssh_gap")
class CalculateSshGapTool(EnvironmentTool):
    """计算SSH模型的能隙。可以排除边缘态来计算体态能隙。"""
    
    name = "calculate_ssh_gap"
    description = "计算SSH模型的能隙。可以排除边缘态来计算体态能隙。"
    arguments = {
        "eigenvalues": {"type": "array", "description": "本征能量（已排序）"},
        "exclude_edge_states": {"type": "boolean", "description": "是否排除能隙中的边缘态（零能态），True表示计算体态能隙，默认True"},
        "edge_threshold": {"type": "number", "description": "判断零能态的阈值，默认0.01"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_ssh_gap 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "eigenvalues" not in args or args["eigenvalues"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenvalues")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.topological_systems import calculate_ssh_gap
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["eigenvalues", "exclude_edge_states", "edge_threshold"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_ssh_gap(**func_kwargs)
            
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


@Toolbox.register(name="identify_edge_states")
class IdentifyEdgeStatesTool(EnvironmentTool):
    """识别SSH模型的边缘态。边缘态的特征是能量接近零且波函数局域在边缘。"""
    
    name = "identify_edge_states"
    description = "识别SSH模型的边缘态。边缘态的特征是能量接近零且波函数局域在边缘。"
    arguments = {
        "eigenvalues": {"type": "array", "description": "本征能量"},
        "eigenvectors": {"type": "object", "description": "本征态（列向量）"},
        "gap_threshold": {"type": "number", "description": "判断是否在能隙中的阈值，默认0.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_edge_states 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "eigenvalues" not in args or args["eigenvalues"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenvalues")
            if "eigenvectors" not in args or args["eigenvectors"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenvectors")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.topological_systems import identify_edge_states
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["eigenvalues", "eigenvectors", "gap_threshold"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = identify_edge_states(**func_kwargs)
            
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


@Toolbox.register(name="scan_ssh_model")
class ScanSshModelTool(EnvironmentTool):
    """扫描SSH模型的拓扑相变。固定t₁+t₂扫描δ参数。"""
    
    name = "scan_ssh_model"
    description = "扫描SSH模型的拓扑相变。固定t₁+t₂扫描δ参数。"
    arguments = {
        "N": {"type": "integer", "description": "单胞数"},
        "t_sum": {"type": "number", "description": "t₁ + t₂ 的值"},
        "delta_values": {"type": "array", "description": "δ = (t₁ - t₂)/(t₁ + t₂) 值列表"},
        "periodic": {"type": "boolean", "description": "边界条件，默认False"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 scan_ssh_model 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "N" not in args or args["N"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N")
            if "t_sum" not in args or args["t_sum"] is None:
                return Observation(self.name, "错误: 缺少必需参数 t_sum")
            if "delta_values" not in args or args["delta_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_values")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.topological_systems import scan_ssh_model
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["N", "t_sum", "delta_values", "periodic"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = scan_ssh_model(**func_kwargs)
            
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


@Toolbox.register(name="calculate_parallel_resistance")
class CalculateParallelResistanceTool(EnvironmentTool):
    """计算并联电阻的等效电阻值。

在凝聚态物理中，电子传输网络和量子电路模拟中常需要计算等效电阻。
此函数基于并联电阻公式: 1/R_eq = 1/R₁ + 1/R₂ + ... + 1/Rₙ

Parameters:
-----------
resistances : List[float]
    并联电阻值列表，单位为欧姆(Ω)
    
Returns:
--------
float
    等效电阻值，单位为欧姆(Ω)

Examples:
---------
>>> calculate_parallel_resistance([10.0, 20.0])
6.666666666666667  # 约等于6.67Ω"""
    
    name = "calculate_parallel_resistance"
    description = "计算并联电阻的等效电阻值。

在凝聚态物理中，电子传输网络和量子电路模拟中常需要计算等效电阻。
此函数基于并联电阻公式: 1/R_eq = 1/R₁ + 1/R₂ + ... + 1/Rₙ

Parameters:
-----------
resistances : List[float]
    并联电阻值列表，单位为欧姆(Ω)
    
Returns:
--------
float
    等效电阻值，单位为欧姆(Ω)

Examples:
---------
>>> calculate_parallel_resistance([10.0, 20.0])
6.666666666666667  # 约等于6.67Ω"
    arguments = {
        "resistances": {"type": "string", "description": "参数 resistances"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_parallel_resistance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import calculate_parallel_resistance
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["resistances"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_parallel_resistance(**func_kwargs)
            
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


@Toolbox.register(name="calculate_series_resistance")
class CalculateSeriesResistanceTool(EnvironmentTool):
    """计算串联电阻的等效电阻值。

在凝聚态物理中，电子传输通道和量子点阵列中常需要计算串联电阻。
此函数基于串联电阻公式: R_eq = R₁ + R₂ + ... + Rₙ

Parameters:
-----------
resistances : List[float]
    串联电阻值列表，单位为欧姆(Ω)
    
Returns:
--------
float
    等效电阻值，单位为欧姆(Ω)

Examples:
---------
>>> calculate_series_resistance([10.0, 20.0])
30.0  # 30Ω"""
    
    name = "calculate_series_resistance"
    description = "计算串联电阻的等效电阻值。

在凝聚态物理中，电子传输通道和量子点阵列中常需要计算串联电阻。
此函数基于串联电阻公式: R_eq = R₁ + R₂ + ... + Rₙ

Parameters:
-----------
resistances : List[float]
    串联电阻值列表，单位为欧姆(Ω)
    
Returns:
--------
float
    等效电阻值，单位为欧姆(Ω)

Examples:
---------
>>> calculate_series_resistance([10.0, 20.0])
30.0  # 30Ω"
    arguments = {
        "resistances": {"type": "string", "description": "参数 resistances"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_series_resistance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import calculate_series_resistance
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["resistances"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_series_resistance(**func_kwargs)
            
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


@Toolbox.register(name="solve_complex_circuit")
class SolveComplexCircuitTool(EnvironmentTool):
    """求解复杂电路网络的等效电阻。

通过递归方式处理复杂的电路结构，包含串联和并联组合。
在凝聚态物理中，此类计算对于理解电子传输网络、量子点阵列和纳米线网络至关重要。

Parameters:
-----------
circuit_structure : Dict[str, List[Union[str, List[str]]]]
    电路结构描述，格式为:
    {
        "type": "series" 或 "parallel",
        "components": [
            "R1",  # 直接电阻
            ["parallel", ["R2", "R3"]],  # 嵌套结构
            ...
        ]
    }
resistances : Dict[str, float]
    电阻元件名称到电阻值的映射，单位为欧姆(Ω)
    
Returns:
--------
float
    整个电路的等效电阻值，单位为欧姆(Ω)"""
    
    name = "solve_complex_circuit"
    description = "求解复杂电路网络的等效电阻。

通过递归方式处理复杂的电路结构，包含串联和并联组合。
在凝聚态物理中，此类计算对于理解电子传输网络、量子点阵列和纳米线网络至关重要。

Parameters:
-----------
circuit_structure : Dict[str, List[Union[str, List[str]]]]
    电路结构描述，格式为:
    {
        "type": "series" 或 "parallel",
        "components": [
            "R1",  # 直接电阻
            ["parallel", ["R2", "R3"]],  # 嵌套结构
            ...
        ]
    }
resistances : Dict[str, float]
    电阻元件名称到电阻值的映射，单位为欧姆(Ω)
    
Returns:
--------
float
    整个电路的等效电阻值，单位为欧姆(Ω)"
    arguments = {
        "circuit_structure": {"type": "string", "description": "参数 circuit_structure"},
        "resistances": {"type": "string", "description": "参数 resistances"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_complex_circuit 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import solve_complex_circuit
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["circuit_structure", "resistances"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve_complex_circuit(**func_kwargs)
            
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


@Toolbox.register(name="visualize_circuit")
class VisualizeCircuitTool(EnvironmentTool):
    """可视化电路结构并显示等效电阻。

使用NetworkX创建电路的图形表示，便于理解电路拓扑结构。

Parameters:
-----------
circuit_structure : Dict[str, List[Union[str, List[str]]]]
    电路结构描述，与solve_complex_circuit函数使用相同格式
resistances : Dict[str, float]
    电阻元件名称到电阻值的映射，单位为欧姆(Ω)
filename : str, optional
    图像保存的文件名，默认为"circuit_visualization.png""""
    
    name = "visualize_circuit"
    description = "可视化电路结构并显示等效电阻。

使用NetworkX创建电路的图形表示，便于理解电路拓扑结构。

Parameters:
-----------
circuit_structure : Dict[str, List[Union[str, List[str]]]]
    电路结构描述，与solve_complex_circuit函数使用相同格式
resistances : Dict[str, float]
    电阻元件名称到电阻值的映射，单位为欧姆(Ω)
filename : str, optional
    图像保存的文件名，默认为"circuit_visualization.png""
    arguments = {
        "circuit_structure": {"type": "string", "description": "参数 circuit_structure"},
        "resistances": {"type": "string", "description": "参数 resistances"},
        "filename": {"type": "string", "description": "参数 filename"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_circuit 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import visualize_circuit
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["circuit_structure", "resistances", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_circuit(**func_kwargs)
            
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


@Toolbox.register(name="calculate_expectation_value")
class CalculateExpectationValueTool(EnvironmentTool):
    """计算量子力学中的期望值。

在量子多体系统模拟中，期望值是预测物理量测量结果的关键。
此函数计算给定密度矩阵和算符的期望值: ⟨O⟩ = Tr(ρO)

Parameters:
-----------
operator : np.ndarray
    要计算期望值的量子算符，形状为(n, n)的矩阵
density_matrix : np.ndarray
    系统的密度矩阵，形状为(n, n)的矩阵
    
Returns:
--------
complex
    算符的期望值，对于厄米算符，结果应为实数"""
    
    name = "calculate_expectation_value"
    description = "计算量子力学中的期望值。

在量子多体系统模拟中，期望值是预测物理量测量结果的关键。
此函数计算给定密度矩阵和算符的期望值: ⟨O⟩ = Tr(ρO)

Parameters:
-----------
operator : np.ndarray
    要计算期望值的量子算符，形状为(n, n)的矩阵
density_matrix : np.ndarray
    系统的密度矩阵，形状为(n, n)的矩阵
    
Returns:
--------
complex
    算符的期望值，对于厄米算符，结果应为实数"
    arguments = {
        "operator": {"type": "string", "description": "参数 operator"},
        "density_matrix": {"type": "string", "description": "参数 density_matrix"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_expectation_value 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import calculate_expectation_value
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["operator", "density_matrix"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_expectation_value(**func_kwargs)
            
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


@Toolbox.register(name="visualize_time_evolution")
class VisualizeTimeEvolutionTool(EnvironmentTool):
    """可视化量子系统时间演化的结果。

绘制算符期望值随时间的变化，帮助理解量子系统的动力学行为。

Parameters:
-----------
time_points : np.ndarray
    时间点数组
results : Dict[str, np.ndarray]
    simulate_time_evolution函数返回的结果字典
filename : str, optional
    图像保存的文件名，默认为"time_evolution.png""""
    
    name = "visualize_time_evolution"
    description = "可视化量子系统时间演化的结果。

绘制算符期望值随时间的变化，帮助理解量子系统的动力学行为。

Parameters:
-----------
time_points : np.ndarray
    时间点数组
results : Dict[str, np.ndarray]
    simulate_time_evolution函数返回的结果字典
filename : str, optional
    图像保存的文件名，默认为"time_evolution.png""
    arguments = {
        "time_points": {"type": "string", "description": "参数 time_points"},
        "results": {"type": "string", "description": "参数 results"},
        "filename": {"type": "string", "description": "参数 filename"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_time_evolution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import visualize_time_evolution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["time_points", "results", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_time_evolution(**func_kwargs)
            
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


@Toolbox.register(name="calculate_dft_energy")
class CalculateDftEnergyTool(EnvironmentTool):
    """模拟第一性原理计算中的密度泛函理论(DFT)能量计算。

此函数提供了一个简化的DFT能量计算模型，用于材料性质预测。
实际应用中通常会使用专业软件包如VASP、Quantum ESPRESSO等。

Parameters:
-----------
atomic_positions : np.ndarray
    原子位置坐标，形状为(n_atoms, 3)，单位为埃(Å)
atomic_numbers : np.ndarray
    原子序数数组，长度为n_atoms
functional : str, optional
    使用的交换关联泛函，默认为'LDA'
basis_set : str, optional
    使用的基组，默认为'minimal'
    
Returns:
--------
float
    系统的总能量，单位为电子伏特(eV)"""
    
    name = "calculate_dft_energy"
    description = "模拟第一性原理计算中的密度泛函理论(DFT)能量计算。

此函数提供了一个简化的DFT能量计算模型，用于材料性质预测。
实际应用中通常会使用专业软件包如VASP、Quantum ESPRESSO等。

Parameters:
-----------
atomic_positions : np.ndarray
    原子位置坐标，形状为(n_atoms, 3)，单位为埃(Å)
atomic_numbers : np.ndarray
    原子序数数组，长度为n_atoms
functional : str, optional
    使用的交换关联泛函，默认为'LDA'
basis_set : str, optional
    使用的基组，默认为'minimal'
    
Returns:
--------
float
    系统的总能量，单位为电子伏特(eV)"
    arguments = {
        "atomic_positions": {"type": "string", "description": "参数 atomic_positions"},
        "atomic_numbers": {"type": "string", "description": "参数 atomic_numbers"},
        "functional": {"type": "string", "description": "参数 functional"},
        "basis_set": {"type": "string", "description": "参数 basis_set"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_dft_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import calculate_dft_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["atomic_positions", "atomic_numbers", "functional", "basis_set"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_dft_energy(**func_kwargs)
            
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


@Toolbox.register(name="optimize_crystal_structure")
class OptimizeCrystalStructureTool(EnvironmentTool):
    """优化晶体结构以找到能量最低的构型。

在材料科学和凝聚态物理中，结构优化是预测稳定材料结构的关键步骤。
此函数使用数值优化方法寻找能量最低的原子构型。

Parameters:
-----------
initial_positions : np.ndarray
    初始原子位置坐标，形状为(n_atoms, 3)，单位为埃(Å)
atomic_numbers : np.ndarray
    原子序数数组，长度为n_atoms
functional : str, optional
    使用的交换关联泛函，默认为'LDA'
basis_set : str, optional
    使用的基组，默认为'minimal'
    
Returns:
--------
Tuple[np.ndarray, float]
    优化后的原子位置坐标和对应的能量值"""
    
    name = "optimize_crystal_structure"
    description = "优化晶体结构以找到能量最低的构型。

在材料科学和凝聚态物理中，结构优化是预测稳定材料结构的关键步骤。
此函数使用数值优化方法寻找能量最低的原子构型。

Parameters:
-----------
initial_positions : np.ndarray
    初始原子位置坐标，形状为(n_atoms, 3)，单位为埃(Å)
atomic_numbers : np.ndarray
    原子序数数组，长度为n_atoms
functional : str, optional
    使用的交换关联泛函，默认为'LDA'
basis_set : str, optional
    使用的基组，默认为'minimal'
    
Returns:
--------
Tuple[np.ndarray, float]
    优化后的原子位置坐标和对应的能量值"
    arguments = {
        "initial_positions": {"type": "string", "description": "参数 initial_positions"},
        "atomic_numbers": {"type": "string", "description": "参数 atomic_numbers"},
        "functional": {"type": "string", "description": "参数 functional"},
        "basis_set": {"type": "string", "description": "参数 basis_set"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 optimize_crystal_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import optimize_crystal_structure
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["initial_positions", "atomic_numbers", "functional", "basis_set"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = optimize_crystal_structure(**func_kwargs)
            
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


@Toolbox.register(name="energy_function")
class EnergyFunctionTool(EnvironmentTool):
    """"""
    
    name = "energy_function"
    description = ""
    arguments = {
        "positions_flat": {"type": "string", "description": "参数 positions_flat"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 energy_function 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_10979 import energy_function
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["positions_flat"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = energy_function(**func_kwargs)
            
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


@Toolbox.register(name="bernoulli_function")
class BernoulliFunctionTool(EnvironmentTool):
    """计算Bernoulli函数 B(x) = x / (exp(x) - 1)，使用数值稳定的算法避免在x≈0时的除零问题和大x时的溢出。"""
    
    name = "bernoulli_function"
    description = "计算Bernoulli函数 B(x) = x / (exp(x) - 1)，使用数值稳定的算法避免在x≈0时的除零问题和大x时的溢出。"
    arguments = {
        "x": {"type": "number", "description": "无量纲参数（通常是 δψ/Vt）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 bernoulli_function 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "x" not in args or args["x"] is None:
                return Observation(self.name, "错误: 缺少必需参数 x")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import bernoulli_function
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["x"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = bernoulli_function(**func_kwargs)
            
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


@Toolbox.register(name="calculate_thermal_voltage")
class CalculateThermalVoltageTool(EnvironmentTool):
    """计算给定温度下的热电压 Vt = kB*T/q。"""
    
    name = "calculate_thermal_voltage"
    description = "计算给定温度下的热电压 Vt = kB*T/q。"
    arguments = {
        "temperature": {"type": "number", "description": "温度 (K)，范围 [1, 1000]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_thermal_voltage 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "temperature" not in args or args["temperature"] is None:
                return Observation(self.name, "错误: 缺少必需参数 temperature")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import calculate_thermal_voltage
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["temperature"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_thermal_voltage(**func_kwargs)
            
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


@Toolbox.register(name="calculate_potential_difference")
class CalculatePotentialDifferenceTool(EnvironmentTool):
    """计算两个网格点之间的电势差。"""
    
    name = "calculate_potential_difference"
    description = "计算两个网格点之间的电势差。"
    arguments = {
        "psi_i": {"type": "number", "description": "网格点i的电势 (V)"},
        "psi_j": {"type": "number", "description": "网格点j的电势 (V)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_potential_difference 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "psi_i" not in args or args["psi_i"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_i")
            if "psi_j" not in args or args["psi_j"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_j")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import calculate_potential_difference
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["psi_i", "psi_j"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_potential_difference(**func_kwargs)
            
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


@Toolbox.register(name="calculate_normalized_potential")
class CalculateNormalizedPotentialTool(EnvironmentTool):
    """计算归一化电势 δψ/Vt。"""
    
    name = "calculate_normalized_potential"
    description = "计算归一化电势 δψ/Vt。"
    arguments = {
        "delta_psi": {"type": "number", "description": "电势差 (V)"},
        "vt": {"type": "number", "description": "热电压 (V)，必须 > 0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_normalized_potential 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "delta_psi" not in args or args["delta_psi"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_psi")
            if "vt" not in args or args["vt"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vt")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import calculate_normalized_potential
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["delta_psi", "vt"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_normalized_potential(**func_kwargs)
            
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


@Toolbox.register(name="scharfetter_gummel_current")
class ScharfetterGummelCurrentTool(EnvironmentTool):
    """计算Scharfetter-Gummel离散化的电子电流密度。"""
    
    name = "scharfetter_gummel_current"
    description = "计算Scharfetter-Gummel离散化的电子电流密度。"
    arguments = {
        "n_i": {"type": "number", "description": "网格点i的电子浓度 (cm^-3)"},
        "n_j": {"type": "number", "description": "网格点j的电子浓度 (cm^-3)"},
        "psi_i": {"type": "number", "description": "网格点i的电势 (V)"},
        "psi_j": {"type": "number", "description": "网格点j的电势 (V)"},
        "mu_n": {"type": "number", "description": "电子迁移率 (cm^2/V·s)"},
        "dx": {"type": "number", "description": "网格间距 (cm)"},
        "vt": {"type": "number", "description": "热电压 (V)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 scharfetter_gummel_current 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_i" not in args or args["n_i"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_i")
            if "n_j" not in args or args["n_j"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_j")
            if "psi_i" not in args or args["psi_i"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_i")
            if "psi_j" not in args or args["psi_j"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_j")
            if "mu_n" not in args or args["mu_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_n")
            if "dx" not in args or args["dx"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dx")
            if "vt" not in args or args["vt"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vt")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import scharfetter_gummel_current
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_i", "n_j", "psi_i", "psi_j", "mu_n", "dx", "vt"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = scharfetter_gummel_current(**func_kwargs)
            
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


@Toolbox.register(name="verify_sg_formula_structure")
class VerifySgFormulaStructureTool(EnvironmentTool):
    """验证Scharfetter-Gummel公式的结构正确性。"""
    
    name = "verify_sg_formula_structure"
    description = "验证Scharfetter-Gummel公式的结构正确性。"
    arguments = {
        "n_i": {"type": "number", "description": "网格点i的电子浓度"},
        "n_j": {"type": "number", "description": "网格点j (即i+1) 的电子浓度"},
        "delta_psi": {"type": "number", "description": "电势差 δψ_{i+1}"},
        "mu_n": {"type": "number", "description": "电子迁移率"},
        "dx": {"type": "number", "description": "网格间距"},
        "vt": {"type": "number", "description": "热电压"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 verify_sg_formula_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_i" not in args or args["n_i"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_i")
            if "n_j" not in args or args["n_j"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_j")
            if "delta_psi" not in args or args["delta_psi"] is None:
                return Observation(self.name, "错误: 缺少必需参数 delta_psi")
            if "mu_n" not in args or args["mu_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_n")
            if "dx" not in args or args["dx"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dx")
            if "vt" not in args or args["vt"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vt")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import verify_sg_formula_structure
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_i", "n_j", "delta_psi", "mu_n", "dx", "vt"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = verify_sg_formula_structure(**func_kwargs)
            
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


@Toolbox.register(name="create_semiconductor_database")
class CreateSemiconductorDatabaseTool(EnvironmentTool):
    """创建半导体材料参数数据库。"""
    
    name = "create_semiconductor_database"
    description = "创建半导体材料参数数据库。"
    arguments = {
        "db_path": {"type": "string", "description": "数据库文件路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_semiconductor_database 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import create_semiconductor_database
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["db_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_semiconductor_database(**func_kwargs)
            
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


@Toolbox.register(name="query_material_properties")
class QueryMaterialPropertiesTool(EnvironmentTool):
    """查询半导体材料的物理参数。"""
    
    name = "query_material_properties"
    description = "查询半导体材料的物理参数。"
    arguments = {
        "material_name": {"type": "string", "description": "材料名称（如 'Si', 'GaAs'）"},
        "db_path": {"type": "string", "description": "数据库路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 query_material_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "material_name" not in args or args["material_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 material_name")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import query_material_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["material_name", "db_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = query_material_properties(**func_kwargs)
            
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


@Toolbox.register(name="plot_bernoulli_function")
class PlotBernoulliFunctionTool(EnvironmentTool):
    """绘制Bernoulli函数曲线。"""
    
    name = "plot_bernoulli_function"
    description = "绘制Bernoulli函数曲线。"
    arguments = {
        "x_range": {"type": "array", "description": "x轴范围 [x_min, x_max]"},
        "num_points": {"type": "integer", "description": "采样点数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_bernoulli_function 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import plot_bernoulli_function
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["x_range", "num_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_bernoulli_function(**func_kwargs)
            
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


@Toolbox.register(name="analyze_sg_discretization_accuracy")
class AnalyzeSgDiscretizationAccuracyTool(EnvironmentTool):
    """分析Scharfetter-Gummel离散化在整个器件中的精度。"""
    
    name = "analyze_sg_discretization_accuracy"
    description = "分析Scharfetter-Gummel离散化在整个器件中的精度。"
    arguments = {
        "n_profile": {"type": "array", "description": "电子浓度分布 (cm^-3)"},
        "psi_profile": {"type": "array", "description": "电势分布 (V)"},
        "mu_n": {"type": "number", "description": "电子迁移率 (cm^2/V·s)"},
        "dx": {"type": "number", "description": "网格间距 (cm)"},
        "vt": {"type": "number", "description": "热电压 (V)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_sg_discretization_accuracy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_profile" not in args or args["n_profile"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_profile")
            if "psi_profile" not in args or args["psi_profile"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_profile")
            if "mu_n" not in args or args["mu_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_n")
            if "dx" not in args or args["dx"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dx")
            if "vt" not in args or args["vt"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vt")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.semiconductor_drift_diffusion_toolkit_claude_177 import analyze_sg_discretization_accuracy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_profile", "psi_profile", "mu_n", "dx", "vt"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_sg_discretization_accuracy(**func_kwargs)
            
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


@Toolbox.register(name="electron_motion_in_field")
class ElectronMotionInFieldTool(EnvironmentTool):
    """计算电子在恒定电场下的运动

基于经典力学和半导体物理学原理，计算电子在恒定电场下的加速运动。
F = qE = m*a，其中q是电子电荷，E是电场强度，m是有效质量，a是加速度。

Parameters:
-----------
electric_field : float
    电场强度，单位为V/m
effective_mass : float
    电子的有效质量，以电子静止质量(m_e)为单位
time_span : float or tuple
    如果是float，表示从0到该值的时间范围；如果是tuple，表示(t_start, t_end)
initial_velocity : float, optional
    初始速度，单位为m/s，默认为0
    
Returns:
--------
tuple
    (时间数组, 速度数组, 位置数组)，单位分别为s, m/s, m"""
    
    name = "electron_motion_in_field"
    description = "计算电子在恒定电场下的运动

基于经典力学和半导体物理学原理，计算电子在恒定电场下的加速运动。
F = qE = m*a，其中q是电子电荷，E是电场强度，m是有效质量，a是加速度。

Parameters:
-----------
electric_field : float
    电场强度，单位为V/m
effective_mass : float
    电子的有效质量，以电子静止质量(m_e)为单位
time_span : float or tuple
    如果是float，表示从0到该值的时间范围；如果是tuple，表示(t_start, t_end)
initial_velocity : float, optional
    初始速度，单位为m/s，默认为0
    
Returns:
--------
tuple
    (时间数组, 速度数组, 位置数组)，单位分别为s, m/s, m"
    arguments = {
        "electric_field": {"type": "string", "description": "参数 electric_field"},
        "effective_mass": {"type": "string", "description": "参数 effective_mass"},
        "time_span": {"type": "string", "description": "参数 time_span"},
        "initial_velocity": {"type": "string", "description": "参数 initial_velocity"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 electron_motion_in_field 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_18998 import electron_motion_in_field
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["electric_field", "effective_mass", "time_span", "initial_velocity"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = electron_motion_in_field(**func_kwargs)
            
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


@Toolbox.register(name="band_structure_1d")
class BandStructure1dTool(EnvironmentTool):
    """计算一维晶体的能带结构

基于不同的能带模型（抛物线、紧束缚等）计算能带结构。

Parameters:
-----------
k_range : tuple or array
    如果是tuple，表示(k_min, k_max, num_points)；如果是array，直接使用该k点数组
band_params : dict
    能带参数，根据band_type不同而不同：
    - 'parabolic': {'effective_mass': float, 'band_edge': float}
    - 'tight_binding': {'hopping': float, 'lattice_constant': float}
    - 'kronig_penney': {'V0': float, 'a': float, 'b': float}
band_type : str, optional
    能带模型类型，可选'parabolic'（抛物线近似）, 'tight_binding'（紧束缚模型）, 
    'kronig_penney'（Kronig-Penney模型）
    
Returns:
--------
tuple
    (k点数组, 能量数组)，单位分别为1/m和eV"""
    
    name = "band_structure_1d"
    description = "计算一维晶体的能带结构

基于不同的能带模型（抛物线、紧束缚等）计算能带结构。

Parameters:
-----------
k_range : tuple or array
    如果是tuple，表示(k_min, k_max, num_points)；如果是array，直接使用该k点数组
band_params : dict
    能带参数，根据band_type不同而不同：
    - 'parabolic': {'effective_mass': float, 'band_edge': float}
    - 'tight_binding': {'hopping': float, 'lattice_constant': float}
    - 'kronig_penney': {'V0': float, 'a': float, 'b': float}
band_type : str, optional
    能带模型类型，可选'parabolic'（抛物线近似）, 'tight_binding'（紧束缚模型）, 
    'kronig_penney'（Kronig-Penney模型）
    
Returns:
--------
tuple
    (k点数组, 能量数组)，单位分别为1/m和eV"
    arguments = {
        "k_range": {"type": "string", "description": "参数 k_range"},
        "band_params": {"type": "string", "description": "参数 band_params"},
        "band_type": {"type": "string", "description": "参数 band_type"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 band_structure_1d 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_18998 import band_structure_1d
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["k_range", "band_params", "band_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = band_structure_1d(**func_kwargs)
            
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


@Toolbox.register(name="monte_carlo_phase_transition")
class MonteCarloPhaseTransitionTool(EnvironmentTool):
    """使用蒙特卡洛方法模拟二维Ising模型的相变

基于Metropolis算法实现的蒙特卡洛模拟，用于研究二维Ising模型的相变现象。

Parameters:
-----------
lattice_size : int
    晶格大小（N x N）
temperature_range : tuple or array
    如果是tuple，表示(T_min, T_max, num_points)；如果是array，直接使用该温度数组
J : float, optional
    交换耦合常数，默认为1.0
num_steps : int, optional
    每个温度点的蒙特卡洛步数，默认为10000
equilibration : int, optional
    平衡化步数（在计算物理量前的热化步数），默认为1000
    
Returns:
--------
tuple
    (温度数组, 能量数组, 磁化强度数组, 比热容数组, 磁化率数组)"""
    
    name = "monte_carlo_phase_transition"
    description = "使用蒙特卡洛方法模拟二维Ising模型的相变

基于Metropolis算法实现的蒙特卡洛模拟，用于研究二维Ising模型的相变现象。

Parameters:
-----------
lattice_size : int
    晶格大小（N x N）
temperature_range : tuple or array
    如果是tuple，表示(T_min, T_max, num_points)；如果是array，直接使用该温度数组
J : float, optional
    交换耦合常数，默认为1.0
num_steps : int, optional
    每个温度点的蒙特卡洛步数，默认为10000
equilibration : int, optional
    平衡化步数（在计算物理量前的热化步数），默认为1000
    
Returns:
--------
tuple
    (温度数组, 能量数组, 磁化强度数组, 比热容数组, 磁化率数组)"
    arguments = {
        "lattice_size": {"type": "string", "description": "参数 lattice_size"},
        "temperature_range": {"type": "string", "description": "参数 temperature_range"},
        "J": {"type": "string", "description": "参数 J"},
        "num_steps": {"type": "string", "description": "参数 num_steps"},
        "equilibration": {"type": "string", "description": "参数 equilibration"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 monte_carlo_phase_transition 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_18998 import monte_carlo_phase_transition
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["lattice_size", "temperature_range", "J", "num_steps", "equilibration"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = monte_carlo_phase_transition(**func_kwargs)
            
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


@Toolbox.register(name="quantum_state_evolution")
class QuantumStateEvolutionTool(EnvironmentTool):
    """计算量子态在给定哈密顿量下的时间演化

基于量子力学的薛定谔方程，计算量子态随时间的演化。

Parameters:
-----------
hamiltonian : numpy.ndarray
    哈密顿量矩阵，形状为(n, n)
initial_state : numpy.ndarray
    初始量子态，形状为(n,)
time_span : tuple or array
    如果是tuple，表示(t_start, t_end, num_points)；如果是array，直接使用该时间数组
method : str, optional
    演化方法，可选'exact'（精确对角化）或'runge_kutta'（龙格-库塔法），默认为'exact'
    
Returns:
--------
tuple
    (时间数组, 量子态数组)，量子态数组形状为(time_points, n)"""
    
    name = "quantum_state_evolution"
    description = "计算量子态在给定哈密顿量下的时间演化

基于量子力学的薛定谔方程，计算量子态随时间的演化。

Parameters:
-----------
hamiltonian : numpy.ndarray
    哈密顿量矩阵，形状为(n, n)
initial_state : numpy.ndarray
    初始量子态，形状为(n,)
time_span : tuple or array
    如果是tuple，表示(t_start, t_end, num_points)；如果是array，直接使用该时间数组
method : str, optional
    演化方法，可选'exact'（精确对角化）或'runge_kutta'（龙格-库塔法），默认为'exact'
    
Returns:
--------
tuple
    (时间数组, 量子态数组)，量子态数组形状为(time_points, n)"
    arguments = {
        "hamiltonian": {"type": "string", "description": "参数 hamiltonian"},
        "initial_state": {"type": "string", "description": "参数 initial_state"},
        "time_span": {"type": "string", "description": "参数 time_span"},
        "method": {"type": "string", "description": "参数 method"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 quantum_state_evolution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_18998 import quantum_state_evolution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["hamiltonian", "initial_state", "time_span", "method"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = quantum_state_evolution(**func_kwargs)
            
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


@Toolbox.register(name="schrodinger_eq")
class SchrodingerEqTool(EnvironmentTool):
    """"""
    
    name = "schrodinger_eq"
    description = ""
    arguments = {
        "t": {"type": "string", "description": "参数 t"},
        "psi": {"type": "string", "description": "参数 psi"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 schrodinger_eq 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.condensed_matter_toolkit_18998 import schrodinger_eq
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["t", "psi"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = schrodinger_eq(**func_kwargs)
            
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


@Toolbox.register(name="construct_anderson_hamiltonian_1d")
class ConstructAndersonHamiltonian1dTool(EnvironmentTool):
    """构建一维Anderson模型哈密顿量。用于研究无序系统中的局域化现象。"""
    
    name = "construct_anderson_hamiltonian_1d"
    description = "构建一维Anderson模型哈密顿量。用于研究无序系统中的局域化现象。"
    arguments = {
        "N": {"type": "integer", "description": "系统格点数"},
        "t": {"type": "number", "description": "跃迁积分（hopping amplitude），默认1.0"},
        "disorder_strength": {"type": "number", "description": "无序强度W（随机势的范围），默认0.0"},
        "periodic": {"type": "boolean", "description": "是否使用周期性边界条件，默认False"},
        "seed": {"type": "integer", "description": "随机数种子，用于可重复性"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_anderson_hamiltonian_1d 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "N" not in args or args["N"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.disorder_localization import construct_anderson_hamiltonian_1d
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["N", "t", "disorder_strength", "periodic", "seed"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_anderson_hamiltonian_1d(**func_kwargs)
            
            # 处理返回值：将 scipy 稀疏矩阵和 numpy 数组转换为 JSON 可序列化格式
            result = convert_to_json_serializable(result)
            
            # 处理返回值格式（tuple 转换为字典）
            if isinstance(result, tuple):
                # 将元组转换为字典，包含 hamiltonian 和 potential
                result_dict = {
                    "hamiltonian": result[0] if len(result) > 0 else None, 
                    "potential": result[1] if len(result) > 1 else None
                }
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            elif isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_inverse_participation_ratio")
class CalculateInverseParticipationRatioTool(EnvironmentTool):
    """计算逆参与率（IPR）。用于判断波函数的局域化程度。"""
    
    name = "calculate_inverse_participation_ratio"
    description = "计算逆参与率（IPR）。用于判断波函数的局域化程度。"
    arguments = {
        "wavefunction": {"type": "array", "description": "波函数（本征态），一维数组"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_inverse_participation_ratio 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wavefunction" not in args or args["wavefunction"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wavefunction")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.disorder_localization import calculate_inverse_participation_ratio
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wavefunction"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_inverse_participation_ratio(**func_kwargs)
            
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


@Toolbox.register(name="analyze_localization_disorder_average")
class AnalyzeLocalizationDisorderAverageTool(EnvironmentTool):
    """对多个无序样本进行平均分析。Anderson局域化研究的标准方法。"""
    
    name = "analyze_localization_disorder_average"
    description = "对多个无序样本进行平均分析。Anderson局域化研究的标准方法。"
    arguments = {
        "N": {"type": "integer", "description": "系统格点数"},
        "t": {"type": "number", "description": "跃迁积分"},
        "disorder_strengths": {"type": "array", "description": "无序强度列表（W值），例如 [0,1,2,4,8]"},
        "num_samples": {"type": "integer", "description": "样本数量，默认100"},
        "energy_window": {"type": "array", "description": "能量窗口 [E_min, E_max]，可选"},
        "periodic": {"type": "boolean", "description": "是否使用周期性边界条件，默认False"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_localization_disorder_average 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "N" not in args or args["N"] is None:
                return Observation(self.name, "错误: 缺少必需参数 N")
            if "t" not in args or args["t"] is None:
                return Observation(self.name, "错误: 缺少必需参数 t")
            if "disorder_strengths" not in args or args["disorder_strengths"] is None:
                return Observation(self.name, "错误: 缺少必需参数 disorder_strengths")
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.disorder_localization import analyze_localization_disorder_average
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["N", "t", "disorder_strengths", "num_samples", "energy_window", "periodic"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_localization_disorder_average(**func_kwargs)
            
            # 处理返回值：将 numpy 数组转换为列表以便 JSON 序列化
            result = convert_to_json_serializable(result)
            
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


@Toolbox.register(name="test1_tight_binding_consistency")
class Test1TightBindingConsistencyTool(EnvironmentTool):
    """测试1：紧束缚模型一致性"""
    
    name = "test1_tight_binding_consistency"
    description = "测试1：紧束缚模型一致性"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 test1_tight_binding_consistency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.cross_validation import test1_tight_binding_consistency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = test1_tight_binding_consistency(**func_kwargs)
            
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


@Toolbox.register(name="test2_ising_small_system")
class Test2IsingSmallSystemTool(EnvironmentTool):
    """测试2：Ising模型（小系统精确验证）"""
    
    name = "test2_ising_small_system"
    description = "测试2：Ising模型（小系统精确验证）"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 test2_ising_small_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.cross_validation import test2_ising_small_system
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = test2_ising_small_system(**func_kwargs)
            
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


@Toolbox.register(name="test3_entanglement_entropy")
class Test3EntanglementEntropyTool(EnvironmentTool):
    """测试3：量子纠缠熵"""
    
    name = "test3_entanglement_entropy"
    description = "测试3：量子纠缠熵"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 test3_entanglement_entropy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.cross_validation import test3_entanglement_entropy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = test3_entanglement_entropy(**func_kwargs)
            
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


@Toolbox.register(name="test4_numerical_stability")
class Test4NumericalStabilityTool(EnvironmentTool):
    """测试4：数值稳定性"""
    
    name = "test4_numerical_stability"
    description = "测试4：数值稳定性"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 test4_numerical_stability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.cross_validation import test4_numerical_stability
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = test4_numerical_stability(**func_kwargs)
            
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


@Toolbox.register(name="check_optional_packages")
class CheckOptionalPackagesTool(EnvironmentTool):
    """检查可选包的安装情况"""
    
    name = "check_optional_packages"
    description = "检查可选包的安装情况"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 check_optional_packages 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.cross_validation import check_optional_packages
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = check_optional_packages(**func_kwargs)
            
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


@Toolbox.register(name="run_all_tests")
class RunAllTestsTool(EnvironmentTool):
    """运行所有测试"""
    
    name = "run_all_tests"
    description = "运行所有测试"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 run_all_tests 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.condensed_matter_physics.cross_validation import run_all_tests
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = run_all_tests(**func_kwargs)
            
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

