#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crystallography 工具注册模块
使用 gym.tool.EnvironmentTool 为 crystallography 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.materials_science.crystallography.crystallography_toolkit_0005 import *  # 动态导入
# from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import *  # 动态导入
# from toolkits.materials_science.crystallography.materials_toolkit_M002_0000 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="identify_lattice_system")
class IdentifyLatticeSystemTool(EnvironmentTool):
    """根据晶格参数识别晶系类型。基于晶格常数和角度的数值关系，判断7大晶系。"""
    
    name = "identify_lattice_system"
    description = "根据晶格参数识别晶系类型。基于晶格常数和角度的数值关系，判断7大晶系。"
    arguments = {
        "a": {"type": "number", "description": "晶格常数a (Å)"},
        "b": {"type": "number", "description": "晶格常数b (Å)"},
        "c": {"type": "number", "description": "晶格常数c (Å)"},
        "alpha": {"type": "number", "description": "晶格角度α (度)"},
        "beta": {"type": "number", "description": "晶格角度β (度)"},
        "gamma": {"type": "number", "description": "晶格角度γ (度)"},
        "tolerance": {"type": "number", "description": "判断相等的角度容差 (度)，默认5.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_lattice_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            a = args.get("a")
            if a is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            b = args.get("b")
            if b is None:
                return Observation(self.name, "错误: 缺少必需参数 b")
            c = args.get("c")
            if c is None:
                return Observation(self.name, "错误: 缺少必需参数 c")
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            beta = args.get("beta")
            if beta is None:
                return Observation(self.name, "错误: 缺少必需参数 beta")
            gamma = args.get("gamma")
            if gamma is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0005 import identify_lattice_system
            
            # 调用函数
            result = identify_lattice_system(a, b, c, alpha, beta, gamma, tolerance)
            
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


@Toolbox.register(name="identify_lattice_type")
class IdentifyLatticeTypeTool(EnvironmentTool):
    """识别格子类型（P、I、F、A、B、C、R）。基于原子位置分布，判断原始、体心、面心等格子类型。"""
    
    name = "identify_lattice_type"
    description = "识别格子类型（P、I、F、A、B、C、R）。基于原子位置分布，判断原始、体心、面心等格子类型。"
    arguments = {
        "positions": {"type": "array", "description": "原子位置数组，形状为(n, 3)"},
        "crystal_system": {"type": "string", "description": "晶系名称", "enum": ["cubic", "tetragonal", "orthorhombic", "hexagonal", "rhombohedral", "monoclinic", "triclinic"]}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_lattice_type 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            positions = args.get("positions")
            if positions is None:
                return Observation(self.name, "错误: 缺少必需参数 positions")
            crystal_system = args.get("crystal_system")
            if crystal_system is None:
                return Observation(self.name, "错误: 缺少必需参数 crystal_system")
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0005 import identify_lattice_type
            
            # 调用函数
            result = identify_lattice_type(positions, crystal_system)
            
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


@Toolbox.register(name="detect_symmetry_operations")
class DetectSymmetryOperationsTool(EnvironmentTool):
    """检测晶体结构中的对称操作。识别反演中心、旋转轴、镜面、滑移面和螺旋轴。"""
    
    name = "detect_symmetry_operations"
    description = "检测晶体结构中的对称操作。识别反演中心、旋转轴、镜面、滑移面和螺旋轴。"
    arguments = {
        "positions": {"type": "array", "description": "原子位置数组，形状为(n, 3)"},
        "tolerance": {"type": "number", "description": "对称性判断容差 (Å)，默认0.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_symmetry_operations 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            positions = args.get("positions")
            if positions is None:
                return Observation(self.name, "错误: 缺少必需参数 positions")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0005 import detect_symmetry_operations
            
            # 调用函数
            result = detect_symmetry_operations(positions, tolerance)
            
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


@Toolbox.register(name="determine_space_group")
class DetermineSpaceGroupTool(EnvironmentTool):
    """基于晶系、格子类型和对称操作确定空间群符号。使用Hermann-Mauguin记号法。"""
    
    name = "determine_space_group"
    description = "基于晶系、格子类型和对称操作确定空间群符号。使用Hermann-Mauguin记号法。"
    arguments = {
        "crystal_system": {"type": "string", "description": "晶系名称", "enum": ["cubic", "tetragonal", "orthorhombic", "hexagonal", "rhombohedral", "monoclinic", "triclinic"]},
        "lattice_type": {"type": "string", "description": "格子类型符号", "enum": ["P", "I", "F", "A", "B", "C", "R"]},
        "symmetry_ops": {"type": "object", "description": "检测到的对称操作字典"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 determine_space_group 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            crystal_system = args.get("crystal_system")
            if crystal_system is None:
                return Observation(self.name, "错误: 缺少必需参数 crystal_system")
            lattice_type = args.get("lattice_type")
            if lattice_type is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_type")
            symmetry_ops = args.get("symmetry_ops")
            if symmetry_ops is None:
                return Observation(self.name, "错误: 缺少必需参数 symmetry_ops")
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0005 import determine_space_group
            
            # 调用函数
            result = determine_space_group(crystal_system, lattice_type, symmetry_ops)
            
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


@Toolbox.register(name="visualize_crystal_structure")
class VisualizeCrystalStructureTool(EnvironmentTool):
    """可视化晶体结构。生成3D结构图，显示原子、化学键和单胞。"""
    
    name = "visualize_crystal_structure"
    description = "可视化晶体结构。生成3D结构图，显示原子、化学键和单胞。"
    arguments = {
        "positions": {"type": "array", "description": "原子位置数组"},
        "elements": {"type": "array", "description": "元素符号列表"},
        "lattice_params": {"type": "object", "description": "晶格参数字典"},
        "space_group": {"type": "string", "description": "空间群符号"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_crystal_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            positions = args.get("positions")
            if positions is None:
                return Observation(self.name, "错误: 缺少必需参数 positions")
            elements = args.get("elements", None)
            lattice_params = args.get("lattice_params", None)
            space_group = args.get("space_group", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0005 import visualize_crystal_structure
            
            # 调用函数
            result = visualize_crystal_structure(positions, elements, lattice_params, space_group)
            
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


@Toolbox.register(name="identify_crystal_system")
class IdentifyCrystalSystemTool(EnvironmentTool):
    """根据晶格参数识别晶系。基于晶格常数的比值关系和角度判断晶系类型（立方、四方、六方、正交、单斜、三斜）。"""
    
    name = "identify_crystal_system"
    description = "根据晶格参数识别晶系。基于晶格常数的比值关系和角度判断晶系类型（立方、四方、六方、正交、单斜、三斜）。"
    arguments = {
        "lattice_params": {"type": "object", "description": "晶格参数字典，包含 {'a': float, 'b': float, 'c': float, 'alpha': float, 'beta': float, 'gamma': float}，长度单位Å，角度单位度"},
        "tolerance": {"type": "number", "description": "判断相等的相对容差，默认0.01（1%）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 identify_crystal_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice_params = args.get("lattice_params")
            if lattice_params is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_params")
            tolerance = args.get("tolerance", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import identify_crystal_system
            
            # 调用函数
            result = identify_crystal_system(lattice_params, tolerance)
            
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


@Toolbox.register(name="analyze_symmetry_operations_from_parameters")
class AnalyzeSymmetryOperationsFromParametersTool(EnvironmentTool):
    """基于参数分析对称操作（OpenAI Function Calling兼容）。从晶格参数和原子坐标构建Structure对象，然后分析对称操作。"""
    
    name = "analyze_symmetry_operations_from_parameters"
    description = "基于参数分析对称操作（OpenAI Function Calling兼容）。从晶格参数和原子坐标构建Structure对象，然后分析对称操作。"
    arguments = {
        "lattice_params": {"type": "object", "description": "晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}，长度单位Å，角度单位度"},
        "species": {"type": "array", "description": "原子种类列表，如['Al', 'Ir']"},
        "coords": {"type": "array", "description": "分数坐标列表，每个坐标为[x, y, z]，范围0-1"},
        "symprec": {"type": "number", "description": "对称性分析精度/Å，默认0.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_symmetry_operations_from_parameters 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice_params = args.get("lattice_params")
            if lattice_params is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_params")
            species = args.get("species")
            if species is None:
                return Observation(self.name, "错误: 缺少必需参数 species")
            coords = args.get("coords")
            if coords is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            symprec = args.get("symprec", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import analyze_symmetry_operations_from_parameters
            
            # 调用函数
            result = analyze_symmetry_operations_from_parameters(lattice_params, species, coords, symprec)
            
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


@Toolbox.register(name="get_spacegroup_from_parameters")
class GetSpacegroupFromParametersTool(EnvironmentTool):
    """基于参数获取空间群信息（OpenAI Function Calling兼容）。从晶格参数和原子坐标构建Structure对象，然后获取空间群信息。"""
    
    name = "get_spacegroup_from_parameters"
    description = "基于参数获取空间群信息（OpenAI Function Calling兼容）。从晶格参数和原子坐标构建Structure对象，然后获取空间群信息。"
    arguments = {
        "lattice_params": {"type": "object", "description": "晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}，长度单位Å，角度单位度"},
        "species": {"type": "array", "description": "原子种类列表，如['Al', 'Ir']"},
        "coords": {"type": "array", "description": "分数坐标列表，每个坐标为[x, y, z]，范围0-1"},
        "symprec": {"type": "number", "description": "对称性分析精度/Å，默认0.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_spacegroup_from_parameters 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice_params = args.get("lattice_params")
            if lattice_params is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_params")
            species = args.get("species")
            if species is None:
                return Observation(self.name, "错误: 缺少必需参数 species")
            coords = args.get("coords")
            if coords is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            symprec = args.get("symprec", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import get_spacegroup_from_parameters
            
            # 调用函数
            result = get_spacegroup_from_parameters(lattice_params, species, coords, symprec)
            
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


@Toolbox.register(name="verify_p63mmc_symmetry_from_parameters")
class VerifyP63mmcSymmetryFromParametersTool(EnvironmentTool):
    """基于参数验证P6₃/mmc对称性（OpenAI Function Calling兼容）。从晶格参数和原子坐标构建Structure对象，然后验证是否符合P6₃/mmc空间群。"""
    
    name = "verify_p63mmc_symmetry_from_parameters"
    description = "基于参数验证P6₃/mmc对称性（OpenAI Function Calling兼容）。从晶格参数和原子坐标构建Structure对象，然后验证是否符合P6₃/mmc空间群。"
    arguments = {
        "lattice_params": {"type": "object", "description": "晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}，长度单位Å，角度单位度"},
        "species": {"type": "array", "description": "原子种类列表，如['Al', 'Ir']"},
        "coords": {"type": "array", "description": "分数坐标列表，每个坐标为[x, y, z]，范围0-1"},
        "symprec": {"type": "number", "description": "对称性分析精度/Å，默认0.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 verify_p63mmc_symmetry_from_parameters 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice_params = args.get("lattice_params")
            if lattice_params is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_params")
            species = args.get("species")
            if species is None:
                return Observation(self.name, "错误: 缺少必需参数 species")
            coords = args.get("coords")
            if coords is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            symprec = args.get("symprec", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import verify_p63mmc_symmetry_from_parameters
            
            # 调用函数
            result = verify_p63mmc_symmetry_from_parameters(lattice_params, species, coords, symprec)
            
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


@Toolbox.register(name="create_hexagonal_test_structure")
class CreateHexagonalTestStructureTool(EnvironmentTool):
    """创建六方晶系测试结构。根据给定的晶格参数和原子坐标构建六方晶胞，用于空间群验证。"""
    
    name = "create_hexagonal_test_structure"
    description = "创建六方晶系测试结构。根据给定的晶格参数和原子坐标构建六方晶胞，用于空间群验证。"
    arguments = {
        "a": {"type": "number", "description": "六方晶格的a轴长度/Å（a=b）"},
        "c": {"type": "number", "description": "六方晶格的c轴长度/Å"},
        "species": {"type": "array", "description": "原子种类列表，如['Al', 'Ir']"},
        "coords": {"type": "array", "description": "分数坐标列表，每个坐标为[x, y, z]，范围0-1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_hexagonal_test_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            a = args.get("a")
            if a is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            c = args.get("c")
            if c is None:
                return Observation(self.name, "错误: 缺少必需参数 c")
            species = args.get("species")
            if species is None:
                return Observation(self.name, "错误: 缺少必需参数 species")
            coords = args.get("coords")
            if coords is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import create_hexagonal_test_structure
            
            # 调用函数
            result = create_hexagonal_test_structure(a, c, species, coords)
            
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


@Toolbox.register(name="determine_spacegroup_from_lattice")
class DetermineSpacegroupFromLatticeTool(EnvironmentTool):
    """从晶格参数和原子坐标确定空间群。组合调用晶系识别、结构创建和对称性分析完成完整的空间群确定流程。"""
    
    name = "determine_spacegroup_from_lattice"
    description = "从晶格参数和原子坐标确定空间群。组合调用晶系识别、结构创建和对称性分析完成完整的空间群确定流程。"
    arguments = {
        "lattice_params": {"type": "object", "description": "晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}，长度单位Å，角度单位度"},
        "species": {"type": "array", "description": "原子种类列表，如['Al', 'Ir']"},
        "coords": {"type": "array", "description": "分数坐标列表，每个坐标为[x, y, z]，范围0-1"},
        "symprec": {"type": "number", "description": "对称性分析精度/Å，默认0.1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 determine_spacegroup_from_lattice 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            lattice_params = args.get("lattice_params")
            if lattice_params is None:
                return Observation(self.name, "错误: 缺少必需参数 lattice_params")
            species = args.get("species")
            if species is None:
                return Observation(self.name, "错误: 缺少必需参数 species")
            coords = args.get("coords")
            if coords is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            symprec = args.get("symprec", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import determine_spacegroup_from_lattice
            
            # 调用函数
            result = determine_spacegroup_from_lattice(lattice_params, species, coords, symprec)
            
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


@Toolbox.register(name="batch_analyze_hexagonal_structures")
class BatchAnalyzeHexagonalStructuresTool(EnvironmentTool):
    """批量分析不同晶格参数的六方结构。扫描a和c参数空间，识别每个结构的空间群，用于参数-对称性关系研究。"""
    
    name = "batch_analyze_hexagonal_structures"
    description = "批量分析不同晶格参数的六方结构。扫描a和c参数空间，识别每个结构的空间群，用于参数-对称性关系研究。"
    arguments = {
        "a_values": {"type": "array", "description": "a轴长度列表/Å"},
        "c_values": {"type": "array", "description": "c轴长度列表/Å"},
        "species": {"type": "array", "description": "原子种类，默认['Al', 'Ir']"},
        "coords": {"type": "array", "description": "分数坐标，默认None时使用[[0,0,0], [1/3,2/3,1/2]]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_analyze_hexagonal_structures 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            a_values = args.get("a_values")
            if a_values is None:
                return Observation(self.name, "错误: 缺少必需参数 a_values")
            c_values = args.get("c_values")
            if c_values is None:
                return Observation(self.name, "错误: 缺少必需参数 c_values")
            species = args.get("species", None)
            coords = args.get("coords", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import batch_analyze_hexagonal_structures
            
            # 调用函数
            result = batch_analyze_hexagonal_structures(a_values, c_values, species, coords)
            
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


@Toolbox.register(name="plot_symmetry_analysis")
class PlotSymmetryAnalysisTool(EnvironmentTool):
    """可视化对称性分析结果。生成对称元素分布图，包括旋转轴、镜面、滑移面的统计可视化。"""
    
    name = "plot_symmetry_analysis"
    description = "可视化对称性分析结果。生成对称元素分布图，包括旋转轴、镜面、滑移面的统计可视化。"
    arguments = {
        "symmetry_data": {"type": "object", "description": "analyze_symmetry_operations()的返回结果"},
        "plot_type": {"type": "string", "description": "'interactive'使用plotly, 'static'使用matplotlib", "enum": ["interactive", "static"]},
        "save_path": {"type": "string", "description": "图片保存路径，默认'./images/'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_symmetry_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            symmetry_data = args.get("symmetry_data")
            if symmetry_data is None:
                return Observation(self.name, "错误: 缺少必需参数 symmetry_data")
            plot_type = args.get("plot_type", None)
            save_path = args.get("save_path", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.crystallography_toolkit_0006 import plot_symmetry_analysis
            
            # 调用函数
            result = plot_symmetry_analysis(symmetry_data, plot_type, save_path)
            
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


@Toolbox.register(name="fetch_property_from_database")
class FetchPropertyFromDatabaseTool(EnvironmentTool):
    """从Materials Project（mp-api）获取指定材料的性质或结构。"""
    
    name = "fetch_property_from_database"
    description = "从Materials Project（mp-api）获取指定材料的性质或结构。"
    arguments = {
        "identifier": {"type": "string", "description": "材料ID如'mp-XXXX'。"},
        "property_name": {"type": "string", "description": "请求的属性名称，如'spacegroup.symbol'或'structure'。"},
        "fields": {"type": "array", "description": "可选，额外需要返回的字段列表。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_property_from_database 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            identifier = args.get("identifier")
            if identifier is None:
                return Observation(self.name, "错误: 缺少必需参数 identifier")
            property_name = args.get("property_name")
            if property_name is None:
                return Observation(self.name, "错误: 缺少必需参数 property_name")
            fields = args.get("fields", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.materials_toolkit_M002_0000 import fetch_property_from_database
            
            # 调用函数
            result = fetch_property_from_database(identifier, property_name, fields)
            
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


@Toolbox.register(name="analyze_space_group")
class AnalyzeSpaceGroupTool(EnvironmentTool):
    """基于Pymatgen对结构进行空间群识别。"""
    
    name = "analyze_space_group"
    description = "基于Pymatgen对结构进行空间群识别。"
    arguments = {
        "structure_input": {"type": "string", "description": "结构来源：CIF文件路径；也可接受pymatgen结构字典。"},
        "symprec": {"type": "number", "description": "对称识别数值容差，典型范围1e-5到1e-1。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_space_group 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            structure_input = args.get("structure_input")
            if structure_input is None:
                return Observation(self.name, "错误: 缺少必需参数 structure_input")
            symprec = args.get("symprec", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.materials_toolkit_M002_0000 import analyze_space_group
            
            # 调用函数
            result = analyze_space_group(structure_input, symprec)
            
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


@Toolbox.register(name="calibrate_spacegroup_notation")
class CalibrateSpacegroupNotationTool(EnvironmentTool):
    """将空间群符号转换到指定目标记号以进行统一比较。"""
    
    name = "calibrate_spacegroup_notation"
    description = "将空间群符号转换到指定目标记号以进行统一比较。"
    arguments = {
        "symbol": {"type": "string", "description": "识别得到的空间群符号。"},
        "target": {"type": "string", "description": "目标符号。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calibrate_spacegroup_notation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            symbol = args.get("symbol")
            if symbol is None:
                return Observation(self.name, "错误: 缺少必需参数 symbol")
            target = args.get("target", None)
            
            # 导入并调用原始函数
            from toolkits.materials_science.crystallography.materials_toolkit_M002_0000 import calibrate_spacegroup_notation
            
            # 调用函数
            result = calibrate_spacegroup_notation(symbol, target)
            
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

def register_crystallography_tools(environment):
    """
    将所有 crystallography 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

