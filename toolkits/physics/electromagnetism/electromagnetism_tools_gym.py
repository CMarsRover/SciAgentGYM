#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
electromagnetism 工具注册模块
使用 gym.tool.EnvironmentTool 为 electromagnetism 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool, convert_to_json_serializable
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.physics.electromagnetism.circuit_analyzer_5496 import *  # 动态导入
# from toolkits.physics.electromagnetism.electromagnetic_field_calculator_197 import *  # 动态导入
# from toolkits.physics.electromagnetism.electromagnetic_field_solver_157 import *  # 动态导入
# from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import *  # 动态导入
# from toolkits.physics.electromagnetism.magnetic_materials_analyzer_18856 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="calculate_magnetic_field_line_current")
class CalculateMagneticFieldLineCurrentTool(EnvironmentTool):
    """计算无限长直线电流在空间某点产生的磁感应强度。基于毕奥-萨伐尔定律，无限长直线电流在距离为r的点产生的磁场大小为: B = (μ₀*I)/(2πr)，方向由右手螺旋定则确定。"""
    
    name = "calculate_magnetic_field_line_current"
    description = "计算无限长直线电流在空间某点产生的磁感应强度。基于毕奥-萨伐尔定律，无限长直线电流在距离为r的点产生的磁场大小为: B = (μ₀*I)/(2πr)，方向由右手螺旋定则确定。"
    arguments = {
        "r_vector": {"type": "array", "description": "场点位置矢量，形状为(3,)，表示(x,y,z)坐标"},
        "current_vector": {"type": "array", "description": "电流方向的单位矢量，形状为(3,)"},
        "current_position": {"type": "array", "description": "电流线所在直线上一点的位置矢量，形状为(3,)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_magnetic_field_line_current 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            r_vector = args.get("r_vector")
            if r_vector is None:
                return Observation(self.name, "错误: 缺少必需参数 r_vector")
            current_vector = args.get("current_vector")
            if current_vector is None:
                return Observation(self.name, "错误: 缺少必需参数 current_vector")
            current_position = args.get("current_position")
            if current_position is None:
                return Observation(self.name, "错误: 缺少必需参数 current_position")
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_157 import calculate_magnetic_field_line_current
            
            # 调用函数
            result = calculate_magnetic_field_line_current(r_vector, current_vector, current_position)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_total_magnetic_field")
class CalculateTotalMagneticFieldTool(EnvironmentTool):
    """计算多个电流源在空间某点产生的总磁感应强度。"""
    
    name = "calculate_total_magnetic_field"
    description = "计算多个电流源在空间某点产生的总磁感应强度。"
    arguments = {
        "r_vector": {"type": "array", "description": "场点位置矢量，形状为(3,)，表示(x,y,z)坐标"},
        "current_sources": {"type": "array", "description": "电流源列表，每个电流源为一个字典，包含：'position': 电流线上一点的位置矢量，'direction': 电流方向的单位矢量，'magnitude': 电流大小，单位为安培(A)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_total_magnetic_field 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            r_vector = args.get("r_vector")
            if r_vector is None:
                return Observation(self.name, "错误: 缺少必需参数 r_vector")
            current_sources = args.get("current_sources")
            if current_sources is None:
                return Observation(self.name, "错误: 缺少必需参数 current_sources")
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_157 import calculate_total_magnetic_field
            
            # 调用函数
            result = calculate_total_magnetic_field(r_vector, current_sources)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_conical_conductor_potential")
class CalculateConicalConductorPotentialTool(EnvironmentTool):
    """计算无限长导体圆锥在导体平面上的电位分布。该函数基于拉普拉斯方程在球坐标系中的解，适用于轴对称电位分布问题。"""
    
    name = "calculate_conical_conductor_potential"
    description = "计算无限长导体圆锥在导体平面上的电位分布。该函数基于拉普拉斯方程在球坐标系中的解，适用于轴对称电位分布问题。"
    arguments = {
        "theta": {"type": "number", "description": "极角，单位为弧度，有效范围为[alpha, pi/2]"},
        "alpha": {"type": "number", "description": "圆锥半夹角，单位为弧度，范围(0, pi/2)"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_conical_conductor_potential 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            theta = args.get("theta")
            if theta is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import calculate_conical_conductor_potential
            
            # 调用函数
            result = calculate_conical_conductor_potential(theta, alpha, phi_0)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="derive_potential_constants")
class DerivePotentialConstantsTool(EnvironmentTool):
    """从轴对称拉普拉斯角向方程推导电位分布常数。"""
    
    name = "derive_potential_constants"
    description = "从轴对称拉普拉斯角向方程推导电位分布常数。"
    arguments = {
        "alpha": {"type": "number", "description": "圆锥半夹角，弧度，(0, π/2)"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"},
        "verbose": {"type": "boolean", "description": "若 True，打印推导与线性求解细节，默认为False"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 derive_potential_constants 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            verbose = args.get("verbose", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import derive_potential_constants
            
            # 调用函数
            result = derive_potential_constants(alpha, phi_0, verbose)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="potential_via_integration")
class PotentialViaIntegrationTool(EnvironmentTool):
    """使用显式积分推导得到的解析式计算电位，并可选打印推导步骤与常数。"""
    
    name = "potential_via_integration"
    description = "使用显式积分推导得到的解析式计算电位，并可选打印推导步骤与常数。"
    arguments = {
        "theta": {"type": "number", "description": "极角（弧度），取值区间 [alpha, π/2]"},
        "alpha": {"type": "number", "description": "圆锥半夹角（弧度），(0, π/2)"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"},
        "verbose": {"type": "boolean", "description": "若为 True，则打印积分推导关键步骤与 A、B 常数，默认为False"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 potential_via_integration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            theta = args.get("theta")
            if theta is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            verbose = args.get("verbose", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import potential_via_integration
            
            # 调用函数
            result = potential_via_integration(theta, alpha, phi_0, verbose)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_electric_field")
class CalculateElectricFieldTool(EnvironmentTool):
    """计算无限长导体圆锥在导体平面上的电场分布。电场是电位的负梯度，在球坐标系中分为径向和角向分量。"""
    
    name = "calculate_electric_field"
    description = "计算无限长导体圆锥在导体平面上的电场分布。电场是电位的负梯度，在球坐标系中分为径向和角向分量。"
    arguments = {
        "r": {"type": "number", "description": "径向距离，单位为米"},
        "theta": {"type": "number", "description": "极角，单位为弧度，有效范围为[alpha, pi/2]"},
        "alpha": {"type": "number", "description": "圆锥半夹角，单位为弧度，范围(0, pi/2)"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_electric_field 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            r = args.get("r")
            if r is None:
                return Observation(self.name, "错误: 缺少必需参数 r")
            theta = args.get("theta")
            if theta is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import calculate_electric_field
            
            # 调用函数
            result = calculate_electric_field(r, theta, alpha, phi_0)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_charge_density")
class CalculateChargeDensityTool(EnvironmentTool):
    """计算导体表面的电荷密度分布。根据高斯定理，导体表面的电荷密度等于ε₀乘以电场的法向分量。"""
    
    name = "calculate_charge_density"
    description = "计算导体表面的电荷密度分布。根据高斯定理，导体表面的电荷密度等于ε₀乘以电场的法向分量。"
    arguments = {
        "r": {"type": "number", "description": "径向距离，单位为米"},
        "theta": {"type": "number", "description": "极角，单位为弧度"},
        "alpha": {"type": "number", "description": "圆锥半夹角，单位为弧度"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"},
        "epsilon_0": {"type": "number", "description": "真空介电常数，默认为8.85e-12 F/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_charge_density 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            r = args.get("r")
            if r is None:
                return Observation(self.name, "错误: 缺少必需参数 r")
            theta = args.get("theta")
            if theta is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            epsilon_0 = args.get("epsilon_0", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import calculate_charge_density
            
            # 调用函数
            result = calculate_charge_density(r, theta, alpha, phi_0, epsilon_0)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="plot_potential_distribution")
class PlotPotentialDistributionTool(EnvironmentTool):
    """绘制电位分布图。"""
    
    name = "plot_potential_distribution"
    description = "绘制电位分布图。"
    arguments = {
        "alpha": {"type": "number", "description": "圆锥半夹角，单位为弧度"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"},
        "resolution": {"type": "integer", "description": "网格分辨率，默认为100"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_potential_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            resolution = args.get("resolution", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import plot_potential_distribution
            
            # 调用函数
            result = plot_potential_distribution(alpha, phi_0, resolution)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="plot_electric_field")
class PlotElectricFieldTool(EnvironmentTool):
    """绘制电场分布图。"""
    
    name = "plot_electric_field"
    description = "绘制电场分布图。"
    arguments = {
        "alpha": {"type": "number", "description": "圆锥半夹角，单位为弧度"},
        "phi_0": {"type": "number", "description": "圆锥表面的电位值，默认为1.0"},
        "resolution": {"type": "integer", "description": "网格分辨率，默认为20"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_electric_field 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            alpha = args.get("alpha")
            if alpha is None:
                return Observation(self.name, "错误: 缺少必需参数 alpha")
            phi_0 = args.get("phi_0", None)
            resolution = args.get("resolution", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_solver_161 import plot_electric_field
            
            # 调用函数
            result = plot_electric_field(alpha, phi_0, resolution)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_mutual_inductance_wire_loop")
class CalculateMutualInductanceWireLoopTool(EnvironmentTool):
    """计算直导线与任意形状闭合回路之间的互感系数。基于比奥-萨伐尔定律和法拉第电磁感应定律，计算直导线产生的磁场通过闭合回路的磁通量，从而得到互感系数。"""
    
    name = "calculate_mutual_inductance_wire_loop"
    description = "计算直导线与任意形状闭合回路之间的互感系数。基于比奥-萨伐尔定律和法拉第电磁感应定律，计算直导线产生的磁场通过闭合回路的磁通量，从而得到互感系数。"
    arguments = {
        "geometry_func": {"type": "object", "description": "描述闭合回路几何形状的函数，接收参数t(参数方程的参数)，返回回路上对应点的坐标(x,y,z)和切向量(dx,dy,dz)"},
        "wire_position": {"type": "array", "description": "直导线的位置坐标 (x, y, z)"},
        "current_direction": {"type": "array", "description": "直导线的电流方向单位向量 (dx, dy, dz)"},
        "integration_limits": {"type": "array", "description": "积分参数的上下限 (t_min, t_max)"},
        "num_points": {"type": "integer", "description": "数值积分使用的点数，默认为1000"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_mutual_inductance_wire_loop 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            geometry_func = args.get("geometry_func")
            if geometry_func is None:
                return Observation(self.name, "错误: 缺少必需参数 geometry_func")
            wire_position = args.get("wire_position")
            if wire_position is None:
                return Observation(self.name, "错误: 缺少必需参数 wire_position")
            current_direction = args.get("current_direction")
            if current_direction is None:
                return Observation(self.name, "错误: 缺少必需参数 current_direction")
            integration_limits = args.get("integration_limits")
            if integration_limits is None:
                return Observation(self.name, "错误: 缺少必需参数 integration_limits")
            num_points = args.get("num_points", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_calculator_197 import calculate_mutual_inductance_wire_loop
            
            # 调用函数
            result = calculate_mutual_inductance_wire_loop(geometry_func, wire_position, current_direction, integration_limits, num_points)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_mutual_inductance_wire_triangle")
class CalculateMutualInductanceWireTriangleTool(EnvironmentTool):
    """计算无限长直导线与等边三角形线框之间的互感。"""
    
    name = "calculate_mutual_inductance_wire_triangle"
    description = "计算无限长直导线与等边三角形线框之间的互感。"
    arguments = {
        "d": {"type": "number", "description": "直导线到三角形最近顶点的距离，单位为米"},
        "a": {"type": "number", "description": "等边三角形的边长，单位为米"},
        "analytical": {"type": "boolean", "description": "是否使用解析解，默认为True。若为False则使用数值积分"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_mutual_inductance_wire_triangle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            d = args.get("d")
            if d is None:
                return Observation(self.name, "错误: 缺少必需参数 d")
            a = args.get("a")
            if a is None:
                return Observation(self.name, "错误: 缺少必需参数 a")
            analytical = args.get("analytical", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_calculator_197 import calculate_mutual_inductance_wire_triangle
            
            # 调用函数
            result = calculate_mutual_inductance_wire_triangle(d, a, analytical)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="infinite_wire_magnetic_field")
class InfiniteWireMagneticFieldTool(EnvironmentTool):
    """计算无限长直导线在给定点产生的磁场。"""
    
    name = "infinite_wire_magnetic_field"
    description = "计算无限长直导线在给定点产生的磁场。"
    arguments = {
        "point": {"type": "array", "description": "观测点的坐标 (x, y, z)"},
        "wire_position": {"type": "array", "description": "导线的位置坐标 (x, y, z)，表示导线上的一点"},
        "current_direction": {"type": "array", "description": "导线的方向单位向量 (dx, dy, dz)"},
        "current": {"type": "number", "description": "电流大小，单位为安培，默认为1.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 infinite_wire_magnetic_field 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            point = args.get("point")
            if point is None:
                return Observation(self.name, "错误: 缺少必需参数 point")
            wire_position = args.get("wire_position")
            if wire_position is None:
                return Observation(self.name, "错误: 缺少必需参数 wire_position")
            current_direction = args.get("current_direction")
            if current_direction is None:
                return Observation(self.name, "错误: 缺少必需参数 current_direction")
            current = args.get("current", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.electromagnetic_field_calculator_197 import infinite_wire_magnetic_field
            
            # 调用函数
            result = infinite_wire_magnetic_field(point, wire_position, current_direction, current)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="essential_circuit_analysis_guide")
class EssentialCircuitAnalysisGuideTool(EnvironmentTool):
    """🔧 等效电路分析工具 - 基本阅读指南。这是一个多模态分析工具，帮助您正确分析电路图并计算等效电阻。"""
    
    name = "essential_circuit_analysis_guide"
    description = "🔧 等效电路分析工具 - 基本阅读指南。这是一个多模态分析工具，帮助您正确分析电路图并计算等效电阻。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 essential_circuit_analysis_guide 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            

            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import essential_circuit_analysis_guide
            
            # 调用函数
            result = essential_circuit_analysis_guide()
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_parallel_resistance")
class CalculateParallelResistanceTool(EnvironmentTool):
    """计算并联电阻的等效电阻。在并联连接中，等效电阻的倒数是所有单个电阻倒数的总和。公式：1/R_eq = 1/R1 + 1/R2 + ... + 1/Rn"""
    
    name = "calculate_parallel_resistance"
    description = "计算并联电阻的等效电阻。在并联连接中，等效电阻的倒数是所有单个电阻倒数的总和。公式：1/R_eq = 1/R1 + 1/R2 + ... + 1/Rn"
    arguments = {
        "resistances": {"type": "array", "description": "电阻值列表，单位为欧姆(Ω)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_parallel_resistance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            resistances = args.get("resistances")
            if resistances is None:
                return Observation(self.name, "错误: 缺少必需参数 resistances")
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import calculate_parallel_resistance
            
            # 调用函数
            result = calculate_parallel_resistance(resistances)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_series_resistance")
class CalculateSeriesResistanceTool(EnvironmentTool):
    """计算串联电阻的等效电阻。在串联连接中，等效电阻是所有单个电阻的总和。公式：R_eq = R1 + R2 + ... + Rn"""
    
    name = "calculate_series_resistance"
    description = "计算串联电阻的等效电阻。在串联连接中，等效电阻是所有单个电阻的总和。公式：R_eq = R1 + R2 + ... + Rn"
    arguments = {
        "resistances": {"type": "array", "description": "电阻值列表，单位为欧姆(Ω)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_series_resistance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            resistances = args.get("resistances")
            if resistances is None:
                return Observation(self.name, "错误: 缺少必需参数 resistances")
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import calculate_series_resistance
            
            # 调用函数
            result = calculate_series_resistance(resistances)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="essential_circuit_analysis_guide")
class EssentialCircuitAnalysisGuideTool(EnvironmentTool):
    """🔧 等效电路分析工具 - 基本阅读指南。这是一个多模态分析工具，帮助您正确分析电路图并计算等效电阻。"""
    
    name = "essential_circuit_analysis_guide"
    description = "🔧 等效电路分析工具 - 基本阅读指南。这是一个多模态分析工具，帮助您正确分析电路图并计算等效电阻。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 essential_circuit_analysis_guide 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            

            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import essential_circuit_analysis_guide
            
            # 调用函数
            result = essential_circuit_analysis_guide()
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_parallel_resistance")
class CalculateParallelResistanceTool(EnvironmentTool):
    """计算并联电阻的等效电阻。在并联连接中，等效电阻的倒数是所有单个电阻倒数的总和。公式：1/R_eq = 1/R1 + 1/R2 + ... + 1/Rn"""
    
    name = "calculate_parallel_resistance"
    description = "计算并联电阻的等效电阻。在并联连接中，等效电阻的倒数是所有单个电阻倒数的总和。公式：1/R_eq = 1/R1 + 1/R2 + ... + 1/Rn"
    arguments = {
        "resistances": {"type": "array", "description": "电阻值列表，单位为欧姆(Ω)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_parallel_resistance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            resistances = args.get("resistances")
            if resistances is None:
                return Observation(self.name, "错误: 缺少必需参数 resistances")
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import calculate_parallel_resistance
            
            # 调用函数
            result = calculate_parallel_resistance(resistances)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_series_resistance")
class CalculateSeriesResistanceTool(EnvironmentTool):
    """计算串联电阻的等效电阻。在串联连接中，等效电阻是所有单个电阻的总和。公式：R_eq = R1 + R2 + ... + Rn"""
    
    name = "calculate_series_resistance"
    description = "计算串联电阻的等效电阻。在串联连接中，等效电阻是所有单个电阻的总和。公式：R_eq = R1 + R2 + ... + Rn"
    arguments = {
        "resistances": {"type": "array", "description": "电阻值列表，单位为欧姆(Ω)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_series_resistance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            resistances = args.get("resistances")
            if resistances is None:
                return Observation(self.name, "错误: 缺少必需参数 resistances")
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import calculate_series_resistance
            
            # 调用函数
            result = calculate_series_resistance(resistances)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="essential_circuit_analysis_guide")
class EssentialCircuitAnalysisGuideTool(EnvironmentTool):
    """🔧 等效电路分析工具 - 基本阅读指南。这是一个多模态分析工具，帮助您正确分析电路图并计算等效电阻。"""
    
    name = "essential_circuit_analysis_guide"
    description = "🔧 等效电路分析工具 - 基本阅读指南。这是一个多模态分析工具，帮助您正确分析电路图并计算等效电阻。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 essential_circuit_analysis_guide 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            

            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.circuit_analyzer_5496 import essential_circuit_analysis_guide
            
            # 调用函数
            result = essential_circuit_analysis_guide()
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_flux_density")
class CalculateFluxDensityTool(EnvironmentTool):
    """计算给定材料与磁场强度(H)的磁通密度(B)"""
    
    name = "calculate_flux_density"
    description = "计算给定材料与磁场强度(H)的磁通密度(B)"
    arguments = {
        "material_name": {"type": "string", "description": "磁性材料名称"},
        "h_value": {"type": "number", "description": "磁场强度(A/m)"},
        "curve_points": {"type": "object", "description": "可选自定义B–H数据点"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_flux_density 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            material_name = args.get("material_name")
            if material_name is None:
                return Observation(self.name, "错误: 缺少必需参数 material_name")
            h_value = args.get("h_value")
            if h_value is None:
                return Observation(self.name, "错误: 缺少必需参数 h_value")
            curve_points = args.get("curve_points", None)
            
            # 导入并调用原始函数
            from toolkits.physics.electromagnetism.magnetic_materials_analyzer_18856 import calculate_flux_density
            
            # 调用函数
            result = calculate_flux_density(material_name, h_value, curve_points)
            
            # 处理返回值：转换为 JSON 可序列化格式
            if isinstance(result, (dict, list)):
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": convert_to_json_serializable(list(result))}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                # 转换 numpy 类型（如 int64, float64）为 Python 原生类型
                result = convert_to_json_serializable(result)
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


# ==================== 工具注册函数 ====================

def register_electromagnetism_tools(environment):
    """
    将所有 electromagnetism 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

