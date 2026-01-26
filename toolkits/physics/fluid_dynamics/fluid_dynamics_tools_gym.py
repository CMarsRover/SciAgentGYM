#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fluid_dynamics 工具注册模块
使用 gym.tool.EnvironmentTool 为 fluid_dynamics 目录中的工具提供统一的注册与调用接口

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


@Toolbox.register(name="calculate_system_acceleration")
class CalculateSystemAccelerationTool(EnvironmentTool):
    """计算滑轮系统的加速度

物理原理：
- 对m2：T - m2*g = -m2*a (向下为正)
- 对m1+水：T - f = (m1+m_water)*a
- 摩擦力：f = μ*N = μ*(m1+m_water)*g

联立求解：a = (m2*g - μ*(m1+m_water)*g) / (m1+m_water+m2)

Parameters:
-----------
m1 : float
    容器质量 (kg)
m2 : float
    重物质量 (kg)
mu : float
    摩擦系数
g : float
    重力加速度 (m/s^2)

Returns:
--------
dict : {'result': float, 'metadata': dict}
    加速度值及计算详情"""
    
    name = "calculate_system_acceleration"
    description = "计算滑轮系统的加速度

物理原理：
- 对m2：T - m2*g = -m2*a (向下为正)
- 对m1+水：T - f = (m1+m_water)*a
- 摩擦力：f = μ*N = μ*(m1+m_water)*g

联立求解：a = (m2*g - μ*(m1+m_water)*g) / (m1+m_water+m2)

Parameters:
-----------
m1 : float
    容器质量 (kg)
m2 : float
    重物质量 (kg)
mu : float
    摩擦系数
g : float
    重力加速度 (m/s^2)

Returns:
--------
dict : {'result': float, 'metadata': dict}
    加速度值及计算详情"
    arguments = {
        "m1": {"type": "string", "description": "参数 m1"},
        "m2": {"type": "string", "description": "参数 m2"},
        "mu": {"type": "string", "description": "参数 mu"},
        "g": {"type": "string", "description": "参数 g"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_system_acceleration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import calculate_system_acceleration
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["m1", "m2", "mu", "g"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_system_acceleration(**func_kwargs)
            
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


@Toolbox.register(name="calculate_water_mass")
class CalculateWaterMassTool(EnvironmentTool):
    """计算容器中水的质量

Parameters:
-----------
base_area : float
    容器底面积 (m^2)
height : float
    水位高度 (m)
density : float
    水的密度 (kg/m^3), 默认1000

Returns:
--------
dict : {'result': float, 'metadata': dict}"""
    
    name = "calculate_water_mass"
    description = "计算容器中水的质量

Parameters:
-----------
base_area : float
    容器底面积 (m^2)
height : float
    水位高度 (m)
density : float
    水的密度 (kg/m^3), 默认1000

Returns:
--------
dict : {'result': float, 'metadata': dict}"
    arguments = {
        "base_area": {"type": "string", "description": "参数 base_area"},
        "height": {"type": "string", "description": "参数 height"},
        "density": {"type": "string", "description": "参数 density"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_water_mass 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import calculate_water_mass
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["base_area", "height", "density"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_water_mass(**func_kwargs)
            
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


@Toolbox.register(name="calculate_tilt_angle")
class CalculateTiltAngleTool(EnvironmentTool):
    """计算液体表面相对于水平面的倾斜角度

物理原理：
在加速参考系中，液体受到惯性力和重力的合力
液面垂直于合力方向
tan(θ) = a/g

Parameters:
-----------
acceleration : float
    容器加速度 (m/s^2)
g : float
    重力加速度 (m/s^2)

Returns:
--------
dict : {'result': float, 'metadata': dict}
    倾斜角度(弧度)及相关信息"""
    
    name = "calculate_tilt_angle"
    description = "计算液体表面相对于水平面的倾斜角度

物理原理：
在加速参考系中，液体受到惯性力和重力的合力
液面垂直于合力方向
tan(θ) = a/g

Parameters:
-----------
acceleration : float
    容器加速度 (m/s^2)
g : float
    重力加速度 (m/s^2)

Returns:
--------
dict : {'result': float, 'metadata': dict}
    倾斜角度(弧度)及相关信息"
    arguments = {
        "acceleration": {"type": "string", "description": "参数 acceleration"},
        "g": {"type": "string", "description": "参数 g"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_tilt_angle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import calculate_tilt_angle
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["acceleration", "g"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_tilt_angle(**func_kwargs)
            
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


@Toolbox.register(name="calculate_height_difference")
class CalculateHeightDifferenceTool(EnvironmentTool):
    """计算液面两端的高度差

几何关系：Δh = (a/2) * tan(θ)
其中a是容器宽度，θ是倾斜角

Parameters:
-----------
base_width : float
    容器宽度 (m)
tilt_angle : float
    倾斜角度 (弧度)

Returns:
--------
dict : {'result': float, 'metadata': dict}
    高度差(m)"""
    
    name = "calculate_height_difference"
    description = "计算液面两端的高度差

几何关系：Δh = (a/2) * tan(θ)
其中a是容器宽度，θ是倾斜角

Parameters:
-----------
base_width : float
    容器宽度 (m)
tilt_angle : float
    倾斜角度 (弧度)

Returns:
--------
dict : {'result': float, 'metadata': dict}
    高度差(m)"
    arguments = {
        "base_width": {"type": "string", "description": "参数 base_width"},
        "tilt_angle": {"type": "string", "description": "参数 tilt_angle"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_height_difference 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import calculate_height_difference
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["base_width", "tilt_angle"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_height_difference(**func_kwargs)
            
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


@Toolbox.register(name="calculate_minimum_height")
class CalculateMinimumHeightTool(EnvironmentTool):
    """计算防止溢出的最小容器高度

临界条件：H = h + Δh
其中h是初始水位，Δh是液面高度差

Parameters:
-----------
initial_height : float
    初始水位高度 (m)
height_difference : float
    液面高度差 (m)

Returns:
--------
dict : {'result': float, 'metadata': dict}"""
    
    name = "calculate_minimum_height"
    description = "计算防止溢出的最小容器高度

临界条件：H = h + Δh
其中h是初始水位，Δh是液面高度差

Parameters:
-----------
initial_height : float
    初始水位高度 (m)
height_difference : float
    液面高度差 (m)

Returns:
--------
dict : {'result': float, 'metadata': dict}"
    arguments = {
        "initial_height": {"type": "string", "description": "参数 initial_height"},
        "height_difference": {"type": "string", "description": "参数 height_difference"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_minimum_height 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import calculate_minimum_height
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["initial_height", "height_difference"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_minimum_height(**func_kwargs)
            
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


@Toolbox.register(name="solve_minimum_container_height")
class SolveMinimumContainerHeightTool(EnvironmentTool):
    """求解防止水溢出的最小容器高度（完整流程）

Parameters:
-----------
container_mass : float
    容器质量 (kg)
hanging_mass : float
    悬挂重物质量 (kg)
friction_coef : float
    摩擦系数
base_width : float
    容器底边长 (m)
initial_water_height : float
    初始水位高度 (m)
water_density : float
    水的密度 (kg/m^3)

Returns:
--------
dict : 包含最终结果和所有中间步骤"""
    
    name = "solve_minimum_container_height"
    description = "求解防止水溢出的最小容器高度（完整流程）

Parameters:
-----------
container_mass : float
    容器质量 (kg)
hanging_mass : float
    悬挂重物质量 (kg)
friction_coef : float
    摩擦系数
base_width : float
    容器底边长 (m)
initial_water_height : float
    初始水位高度 (m)
water_density : float
    水的密度 (kg/m^3)

Returns:
--------
dict : 包含最终结果和所有中间步骤"
    arguments = {
        "container_mass": {"type": "string", "description": "参数 container_mass"},
        "hanging_mass": {"type": "string", "description": "参数 hanging_mass"},
        "friction_coef": {"type": "string", "description": "参数 friction_coef"},
        "base_width": {"type": "string", "description": "参数 base_width"},
        "initial_water_height": {"type": "string", "description": "参数 initial_water_height"},
        "water_density": {"type": "string", "description": "参数 water_density"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_minimum_container_height 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import solve_minimum_container_height
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["container_mass", "hanging_mass", "friction_coef", "base_width", "initial_water_height", "water_density"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve_minimum_container_height(**func_kwargs)
            
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


@Toolbox.register(name="analyze_parameter_sensitivity")
class AnalyzeParameterSensitivityTool(EnvironmentTool):
    """分析参数敏感性

Parameters:
-----------
base_params : dict
    基准参数字典，包含：
    - container_mass, hanging_mass, friction_coef, 
      base_width, initial_water_height
vary_param : str
    要变化的参数名称
param_range : list
    参数变化范围

Returns:
--------
dict : 敏感性分析结果"""
    
    name = "analyze_parameter_sensitivity"
    description = "分析参数敏感性

Parameters:
-----------
base_params : dict
    基准参数字典，包含：
    - container_mass, hanging_mass, friction_coef, 
      base_width, initial_water_height
vary_param : str
    要变化的参数名称
param_range : list
    参数变化范围

Returns:
--------
dict : 敏感性分析结果"
    arguments = {
        "base_params": {"type": "string", "description": "参数 base_params"},
        "vary_param": {"type": "string", "description": "参数 vary_param"},
        "param_range": {"type": "string", "description": "参数 param_range"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_parameter_sensitivity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import analyze_parameter_sensitivity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["base_params", "vary_param", "param_range"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_parameter_sensitivity(**func_kwargs)
            
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


@Toolbox.register(name="visualize_liquid_surface")
class VisualizeLiquidSurfaceTool(EnvironmentTool):
    """可视化液体表面倾斜情况

Parameters:
-----------
base_width : float
    容器宽度 (m)
initial_height : float
    初始水位 (m)
tilt_angle : float
    倾斜角度 (弧度)
min_height : float
    最小容器高度 (m)
save_path : str
    保存路径

Returns:
--------
dict : 图像信息"""
    
    name = "visualize_liquid_surface"
    description = "可视化液体表面倾斜情况

Parameters:
-----------
base_width : float
    容器宽度 (m)
initial_height : float
    初始水位 (m)
tilt_angle : float
    倾斜角度 (弧度)
min_height : float
    最小容器高度 (m)
save_path : str
    保存路径

Returns:
--------
dict : 图像信息"
    arguments = {
        "base_width": {"type": "string", "description": "参数 base_width"},
        "initial_height": {"type": "string", "description": "参数 initial_height"},
        "tilt_angle": {"type": "string", "description": "参数 tilt_angle"},
        "min_height": {"type": "string", "description": "参数 min_height"},
        "save_path": {"type": "string", "description": "参数 save_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_liquid_surface 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import visualize_liquid_surface
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["base_width", "initial_height", "tilt_angle", "min_height", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_liquid_surface(**func_kwargs)
            
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


@Toolbox.register(name="plot_sensitivity_analysis")
class PlotSensitivityAnalysisTool(EnvironmentTool):
    """绘制参数敏感性分析图

Parameters:
-----------
sensitivity_data : list
    敏感性数据，格式：[{'param_value': x, 'min_height': y}, ...]
param_name : str
    参数名称
param_unit : str
    参数单位
save_path : str
    保存路径

Returns:
--------
dict : 图像信息"""
    
    name = "plot_sensitivity_analysis"
    description = "绘制参数敏感性分析图

Parameters:
-----------
sensitivity_data : list
    敏感性数据，格式：[{'param_value': x, 'min_height': y}, ...]
param_name : str
    参数名称
param_unit : str
    参数单位
save_path : str
    保存路径

Returns:
--------
dict : 图像信息"
    arguments = {
        "sensitivity_data": {"type": "string", "description": "参数 sensitivity_data"},
        "param_name": {"type": "string", "description": "参数 param_name"},
        "param_unit": {"type": "string", "description": "参数 param_unit"},
        "save_path": {"type": "string", "description": "参数 save_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_sensitivity_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import plot_sensitivity_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["sensitivity_data", "param_name", "param_unit", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_sensitivity_analysis(**func_kwargs)
            
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


@Toolbox.register(name="create_force_diagram")
class CreateForceDiagramTool(EnvironmentTool):
    """绘制系统受力分析图

Parameters:
-----------
container_mass : float
    容器质量 (kg)
water_mass : float
    水的质量 (kg)
hanging_mass : float
    悬挂物质量 (kg)
friction_force : float
    摩擦力 (N)
tension : float
    绳子张力 (N)
acceleration : float
    加速度 (m/s^2)
save_path : str
    保存路径

Returns:
--------
dict : 图像信息"""
    
    name = "create_force_diagram"
    description = "绘制系统受力分析图

Parameters:
-----------
container_mass : float
    容器质量 (kg)
water_mass : float
    水的质量 (kg)
hanging_mass : float
    悬挂物质量 (kg)
friction_force : float
    摩擦力 (N)
tension : float
    绳子张力 (N)
acceleration : float
    加速度 (m/s^2)
save_path : str
    保存路径

Returns:
--------
dict : 图像信息"
    arguments = {
        "container_mass": {"type": "string", "description": "参数 container_mass"},
        "water_mass": {"type": "string", "description": "参数 water_mass"},
        "hanging_mass": {"type": "string", "description": "参数 hanging_mass"},
        "friction_force": {"type": "string", "description": "参数 friction_force"},
        "tension": {"type": "string", "description": "参数 tension"},
        "acceleration": {"type": "string", "description": "参数 acceleration"},
        "save_path": {"type": "string", "description": "参数 save_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_force_diagram 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.fluid_mechanics_toolkit_claude_4 import create_force_diagram
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["container_mass", "water_mass", "hanging_mass", "friction_force", "tension", "acceleration", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_force_diagram(**func_kwargs)
            
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


@Toolbox.register(name="calculate_total_variation_coefficient")
class CalculateTotalVariationCoefficientTool(EnvironmentTool):
    """计算污水总变化系数 Kz

根据《室外排水设计标准》GB 50014-2021，总变化系数用于将平均流量转换为设计流量。

Parameters:
-----------
average_flow : float
    平均流量 (L/s)
coefficient_type : str
    系数类型，可选 "domestic"(生活污水) 或 "industrial"(工业废水)

Returns:
--------
dict : {
    'result': float,  # 总变化系数 Kz
    'metadata': {
        'formula': str,  # 使用的公式
        'flow_range': str,  # 流量范围
        'coefficient_type': str
    }
}

公式依据：
- 生活污水：Kz = 2.3 / Q^0.07 (Q < 20 L/s)
- 生活污水：Kz = 1.98 / Q^0.04 (20 ≤ Q ≤ 1000 L/s)
- 工业废水：Kz = 1.3 ~ 1.5 (相对稳定)"""
    
    name = "calculate_total_variation_coefficient"
    description = "计算污水总变化系数 Kz

根据《室外排水设计标准》GB 50014-2021，总变化系数用于将平均流量转换为设计流量。

Parameters:
-----------
average_flow : float
    平均流量 (L/s)
coefficient_type : str
    系数类型，可选 "domestic"(生活污水) 或 "industrial"(工业废水)

Returns:
--------
dict : {
    'result': float,  # 总变化系数 Kz
    'metadata': {
        'formula': str,  # 使用的公式
        'flow_range': str,  # 流量范围
        'coefficient_type': str
    }
}

公式依据：
- 生活污水：Kz = 2.3 / Q^0.07 (Q < 20 L/s)
- 生活污水：Kz = 1.98 / Q^0.04 (20 ≤ Q ≤ 1000 L/s)
- 工业废水：Kz = 1.3 ~ 1.5 (相对稳定)"
    arguments = {
        "average_flow": {"type": "string", "description": "参数 average_flow"},
        "coefficient_type": {"type": "string", "description": "参数 coefficient_type"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_total_variation_coefficient 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import calculate_total_variation_coefficient
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["average_flow", "coefficient_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_total_variation_coefficient(**func_kwargs)
            
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


@Toolbox.register(name="calculate_design_flow")
class CalculateDesignFlowTool(EnvironmentTool):
    """计算设计流量

Parameters:
-----------
average_flow : float
    平均流量 (L/s)
total_variation_coefficient : float
    总变化系数 Kz

Returns:
--------
dict : {
    'result': float,  # 设计流量 (L/s)
    'metadata': {
        'formula': str,
        'average_flow': float,
        'kz': float
    }
}

公式：Qd = Kz * Qavg"""
    
    name = "calculate_design_flow"
    description = "计算设计流量

Parameters:
-----------
average_flow : float
    平均流量 (L/s)
total_variation_coefficient : float
    总变化系数 Kz

Returns:
--------
dict : {
    'result': float,  # 设计流量 (L/s)
    'metadata': {
        'formula': str,
        'average_flow': float,
        'kz': float
    }
}

公式：Qd = Kz * Qavg"
    arguments = {
        "average_flow": {"type": "string", "description": "参数 average_flow"},
        "total_variation_coefficient": {"type": "string", "description": "参数 total_variation_coefficient"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_design_flow 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import calculate_design_flow
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["average_flow", "total_variation_coefficient"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_design_flow(**func_kwargs)
            
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


@Toolbox.register(name="calculate_concentrated_flow_coefficient")
class CalculateConcentratedFlowCoefficientTool(EnvironmentTool):
    """计算集中流量的变化系数

Parameters:
-----------
flow_rate : float
    集中流量 (L/s)
source_type : str
    来源类型，可选 "factory"(工厂), "hospital"(医院), "school"(学校)

Returns:
--------
dict : {
    'result': float,  # 集中流量变化系数
    'metadata': {
        'source_type': str,
        'coefficient_range': str
    }
}"""
    
    name = "calculate_concentrated_flow_coefficient"
    description = "计算集中流量的变化系数

Parameters:
-----------
flow_rate : float
    集中流量 (L/s)
source_type : str
    来源类型，可选 "factory"(工厂), "hospital"(医院), "school"(学校)

Returns:
--------
dict : {
    'result': float,  # 集中流量变化系数
    'metadata': {
        'source_type': str,
        'coefficient_range': str
    }
}"
    arguments = {
        "flow_rate": {"type": "string", "description": "参数 flow_rate"},
        "source_type": {"type": "string", "description": "参数 source_type"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_concentrated_flow_coefficient 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import calculate_concentrated_flow_coefficient
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["flow_rate", "source_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_concentrated_flow_coefficient(**func_kwargs)
            
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


@Toolbox.register(name="sum_flows_at_node")
class SumFlowsAtNodeTool(EnvironmentTool):
    """计算节点处的流量汇总

Parameters:
-----------
upstream_flows : List[float]
    上游各管段流入的流量列表 (L/s)
concentrated_flow : float
    节点处的集中流量 (L/s)，默认为0

Returns:
--------
dict : {
    'result': float,  # 节点总流量 (L/s)
    'metadata': {
        'upstream_count': int,
        'upstream_total': float,
        'concentrated_flow': float
    }
}"""
    
    name = "sum_flows_at_node"
    description = "计算节点处的流量汇总

Parameters:
-----------
upstream_flows : List[float]
    上游各管段流入的流量列表 (L/s)
concentrated_flow : float
    节点处的集中流量 (L/s)，默认为0

Returns:
--------
dict : {
    'result': float,  # 节点总流量 (L/s)
    'metadata': {
        'upstream_count': int,
        'upstream_total': float,
        'concentrated_flow': float
    }
}"
    arguments = {
        "upstream_flows": {"type": "string", "description": "参数 upstream_flows"},
        "concentrated_flow": {"type": "string", "description": "参数 concentrated_flow"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 sum_flows_at_node 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import sum_flows_at_node
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["upstream_flows", "concentrated_flow"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = sum_flows_at_node(**func_kwargs)
            
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


@Toolbox.register(name="calculate_pipe_velocity")
class CalculatePipeVelocityTool(EnvironmentTool):
    """计算管道流速

Parameters:
-----------
flow_rate : float
    流量 (L/s)
diameter : float
    管道直径 (mm)

Returns:
--------
dict : {
    'result': float,  # 流速 (m/s)
    'metadata': {
        'flow_rate': float,
        'diameter': float,
        'area': float,  # 管道截面积 (m²)
        'velocity_check': str  # 流速是否满足规范
    }
}

规范要求：污水管道流速应在 0.6 ~ 3.0 m/s 之间"""
    
    name = "calculate_pipe_velocity"
    description = "计算管道流速

Parameters:
-----------
flow_rate : float
    流量 (L/s)
diameter : float
    管道直径 (mm)

Returns:
--------
dict : {
    'result': float,  # 流速 (m/s)
    'metadata': {
        'flow_rate': float,
        'diameter': float,
        'area': float,  # 管道截面积 (m²)
        'velocity_check': str  # 流速是否满足规范
    }
}

规范要求：污水管道流速应在 0.6 ~ 3.0 m/s 之间"
    arguments = {
        "flow_rate": {"type": "string", "description": "参数 flow_rate"},
        "diameter": {"type": "string", "description": "参数 diameter"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_pipe_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import calculate_pipe_velocity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["flow_rate", "diameter"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_pipe_velocity(**func_kwargs)
            
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


@Toolbox.register(name="calculate_section_design_flow")
class CalculateSectionDesignFlowTool(EnvironmentTool):
    """计算管段设计流量（综合考虑本段生活污水、上游来水、集中流量）

Parameters:
-----------
section_domestic_flow : float
    本管段生活污水平均流量 (L/s)
upstream_design_flow : float
    上游管段设计流量 (L/s)，默认为0
concentrated_flow : float
    集中流量（如工厂废水）(L/s)，默认为0
concentrated_source_type : str
    集中流量来源类型

Returns:
--------
dict : {
    'result': float,  # 管段总设计流量 (L/s)
    'metadata': {
        'section_domestic_design': float,
        'upstream_design': float,
        'concentrated_design': float,
        'calculation_steps': list
    }
}"""
    
    name = "calculate_section_design_flow"
    description = "计算管段设计流量（综合考虑本段生活污水、上游来水、集中流量）

Parameters:
-----------
section_domestic_flow : float
    本管段生活污水平均流量 (L/s)
upstream_design_flow : float
    上游管段设计流量 (L/s)，默认为0
concentrated_flow : float
    集中流量（如工厂废水）(L/s)，默认为0
concentrated_source_type : str
    集中流量来源类型

Returns:
--------
dict : {
    'result': float,  # 管段总设计流量 (L/s)
    'metadata': {
        'section_domestic_design': float,
        'upstream_design': float,
        'concentrated_design': float,
        'calculation_steps': list
    }
}"
    arguments = {
        "section_domestic_flow": {"type": "string", "description": "参数 section_domestic_flow"},
        "upstream_design_flow": {"type": "string", "description": "参数 upstream_design_flow"},
        "concentrated_flow": {"type": "string", "description": "参数 concentrated_flow"},
        "concentrated_source_type": {"type": "string", "description": "参数 concentrated_source_type"},
        "infiltration_ratio": {"type": "string", "description": "参数 infiltration_ratio"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_section_design_flow 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import calculate_section_design_flow
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["section_domestic_flow", "upstream_design_flow", "concentrated_flow", "concentrated_source_type", "infiltration_ratio"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_section_design_flow(**func_kwargs)
            
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


@Toolbox.register(name="design_pipe_diameter")
class DesignPipeDiameterTool(EnvironmentTool):
    """根据设计流量和目标流速选择标准管径

Parameters:
-----------
design_flow : float
    设计流量 (L/s)
target_velocity : float
    目标流速 (m/s)，默认1.0
standard_diameters : List[int]
    标准管径列表 (mm)，默认使用常用管径

Returns:
--------
dict : {
    'result': int,  # 推荐管径 (mm)
    'metadata': {
        'calculated_diameter': float,
        'actual_velocity': float,
        'velocity_check': str
    }
}"""
    
    name = "design_pipe_diameter"
    description = "根据设计流量和目标流速选择标准管径

Parameters:
-----------
design_flow : float
    设计流量 (L/s)
target_velocity : float
    目标流速 (m/s)，默认1.0
standard_diameters : List[int]
    标准管径列表 (mm)，默认使用常用管径

Returns:
--------
dict : {
    'result': int,  # 推荐管径 (mm)
    'metadata': {
        'calculated_diameter': float,
        'actual_velocity': float,
        'velocity_check': str
    }
}"
    arguments = {
        "design_flow": {"type": "string", "description": "参数 design_flow"},
        "target_velocity": {"type": "string", "description": "参数 target_velocity"},
        "standard_diameters": {"type": "string", "description": "参数 standard_diameters"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 design_pipe_diameter 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import design_pipe_diameter
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["design_flow", "target_velocity", "standard_diameters"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = design_pipe_diameter(**func_kwargs)
            
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


@Toolbox.register(name="analyze_pipeline_network")
class AnalyzePipelineNetworkTool(EnvironmentTool):
    """分析整个管网系统的流量分布

Parameters:
-----------
sections : List[Dict]
    管段信息列表，每个管段包含：
    {
        'name': str,  # 管段名称，如 "1-2"
        'domestic_flow': float,  # 本段生活污水流量
        'upstream_sections': List[str],  # 上游管段名称列表
        'concentrated_flow': float,  # 集中流量（可选）
        'concentrated_type': str  # 集中流量类型（可选）
    }

Returns:
--------
dict : {
    'result': dict,  # {section_name: design_flow}
    'metadata': {
        'total_sections': int,
        'max_flow_section': str,
        'calculation_order': List[str]
    }
}"""
    
    name = "analyze_pipeline_network"
    description = "分析整个管网系统的流量分布

Parameters:
-----------
sections : List[Dict]
    管段信息列表，每个管段包含：
    {
        'name': str,  # 管段名称，如 "1-2"
        'domestic_flow': float,  # 本段生活污水流量
        'upstream_sections': List[str],  # 上游管段名称列表
        'concentrated_flow': float,  # 集中流量（可选）
        'concentrated_type': str  # 集中流量类型（可选）
    }

Returns:
--------
dict : {
    'result': dict,  # {section_name: design_flow}
    'metadata': {
        'total_sections': int,
        'max_flow_section': str,
        'calculation_order': List[str]
    }
}"
    arguments = {
        "sections": {"type": "string", "description": "参数 sections"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_pipeline_network 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import analyze_pipeline_network
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["sections"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_pipeline_network(**func_kwargs)
            
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


@Toolbox.register(name="visualize_flow_distribution")
class VisualizeFlowDistributionTool(EnvironmentTool):
    """可视化管网流量分布

Parameters:
-----------
section_flows : Dict[str, float]
    管段流量字典 {section_name: flow_value}
output_path : str
    输出图像路径

Returns:
--------
dict : {
    'result': str,  # 图像文件路径
    'metadata': {
        'file_type': str,
        'sections_count': int
    }
}"""
    
    name = "visualize_flow_distribution"
    description = "可视化管网流量分布

Parameters:
-----------
section_flows : Dict[str, float]
    管段流量字典 {section_name: flow_value}
output_path : str
    输出图像路径

Returns:
--------
dict : {
    'result': str,  # 图像文件路径
    'metadata': {
        'file_type': str,
        'sections_count': int
    }
}"
    arguments = {
        "section_flows": {"type": "string", "description": "参数 section_flows"},
        "output_path": {"type": "string", "description": "参数 output_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_flow_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import visualize_flow_distribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["section_flows", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_flow_distribution(**func_kwargs)
            
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


@Toolbox.register(name="visualize_network_schematic")
class VisualizeNetworkSchematicTool(EnvironmentTool):
    """绘制管网系统示意图

Parameters:
-----------
sections : List[Dict]
    管段信息列表
section_flows : Dict[str, float]
    管段流量字典
output_path : str
    输出图像路径

Returns:
--------
dict : {
    'result': str,  # 图像文件路径
    'metadata': {
        'file_type': str,
        'nodes_count': int
    }
}"""
    
    name = "visualize_network_schematic"
    description = "绘制管网系统示意图

Parameters:
-----------
sections : List[Dict]
    管段信息列表
section_flows : Dict[str, float]
    管段流量字典
output_path : str
    输出图像路径

Returns:
--------
dict : {
    'result': str,  # 图像文件路径
    'metadata': {
        'file_type': str,
        'nodes_count': int
    }
}"
    arguments = {
        "sections": {"type": "string", "description": "参数 sections"},
        "section_flows": {"type": "string", "description": "参数 section_flows"},
        "output_path": {"type": "string", "description": "参数 output_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_network_schematic 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import visualize_network_schematic
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["sections", "section_flows", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_network_schematic(**func_kwargs)
            
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


@Toolbox.register(name="generate_calculation_report")
class GenerateCalculationReportTool(EnvironmentTool):
    """生成详细的计算报告

Parameters:
-----------
problem_description : str
    问题描述
calculation_results : Dict
    计算结果字典
output_path : str
    输出文件路径

Returns:
--------
dict : {
    'result': str,  # 报告文件路径
    'metadata': {
        'file_type': str,
        'timestamp': str
    }
}"""
    
    name = "generate_calculation_report"
    description = "生成详细的计算报告

Parameters:
-----------
problem_description : str
    问题描述
calculation_results : Dict
    计算结果字典
output_path : str
    输出文件路径

Returns:
--------
dict : {
    'result': str,  # 报告文件路径
    'metadata': {
        'file_type': str,
        'timestamp': str
    }
}"
    arguments = {
        "problem_description": {"type": "string", "description": "参数 problem_description"},
        "calculation_results": {"type": "string", "description": "参数 calculation_results"},
        "output_path": {"type": "string", "description": "参数 output_path"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 generate_calculation_report 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import generate_calculation_report
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["problem_description", "calculation_results", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = generate_calculation_report(**func_kwargs)
            
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


@Toolbox.register(name="calculate_section")
class CalculateSectionTool(EnvironmentTool):
    """"""
    
    name = "calculate_section"
    description = ""
    arguments = {
        "section_info": {"type": "string", "description": "参数 section_info"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_section 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.wastewater_engineering_toolkit_claude_649 import calculate_section
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["section_info"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_section(**func_kwargs)
            
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


@Toolbox.register(name="captive_bubble_to_sessile_drop")
class CaptiveBubbleToSessileDropTool(EnvironmentTool):
    """将captive bubble接触角转换为sessile drop接触角。理论基础：在captive bubble法中，气泡在液体中附着在固体表面下方；在sessile drop法中，液滴在空气中附着在固体表面上方；两者的接触角互补：θ_captive + θ_sessile = 180°。对于疏水表面（captive角度较小），实际测量显示平衡角可能略大于理论值，需要根据接触角大小进行修正。"""
    
    name = "captive_bubble_to_sessile_drop"
    description = "将captive bubble接触角转换为sessile drop接触角。理论基础：在captive bubble法中，气泡在液体中附着在固体表面下方；在sessile drop法中，液滴在空气中附着在固体表面上方；两者的接触角互补：θ_captive + θ_sessile = 180°。对于疏水表面（captive角度较小），实际测量显示平衡角可能略大于理论值，需要根据接触角大小进行修正。"
    arguments = {
        "captive_angle": {"type": "number", "description": "captive bubble测量的接触角，单位：度，范围0-180"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 captive_bubble_to_sessile_drop 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "captive_angle" not in args or args["captive_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 captive_angle")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import captive_bubble_to_sessile_drop
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["captive_angle"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = captive_bubble_to_sessile_drop(**func_kwargs)
            
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


@Toolbox.register(name="calculate_contact_angle_hysteresis")
class CalculateContactAngleHysteresisTool(EnvironmentTool):
    """基于表面能和粗糙度计算接触角滞后。理论模型：Wenzel模型和经验关系，接触角滞后Δθ = θ_adv - θ_rec。"""
    
    name = "calculate_contact_angle_hysteresis"
    description = "基于表面能和粗糙度计算接触角滞后。理论模型：Wenzel模型和经验关系，接触角滞后Δθ = θ_adv - θ_rec。"
    arguments = {
        "surface_energy": {"type": "number", "description": "固体表面能，单位：mN/m，必须为正值"},
        "roughness_factor": {"type": "number", "description": "粗糙度因子，必须 >= 1.0，默认为1.0"},
        "liquid_surface_tension": {"type": "number", "description": "液体表面张力，单位：mN/m，默认为72.8（水在20°C）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_contact_angle_hysteresis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "surface_energy" not in args or args["surface_energy"] is None:
                return Observation(self.name, "错误: 缺少必需参数 surface_energy")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import calculate_contact_angle_hysteresis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["surface_energy", "roughness_factor", "liquid_surface_tension"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_contact_angle_hysteresis(**func_kwargs)
            
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


@Toolbox.register(name="estimate_advancing_receding_angles")
class EstimateAdvancingRecedingAnglesTool(EnvironmentTool):
    """基于平衡接触角和滞后估算前进角和后退角。理论：θ_equilibrium ≈ (θ_adv + θ_rec) / 2，Δθ = θ_adv - θ_rec。"""
    
    name = "estimate_advancing_receding_angles"
    description = "基于平衡接触角和滞后估算前进角和后退角。理论：θ_equilibrium ≈ (θ_adv + θ_rec) / 2，Δθ = θ_adv - θ_rec。"
    arguments = {
        "equilibrium_angle": {"type": "number", "description": "平衡接触角，单位：度，范围0-180"},
        "hysteresis": {"type": "number", "description": "接触角滞后，单位：度，必须为正值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 estimate_advancing_receding_angles 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "equilibrium_angle" not in args or args["equilibrium_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 equilibrium_angle")
            if "hysteresis" not in args or args["hysteresis"] is None:
                return Observation(self.name, "错误: 缺少必需参数 hysteresis")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import estimate_advancing_receding_angles
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["equilibrium_angle", "hysteresis"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = estimate_advancing_receding_angles(**func_kwargs)
            
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


@Toolbox.register(name="query_surface_properties")
class QuerySurfacePropertiesTool(EnvironmentTool):
    """从数据库查询固体表面性质，包括表面能、极性分量、色散分量和粗糙度因子。"""
    
    name = "query_surface_properties"
    description = "从数据库查询固体表面性质，包括表面能、极性分量、色散分量和粗糙度因子。"
    arguments = {
        "db_path": {"type": "string", "description": "数据库文件路径"},
        "material_name": {"type": "string", "description": "材料名称，如'PTFE'、'glass'、'silicon'等"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 query_surface_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "db_path" not in args or args["db_path"] is None:
                return Observation(self.name, "错误: 缺少必需参数 db_path")
            if "material_name" not in args or args["material_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 material_name")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import query_surface_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["db_path", "material_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = query_surface_properties(**func_kwargs)
            
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


@Toolbox.register(name="estimate_surface_properties_from_equilibrium_angle")
class EstimateSurfacePropertiesFromEquilibriumAngleTool(EnvironmentTool):
    """从平衡接触角估计表面性质和粗糙度。工作流程：1.从平衡角估算表面能（使用Young's方程）；2.如果目标滞后值已知，从滞后值反推粗糙度；3.如果目标滞后值未知，使用默认粗糙度估算。"""
    
    name = "estimate_surface_properties_from_equilibrium_angle"
    description = "从平衡接触角估计表面性质和粗糙度。工作流程：1.从平衡角估算表面能（使用Young's方程）；2.如果目标滞后值已知，从滞后值反推粗糙度；3.如果目标滞后值未知，使用默认粗糙度估算。"
    arguments = {
        "equilibrium_angle": {"type": "number", "description": "平衡接触角，单位：度，范围0-180"},
        "target_hysteresis": {"type": "number", "description": "目标滞后值，单位：度，如果为None则从表面性质估算"},
        "liquid_surface_tension": {"type": "number", "description": "液体表面张力，单位：mN/m，默认为72.8（水）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 estimate_surface_properties_from_equilibrium_angle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "equilibrium_angle" not in args or args["equilibrium_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 equilibrium_angle")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import estimate_surface_properties_from_equilibrium_angle
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["equilibrium_angle", "target_hysteresis", "liquid_surface_tension"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = estimate_surface_properties_from_equilibrium_angle(**func_kwargs)
            
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


@Toolbox.register(name="analyze_wetting_behavior")
class AnalyzeWettingBehaviorTool(EnvironmentTool):
    """分析润湿行为类型。分类标准：θ < 90°为亲水（hydrophilic），90° < θ < 150°为疏水（hydrophobic），θ > 150°为超疏水（superhydrophobic）。"""
    
    name = "analyze_wetting_behavior"
    description = "分析润湿行为类型。分类标准：θ < 90°为亲水（hydrophilic），90° < θ < 150°为疏水（hydrophobic），θ > 150°为超疏水（superhydrophobic）。"
    arguments = {
        "contact_angle": {"type": "number", "description": "接触角，单位：度"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_wetting_behavior 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "contact_angle" not in args or args["contact_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 contact_angle")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import analyze_wetting_behavior
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["contact_angle"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_wetting_behavior(**func_kwargs)
            
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


@Toolbox.register(name="calculate_surface_energy_from_contact_angles")
class CalculateSurfaceEnergyFromContactAnglesTool(EnvironmentTool):
    """基于前进角和后退角反推表面能（使用Owens-Wendt方法）。使用Young's equation简化形式计算。"""
    
    name = "calculate_surface_energy_from_contact_angles"
    description = "基于前进角和后退角反推表面能（使用Owens-Wendt方法）。使用Young's equation简化形式计算。"
    arguments = {
        "advancing_angle": {"type": "number", "description": "前进角，单位：度"},
        "receding_angle": {"type": "number", "description": "后退角，单位：度"},
        "liquid_surface_tension": {"type": "number", "description": "液体表面张力，单位：mN/m，默认为72.8（水）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_surface_energy_from_contact_angles 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "advancing_angle" not in args or args["advancing_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 advancing_angle")
            if "receding_angle" not in args or args["receding_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 receding_angle")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import calculate_surface_energy_from_contact_angles
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["advancing_angle", "receding_angle", "liquid_surface_tension"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_surface_energy_from_contact_angles(**func_kwargs)
            
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


@Toolbox.register(name="plot_contact_angle_diagram")
class PlotContactAngleDiagramTool(EnvironmentTool):
    """绘制接触角对比图，包含captive bubble、前进角和后退角三种状态的可视化。"""
    
    name = "plot_contact_angle_diagram"
    description = "绘制接触角对比图，包含captive bubble、前进角和后退角三种状态的可视化。"
    arguments = {
        "captive_angle": {"type": "number", "description": "captive bubble角度，单位：度"},
        "advancing_angle": {"type": "number", "description": "前进角，单位：度"},
        "receding_angle": {"type": "number", "description": "后退角，单位：度"},
        "output_path": {"type": "string", "description": "输出图像路径，默认为'./tool_images/contact_angle_comparison.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_contact_angle_diagram 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "captive_angle" not in args or args["captive_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 captive_angle")
            if "advancing_angle" not in args or args["advancing_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 advancing_angle")
            if "receding_angle" not in args or args["receding_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 receding_angle")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import plot_contact_angle_diagram
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["captive_angle", "advancing_angle", "receding_angle", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_contact_angle_diagram(**func_kwargs)
            
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


@Toolbox.register(name="plot_hysteresis_analysis")
class PlotHysteresisAnalysisTool(EnvironmentTool):
    """绘制接触角滞后分析图，包含接触角分布和能量势垒模型。"""
    
    name = "plot_hysteresis_analysis"
    description = "绘制接触角滞后分析图，包含接触角分布和能量势垒模型。"
    arguments = {
        "equilibrium_angle": {"type": "number", "description": "平衡角，单位：度"},
        "hysteresis": {"type": "number", "description": "滞后值，单位：度"},
        "advancing_angle": {"type": "number", "description": "前进角，单位：度"},
        "receding_angle": {"type": "number", "description": "后退角，单位：度"},
        "output_path": {"type": "string", "description": "输出图像路径，默认为'./tool_images/hysteresis_analysis.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_hysteresis_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "equilibrium_angle" not in args or args["equilibrium_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 equilibrium_angle")
            if "hysteresis" not in args or args["hysteresis"] is None:
                return Observation(self.name, "错误: 缺少必需参数 hysteresis")
            if "advancing_angle" not in args or args["advancing_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 advancing_angle")
            if "receding_angle" not in args or args["receding_angle"] is None:
                return Observation(self.name, "错误: 缺少必需参数 receding_angle")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import plot_hysteresis_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["equilibrium_angle", "hysteresis", "advancing_angle", "receding_angle", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_hysteresis_analysis(**func_kwargs)
            
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


@Toolbox.register(name="plot_wetting_phase_diagram")
class PlotWettingPhaseDiagramTool(EnvironmentTool):
    """绘制润湿相图，展示接触角与表面能的关系，标注亲水和疏水区域。"""
    
    name = "plot_wetting_phase_diagram"
    description = "绘制润湿相图，展示接触角与表面能的关系，标注亲水和疏水区域。"
    arguments = {
        "contact_angles": {"type": "array", "description": "接触角列表，单位：度"},
        "surface_energies": {"type": "array", "description": "对应的表面能列表，单位：mN/m"},
        "output_path": {"type": "string", "description": "输出图像路径，默认为'./tool_images/wetting_phase_diagram.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_wetting_phase_diagram 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "contact_angles" not in args or args["contact_angles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 contact_angles")
            if "surface_energies" not in args or args["surface_energies"] is None:
                return Observation(self.name, "错误: 缺少必需参数 surface_energies")
            
            # 导入并调用原始函数
            from toolkits.physics.fluid_dynamics.surface_wetting_toolkit_claude_43 import plot_wetting_phase_diagram
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["contact_angles", "surface_energies", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_wetting_phase_diagram(**func_kwargs)
            
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

