#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
thermodynamics 工具注册模块
使用 gym.tool.EnvironmentTool 为 thermodynamics 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.physics.thermodynamics.thermodynamic_solver_50 import *  # 动态导入
# from toolkits.physics.thermodynamics.thermodynamics_solver_215 import *  # 动态导入
# from toolkits.physics.thermodynamics.thermodynamics_solver_8030 import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="solve_thermodynamic_cycle")
class SolveThermodynamicCycleTool(EnvironmentTool):
    """求解由两条等容线和两条等压线构成的循环过程中状态b的温度"""
    
    name = "solve_thermodynamic_cycle"
    description = "求解由两条等容线和两条等压线构成的循环过程中状态b的温度"
    arguments = {
        "T1": {"type": "number", "description": "状态a的温度，单位：K"},
        "T3": {"type": "number", "description": "状态c的温度，单位：K"},
        "n": {"type": "number", "description": "物质的量，默认为1 mol"},
        "R": {"type": "number", "description": "气体常数，默认为8.314 J/(mol·K)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_thermodynamic_cycle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            T1 = args.get("T1")
            if T1 is None:
                return Observation(self.name, "错误: 缺少必需参数 T1")
            T3 = args.get("T3")
            if T3 is None:
                return Observation(self.name, "错误: 缺少必需参数 T3")
            n = args.get("n", None)
            R = args.get("R", None)
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamics_solver_215 import solve_thermodynamic_cycle
            
            # 调用函数
            result = solve_thermodynamic_cycle(T1, T3, n, R)
            
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


@Toolbox.register(name="plot_thermodynamic_cycle")
class PlotThermodynamicCycleTool(EnvironmentTool):
    """绘制热力学循环的P-V图"""
    
    name = "plot_thermodynamic_cycle"
    description = "绘制热力学循环的P-V图"
    arguments = {
        "T1": {"type": "number", "description": ""},
        "T3": {"type": "number", "description": ""},
        "n": {"type": "number", "description": ""},
        "R": {"type": "number", "description": ""}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_thermodynamic_cycle 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            T1 = args.get("T1")
            if T1 is None:
                return Observation(self.name, "错误: 缺少必需参数 T1")
            T3 = args.get("T3")
            if T3 is None:
                return Observation(self.name, "错误: 缺少必需参数 T3")
            n = args.get("n", None)
            R = args.get("R", None)
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamics_solver_215 import plot_thermodynamic_cycle
            
            # 调用函数
            result = plot_thermodynamic_cycle(T1, T3, n, R)
            
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


@Toolbox.register(name="ideal_gas_expansion")
class IdealGasExpansionTool(EnvironmentTool):
    """计算理想气体绝热自由膨胀过程的最终状态和热力学参数。在绝热自由膨胀过程中，气体从初始体积膨胀到更大体积，系统不与外界交换热量(绝热)，且不对外做功(自由膨胀)。"""
    
    name = "ideal_gas_expansion"
    description = "计算理想气体绝热自由膨胀过程的最终状态和热力学参数。在绝热自由膨胀过程中，气体从初始体积膨胀到更大体积，系统不与外界交换热量(绝热)，且不对外做功(自由膨胀)。"
    arguments = {
        "p0": {"type": "number", "description": "初始压强，单位：Pa"},
        "V_ratio": {"type": "number", "description": "膨胀后体积与初始体积的比值，默认为2.0"},
        "gamma": {"type": "number", "description": "气体的绝热指数(Cp/Cv)，默认为1.4(双原子分子)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 ideal_gas_expansion 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            p0 = args.get("p0")
            if p0 is None:
                return Observation(self.name, "错误: 缺少必需参数 p0")
            V_ratio = args.get("V_ratio", None)
            gamma = args.get("gamma", None)
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamic_solver_50 import ideal_gas_expansion
            
            # 调用函数
            result = ideal_gas_expansion(p0, V_ratio, gamma)
            
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


@Toolbox.register(name="rankine_cycle_efficiency")
class RankineCycleEfficiencyTool(EnvironmentTool):
    """计算理想朗肯循环的热效率和关键状态点参数。朗肯循环是蒸汽动力装置的理想热力循环，包括等熵压缩、等压加热、等熵膨胀和等压冷凝四个过程。"""
    
    name = "rankine_cycle_efficiency"
    description = "计算理想朗肯循环的热效率和关键状态点参数。朗肯循环是蒸汽动力装置的理想热力循环，包括等熵压缩、等压加热、等熵膨胀和等压冷凝四个过程。"
    arguments = {
        "T_high": {"type": "number", "description": "锅炉出口蒸汽温度，单位：K"},
        "P_high": {"type": "number", "description": "锅炉出口蒸汽压力，单位：Pa"},
        "P_low": {"type": "number", "description": "冷凝器压力，单位：Pa"},
        "fluid": {"type": "string", "description": "工质名称，默认为'Water'"},
        "eta_turbine": {"type": "number", "description": "汽轮机等熵效率，范围[0,1]，默认为1.0(理想)"},
        "eta_pump": {"type": "number", "description": "泵的等熵效率，范围[0,1]，默认为1.0(理想)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 rankine_cycle_efficiency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            T_high = args.get("T_high")
            if T_high is None:
                return Observation(self.name, "错误: 缺少必需参数 T_high")
            P_high = args.get("P_high")
            if P_high is None:
                return Observation(self.name, "错误: 缺少必需参数 P_high")
            P_low = args.get("P_low")
            if P_low is None:
                return Observation(self.name, "错误: 缺少必需参数 P_low")
            fluid = args.get("fluid", None)
            eta_turbine = args.get("eta_turbine", None)
            eta_pump = args.get("eta_pump", None)
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamic_solver_50 import rankine_cycle_efficiency
            
            # 调用函数
            result = rankine_cycle_efficiency(T_high, P_high, P_low, fluid, eta_turbine, eta_pump)
            
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


@Toolbox.register(name="heat_diffusion_solver")
class HeatDiffusionSolverTool(EnvironmentTool):
    """求解一维热传导方程的数值解。使用有限差分法求解一维非稳态热传导方程：ρc_p(∂T/∂t) = k(∂²T/∂x²)"""
    
    name = "heat_diffusion_solver"
    description = "求解一维热传导方程的数值解。使用有限差分法求解一维非稳态热传导方程：ρc_p(∂T/∂t) = k(∂²T/∂x²)"
    arguments = {
        "L": {"type": "number", "description": "计算域长度，单位：m"},
        "nx": {"type": "integer", "description": "空间网格点数"},
        "T_left": {"type": "number", "description": "左边界温度，单位：K"},
        "T_right": {"type": "number", "description": "右边界温度，单位：K"},
        "k": {"type": "number", "description": "热导率，单位：W/(m·K)"},
        "rho": {"type": "number", "description": "密度，单位：kg/m³"},
        "c_p": {"type": "number", "description": "比热容，单位：J/(kg·K)"},
        "t_final": {"type": "number", "description": "模拟总时间，单位：s"},
        "nt": {"type": "integer", "description": "时间步数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 heat_diffusion_solver 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            L = args.get("L")
            if L is None:
                return Observation(self.name, "错误: 缺少必需参数 L")
            nx = args.get("nx")
            if nx is None:
                return Observation(self.name, "错误: 缺少必需参数 nx")
            T_left = args.get("T_left")
            if T_left is None:
                return Observation(self.name, "错误: 缺少必需参数 T_left")
            T_right = args.get("T_right")
            if T_right is None:
                return Observation(self.name, "错误: 缺少必需参数 T_right")
            k = args.get("k")
            if k is None:
                return Observation(self.name, "错误: 缺少必需参数 k")
            rho = args.get("rho")
            if rho is None:
                return Observation(self.name, "错误: 缺少必需参数 rho")
            c_p = args.get("c_p")
            if c_p is None:
                return Observation(self.name, "错误: 缺少必需参数 c_p")
            t_final = args.get("t_final")
            if t_final is None:
                return Observation(self.name, "错误: 缺少必需参数 t_final")
            nt = args.get("nt")
            if nt is None:
                return Observation(self.name, "错误: 缺少必需参数 nt")
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamic_solver_50 import heat_diffusion_solver
            
            # 调用函数
            result = heat_diffusion_solver(L, nx, T_left, T_right, k, rho, c_p, t_final, nt)
            
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


@Toolbox.register(name="phase_field_simulation")
class PhaseFieldSimulationTool(EnvironmentTool):
    """使用相场方法模拟材料相变过程。基于Allen-Cahn方程模拟相场演化: ∂φ/∂t = D∇²φ + φ(1-φ)(φ-0.5)"""
    
    name = "phase_field_simulation"
    description = "使用相场方法模拟材料相变过程。基于Allen-Cahn方程模拟相场演化: ∂φ/∂t = D∇²φ + φ(1-φ)(φ-0.5)"
    arguments = {
        "nx": {"type": "integer", "description": "x方向的网格点数"},
        "ny": {"type": "integer", "description": "y方向的网格点数"},
        "dx": {"type": "number", "description": "x方向的网格间距，单位：无量纲"},
        "dy": {"type": "number", "description": "y方向的网格间距，单位：无量纲"},
        "D": {"type": "number", "description": "扩散系数，单位：无量纲"},
        "time_steps": {"type": "integer", "description": "模拟的时间步数"},
        "dt": {"type": "number", "description": "时间步长，单位：无量纲"},
        "initial_radius": {"type": "number", "description": "初始相场圆形区域的半径，默认为10"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 phase_field_simulation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            nx = args.get("nx")
            if nx is None:
                return Observation(self.name, "错误: 缺少必需参数 nx")
            ny = args.get("ny")
            if ny is None:
                return Observation(self.name, "错误: 缺少必需参数 ny")
            dx = args.get("dx")
            if dx is None:
                return Observation(self.name, "错误: 缺少必需参数 dx")
            dy = args.get("dy")
            if dy is None:
                return Observation(self.name, "错误: 缺少必需参数 dy")
            D = args.get("D")
            if D is None:
                return Observation(self.name, "错误: 缺少必需参数 D")
            time_steps = args.get("time_steps")
            if time_steps is None:
                return Observation(self.name, "错误: 缺少必需参数 time_steps")
            dt = args.get("dt")
            if dt is None:
                return Observation(self.name, "错误: 缺少必需参数 dt")
            initial_radius = args.get("initial_radius", None)
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamic_solver_50 import phase_field_simulation
            
            # 调用函数
            result = phase_field_simulation(nx, ny, dx, dy, D, time_steps, dt, initial_radius)
            
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


@Toolbox.register(name="calculate_internal_energy_change")
class CalculateInternalEnergyChangeTool(EnvironmentTool):
    """计算热力学系统内能变化，基于热力学第一定律。"""
    
    name = "calculate_internal_energy_change"
    description = "计算热力学系统内能变化，基于热力学第一定律。"
    arguments = {
        "heat_flow": {"type": "number", "description": "流入系统的热量，单位为焦耳(J)。正值表示热量流入系统，负值表示热量流出系统。"},
        "work_done": {"type": "number", "description": "对系统做的功，单位为焦耳(J)。正值表示对系统做功，负值表示系统对外做功。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_internal_energy_change 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            heat_flow = args.get("heat_flow")
            if heat_flow is None:
                return Observation(self.name, "错误: 缺少必需参数 heat_flow")
            work_done = args.get("work_done")
            if work_done is None:
                return Observation(self.name, "错误: 缺少必需参数 work_done")
            
            # 导入并调用原始函数
            from toolkits.physics.thermodynamics.thermodynamics_solver_8030 import calculate_internal_energy_change
            
            # 调用函数
            result = calculate_internal_energy_change(heat_flow, work_done)
            
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

def register_thermodynamics_tools(environment):
    """
    将所有 thermodynamics 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

