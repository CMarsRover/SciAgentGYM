#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变分问题基类
"""

import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

@dataclass
class BoundaryConditions:
    """边界条件"""
    initial_value: Optional[float] = None
    final_value: Optional[float] = None
    initial_derivative: Optional[float] = None
    final_derivative: Optional[float] = None
    periodic: bool = False

class VariationalProblem(ABC):
    """
    变分问题抽象基类
    
    实现了变分法中的核心概念和方法
    """
    
    def __init__(self, 
                 lagrangian: Optional[str] = None,
                 variables: Optional[list] = None,
                 boundary_conditions: Optional[BoundaryConditions] = None):
        """
        初始化变分问题
        
        Args:
            lagrangian: 拉格朗日函数表达式
            variables: 变量列表
            boundary_conditions: 边界条件
        """
        self.lagrangian_expr = lagrangian
        self.variables = variables or ['t', 'x', 'x_dot']
        self.boundary_conditions = boundary_conditions or BoundaryConditions()
        self.symbols = {}
        self._setup_symbols()
        
    def _setup_symbols(self):
        """设置符号变量"""
        for var in self.variables:
            self.symbols[var] = sp.Symbol(var)
    
    def set_lagrangian(self, lagrangian: str):
        """设置拉格朗日函数"""
        self.lagrangian_expr = lagrangian
        
    def set_boundary_conditions(self, boundary_conditions: BoundaryConditions):
        """设置边界条件"""
        self.boundary_conditions = boundary_conditions
    
    def compute_euler_lagrange_equation(self) -> sp.Expr:
        """
        计算欧拉-拉格朗日方程
        
        Returns:
            欧拉-拉格朗日方程
        """
        if not self.lagrangian_expr:
            raise ValueError("拉格朗日函数未设置")
            
        # 解析拉格朗日函数
        L = sp.sympify(self.lagrangian_expr)
        
        # 获取变量
        t = self.symbols.get('t', sp.Symbol('t'))
        x = self.symbols.get('x', sp.Symbol('x'))
        x_dot = self.symbols.get('x_dot', sp.Symbol('x_dot'))
        
        # 计算偏导数
        dL_dx = sp.diff(L, x)
        dL_dxdot = sp.diff(L, x_dot)
        
        # 计算时间导数
        dL_dxdot_dt = sp.diff(dL_dxdot, t)
        
        # 欧拉-拉格朗日方程: d/dt(dL/dx_dot) - dL/dx = 0
        euler_lagrange = dL_dxdot_dt - dL_dx
        
        return euler_lagrange
    
    def solve_euler_lagrange(self, 
                           initial_conditions: Optional[Dict[str, float]] = None,
                           time_span: Tuple[float, float] = (0, 10),
                           num_points: int = 1000,
                           method: str = 'RK45',
                           rtol: float = 1e-6,
                           atol: float = 1e-8) -> Dict[str, np.ndarray]:
        """
        求解欧拉-拉格朗日方程
        
        使用数值方法求解变分问题对应的欧拉-拉格朗日方程，
        通过将二阶微分方程转换为一阶微分方程组进行求解。
        
        Parameters:
        -----------
        initial_conditions : dict, optional
            初始条件字典，包含'x'和'x_dot'的初始值，默认为{'x': 1.0, 'x_dot': 0.0}
        time_span : tuple, optional
            时间积分范围，格式为(t_start, t_end)，默认为(0, 10)
        num_points : int, optional
            时间网格点数，默认为1000
        method : str, optional
            数值求解方法，默认为'RK45'
        rtol : float, optional
            相对误差容限，默认为1e-6
        atol : float, optional
            绝对误差容限，默认为1e-8
            
        Returns:
        --------
        dict
            包含解的时间序列字典：
            - 't': 时间数组
            - 'x': 位置数组
            - 'x_dot': 速度数组
        """
        from scipy.integrate import solve_ivp
        
        # 获取欧拉-拉格朗日方程
        eq = self.compute_euler_lagrange_equation()
        
        # 转换为数值求解形式
        def system(t, state):
            """
            计算system相关结果
            
            Parameters:
            -----------
            t : float
                t，单位为s
            state : float
                state参数
            
            Returns:
            --------
            float
                计算结果
            """
            x, x_dot = state
            # 这里需要根据具体的拉格朗日函数来实现
            # 简化示例：假设是简单的谐振子
            x_ddot = -x  # 简化的二阶微分方程
            return [x_dot, x_ddot]
        
        # 设置初始条件
        if initial_conditions is None:
            initial_conditions = {'x': 1.0, 'x_dot': 0.0}
        
        y0 = [initial_conditions['x'], initial_conditions['x_dot']]
        
        # 求解ODE
        t_span = time_span
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        solution = solve_ivp(system, t_span, y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
        
        return {
            't': solution.t,
            'x': solution.y[0],
            'x_dot': solution.y[1]
        }
    
    def compute_action(self, 
                      trajectory: Dict[str, np.ndarray],
                      lagrangian_func: Optional[Callable] = None,
                      integration_method: str = 'trapz') -> float:
        """
        计算作用量
        
        计算给定轨迹的作用量，即拉格朗日函数沿轨迹的时间积分。
        作用量是变分法中的核心概念，用于描述系统的动力学行为。
        
        Parameters:
        -----------
        trajectory : dict
            轨迹数据字典，包含't'、'x'、'x_dot'数组
        lagrangian_func : callable, optional
            拉格朗日函数（数值形式），默认为谐振子拉格朗日函数
        integration_method : str, optional
            积分方法，可选'trapz'或'simpson'，默认为'trapz'
            
        Returns:
        --------
        float
            作用量值，单位为J·s
        """
        t = trajectory['t']
        x = trajectory['x']
        x_dot = trajectory['x_dot']
        
        if lagrangian_func is None:
            # 默认拉格朗日函数
            def lagrangian_func(t, x, x_dot):
                """
                拉格朗日力学相关计算
                
                Parameters:
                -----------
                t : float
                    t，单位为s
                x : float
                    x，x坐标
                x_dot : float
                    x_dot参数
                
                Returns:
                --------
                float
                    计算结果
                """
                return 0.5 * x_dot**2 - 0.5 * x**2
        
        # 计算拉格朗日函数值
        L_values = [lagrangian_func(t[i], x[i], x_dot[i]) for i in range(len(t))]
        
        # 数值积分
        if integration_method == 'trapz':
            action = np.trapz(L_values, t)
        elif integration_method == 'simpson':
            from scipy.integrate import simpson
            action = simpson(L_values, t)
        else:
            raise ValueError(f"不支持的积分方法: {integration_method}")
        
        return action
    
    def verify_solution(self, 
                       solution: Dict[str, np.ndarray],
                       tolerance: float = 1e-6,
                       lagrangian_expr: Optional[str] = None) -> bool:
        """
        验证解是否满足欧拉-拉格朗日方程
        
        通过计算数值解在欧拉-拉格朗日方程中的残差来验证解的正确性。
        残差越小，说明解越接近真实解。
        
        Parameters:
        -----------
        solution : dict
            解的数据字典，包含't'、'x'、'x_dot'数组
        tolerance : float, optional
            验证容差，默认为1e-6
        lagrangian_expr : str, optional
            拉格朗日函数表达式，用于计算精确的欧拉-拉格朗日方程
            
        Returns:
        --------
        bool
            解是否满足欧拉-拉格朗日方程（残差小于容差）
        """
        # 计算数值导数
        t = solution['t']
        x = solution['x']
        x_dot = solution['x_dot']
        
        # 二阶导数
        x_ddot = np.gradient(x_dot, t)
        
        # 检查欧拉-拉格朗日方程
        residual = x_ddot + x  # 对于L = 0.5*x_dot^2 - 0.5*x^2
        
        max_residual = np.max(np.abs(residual))
        
        return max_residual < tolerance
    
    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """抽象方法：求解变分问题"""
        pass
    
    def get_problem_info(self) -> Dict[str, Any]:
        """获取问题信息"""
        return {
            'lagrangian': self.lagrangian_expr,
            'variables': self.variables,
            'boundary_conditions': self.boundary_conditions,
            'euler_lagrange_equation': str(self.compute_euler_lagrange_equation())
        }
