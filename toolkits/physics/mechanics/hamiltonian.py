#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哈密顿系统模块
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, Optional, List, Tuple
from scipy.integrate import solve_ivp
from .lagrangian_mechanics import LagrangianMechanics
from ..core.lagrangian import Lagrangian

class HamiltonianSystem(LagrangianMechanics):
    """
    哈密顿系统
    
    实现了哈密顿力学方法
    """
    
    def __init__(self, 
                 lagrangian: Optional[Lagrangian] = None,
                 hamiltonian: Optional[str] = None,
                 coordinates: Optional[List[str]] = None):
        """
        初始化哈密顿系统
        
        Args:
            lagrangian: 拉格朗日函数
            hamiltonian: 哈密顿函数
            coordinates: 坐标列表
        """
        self.hamiltonian_expr = hamiltonian
        self.coordinates = coordinates or ['q', 'p']
        
        if lagrangian:
            super().__init__(lagrangian=lagrangian, coordinates=coordinates)
            # 从拉格朗日函数计算哈密顿函数
            self.hamiltonian_expr = str(self.compute_hamiltonian())
        else:
            super().__init__()
    
    def set_hamiltonian(self, hamiltonian: str):
        """设置哈密顿函数"""
        self.hamiltonian_expr = hamiltonian
    
    def compute_hamilton_equations(self) -> Dict[str, sp.Expr]:
        """计算哈密顿方程"""
        if not self.hamiltonian_expr:
            raise ValueError("哈密顿函数未设置")
        
        H = sp.sympify(self.hamiltonian_expr)
        equations = {}
        
        # 获取变量
        q = sp.Symbol('q')
        p = sp.Symbol('p')
        t = sp.Symbol('t')
        
        # 哈密顿方程
        # dq/dt = ∂H/∂p
        # dp/dt = -∂H/∂q
        
        if p in H.free_symbols:
            equations['dq_dt'] = sp.diff(H, p)
        if q in H.free_symbols:
            equations['dp_dt'] = -sp.diff(H, q)
        
        return equations
    
    def solve_hamilton_equations(self,
                                initial_conditions: Dict[str, float],
                                time_span: Tuple[float, float] = (0, 10),
                                num_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        求解哈密顿方程
        
        Args:
            initial_conditions: 初始条件
            time_span: 时间范围
            num_points: 时间点数
            
        Returns:
            解的时间序列
        """
        if not self.hamiltonian_expr:
            raise ValueError("哈密顿函数未设置")
        
        # 获取哈密顿方程
        equations = self.compute_hamilton_equations()
        
        # 转换为数值求解形式
        def system(t, state):
            q, p = state
            
            # 这里需要根据具体的哈密顿函数来实现
            # 简化示例：谐振子 H = p²/2m + kq²/2
            m = 1.0
            k = 1.0
            
            dq_dt = p / m
            dp_dt = -k * q
            
            return [dq_dt, dp_dt]
        
        # 设置初始条件
        y0 = [initial_conditions.get('q', 0.0), initial_conditions.get('p', 0.0)]
        
        # 求解ODE
        t_eval = np.linspace(time_span[0], time_span[1], num_points)
        solution = solve_ivp(system, time_span, y0, t_eval=t_eval, method='RK45')
        
        return {
            't': solution.t,
            'q': solution.y[0],
            'p': solution.y[1]
        }
    
    def compute_poisson_bracket(self, 
                               f: str, 
                               g: str) -> sp.Expr:
        """
        计算泊松括号 {f, g}
        
        Args:
            f: 函数f
            g: 函数g
            
        Returns:
            泊松括号
        """
        f_expr = sp.sympify(f)
        g_expr = sp.sympify(g)
        
        q = sp.Symbol('q')
        p = sp.Symbol('p')
        
        # 泊松括号 {f, g} = ∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q
        poisson_bracket = (sp.diff(f_expr, q) * sp.diff(g_expr, p) - 
                          sp.diff(f_expr, p) * sp.diff(g_expr, q))
        
        return poisson_bracket
    
    def check_integrability(self) -> Dict[str, Any]:
        """检查可积性"""
        if not self.hamiltonian_expr:
            return {'is_integrable': False, 'reason': '哈密顿函数未设置'}
        
        # 检查是否有足够的守恒量
        H = sp.sympify(self.hamiltonian_expr)
        q = sp.Symbol('q')
        p = sp.Symbol('p')
        
        # 检查能量守恒
        energy_conserved = sp.diff(H, sp.Symbol('t')) == 0
        
        # 检查其他守恒量（简化版本）
        # 在实际应用中，需要更复杂的分析
        
        return {
            'is_integrable': energy_conserved,
            'energy_conserved': energy_conserved,
            'degrees_of_freedom': 1,  # 简化假设
            'conserved_quantities': ['energy'] if energy_conserved else []
        }
    
    def compute_action_angle_variables(self,
                                     trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算作用量-角变量
        
        Args:
            trajectory: 轨迹数据
            
        Returns:
            作用量-角变量
        """
        if 'q' not in trajectory or 'p' not in trajectory:
            raise ValueError("需要位置和动量数据")
        
        q = trajectory['q']
        p = trajectory['p']
        
        # 计算作用量（相空间面积）
        # 对于周期运动，作用量是相空间中的面积
        action = np.trapz(p, q)
        
        # 计算角变量（简化版本）
        # 对于谐振子，角变量是相位
        omega = 1.0  # 假设频率为1
        t = trajectory['t']
        angle = omega * t
        
        return {
            'action': np.full_like(t, action),
            'angle': angle
        }
    
    def solve(self, **kwargs) -> Dict[str, Any]:
        """求解哈密顿系统"""
        initial_conditions = kwargs.get('initial_conditions', {})
        time_span = kwargs.get('time_span', (0, 10))
        num_points = kwargs.get('num_points', 1000)
        
        # 求解哈密顿方程
        trajectory = self.solve_hamilton_equations(
            initial_conditions, time_span, num_points
        )
        
        # 计算能量
        if self.hamiltonian_expr:
            energy = self.compute_energy_from_hamiltonian(trajectory)
            trajectory['energy'] = energy
        
        # 分析可积性
        integrability = self.check_integrability()
        
        # 计算作用量-角变量
        try:
            action_angle = self.compute_action_angle_variables(trajectory)
            trajectory.update(action_angle)
        except:
            pass
        
        return {
            'trajectory': trajectory,
            'hamilton_equations': {k: str(v) for k, v in self.compute_hamilton_equations().items()},
            'integrability': integrability,
            'hamiltonian': self.hamiltonian_expr
        }
    
    def compute_energy_from_hamiltonian(self, 
                                      trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        """从哈密顿函数计算能量"""
        if not self.hamiltonian_expr:
            return np.zeros_like(trajectory['t'])
        
        # 转换为数值函数
        H = sp.sympify(self.hamiltonian_expr)
        H_func = sp.lambdify([sp.Symbol('q'), sp.Symbol('p')], H)
        
        energy = []
        for i in range(len(trajectory['t'])):
            e = H_func(trajectory['q'][i], trajectory['p'][i])
            energy.append(e)
        
        return np.array(energy)
