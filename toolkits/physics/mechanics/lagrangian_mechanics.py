#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日力学模块
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, Optional, List, Tuple
from scipy.integrate import solve_ivp
from ..core.problem import VariationalProblem, BoundaryConditions
from ..core.lagrangian import Lagrangian

class LagrangianMechanics(VariationalProblem):
    """
    拉格朗日力学系统
    
    实现了经典力学中的拉格朗日方法
    """
    
    def __init__(self, 
                 lagrangian: Optional[Lagrangian] = None,
                 coordinates: Optional[List[str]] = None,
                 constraints: Optional[List[str]] = None):
        """
        初始化拉格朗日力学系统
        
        Args:
            lagrangian: 拉格朗日函数
            coordinates: 广义坐标
            constraints: 约束条件
        """
        self.lagrangian = lagrangian
        self.coordinates = coordinates or ['x', 'y']
        self.constraints = constraints or []
        self.generalized_forces = {}
        
        # 调用父类初始化
        if lagrangian:
            super().__init__(
                lagrangian=lagrangian.expression,
                variables=['t'] + coordinates + [f'{coord}_dot' for coord in coordinates]
            )
        else:
            super().__init__()
    
    def add_generalized_force(self, coordinate: str, force: str):
        """添加广义力"""
        self.generalized_forces[coordinate] = force
    
    def compute_generalized_momenta(self) -> Dict[str, sp.Expr]:
        """计算广义动量"""
        if not self.lagrangian:
            raise ValueError("拉格朗日函数未设置")
        
        momenta = {}
        for coord in self.coordinates:
            coord_dot = f'{coord}_dot'
            if coord_dot in self.lagrangian.symbols:
                momenta[coord] = sp.diff(self.lagrangian.sympy_expr, self.lagrangian.symbols[coord_dot])
        
        return momenta
    
    def compute_hamiltonian(self) -> sp.Expr:
        """计算哈密顿函数"""
        if not self.lagrangian:
            raise ValueError("拉格朗日函数未设置")
        
        # 计算广义动量
        momenta = self.compute_generalized_momenta()
        
        # 哈密顿函数 H = Σ p_i q_i_dot - L
        H = 0
        for coord in self.coordinates:
            coord_dot = f'{coord}_dot'
            if coord in momenta and coord_dot in self.lagrangian.symbols:
                H += momenta[coord] * self.lagrangian.symbols[coord_dot]
        
        H -= self.lagrangian.sympy_expr
        return H
    
    def compute_lagrange_equations(self) -> Dict[str, sp.Expr]:
        """计算拉格朗日方程"""
        if not self.lagrangian:
            raise ValueError("拉格朗日函数未设置")
        
        equations = {}
        t = self.lagrangian.symbols.get('t', sp.Symbol('t'))
        
        for coord in self.coordinates:
            coord_dot = f'{coord}_dot'
            if coord in self.lagrangian.symbols and coord_dot in self.lagrangian.symbols:
                # 计算偏导数
                dL_dq = sp.diff(self.lagrangian.sympy_expr, self.lagrangian.symbols[coord])
                dL_dqdot = sp.diff(self.lagrangian.sympy_expr, self.lagrangian.symbols[coord_dot])
                
                # 计算时间导数
                dL_dqdot_dt = sp.diff(dL_dqdot, t)
                
                # 拉格朗日方程: d/dt(dL/dq_dot) - dL/dq = Q
                Q = self.generalized_forces.get(coord, 0)
                equation = dL_dqdot_dt - dL_dq - Q
                equations[coord] = equation
        
        return equations
    
    def solve_motion(self, 
                    initial_conditions: Dict[str, float],
                    time_span: Tuple[float, float] = (0, 10),
                    num_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        求解运动方程
        
        Args:
            initial_conditions: 初始条件
            time_span: 时间范围
            num_points: 时间点数
            
        Returns:
            运动轨迹
        """
        if not self.lagrangian:
            raise ValueError("拉格朗日函数未设置")
        
        # 获取拉格朗日方程
        equations = self.compute_lagrange_equations()
        
        # 转换为数值求解形式
        def system(t, state):
            """
            计算system相关结果
            
            Parameters:
            -----------
            t : float
                t, 单位为s
            state : array
                state参数
            
            Returns:
            --------
            array
                计算结果
            """
            # state = [q1, q2, ..., q1_dot, q2_dot, ...]
            n_coords = len(self.coordinates)
            q = state[:n_coords]
            q_dot = state[n_coords:]
            
            # 计算二阶导数
            q_ddot = []
            for i, coord in enumerate(self.coordinates):
                # 这里需要根据具体的拉格朗日函数来实现
                # 简化示例：假设是简单的谐振子
                if coord == 'x':
                    q_ddot.append(-q[i])  # 简化的二阶微分方程
                elif coord == 'y':
                    q_ddot.append(-q[i])
                else:
                    q_ddot.append(0)
            
            return list(q_dot) + q_ddot
        
        # 设置初始条件
        y0 = []
        for coord in self.coordinates:
            y0.append(initial_conditions.get(coord, 0.0))
        for coord in self.coordinates:
            coord_dot = f'{coord}_dot'
            y0.append(initial_conditions.get(coord_dot, 0.0))
        
        # 求解ODE
        t_eval = np.linspace(time_span[0], time_span[1], num_points)
        solution = solve_ivp(system, time_span, y0, t_eval=t_eval, method='RK45')
        
        # 整理结果
        result = {'t': solution.t}
        n_coords = len(self.coordinates)
        for i, coord in enumerate(self.coordinates):
            result[coord] = solution.y[i]
            result[f'{coord}_dot'] = solution.y[i + n_coords]
        
        return result
    
    def compute_energy(self, trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        """计算能量"""
        if not self.lagrangian:
            raise ValueError("拉格朗日函数未设置")
        
        # 计算哈密顿函数
        H = self.compute_hamiltonian()
        
        # 转换为数值函数
        H_func = sp.lambdify(list(self.lagrangian.symbols.values()), H)
        
        # 计算能量
        energy = []
        for i in range(len(trajectory['t'])):
            args = [trajectory['t'][i]]
            for coord in self.coordinates:
                args.append(trajectory[coord][i])
            for coord in self.coordinates:
                coord_dot = f'{coord}_dot'
                args.append(trajectory[coord_dot][i])
            energy.append(H_func(*args))
        
        return np.array(energy)
    
    def solve(self, **kwargs) -> Dict[str, Any]:
        """求解变分问题"""
        initial_conditions = kwargs.get('initial_conditions', {})
        time_span = kwargs.get('time_span', (0, 10))
        num_points = kwargs.get('num_points', 1000)
        
        trajectory = self.solve_motion(initial_conditions, time_span, num_points)
        energy = self.compute_energy(trajectory)
        
        return {
            'trajectory': trajectory,
            'energy': energy,
            'lagrange_equations': {k: str(v) for k, v in self.compute_lagrange_equations().items()},
            'hamiltonian': str(self.compute_hamiltonian()),
            'generalized_momenta': {k: str(v) for k, v in self.compute_generalized_momenta().items()}
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'coordinates': self.coordinates,
            'constraints': self.constraints,
            'generalized_forces': self.generalized_forces
        }
        
        if self.lagrangian:
            info.update({
                'lagrangian': self.lagrangian.get_info(),
                'lagrange_equations': {k: str(v) for k, v in self.compute_lagrange_equations().items()},
                'hamiltonian': str(self.compute_hamiltonian()),
                'generalized_momenta': {k: str(v) for k, v in self.compute_generalized_momenta().items()}
            })
        
        return info
