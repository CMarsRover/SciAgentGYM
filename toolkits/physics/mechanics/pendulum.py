#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单摆系统模块
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, Optional, Tuple
from .lagrangian_mechanics import LagrangianMechanics
from ..core.lagrangian import Lagrangian

class PendulumSystem(LagrangianMechanics):
    """
    单摆系统
    
    实现了简单单摆和双摆的动力学分析
    """
    
    def __init__(self, 
                 length: float = 1.0,
                 mass: float = 1.0,
                 gravity: float = 9.81,
                 damping: float = 0.0):
        """
        初始化单摆系统
        
        Args:
            length: 摆长
            mass: 质量
            gravity: 重力加速度
            damping: 阻尼系数
        """
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.damping = damping
        
        # 创建拉格朗日函数
        lagrangian = Lagrangian(
            expression=f"0.5 * {mass} * {length}^2 * theta_dot^2 + {mass} * {gravity} * {length} * cos(theta)",
            variables=['t', 'theta', 'theta_dot'],
            parameters={'m': mass, 'l': length, 'g': gravity}
        )
        
        super().__init__(
            lagrangian=lagrangian,
            coordinates=['theta']
        )
    
    def get_natural_frequency(self) -> float:
        """计算自然频率"""
        return np.sqrt(self.gravity / self.length)
    
    def get_period(self) -> float:
        """计算小角度近似下的周期"""
        return 2 * np.pi / self.get_natural_frequency()
    
    def solve_small_angle_approximation(self,
                                       initial_angle: float,
                                       initial_angular_velocity: float = 0.0,
                                       time_span: Tuple[float, float] = (0, 10),
                                       num_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        小角度近似解
        
        Args:
            initial_angle: 初始角度
            initial_angular_velocity: 初始角速度
            time_span: 时间范围
            num_points: 时间点数
            
        Returns:
            解析解
        """
        omega = self.get_natural_frequency()
        t = np.linspace(time_span[0], time_span[1], num_points)
        
        # 解析解：θ(t) = θ₀ cos(ωt) + (θ̇₀/ω) sin(ωt)
        theta = (initial_angle * np.cos(omega * t) + 
                (initial_angular_velocity / omega) * np.sin(omega * t))
        
        # 角速度
        theta_dot = (-initial_angle * omega * np.sin(omega * t) + 
                    initial_angular_velocity * np.cos(omega * t))
        
        return {
            't': t,
            'theta': theta,
            'theta_dot': theta_dot,
            'method': 'small_angle_approximation'
        }
    
    def solve_numerical(self,
                       initial_angle: float,
                       initial_angular_velocity: float = 0.0,
                       time_span: Tuple[float, float] = (0, 10),
                       num_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        数值解（考虑非线性效应）
        
        Args:
            initial_angle: 初始角度
            initial_angular_velocity: 初始角速度
            time_span: 时间范围
            num_points: 时间点数
            
        Returns:
            数值解
        """
        initial_conditions = {
            'theta': initial_angle,
            'theta_dot': initial_angular_velocity
        }
        
        solution = self.solve(
            initial_conditions=initial_conditions,
            time_span=time_span,
            num_points=num_points
        )
        
        solution['method'] = 'numerical'
        return solution
    
    def compute_energy(self, trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        """计算能量"""
        t = trajectory['t']
        theta = trajectory['theta']
        theta_dot = trajectory['theta_dot']
        
        # 动能
        kinetic_energy = 0.5 * self.mass * self.length**2 * theta_dot**2
        
        # 势能
        potential_energy = self.mass * self.gravity * self.length * (1 - np.cos(theta))
        
        # 总能量
        total_energy = kinetic_energy + potential_energy
        
        return total_energy
    
    def analyze_stability(self, 
                         equilibrium_points: Optional[list] = None) -> Dict[str, Any]:
        """
        分析稳定性
        
        Args:
            equilibrium_points: 平衡点列表
            
        Returns:
            稳定性分析结果
        """
        if equilibrium_points is None:
            equilibrium_points = [0, np.pi]  # 下平衡点和上平衡点
        
        stability_analysis = {}
        
        for i, point in enumerate(equilibrium_points):
            # 计算雅可比矩阵
            # 对于单摆：θ̈ + (g/l) sin(θ) = 0
            # 线性化：θ̈ + (g/l) θ = 0 (在θ=0附近)
            # 或者：θ̈ - (g/l) θ = 0 (在θ=π附近)
            
            if abs(point) < 1e-6:  # 下平衡点
                eigenvalue = 1j * np.sqrt(self.gravity / self.length)  # 纯虚数，稳定
                stability = "稳定"
            else:  # 上平衡点
                eigenvalue = np.sqrt(self.gravity / self.length)  # 正实数，不稳定
                stability = "不稳定"
            
            stability_analysis[f"平衡点_{i}"] = {
                'position': point,
                'eigenvalue': eigenvalue,
                'stability': stability
            }
        
        return stability_analysis
    
    def get_phase_portrait_data(self,
                               theta_range: Tuple[float, float] = (-np.pi, np.pi),
                               theta_dot_range: Tuple[float, float] = (-3, 3),
                               n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        获取相图数据
        
        Args:
            theta_range: 角度范围
            theta_dot_range: 角速度范围
            n_points: 网格点数
            
        Returns:
            相图数据
        """
        theta = np.linspace(theta_range[0], theta_range[1], n_points)
        theta_dot = np.linspace(theta_dot_range[0], theta_dot_range[1], n_points)
        Theta, Theta_dot = np.meshgrid(theta, theta_dot)
        
        # 计算导数
        Theta_ddot = -(self.gravity / self.length) * np.sin(Theta)
        
        return {
            'theta': Theta,
            'theta_dot': Theta_dot,
            'theta_ddot': Theta_ddot
        }
    
    def solve(self, **kwargs) -> Dict[str, Any]:
        """求解单摆运动"""
        solution = super().solve(**kwargs)
        
        # 添加单摆特有的分析
        solution['natural_frequency'] = self.get_natural_frequency()
        solution['period'] = self.get_period()
        solution['stability'] = self.analyze_stability()
        
        return solution
