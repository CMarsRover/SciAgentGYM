#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, Optional, List, Tuple
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_solution(solution: Dict[str, np.ndarray], 
                 title: str = "变分问题解",
                 figsize: Tuple[int, int] = (12, 8),
                 save_path: Optional[str] = None) -> None:
    """
    绘制变分问题的解
    
    Args:
        solution: 解的数据字典
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    t = solution['t']
    
    # 位置图
    if 'x' in solution:
        axes[0, 0].plot(t, solution['x'], 'b-', linewidth=2, label='x(t)')
        axes[0, 0].set_xlabel('时间 t')
        axes[0, 0].set_ylabel('位置 x')
        axes[0, 0].set_title('位置随时间变化')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # 速度图
    if 'x_dot' in solution:
        axes[0, 1].plot(t, solution['x_dot'], 'r-', linewidth=2, label='ẋ(t)')
        axes[0, 1].set_xlabel('时间 t')
        axes[0, 1].set_ylabel('速度 ẋ')
        axes[0, 1].set_title('速度随时间变化')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # 相图
    if 'x' in solution and 'x_dot' in solution:
        axes[1, 0].plot(solution['x'], solution['x_dot'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('位置 x')
        axes[1, 0].set_ylabel('速度 ẋ')
        axes[1, 0].set_title('相图')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 能量图
    if 'energy' in solution:
        axes[1, 1].plot(t, solution['energy'], 'm-', linewidth=2, label='能量')
        axes[1, 1].set_xlabel('时间 t')
        axes[1, 1].set_ylabel('能量 E')
        axes[1, 1].set_title('能量随时间变化')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_trajectory(trajectory: Dict[str, np.ndarray],
                   title: str = "运动轨迹",
                   figsize: Tuple[int, int] = (12, 8),
                   save_path: Optional[str] = None) -> None:
    """
    绘制运动轨迹
    
    Args:
        trajectory: 轨迹数据
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    t = trajectory['t']
    
    # 检查是否有多个坐标
    coords = [key for key in trajectory.keys() if key not in ['t'] and not key.endswith('_dot')]
    
    if len(coords) >= 2:
        # 2D轨迹图
        axes[0, 0].plot(trajectory[coords[0]], trajectory[coords[1]], 'b-', linewidth=2)
        axes[0, 0].set_xlabel(f'{coords[0]}')
        axes[0, 0].set_ylabel(f'{coords[1]}')
        axes[0, 0].set_title('2D轨迹')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
    
    # 时间序列图
    for i, coord in enumerate(coords[:2]):
        axes[0, 1].plot(t, trajectory[coord], linewidth=2, label=f'{coord}(t)')
    axes[0, 1].set_xlabel('时间 t')
    axes[0, 1].set_ylabel('坐标')
    axes[0, 1].set_title('坐标随时间变化')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 速度图
    for i, coord in enumerate(coords[:2]):
        coord_dot = f'{coord}_dot'
        if coord_dot in trajectory:
            axes[1, 0].plot(t, trajectory[coord_dot], linewidth=2, label=f'{coord_dot}(t)')
    axes[1, 0].set_xlabel('时间 t')
    axes[1, 0].set_ylabel('速度')
    axes[1, 0].set_title('速度随时间变化')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 相图
    if len(coords) >= 2:
        coord1_dot = f'{coords[0]}_dot'
        coord2_dot = f'{coords[1]}_dot'
        if coord1_dot in trajectory and coord2_dot in trajectory:
            axes[1, 1].plot(trajectory[coord1_dot], trajectory[coord2_dot], 'g-', linewidth=2)
            axes[1, 1].set_xlabel(f'{coord1_dot}')
            axes[1, 1].set_ylabel(f'{coord2_dot}')
            axes[1, 1].set_title('速度相图')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_surface(x: np.ndarray, 
                y: np.ndarray, 
                z: np.ndarray,
                title: str = "曲面图",
                figsize: Tuple[int, int] = (10, 8),
                save_path: Optional[str] = None) -> None:
    """
    绘制3D曲面图
    
    Args:
        x: x坐标网格
        y: y坐标网格
        z: z坐标值
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_phase_portrait(system_func: callable,
                       x_range: Tuple[float, float] = (-3, 3),
                       y_range: Tuple[float, float] = (-3, 3),
                       n_points: int = 20,
                       title: str = "相图",
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None) -> None:
    """
    绘制相图
    
    Args:
        system_func: 系统函数
        x_range: x范围
        y_range: y范围
        n_points: 网格点数
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    U, V = np.zeros_like(X), np.zeros_like(Y)
    
    for i in range(n_points):
        for j in range(n_points):
            U[i, j], V[i, j] = system_func(X[i, j], Y[i, j])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制向量场
    ax.quiver(X, Y, U, V, alpha=0.6)
    
    # 绘制零增长线
    ax.contour(X, Y, U, levels=[0], colors='red', linewidths=2, label='ẋ=0')
    ax.contour(X, Y, V, levels=[0], colors='blue', linewidths=2, label='ẏ=0')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_energy_landscape(energy_func: callable,
                         x_range: Tuple[float, float] = (-3, 3),
                         y_range: Tuple[float, float] = (-3, 3),
                         n_points: int = 100,
                         title: str = "能量景观",
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> None:
    """
    绘制能量景观
    
    Args:
        energy_func: 能量函数
        x_range: x范围
        y_range: y范围
        n_points: 网格点数
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = energy_func(X[i, j], Y[i, j])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制等高线图
    contour = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 绘制填充等高线图
    filled_contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # 添加颜色条
    plt.colorbar(filled_contour, ax=ax, label='能量')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
