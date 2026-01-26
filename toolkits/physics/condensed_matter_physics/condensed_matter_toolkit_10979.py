# Filename: condensed_matter_toolkit.py

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
from scipy.optimize import minimize
import networkx as nx
from typing import List, Tuple, Dict, Union, Optional, Callable

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保图片保存目录存在
if not os.path.exists("./images"):
    os.makedirs("./images")

def calculate_parallel_resistance(resistances: List[float]) -> float:
    """
    计算并联电阻的等效电阻值。
    
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
    6.666666666666667  # 约等于6.67Ω
    """
    if not resistances:
        raise ValueError("电阻列表不能为空")
    
    inverse_sum = sum(1.0 / r for r in resistances)
    return 1.0 / inverse_sum

def calculate_series_resistance(resistances: List[float]) -> float:
    """
    计算串联电阻的等效电阻值。
    
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
    30.0  # 30Ω
    """
    if not resistances:
        raise ValueError("电阻列表不能为空")
    
    return sum(resistances)

def solve_complex_circuit(circuit_structure: Dict[str, List[Union[str, List[str]]]],
                         resistances: Dict[str, float]) -> float:
    """
    求解复杂电路网络的等效电阻。
    
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
        整个电路的等效电阻值，单位为欧姆(Ω)
    """
    def _solve_subcircuit(structure):
        if isinstance(structure, str):
            # 基本电阻元件
            return resistances[structure]
        
        circuit_type = structure[0]
        components = structure[1]
        
        # 计算所有子组件的等效电阻
        component_resistances = [_solve_subcircuit(comp) for comp in components]
        
        # 根据连接类型计算等效电阻
        if circuit_type == "series":
            return calculate_series_resistance(component_resistances)
        elif circuit_type == "parallel":
            return calculate_parallel_resistance(component_resistances)
        else:
            raise ValueError(f"未知的电路连接类型: {circuit_type}")
    
    return _solve_subcircuit([circuit_structure["type"], circuit_structure["components"]])

def visualize_circuit(circuit_structure: Dict[str, List[Union[str, List[str]]]],
                     resistances: Dict[str, float],
                     filename: str = "circuit_visualization.png") -> None:
    """
    可视化电路结构并显示等效电阻。
    
    使用NetworkX创建电路的图形表示，便于理解电路拓扑结构。
    
    Parameters:
    -----------
    circuit_structure : Dict[str, List[Union[str, List[str]]]]
        电路结构描述，与solve_complex_circuit函数使用相同格式
    resistances : Dict[str, float]
        电阻元件名称到电阻值的映射，单位为欧姆(Ω)
    filename : str, optional
        图像保存的文件名，默认为"circuit_visualization.png"
    """
    G = nx.Graph()
    
    def _add_components(structure, parent=None, level=0, pos=0):
        if isinstance(structure, str):
            # 基本电阻元件
            node_id = structure
            G.add_node(node_id, label=f"{node_id}\n{resistances[node_id]}Ω")
            if parent:
                G.add_edge(parent, node_id)
            return node_id
        
        circuit_type = structure[0]
        components = structure[1]
        
        # 创建一个表示此子电路的节点
        node_id = f"{circuit_type}_{level}_{pos}"
        G.add_node(node_id, label=f"{circuit_type}")
        if parent:
            G.add_edge(parent, node_id)
        
        # 添加所有子组件
        for i, comp in enumerate(components):
            _add_components(comp, node_id, level+1, i)
        
        return node_id
    
    _add_components([circuit_structure["type"], circuit_structure["components"]])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue")
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=2)
    
    # 添加标签
    labels = {node: data.get("label", node) for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # 计算并显示等效电阻
    eq_resistance = solve_complex_circuit(circuit_structure, resistances)
    plt.title(f"电路结构可视化 - 等效电阻: {eq_resistance:.2f}Ω")
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"./images/{filename}")
    plt.close()

def calculate_density_matrix(hamiltonian: np.ndarray, 
                            temperature: float = 1.0, 
                            kb: float = 1.0) -> np.ndarray:
    """
    计算给定哈密顿量的密度矩阵。
    
    在量子多体系统模拟中，密度矩阵是描述系统量子态的基本工具。
    此函数基于玻尔兹曼分布计算热平衡态的密度矩阵。
    
    Parameters:
    -----------
    hamiltonian : np.ndarray
        系统的哈密顿量，形状为(n, n)的厄米矩阵
    temperature : float, optional
        系统温度，默认为1.0（归一化单位）
    kb : float, optional
        玻尔兹曼常数，默认为1.0（归一化单位）
        
    Returns:
    --------
    np.ndarray
        密度矩阵，形状与哈密顿量相同
    """
    # 确保哈密顿量是厄米矩阵
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("哈密顿量必须是厄米矩阵")
    
    # 对角化哈密顿量，获取本征值和本征向量
    eigenvalues, eigenvectors = linalg.eigh(hamiltonian)
    
    # 计算玻尔兹曼因子
    beta = 1.0 / (kb * temperature)
    
    # 计算配分函数
    boltzmann_factors = np.exp(-beta * eigenvalues)
    partition_function = np.sum(boltzmann_factors)
    
    # 构建密度矩阵
    density_matrix = np.zeros_like(hamiltonian, dtype=complex)
    for i in range(len(eigenvalues)):
        # |ψ⟩⟨ψ| 投影算符
        projection = np.outer(eigenvectors[:, i], eigenvectors[:, i].conj())
        # 加权求和
        density_matrix += np.exp(-beta * eigenvalues[i]) * projection
    
    # 归一化
    density_matrix /= partition_function
    
    return density_matrix

def calculate_expectation_value(operator: np.ndarray, 
                               density_matrix: np.ndarray) -> complex:
    """
    计算量子力学中的期望值。
    
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
        算符的期望值，对于厄米算符，结果应为实数
    """
    # 验证输入维度匹配
    if operator.shape != density_matrix.shape:
        raise ValueError("算符和密度矩阵的维度必须匹配")
    
    # 计算迹: Tr(ρO)
    expectation = np.trace(np.matmul(density_matrix, operator))
    
    return expectation

def simulate_time_evolution(hamiltonian: np.ndarray, 
                           initial_state: np.ndarray, 
                           time_points: np.ndarray,
                           operators: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    模拟量子系统的时间演化。
    
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
        - 算符名称: 对应算符在每个时间点的期望值
    """
    # 确保哈密顿量是厄米矩阵
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("哈密顿量必须是厄米矩阵")
    
    # 对角化哈密顿量
    eigenvalues, eigenvectors = linalg.eigh(hamiltonian)
    
    # 初始化结果字典
    results = {}
    n = hamiltonian.shape[0]
    states = np.zeros((len(time_points), n), dtype=complex)
    
    # 将初始态投影到本征基上
    initial_coeffs = eigenvectors.conj().T @ initial_state
    
    # 计算每个时间点的量子态
    for i, t in enumerate(time_points):
        # 时间演化: |ψ(t)⟩ = ∑ᵢ e^(-iEᵢt/ħ) |Eᵢ⟩⟨Eᵢ|ψ(0)⟩
        # 这里设ħ=1
        evolved_coeffs = initial_coeffs * np.exp(-1j * eigenvalues * t)
        evolved_state = eigenvectors @ evolved_coeffs
        states[i] = evolved_state
    
    results['states'] = states
    
    # 计算算符的期望值
    if operators:
        for name, op in operators.items():
            expectations = np.zeros(len(time_points), dtype=complex)
            for i, state in enumerate(states):
                # 计算期望值: ⟨ψ|O|ψ⟩
                expectations[i] = np.vdot(state, op @ state)
            results[name] = expectations
    
    return results

def visualize_time_evolution(time_points: np.ndarray, 
                            results: Dict[str, np.ndarray],
                            filename: str = "time_evolution.png") -> None:
    """
    可视化量子系统时间演化的结果。
    
    绘制算符期望值随时间的变化，帮助理解量子系统的动力学行为。
    
    Parameters:
    -----------
    time_points : np.ndarray
        时间点数组
    results : Dict[str, np.ndarray]
        simulate_time_evolution函数返回的结果字典
    filename : str, optional
        图像保存的文件名，默认为"time_evolution.png"
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制各算符期望值随时间的变化
    for name, values in results.items():
        if name != 'states':  # 跳过量子态数组
            if np.iscomplex(values).any():
                plt.plot(time_points, values.real, label=f"Re({name})")
                plt.plot(time_points, values.imag, '--', label=f"Im({name})")
            else:
                plt.plot(time_points, values.real, label=name)
    
    plt.xlabel('时间')
    plt.ylabel('期望值')
    plt.title('量子系统时间演化')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./images/{filename}")
    plt.close()

def calculate_dft_energy(atomic_positions: np.ndarray, 
                        atomic_numbers: np.ndarray,
                        functional: str = 'LDA',
                        basis_set: str = 'minimal') -> float:
    """
    模拟第一性原理计算中的密度泛函理论(DFT)能量计算。
    
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
        系统的总能量，单位为电子伏特(eV)
    """
    # 这是一个简化模型，实际DFT计算需要专业软件包
    # 此处仅演示计算流程和接口设计
    
    n_atoms = len(atomic_numbers)
    if atomic_positions.shape[0] != n_atoms:
        raise ValueError("原子位置和原子序数数组长度不匹配")
    
    # 简化的能量计算模型
    # 1. 计算原子间距离矩阵
    distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(atomic_positions[i] - atomic_positions[j])
            distances[i, j] = distances[j, i] = dist
    
    # 2. 简化的能量模型（仅作演示）
    # 实际DFT计算涉及电子密度、交换关联能等复杂计算
    kinetic_energy = sum(atomic_numbers**2.4) * 0.5  # 简化的动能项
    
    # 简化的库仑相互作用
    coulomb_energy = 0.0
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if distances[i, j] > 0.1:  # 避免除零
                coulomb_energy += atomic_numbers[i] * atomic_numbers[j] / distances[i, j]
    
    # 简化的交换关联能
    xc_factor = 0.7 if functional == 'LDA' else 0.8  # 不同泛函的简化处理
    xc_energy = -xc_factor * sum(atomic_numbers**(2/3))
    
    # 基组效应的简化处理
    basis_factor = 1.0
    if basis_set == 'minimal':
        basis_factor = 0.9
    elif basis_set == 'extended':
        basis_factor = 1.1
    
    # 总能量
    total_energy = (kinetic_energy + coulomb_energy + xc_energy) * basis_factor
    
    return total_energy

def optimize_crystal_structure(initial_positions: np.ndarray,
                              atomic_numbers: np.ndarray,
                              functional: str = 'LDA',
                              basis_set: str = 'minimal') -> Tuple[np.ndarray, float]:
    """
    优化晶体结构以找到能量最低的构型。
    
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
        优化后的原子位置坐标和对应的能量值
    """
    n_atoms = len(atomic_numbers)
    
    # 定义目标函数：计算给定构型的能量
    def energy_function(positions_flat):
        positions = positions_flat.reshape(n_atoms, 3)
        return calculate_dft_energy(positions, atomic_numbers, functional, basis_set)
    
    # 使用scipy的优化器进行结构优化
    initial_positions_flat = initial_positions.flatten()
    result = minimize(energy_function, initial_positions_flat, method='BFGS')
    
    # 重塑优化后的位置
    optimized_positions = result.x.reshape(n_atoms, 3)
    optimized_energy = result.fun
    
    return optimized_positions, optimized_energy

def main():
    """
    主函数：演示如何使用工具函数求解凝聚态物理中的问题
    """
    print("凝聚态物理工具包演示")
    print("=" * 50)
    
    # 示例1：复杂电路网络分析
    print("\n1. 复杂电路网络分析")
    print("-" * 30)
    
    # 定义电阻值
    resistances = {
        "R1": 42.0,  # 欧姆
        "R2": 75.0,
        "R3": 33.0,
        "R4": 61.0,
        "R5": 12.5,
        "R6": 27.0
    }
    
    # 根据图中的电路结构定义
    # R1和R2并联，然后与(R3和R4串联)并联，最后与R5并联，整体与R6串联
    circuit_structure = {
        "type": "series",
        "components": [
            ["parallel", [
                ["parallel", ["R1", "R2"]],
                ["series", ["R3", "R4"]],
                "R5"
            ]],
            "R6"
        ]
    }
    
    # 计算等效电阻
    equivalent_resistance = solve_complex_circuit(circuit_structure, resistances)
    print(f"等效电阻: {equivalent_resistance:.2f}Ω")
    
    # 可视化电路
    visualize_circuit(circuit_structure, resistances, "complex_circuit.png")
    print(f"电路可视化已保存至 ./images/complex_circuit.png")
    
    # 示例2：量子多体系统模拟
    print("\n2. 量子多体系统模拟")
    print("-" * 30)
    
    # 定义一个简单的量子系统哈密顿量（2x2矩阵表示单粒子系统）
    # 例如，一个简单的两能级系统
    h = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ])
    
    # 计算密度矩阵
    temperature = 0.5  # 归一化温度
    density_matrix = calculate_density_matrix(h, temperature)
    print("密度矩阵:")
    print(density_matrix)
    
    # 定义一个观测算符（例如，泡利矩阵σz）
    sigma_z = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ])
    
    # 计算期望值
    expectation = calculate_expectation_value(sigma_z, density_matrix)
    print(f"σz的期望值: {expectation.real:.4f}")
    
    # 时间演化模拟
    initial_state = np.array([1.0, 0.0])  # 初始态|0⟩
    time_points = np.linspace(0, 10, 100)  # 时间点
    
    # 定义要计算期望值的算符
    operators = {
        "sigma_z": sigma_z,
        "sigma_x": np.array([[0, 1], [1, 0]])
    }
    
    # 计算时间演化
    evolution_results = simulate_time_evolution(h, initial_state, time_points, operators)
    
    # 可视化时间演化
    visualize_time_evolution(time_points, evolution_results, "quantum_evolution.png")
    print(f"量子演化可视化已保存至 ./images/quantum_evolution.png")
    
    # 示例3：材料结构优化
    print("\n3. 材料结构优化")
    print("-" * 30)
    
    # 定义一个简单分子的初始结构（例如，水分子H2O）
    initial_positions = np.array([
        [0.0, 0.0, 0.0],    # O
        [0.8, 0.6, 0.0],    # H
        [-0.8, 0.6, 0.0]    # H
    ])
    atomic_numbers = np.array([8, 1, 1])  # O, H, H
    
    # 优化结构
    optimized_positions, optimized_energy = optimize_crystal_structure(
        initial_positions, atomic_numbers, functional='LDA', basis_set='minimal'
    )
    
    print("初始原子位置:")
    print(initial_positions)
    print("\n优化后原子位置:")
    print(optimized_positions)
    print(f"\n优化后能量: {optimized_energy:.4f} eV")
    
    print("\n凝聚态物理工具包演示完成")

if __name__ == "__main__":
    main()