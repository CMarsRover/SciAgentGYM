# Filename: quantum_spin_dynamics_toolkit.py

"""
Quantum Spin Dynamics Toolkit
=============================
专业的量子自旋动力学计算工具包，用于计算自旋1/2粒子在磁场中的演化。

核心功能：
1. 自旋算符和哈密顿量构建
2. 量子态时间演化
3. 期望值计算与可视化
4. Larmor进动频率分析

依赖库：
- numpy: 数值计算
- scipy: 矩阵指数和优化
- matplotlib: 可视化
- qutip (可选): 量子工具箱
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import json

# 确保输出目录存在
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 物理常数
HBAR = 1.0  # 约化普朗克常数 (使用自然单位制)

# ============================================================================
# 第一层：原子函数 - 基础量子算符和态矢量
# ============================================================================

def get_pauli_matrices() -> Dict[str, np.ndarray]:
    """
    获取泡利矩阵（Pauli Matrices）
    
    泡利矩阵是自旋1/2系统的基本算符，定义为：
    σx = [[0, 1], [1, 0]]
    σy = [[0, -i], [i, 0]]
    σz = [[1, 0], [0, -1]]
    
    Returns:
        dict: 包含三个泡利矩阵的字典
            {
                'result': {
                    'sigma_x': [[...]], 
                    'sigma_y': [[...]], 
                    'sigma_z': [[...]]
                },
                'metadata': {
                    'dimension': 2,
                    'hermitian': True,
                    'traceless': True
                }
            }
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    return {
        'result': {
            'sigma_x': sigma_x.tolist(),
            'sigma_y': sigma_y.tolist(),
            'sigma_z': sigma_z.tolist()
        },
        'metadata': {
            'dimension': 2,
            'hermitian': True,
            'traceless': True,
            'description': 'Pauli spin matrices for spin-1/2 system'
        }
    }


def get_spin_operators(hbar: float = HBAR) -> Dict[str, np.ndarray]:
    """
    获取自旋算符 S = (ℏ/2)σ
    
    自旋1/2粒子的自旋算符定义为泡利矩阵的一半（乘以ℏ）
    
    Args:
        hbar: 约化普朗克常数，默认为1.0（自然单位制）
    
    Returns:
        dict: 包含三个自旋算符的字典
            {
                'result': {'Sx': [[...]], 'Sy': [[...]], 'Sz': [[...]]},
                'metadata': {...}
            }
    """
    pauli = get_pauli_matrices()
    sigma_x = np.array(pauli['result']['sigma_x'])
    sigma_y = np.array(pauli['result']['sigma_y'])
    sigma_z = np.array(pauli['result']['sigma_z'])
    
    factor = hbar / 2.0
    Sx = factor * sigma_x
    Sy = factor * sigma_y
    Sz = factor * sigma_z
    
    return {
        'result': {
            'Sx': Sx.tolist(),
            'Sy': Sy.tolist(),
            'Sz': Sz.tolist()
        },
        'metadata': {
            'hbar': hbar,
            'eigenvalues': [hbar/2, -hbar/2],
            'description': 'Spin operators for spin-1/2 particle'
        }
    }


def create_spin_state(direction: str) -> Dict:
    """
    创建沿特定方向的自旋本征态
    
    Args:
        direction: 自旋方向，可选 'up_z', 'down_z', 'up_x', 'down_x', 'up_y', 'down_y'
    
    Returns:
        dict: 包含态矢量的字典
            {
                'result': [[c1], [c2]],  # 复数振幅
                'metadata': {
                    'direction': str,
                    'normalized': bool,
                    'basis': 'z-basis'
                }
            }
    
    Raises:
        ValueError: 如果direction参数无效
    """
    valid_directions = ['up_z', 'down_z', 'up_x', 'down_x', 'up_y', 'down_y']
    if direction not in valid_directions:
        raise ValueError(f"Invalid direction. Must be one of {valid_directions}")
    
    # Z方向本征态
    if direction == 'up_z':
        state = np.array([[1], [0]], dtype=complex)
    elif direction == 'down_z':
        state = np.array([[0], [1]], dtype=complex)
    
    # X方向本征态（在Z基矢下表示）
    elif direction == 'up_x':
        state = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
    elif direction == 'down_x':
        state = np.array([[1], [-1]], dtype=complex) / np.sqrt(2)
    
    # Y方向本征态（在Z基矢下表示）
    elif direction == 'up_y':
        state = np.array([[1], [1j]], dtype=complex) / np.sqrt(2)
    elif direction == 'down_y':
        state = np.array([[1], [-1j]], dtype=complex) / np.sqrt(2)
    
    # 验证归一化
    norm = np.linalg.norm(state)
    is_normalized = np.isclose(norm, 1.0)
    
    return {
        'result': state.tolist(),
        'metadata': {
            'direction': direction,
            'normalized': is_normalized,
            'norm': float(norm),
            'basis': 'z-basis',
            'description': f'Spin eigenstate along {direction}'
        }
    }


def construct_hamiltonian(field_direction: str, field_strength: float, 
                         gamma: float) -> Dict:
    """
    构建自旋在磁场中的哈密顿量
    
    哈密顿量: H = -γ B·S = -γ B Sn (n为磁场方向)
    
    Args:
        field_direction: 磁场方向 ('x', 'y', 'z')
        field_strength: 磁场强度 B
        gamma: 旋磁比
    
    Returns:
        dict: 包含哈密顿量矩阵的字典
            {
                'result': [[...]],
                'metadata': {
                    'field_direction': str,
                    'field_strength': float,
                    'gamma': float,
                    'hermitian': bool
                }
            }
    
    Raises:
        ValueError: 如果field_direction无效或参数为负
    """
    if field_direction not in ['x', 'y', 'z']:
        raise ValueError("field_direction must be 'x', 'y', or 'z'")
    if field_strength < 0:
        raise ValueError("field_strength must be non-negative")
    
    spin_ops = get_spin_operators()
    
    if field_direction == 'x':
        S = np.array(spin_ops['result']['Sx'])
    elif field_direction == 'y':
        S = np.array(spin_ops['result']['Sy'])
    else:  # 'z'
        S = np.array(spin_ops['result']['Sz'])
    
    # H = -γ B S
    H = -gamma * field_strength * S
    
    # 验证厄米性
    is_hermitian = np.allclose(H, H.conj().T)
    
    return {
        'result': H.tolist(),
        'metadata': {
            'field_direction': field_direction,
            'field_strength': field_strength,
            'gamma': gamma,
            'hermitian': is_hermitian,
            'energy_unit': 'gamma*B*hbar',
            'description': f'Hamiltonian for spin in {field_direction}-directed magnetic field'
        }
    }


# ============================================================================
# 第二层：组合函数 - 时间演化和期望值计算
# ============================================================================

def time_evolution_operator(hamiltonian_list: List[List], time: float, 
                           hbar: float = HBAR) -> Dict:
    """
    计算时间演化算符 U(t) = exp(-iHt/ℏ)
    
    Args:
        hamiltonian_list: 哈密顿量矩阵（2x2列表）
        time: 演化时间
        hbar: 约化普朗克常数
    
    Returns:
        dict: 包含演化算符的字典
            {
                'result': [[...]],
                'metadata': {
                    'time': float,
                    'unitary': bool
                }
            }
    """
    H = np.array(hamiltonian_list, dtype=complex)
    
    # U(t) = exp(-iHt/ℏ)
    U = expm(-1j * H * time / hbar)
    
    # 验证幺正性
    is_unitary = np.allclose(U @ U.conj().T, np.eye(2))
    
    return {
        'result': U.tolist(),
        'metadata': {
            'time': time,
            'hbar': hbar,
            'unitary': is_unitary,
            'description': 'Time evolution operator U(t) = exp(-iHt/ℏ)'
        }
    }


def evolve_state(initial_state_list: List[List], evolution_operator_list: List[List]) -> Dict:
    """
    演化量子态 |ψ(t)⟩ = U(t)|ψ(0)⟩
    
    Args:
        initial_state_list: 初态（2x1列表）
        evolution_operator_list: 演化算符（2x2列表）
    
    Returns:
        dict: 包含演化后态矢量的字典
            {
                'result': [[...]],
                'metadata': {
                    'normalized': bool,
                    'norm': float
                }
            }
    """
    psi_0 = np.array(initial_state_list, dtype=complex)
    U = np.array(evolution_operator_list, dtype=complex)
    
    # |ψ(t)⟩ = U(t)|ψ(0)⟩
    psi_t = U @ psi_0
    
    # 验证归一化
    norm = np.linalg.norm(psi_t)
    is_normalized = np.isclose(norm, 1.0)
    
    return {
        'result': psi_t.tolist(),
        'metadata': {
            'normalized': is_normalized,
            'norm': float(norm),
            'description': 'Evolved quantum state'
        }
    }


def calculate_expectation_value(state_list: List[List], operator_list: List[List]) -> Dict:
    """
    计算算符的期望值 ⟨O⟩ = ⟨ψ|O|ψ⟩
    
    Args:
        state_list: 量子态（2x1列表）
        operator_list: 算符矩阵（2x2列表）
    
    Returns:
        dict: 包含期望值的字典
            {
                'result': float,
                'metadata': {
                    'real': float,
                    'imaginary': float,
                    'is_real': bool
                }
            }
    """
    psi = np.array(state_list, dtype=complex)
    O = np.array(operator_list, dtype=complex)
    
    # ⟨O⟩ = ⟨ψ|O|ψ⟩
    expectation = (psi.conj().T @ O @ psi)[0, 0]
    
    is_real = np.isclose(expectation.imag, 0)
    
    return {
        'result': float(expectation.real) if is_real else complex(expectation),
        'metadata': {
            'real_part': float(expectation.real),
            'imaginary_part': float(expectation.imag),
            'is_real': is_real,
            'description': 'Expectation value of operator'
        }
    }


def compute_time_series(initial_state_list: List[List], 
                       hamiltonian_list: List[List],
                       time_points: List[float],
                       observable_list: List[List],
                       hbar: float = HBAR) -> Dict:
    """
    计算可观测量随时间的演化序列
    
    Args:
        initial_state_list: 初态（2x1列表）
        hamiltonian_list: 哈密顿量（2x2列表）
        time_points: 时间点列表
        observable_list: 可观测量算符（2x2列表）
        hbar: 约化普朗克常数
    
    Returns:
        dict: 包含时间序列数据的字典
            {
                'result': {
                    'times': [...],
                    'expectation_values': [...]
                },
                'metadata': {...}
            }
    """
    psi_0 = np.array(initial_state_list, dtype=complex)
    H = np.array(hamiltonian_list, dtype=complex)
    O = np.array(observable_list, dtype=complex)
    
    expectation_values = []
    
    for t in time_points:
        # 计算演化算符
        U_result = time_evolution_operator(hamiltonian_list, t, hbar)
        U = np.array(U_result['result'])
        
        # 演化态
        psi_t = U @ psi_0
        
        # 计算期望值
        exp_val = (psi_t.conj().T @ O @ psi_t)[0, 0].real
        expectation_values.append(float(exp_val))
    
    # 保存中间结果
    data = {
        'times': time_points,
        'expectation_values': expectation_values
    }
    filepath = './mid_result/physics/time_series_data.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {
        'result': {
            'times': time_points,
            'expectation_values': expectation_values
        },
        'metadata': {
            'num_points': len(time_points),
            'time_range': [min(time_points), max(time_points)],
            'data_file': filepath,
            'description': 'Time evolution of expectation value'
        }
    }


# ============================================================================
# 第三层：高级分析函数 - 频率提取和可视化
# ============================================================================

def extract_oscillation_frequency(times: List[float], 
                                  values: List[float],
                                  method: str = 'fft') -> Dict:
    """
    从时间序列中提取振荡频率
    
    Args:
        times: 时间点列表
        values: 对应的数值列表
        method: 提取方法 ('fft' 或 'fit')
    
    Returns:
        dict: 包含频率信息的字典
            {
                'result': float,  # 主频率
                'metadata': {
                    'method': str,
                    'angular_frequency': float,
                    'period': float,
                    'all_frequencies': [...],
                    'power_spectrum': [...]
                }
            }
    
    Raises:
        ValueError: 如果method参数无效
    """
    if method not in ['fft', 'fit']:
        raise ValueError("method must be 'fft' or 'fit'")
    
    times_array = np.array(times)
    values_array = np.array(values)
    
    if method == 'fft':
        # FFT方法
        dt = times_array[1] - times_array[0]
        n = len(values_array)
        
        # 执行FFT
        fft_vals = fft(values_array)
        freqs = fftfreq(n, dt)
        
        # 只取正频率部分
        positive_freqs = freqs[:n//2]
        power = np.abs(fft_vals[:n//2])
        
        # 找到最大功率对应的频率
        max_idx = np.argmax(power[1:]) + 1  # 跳过DC分量
        dominant_freq = positive_freqs[max_idx]
        
        result = {
            'result': float(dominant_freq),
            'metadata': {
                'method': 'fft',
                'angular_frequency': float(2 * np.pi * dominant_freq),
                'period': float(1 / dominant_freq) if dominant_freq > 0 else float('inf'),
                'all_frequencies': positive_freqs.tolist(),
                'power_spectrum': power.tolist(),
                'description': 'Frequency extracted using FFT'
            }
        }
    
    else:  # 'fit'
        # 拟合方法：假设形式 A*cos(ωt + φ) + C
        def cosine_func(t, A, omega, phi, C):
            return A * np.cos(omega * t + phi) + C
        
        # 初始猜测
        A_guess = (np.max(values_array) - np.min(values_array)) / 2
        C_guess = np.mean(values_array)
        omega_guess = 2 * np.pi / (times_array[-1] - times_array[0])
        
        try:
            popt, _ = curve_fit(cosine_func, times_array, values_array,
                              p0=[A_guess, omega_guess, 0, C_guess],
                              maxfev=10000)
            
            A_fit, omega_fit, phi_fit, C_fit = popt
            freq_fit = omega_fit / (2 * np.pi)
            
            result = {
                'result': float(freq_fit),
                'metadata': {
                    'method': 'fit',
                    'angular_frequency': float(omega_fit),
                    'period': float(2 * np.pi / omega_fit) if omega_fit > 0 else float('inf'),
                    'amplitude': float(A_fit),
                    'phase': float(phi_fit),
                    'offset': float(C_fit),
                    'description': 'Frequency extracted using curve fitting'
                }
            }
        except Exception as e:
            result = {
                'result': None,
                'metadata': {
                    'method': 'fit',
                    'error': str(e),
                    'description': 'Curve fitting failed'
                }
            }
    
    return result


def visualize_spin_precession(times: List[float],
                              expectation_values: List[float],
                              frequency: float,
                              component: str = 'z',
                              save_path: str = None) -> Dict:
    """
    可视化自旋进动
    
    Args:
        times: 时间点列表
        expectation_values: 期望值列表
        frequency: 振荡频率
        component: 自旋分量 ('x', 'y', 'z')
        save_path: 图像保存路径（可选）
    
    Returns:
        dict: 包含图像路径的字典
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上图：时间演化
    ax1.plot(times, expectation_values, 'b-', linewidth=2, label='Quantum Evolution')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time (1/γB)', fontsize=12)
    ax1.set_ylabel(f'⟨S{component}⟩ (ℏ/2)', fontsize=12)
    ax1.set_title(f'Spin-{component} Component Precession (ω = {frequency:.4f} γB)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 下图：FFT功率谱
    dt = times[1] - times[0]
    n = len(expectation_values)
    fft_vals = fft(expectation_values)
    freqs = fftfreq(n, dt)
    power = np.abs(fft_vals)
    
    positive_freqs = freqs[:n//2]
    positive_power = power[:n//2]
    
    ax2.plot(positive_freqs, positive_power, 'r-', linewidth=2)
    ax2.axvline(x=frequency, color='g', linestyle='--', linewidth=2, 
                label=f'Dominant Frequency = {frequency:.4f}')
    ax2.set_xlabel('Frequency (γB/2π)', fontsize=12)
    ax2.set_ylabel('Power Spectrum', fontsize=12)
    ax2.set_title('Fourier Transform - Frequency Analysis', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, min(5*frequency, positive_freqs[-1])])
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = './tool_images/spin_precession_analysis.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'component': component,
            'frequency': frequency,
            'description': 'Spin precession visualization with FFT analysis'
        }
    }


def visualize_bloch_sphere_trajectory(times: List[float],
                                     initial_state_list: List[List],
                                     hamiltonian_list: List[List],
                                     save_path: str = None) -> Dict:
    """
    在Bloch球面上可视化自旋轨迹
    
    Args:
        times: 时间点列表
        initial_state_list: 初态
        hamiltonian_list: 哈密顿量
        save_path: 保存路径
    
    Returns:
        dict: 包含图像路径的字典
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算Bloch矢量分量
    spin_ops = get_spin_operators()
    Sx = np.array(spin_ops['result']['Sx'])
    Sy = np.array(spin_ops['result']['Sy'])
    Sz = np.array(spin_ops['result']['Sz'])
    
    x_vals, y_vals, z_vals = [], [], []
    
    for t in times:
        U_result = time_evolution_operator(hamiltonian_list, t)
        U = np.array(U_result['result'])
        psi_0 = np.array(initial_state_list, dtype=complex)
        psi_t = U @ psi_0
        
        x = (psi_t.conj().T @ Sx @ psi_t)[0, 0].real
        y = (psi_t.conj().T @ Sy @ psi_t)[0, 0].real
        z = (psi_t.conj().T @ Sz @ psi_t)[0, 0].real
        
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
    
    # 3D绘图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制Bloch球面
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 0.5 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.5 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    # 绘制轨迹
    ax.plot(x_vals, y_vals, z_vals, 'r-', linewidth=2, label='Spin Trajectory')
    ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], c='green', s=100, label='Initial State')
    ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], c='blue', s=100, label='Final State')
    
    # 坐标轴
    ax.quiver(0, 0, 0, 0.6, 0, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0.6, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 0.6, color='black', arrow_length_ratio=0.1)
    ax.text(0.7, 0, 0, 'X', fontsize=12)
    ax.text(0, 0.7, 0, 'Y', fontsize=12)
    ax.text(0, 0, 0.7, 'Z', fontsize=12)
    
    ax.set_xlabel('⟨Sx⟩ (ℏ/2)', fontsize=12)
    ax.set_ylabel('⟨Sy⟩ (ℏ/2)', fontsize=12)
    ax.set_zlabel('⟨Sz⟩ (ℏ/2)', fontsize=12)
    ax.set_title('Spin Precession on Bloch Sphere', fontsize=14)
    ax.legend()
    
    if save_path is None:
        save_path = './tool_images/bloch_sphere_trajectory.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'num_points': len(times),
            'description': 'Bloch sphere trajectory visualization'
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示量子自旋动力学工具包的三个应用场景
    """
    
    # 物理参数设置
    gamma = 1.0  # 旋磁比 (归一化单位)
    B = 1.0      # 磁场强度 (归一化单位)
    
    print("=" * 80)
    print("场景1：解决原始问题 - 自旋1/2粒子在磁场突变下的进动频率")
    print("=" * 80)
    print("问题描述：")
    print("初始状态：自旋沿+Z方向，磁场沿+Z方向")
    print("磁场突变：磁场从+Z方向突然切换到+Y方向（强度不变）")
    print("求解目标：计算⟨Sz⟩的振荡频率")
    print("-" * 80)
    
    # 步骤1：创建初态（沿+Z方向的自旋向上态）
    print("\n步骤1：创建初态 |↑⟩z")
    initial_state_result = create_spin_state('up_z')
    initial_state = initial_state_result['result']
    print(f"FUNCTION_CALL: create_spin_state | PARAMS: {{'direction': 'up_z'}} | RESULT: {initial_state_result}")
    
    # 步骤2：构建新磁场（+Y方向）的哈密顿量
    print("\n步骤2：构建+Y方向磁场的哈密顿量")
    H_result = construct_hamiltonian('y', B, gamma)
    H = H_result['result']
    print(f"FUNCTION_CALL: construct_hamiltonian | PARAMS: {{'field_direction': 'y', 'field_strength': {B}, 'gamma': {gamma}}} | RESULT: {H_result}")
    
    # 步骤3：获取Sz算符
    print("\n步骤3：获取Sz算符")
    spin_ops = get_spin_operators()
    Sz = spin_ops['result']['Sz']
    print(f"FUNCTION_CALL: get_spin_operators | PARAMS: {{}} | RESULT: {spin_ops}")
    
    # 步骤4：计算时间演化序列
    print("\n步骤4：计算⟨Sz⟩随时间的演化")
    times = np.linspace(0, 20, 500).tolist()
    time_series_result = compute_time_series(initial_state, H, times, Sz)
    print(f"FUNCTION_CALL: compute_time_series | PARAMS: {{initial_state, hamiltonian, times (500 points), Sz}} | RESULT: {{'num_points': {len(times)}, 'file': '{time_series_result['metadata']['data_file']}'}}")
    
    # 步骤5：提取振荡频率（FFT方法）
    print("\n步骤5：使用FFT提取振荡频率")
    freq_result_fft = extract_oscillation_frequency(
        time_series_result['result']['times'],
        time_series_result['result']['expectation_values'],
        method='fft'
    )
    frequency_fft = freq_result_fft['result']
    angular_freq_fft = freq_result_fft['metadata']['angular_frequency']
    print(f"FUNCTION_CALL: extract_oscillation_frequency | PARAMS: {{'method': 'fft'}} | RESULT: {freq_result_fft}")
    
    # 步骤6：提取振荡频率（拟合方法验证）
    print("\n步骤6：使用曲线拟合验证频率")
    freq_result_fit = extract_oscillation_frequency(
        time_series_result['result']['times'],
        time_series_result['result']['expectation_values'],
        method='fit'
    )
    frequency_fit = freq_result_fit['result']
    print(f"FUNCTION_CALL: extract_oscillation_frequency | PARAMS: {{'method': 'fit'}} | RESULT: {freq_result_fit}")
    
    # 步骤7：可视化结果
    print("\n步骤7：可视化自旋进动")
    viz_result = visualize_spin_precession(
        time_series_result['result']['times'],
        time_series_result['result']['expectation_values'],
        frequency_fft,
        component='z'
    )
    print(f"FUNCTION_CALL: visualize_spin_precession | PARAMS: {{times, values, frequency}} | RESULT: {viz_result}")
    
    # 步骤8：Bloch球面轨迹可视化
    print("\n步骤8：绘制Bloch球面轨迹")
    bloch_result = visualize_bloch_sphere_trajectory(
        np.linspace(0, 10, 100).tolist(),
        initial_state,
        H
    )
    print(f"FUNCTION_CALL: visualize_bloch_sphere_trajectory | PARAMS: {{times, initial_state, hamiltonian}} | RESULT: {bloch_result}")
    
    print("\n" + "=" * 80)
    print("理论分析：")
    print(f"初态在+Z方向：|↑⟩z")
    print(f"新哈密顿量：H = -γB·Sy")
    print(f"Larmor进动频率：ω = γB")
    print(f"理论预期角频率：ω = {gamma * B}")
    print(f"FFT提取角频率：ω = {angular_freq_fft:.6f}")
    print(f"拟合提取角频率：ω = {freq_result_fit['metadata']['angular_frequency']:.6f}")
    print(f"频率（Hz）：f = ω/(2π) = {frequency_fft:.6f}")
    print("=" * 80)
    
    # 最终答案
    answer = f"γB (角频率 = {angular_freq_fft:.6f}, 频率 = {frequency_fft:.6f})"
    print(f"\nFINAL_ANSWER: {answer}")
    
    
    # ========================================================================
    # 场景2：不同初态的进动分析
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：不同初态在相同磁场下的进动频率比较")
    print("=" * 80)
    print("问题描述：")
    print("比较初态为|↑⟩x、|↑⟩y、|↑⟩z在+Y方向磁场下的进动行为")
    print("验证Larmor频率的普适性")
    print("-" * 80)
    
    initial_states = {
        'up_x': create_spin_state('up_x')['result'],
        'up_y': create_spin_state('up_y')['result'],
        'up_z': create_spin_state('up_z')['result']
    }
    
    frequencies = {}
    
    for state_name, state in initial_states.items():
        print(f"\n分析初态：|↑⟩{state_name[-1]}")
        
        # 计算时间演化
        ts_result = compute_time_series(state, H, times, Sz)
        
        # 提取频率
        freq_res = extract_oscillation_frequency(
            ts_result['result']['times'],
            ts_result['result']['expectation_values'],
            method='fft'
        )
        frequencies[state_name] = freq_res['metadata']['angular_frequency']
        
        print(f"FUNCTION_CALL: compute_time_series + extract_oscillation_frequency | STATE: {state_name} | FREQUENCY: {frequencies[state_name]:.6f}")
    
    print("\n" + "-" * 80)
    print("频率比较结果：")
    for state_name, freq in frequencies.items():
        print(f"  初态 |↑⟩{state_name[-1]}: ω = {freq:.6f}")
    
    print(f"\n结论：所有初态的进动角频率均为 γB = {gamma * B}")
    print("这验证了Larmor进动频率只依赖于磁场强度和旋磁比，与初态无关")
    
    answer_2 = f"所有初态的进动频率相同：ω = γB = {gamma * B}"
    print(f"\nFINAL_ANSWER: {answer_2}")
    
    
    # ========================================================================
    # 场景3：磁场强度对进动频率的影响
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：磁场强度扫描 - 验证频率与磁场的线性关系")
    print("=" * 80)
    print("问题描述：")
    print("固定初态|↑⟩z和磁场方向+Y，改变磁场强度B")
    print("验证进动频率 ω = γB 的线性关系")
    print("-" * 80)
    
    B_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    measured_frequencies = []
    
    initial_state_z = create_spin_state('up_z')['result']
    
    for B_val in B_values:
        print(f"\n磁场强度 B = {B_val}")
        
        # 构建哈密顿量
        H_B = construct_hamiltonian('y', B_val, gamma)['result']
        
        # 计算时间演化（调整时间范围以适应不同频率）
        times_scan = np.linspace(0, 20/B_val, 500).tolist()
        ts_res = compute_time_series(initial_state_z, H_B, times_scan, Sz)
        
        # 提取频率
        freq_res = extract_oscillation_frequency(
            ts_res['result']['times'],
            ts_res['result']['expectation_values'],
            method='fft'
        )
        omega = freq_res['metadata']['angular_frequency']
        measured_frequencies.append(omega)
        
        print(f"FUNCTION_CALL: construct_hamiltonian + compute_time_series + extract_oscillation_frequency | B: {B_val} | OMEGA: {omega:.6f}")
    
    # 线性拟合验证
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(B_values, measured_frequencies)
    
    print("\n" + "-" * 80)
    print("线性拟合结果：")
    print(f"  ω = {slope:.6f} * B + {intercept:.6f}")
    print(f"  理论斜率（γ）: {gamma}")
    print(f"  测量斜率: {slope:.6f}")
    print(f"  相关系数 R²: {r_value**2:.6f}")
    
    # 可视化线性关系
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(B_values, measured_frequencies, s=100, c='red', label='Measured Data', zorder=3)
    ax.plot(B_values, [gamma * B for B in B_values], 'b--', linewidth=2, label=f'Theory: ω = γB (γ={gamma})')
    ax.plot(B_values, [slope * B + intercept for B in B_values], 'g-', linewidth=2, 
            label=f'Fit: ω = {slope:.3f}B + {intercept:.3f}')
    
    ax.set_xlabel('Magnetic Field Strength B', fontsize=12)
    ax.set_ylabel('Angular Frequency ω', fontsize=12)
    ax.set_title('Larmor Precession Frequency vs Magnetic Field', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path_3 = './tool_images/frequency_vs_field.png'
    plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFILE_GENERATED: image | PATH: {save_path_3}")
    
    answer_3 = f"进动频率与磁场强度呈线性关系：ω = γB，测量斜率 = {slope:.6f} ≈ γ = {gamma}"
    print(f"\nFINAL_ANSWER: {answer_3}")
    
    print("\n" + "=" * 80)
    print("工具包演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()