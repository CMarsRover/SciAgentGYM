# Filename: optical_wave_propagation.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize
from scipy.signal import find_peaks

def calculate_reflection_coefficient(n1, n2, theta_i=0):
    """
    计算光波在两种介质界面上的反射系数（振幅反射率）
    
    基于菲涅耳方程计算光波在两种不同折射率介质界面上的反射系数。
    对于垂直入射(θ=0)，TE和TM偏振的反射系数相同。
    
    Parameters:
    -----------
    n1 : float
        入射介质的折射率
    n2 : float
        透射介质的折射率
    theta_i : float, optional
        入射角（弧度），默认为0（垂直入射）
        
    Returns:
    --------
    float
        反射系数r，表示反射波振幅与入射波振幅之比
    """
    # 使用斯涅尔定律计算透射角
    if np.sin(theta_i) > n2/n1:  # 全反射情况
        return 1.0
    
    theta_t = np.arcsin(n1 * np.sin(theta_i) / n2)
    
    # 计算垂直偏振(TE)的反射系数
    r_TE = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
    
    # 计算平行偏振(TM)的反射系数
    r_TM = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
    
    # 对于非偏振光，返回两种偏振的平均值
    return (r_TE**2 + r_TM**2) / 2

def calculate_power_reflection(n1, n2, theta_i=0):
    """
    计算光波在两种介质界面上的功率反射率
    
    功率反射率等于反射系数的平方，表示反射功率与入射功率之比。
    
    Parameters:
    -----------
    n1 : float
        入射介质的折射率
    n2 : float
        透射介质的折射率
    theta_i : float, optional
        入射角（弧度），默认为0（垂直入射）
        
    Returns:
    --------
    float
        功率反射率R，表示反射功率与入射功率之比
    """
    # 对于垂直入射，反射系数简化为
    if theta_i == 0:
        r = (n1 - n2) / (n1 + n2)
        return r**2
    else:
        # 使用更通用的计算方法
        r = calculate_reflection_coefficient(n1, n2, theta_i)
        return r

def angular_spectrum_propagation(field, dx, dy, wavelength, z):
    """
    使用角谱方法计算光场在自由空间中的传播
    
    基于傅里叶光学原理，通过频域中的传递函数计算光场传播。
    
    Parameters:
    -----------
    field : ndarray
        输入光场的复振幅分布
    dx, dy : float
        x和y方向的空间采样间隔（米）
    wavelength : float
        光的波长（米）
    z : float
        传播距离（米）
        
    Returns:
    --------
    ndarray
        传播后的光场复振幅分布
    """
    # 获取场的尺寸
    ny, nx = field.shape
    
    # 计算频域采样间隔
    dfx = 1 / (nx * dx)
    dfy = 1 / (ny * dy)
    
    # 创建频域网格
    fx = np.arange(-nx/2, nx/2) * dfx
    fy = np.arange(-ny/2, ny/2) * dfy
    FX, FY = np.meshgrid(fx, fy)
    
    # 计算传递函数
    k = 2 * np.pi / wavelength  # 波数
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength*FX)**2 - (wavelength*FY)**2))
    
    # 对于不满足傍轴近似的频率分量，设置为0
    H[np.real(1 - (wavelength*FX)**2 - (wavelength*FY)**2) < 0] = 0
    
    # 执行角谱传播
    F = fftshift(fft2(field))
    F_prop = F * H
    field_prop = ifft2(ifftshift(F_prop))
    
    return field_prop

def optimize_optical_system(initial_params, merit_function, bounds=None, method='Nelder-Mead'):
    """
    优化光学系统参数以最小化给定的评价函数
    
    使用数值优化方法找到光学系统的最佳参数配置。
    
    Parameters:
    -----------
    initial_params : array_like
        初始参数值
    merit_function : callable
        评价函数，接受参数并返回需要最小化的标量值
    bounds : sequence or None, optional
        参数的边界约束，格式为[(min1, max1), (min2, max2), ...]
    method : str, optional
        优化算法，默认为'Nelder-Mead'
        
    Returns:
    --------
    OptimizeResult
        优化结果，包含最优参数和其他信息
    """
    result = minimize(merit_function, initial_params, method=method, bounds=bounds)
    return result

def analyze_spectrum(wavelengths, intensities, prominence=0.1, width=None, height=None):
    """
    分析光谱数据，识别峰值并提取特征
    
    对光谱数据进行峰值检测和特征提取，用于光谱分析。
    
    Parameters:
    -----------
    wavelengths : array_like
        波长数据（纳米）
    intensities : array_like
        对应的强度数据
    prominence : float, optional
        峰值检测的突出度阈值
    width : float or None, optional
        峰值检测的宽度约束
    height : float or None, optional
        峰值检测的高度阈值
        
    Returns:
    --------
    dict
        包含峰值位置、高度、宽度等特征的字典
    """
    # 归一化强度数据
    norm_intensities = intensities / np.max(intensities)
    
    # 查找峰值
    peaks, properties = find_peaks(norm_intensities, 
                                  prominence=prominence,
                                  width=width,
                                  height=height)
    
    # 提取峰值特征
    peak_wavelengths = wavelengths[peaks]
    peak_intensities = norm_intensities[peaks]
    peak_widths = properties.get('widths', np.zeros_like(peaks))
    peak_prominences = properties.get('prominences', np.zeros_like(peaks))
    
    # 返回特征字典
    return {
        'peak_indices': peaks,
        'peak_wavelengths': peak_wavelengths,
        'peak_intensities': peak_intensities,
        'peak_widths': peak_widths,
        'peak_prominences': peak_prominences
    }

def calculate_prism_reflection(epsilon_r, incident_angle=0):
    """
    计算棱镜表面的反射功率比
    
    计算光波在棱镜表面的反射功率与入射功率之比，适用于各种棱镜几何形状。
    
    Parameters:
    -----------
    epsilon_r : float
        棱镜材料的相对介电常数
    incident_angle : float, optional
        入射角（弧度），默认为0（垂直入射）
        
    Returns:
    --------
    float
        反射功率与入射功率之比
    """
    # 计算折射率（n = √ε_r）
    n_prism = np.sqrt(epsilon_r)
    n_air = 1.0
    
    # 计算功率反射率
    power_ratio = calculate_power_reflection(n_air, n_prism, incident_angle)
    
    return power_ratio

def main():
    """
    主函数：演示如何使用工具函数求解当前问题
    
    当平面波向等腰直角玻璃棱镜的底边垂直投射时，
    若玻璃的相对介电常数ε_r=4，计算反射功率W^r与入射功率W^i之比。
    """
    # 问题参数
    epsilon_r = 4.0  # 玻璃的相对介电常数
    incident_angle = 0  # 垂直入射（弧度）
    
    # 计算反射功率比
    reflection_ratio = calculate_prism_reflection(epsilon_r, incident_angle)
    
    # 输出结果
    print(f"问题：当平面波向等腰直角玻璃棱镜的底边垂直投射时，若玻璃的相对介电常数ε_r={epsilon_r}，")
    print(f"计算反射功率W^r与入射功率W^i之比。")
    print("\n计算过程：")
    print(f"1. 玻璃的折射率 n = √ε_r = √{epsilon_r} = {np.sqrt(epsilon_r)}")
    print(f"2. 空气的折射率 n_air = 1.0")
    print(f"3. 对于垂直入射，反射系数 r = (n_air - n_glass)/(n_air + n_glass) = (1 - {np.sqrt(epsilon_r)})/(1 + {np.sqrt(epsilon_r)})")
    print(f"4. 功率反射率 R = |r|² = {reflection_ratio}")
    print("\n结果：")
    print(f"反射功率与入射功率之比 W^r/W^i = {reflection_ratio:.6f}")
    
    # 可选：绘制不同介电常数下的反射率变化
    epsilon_values = np.linspace(1, 10, 100)
    reflection_values = [calculate_prism_reflection(eps) for eps in epsilon_values]
    
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.plot(epsilon_values, reflection_values)
    plt.axvline(x=epsilon_r, color='r', linestyle='--', label=f'ε_r = {epsilon_r}')
    plt.axhline(y=reflection_ratio, color='g', linestyle='--', label=f'反射率 = {reflection_ratio:.4f}')
    plt.xlabel('相对介电常数 ε_r')
    plt.ylabel('反射功率比 W^r/W^i')
    plt.title('不同介电常数下的反射功率比')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()