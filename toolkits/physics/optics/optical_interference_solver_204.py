# Filename: optical_interference_solver.py

import numpy as np
from scipy import optimize

def calculate_thin_film_interference(n1, n2, n3, d, wavelength_range=(380, 780), incidence_angle=0):
    """
    计算薄膜干涉中反射光的增强和减弱波长。
    
    该函数基于薄膜干涉理论，计算在给定折射率和薄膜厚度条件下，
    哪些波长的光在反射时会发生增强或减弱。考虑了半波损失的情况。
    
    Parameters:
    -----------
    n1 : float
        入射介质的折射率
    n2 : float
        薄膜介质的折射率
    n3 : float
        基底介质的折射率
    d : float
        薄膜厚度，单位为纳米(nm)
    wavelength_range : tuple, optional
        需要考察的波长范围(nm)，默认为可见光范围(380, 780)
    incidence_angle : float, optional
        入射角，单位为度，默认为0度(垂直入射)
        
    Returns:
    --------
    dict
        包含以下键值对:
        - 'enhanced_wavelengths': 反射增强的波长列表(nm)
        - 'weakened_wavelengths': 反射减弱的波长列表(nm)
        - 'reflection_coefficient': 各波长对应的反射系数函数
        - 'phase_difference': 各波长对应的相位差函数
    """
    # 将入射角从度转换为弧度
    theta_i = np.radians(incidence_angle)
    
    # 计算折射角（斯涅尔定律）
    theta_t = np.arcsin(n1 * np.sin(theta_i) / n2) if theta_i != 0 else 0
    
    # 确定半波损失情况
    r12_phase_shift = 0 if n1 > n2 else np.pi  # 上表面反射相移
    r23_phase_shift = np.pi if n2 < n3 else 0  # 下表面反射相移
    total_phase_shift = r12_phase_shift - r23_phase_shift
    # print("total_phase_shift",total_phase_shift)
    # 确定干涉条件
    if total_phase_shift == 0:  # 相位差为0或2π的整数倍
        # 增强条件: 2n2*d*cos(theta_t) = m*lambda
        enhanced_condition = lambda m: 2 * n2 * d * np.cos(theta_t) / m
        # 减弱条件: 2n2*d*cos(theta_t) = (m+1/2)*lambda
        weakened_condition = lambda m: 2 * n2 * d * np.cos(theta_t) / (m + 0.5)
    else:  # 相位差为π
        # 增强条件: 2n2*d*cos(theta_t) = (m+1/2)*lambda
        enhanced_condition = lambda m: 2 * n2 * d * np.cos(theta_t) / (m + 0.5)
        # 减弱条件: 2n2*d*cos(theta_t) = m*lambda
        weakened_condition = lambda m: 2 * n2 * d * np.cos(theta_t) / m
    # print("enhanced_condition",enhanced_condition)
    # 计算可见光范围内的增强波长
    enhanced_wavelengths = []
    m = 0
    while True:
        if m == 0 and total_phase_shift == 0:
            m += 1
            continue  # 避免除以零
        wavelength = enhanced_condition(m)
        if wavelength_range[0] <= wavelength <= wavelength_range[1]:
            enhanced_wavelengths.append(wavelength)
        elif wavelength < wavelength_range[0]:
            break
        m += 1
    
    # 计算可见光范围内的减弱波长
    weakened_wavelengths = []
    m = 1  # 从1开始避免除以零
    while True:
        wavelength = weakened_condition(m)
        if wavelength_range[0] <= wavelength <= wavelength_range[1]:
            weakened_wavelengths.append(wavelength)
        elif wavelength < wavelength_range[0]:
            break
        m += 1
    
    # 定义反射系数计算函数
    def reflection_coefficient(wavelength):
        """计算给定波长的反射系数"""
        k0 = 2 * np.pi / wavelength
        # 计算界面反射系数
        r12 = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
        theta_t2 = np.arcsin(n2 * np.sin(theta_t) / n3) if theta_t != 0 else 0
        r23 = (n2 * np.cos(theta_t) - n3 * np.cos(theta_t2)) / (n2 * np.cos(theta_t) + n3 * np.cos(theta_t2))
        
        # 计算相位差
        delta = 2 * k0 * n2 * d * np.cos(theta_t)
        
        # 计算总反射系数
        r = (r12 + r23 * np.exp(1j * delta)) / (1 + r12 * r23 * np.exp(1j * delta))
        return np.abs(r)**2
    
    # 定义相位差计算函数
    def phase_difference(wavelength):
        """计算给定波长的相位差"""
        k0 = 2 * np.pi / wavelength
        delta = 2 * k0 * n2 * d * np.cos(theta_t)
        return delta
    return {
        'enhanced_wavelengths': enhanced_wavelengths,
        'weakened_wavelengths': weakened_wavelengths,
        'reflection_coefficient': reflection_coefficient,
        'phase_difference': phase_difference
    }

def find_extrema_wavelengths(n1, n2, n3, d, wavelength_range=(380, 780), num_points=1000):
    """
    通过数值方法找出反射系数的极值点，确定反射增强和减弱的精确波长。
    
    Parameters:
    -----------
    n1, n2, n3 : float
        三种介质的折射率
    d : float
        薄膜厚度(nm)
    wavelength_range : tuple, optional
        波长范围(nm)
    num_points : int, optional
        用于初始扫描的点数
        
    Returns:
    --------
    dict
        包含反射增强和减弱的精确波长
    """
    # 获取干涉计算结果
    result = calculate_thin_film_interference(n1, n2, n3, d, wavelength_range)
    reflection_func = result['reflection_coefficient']
    print("reflection_func",reflection_func)
    # 创建波长数组进行扫描
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    reflections = np.array([reflection_func(wl) for wl in wavelengths])
    
    # 找出局部极大值点（反射增强）
    max_indices = []
    for i in range(1, len(reflections)-1):
        if reflections[i] > reflections[i-1] and reflections[i] > reflections[i+1]:
            max_indices.append(i)
    
    # 找出局部极小值点（反射减弱）
    min_indices = []
    for i in range(1, len(reflections)-1):
        if reflections[i] < reflections[i-1] and reflections[i] < reflections[i+1]:
            min_indices.append(i)
    
    # 使用优化方法精确定位极值点
    enhanced_wavelengths = []
    for idx in max_indices:
        initial_guess = wavelengths[idx]
        # 定义负反射系数函数（用于求最大值）
        neg_reflection = lambda wl: -reflection_func(wl)
        result = optimize.minimize(neg_reflection, initial_guess, bounds=[(wavelength_range[0], wavelength_range[1])])
        if result.success:
            enhanced_wavelengths.append(result.x[0])
    
    weakened_wavelengths = []
    for idx in min_indices:
        initial_guess = wavelengths[idx]
        result = optimize.minimize(reflection_func, initial_guess, bounds=[(wavelength_range[0], wavelength_range[1])])
        if result.success:
            weakened_wavelengths.append(result.x[0])
    
    return {
        'enhanced_wavelengths': sorted(enhanced_wavelengths),
        'weakened_wavelengths': sorted(weakened_wavelengths)
    }

def wavelength_to_color_name(wavelength):
    """
    将波长转换为对应的颜色名称。
    
    Parameters:
    -----------
    wavelength : float
        光的波长(nm)
        
    Returns:
    --------
    str
        对应的颜色名称
    """
    if 380 <= wavelength < 450:
        return "紫色"
    elif 450 <= wavelength < 485:
        return "蓝色"
    elif 485 <= wavelength < 500:
        return "青色"
    elif 500 <= wavelength < 565:
        return "绿色"
    elif 565 <= wavelength < 590:
        return "黄色"
    elif 590 <= wavelength < 625:
        return "橙色"
    elif 625 <= wavelength < 780:
        return "红色"
    else:
        return "不在可见光范围"

def main():
    """
    主函数：演示如何使用工具函数求解薄膜干涉问题
    """
    # 问题参数定义
    n1 = 1.5  # 假设入射介质折射率为1.5（大于n2）
    n2 = 1.4  # 薄膜折射率
    n3 = 1.6  # 假设基底折射率为1.6（大于n2）
    d = 350   # 薄膜厚度(nm)
    
    # 计算反射光中得到加强的波长
    result = calculate_thin_film_interference(n1, n2, n3, d)
    enhanced_wavelengths = result['enhanced_wavelengths']
    
    # 输出结果
    print(f"薄膜参数: n1={n1}, n2={n2}, n3={n3}, d={d} nm")
    print("\n反射光中得到加强的可见光波长:")
    for wl in enhanced_wavelengths:
        color = wavelength_to_color_name(wl)
        print(f"  {wl:.2f} nm ({color})")
    
    # 使用数值方法验证结果
    numerical_result = find_extrema_wavelengths(n1, n2, n3, d)
    print("\n通过数值方法确定的反射增强波长:")
    for wl in numerical_result['enhanced_wavelengths']:
        color = wavelength_to_color_name(wl)
        print(f"  {wl:.2f} nm ({color})")

if __name__ == "__main__":
    main()