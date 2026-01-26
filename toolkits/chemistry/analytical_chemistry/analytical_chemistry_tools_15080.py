# Filename: analytical_chemistry_tools.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def _save_figure(fig, filename):
    """
    保存图像到 ./images 目录下。
    filename: 文件名，例如 'horwitz_trumpet.png'
    """
    images_dir = os.path.join("./images")
    os.makedirs(images_dir, exist_ok=True)
    save_path = os.path.join(images_dir, filename)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')

def horwitz_trumpet(concentration, plateau_level=1e-7):
    """
    Calculate the Horwitz trumpet relative standard deviation for interlaboratory results.
    
    The Horwitz trumpet describes the relationship between analyte concentration and 
    expected relative standard deviation in analytical measurements across different laboratories.
    
    Parameters:
    -----------
    concentration : float or numpy.ndarray
        Concentration of analyte in g/g (weight fraction)
    plateau_level : float, optional
        Concentration below which the RSD plateaus, default is 1e-7 g/g
        
    Returns:
    --------
    float or numpy.ndarray
        Predicted interlaboratory relative standard deviation (%)
    
    References:
    -----------
    Horwitz, W. (1982). Evaluation of Analytical Methods Used for Regulation of Foods and Drugs.
    Analytical Chemistry, 54, 67A.
    
    Thompson, M. (2013). An Emergent Optimum Precision in Chemical Measurement at Low Concentrations.
    Analytical Methods, 5, 4518.
    """
    concentration = np.asarray(concentration)
    is_scalar = concentration.ndim == 0
    if is_scalar:
        concentration = concentration.reshape(1)
    
    # Apply the Horwitz equation: RSD = 2^(1-0.5*log10(C))
    # where C is concentration in g/g
    rsd = 2**(1 - 0.5 * np.log10(concentration))
    
    # Apply plateau for very low concentrations (Thompson modification)
    mask = concentration < plateau_level
    if np.any(mask):
        plateau_rsd = 2**(1 - 0.5 * np.log10(plateau_level))
        rsd[mask] = plateau_rsd
    
    # Horwitz公式给出的即为“百分数值”（例如1wt%对应4，即4%），无需再乘以100
    result = rsd
    
    # Return scalar if input was scalar
    if is_scalar:
        return result[0]
    return result

def intra_laboratory_rsd(interlaboratory_rsd, factor=0.6):
    """
    Estimate the within-laboratory RSD based on interlaboratory RSD.
    
    Parameters:
    -----------
    interlaboratory_rsd : float or numpy.ndarray
        Relative standard deviation between laboratories (%)
    factor : float, optional
        Conversion factor, typically 0.5-0.7, default is 0.6
        
    Returns:
    --------
    float or numpy.ndarray
        Estimated within-laboratory relative standard deviation (%)
    """
    return interlaboratory_rsd * factor

def baseline_correction(spectrum, method='als', **kwargs):
    """
    Perform baseline correction on spectral data.
    
    Parameters:
    -----------
    spectrum : numpy.ndarray
        The spectral data to be corrected
    method : str, optional
        Method for baseline correction: 'als' (Asymmetric Least Squares),
        'polynomial', or 'rubberband'
    **kwargs : dict
        Additional parameters for the specific correction method
        
    Returns:
    --------
    numpy.ndarray
        Baseline-corrected spectrum
    """
    if method == 'als':
        # Default parameters
        lam = kwargs.get('lam', 1e5)
        p = kwargs.get('p', 0.001)
        niter = kwargs.get('niter', 10)
        
        L = len(spectrum)
        # Create second derivative matrix
        D = np.zeros((L-2, L))
        for i in range(L-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        w = np.ones(L)
        
        for i in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T.dot(D)
            z = np.linalg.solve(Z, w * spectrum)
            w = p * (spectrum > z) + (1 - p) * (spectrum <= z)
            
        return spectrum - z
    
    elif method == 'polynomial':
        degree = kwargs.get('degree', 3)
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, degree)
        baseline = np.polyval(coeffs, x)
        return spectrum - baseline
    
    elif method == 'rubberband':
        # Simplified rubberband method
        x = np.arange(len(spectrum))
        hull_indices = signal.argrelmin(spectrum, order=20)[0]
        
        if len(hull_indices) < 2:
            return spectrum
            
        hull_indices = np.append(0, np.append(hull_indices, len(spectrum)-1))
        baseline = np.interp(x, hull_indices, spectrum[hull_indices])
        return spectrum - baseline
    
    else:
        raise ValueError(f"Unknown baseline correction method: {method}")

def peak_detection(spectrum, x_values=None, threshold=0.1, min_distance=5, prominence=None):
    """
    Detect peaks in spectral data.
    
    Parameters:
    -----------
    spectrum : numpy.ndarray
        The spectral data for peak detection
    x_values : numpy.ndarray, optional
        The x-axis values corresponding to the spectrum
    threshold : float, optional
        Minimum intensity threshold for peak detection
    min_distance : int, optional
        Minimum distance between peaks
    prominence : float, optional
        Minimum prominence of peaks
        
    Returns:
    --------
    tuple
        (peak_indices, peak_properties) where peak_indices are the indices of detected peaks
        and peak_properties is a dictionary containing properties of the peaks
    """
    if x_values is None:
        x_values = np.arange(len(spectrum))
    
    # Find peaks using scipy.signal
    peaks, properties = signal.find_peaks(
        spectrum, 
        height=threshold,
        distance=min_distance,
        prominence=prominence
    )
    
    # Calculate additional peak properties
    peak_heights = spectrum[peaks]
    peak_positions = x_values[peaks]
    
    # Calculate peak widths
    widths, width_heights, left_ips, right_ips = signal.peak_widths(
        spectrum, peaks, rel_height=0.5
    )
    
    # Prepare results
    peak_properties = {
        'heights': peak_heights,
        'positions': peak_positions,
        'widths': widths,
        'left_bases': left_ips,
        'right_bases': right_ips
    }
    
    return peaks, peak_properties

def multivariate_analysis(data, method='pca', n_components=2, **kwargs):
    """
    Perform multivariate analysis on analytical data.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        The data matrix where rows are samples and columns are variables
    method : str, optional
        Method for multivariate analysis: 'pca' (Principal Component Analysis)
    n_components : int, optional
        Number of components to extract
    **kwargs : dict
        Additional parameters for the specific analysis method
        
    Returns:
    --------
    dict
        Results of the multivariate analysis, including scores, loadings, and explained variance
    """
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        data_values = data.values
    else:
        data_values = data
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    if method == 'pca':
        # Perform PCA
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(data_scaled)
        
        results = {
            'scores': scores,
            'loadings': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'n_components': n_components,
            'scaler': scaler,
            'model': pca
        }
        
        return results
    
    else:
        raise ValueError(f"Unknown multivariate analysis method: {method}")

def fit_calibration_curve(concentrations, responses, model='linear', **kwargs):
    """
    Fit a calibration curve for analytical measurements.
    
    Parameters:
    -----------
    concentrations : numpy.ndarray
        Known concentrations of standards
    responses : numpy.ndarray
        Measured responses corresponding to the standards
    model : str, optional
        Model type: 'linear', 'quadratic', or 'custom'
    **kwargs : dict
        Additional parameters including custom function for 'custom' model
        
    Returns:
    --------
    dict
        Calibration results including fitted parameters, function for prediction,
        and goodness of fit metrics
    """
    if model == 'linear':
        def fit_func(x, a, b):
            return a * x + b
        
        p0 = [1, 0]  # Initial guess
        
    elif model == 'quadratic':
        def fit_func(x, a, b, c):
            return a * x**2 + b * x + c
        
        p0 = [0, 1, 0]  # Initial guess
        
    elif model == 'custom':
        fit_func = kwargs.get('function')
        p0 = kwargs.get('initial_params')
        
        if fit_func is None:
            raise ValueError("Custom function must be provided for 'custom' model")
        if p0 is None:
            raise ValueError("Initial parameters must be provided for 'custom' model")
    
    else:
        raise ValueError(f"Unknown calibration model: {model}")
    
    # Fit the model
    params, covariance = curve_fit(fit_func, concentrations, responses, p0=p0)
    
    # Calculate R-squared
    residuals = responses - fit_func(concentrations, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((responses - np.mean(responses))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create prediction function
    def predict(x):
        return fit_func(x, *params)
    
    # Calculate standard error of parameters
    param_errors = np.sqrt(np.diag(covariance))
    
    # Calculate detection limit (simplified approach)
    # Using 3 * standard deviation of the residuals at the lowest concentration
    sorted_indices = np.argsort(concentrations)
    lowest_conc_idx = sorted_indices[0]
    residual_std = np.std(residuals)
    detection_limit = 3 * residual_std / (params[0] if model == 'linear' else None)
    
    results = {
        'model': model,
        'parameters': params,
        'parameter_errors': param_errors,
        'covariance': covariance,
        'r_squared': r_squared,
        'residuals': residuals,
        'predict': predict,
        'detection_limit': detection_limit if model == 'linear' else None
    }
    
    return results

def plot_horwitz_trumpet(conc_range=None, show_plateau=True, annotate=True):
    """
    Plot the Horwitz trumpet curve showing the relationship between
    concentration and relative standard deviation.
    
    Parameters:
    -----------
    conc_range : tuple, optional
        Range of concentrations to plot (min, max) in g/g
        Default is (1e-10, 1.0)
    show_plateau : bool, optional
        Whether to show the plateau at low concentrations
    annotate : bool, optional
        Whether to add annotations to the plot
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axis objects
    """
    if conc_range is None:
        conc_range = (1e-10, 1.0)
    
    # Set up concentrations on log scale
    concentrations = np.logspace(np.log10(conc_range[0]), np.log10(conc_range[1]), 1000)
    
    # Calculate RSDs
    rsds = horwitz_trumpet(concentrations, plateau_level=1e-7 if show_plateau else 0)
    
    # Create the plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the trumpet curve
    ax.semilogx(concentrations, rsds, 'b-', linewidth=2)
    ax.semilogx(concentrations, -rsds, 'b-', linewidth=2)
    
    # Fill the trumpet
    ax.fill_between(concentrations, rsds, -rsds, alpha=0.2, color='green')
    
    # Set labels and title
    ax.set_xlabel('浓度 (g analyte/g sample)', fontsize=12)
    ax.set_ylabel('相对标准偏差 (%)', fontsize=12)
    ax.set_title('Horwitz Trumpet: 浓度与相对标准偏差的关系', fontsize=14)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add concentration labels on x-axis
    conc_labels = [
        (1, '100%'), 
        (1e-2, '1%'), 
        (1e-4, '0.01%'), 
        (1e-6, '1 ppm'), 
        (1e-9, '1 ppb'), 
        (1e-12, '1 ppt')
    ]
    
    ax.set_xticks([c[0] for c in conc_labels])
    ax.set_xticklabels([c[1] for c in conc_labels])
    
    # Add annotations if requested
    if annotate:
        # Add key points
        key_points = [
            (1e-2, horwitz_trumpet(1e-2), "1% ≈ 4%"),
            (1e-6, horwitz_trumpet(1e-6), "1 ppm ≈ 16%"),
            (1e-10, horwitz_trumpet(1e-10), "10 ppt ≈ 22%")
        ]
        
        for point in key_points:
            ax.annotate(
                point[2], 
                xy=(point[0], point[1]), 
                xytext=(point[0]*1.5, point[1]*1.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
            )
    
    return fig, ax

def main():
    """
    主函数：演示如何使用工具函数解决分析化学中的实际问题
    """
    print("分析化学工具包演示")
    print("-" * 50)
    
    # 示例1：Horwitz Trumpet分析
    print("\n示例1: Horwitz Trumpet分析")
    
    # 计算10% NH3样品的预期相对标准偏差
    concentration = 0.1  # 10 wt% = 0.1 g/g
    inter_lab_rsd = horwitz_trumpet(concentration)
    intra_lab_rsd = intra_laboratory_rsd(inter_lab_rsd)
    
    print(f"样品浓度: 10 wt% NH3 ({concentration} g/g)")
    print(f"实验室间相对标准偏差: {inter_lab_rsd:.2f}%")
    print(f"实验室内相对标准偏差: {intra_lab_rsd:.2f}%")
    
    # 绘制Horwitz Trumpet曲线
    fig, ax = plot_horwitz_trumpet()
    _save_figure(fig, "horwitz_trumpet.png")
    plt.tight_layout()
    plt.show()
    
    # 示例2：光谱数据处理
    print("\n示例2: 光谱数据处理与峰检测")
    
    # 生成模拟光谱数据
    x = np.linspace(0, 1000, 1000)
    # 创建基线
    baseline = 0.2 * np.sin(x/200) + 0.1 * x/1000 + 0.05
    # 添加高斯峰
    peaks = [
        (200, 0.5, 10),  # (位置, 高度, 宽度)
        (400, 0.8, 15),
        (600, 0.3, 20),
        (800, 0.6, 12)
    ]
    
    spectrum = baseline.copy()
    for pos, height, width in peaks:
        spectrum += height * np.exp(-(x - pos)**2 / (2 * width**2))
    
    # 添加噪声
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, size=len(x))
    noisy_spectrum = spectrum + noise
    
    # 基线校正
    corrected_spectrum = baseline_correction(noisy_spectrum, method='als')
    
    # 峰检测
    peak_indices, peak_props = peak_detection(
        corrected_spectrum, 
        x_values=x,
        threshold=0.1,
        min_distance=30
    )
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 原始数据和基线校正
    ax1.plot(x, noisy_spectrum, 'b-', label='原始光谱')
    ax1.plot(x, baseline, 'r--', label='真实基线')
    ax1.plot(x, corrected_spectrum, 'g-', label='校正后光谱')
    ax1.set_title('光谱数据基线校正')
    ax1.set_xlabel('波长/波数')
    ax1.set_ylabel('强度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 峰检测结果
    ax2.plot(x, corrected_spectrum, 'b-')
    ax2.plot(peak_props['positions'], peak_props['heights'], 'ro', label='检测到的峰')
    
    # 标记峰宽
    for i, peak_idx in enumerate(peak_indices):
        ax2.hlines(
            y=corrected_spectrum[peak_idx] / 2,
            xmin=x[int(peak_props['left_bases'][i])],
            xmax=x[int(peak_props['right_bases'][i])],
            color='g', linestyle='--'
        )
    
    ax2.set_title('峰检测结果')
    ax2.set_xlabel('波长/波数')
    ax2.set_ylabel('强度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    _save_figure(fig, "spectrum_processing.png")
    plt.tight_layout()
    plt.show()
    
    # 示例3：多变量分析
    print("\n示例3: 多变量分析与化学计量学")
    
    # 生成模拟数据集 - 假设是不同样品的多种元素含量
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    
    # 创建三个不同的样品组
    group1 = np.random.normal(loc=5, scale=1, size=(n_samples//3, n_features))
    group2 = np.random.normal(loc=7, scale=1.2, size=(n_samples//3, n_features))
    group3 = np.random.normal(loc=3, scale=0.8, size=(n_samples - 2*(n_samples//3), n_features))
    
    # 合并数据
    X = np.vstack([group1, group2, group3])
    
    # 添加一些相关性
    X[:, 1] = X[:, 0] * 0.8 + np.random.normal(0, 0.5, size=n_samples)
    X[:, 3] = X[:, 2] * 0.6 + X[:, 4] * 0.3 + np.random.normal(0, 0.4, size=n_samples)
    
    # 创建样品标签
    y = np.array([0] * (n_samples//3) + [1] * (n_samples//3) + [2] * (n_samples - 2*(n_samples//3)))
    
    # 创建DataFrame
    feature_names = ['Cu', 'Zn', 'Fe', 'Pb', 'Cd']
    df = pd.DataFrame(X, columns=feature_names)
    df['Group'] = y
    
    print("数据集预览:")
    print(df.head())
    
    # 执行PCA分析
    pca_results = multivariate_analysis(df.drop('Group', axis=1), method='pca', n_components=2)
    
    # 提取结果
    scores = pca_results['scores']
    loadings = pca_results['loadings']
    explained_var = pca_results['explained_variance_ratio'] * 100
    
    # 绘制得分图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散点图按组着色
    colors = ['blue', 'red', 'green']
    groups = ['Group A', 'Group B', 'Group C']
    
    for i, group in enumerate(np.unique(y)):
        mask = y == group
        ax1.scatter(
            scores[mask, 0], 
            scores[mask, 1], 
            c=colors[i], 
            label=groups[i],
            alpha=0.7,
            edgecolors='k'
        )
    
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    ax1.set_title('PCA 得分图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制载荷图
    for i, feature in enumerate(feature_names):
        ax2.arrow(
            0, 0, 
            loadings[0, i] * 5, 
            loadings[1, i] * 5,
            head_width=0.1, 
            head_length=0.1, 
            fc=colors[i % len(colors)], 
            ec=colors[i % len(colors)]
        )
        ax2.text(
            loadings[0, i] * 5.2, 
            loadings[1, i] * 5.2, 
            feature, 
            fontsize=12
        )
    
    # 设置轴范围为正方形
    max_val = max(abs(loadings.min()), abs(loadings.max())) * 6
    ax2.set_xlim(-max_val, max_val)
    ax2.set_ylim(-max_val, max_val)
    
    ax2.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    ax2.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    ax2.set_title('PCA 载荷图')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    _save_figure(fig, "pca_plots.png")
    plt.tight_layout()
    plt.show()
    
    # 示例4：校准曲线拟合
    print("\n示例4: 分析校准曲线拟合")
    
    # 生成校准数据
    concentrations = np.array([0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    # 线性响应加噪声
    responses = 0.5 * concentrations + 0.2 + np.random.normal(0, 0.1, size=len(concentrations))
    
    # 拟合校准曲线
    calibration = fit_calibration_curve(concentrations, responses, model='linear')
    
    # 获取拟合参数
    slope, intercept = calibration['parameters']
    r_squared = calibration['r_squared']
    
    # 创建预测曲线
    conc_range = np.linspace(0, max(concentrations) * 1.1, 100)
    predicted_responses = calibration['predict'](conc_range)
    
    # 绘制校准曲线
    fig_cal = plt.figure(figsize=(8, 6))
    plt.scatter(concentrations, responses, color='blue', label='校准点')
    plt.plot(conc_range, predicted_responses, 'r-', label=f'拟合曲线: y = {slope:.3f}x + {intercept:.3f}')
    
    # 添加置信区间（简化计算）
    residual_std = np.std(calibration['residuals'])
    plt.fill_between(
        conc_range,
        predicted_responses - 2 * residual_std,
        predicted_responses + 2 * residual_std,
        alpha=0.2,
        color='gray',
        label='95% 置信区间'
    )
    
    plt.xlabel('浓度')
    plt.ylabel('响应')
    plt.title(f'分析校准曲线 (R² = {r_squared:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_figure(fig_cal, "calibration_curve.png")
    plt.tight_layout()
    plt.show()
    
    print(f"校准曲线方程: y = {slope:.3f}x + {intercept:.3f}")
    print(f"R²: {r_squared:.4f}")
    print(f"检出限: {calibration['detection_limit']:.4f}")

if __name__ == "__main__":
    main()