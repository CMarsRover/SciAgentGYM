# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 可选导入，如果模块不存在则跳过
try:
    from pyscf import gto
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("⚠️ PySCF模块未安装，分子动能积分功能将不可用")

try:
    from sympy import symbols, exp, diff, integrate, oo, simplify, pi
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️ SymPy模块未安装，符号计算功能将不可用")


# ------------------------------
# Config & Constants providers
# ------------------------------

def setup_matplotlib_chinese_fonts():
    """配置中文字体（如缺失将回退英文），避免绘图中文乱码。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
def get_default_molecule():
    """返回分子与基组的默认设置（示例为 HF 分子）。"""
    atom = 'H 0 0 0; F 0 0 0.917'
    basis = '6-31g'
    return atom, basis
def get_fd_default_grid():
    """有限差分动能矩阵的默认网格参数。"""
    return dict(N=500, L=2.0)

def get_planewave_defaults():
    """平面波动能曲线的默认参数。"""
    return dict(k_points=200, G_max=3, a=5.0)


# ------------------------------
# Part 1: 分子动能积分（PySCF）
# ------------------------------

def coding_func_molecular_kinetic(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g'):
    """返回分子动能积分矩阵与分子对象。"""
    if not PYSCF_AVAILABLE:
        raise ImportError("PySCF模块未安装，无法进行分子动能积分。请安装pyscf库。")
    mol = gto.M(atom=atom, basis=basis)
    kinetic_matrix = mol.intor('int1e_kin')  # PySCF动能积分
    return kinetic_matrix, mol


def visual_func_molecular_kinetic(kinetic_matrix, title="Kinetic Energy Integral Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(kinetic_matrix, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.xlabel("Basis Function Index")
    plt.ylabel("Basis Function Index")
    plt.tight_layout()
    plt.show()


def math_func_gaussian_kinetic_integral():
    """1D高斯波函数的动能积分解析结果。"""
    if not SYMPY_AVAILABLE:
        print("⚠️ SymPy模块未安装，无法进行符号计算。")
        return
    x, alpha = symbols('x alpha', real=True, positive=True)
    g = exp(-alpha * x**2)
    laplacian = diff(g, x, 2)
    integrand = g * (-0.5 * laplacian)
    result = integrate(integrand, (x, -oo, oo))
    print("数学推导：1D 高斯函数动能积分")
    print("波函数: φ(x) = exp(-α x²)")
    print(f"动能积分 ∫ φ (-1/2 ∇² φ) dx = {simplify(result)}")
    return result


# ------------------------------
# Part 2: 有限差分近似（无限深势阱）
# ------------------------------

def coding_func_finite_difference_kinetic(N=200, L=10.0):
    """构造一维二阶导数算子的有限差分矩阵，对应动能算子 -1/2 ∂²/∂x²。"""
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    T = -0.5 / dx**2 * (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))
    return T, x


def visual_func_infinite_well_eigenstates(T, x, num_states=5):
    from scipy.linalg import eigh
    energies, states = eigh(T)
    plt.figure(figsize=(10, 6))
    for i in range(min(num_states, len(energies))):
        psi = states[:, i]
        # 使用numpy实现梯形积分以规避弃用警告
        area = np.trapz(psi**2, x)
        psi_norm = psi / np.sqrt(area) if area > 0 else psi
        plt.plot(x, psi_norm * psi_norm, label=f"n={i+1}, E={energies[i]:.4f}")
    plt.title("Infinite Well: Kinetic-dominant Eigenstates (|ψ|²)")
    plt.xlabel("x")
    plt.ylabel("|ψ(x)|²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def math_func_infinite_well_analytical():
    """无限深势阱动能本征值公式：E_n = n²π²/(2L²)。"""
    if not SYMPY_AVAILABLE:
        print("⚠️ SymPy模块未安装，无法进行符号计算。")
        return
    n, L = symbols('n L', positive=True, integer=True)
    E_n = n**2 * pi**2 / (2 * L**2)
    print("数学推导：无限深势阱第 n 能级能量（动能）")
    print(f"E_n = {E_n}")
    return E_n


# ------------------------------
# Part 3: 平面波自由电子模型（倒格矢展开）
# ------------------------------

def coding_func_planewave_kinetic(k_points=100, G_max=5, a=5.0):
    """一维平面波自由电子模型：E(k,G)=0.5*(k+G*b)^2。"""
    b = 2 * pi / a  # 倒格子矢量
    k_list = np.linspace(-pi/a, pi/a, k_points)
    G_range = np.arange(-G_max, G_max + 1)
    E_bands = np.zeros((k_points, len(G_range)))
    for i, k in enumerate(k_list):
        for j, G in enumerate(G_range):
            kG = k + G * b
            E_bands[i, j] = 0.5 * kG**2
    return k_list, E_bands


def visual_func_band_structure(k_list, E_bands, title="1D Free-electron Band Structure"):
    plt.figure(figsize=(8, 6))
    for band in range(E_bands.shape[1]):
        plt.plot(k_list, E_bands[:, band], 'b-', lw=1.5)
    plt.xlabel("k")
    plt.ylabel("Energy E(k) [Hartree]")
    plt.title(title)
    plt.axvline(x=-pi/5, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=pi/5, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def math_func_planewave_kinetic_expression():
    if not SYMPY_AVAILABLE:
        print("⚠️ SymPy模块未安装，无法进行符号计算。")
        return
    k, G, b = symbols('k G b')
    T_expr = 0.5 * (k + G * b)**2
    print("数学表达式：平面波动能（倒格子空间）")
    print(f"T(k, G) = {simplify(T_expr)}")
    return T_expr


# ------------------------------
# Example runner
# ------------------------------
if __name__ == "__main__":
    setup_matplotlib_chinese_fonts()

    # 1) 分子动能积分（PySCF）
    atom, basis = get_default_molecule()
    try:
        T_mol, mol = coding_func_molecular_kinetic(atom=atom, basis=basis)
        print(f"分子: {mol.atom}\n基组: {mol.basis}")
        print("动能积分矩阵:\n", T_mol)
        visual_func_molecular_kinetic(T_mol, title=f"Molecular Kinetic Matrix ({basis})")
        math_func_gaussian_kinetic_integral()
    except ImportError as e:
        print(e)

    # 2) 有限差分本征态
    fd_params = get_fd_default_grid()
    T_fd, x = coding_func_finite_difference_kinetic(**fd_params)
    print(f"有限差分动能矩阵维度: {T_fd.shape}")
    visual_func_infinite_well_eigenstates(T_fd, x, num_states=4)
    try:
        math_func_infinite_well_analytical()
    except ImportError as e:
        print(e)

    # 3) 平面波自由电子模型
    pw_params = get_planewave_defaults()
    k_list, E_bands = coding_func_planewave_kinetic(**pw_params)
    visual_func_band_structure(k_list, E_bands)
    try:
        math_func_planewave_kinetic_expression()
    except ImportError as e:
        print(e)
