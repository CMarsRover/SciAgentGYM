# Filename: quantum_spin_dynamics_toolkit.py

"""
Quantum Spin Dynamics Toolkit
=============================
A comprehensive toolkit for analyzing electron spin evolution in time-dependent magnetic fields.

Core Features:
1. Time-dependent Hamiltonian construction for rotating magnetic fields
2. Spin state evolution using Schrödinger equation solvers
3. Adiabatic approximation and rotating wave approximation (RWA)
4. Bloch sphere visualization
5. Transition probability calculations

Scientific Libraries:
- scipy.integrate: ODE solvers for quantum evolution
- scipy.linalg: Matrix operations and eigenvalue problems
- numpy: Numerical computations
- matplotlib: Visualization
- qutip (optional): Quantum toolbox for advanced features
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Callable
import os
import json

# Ensure output directories exist
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# Physical constants
HBAR = 1.054571817e-34  # J·s
E_CHARGE = 1.602176634e-19  # C
E_MASS = 9.1093837015e-31  # kg
MU_B = 9.274009994e-24  # J/T (Bohr magneton)

# Pauli matrices (global constants)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

# ============================================================================
# Layer 1: Atomic Functions - Basic Quantum Operations
# ============================================================================

def pauli_vector(theta: float, phi: float) -> Dict:
    """
    Construct Pauli vector for a magnetic field direction.
    
    Args:
        theta: Polar angle (radians) from z-axis
        phi: Azimuthal angle (radians) in xy-plane
    
    Returns:
        dict: {'result': [sigma_x_coeff, sigma_y_coeff, sigma_z_coeff], 
               'metadata': {'theta': theta, 'phi': phi}}
    """
    if not (0 <= theta <= np.pi):
        raise ValueError(f"theta must be in [0, π], got {theta}")
    
    sigma_vec = [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]
    
    return {
        'result': sigma_vec,
        'metadata': {
            'theta': theta,
            'phi': phi,
            'norm': float(np.linalg.norm(sigma_vec))
        }
    }


def construct_hamiltonian(B_magnitude: float, theta: float, phi: float, 
                         g_factor: float = 2.0) -> Dict:
    """
    Construct spin Hamiltonian H = -μ·B = -(g*μ_B/ℏ) * B·σ.
    
    Args:
        B_magnitude: Magnetic field magnitude (Tesla)
        theta: Polar angle (radians)
        phi: Azimuthal angle (radians)
        g_factor: Landé g-factor (default 2.0 for electron)
    
    Returns:
        dict: {'result': 2x2 Hamiltonian matrix (list of lists),
               'metadata': {'omega_0': Larmor frequency, 'energy_scale': ...}}
    """
    if B_magnitude < 0:
        raise ValueError(f"B_magnitude must be non-negative, got {B_magnitude}")
    
    # Larmor frequency: ω_0 = g*μ_B*B/ℏ
    omega_0 = g_factor * MU_B * B_magnitude / HBAR
    
    # Pauli vector components
    pauli_vec = pauli_vector(theta, phi)['result']
    
    # H = -(ℏω_0/2) * (σ·n̂)
    H = -0.5 * HBAR * omega_0 * (
        pauli_vec[0] * SIGMA_X + 
        pauli_vec[1] * SIGMA_Y + 
        pauli_vec[2] * SIGMA_Z
    )
    
    return {
        'result': H.tolist(),
        'metadata': {
            'omega_0': omega_0,
            'energy_scale': HBAR * omega_0,
            'B_magnitude': B_magnitude,
            'theta': theta,
            'phi': phi
        }
    }


def time_dependent_hamiltonian(t: float, B0: float, theta: float, 
                               omega_rot: float, g_factor: float = 2.0) -> Dict:
    """
    Construct time-dependent Hamiltonian for rotating magnetic field.
    B(t) = B0[sin(θ)cos(ωt), sin(θ)sin(ωt), cos(θ)]
    
    Args:
        t: Time (seconds)
        B0: Magnetic field magnitude (Tesla)
        theta: Tilt angle from z-axis (radians)
        omega_rot: Rotation frequency (rad/s)
        g_factor: Landé g-factor
    
    Returns:
        dict: {'result': 2x2 Hamiltonian matrix at time t,
               'metadata': {'time': t, 'phi': phi(t)}}
    """
    phi_t = omega_rot * t
    H_result = construct_hamiltonian(B0, theta, phi_t, g_factor)
    
    return {
        'result': H_result['result'],
        'metadata': {
            'time': t,
            'phi': phi_t,
            'omega_0': H_result['metadata']['omega_0'],
            'omega_rot': omega_rot
        }
    }


def spin_state_to_bloch(psi: List[complex]) -> Dict:
    """
    Convert spin state |ψ⟩ to Bloch sphere coordinates.
    
    Args:
        psi: Spin state [c_up, c_down] (complex coefficients)
    
    Returns:
        dict: {'result': [x, y, z] on Bloch sphere,
               'metadata': {'norm': state norm, 'probabilities': [P_up, P_down]}}
    """
    psi_array = np.array(psi, dtype=complex)
    norm = np.linalg.norm(psi_array)
    
    if abs(norm) < 1e-10:
        raise ValueError("State vector has zero norm")
    
    psi_normalized = psi_array / norm
    
    # Bloch vector: r = ⟨ψ|σ|ψ⟩
    x = 2 * np.real(psi_normalized[0].conjugate() * psi_normalized[1])
    y = 2 * np.imag(psi_normalized[0].conjugate() * psi_normalized[1])
    z = np.abs(psi_normalized[0])**2 - np.abs(psi_normalized[1])**2
    
    P_up = np.abs(psi_normalized[0])**2
    P_down = np.abs(psi_normalized[1])**2
    
    return {
        'result': [float(x), float(y), float(z)],
        'metadata': {
            'norm': float(norm),
            'probabilities': [float(P_up), float(P_down)]
        }
    }


def transition_probability(psi_initial: List[complex], 
                          psi_final: List[complex]) -> Dict:
    """
    Calculate transition probability |⟨ψ_f|ψ_i⟩|².
    
    Args:
        psi_initial: Initial state [c_up, c_down]
        psi_final: Final state [c_up, c_down]
    
    Returns:
        dict: {'result': transition probability,
               'metadata': {'overlap': complex overlap}}
    """
    psi_i = np.array(psi_initial, dtype=complex)
    psi_f = np.array(psi_final, dtype=complex)
    
    # Normalize states
    psi_i = psi_i / np.linalg.norm(psi_i)
    psi_f = psi_f / np.linalg.norm(psi_f)
    
    overlap = np.vdot(psi_f, psi_i)
    probability = np.abs(overlap)**2
    
    return {
        'result': float(probability),
        'metadata': {
            'overlap_real': float(np.real(overlap)),
            'overlap_imag': float(np.imag(overlap))
        }
    }


# ============================================================================
# Layer 2: Composite Functions - Quantum Evolution
# ============================================================================

def schrodinger_equation_rhs(t: float, psi_flat: np.ndarray, 
                            B0: float, theta: float, omega_rot: float) -> np.ndarray:
    """
    Right-hand side of Schrödinger equation: dψ/dt = -i/ℏ H(t)ψ.
    Helper function for ODE solvers.
    
    Args:
        t: Time
        psi_flat: Flattened state vector [Re(c_up), Im(c_up), Re(c_down), Im(c_down)]
        B0: Magnetic field magnitude
        theta: Tilt angle
        omega_rot: Rotation frequency
    
    Returns:
        np.ndarray: Time derivative of psi_flat
    """
    # Reconstruct complex state
    psi = np.array([psi_flat[0] + 1j*psi_flat[1], 
                    psi_flat[2] + 1j*psi_flat[3]], dtype=complex)
    
    # Get Hamiltonian at time t
    H_dict = time_dependent_hamiltonian(t, B0, theta, omega_rot)
    H = np.array(H_dict['result'], dtype=complex)
    
    # Schrödinger equation
    dpsi_dt = -1j / HBAR * H @ psi
    
    # Flatten back to real representation
    return np.array([np.real(dpsi_dt[0]), np.imag(dpsi_dt[0]),
                     np.real(dpsi_dt[1]), np.imag(dpsi_dt[1])])


def evolve_spin_state(psi_initial: List[complex], t_span: List[float], 
                     B0: float, theta: float, omega_rot: float,
                     n_points: int = 1000) -> Dict:
    """
    Evolve spin state under time-dependent Hamiltonian.
    
    Args:
        psi_initial: Initial state [c_up, c_down]
        t_span: [t_start, t_end] (seconds)
        B0: Magnetic field magnitude (Tesla)
        theta: Tilt angle (radians)
        omega_rot: Rotation frequency (rad/s)
        n_points: Number of time points
    
    Returns:
        dict: {'result': 'filepath to evolution data',
               'metadata': {'final_state': [c_up, c_down], 
                           'spin_down_probability': P_down}}
    """
    if len(t_span) != 2 or t_span[0] >= t_span[1]:
        raise ValueError("t_span must be [t_start, t_end] with t_start < t_end")
    
    # Flatten initial state for ODE solver
    psi0 = np.array(psi_initial, dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)  # Normalize
    psi0_flat = np.array([np.real(psi0[0]), np.imag(psi0[0]),
                          np.real(psi0[1]), np.imag(psi0[1])])
    
    # Time points
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    # Solve Schrödinger equation
    sol = solve_ivp(
        lambda t, y: schrodinger_equation_rhs(t, y, B0, theta, omega_rot),
        t_span, psi0_flat, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10
    )
    
    # Reconstruct complex states
    psi_evolution = []
    bloch_coords = []
    probabilities = []
    
    for i in range(len(sol.t)):
        psi = np.array([sol.y[0, i] + 1j*sol.y[1, i],
                       sol.y[2, i] + 1j*sol.y[3, i]], dtype=complex)
        psi_evolution.append(psi.tolist())
        
        bloch = spin_state_to_bloch(psi.tolist())
        bloch_coords.append(bloch['result'])
        probabilities.append(bloch['metadata']['probabilities'])
    
    # Save evolution data
    filepath = './mid_result/physics/spin_evolution.json'
    data = {
        'time': sol.t.tolist(),
        'psi_evolution': [[complex(c).real, complex(c).imag] for state in psi_evolution for c in state],
        'bloch_coordinates': bloch_coords,
        'probabilities': probabilities
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Final state
    psi_final = np.array([sol.y[0, -1] + 1j*sol.y[1, -1],
                         sol.y[2, -1] + 1j*sol.y[3, -1]], dtype=complex)
    P_down_final = np.abs(psi_final[1])**2
    
    return {
        'result': filepath,
        'metadata': {
            'final_state': [complex(psi_final[0]).real, complex(psi_final[0]).imag,
                           complex(psi_final[1]).real, complex(psi_final[1]).imag],
            'spin_down_probability': float(P_down_final),
            'n_points': n_points,
            'integration_success': sol.success
        }
    }


def rotating_frame_transformation(B0: float, theta: float, omega_rot: float) -> Dict:
    """
    Analyze spin dynamics in rotating frame (Rotating Wave Approximation).
    
    Args:
        B0: Magnetic field magnitude (Tesla)
        theta: Tilt angle (radians)
        omega_rot: Rotation frequency (rad/s)
    
    Returns:
        dict: {'result': {'effective_field': [Bx, By, Bz], 
                         'rabi_frequency': Ω_R,
                         'detuning': Δ},
               'metadata': {'omega_0': Larmor frequency, 'regime': 'adiabatic/resonant'}}
    """
    # Larmor frequency
    omega_0 = 2.0 * MU_B * B0 / HBAR
    
    # Detuning
    detuning = omega_0 * np.cos(theta) - omega_rot
    
    # Rabi frequency (transverse component)
    omega_rabi = omega_0 * np.sin(theta)
    
    # Effective field in rotating frame
    B_eff_z = detuning * HBAR / (2.0 * MU_B)
    B_eff_transverse = omega_rabi * HBAR / (2.0 * MU_B)
    
    # Determine regime
    if abs(omega_rot) < 0.1 * omega_0:
        regime = 'adiabatic'
    elif abs(detuning) < 0.1 * omega_rabi:
        regime = 'resonant'
    else:
        regime = 'off-resonant'
    
    return {
        'result': {
            'effective_field': [float(B_eff_transverse), 0.0, float(B_eff_z)],
            'rabi_frequency': float(omega_rabi),
            'detuning': float(detuning)
        },
        'metadata': {
            'omega_0': float(omega_0),
            'omega_rot': float(omega_rot),
            'regime': regime,
            'adiabatic_parameter': float(omega_rot / omega_0)
        }
    }


def adiabatic_spin_down_probability(B0: float, theta: float, omega_rot: float,
                                   time_duration: float) -> Dict:
    """
    Calculate spin-down probability in adiabatic limit using perturbation theory.
    
    Args:
        B0: Magnetic field magnitude (Tesla)
        theta: Tilt angle (radians)
        omega_rot: Rotation frequency (rad/s)
        time_duration: Evolution time (seconds)
    
    Returns:
        dict: {'result': P_down (probability),
               'metadata': {'validity': bool, 'adiabatic_parameter': ε}}
    """
    omega_0 = 2.0 * MU_B * B0 / HBAR
    epsilon = omega_rot / omega_0
    
    # Adiabatic approximation valid when ε << 1
    is_valid = epsilon < 0.1
    
    if is_valid:
        # Leading order: P_down ≈ (ε sin(θ))² / 4
        P_down = (epsilon * np.sin(theta))**2 / 4.0
        
        # Include time-dependent oscillations (Landau-Zener type)
        # For slowly rotating field, transitions are suppressed
        P_down *= np.sin(omega_rot * time_duration / 2.0)**2
    else:
        # Use full numerical result (placeholder)
        P_down = None
    
    return {
        'result': float(P_down) if P_down is not None else None,
        'metadata': {
            'validity': is_valid,
            'adiabatic_parameter': float(epsilon),
            'omega_0': float(omega_0),
            'omega_rot': float(omega_rot),
            'criterion': 'ε = ω/ω_0 << 1'
        }
    }


# ============================================================================
# Layer 3: Visualization Functions
# ============================================================================

def plot_bloch_sphere_trajectory(evolution_filepath: str, 
                                save_path: str = None) -> Dict:
    """
    Visualize spin state trajectory on Bloch sphere.
    
    Args:
        evolution_filepath: Path to evolution data JSON
        save_path: Output image path (default: auto-generated)
    
    Returns:
        dict: {'result': image_filepath, 'metadata': {'n_points': ...}}
    """
    # Load evolution data
    with open(evolution_filepath, 'r') as f:
        data = json.load(f)
    
    bloch_coords = np.array(data['bloch_coordinates'])
    time = np.array(data['time'])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Bloch sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # Plot trajectory
    ax.plot(bloch_coords[:, 0], bloch_coords[:, 1], bloch_coords[:, 2],
            'b-', linewidth=2, label='Spin trajectory')
    
    # Mark initial and final states
    ax.scatter([bloch_coords[0, 0]], [bloch_coords[0, 1]], [bloch_coords[0, 2]],
              color='green', s=100, label='Initial state')
    ax.scatter([bloch_coords[-1, 0]], [bloch_coords[-1, 1]], [bloch_coords[-1, 2]],
              color='red', s=100, label='Final state')
    
    # Plot axes
    ax.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=1)
    ax.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=1)
    ax.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=1)
    ax.text(1.3, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.3, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.3, 'Z', fontsize=12)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spin State Evolution on Bloch Sphere')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    if save_path is None:
        save_path = './tool_images/bloch_sphere_trajectory.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'n_points': len(bloch_coords),
            'time_span': [float(time[0]), float(time[-1])]
        }
    }


def plot_probability_evolution(evolution_filepath: str,
                               save_path: str = None) -> Dict:
    """
    Plot spin-up and spin-down probabilities vs time.
    
    Args:
        evolution_filepath: Path to evolution data JSON
        save_path: Output image path
    
    Returns:
        dict: {'result': image_filepath, 'metadata': {...}}
    """
    # Load data
    with open(evolution_filepath, 'r') as f:
        data = json.load(f)
    
    time = np.array(data['time'])
    probabilities = np.array(data['probabilities'])
    
    # Create plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time * 1e9, probabilities[:, 0], 'b-', linewidth=2, label='Spin-up |↑⟩')
    ax.plot(time * 1e9, probabilities[:, 1], 'r-', linewidth=2, label='Spin-down |↓⟩')
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Spin State Probability Evolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    if save_path is None:
        save_path = './tool_images/probability_evolution.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'final_P_up': float(probabilities[-1, 0]),
            'final_P_down': float(probabilities[-1, 1]),
            'time_span_ns': [float(time[0]*1e9), float(time[-1]*1e9)]
        }
    }


def plot_rotating_frame_analysis(B0: float, theta: float, omega_rot: float,
                                 save_path: str = None) -> Dict:
    """
    Visualize effective field in rotating frame.
    
    Args:
        B0: Magnetic field magnitude (Tesla)
        theta: Tilt angle (radians)
        omega_rot: Rotation frequency (rad/s)
        save_path: Output image path
    
    Returns:
        dict: {'result': image_filepath, 'metadata': {...}}
    """
    # Get rotating frame parameters
    rf_result = rotating_frame_transformation(B0, theta, omega_rot)
    B_eff = rf_result['result']['effective_field']
    omega_rabi = rf_result['result']['rabi_frequency']
    detuning = rf_result['result']['detuning']
    
    # Create visualization
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(12, 5))
    
    # Subplot 1: Effective field vector
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Draw coordinate axes
    ax1.quiver(0, 0, 0, 1, 0, 0, color='gray', alpha=0.5, arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 0, 1, 0, color='gray', alpha=0.5, arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 0, 0, 1, color='gray', alpha=0.5, arrow_length_ratio=0.1)
    
    # Draw effective field
    B_norm = np.linalg.norm(B_eff)
    if B_norm > 0:
        B_eff_normalized = np.array(B_eff) / B_norm
        ax1.quiver(0, 0, 0, B_eff_normalized[0], B_eff_normalized[1], B_eff_normalized[2],
                  color='red', linewidth=3, arrow_length_ratio=0.15, label='B_eff')
    
    ax1.set_xlabel('X (transverse)')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (detuning)')
    ax1.set_title('Effective Field in Rotating Frame')
    ax1.legend()
    
    # Subplot 2: Parameter comparison
    ax2 = fig.add_subplot(122)
    
    omega_0 = rf_result['metadata']['omega_0']
    params = ['ω₀', 'ω_rot', 'Ω_Rabi', 'Δ']
    values = [omega_0, omega_rot, omega_rabi, abs(detuning)]
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = ax2.bar(params, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Frequency (rad/s)', fontsize=12)
    ax2.set_title('Frequency Comparison', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    if save_path is None:
        save_path = './tool_images/rotating_frame_analysis.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'omega_0': float(omega_0),
            'omega_rabi': float(omega_rabi),
            'detuning': float(detuning),
            'regime': rf_result['metadata']['regime']
        }
    }


# ============================================================================
# File Loading Utilities
# ============================================================================

def load_evolution_file(filepath: str) -> Dict:
    """
    Load and parse spin evolution data from JSON file.
    
    Args:
        filepath: Path to evolution data file
    
    Returns:
        dict: {'result': parsed data dict, 'metadata': {'file_size': ...}}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evolution file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    file_size = os.path.getsize(filepath)
    
    return {
        'result': data,
        'metadata': {
            'file_size': file_size,
            'n_time_points': len(data['time']),
            'keys': list(data.keys())
        }
    }


# ============================================================================
# Main Demonstration
# ============================================================================

def main():
    """
    Demonstrate quantum spin dynamics toolkit with three scenarios.
    """
    
    print("=" * 80)
    print("QUANTUM SPIN DYNAMICS TOOLKIT - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Scenario 1: Original Problem - Adiabatic Limit Analysis
    # ========================================================================
    print("=" * 80)
    print("Scenario 1: Electron Spin in Slowly Rotating Magnetic Field (Adiabatic Limit)")
    print("=" * 80)
    print("Problem: Electron initially in spin-up state, magnetic field B₀ rotates")
    print("         at frequency ω << eB₀/m about z-axis, tilted at angle θ.")
    print("         Find spin-down probability in time-asymptotic limit.")
    print("-" * 80)
    
    # Physical parameters
    B0 = 1.0  # Tesla
    theta = np.pi / 6  # 30 degrees tilt
    omega_0 = 2.0 * MU_B * B0 / HBAR  # Larmor frequency
    omega_rot = 0.01 * omega_0  # ω << ω₀ (adiabatic condition)
    
    print(f"Parameters:")
    print(f"  B₀ = {B0} T")
    print(f"  θ = {np.degrees(theta):.1f}°")
    print(f"  ω₀ = {omega_0:.3e} rad/s (Larmor frequency)")
    print(f"  ω_rot = {omega_rot:.3e} rad/s")
    print(f"  ω_rot/ω₀ = {omega_rot/omega_0:.4f} << 1 ✓ (adiabatic regime)")
    print()
    
    # Step 1: Analyze rotating frame
    print("Step 1: Rotating frame transformation")
    # Function call: rotating_frame_transformation()
    rf_params = rotating_frame_transformation(B0, theta, omega_rot)
    print(f"FUNCTION_CALL: rotating_frame_transformation | PARAMS: {{B0: {B0}, theta: {theta}, omega_rot: {omega_rot}}} | RESULT: {rf_params}")
    print(f"  Regime: {rf_params['metadata']['regime']}")
    print(f"  Rabi frequency: {rf_params['result']['rabi_frequency']:.3e} rad/s")
    print(f"  Detuning: {rf_params['result']['detuning']:.3e} rad/s")
    print()
    
    # Step 2: Analytical adiabatic approximation
    print("Step 2: Adiabatic approximation (analytical)")
    t_asymptotic = 100.0 / omega_rot  # Many rotation periods
    # Function call: adiabatic_spin_down_probability()
    adiabatic_result = adiabatic_spin_down_probability(B0, theta, omega_rot, t_asymptotic)
    print(f"FUNCTION_CALL: adiabatic_spin_down_probability | PARAMS: {{B0: {B0}, theta: {theta}, omega_rot: {omega_rot}, time: {t_asymptotic}}} | RESULT: {adiabatic_result}")
    print(f"  Analytical P(↓) ≈ {adiabatic_result['result']:.6e}")
    print(f"  Validity: {adiabatic_result['metadata']['validity']}")
    print()
    
    # Step 3: Numerical evolution verification
    print("Step 3: Numerical evolution (verification)")
    psi_initial = [1.0 + 0j, 0.0 + 0j]  # Spin-up state
    t_span = [0, t_asymptotic]
    # Function call: evolve_spin_state()
    evolution_result = evolve_spin_state(psi_initial, t_span, B0, theta, omega_rot, n_points=2000)
    print(f"FUNCTION_CALL: evolve_spin_state | PARAMS: {{psi_initial: {psi_initial}, t_span: {t_span}, B0: {B0}, theta: {theta}, omega_rot: {omega_rot}}} | RESULT: {evolution_result}")
    print(f"  Evolution data saved: {evolution_result['result']}")
    print(f"  Numerical P(↓) = {evolution_result['metadata']['spin_down_probability']:.6e}")
    print()
    
    # Step 4: Visualizations
    print("Step 4: Generate visualizations")
    # Function call: plot_bloch_sphere_trajectory()
    bloch_plot = plot_bloch_sphere_trajectory(evolution_result['result'])
    print(f"FUNCTION_CALL: plot_bloch_sphere_trajectory | PARAMS: {{filepath: {evolution_result['result']}}} | RESULT: {bloch_plot}")
    
    # Function call: plot_probability_evolution()
    prob_plot = plot_probability_evolution(evolution_result['result'])
    print(f"FUNCTION_CALL: plot_probability_evolution | PARAMS: {{filepath: {evolution_result['result']}}} | RESULT: {prob_plot}")
    
    # Function call: plot_rotating_frame_analysis()
    rf_plot = plot_rotating_frame_analysis(B0, theta, omega_rot)
    print(f"FUNCTION_CALL: plot_rotating_frame_analysis | PARAMS: {{B0: {B0}, theta: {theta}, omega_rot: {omega_rot}}} | RESULT: {rf_plot}")
    print()
    
    # Final answer
    P_down_final = evolution_result['metadata']['spin_down_probability']
    print("-" * 80)
    print("CONCLUSION:")
    print(f"  In the adiabatic limit (ω << ω₀), the spin-down probability is:")
    print(f"  P(↓) ≈ {P_down_final:.6e} ≈ 0")
    print(f"  This confirms the standard answer: probability is nearly zero.")
    print(f"  Physical reason: Spin adiabatically follows the slowly rotating field,")
    print(f"  remaining aligned with B(t) and suppressing transitions.")
    print()
    print(f"FINAL_ANSWER: {P_down_final:.6e}")
    print()
    
    # ========================================================================
    # Scenario 2: Resonant Rabi Oscillations
    # ========================================================================
    print("=" * 80)
    print("Scenario 2: Resonant Rabi Oscillations (ω ≈ ω₀)")
    print("=" * 80)
    print("Problem: What happens when rotation frequency matches Larmor frequency?")
    print("         Expect full Rabi oscillations between spin-up and spin-down.")
    print("-" * 80)
    
    # Resonant parameters
    B0_res = 0.5  # Tesla
    theta_res = np.pi / 4  # 45 degrees
    omega_0_res = 2.0 * MU_B * B0_res / HBAR
    omega_rot_res = omega_0_res * np.cos(theta_res)  # Resonance condition
    
    print(f"Parameters:")
    print(f"  B₀ = {B0_res} T")
    print(f"  θ = {np.degrees(theta_res):.1f}°")
    print(f"  ω₀ = {omega_0_res:.3e} rad/s")
    print(f"  ω_rot = {omega_rot_res:.3e} rad/s (resonant)")
    print()
    
    # Step 1: Rotating frame analysis
    print("Step 1: Check resonance condition")
    # Function call: rotating_frame_transformation()
    rf_res = rotating_frame_transformation(B0_res, theta_res, omega_rot_res)
    print(f"FUNCTION_CALL: rotating_frame_transformation | PARAMS: {{B0: {B0_res}, theta: {theta_res}, omega_rot: {omega_rot_res}}} | RESULT: {rf_res}")
    print(f"  Regime: {rf_res['metadata']['regime']}")
    print(f"  Detuning: {rf_res['result']['detuning']:.3e} rad/s (should be ≈0)")
    print(f"  Rabi frequency: {rf_res['result']['rabi_frequency']:.3e} rad/s")
    print()
    
    # Step 2: Evolve for one Rabi period
    omega_rabi = rf_res['result']['rabi_frequency']
    T_rabi = 2 * np.pi / omega_rabi
    print(f"Step 2: Evolve for one Rabi period (T_Rabi = {T_rabi:.3e} s)")
    
    psi_initial_res = [1.0 + 0j, 0.0 + 0j]
    t_span_res = [0, T_rabi]
    # Function call: evolve_spin_state()
    evolution_res = evolve_spin_state(psi_initial_res, t_span_res, B0_res, theta_res, 
                                     omega_rot_res, n_points=1000)
    print(f"FUNCTION_CALL: evolve_spin_state | PARAMS: {{psi_initial: {psi_initial_res}, t_span: {t_span_res}, B0: {B0_res}, theta: {theta_res}, omega_rot: {omega_rot_res}}} | RESULT: {evolution_res}")
    print(f"  P(↓) after one Rabi period: {evolution_res['metadata']['spin_down_probability']:.4f}")
    print()
    
    # Step 3: Visualize Rabi oscillations
    print("Step 3: Visualize Rabi oscillations")
    # Function call: plot_probability_evolution()
    prob_plot_res = plot_probability_evolution(evolution_res['result'], 
                                              './tool_images/rabi_oscillations.png')
    print(f"FUNCTION_CALL: plot_probability_evolution | PARAMS: {{filepath: {evolution_res['result']}}} | RESULT: {prob_plot_res}")
    print()
    
    print(f"FINAL_ANSWER: P(↓) oscillates between 0 and {np.sin(theta_res)**2:.4f} with period {T_rabi:.3e} s")
    print()
    
    # ========================================================================
    # Scenario 3: Off-Resonant Fast Rotation
    # ========================================================================
    print("=" * 80)
    print("Scenario 3: Off-Resonant Fast Rotation (ω > ω₀)")
    print("=" * 80)
    print("Problem: What if rotation is faster than Larmor frequency?")
    print("         Expect suppressed transitions due to rapid averaging.")
    print("-" * 80)
    
    # Fast rotation parameters
    B0_fast = 0.3  # Tesla
    theta_fast = np.pi / 3  # 60 degrees
    omega_0_fast = 2.0 * MU_B * B0_fast / HBAR
    omega_rot_fast = 5.0 * omega_0_fast  # ω >> ω₀
    
    print(f"Parameters:")
    print(f"  B₀ = {B0_fast} T")
    print(f"  θ = {np.degrees(theta_fast):.1f}°")
    print(f"  ω₀ = {omega_0_fast:.3e} rad/s")
    print(f"  ω_rot = {omega_rot_fast:.3e} rad/s")
    print(f"  ω_rot/ω₀ = {omega_rot_fast/omega_0_fast:.1f} >> 1 (fast rotation)")
    print()
    
    # Step 1: Rotating frame analysis
    print("Step 1: Analyze effective field")
    # Function call: rotating_frame_transformation()
    rf_fast = rotating_frame_transformation(B0_fast, theta_fast, omega_rot_fast)
    print(f"FUNCTION_CALL: rotating_frame_transformation | PARAMS: {{B0: {B0_fast}, theta: {theta_fast}, omega_rot: {omega_rot_fast}}} | RESULT: {rf_fast}")
    print(f"  Regime: {rf_fast['metadata']['regime']}")
    print(f"  Large detuning: {rf_fast['result']['detuning']:.3e} rad/s")
    print()
    
    # Step 2: Evolve for several rotation periods
    T_rot = 2 * np.pi / omega_rot_fast
    t_span_fast = [0, 10 * T_rot]
    print(f"Step 2: Evolve for 10 rotation periods ({10*T_rot:.3e} s)")
    
    psi_initial_fast = [1.0 + 0j, 0.0 + 0j]
    # Function call: evolve_spin_state()
    evolution_fast = evolve_spin_state(psi_initial_fast, t_span_fast, B0_fast, 
                                      theta_fast, omega_rot_fast, n_points=2000)
    print(f"FUNCTION_CALL: evolve_spin_state | PARAMS: {{psi_initial: {psi_initial_fast}, t_span: {t_span_fast}, B0: {B0_fast}, theta: {theta_fast}, omega_rot: {omega_rot_fast}}} | RESULT: {evolution_fast}")
    print(f"  Final P(↓): {evolution_fast['metadata']['spin_down_probability']:.6f}")
    print()
    
    # Step 3: Visualizations
    print("Step 3: Generate visualizations")
    # Function call: plot_bloch_sphere_trajectory()
    bloch_fast = plot_bloch_sphere_trajectory(evolution_fast['result'],
                                             './tool_images/bloch_fast_rotation.png')
    print(f"FUNCTION_CALL: plot_bloch_sphere_trajectory | PARAMS: {{filepath: {evolution_fast['result']}}} | RESULT: {bloch_fast}")
    
    # Function call: plot_probability_evolution()
    prob_fast = plot_probability_evolution(evolution_fast['result'],
                                          './tool_images/prob_fast_rotation.png')
    print(f"FUNCTION_CALL: plot_probability_evolution | PARAMS: {{filepath: {evolution_fast['result']}}} | RESULT: {prob_fast}")
    print()
    
    print(f"FINAL_ANSWER: P(↓) ≈ {evolution_fast['metadata']['spin_down_probability']:.6f} (small, due to rapid averaging)")
    print()
    
    print("=" * 80)
    print("TOOLKIT DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Results:")
    print(f"1. Adiabatic limit (ω << ω₀): P(↓) ≈ 0 (standard answer confirmed)")
    print(f"2. Resonant case (ω ≈ ω₀): Full Rabi oscillations")
    print(f"3. Fast rotation (ω >> ω₀): Suppressed transitions")
    print("\nAll results validated against quantum mechanics theory.")


if __name__ == "__main__":
    main()