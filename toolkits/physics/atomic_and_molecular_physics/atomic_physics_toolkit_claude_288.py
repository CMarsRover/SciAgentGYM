# Filename: atomic_physics_toolkit.py

"""
Atomic Physics Toolkit for Hyperfine Structure Calculations
============================================================

This toolkit provides functions for calculating hyperfine structure transitions
in hydrogen-like atoms, with support for arbitrary electron and nuclear spins.

Key Features:
- Hyperfine splitting frequency calculations
- Quantum angular momentum coupling
- Support for modified electron spin scenarios
- Integration with scipy.constants for physical constants

Physical Background:
-------------------
The hyperfine structure arises from the magnetic interaction between the 
electron's magnetic moment and the nuclear magnetic moment. The energy splitting
is given by the Fermi contact term:

ΔE_hfs = (4/3) * μ_0 * g_I * μ_N * g_J * μ_B * |ψ(0)|² * ΔF

where:
- μ_0: vacuum permeability
- g_I: nuclear g-factor
- μ_N: nuclear magneton
- g_J: electron g-factor
- μ_B: Bohr magneton
- |ψ(0)|²: electron probability density at nucleus
- ΔF: change in total angular momentum quantum number

For the ground state hydrogen atom:
ΔE_hfs = A_hfs * [F(F+1) - I(I+1) - J(J+1)] / 2

where A_hfs is the hyperfine structure constant.
"""

import numpy as np
from scipy import constants
from typing import Dict, List, Tuple
import json
import os

# ============================================================================
# LAYER 1: ATOMIC CONSTANTS AND FUNDAMENTAL CALCULATIONS
# ============================================================================

def get_physical_constants() -> Dict[str, float]:
    """
    Retrieve fundamental physical constants from scipy.constants.
    
    Returns:
        dict: Dictionary containing physical constants with metadata
            - 'result': dict of constant values
            - 'metadata': source and units information
    """
    constants_dict = {
        'c': constants.c,  # Speed of light (m/s)
        'h': constants.h,  # Planck constant (J·s)
        'hbar': constants.hbar,  # Reduced Planck constant (J·s)
        'mu_0': constants.mu_0,  # Vacuum permeability (N/A²)
        'mu_B': constants.physical_constants['Bohr magneton'][0],  # Bohr magneton (J/T)
        'mu_N': constants.physical_constants['nuclear magneton'][0],  # Nuclear magneton (J/T)
        'alpha': constants.alpha,  # Fine structure constant
        'm_e': constants.m_e,  # Electron mass (kg)
        'm_p': constants.m_p,  # Proton mass (kg)
        'a_0': constants.physical_constants['Bohr radius'][0],  # Bohr radius (m)
        'R_inf': constants.Rydberg * constants.c * constants.h,  # Rydberg energy (J)
    }
    
    return {
        'result': constants_dict,
        'metadata': {
            'source': 'scipy.constants',
            'units': {
                'c': 'm/s',
                'h': 'J·s',
                'hbar': 'J·s',
                'mu_0': 'N/A²',
                'mu_B': 'J/T',
                'mu_N': 'J/T',
                'alpha': 'dimensionless',
                'm_e': 'kg',
                'm_p': 'kg',
                'a_0': 'm',
                'R_inf': 'J'
            }
        }
    }


def calculate_electron_g_factor(spin: float) -> Dict[str, float]:
    """
    Calculate the electron g-factor for a given spin.
    
    For a pure spin system, g_J ≈ g_s ≈ 2.002319 (Dirac theory gives 2.0)
    We use the approximation g_J ≈ 2 for simplicity in this modified scenario.
    
    Args:
        spin: Electron spin quantum number (e.g., 0.5, 1.5, etc.)
    
    Returns:
        dict: Contains g-factor value and calculation metadata
    
    Raises:
        ValueError: If spin is negative or not a half-integer
    """
    if spin < 0:
        raise ValueError(f"Spin must be non-negative, got {spin}")
    
    # Check if spin is a half-integer
    if not np.isclose((2 * spin) % 1, 0):
        raise ValueError(f"Spin must be a half-integer (n/2), got {spin}")
    
    # For ground state hydrogen (L=0), g_J = g_s ≈ 2
    g_factor = 2.0023193043617  # Precise experimental value
    
    return {
        'result': g_factor,
        'metadata': {
            'spin': spin,
            'approximation': 'Dirac theory with QED corrections',
            'note': 'For L=0 states, g_J equals electron g_s'
        }
    }


def calculate_proton_g_factor() -> Dict[str, float]:
    """
    Get the proton g-factor (nuclear g-factor for hydrogen).
    
    The proton g-factor is an experimentally measured quantity:
    g_p = 5.5856946893 (CODATA 2018)
    
    Returns:
        dict: Contains proton g-factor and metadata
    """
    g_proton = 5.5856946893  # CODATA 2018 value
    
    return {
        'result': g_proton,
        'metadata': {
            'source': 'CODATA 2018',
            'nuclear_spin': 0.5,
            'uncertainty': 1.6e-9
        }
    }


def calculate_wavefunction_at_nucleus(n: int, l: int, a_0: float) -> Dict[str, float]:
    """
    Calculate |ψ(0)|² for hydrogen atom at the nucleus.
    
    For hydrogen atom: |ψ_nlm(0)|² = (1/(π*n³*a_0³)) * δ_l0
    Only s-orbitals (l=0) have non-zero probability at nucleus.
    
    Args:
        n: Principal quantum number (must be positive integer)
        l: Orbital angular momentum quantum number (0 ≤ l < n)
        a_0: Bohr radius in meters
    
    Returns:
        dict: Contains |ψ(0)|² value and calculation details
    
    Raises:
        ValueError: If quantum numbers are invalid
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError(f"Principal quantum number n must be positive integer, got {n}")
    
    if l < 0 or l >= n:
        raise ValueError(f"Orbital quantum number l must satisfy 0 ≤ l < n, got l={l}, n={n}")
    
    if a_0 <= 0:
        raise ValueError(f"Bohr radius must be positive, got {a_0}")
    
    # Only s-orbitals (l=0) have non-zero density at nucleus
    if l == 0:
        psi_squared = 1.0 / (np.pi * n**3 * a_0**3)
    else:
        psi_squared = 0.0
    
    return {
        'result': psi_squared,
        'metadata': {
            'n': n,
            'l': l,
            'orbital_type': 's' if l == 0 else 'non-s',
            'units': 'm^-3',
            'note': 'Only s-orbitals have non-zero density at nucleus'
        }
    }


# ============================================================================
# LAYER 2: QUANTUM ANGULAR MOMENTUM AND HYPERFINE COUPLING
# ============================================================================

def calculate_total_angular_momentum_states(I: float, J: float) -> Dict[str, List[float]]:
    """
    Calculate possible total angular momentum quantum numbers F.
    
    For coupled angular momenta I and J:
    F ranges from |I - J| to I + J in integer steps
    
    Args:
        I: Nuclear spin quantum number
        J: Electronic angular momentum quantum number
    
    Returns:
        dict: Contains list of possible F values and coupling information
    
    Raises:
        ValueError: If I or J are negative or not half-integers
    """
    if I < 0 or J < 0:
        raise ValueError(f"Angular momenta must be non-negative, got I={I}, J={J}")
    
    # Check half-integer condition
    if not (np.isclose((2*I) % 1, 0) and np.isclose((2*J) % 1, 0)):
        raise ValueError(f"I and J must be half-integers, got I={I}, J={J}")
    
    F_min = abs(I - J)
    F_max = I + J
    
    # Generate F values
    F_values = []
    F = F_min
    while F <= F_max + 1e-10:  # Small tolerance for floating point
        F_values.append(F)
        F += 1.0
    
    return {
        'result': F_values,
        'metadata': {
            'I': I,
            'J': J,
            'F_min': F_min,
            'F_max': F_max,
            'num_states': len(F_values),
            'coupling_rule': '|I-J| ≤ F ≤ I+J'
        }
    }


def calculate_hyperfine_constant_A(
    g_I: float,
    g_J: float,
    mu_N: float,
    mu_B: float,
    psi_squared: float,
    mu_0: float
) -> Dict[str, float]:
    """
    Calculate the hyperfine structure constant A for hydrogen.
    
    The Fermi contact term gives:
    A = (8π/3) * μ_0 * g_I * μ_N * g_J * μ_B * |ψ(0)|²
    
    This is the energy coefficient in: E_hfs = A * K
    where K = [F(F+1) - I(I+1) - J(J+1)] / 2
    
    Args:
        g_I: Nuclear g-factor
        g_J: Electronic g-factor
        mu_N: Nuclear magneton (J/T)
        mu_B: Bohr magneton (J/T)
        psi_squared: |ψ(0)|² electron density at nucleus (m^-3)
        mu_0: Vacuum permeability (N/A²)
    
    Returns:
        dict: Contains A constant in Joules and Hz, with calculation details
    """
    if any(x <= 0 for x in [mu_N, mu_B, mu_0]):
        raise ValueError("Physical constants must be positive")
    
    if psi_squared < 0:
        raise ValueError(f"Wavefunction density must be non-negative, got {psi_squared}")
    
    # Calculate A in Joules
    A_joules = (8 * np.pi / 3) * mu_0 * g_I * mu_N * g_J * mu_B * psi_squared
    
    # Convert to frequency (Hz) via E = h*ν
    h = constants.h
    A_hz = A_joules / h
    
    return {
        'result': A_hz,
        'metadata': {
            'A_joules': A_joules,
            'A_hz': A_hz,
            'A_MHz': A_hz / 1e6,
            'g_I': g_I,
            'g_J': g_J,
            'formula': 'A = (8π/3) * μ_0 * g_I * μ_N * g_J * μ_B * |ψ(0)|²',
            'units': 'Hz'
        }
    }


def calculate_hyperfine_energy_shift(A_hz: float, F: float, I: float, J: float) -> Dict[str, float]:
    """
    Calculate the hyperfine energy shift for a given F state.
    
    E_hfs = (h * A / 2) * [F(F+1) - I(I+1) - J(J+1)]
    
    Args:
        A_hz: Hyperfine constant in Hz
        F: Total angular momentum quantum number
        I: Nuclear spin quantum number
        J: Electronic angular momentum quantum number
    
    Returns:
        dict: Contains energy shift in Joules and frequency in Hz
    """
    if A_hz < 0:
        raise ValueError(f"Hyperfine constant must be non-negative, got {A_hz}")
    
    # Calculate the quantum number factor
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    
    # Energy shift in frequency units (Hz)
    delta_nu = (A_hz / 2) * K
    
    # Energy shift in Joules
    h = constants.h
    delta_E = h * delta_nu
    
    return {
        'result': delta_nu,
        'metadata': {
            'delta_E_joules': delta_E,
            'delta_nu_hz': delta_nu,
            'delta_nu_MHz': delta_nu / 1e6,
            'F': F,
            'I': I,
            'J': J,
            'K_factor': K,
            'formula': 'ΔE = (h*A/2) * [F(F+1) - I(I+1) - J(J+1)]'
        }
    }


# ============================================================================
# LAYER 3: COMPLETE HYPERFINE TRANSITION CALCULATIONS
# ============================================================================

def calculate_hyperfine_transition_frequency(
    electron_spin: float,
    nuclear_spin: float,
    n: int = 1,
    l: int = 0
) -> Dict[str, float]:
    """
    Calculate the hyperfine transition frequency for hydrogen-like atom.
    
    This is the main calculation function that combines all lower-level functions
    to compute the frequency of the hyperfine transition between F states.
    
    Args:
        electron_spin: Electron spin quantum number (e.g., 0.5, 1.5)
        nuclear_spin: Nuclear spin quantum number (typically 0.5 for proton)
        n: Principal quantum number (default: 1 for ground state)
        l: Orbital angular momentum quantum number (default: 0 for s-orbital)
    
    Returns:
        dict: Contains transition frequency and complete calculation breakdown
    """
    # Step 1: Get physical constants
    # Function call: get_physical_constants()
    constants_result = get_physical_constants()
    phys_const = constants_result['result']
    print(f"FUNCTION_CALL: get_physical_constants | PARAMS: {{}} | RESULT: {list(phys_const.keys())}")
    
    # Step 2: Calculate electron g-factor
    # Function call: calculate_electron_g_factor()
    g_J_result = calculate_electron_g_factor(electron_spin)
    g_J = g_J_result['result']
    print(f"FUNCTION_CALL: calculate_electron_g_factor | PARAMS: {{'spin': {electron_spin}}} | RESULT: {g_J}")
    
    # Step 3: Get proton g-factor
    # Function call: calculate_proton_g_factor()
    g_I_result = calculate_proton_g_factor()
    g_I = g_I_result['result']
    print(f"FUNCTION_CALL: calculate_proton_g_factor | PARAMS: {{}} | RESULT: {g_I}")
    
    # Step 4: Calculate wavefunction density at nucleus
    # Function call: calculate_wavefunction_at_nucleus()
    psi_result = calculate_wavefunction_at_nucleus(n, l, phys_const['a_0'])
    psi_squared = psi_result['result']
    print(f"FUNCTION_CALL: calculate_wavefunction_at_nucleus | PARAMS: {{'n': {n}, 'l': {l}, 'a_0': {phys_const['a_0']:.6e}}} | RESULT: {psi_squared:.6e}")
    
    # Step 5: For ground state, J = electron_spin (since L=0)
    J = electron_spin
    I = nuclear_spin
    
    # Step 6: Calculate possible F states
    # Function call: calculate_total_angular_momentum_states()
    F_states_result = calculate_total_angular_momentum_states(I, J)
    F_values = F_states_result['result']
    print(f"FUNCTION_CALL: calculate_total_angular_momentum_states | PARAMS: {{'I': {I}, 'J': {J}}} | RESULT: {F_values}")
    
    # Step 7: Calculate hyperfine constant A
    # Function call: calculate_hyperfine_constant_A()
    A_result = calculate_hyperfine_constant_A(
        g_I=g_I,
        g_J=g_J,
        mu_N=phys_const['mu_N'],
        mu_B=phys_const['mu_B'],
        psi_squared=psi_squared,
        mu_0=phys_const['mu_0']
    )
    A_hz = A_result['result']
    print(f"FUNCTION_CALL: calculate_hyperfine_constant_A | PARAMS: {{'g_I': {g_I:.4f}, 'g_J': {g_J:.4f}, ...}} | RESULT: {A_hz:.6e} Hz")
    
    # Step 8: Calculate energy shifts for all F states
    energy_shifts = []
    for F in F_values:
        # Function call: calculate_hyperfine_energy_shift()
        shift_result = calculate_hyperfine_energy_shift(A_hz, F, I, J)
        energy_shifts.append({
            'F': F,
            'frequency_hz': shift_result['result'],
            'frequency_MHz': shift_result['result'] / 1e6
        })
        print(f"FUNCTION_CALL: calculate_hyperfine_energy_shift | PARAMS: {{'A_hz': {A_hz:.6e}, 'F': {F}, 'I': {I}, 'J': {J}}} | RESULT: {shift_result['result']:.6e} Hz")
    
    # Step 9: Calculate transition frequency (between highest and lowest F)
    if len(F_values) >= 2:
        F_upper = max(F_values)
        F_lower = min(F_values)
        
        # Find corresponding energy shifts
        E_upper = next(e['frequency_hz'] for e in energy_shifts if e['F'] == F_upper)
        E_lower = next(e['frequency_hz'] for e in energy_shifts if e['F'] == F_lower)
        
        transition_freq_hz = abs(E_upper - E_lower)
        transition_freq_MHz = transition_freq_hz / 1e6
    else:
        transition_freq_hz = 0.0
        transition_freq_MHz = 0.0
        F_upper = F_lower = F_values[0] if F_values else 0
    
    return {
        'result': transition_freq_MHz,
        'metadata': {
            'electron_spin': electron_spin,
            'nuclear_spin': nuclear_spin,
            'J': J,
            'I': I,
            'F_states': F_values,
            'F_upper': F_upper,
            'F_lower': F_lower,
            'A_constant_MHz': A_hz / 1e6,
            'transition_frequency_hz': transition_freq_hz,
            'transition_frequency_MHz': transition_freq_MHz,
            'energy_shifts': energy_shifts,
            'quantum_numbers': {'n': n, 'l': l},
            'note': f'Transition between F={F_upper} and F={F_lower} states'
        }
    }


def compare_hyperfine_frequencies(
    electron_spins: List[float],
    nuclear_spin: float = 0.5
) -> Dict[str, List[Dict]]:
    """
    Compare hyperfine transition frequencies for different electron spins.
    
    This function is useful for understanding how the hyperfine structure
    changes with modified electron spin values.
    
    Args:
        electron_spins: List of electron spin values to compare
        nuclear_spin: Nuclear spin quantum number (default: 0.5)
    
    Returns:
        dict: Contains comparison table and analysis
    """
    if not electron_spins:
        raise ValueError("electron_spins list cannot be empty")
    
    if nuclear_spin < 0:
        raise ValueError(f"Nuclear spin must be non-negative, got {nuclear_spin}")
    
    results = []
    
    for s_e in electron_spins:
        # Function call: calculate_hyperfine_transition_frequency()
        freq_result = calculate_hyperfine_transition_frequency(
            electron_spin=s_e,
            nuclear_spin=nuclear_spin
        )
        
        results.append({
            'electron_spin': s_e,
            'frequency_MHz': freq_result['result'],
            'frequency_hz': freq_result['metadata']['transition_frequency_hz'],
            'F_states': freq_result['metadata']['F_states'],
            'A_constant_MHz': freq_result['metadata']['A_constant_MHz']
        })
        
        print(f"FUNCTION_CALL: calculate_hyperfine_transition_frequency | PARAMS: {{'electron_spin': {s_e}, 'nuclear_spin': {nuclear_spin}}} | RESULT: {freq_result['result']:.2f} MHz")
    
    # Calculate scaling factors relative to standard hydrogen (s=1/2)
    standard_freq = next((r['frequency_MHz'] for r in results if r['electron_spin'] == 0.5), None)
    
    if standard_freq:
        for r in results:
            r['scaling_factor'] = r['frequency_MHz'] / standard_freq
    
    return {
        'result': results,
        'metadata': {
            'nuclear_spin': nuclear_spin,
            'num_comparisons': len(results),
            'standard_hydrogen_freq_MHz': standard_freq,
            'note': 'Comparison of hyperfine frequencies for different electron spins'
        }
    }


# ============================================================================
# LAYER 4: VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_hyperfine_levels(
    electron_spin: float,
    nuclear_spin: float,
    output_path: str = "./tool_images/hyperfine_levels.png"
) -> Dict[str, str]:
    """
    Create an energy level diagram for hyperfine structure.
    
    Args:
        electron_spin: Electron spin quantum number
        nuclear_spin: Nuclear spin quantum number
        output_path: Path to save the figure
    
    Returns:
        dict: Contains file path and visualization metadata
    """
    import matplotlib.pyplot as plt
    
    # Set font to support both Chinese and English
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Calculate hyperfine structure
    result = calculate_hyperfine_transition_frequency(electron_spin, nuclear_spin)
    energy_shifts = result['metadata']['energy_shifts']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot energy levels
    for shift in energy_shifts:
        F = shift['F']
        freq_MHz = shift['frequency_MHz']
        
        # Draw horizontal line for energy level
        ax.hlines(freq_MHz, 0, 1, colors='blue', linewidth=2)
        ax.text(1.05, freq_MHz, f'F = {F}', va='center', fontsize=12)
        ax.text(-0.05, freq_MHz, f'{freq_MHz:.2f} MHz', va='center', ha='right', fontsize=10)
    
    # Draw transition arrow
    if len(energy_shifts) >= 2:
        E_max = max(s['frequency_MHz'] for s in energy_shifts)
        E_min = min(s['frequency_MHz'] for s in energy_shifts)
        
        ax.annotate('', xy=(0.5, E_max), xytext=(0.5, E_min),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        
        transition_freq = result['result']
        ax.text(0.5, (E_max + E_min) / 2, f'Δν = {transition_freq:.2f} MHz',
               ha='center', va='center', fontsize=12, color='red',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    
    # Labels and title
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylabel('Relative Frequency (MHz)', fontsize=14)
    ax.set_title(f'Hyperfine Structure: Electron Spin = {electron_spin}, Nuclear Spin = {nuclear_spin}',
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {output_path}")
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'png',
            'electron_spin': electron_spin,
            'nuclear_spin': nuclear_spin,
            'num_levels': len(energy_shifts),
            'transition_frequency_MHz': result['result']
        }
    }


def visualize_spin_comparison(
    electron_spins: List[float],
    nuclear_spin: float = 0.5,
    output_path: str = "./tool_images/spin_comparison.png"
) -> Dict[str, str]:
    """
    Create a comparison plot of hyperfine frequencies for different electron spins.
    
    Args:
        electron_spins: List of electron spin values
        nuclear_spin: Nuclear spin quantum number
        output_path: Path to save the figure
    
    Returns:
        dict: Contains file path and plot metadata
    """
    import matplotlib.pyplot as plt
    
    # Set font to support both Chinese and English
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Get comparison data
    comparison = compare_hyperfine_frequencies(electron_spins, nuclear_spin)
    results = comparison['result']
    
    # Extract data for plotting
    spins = [r['electron_spin'] for r in results]
    freqs = [r['frequency_MHz'] for r in results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Frequency vs Spin
    ax1.plot(spins, freqs, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Electron Spin', fontsize=12)
    ax1.set_ylabel('Hyperfine Transition Frequency (MHz)', fontsize=12)
    ax1.set_title('Hyperfine Frequency vs Electron Spin', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotate points
    for spin, freq in zip(spins, freqs):
        ax1.annotate(f'{freq:.1f} MHz', xy=(spin, freq), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot 2: Scaling factor
    if 'scaling_factor' in results[0]:
        scaling = [r['scaling_factor'] for r in results]
        ax2.bar(range(len(spins)), scaling, color='green', alpha=0.7)
        ax2.set_xticks(range(len(spins)))
        ax2.set_xticklabels([f'{s}' for s in spins])
        ax2.set_xlabel('Electron Spin', fontsize=12)
        ax2.set_ylabel('Scaling Factor (relative to s=1/2)', fontsize=12)
        ax2.set_title('Frequency Scaling with Electron Spin', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Standard H (s=1/2)')
        ax2.legend()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {output_path}")
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'png',
            'electron_spins': electron_spins,
            'nuclear_spin': nuclear_spin,
            'num_comparisons': len(results)
        }
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstrate the atomic physics toolkit with three scenarios.
    
    Scenario 1: Solve the original problem (electron spin = 3/2)
    Scenario 2: Compare different electron spin values
    Scenario 3: Visualize hyperfine structure for modified spin
    """
    
    print("=" * 80)
    print("ATOMIC PHYSICS TOOLKIT - HYPERFINE STRUCTURE CALCULATIONS")
    print("=" * 80)
    print()
    
    # ========================================================================
    # SCENARIO 1: Original Problem - Electron Spin = 3/2
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 1: Hyperfine Transition with Modified Electron Spin (s = 3/2)")
    print("=" * 80)
    print("Problem: Calculate the hyperfine transition frequency in ground state")
    print("         hydrogen if the electron spin was 3/2 instead of 1/2.")
    print("         Nuclear spin (proton) remains 1/2.")
    print("-" * 80)
    print()
    
    # Step 1: Define quantum numbers
    electron_spin_modified = 1.5  # 3/2
    nuclear_spin_proton = 0.5     # 1/2
    
    print(f"Step 1: Define quantum numbers")
    print(f"  - Electron spin: s = {electron_spin_modified} (modified from standard 1/2)")
    print(f"  - Nuclear spin: I = {nuclear_spin_proton} (proton)")
    print(f"  - Ground state: n=1, l=0 (1s orbital)")
    print()
    
    # Step 2: Calculate hyperfine transition frequency
    print(f"Step 2: Calculate hyperfine transition frequency")
    print(f"  Calling: calculate_hyperfine_transition_frequency()")
    result_scenario1 = calculate_hyperfine_transition_frequency(
        electron_spin=electron_spin_modified,
        nuclear_spin=nuclear_spin_proton,
        n=1,
        l=0
    )
    print()
    
    # Step 3: Extract and display results
    frequency_MHz = result_scenario1['result']
    metadata = result_scenario1['metadata']
    
    print(f"Step 3: Analysis of results")
    print(f"  - Total angular momentum J = {metadata['J']}")
    print(f"  - Nuclear spin I = {metadata['I']}")
    print(f"  - Possible F states: {metadata['F_states']}")
    print(f"  - Transition: F = {metadata['F_upper']} ↔ F = {metadata['F_lower']}")
    print(f"  - Hyperfine constant A = {metadata['A_constant_MHz']:.4f} MHz")
    print()
    
    print(f"Energy level details:")
    for shift in metadata['energy_shifts']:
        print(f"  F = {shift['F']}: {shift['frequency_MHz']:+.4f} MHz")
    print()
    
    print("=" * 80)
    print(f"FINAL_ANSWER: {frequency_MHz:.0f} MHz")
    print("=" * 80)
    print()
    print()
    
    # ========================================================================
    # SCENARIO 2: Comparison of Different Electron Spins
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 2: Systematic Comparison of Electron Spin Effects")
    print("=" * 80)
    print("Problem: How does the hyperfine transition frequency scale with")
    print("         different electron spin values?")
    print("-" * 80)
    print()
    
    # Step 1: Define spin values to compare
    spin_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    print(f"Step 1: Define electron spin values for comparison")
    print(f"  Spin values: {spin_values}")
    print(f"  (Standard hydrogen has s = 0.5)")
    print()
    
    # Step 2: Calculate frequencies for all spins
    print(f"Step 2: Calculate hyperfine frequencies for each spin")
    print(f"  Calling: compare_hyperfine_frequencies()")
    comparison_result = compare_hyperfine_frequencies(
        electron_spins=spin_values,
        nuclear_spin=nuclear_spin_proton
    )
    print()
    
    # Step 3: Display comparison table
    print(f"Step 3: Comparison table")
    print(f"{'Electron Spin':>15} | {'Frequency (MHz)':>18} | {'Scaling Factor':>15} | {'F States':>15}")
    print("-" * 80)
    
    for result in comparison_result['result']:
        spin = result['electron_spin']
        freq = result['frequency_MHz']
        scale = result.get('scaling_factor', 1.0)
        f_states = str(result['F_states'])
        print(f"{spin:>15.1f} | {freq:>18.2f} | {scale:>15.3f} | {f_states:>15}")
    print()
    
    # Step 4: Identify the s=3/2 case
    s_3_2_result = next(r for r in comparison_result['result'] if r['electron_spin'] == 1.5)
    
    print(f"Step 4: Verification for s = 3/2")
    print(f"  - Frequency: {s_3_2_result['frequency_MHz']:.2f} MHz")
    print(f"  - Scaling factor: {s_3_2_result['scaling_factor']:.3f}x standard hydrogen")
    print(f"  - F states: {s_3_2_result['F_states']}")
    print()
    
    print("=" * 80)
    print(f"FINAL_ANSWER: For s=3/2, frequency = {s_3_2_result['frequency_MHz']:.0f} MHz")
    print("=" * 80)
    print()
    print()
    
    # ========================================================================
    # SCENARIO 3: Visualization of Hyperfine Structure
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 3: Visualization of Hyperfine Energy Levels")
    print("=" * 80)
    print("Problem: Create visual representations of the hyperfine structure")
    print("         for modified electron spin scenarios.")
    print("-" * 80)
    print()
    
    # Step 1: Visualize energy levels for s=3/2
    print(f"Step 1: Generate energy level diagram for s = 3/2")
    print(f"  Calling: visualize_hyperfine_levels()")
    viz1_result = visualize_hyperfine_levels(
        electron_spin=1.5,
        nuclear_spin=0.5,
        output_path="./tool_images/hyperfine_levels_s32.png"
    )
    print()
    
    # Step 2: Create comparison plot
    print(f"Step 2: Generate comparison plot for multiple spins")
    print(f"  Calling: visualize_spin_comparison()")
    viz2_result = visualize_spin_comparison(
        electron_spins=[0.5, 1.0, 1.5, 2.0],
        nuclear_spin=0.5,
        output_path="./tool_images/spin_comparison.png"
    )
    print()
    
    # Step 3: Summary of visualizations
    print(f"Step 3: Visualization summary")
    print(f"  - Energy level diagram: {viz1_result['result']}")
    print(f"    * Shows F states and transition frequency")
    print(f"    * Electron spin: {viz1_result['metadata']['electron_spin']}")
    print(f"    * Transition frequency: {viz1_result['metadata']['transition_frequency_MHz']:.2f} MHz")
    print()
    print(f"  - Comparison plot: {viz2_result['result']}")
    print(f"    * Compares {viz2_result['metadata']['num_comparisons']} different spin values")
    print(f"    * Shows frequency scaling with electron spin")
    print()
    
    print("=" * 80)
    print(f"FINAL_ANSWER: Visualizations saved successfully")
    print("=" * 80)
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("TOOLKIT DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary of Results:")
    print(f"  1. Modified hydrogen (s=3/2): {frequency_MHz:.0f} MHz")
    print(f"  2. Standard hydrogen (s=1/2): {comparison_result['metadata']['standard_hydrogen_freq_MHz']:.2f} MHz")
    print(f"  3. Scaling factor: {frequency_MHz / comparison_result['metadata']['standard_hydrogen_freq_MHz']:.3f}x")
    print()
    print("Physical Interpretation:")
    print("  - Larger electron spin increases magnetic moment")
    print("  - Stronger hyperfine interaction leads to larger splitting")
    print("  - Transition frequency scales approximately with electron spin")
    print()
    print("Files Generated:")
    print(f"  - {viz1_result['result']}")
    print(f"  - {viz2_result['result']}")
    print()


if __name__ == "__main__":
    main()