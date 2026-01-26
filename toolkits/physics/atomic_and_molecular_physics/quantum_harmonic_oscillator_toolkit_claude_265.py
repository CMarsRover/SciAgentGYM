# Filename: quantum_harmonic_oscillator_toolkit.py

"""
Quantum Harmonic Oscillator Toolkit
====================================
A comprehensive toolkit for quantum harmonic oscillator calculations including:
- Wave function manipulation
- Ladder operator calculations
- Expectation value computations
- Energy eigenstate analysis
- Visualization of quantum states

Dependencies:
- numpy: Numerical computations
- scipy: Special functions (Hermite polynomials)
- matplotlib: Visualization
- sympy: Symbolic quantum mechanics
"""

import numpy as np
from scipy.special import hermite, factorial
from scipy.integrate import quad
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import json
import os

# Global constants
HBAR = 1.0  # Reduced Planck constant (normalized units)
OMEGA = 1.0  # Angular frequency (normalized units)
M = 1.0  # Mass (normalized units)

# Create output directories
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# LAYER 1: ATOMIC FUNCTIONS - Basic quantum harmonic oscillator operations
# ============================================================================

def calculate_normalization_constant(n: int) -> float:
    """
    Calculate the normalization constant for the n-th energy eigenstate.
    
    For quantum harmonic oscillator:
    N_n = (m*omega/(pi*hbar))^(1/4) * 1/sqrt(2^n * n!)
    
    Args:
        n: Quantum number (non-negative integer)
        
    Returns:
        dict: {'result': normalization_constant, 'metadata': {...}}
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError(f"Quantum number n must be non-negative, got {n}")
    
    # Calculate normalization constant
    alpha = np.sqrt(M * OMEGA / HBAR)
    N_n = (alpha / np.pi)**(0.25) * (1.0 / np.sqrt(2**n * factorial(n)))
    
    return {
        'result': float(N_n),
        'metadata': {
            'quantum_number': n,
            'alpha': float(alpha),
            'formula': 'N_n = (m*omega/(pi*hbar))^(1/4) * 1/sqrt(2^n * n!)'
        }
    }


def hermite_polynomial_value(n: int, x: float) -> float:
    """
    Calculate the value of the n-th Hermite polynomial at position x.
    
    Uses physicist's Hermite polynomials H_n(x).
    
    Args:
        n: Order of Hermite polynomial (non-negative integer)
        x: Position value
        
    Returns:
        dict: {'result': H_n(x), 'metadata': {...}}
    """
    if n < 0:
        raise ValueError(f"Hermite polynomial order must be non-negative, got {n}")
    
    H_n = hermite(n)
    value = float(H_n(x))
    
    return {
        'result': value,
        'metadata': {
            'order': n,
            'position': float(x),
            'polynomial_type': 'physicist_hermite'
        }
    }


def eigenstate_wavefunction(n: int, x: float) -> float:
    """
    Calculate the wave function value of the n-th energy eigenstate at position x.
    
    phi_n(x) = N_n * H_n(alpha*x) * exp(-alpha^2 * x^2 / 2)
    where alpha = sqrt(m*omega/hbar)
    
    Args:
        n: Quantum number (non-negative integer)
        x: Position value
        
    Returns:
        dict: {'result': phi_n(x), 'metadata': {...}}
    """
    if n < 0:
        raise ValueError(f"Quantum number must be non-negative, got {n}")
    
    # Get normalization constant
    norm_result = calculate_normalization_constant(n)
    N_n = norm_result['result']
    alpha = norm_result['metadata']['alpha']
    
    # Calculate Hermite polynomial value
    hermite_result = hermite_polynomial_value(n, alpha * x)
    H_n_value = hermite_result['result']
    
    # Calculate wave function
    phi_n = N_n * H_n_value * np.exp(-alpha**2 * x**2 / 2)
    
    return {
        'result': float(phi_n),
        'metadata': {
            'quantum_number': n,
            'position': float(x),
            'normalization': N_n,
            'hermite_value': H_n_value,
            'alpha': alpha
        }
    }


def ladder_operator_action(n: int, operator: str) -> Dict:
    """
    Calculate the action of ladder operators on energy eigenstates.
    
    Annihilation operator: a|n> = sqrt(n)|n-1>
    Creation operator: a†|n> = sqrt(n+1)|n+1>
    
    Args:
        n: Quantum number (non-negative integer)
        operator: 'a' for annihilation, 'a_dagger' for creation
        
    Returns:
        dict: {'result': {'coefficient': coeff, 'new_state': new_n}, 'metadata': {...}}
    """
    if n < 0:
        raise ValueError(f"Quantum number must be non-negative, got {n}")
    
    if operator not in ['a', 'a_dagger']:
        raise ValueError(f"Operator must be 'a' or 'a_dagger', got {operator}")
    
    if operator == 'a':
        # Annihilation operator
        if n == 0:
            coefficient = 0.0
            new_state = None
        else:
            coefficient = np.sqrt(n)
            new_state = n - 1
        operator_name = 'annihilation'
    else:
        # Creation operator
        coefficient = np.sqrt(n + 1)
        new_state = n + 1
        operator_name = 'creation'
    
    return {
        'result': {
            'coefficient': float(coefficient),
            'new_state': new_state
        },
        'metadata': {
            'original_state': n,
            'operator': operator_name,
            'formula': f"{operator}|{n}> = {coefficient:.4f}|{new_state}>" if new_state is not None else f"{operator}|0> = 0"
        }
    }


def energy_eigenvalue(n: int) -> float:
    """
    Calculate the energy eigenvalue for the n-th state.
    
    E_n = (n + 1/2) * hbar * omega
    
    Args:
        n: Quantum number (non-negative integer)
        
    Returns:
        dict: {'result': E_n, 'metadata': {...}}
    """
    if n < 0:
        raise ValueError(f"Quantum number must be non-negative, got {n}")
    
    E_n = (n + 0.5) * HBAR * OMEGA
    
    return {
        'result': float(E_n),
        'metadata': {
            'quantum_number': n,
            'hbar': HBAR,
            'omega': OMEGA,
            'formula': 'E_n = (n + 1/2) * hbar * omega'
        }
    }


# ============================================================================
# LAYER 2: COMPOSITE FUNCTIONS - Complex quantum operations
# ============================================================================

def parse_superposition_state(coefficients: List[float], states: List[int]) -> Dict:
    """
    Parse and validate a superposition state.
    
    Checks normalization and returns state information.
    
    Args:
        coefficients: List of complex coefficients (as [real, imag] pairs or floats)
        states: List of quantum numbers
        
    Returns:
        dict: {'result': {'normalized': bool, 'norm': float, 'states': [...]}, 'metadata': {...}}
    """
    if len(coefficients) != len(states):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match number of states ({len(states)})")
    
    # Calculate normalization
    norm_squared = sum(abs(c)**2 for c in coefficients)
    is_normalized = abs(norm_squared - 1.0) < 1e-6
    
    state_info = []
    for c, n in zip(coefficients, states):
        state_info.append({
            'coefficient': float(c),
            'quantum_number': int(n),
            'probability': float(abs(c)**2)
        })
    
    return {
        'result': {
            'normalized': is_normalized,
            'norm_squared': float(norm_squared),
            'states': state_info
        },
        'metadata': {
            'num_states': len(states),
            'normalization_check': 'passed' if is_normalized else 'failed'
        }
    }


def operator_expectation_value(coefficients: List[float], states: List[int], 
                               operator_matrix: List[List[float]]) -> Dict:
    """
    Calculate expectation value of an operator for a superposition state.
    
    <psi|O|psi> = sum_ij c_i* c_j <i|O|j>
    
    Args:
        coefficients: List of state coefficients
        states: List of quantum numbers
        operator_matrix: Matrix elements <i|O|j> as nested list
        
    Returns:
        dict: {'result': expectation_value, 'metadata': {...}}
    """
    n_states = len(states)
    
    if len(coefficients) != n_states:
        raise ValueError("Number of coefficients must match number of states")
    
    if len(operator_matrix) != n_states or any(len(row) != n_states for row in operator_matrix):
        raise ValueError(f"Operator matrix must be {n_states}x{n_states}")
    
    # Calculate expectation value
    expectation = 0.0
    for i in range(n_states):
        for j in range(n_states):
            expectation += coefficients[i] * coefficients[j] * operator_matrix[i][j]
    
    return {
        'result': float(expectation),
        'metadata': {
            'num_states': n_states,
            'states': states,
            'calculation': 'sum_ij c_i* c_j <i|O|j>'
        }
    }


def number_operator_matrix(states: List[int]) -> Dict:
    """
    Construct the number operator matrix N = a†a for given states.
    
    <m|N|n> = n * delta_mn
    
    Args:
        states: List of quantum numbers
        
    Returns:
        dict: {'result': matrix (as nested list), 'metadata': {...}}
    """
    n_states = len(states)
    matrix = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
    
    for i, n in enumerate(states):
        matrix[i][i] = float(n)
    
    return {
        'result': matrix,
        'metadata': {
            'operator': 'number_operator',
            'dimension': n_states,
            'states': states,
            'diagonal_elements': [float(n) for n in states]
        }
    }


def hamiltonian_operator_matrix(states: List[int]) -> Dict:
    """
    Construct the Hamiltonian operator matrix H = (N + 1/2)ℏω.
    
    This is equivalent to (a†a + 1/2)ℏω.
    
    Args:
        states: List of quantum numbers
        
    Returns:
        dict: {'result': matrix (as nested list), 'metadata': {...}}
    """
    n_states = len(states)
    matrix = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
    
    for i, n in enumerate(states):
        matrix[i][i] = (n + 0.5) * HBAR * OMEGA
    
    return {
        'result': matrix,
        'metadata': {
            'operator': 'hamiltonian',
            'dimension': n_states,
            'states': states,
            'formula': 'H = (N + 1/2)ℏω = (a†a + 1/2)ℏω',
            'eigenvalues': [(n + 0.5) * HBAR * OMEGA for n in states]
        }
    }


def calculate_operator_aa_dagger_expectation(coefficients: List[float], 
                                             states: List[int]) -> Dict:
    """
    Calculate expectation value of aa† operator.
    
    aa†|n> = (N + 1)|n> where N is the number operator.
    So <psi|aa†|psi> = <psi|(N + 1)|psi> = <N> + 1
    
    Args:
        coefficients: List of state coefficients
        states: List of quantum numbers
        
    Returns:
        dict: {'result': <aa†>, 'metadata': {...}}
    """
    # aa† = N + 1, so we need <N> + 1
    n_states = len(states)
    
    # Calculate <N>
    expectation_N = 0.0
    for i, (c, n) in enumerate(zip(coefficients, states)):
        expectation_N += abs(c)**2 * n
    
    expectation_aa_dagger = expectation_N + 1.0
    
    return {
        'result': float(expectation_aa_dagger),
        'metadata': {
            'expectation_N': float(expectation_N),
            'formula': '<aa†> = <N> + 1',
            'states': states,
            'probabilities': [abs(c)**2 for c in coefficients]
        }
    }


def calculate_hamiltonian_expectation(coefficients: List[float], 
                                     states: List[int]) -> Dict:
    """
    Calculate expectation value of Hamiltonian H = (aa† + 1/2)ℏω.
    
    Since aa† = N + 1, we have:
    H = (N + 1 + 1/2)ℏω = (N + 3/2)ℏω
    
    But the standard form is H = (a†a + 1/2)ℏω = (N + 1/2)ℏω
    
    For the operator (aa† + 1/2)ℏω:
    <H> = (<N> + 1 + 1/2)ℏω = (<N> + 3/2)ℏω
    
    Args:
        coefficients: List of state coefficients
        states: List of quantum numbers
        
    Returns:
        dict: {'result': <H>, 'metadata': {...}}
    """
    # Calculate <N>
    expectation_N = 0.0
    for c, n in zip(coefficients, states):
        expectation_N += abs(c)**2 * n
    
    # For (aa† + 1/2)ℏω = (N + 1 + 1/2)ℏω = (N + 3/2)ℏω
    expectation_H = (expectation_N + 1.5) * HBAR * OMEGA
    
    return {
        'result': float(expectation_H),
        'metadata': {
            'expectation_N': float(expectation_N),
            'formula': '<aa† + 1/2>ℏω = (<N> + 3/2)ℏω',
            'hbar_omega': HBAR * OMEGA,
            'states': states,
            'individual_contributions': [
                {
                    'state': n,
                    'coefficient': c,
                    'probability': abs(c)**2,
                    'contribution': abs(c)**2 * (n + 1.5) * HBAR * OMEGA
                }
                for c, n in zip(coefficients, states)
            ]
        }
    }


def superposition_wavefunction_values(coefficients: List[float], 
                                      states: List[int], 
                                      x_values: List[float]) -> Dict:
    """
    Calculate wave function values for a superposition state.
    
    psi(x) = sum_n c_n * phi_n(x)
    
    Args:
        coefficients: List of state coefficients
        states: List of quantum numbers
        x_values: List of position values
        
    Returns:
        dict: {'result': list of psi(x) values, 'metadata': {...}}
    """
    psi_values = []
    
    for x in x_values:
        psi_x = 0.0
        for c, n in zip(coefficients, states):
            phi_result = eigenstate_wavefunction(n, x)
            psi_x += c * phi_result['result']
        psi_values.append(float(psi_x))
    
    return {
        'result': psi_values,
        'metadata': {
            'num_points': len(x_values),
            'x_range': [float(min(x_values)), float(max(x_values))],
            'num_states': len(states),
            'states': states
        }
    }


# ============================================================================
# LAYER 3: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_energy_eigenstates(states: List[int], x_range: Tuple[float, float], 
                           num_points: int = 500) -> Dict:
    """
    Plot energy eigenstates of quantum harmonic oscillator.
    
    Args:
        states: List of quantum numbers to plot
        x_range: Tuple of (x_min, x_max)
        num_points: Number of points for plotting
        
    Returns:
        dict: {'result': filepath, 'metadata': {...}}
    """
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    for n in states:
        phi_values = []
        for x in x_values:
            result = eigenstate_wavefunction(n, x)
            phi_values.append(result['result'])
        
        # Offset by energy level for visualization
        energy = energy_eigenvalue(n)['result']
        plt.plot(x_values, np.array(phi_values) + energy, 
                label=f'n={n}, E={energy:.2f}ℏω')
    
    # Plot potential
    V = 0.5 * M * OMEGA**2 * x_values**2
    plt.plot(x_values, V, 'k--', linewidth=2, label='Potential V(x)')
    
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel('Energy / Wave function', fontsize=12)
    plt.title('Quantum Harmonic Oscillator Energy Eigenstates', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filepath = './tool_images/energy_eigenstates.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'states_plotted': states,
            'x_range': x_range,
            'num_points': num_points
        }
    }


def plot_superposition_state(coefficients: List[float], states: List[int],
                            x_range: Tuple[float, float], num_points: int = 500) -> Dict:
    """
    Plot superposition state wave function and probability density.
    
    Args:
        coefficients: List of state coefficients
        states: List of quantum numbers
        x_range: Tuple of (x_min, x_max)
        num_points: Number of points for plotting
        
    Returns:
        dict: {'result': filepath, 'metadata': {...}}
    """
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate superposition wave function
    result = superposition_wavefunction_values(coefficients, states, x_values.tolist())
    psi_values = np.array(result['result'])
    prob_density = psi_values**2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot wave function
    ax1.plot(x_values, psi_values, 'b-', linewidth=2, label='ψ(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Wave function ψ(x)', fontsize=12)
    ax1.set_title('Superposition State Wave Function', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot probability density
    ax2.fill_between(x_values, prob_density, alpha=0.5, color='red', label='|ψ(x)|²')
    ax2.plot(x_values, prob_density, 'r-', linewidth=2)
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Probability density |ψ(x)|²', fontsize=12)
    ax2.set_title('Probability Density', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add state composition text
    state_text = 'State composition:\n'
    for c, n in zip(coefficients, states):
        state_text += f'c_{n} = {c:.4f}, |c_{n}|² = {abs(c)**2:.4f}\n'
    ax2.text(0.02, 0.98, state_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    filepath = './tool_images/superposition_state.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'states': states,
            'coefficients': [float(c) for c in coefficients],
            'x_range': x_range,
            'num_points': num_points
        }
    }


def plot_expectation_values(coefficients: List[float], states: List[int]) -> Dict:
    """
    Plot expectation values of various operators for the superposition state.
    
    Args:
        coefficients: List of state coefficients
        states: List of quantum numbers
        
    Returns:
        dict: {'result': filepath, 'metadata': {...}}
    """
    # Calculate various expectation values
    operators = []
    values = []
    
    # <N>
    expectation_N = sum(abs(c)**2 * n for c, n in zip(coefficients, states))
    operators.append('<N>')
    values.append(expectation_N)
    
    # <aa†>
    result_aa_dagger = calculate_operator_aa_dagger_expectation(coefficients, states)
    operators.append('<aa†>')
    values.append(result_aa_dagger['result'])
    
    # <H> for standard Hamiltonian (a†a + 1/2)ℏω
    expectation_H_standard = (expectation_N + 0.5) * HBAR * OMEGA
    operators.append('<H_standard>')
    values.append(expectation_H_standard)
    
    # <H> for (aa† + 1/2)ℏω
    result_H = calculate_hamiltonian_expectation(coefficients, states)
    operators.append('<(aa†+1/2)ℏω>')
    values.append(result_H['result'])
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    bars = plt.bar(operators, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.ylabel('Expectation Value', fontsize=12)
    plt.title('Operator Expectation Values for Superposition State', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add state information
    state_text = 'State: ' + ' + '.join([f'{c:.4f}|{n}>' for c, n in zip(coefficients, states)])
    plt.text(0.5, 0.95, state_text, transform=plt.gca().transAxes,
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    filepath = './tool_images/expectation_values.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'operators': operators,
            'values': [float(v) for v in values],
            'states': states,
            'coefficients': [float(c) for c in coefficients]
        }
    }


# ============================================================================
# MAIN FUNCTION - Demonstration scenarios
# ============================================================================

def main():
    """
    Demonstrate the quantum harmonic oscillator toolkit with three scenarios.
    """
    
    print("=" * 80)
    print("QUANTUM HARMONIC OSCILLATOR TOOLKIT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # SCENARIO 1: Solve the original problem
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 1: Calculate <(aa† + 1/2)ℏω> for given superposition state")
    print("=" * 80)
    print("Problem: A system with wave function at t=0:")
    print("ψ(x) = (1/√3)φ₀(x) + (1/√2)φ₁(x) + (1/√6)φ₃(x)")
    print("Calculate the expectation value of (aa† + 1/2)ℏω")
    print("-" * 80)
    
    # Step 1: Define the superposition state
    print("\nStep 1: Parse and validate the superposition state")
    coefficients_1 = [1/np.sqrt(3), 1/np.sqrt(2), 1/np.sqrt(6)]
    states_1 = [0, 1, 3]
    
    # Function call: parse_superposition_state()
    parse_result = parse_superposition_state(coefficients_1, states_1)
    print(f"FUNCTION_CALL: parse_superposition_state | PARAMS: {{'coefficients': {coefficients_1}, 'states': {states_1}}} | RESULT: {parse_result}")
    
    print(f"\nNormalization check: {parse_result['result']['normalized']}")
    print(f"Norm squared: {parse_result['result']['norm_squared']:.6f}")
    print("\nState probabilities:")
    for state_info in parse_result['result']['states']:
        print(f"  |{state_info['quantum_number']}>: coefficient = {state_info['coefficient']:.4f}, "
              f"probability = {state_info['probability']:.4f}")
    
    # Step 2: Calculate <N> (number operator expectation)
    print("\nStep 2: Calculate expectation value of number operator <N>")
    expectation_N = sum(abs(c)**2 * n for c, n in zip(coefficients_1, states_1))
    print(f"<N> = {expectation_N:.6f}")
    
    # Step 3: Calculate <aa†>
    print("\nStep 3: Calculate expectation value of aa† operator")
    # Function call: calculate_operator_aa_dagger_expectation()
    aa_dagger_result = calculate_operator_aa_dagger_expectation(coefficients_1, states_1)
    print(f"FUNCTION_CALL: calculate_operator_aa_dagger_expectation | PARAMS: {{'coefficients': {coefficients_1}, 'states': {states_1}}} | RESULT: {aa_dagger_result}")
    
    print(f"<aa†> = <N> + 1 = {aa_dagger_result['result']:.6f}")
    
    # Step 4: Calculate <(aa† + 1/2)ℏω>
    print("\nStep 4: Calculate expectation value of (aa† + 1/2)ℏω")
    # Function call: calculate_hamiltonian_expectation()
    hamiltonian_result = calculate_hamiltonian_expectation(coefficients_1, states_1)
    print(f"FUNCTION_CALL: calculate_hamiltonian_expectation | PARAMS: {{'coefficients': {coefficients_1}, 'states': {states_1}}} | RESULT: {hamiltonian_result}")
    
    print(f"\n<(aa† + 1/2)ℏω> = (<N> + 3/2)ℏω")
    print(f"                = ({expectation_N:.6f} + 1.5) × {HBAR * OMEGA}")
    print(f"                = {hamiltonian_result['result']:.6f}ℏω")
    
    # Verify against standard answer
    expected_answer = 1.5 * HBAR * OMEGA
    print(f"\nExpected answer: (3/2)ℏω = {expected_answer:.6f}ℏω")
    print(f"Calculated answer: {hamiltonian_result['result']:.6f}ℏω")
    print(f"Match: {abs(hamiltonian_result['result'] - expected_answer) < 1e-6}")
    
    # Step 5: Visualize the superposition state
    print("\nStep 5: Visualize the superposition state")
    # Function call: plot_superposition_state()
    plot_result_1 = plot_superposition_state(coefficients_1, states_1, (-5, 5))
    print(f"FUNCTION_CALL: plot_superposition_state | PARAMS: {{'coefficients': {coefficients_1}, 'states': {states_1}, 'x_range': (-5, 5)}} | RESULT: {plot_result_1}")
    
    # Step 6: Visualize expectation values
    print("\nStep 6: Visualize operator expectation values")
    # Function call: plot_expectation_values()
    plot_result_2 = plot_expectation_values(coefficients_1, states_1)
    print(f"FUNCTION_CALL: plot_expectation_values | PARAMS: {{'coefficients': {coefficients_1}, 'states': {states_1}}} | RESULT: {plot_result_2}")
    
    print(f"\nFINAL_ANSWER: {hamiltonian_result['result']:.6f}ℏω = (3/2)ℏω")
    
    # ========================================================================
    # SCENARIO 2: Compare different operators for equal superposition
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 2: Compare operators for equal superposition of ground and first excited state")
    print("=" * 80)
    print("Problem: For state ψ = (1/√2)|0> + (1/√2)|1>")
    print("Compare <N>, <aa†>, <a†a>, and <(aa† + 1/2)ℏω>")
    print("-" * 80)
    
    # Step 1: Define equal superposition state
    print("\nStep 1: Define equal superposition state")
    coefficients_2 = [1/np.sqrt(2), 1/np.sqrt(2)]
    states_2 = [0, 1]
    
    # Function call: parse_superposition_state()
    parse_result_2 = parse_superposition_state(coefficients_2, states_2)
    print(f"FUNCTION_CALL: parse_superposition_state | PARAMS: {{'coefficients': {coefficients_2}, 'states': {states_2}}} | RESULT: {parse_result_2}")
    
    # Step 2: Calculate <N>
    print("\nStep 2: Calculate <N>")
    expectation_N_2 = sum(abs(c)**2 * n for c, n in zip(coefficients_2, states_2))
    print(f"<N> = {expectation_N_2:.6f}")
    
    # Step 3: Calculate <aa†>
    print("\nStep 3: Calculate <aa†>")
    # Function call: calculate_operator_aa_dagger_expectation()
    aa_dagger_result_2 = calculate_operator_aa_dagger_expectation(coefficients_2, states_2)
    print(f"FUNCTION_CALL: calculate_operator_aa_dagger_expectation | PARAMS: {{'coefficients': {coefficients_2}, 'states': {states_2}}} | RESULT: {aa_dagger_result_2}")
    
    # Step 4: Calculate <(aa† + 1/2)ℏω>
    print("\nStep 4: Calculate <(aa† + 1/2)ℏω>")
    # Function call: calculate_hamiltonian_expectation()
    hamiltonian_result_2 = calculate_hamiltonian_expectation(coefficients_2, states_2)
    print(f"FUNCTION_CALL: calculate_hamiltonian_expectation | PARAMS: {{'coefficients': {coefficients_2}, 'states': {states_2}}} | RESULT: {hamiltonian_result_2}")
    
    # Step 5: Calculate standard Hamiltonian <(a†a + 1/2)ℏω>
    print("\nStep 5: Calculate standard Hamiltonian <(a†a + 1/2)ℏω>")
    hamiltonian_standard_2 = (expectation_N_2 + 0.5) * HBAR * OMEGA
    print(f"<(a†a + 1/2)ℏω> = {hamiltonian_standard_2:.6f}ℏω")
    
    print("\nComparison:")
    print(f"  <N> = {expectation_N_2:.6f}")
    print(f"  <aa†> = {aa_dagger_result_2['result']:.6f}")
    print(f"  <a†a> = <N> = {expectation_N_2:.6f}")
    print(f"  <(aa† + 1/2)ℏω> = {hamiltonian_result_2['result']:.6f}ℏω")
    print(f"  <(a†a + 1/2)ℏω> = {hamiltonian_standard_2:.6f}ℏω")
    
    # Step 6: Visualize
    print("\nStep 6: Visualize expectation values")
    # Function call: plot_expectation_values()
    plot_result_3 = plot_expectation_values(coefficients_2, states_2)
    print(f"FUNCTION_CALL: plot_expectation_values | PARAMS: {{'coefficients': {coefficients_2}, 'states': {states_2}}} | RESULT: {plot_result_3}")
    
    print(f"\nFINAL_ANSWER: <(aa† + 1/2)ℏω> = {hamiltonian_result_2['result']:.6f}ℏω")
    
    # ========================================================================
    # SCENARIO 3: Analyze ladder operator actions on pure states
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 3: Analyze ladder operator actions and energy calculations")
    print("=" * 80)
    print("Problem: For pure state |2>, calculate:")
    print("1. Action of a and a† operators")
    print("2. Expectation values <aa†> and <a†a>")
    print("3. Energy eigenvalue and verify with operator expectation")
    print("-" * 80)
    
    # Step 1: Define pure state |2>
    print("\nStep 1: Analyze pure state |2>")
    n_state = 2
    coefficients_3 = [1.0]
    states_3 = [2]
    
    # Step 2: Calculate ladder operator actions
    print("\nStep 2: Calculate ladder operator actions")
    # Function call: ladder_operator_action() for annihilation
    a_action = ladder_operator_action(n_state, 'a')
    print(f"FUNCTION_CALL: ladder_operator_action | PARAMS: {{'n': {n_state}, 'operator': 'a'}} | RESULT: {a_action}")
    print(f"a|2> = {a_action['result']['coefficient']:.4f}|{a_action['result']['new_state']}>")
    
    # Function call: ladder_operator_action() for creation
    a_dagger_action = ladder_operator_action(n_state, 'a_dagger')
    print(f"FUNCTION_CALL: ladder_operator_action | PARAMS: {{'n': {n_state}, 'operator': 'a_dagger'}} | RESULT: {a_dagger_action}")
    print(f"a†|2> = {a_dagger_action['result']['coefficient']:.4f}|{a_dagger_action['result']['new_state']}>")
    
    # Step 3: Calculate <aa†> for pure state
    print("\nStep 3: Calculate <aa†> for pure state |2>")
    # Function call: calculate_operator_aa_dagger_expectation()
    aa_dagger_result_3 = calculate_operator_aa_dagger_expectation(coefficients_3, states_3)
    print(f"FUNCTION_CALL: calculate_operator_aa_dagger_expectation | PARAMS: {{'coefficients': {coefficients_3}, 'states': {states_3}}} | RESULT: {aa_dagger_result_3}")
    print(f"<2|aa†|2> = <2|N+1|2> = {aa_dagger_result_3['result']:.6f}")
    
    # Step 4: Calculate <a†a> for pure state
    print("\nStep 4: Calculate <a†a> for pure state |2>")
    expectation_a_dagger_a = n_state  # For pure state |n>, <a†a> = n
    print(f"<2|a†a|2> = <2|N|2> = {expectation_a_dagger_a}")
    
    # Step 5: Calculate energy eigenvalue
    print("\nStep 5: Calculate energy eigenvalue")
    # Function call: energy_eigenvalue()
    energy_result = energy_eigenvalue(n_state)
    print(f"FUNCTION_CALL: energy_eigenvalue | PARAMS: {{'n': {n_state}}} | RESULT: {energy_result}")
    print(f"E_2 = (2 + 1/2)ℏω = {energy_result['result']:.6f}ℏω")
    
    # Step 6: Verify with operator expectation
    print("\nStep 6: Verify energy with <(aa† + 1/2)ℏω>")
    # Function call: calculate_hamiltonian_expectation()
    hamiltonian_result_3 = calculate_hamiltonian_expectation(coefficients_3, states_3)
    print(f"FUNCTION_CALL: calculate_hamiltonian_expectation | PARAMS: {{'coefficients': {coefficients_3}, 'states': {states_3}}} | RESULT: {hamiltonian_result_3}")
    print(f"<(aa† + 1/2)ℏω> = (2 + 3/2)ℏω = {hamiltonian_result_3['result']:.6f}ℏω")
    
    # Step 7: Visualize energy eigenstates
    print("\nStep 7: Visualize energy eigenstates")
    # Function call: plot_energy_eigenstates()
    plot_result_4 = plot_energy_eigenstates([0, 1, 2, 3], (-4, 4))
    print(f"FUNCTION_CALL: plot_energy_eigenstates | PARAMS: {{'states': [0, 1, 2, 3], 'x_range': (-4, 4)}} | RESULT: {plot_result_4}")
    
    print("\nVerification:")
    print(f"  <aa†> = {aa_dagger_result_3['result']:.6f}")
    print(f"  <a†a> = {expectation_a_dagger_a}")
    print(f"  E_2 = {energy_result['result']:.6f}ℏω")
    print(f"  <(aa† + 1/2)ℏω> = {hamiltonian_result_3['result']:.6f}ℏω")
    print(f"  Note: For pure state |2>, <(aa† + 1/2)ℏω> = (2 + 3/2)ℏω = 3.5ℏω")
    
    print(f"\nFINAL_ANSWER: For pure state |2>, <(aa† + 1/2)ℏω> = {hamiltonian_result_3['result']:.6f}ℏω")
    
    print("\n" + "=" * 80)
    print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()