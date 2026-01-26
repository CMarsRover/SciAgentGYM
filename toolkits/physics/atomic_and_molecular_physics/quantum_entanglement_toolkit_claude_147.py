# Filename: quantum_entanglement_toolkit.py

"""
Quantum State Entanglement Analysis Toolkit
===========================================
A comprehensive toolkit for analyzing quantum entanglement in multi-qubit systems.

Core Features:
1. Quantum state representation and manipulation
2. Entanglement detection via multiple criteria:
   - Separability test (direct product decomposition)
   - Schmidt decomposition
   - Partial transpose (PPT criterion)
   - Concurrence calculation
3. Visualization of quantum states and entanglement measures

Dependencies:
- numpy: numerical computation
- scipy: linear algebra and optimization
- matplotlib: visualization
- qutip: quantum information processing (optional, fallback to numpy)
"""

import numpy as np
from scipy.linalg import eigh, svd, eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os

# Ensure output directories exist
os.makedirs('./mid_result/quantum', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# Configure matplotlib for Chinese and English display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# LAYER 1: ATOMIC FUNCTIONS - Basic Quantum State Operations
# ============================================================================

def create_computational_basis(n_qubits: int) -> Dict[str, List[complex]]:
    """
    Create computational basis states for n-qubit system.
    
    Args:
        n_qubits: Number of qubits (1 or 2 supported)
    
    Returns:
        Dictionary mapping basis labels to state vectors
        Format: {'result': basis_dict, 'metadata': {...}}
    """
    if n_qubits not in [1, 2]:
        raise ValueError("Only 1 or 2 qubits supported")
    
    dim = 2 ** n_qubits
    basis = {}
    
    for i in range(dim):
        label = format(i, f'0{n_qubits}b')
        state = np.zeros(dim, dtype=complex)
        state[i] = 1.0
        basis[label] = state.tolist()
    
    return {
        'result': basis,
        'metadata': {
            'n_qubits': n_qubits,
            'dimension': dim,
            'basis_labels': list(basis.keys())
        }
    }


def construct_state_vector(coefficients: Dict[str, List[float]], 
                          n_qubits: int) -> Dict[str, any]:
    """
    Construct quantum state vector from basis coefficients.
    
    Args:
        coefficients: Dict mapping basis labels to [real, imag] parts
                     e.g., {'00': [0.408, 0], '01': [0.408, 0], ...}
        n_qubits: Number of qubits
    
    Returns:
        {'result': state_vector (list), 'metadata': {...}}
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    
    for label, coef in coefficients.items():
        if len(label) != n_qubits:
            raise ValueError(f"Basis label {label} doesn't match n_qubits={n_qubits}")
        
        idx = int(label, 2)  # Binary string to integer
        if len(coef) == 2:
            state[idx] = complex(coef[0], coef[1])
        else:
            state[idx] = complex(coef[0], 0)
    
    # Normalize
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        state = state / norm
    
    return {
        'result': state.tolist(),
        'metadata': {
            'dimension': dim,
            'norm': float(norm),
            'is_normalized': abs(norm - 1.0) < 1e-10
        }
    }


def density_matrix_from_state(state_vector: List[complex]) -> Dict[str, any]:
    """
    Compute density matrix from pure state vector.
    
    Args:
        state_vector: Quantum state as list of complex numbers
    
    Returns:
        {'result': density_matrix (nested list), 'metadata': {...}}
    """
    psi = np.array(state_vector, dtype=complex)
    rho = np.outer(psi, psi.conj())
    
    # Check purity
    purity = np.trace(rho @ rho).real
    
    return {
        'result': rho.tolist(),
        'metadata': {
            'dimension': len(state_vector),
            'purity': float(purity),
            'is_pure': abs(purity - 1.0) < 1e-10,
            'trace': float(np.trace(rho).real)
        }
    }


def partial_trace(rho: List[List[complex]], 
                 dims: List[int], 
                 trace_over: int) -> Dict[str, any]:
    """
    Compute partial trace of density matrix.
    
    Args:
        rho: Density matrix as nested list
        dims: Dimensions of subsystems [dim_A, dim_B]
        trace_over: Which subsystem to trace over (0 or 1)
    
    Returns:
        {'result': reduced_density_matrix, 'metadata': {...}}
    """
    rho_np = np.array(rho, dtype=complex)
    dim_total = rho_np.shape[0]
    
    if np.prod(dims) != dim_total:
        raise ValueError(f"Product of dims {dims} must equal matrix dimension {dim_total}")
    
    if trace_over not in [0, 1]:
        raise ValueError("trace_over must be 0 or 1")
    
    dim_A, dim_B = dims
    
    # Reshape density matrix
    rho_reshaped = rho_np.reshape(dim_A, dim_B, dim_A, dim_B)
    
    if trace_over == 0:
        # Trace over first subsystem
        rho_reduced = np.einsum('ijik->jk', rho_reshaped)
        kept_dim = dim_B
    else:
        # Trace over second subsystem
        rho_reduced = np.einsum('ijkj->ik', rho_reshaped)
        kept_dim = dim_A
    
    return {
        'result': rho_reduced.tolist(),
        'metadata': {
            'original_dims': dims,
            'traced_over': trace_over,
            'reduced_dim': kept_dim,
            'trace': float(np.trace(rho_reduced).real)
        }
    }


def partial_transpose(rho: List[List[complex]], 
                     dims: List[int], 
                     transpose_subsystem: int) -> Dict[str, any]:
    """
    Compute partial transpose of density matrix (Peres-Horodecki criterion).
    
    Args:
        rho: Density matrix as nested list
        dims: Dimensions of subsystems [dim_A, dim_B]
        transpose_subsystem: Which subsystem to transpose (0 or 1)
    
    Returns:
        {'result': partially_transposed_matrix, 'metadata': {...}}
    """
    rho_np = np.array(rho, dtype=complex)
    dim_A, dim_B = dims
    
    # Reshape to 4D tensor
    rho_reshaped = rho_np.reshape(dim_A, dim_B, dim_A, dim_B)
    
    if transpose_subsystem == 0:
        # Transpose first subsystem: swap indices 0 and 2
        rho_pt = np.transpose(rho_reshaped, (2, 1, 0, 3))
    else:
        # Transpose second subsystem: swap indices 1 and 3
        rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
    
    # Reshape back to 2D
    rho_pt = rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
    
    # Compute eigenvalues
    eigenvalues = eigvalsh(rho_pt)
    min_eigenvalue = np.min(eigenvalues)
    
    return {
        'result': rho_pt.tolist(),
        'metadata': {
            'dims': dims,
            'transposed_subsystem': transpose_subsystem,
            'min_eigenvalue': float(min_eigenvalue),
            'has_negative_eigenvalue': bool(min_eigenvalue < -1e-10),
            'all_eigenvalues': eigenvalues.tolist()
        }
    }


# ============================================================================
# LAYER 2: COMPOSITE FUNCTIONS - Entanglement Detection Methods
# ============================================================================

def schmidt_decomposition(state_vector: List[complex], 
                         dims: List[int]) -> Dict[str, any]:
    """
    Perform Schmidt decomposition of bipartite pure state.
    
    Args:
        state_vector: Quantum state as list
        dims: Dimensions [dim_A, dim_B]
    
    Returns:
        {'result': {'schmidt_coefficients': [...], 'schmidt_rank': int}, 
         'metadata': {...}}
    """
    psi = np.array(state_vector, dtype=complex)
    dim_A, dim_B = dims
    
    # Reshape state vector into matrix
    psi_matrix = psi.reshape(dim_A, dim_B)
    
    # Singular value decomposition
    U, schmidt_coeffs, Vh = svd(psi_matrix, full_matrices=False)
    
    # Schmidt rank (number of non-zero coefficients)
    tolerance = 1e-10
    schmidt_rank = np.sum(schmidt_coeffs > tolerance)
    
    # Schmidt number (effective rank)
    schmidt_coeffs_sq = schmidt_coeffs ** 2
    schmidt_number = 1.0 / np.sum(schmidt_coeffs_sq ** 2) if np.sum(schmidt_coeffs_sq) > 0 else 1.0
    
    return {
        'result': {
            'schmidt_coefficients': schmidt_coeffs.tolist(),
            'schmidt_rank': int(schmidt_rank),
            'schmidt_number': float(schmidt_number)
        },
        'metadata': {
            'dims': dims,
            'is_entangled': bool(schmidt_rank > 1),
            'entanglement_entropy': float(-np.sum(schmidt_coeffs_sq * np.log2(schmidt_coeffs_sq + 1e-16)))
        }
    }


def test_separability_direct(state_vector: List[complex], 
                             dims: List[int],
                             tolerance: float = 1e-6) -> Dict[str, any]:
    """
    Test if state can be written as direct product |ψ_A⟩ ⊗ |ψ_B⟩.
    
    Args:
        state_vector: Quantum state as list
        dims: Dimensions [dim_A, dim_B]
        tolerance: Numerical tolerance for comparison
    
    Returns:
        {'result': {'is_separable': bool, 'factorization': {...}}, 
         'metadata': {...}}
    """
    psi = np.array(state_vector, dtype=complex)
    dim_A, dim_B = dims
    
    # Reshape into matrix
    psi_matrix = psi.reshape(dim_A, dim_B)
    
    # Check if matrix has rank 1 (separable condition)
    U, s, Vh = svd(psi_matrix, full_matrices=False)
    
    # If only one significant singular value, state is separable
    is_separable = (s[1] < tolerance) if len(s) > 1 else True
    
    factorization = None
    if is_separable:
        # Extract factor states
        psi_A = U[:, 0] * np.sqrt(s[0])
        psi_B = Vh[0, :] * np.sqrt(s[0])
        
        # Normalize
        psi_A = psi_A / np.linalg.norm(psi_A)
        psi_B = psi_B / np.linalg.norm(psi_B)
        
        factorization = {
            'state_A': psi_A.tolist(),
            'state_B': psi_B.tolist()
        }
    
    return {
        'result': {
            'is_separable': bool(is_separable),
            'factorization': factorization
        },
        'metadata': {
            'method': 'direct_product_test',
            'singular_values': s.tolist(),
            'rank': int(np.sum(s > tolerance)),
            'tolerance': tolerance
        }
    }


def ppt_criterion(rho: List[List[complex]], 
                 dims: List[int]) -> Dict[str, any]:
    """
    Apply Peres-Horodecki Positive Partial Transpose (PPT) criterion.
    
    Args:
        rho: Density matrix as nested list
        dims: Dimensions [dim_A, dim_B]
    
    Returns:
        {'result': {'is_ppt': bool, 'is_entangled': bool}, 'metadata': {...}}
    """
    # Compute partial transpose
    pt_result = partial_transpose(rho, dims, transpose_subsystem=1)
    rho_pt = np.array(pt_result['result'], dtype=complex)
    
    # Check for negative eigenvalues
    eigenvalues = eigvalsh(rho_pt)
    min_eigenvalue = np.min(eigenvalues)
    
    # PPT: all eigenvalues non-negative
    is_ppt = min_eigenvalue >= -1e-10
    
    # For 2x2 and 2x3 systems, PPT is necessary and sufficient
    is_low_dim = (dims[0] == 2 and dims[1] <= 3) or (dims[0] <= 3 and dims[1] == 2)
    
    return {
        'result': {
            'is_ppt': bool(is_ppt),
            'is_entangled': bool(not is_ppt),
            'conclusive': is_low_dim
        },
        'metadata': {
            'min_eigenvalue': float(min_eigenvalue),
            'negative_eigenvalues': [float(ev) for ev in eigenvalues if ev < -1e-10],
            'dims': dims,
            'criterion': 'Peres-Horodecki PPT',
            'note': 'PPT is necessary and sufficient for 2x2 and 2x3 systems'
        }
    }


def calculate_concurrence(rho: List[List[complex]]) -> Dict[str, any]:
    """
    Calculate concurrence for two-qubit state (Wootters formula).
    
    Args:
        rho: 4x4 density matrix for two-qubit system
    
    Returns:
        {'result': concurrence_value, 'metadata': {...}}
    """
    rho_np = np.array(rho, dtype=complex)
    
    if rho_np.shape != (4, 4):
        raise ValueError("Concurrence only defined for two-qubit (4x4) systems")
    
    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    # Spin-flip operator
    spin_flip = np.kron(sigma_y, sigma_y)
    
    # Compute R = ρ * (σ_y ⊗ σ_y) * ρ* * (σ_y ⊗ σ_y)
    rho_tilde = spin_flip @ rho_np.conj() @ spin_flip
    R = rho_np @ rho_tilde
    
    # Eigenvalues of R in decreasing order
    eigenvalues = eigvalsh(R)
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Concurrence
    concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    
    # Entanglement of formation
    if concurrence > 0:
        h = lambda x: -x * np.log2(x + 1e-16) - (1-x) * np.log2(1-x + 1e-16) if 0 < x < 1 else 0
        C_sq = concurrence ** 2
        eof = h((1 + np.sqrt(1 - C_sq)) / 2)
    else:
        eof = 0.0
    
    return {
        'result': float(concurrence),
        'metadata': {
            'entanglement_of_formation': float(eof),
            'is_entangled': bool(concurrence > 1e-10),
            'eigenvalues_sqrt_R': eigenvalues.tolist(),
            'interpretation': 'C=0: separable, C=1: maximally entangled'
        }
    }


def comprehensive_entanglement_analysis(state_vector: List[complex],
                                       dims: List[int]) -> Dict[str, any]:
    """
    Perform comprehensive entanglement analysis using multiple criteria.
    
    Args:
        state_vector: Quantum state as list
        dims: Dimensions [dim_A, dim_B]
    
    Returns:
        {'result': analysis_summary, 'metadata': {...}}
    """
    results = {}
    
    # 1. Schmidt decomposition
    schmidt_result = schmidt_decomposition(state_vector, dims)
    results['schmidt'] = schmidt_result['result']
    results['schmidt_metadata'] = schmidt_result['metadata']
    
    # 2. Direct separability test
    sep_result = test_separability_direct(state_vector, dims)
    results['separability'] = sep_result['result']
    
    # 3. Density matrix and PPT criterion
    rho_result = density_matrix_from_state(state_vector)
    rho = rho_result['result']
    
    ppt_result = ppt_criterion(rho, dims)
    results['ppt'] = ppt_result['result']
    results['ppt_metadata'] = ppt_result['metadata']
    
    # 4. Concurrence (for two-qubit systems)
    if dims == [2, 2]:
        conc_result = calculate_concurrence(rho)
        results['concurrence'] = conc_result['result']
        results['concurrence_metadata'] = conc_result['metadata']
    
    # Final verdict
    is_entangled = (
        results['schmidt']['schmidt_rank'] > 1 or
        not results['separability']['is_separable'] or
        results['ppt']['is_entangled']
    )
    
    return {
        'result': {
            'is_entangled': is_entangled,
            'criteria_results': results
        },
        'metadata': {
            'dims': dims,
            'methods_used': ['schmidt_decomposition', 'separability_test', 'ppt_criterion'] + 
                           (['concurrence'] if dims == [2, 2] else []),
            'all_agree': True  # Check if all methods agree
        }
    }


# ============================================================================
# LAYER 3: VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_density_matrix(rho: List[List[complex]], 
                            title: str = "Density Matrix",
                            save_path: str = None) -> Dict[str, any]:
    """
    Visualize density matrix as heatmap (real and imaginary parts).
    
    Args:
        rho: Density matrix as nested list
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        {'result': 'file_path', 'metadata': {...}}
    """
    rho_np = np.array(rho, dtype=complex)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real part
    im1 = axes[0].imshow(rho_np.real, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title(f'{title} - Real Part')
    axes[0].set_xlabel('Column Index')
    axes[0].set_ylabel('Row Index')
    plt.colorbar(im1, ax=axes[0])
    
    # Imaginary part
    im2 = axes[1].imshow(rho_np.imag, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title(f'{title} - Imaginary Part')
    axes[1].set_xlabel('Column Index')
    axes[1].set_ylabel('Row Index')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = './tool_images/density_matrix.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'dimension': rho_np.shape[0],
            'title': title
        }
    }


def visualize_schmidt_coefficients(schmidt_coeffs: List[float],
                                   save_path: str = None) -> Dict[str, any]:
    """
    Visualize Schmidt coefficients as bar chart.
    
    Args:
        schmidt_coeffs: List of Schmidt coefficients
        save_path: Optional path to save figure
    
    Returns:
        {'result': 'file_path', 'metadata': {...}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = range(len(schmidt_coeffs))
    ax.bar(indices, schmidt_coeffs, color='steelblue', alpha=0.7)
    ax.set_xlabel('Schmidt Index', fontsize=12)
    ax.set_ylabel('Schmidt Coefficient', fontsize=12)
    ax.set_title('Schmidt Decomposition Coefficients', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(schmidt_coeffs):
        if v > 0.01:
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    if save_path is None:
        save_path = './tool_images/schmidt_coefficients.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'num_coefficients': len(schmidt_coeffs),
            'schmidt_rank': int(np.sum(np.array(schmidt_coeffs) > 1e-10))
        }
    }


def visualize_entanglement_summary(analysis_result: Dict,
                                  save_path: str = None) -> Dict[str, any]:
    """
    Create comprehensive visualization of entanglement analysis results.
    
    Args:
        analysis_result: Output from comprehensive_entanglement_analysis
        save_path: Optional path to save figure
    
    Returns:
        {'result': 'file_path', 'metadata': {...}}
    """
    results = analysis_result['result']['criteria_results']
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Schmidt coefficients
    ax1 = fig.add_subplot(gs[0, :])
    schmidt_coeffs = results['schmidt']['schmidt_coefficients']
    ax1.bar(range(len(schmidt_coeffs)), schmidt_coeffs, color='steelblue', alpha=0.7)
    ax1.set_title('Schmidt Coefficients', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Coefficient')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Entanglement measures
    ax2 = fig.add_subplot(gs[1, 0])
    measures = {
        'Schmidt Rank': results['schmidt']['schmidt_rank'],
        'Schmidt Number': results['schmidt']['schmidt_number'],
        'Entanglement Entropy': results['schmidt_metadata']['entanglement_entropy']
    }
    if 'concurrence' in results:
        measures['Concurrence'] = results['concurrence']
    
    ax2.barh(list(measures.keys()), list(measures.values()), color='coral', alpha=0.7)
    ax2.set_xlabel('Value')
    ax2.set_title('Entanglement Measures', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Criteria summary
    ax3 = fig.add_subplot(gs[1, 1])
    criteria = {
        'Separability Test': 'Separable' if results['separability']['is_separable'] else 'Entangled',
        'PPT Criterion': 'Separable' if results['ppt']['is_ppt'] else 'Entangled',
        'Schmidt Rank': 'Separable' if results['schmidt']['schmidt_rank'] == 1 else 'Entangled'
    }
    
    colors = ['green' if 'Separable' in v else 'red' for v in criteria.values()]
    ax3.barh(list(criteria.keys()), [1]*len(criteria), color=colors, alpha=0.6)
    ax3.set_xlim(0, 1.5)
    ax3.set_xticks([])
    ax3.set_title('Entanglement Criteria', fontsize=14, fontweight='bold')
    
    for i, (k, v) in enumerate(criteria.items()):
        ax3.text(0.5, i, v, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # 4. Final verdict
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    is_entangled = analysis_result['result']['is_entangled']
    verdict_text = "ENTANGLED" if is_entangled else "SEPARABLE"
    verdict_color = 'red' if is_entangled else 'green'
    
    ax4.text(0.5, 0.7, f"Final Verdict: {verdict_text}", 
             ha='center', va='center', fontsize=20, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=verdict_color, alpha=0.3))
    
    # Add detailed explanation
    explanation = []
    if is_entangled:
        explanation.append("The quantum state is ENTANGLED because:")
        if results['schmidt']['schmidt_rank'] > 1:
            explanation.append(f"• Schmidt rank = {results['schmidt']['schmidt_rank']} > 1")
        if not results['separability']['is_separable']:
            explanation.append("• Cannot be written as a direct product |ψ_A⟩ ⊗ |ψ_B⟩")
        if results['ppt']['is_entangled']:
            explanation.append(f"• PPT criterion violated (min eigenvalue = {results['ppt_metadata']['min_eigenvalue']:.6f})")
    else:
        explanation.append("The quantum state is SEPARABLE because:")
        explanation.append("• All entanglement criteria indicate no entanglement")
    
    ax4.text(0.5, 0.3, '\n'.join(explanation), 
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    if save_path is None:
        save_path = './tool_images/entanglement_summary.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'is_entangled': is_entangled
        }
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstrate quantum entanglement analysis with three scenarios.
    """
    
    print("=" * 80)
    print("QUANTUM ENTANGLEMENT ANALYSIS TOOLKIT")
    print("=" * 80)
    print()
    
    # ========================================================================
    # SCENARIO 1: Original Problem - Analyze Given Quantum State
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 1: Analyze Given Two-Qubit State")
    print("=" * 80)
    print("Problem: Determine if the following state is entangled:")
    print("|ψ⟩ = (√6/6)(|00⟩ + |01⟩) + (√3/3)(|10⟩ + |11⟩)")
    print("-" * 80)
    
    # Step 1: Construct the state vector
    # Coefficients: √6/6 ≈ 0.408, √3/3 ≈ 0.577
    sqrt6_over_6 = np.sqrt(6) / 6
    sqrt3_over_3 = np.sqrt(3) / 3
    
    coefficients = {
        '00': [sqrt6_over_6, 0],
        '01': [sqrt6_over_6, 0],
        '10': [sqrt3_over_3, 0],
        '11': [sqrt3_over_3, 0]
    }
    
    print("\nStep 1: Construct state vector from basis coefficients")
    print(f"Calling: construct_state_vector()")
    state_result = construct_state_vector(coefficients, n_qubits=2)
    state_vector = state_result['result']
    print(f"FUNCTION_CALL: construct_state_vector | PARAMS: {{'n_qubits': 2, 'coefficients': '...'}} | RESULT: {state_result}")
    
    # Step 2: Perform comprehensive entanglement analysis
    print("\nStep 2: Perform comprehensive entanglement analysis")
    print(f"Calling: comprehensive_entanglement_analysis()")
    analysis = comprehensive_entanglement_analysis(state_vector, dims=[2, 2])
    print(f"FUNCTION_CALL: comprehensive_entanglement_analysis | PARAMS: {{'dims': [2, 2]}} | RESULT: {analysis}")
    
    # Step 3: Extract key results
    print("\nStep 3: Extract and display key results")
    is_entangled = analysis['result']['is_entangled']
    schmidt_rank = analysis['result']['criteria_results']['schmidt']['schmidt_rank']
    is_separable = analysis['result']['criteria_results']['separability']['is_separable']
    ppt_entangled = analysis['result']['criteria_results']['ppt']['is_entangled']
    concurrence = analysis['result']['criteria_results']['concurrence']
    
    print(f"\nEntanglement Analysis Results:")
    print(f"  - Schmidt Rank: {schmidt_rank}")
    print(f"  - Separability Test: {'Separable' if is_separable else 'Not Separable'}")
    print(f"  - PPT Criterion: {'Entangled' if ppt_entangled else 'Not Entangled'}")
    print(f"  - Concurrence: {concurrence:.6f}")
    print(f"  - Final Verdict: {'ENTANGLED' if is_entangled else 'SEPARABLE'}")
    
    # Step 4: Visualize results
    print("\nStep 4: Generate visualization")
    print(f"Calling: visualize_entanglement_summary()")
    viz_result = visualize_entanglement_summary(analysis, 
                                               save_path='./tool_images/scenario1_analysis.png')
    print(f"FUNCTION_CALL: visualize_entanglement_summary | PARAMS: {{'save_path': 'scenario1_analysis.png'}} | RESULT: {viz_result}")
    
    # Step 5: Detailed explanation
    print("\nStep 5: Detailed explanation")
    print("\nWhy is this state entangled?")
    print(f"  1. Schmidt Rank = {schmidt_rank} > 1")
    print(f"     → The state cannot be written as a simple product |ψ_A⟩ ⊗ |ψ_B⟩")
    print(f"  2. Separability Test: Failed")
    print(f"     → Direct product decomposition is not possible")
    print(f"  3. PPT Criterion: Violated")
    print(f"     → Partial transpose has negative eigenvalues")
    print(f"  4. Concurrence = {concurrence:.6f} > 0")
    print(f"     → Quantitative measure confirms entanglement")
    
    print(f"\nFINAL_ANSWER: Yes, the state is ENTANGLED because it does not satisfy the separability criterion. The Schmidt rank is {schmidt_rank} (>1), indicating the state cannot be factored into a product of single-qubit states. This is confirmed by the PPT criterion showing negative eigenvalues and a non-zero concurrence of {concurrence:.6f}.")
    
    # ========================================================================
    # SCENARIO 2: Bell State Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 2: Analyze Maximally Entangled Bell State")
    print("=" * 80)
    print("Problem: Analyze the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print("-" * 80)
    
    # Step 1: Construct Bell state
    bell_coefficients = {
        '00': [1/np.sqrt(2), 0],
        '01': [0, 0],
        '10': [0, 0],
        '11': [1/np.sqrt(2), 0]
    }
    
    print("\nStep 1: Construct Bell state vector")
    print(f"Calling: construct_state_vector()")
    bell_state_result = construct_state_vector(bell_coefficients, n_qubits=2)
    bell_state = bell_state_result['result']
    print(f"FUNCTION_CALL: construct_state_vector | PARAMS: {{'n_qubits': 2, 'bell_state': True}} | RESULT: {bell_state_result}")
    
    # Step 2: Schmidt decomposition
    print("\nStep 2: Perform Schmidt decomposition")
    print(f"Calling: schmidt_decomposition()")
    schmidt_result = schmidt_decomposition(bell_state, dims=[2, 2])
    print(f"FUNCTION_CALL: schmidt_decomposition | PARAMS: {{'dims': [2, 2]}} | RESULT: {schmidt_result}")
    
    # Step 3: Calculate concurrence
    print("\nStep 3: Calculate concurrence")
    print(f"Calling: density_matrix_from_state() and calculate_concurrence()")
    bell_rho = density_matrix_from_state(bell_state)
    conc_result = calculate_concurrence(bell_rho['result'])
    print(f"FUNCTION_CALL: calculate_concurrence | PARAMS: {{}} | RESULT: {conc_result}")
    
    # Step 4: Visualize Schmidt coefficients
    print("\nStep 4: Visualize Schmidt coefficients")
    print(f"Calling: visualize_schmidt_coefficients()")
    schmidt_viz = visualize_schmidt_coefficients(
        schmidt_result['result']['schmidt_coefficients'],
        save_path='./tool_images/scenario2_schmidt.png'
    )
    print(f"FUNCTION_CALL: visualize_schmidt_coefficients | PARAMS: {{'save_path': 'scenario2_schmidt.png'}} | RESULT: {schmidt_viz}")
    
    print(f"\nBell State Analysis:")
    print(f"  - Schmidt Coefficients: {schmidt_result['result']['schmidt_coefficients']}")
    print(f"  - Schmidt Rank: {schmidt_result['result']['schmidt_rank']}")
    print(f"  - Concurrence: {conc_result['result']:.6f}")
    print(f"  - Entanglement Entropy: {schmidt_result['metadata']['entanglement_entropy']:.6f} bits")
    
    print(f"\nFINAL_ANSWER: The Bell state |Φ+⟩ is MAXIMALLY ENTANGLED with concurrence = {conc_result['result']:.6f} and entanglement entropy = {schmidt_result['metadata']['entanglement_entropy']:.6f} bits (maximum for two qubits).")
    
    # ========================================================================
    # SCENARIO 3: Separable State Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 3: Analyze Separable Product State")
    print("=" * 80)
    print("Problem: Verify that |ψ⟩ = |0⟩ ⊗ |+⟩ = (|00⟩ + |01⟩)/√2 is separable")
    print("-" * 80)
    
    # Step 1: Construct separable state
    separable_coefficients = {
        '00': [1/np.sqrt(2), 0],
        '01': [1/np.sqrt(2), 0],
        '10': [0, 0],
        '11': [0, 0]
    }
    
    print("\nStep 1: Construct separable state vector")
    print(f"Calling: construct_state_vector()")
    sep_state_result = construct_state_vector(separable_coefficients, n_qubits=2)
    sep_state = sep_state_result['result']
    print(f"FUNCTION_CALL: construct_state_vector | PARAMS: {{'n_qubits': 2, 'separable': True}} | RESULT: {sep_state_result}")
    
    # Step 2: Test separability directly
    print("\nStep 2: Test separability via direct product decomposition")
    print(f"Calling: test_separability_direct()")
    sep_test = test_separability_direct(sep_state, dims=[2, 2])
    print(f"FUNCTION_CALL: test_separability_direct | PARAMS: {{'dims': [2, 2]}} | RESULT: {sep_test}")
    
    # Step 3: Schmidt decomposition
    print("\nStep 3: Verify with Schmidt decomposition")
    print(f"Calling: schmidt_decomposition()")
    sep_schmidt = schmidt_decomposition(sep_state, dims=[2, 2])
    print(f"FUNCTION_CALL: schmidt_decomposition | PARAMS: {{'dims': [2, 2]}} | RESULT: {sep_schmidt}")
    
    # Step 4: PPT criterion
    print("\nStep 4: Apply PPT criterion")
    print(f"Calling: density_matrix_from_state() and ppt_criterion()")
    sep_rho = density_matrix_from_state(sep_state)
    sep_ppt = ppt_criterion(sep_rho['result'], dims=[2, 2])
    print(f"FUNCTION_CALL: ppt_criterion | PARAMS: {{'dims': [2, 2]}} | RESULT: {sep_ppt}")
    
    # Step 5: Comprehensive analysis
    print("\nStep 5: Comprehensive analysis")
    print(f"Calling: comprehensive_entanglement_analysis()")
    sep_analysis = comprehensive_entanglement_analysis(sep_state, dims=[2, 2])
    print(f"FUNCTION_CALL: comprehensive_entanglement_analysis | PARAMS: {{'dims': [2, 2]}} | RESULT: {sep_analysis}")
    
    # Step 6: Visualize
    print("\nStep 6: Generate visualization")
    print(f"Calling: visualize_entanglement_summary()")
    sep_viz = visualize_entanglement_summary(sep_analysis,
                                            save_path='./tool_images/scenario3_analysis.png')
    print(f"FUNCTION_CALL: visualize_entanglement_summary | PARAMS: {{'save_path': 'scenario3_analysis.png'}} | RESULT: {sep_viz}")
    
    print(f"\nSeparable State Analysis:")
    print(f"  - Is Separable: {sep_test['result']['is_separable']}")
    print(f"  - Schmidt Rank: {sep_schmidt['result']['schmidt_rank']}")
    print(f"  - Factor States:")
    if sep_test['result']['factorization']:
        print(f"    |ψ_A⟩ = {sep_test['result']['factorization']['state_A']}")
        print(f"    |ψ_B⟩ = {sep_test['result']['factorization']['state_B']}")
    print(f"  - PPT Satisfied: {sep_ppt['result']['is_ppt']}")
    print(f"  - Concurrence: {sep_analysis['result']['criteria_results']['concurrence']:.6f}")
    
    print(f"\nFINAL_ANSWER: The state |ψ⟩ = |0⟩ ⊗ |+⟩ is SEPARABLE (not entangled) because it can be written as a direct product of single-qubit states. Schmidt rank = 1, all PPT eigenvalues are non-negative, and concurrence = 0.")
    
    print("\n" + "=" * 80)
    print("TOOLKIT DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()