## Task Objective
Based on knowledge in physics, chemistry, and materials science, build professional, highly generalizable retrieval and computational toolkits that support multi-scenario problem solving and are compatible with OpenAI Function Calling specifications.

---

## Execution Requirements

### 1. Domain Knowledge and Mathematical Modeling
- **Identify Core Scientific Principles**: Clearly define the physical laws, chemical reaction mechanisms, or material properties involved in the problem
- **Mathematical Model Construction**: Transform scientific problems into mathematical equation systems, optimization problems, or numerical calculations
- **Solution Path Design**: Decompose into reusable computational units, considering branch logic for different parameter intervals

### 2. Professional Tool Library Selection
**Must prioritize domain-specific libraries**, including but not limited to:

**Physics Computing:**
- `scipy.integrate` - Differential equation solving
- `scipy.optimize` - Optimization and fitting
- `sympy` - Symbolic computation

**Chemistry Computing:**
- `rdkit` - Molecular processing and property calculation
- `pubchempy` - PubChem database access
- `mendeleev` - Periodic table data
- `chempy` - Stoichiometry and reaction equilibrium

**Materials Science:**
- `pymatgen` - Crystal structure analysis and phase diagram calculation
- `ase` - Atomic Simulation Environment
- `mp-api` - Materials Project database interface
- `matminer` - Material data mining

**Visualization Tools:**
- `plotly` - Interactive scientific visualization (priority)
- `matplotlib` - Basic plotting
- `seaborn` - Statistical charts

### 3. Tool Function Design Principles

#### 3.1 Generality and Composability
- **Single Responsibility**: Each function does one thing, facilitating combined calls
- **Generalized Parameters**: Avoid hardcoding, support array/batch input
- **Branch Logic**: Use decorators or strategy patterns to handle different physical intervals/chemical environments
- **Tool Chain Design**: High-level functions call low-level functions, forming a tool ecosystem

#### 3.2 OpenAI Function Calling Compatibility Specification
Each tool function must include:
````python
def tool_function_name(param1: float, param2: str, param3: Optional[list] = None) -> dict:
    """
    [Concise one-sentence description - for Function Calling description]
    
    Detailed scientific principle explanation (2-3 sentences)
    ### 🔧 Updated Code Quality Checklist
    - [ ] **All function parameter types are JSON serializable**: str, int, float, bool, List, Dict
    - [ ] **Python object construction logic is inside the function**, not passed as parameters
    - [ ] **Support multiple input formats**: file paths, database IDs, string representations
    - [ ] **Example code uses basic types for calls**, no Python objects involved
    
    Args:
        param1: Physical/chemical meaning of parameter, units, value range (e.g., temperature/K, range 273-373)
        param2: Parameter description (e.g., element symbol, examples 'C','N','O')
        param3: Optional parameter description, default value selection rationale
    
    Returns:
        dict: {
            'result': Main calculation result (with unit description),
            'metadata': {Additional information such as convergence status, databases used, etc.}
        }
    
    Example:
        >>> result = tool_function_name(300.0, 'H2O')
        >>> print(result['result'])
    """
    # Implementation code
    return {"result": value, "metadata": {}}
````


### 4. Toolkit Architecture Design
#### 🎯 Design Principle Summary: Layered Architecture Design
#### Output Specification: Ensure each tool's return format is unified `{'result': ..., 'metadata': {...}}` format

##### Atomic Functions (Layer 1)

- [ ] **Complete boundary condition checks** (type, range, special values, function testing)
- [ ] **Can be used independently**:
  - [ ] Ensure unified return format; if return information is a file path, also provide the specific path, save all process files uniformly in `./mid_result/{subject}` where `subject` is determined by you based on the problem, choose from `physics/chemistry/biology/materials`;
  - [ ] If encountering non-serializable data structures like `scipy.sparse.csr_matrix` which is CSR (Compressed Sparse Row) format, an efficient data structure for storing sparse matrices, you need to return as follows:
   `
        summary = f"""Sparse Matrix (CSR format):
            - Shape: {sparse_mat.shape}
            - Non-zero elements: {sparse_mat.nnz} / {sparse_mat.shape[0] * sparse_mat.shape[1]}
            - Sparsity: {(1 - sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1])) * 100:.2f}%
            - Data type: {sparse_mat.dtype}
            Saved to: {filepath}
            Can be loaded with scipy.sparse.load_npz()
            """
        return {
            'type': 'sparse_matrix',
            'summary': summary,
            'filepath': filepath,
            'metadata': {
                'shape': sparse_mat.shape,
                'nnz': sparse_mat.nnz,
                'format': 'csr'
            }
        }
    `
- [ ] **Atomic functions include comprehensive boundary condition checks**
  - [ ] Parameter type checks
  - [ ] Parameter range checks
  - [ ] Special value handling (zero values, empty values, extreme values)
  - [ ] Function call testing (for callable parameters)
- [ ] **Retain atomic functions for advanced users**

##### Composite Functions (Layer 2)

- [ ] **Composite functions are for complex solving tasks that require multi-step mathematical/physical/engineering calculations, integrating outputs from multiple atomic functions, and performing high-level analysis and reasoning. Not simple atomic function concatenation, but implementing domain-specific complex algorithms or analysis workflows.**
- [ ] **Input parameters are fully serializable**
  - [ ] If atomic function returns file path → composite function should first understand whether output information contains sparse matrices or special values to construct input
  - [ ] If atomic function returns sparse matrix dict → composite function accepts dict format matrix representation
  - [ ] If atomic function returns array → composite function accepts list format array representation
  - [ ] Avoid parameter types that LLMs cannot construct:
        ❌ numpy.ndarray
        ❌ scipy.sparse.csr_matrix
        ❌ pandas.DataFrame
        ❌ Custom class instances
        ✅ Serializable representations of the above objects (list/dict)
- [ ] **Composite functions calling multiple atomic functions internally must include comments**: `## using file.func_name, and get ** returns`,
- [ ] **All function returns follow the unified output specification above**

##### Visualization Functions (Layer 3)

- [ ] **Perform domain-specific visualization**
- [ ] **No HTML visualization methods**
- [ ] **Automatic handling**: Tools automatically save files to `./tool_images/filename.extension`
- [ ] **Must print full path after saving file**, format: `print(f"image_saved_path: {file_type}")`
- [ ] **Return format still follows output specification**: `{'result': ..., 'metadata': {...}}`

````python
# Filename: <domain>_toolkit.py (e.g., materials_toolkit.py)

# ============ Layer 1: Atomic Tool Functions (Atomic Tools) ============
def fetch_property_from_database(identifier: str, property_name: str) -> dict:
    """Fetch basic data from free database"""
    pass

def construct_hamiltonian(system_size, interaction_matrix, potential=None, periodic=False) -> dict:
    """
    Construct Hamiltonian (returns sparse matrix)
    Returns:
        return {
            'type': 'sparse_matrix',
            'summary': summary,
            'filepath': filepath,
            'metadata': {
                'shape': sparse_mat.shape,
                'nnz': sparse_mat.nnz,
                'format': 'csr'
            }
        }
    """
    # === Complete boundary checks ===
    if isinstance(system_size, int):
        ...
    return {'type': 'sparse_matrix', ... , 'metadata': {...}}

# ============ Layer 2: Composite Tool Functions (Composite Tools) ============

def analyze_commutator(matrix_a: list, matrix_b: list) -> dict:
    """
    Calculate and analyze the commutator of two operators [A, B] = AB - BA
    
    Physical meaning:
    - [A, B] = 0: Two operators commute, can be measured simultaneously
    - [A, B] ≠ 0: Two operators do not commute, uncertainty relation exists
    
    Args:
        matrix_a: First matrix (operator A), format as nested list or sparse dict
        matrix_b: Second matrix (operator B), same format as above
        
    Returns:
        dict: {'result': summary, 'metadata': {...}}
    """
    # === Fully serializable parameter checks ===
    if not isinstance(matrix_a, (list, dict)):
        raise TypeError("matrix_a must be list or dict")
    if not isinstance(matrix_b, (list, dict)):
        raise TypeError("matrix_b must be list or dict")
    
    # === Deserialization ===
    from scipy.sparse import csr_matrix
    
    ## Deserialize matrix_a
    if isinstance(matrix_a, dict):
        A = csr_matrix(
            (matrix_a['data'], (matrix_a['row'], matrix_a['col'])),
            shape=tuple(matrix_a['shape'])
        )
    else:
        A = np.array(matrix_a)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"matrix_a must be a square matrix")
    
    ## Deserialize matrix_b
    ...
    
    # Check dimension matching
    if A.shape != B.shape:
        raise ValueError(f"Two matrices have mismatched dimensions: {A.shape} vs {B.shape}")
    
    # === Core calculation: Commutator ===
    commutator = A @ B - B @ A
    ... 

# ============ Layer 3: Visualization Tools (Visualization - as needed) ============

def visualize_domain_specific(data: dict, domain: str, vis_type: str,
                               save_dir: str = './images/',
                               filename: str = None) -> str:
    """
    Domain-specific visualization tool - Use standard visualization methods for each discipline, generate based on the current problem's domain
    
    Args:
        data: Data to visualize
        domain: Domain type 'chemistry', 'materials', 'physics'
        vis_type: Specific visualization type (see descriptions below)
        save_dir: Save directory
    
    Example domain-specific visualization types:
    
    【Chemistry Domain】
    - 'molecule_2d': 2D molecular structure diagram (using RDKit)
    - 'molecule_3d': 3D molecular conformation (using Py3Dmol or RDKit)
    - 'reaction_scheme': Reaction equation diagram
    - 'spectrum': Spectrum diagram (IR/NMR/UV-Vis)
    - 'energy_diagram': Energy level diagram/potential energy surface
    
    【Materials Science Domain】
    - 'crystal_structure': Crystal structure diagram (using ASE/Pymatgen)
    - 'phase_diagram': Phase diagram (using Pymatgen)
    - 'band_structure': Band structure
    - 'dos': Density of States diagram
    - 'xrd_pattern': XRD diffraction pattern
    
    【Physics Domain】
    - 'field_distribution': Field distribution diagram (electric/magnetic/temperature fields)
    - 'trajectory': Particle trajectory diagram
    - 'phase_space': Phase space diagram
    - 'waveform': Waveform diagram (sound/electromagnetic waves)
    - 'vector_field': Vector field diagram
    
    【Statistical Analysis (General)】
    - 'histogram': Histogram (distribution analysis)
    - 'box_plot': Box plot (outlier detection)
    - 'correlation_heatmap': Correlation heatmap
    - 'pie_chart': Pie chart (component proportion)
    
    Returns:
        str: Saved image path
    """
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Call professional library based on domain, example for materials domain == 'materials'
    # Use ASE to visualize crystal structure
    from ase.io import write
    from ase import Atoms
    
    atoms = data.get('structure')  # ASE Atoms object
    save_file = os.path.join(save_path, f"{plot_type}_result.jpg")  # or .png
    # Save logic
    return save_file


# ============ Layer 4: Main Flow Demonstration ============
def main():
    """
    Demonstrate toolkit solving 【current problem】+ 【at least 2 related scenarios】
    
    ⚠️ Must strictly follow the format below:
    """
    
    print("=" * 60)
    print("Scenario 1: Original Problem Solving")
    print("=" * 60)
    print("Problem Description: [Describe the specific problem to solve in 1-2 sentences]")
    print("-" * 60)
    
    # Step 1: [Describe what this step does]
    # Call function: function_name_1()
    result1 = function_name_1(param1, param2)
    print(f"Step 1 result: {result1['result']}")
    
    # Step 2: [Describe what this step does]
    # Call function: function_name_2(), this function internally calls function_name_1()
    result2 = function_name_2(result1['result'], param3)
    print(f"Step 2 result: {result2['result']}")
    
    # Step 3: [If there are more steps]
    # Call function: function_name_3()
    final_result1 = function_name_3(result2)
    print(f"✓ Scenario 1 final answer: {final_result1['result']}\n")
    # Scenarios 2 and 3 follow the same pattern
    
    print("=" * 60)
    print("Toolkit demonstration complete")
    print("=" * 60)
    print("Summary:")
    print("- Scenario 1 demonstrates the complete workflow for solving the original problem")
    print("- Scenario 2 demonstrates the tool's parameter generalization capability")
    print("- Scenario 3 demonstrates the tool's database integration capability")


if __name__ == "__main__":
    main()
````

---

### 🔧 Code Quality Checklist

Before outputting code, confirm:
- [ ] Use at least 2 domain-specific libraries (not numpy/matplotlib)
- [ ] Include 3 layers of tool functions (atomic → composite → demonstration)
- [ ] Each function has type hints and return dict format
- [ ] main function strictly follows template format, includes 3 scenarios
- [ ] **Each scenario has "Problem Description" and "Call Function" comments**
- [ ] **Each step uses comments to indicate the specific function name called**
- [ ] Visualization code includes save logic and font configuration
- [ ] No hardcoded values (constants use UPPER_CASE naming at file top)
- [ ] Clear call relationships between tool functions

---

## 📤 Output Format
````python
# Filename: <domain>_toolkit.py
"""
<Domain Name> Computational Toolkit

Main Functions:
1. [Function A]: Implement Y calculation based on X library
2. [Function B]: Call Z database to retrieve W data
3. [Function C]: Combined analysis to complete P problem

Dependencies:
pip install numpy scipy rdkit pymatgen plotly
"""

import numpy as np
from typing import Optional, Union, List, Dict
# Import domain-specific libraries
from rdkit import Chem
from pymatgen.core import Structure

# Global constants
PLANCK_CONSTANT = 6.62607015e-34  # J·s

# [Complete code implementation...]
````

---

## ⚠️ Key Requirements Emphasis

### Hard Requirements for main() Function:

1. **Must include 3 scenarios**:
   - Scenario 1: Solve the given original problem (complete solution process)
   - Scenario 2: Parameter scanning or condition change analysis
   - Scenario 3: Database batch query or cross-object comparison

2. **Each scenario must include**:
   - `print("=" * 60)` separator line
   - `print("Scenario X: XXX")` title
   - `print("Problem Description: XXX")` explaining what this scenario does
   - Add `# Call function: function_name()` comment before each calculation step
   - Use `print(f"✓ Scenario X complete: ...")` to mark completion

3. **Function calls must be traceable**:
   - Scenario 1 must demonstrate call chains between functions (e.g., function B calls function A)
   - Comments clearly state: `# Call function: xxx(), this function internally calls yyy()`

4. **Output Format Requirements ⚠️**:
    **Structured Function Call Output**: Each calculation step must use standardized format, clearly annotating function name, parameters, and results:
      - Standard format: `print(f"FUNCTION_CALL: {function_name} | PARAMS: {params} | RESULT: {result}")`
      - Simplified format: `print(f"[CALL] {function_name}({params}) -> {result}")`
      - Recommended example: `print(f"FUNCTION_CALL: calculate_velocity | PARAMS: time=5.0, acceleration=9.8 | RESULT: 49.0")`
    **File Generation Output**: If code generates files, must print file path:
      - Format: `print(f"FILE_GENERATED: {file_type} | PATH: {file_path}")`
      - Example: `print(f"FILE_GENERATED: Plot | PATH: ./images/velocity_plot.png")`
    **Final Answer Output**: Must use standard format to output final answer:
        - Format: `print(f"FINAL_ANSWER: {answer}")`
    **Output Order**: Ensure final answer is on the last line of all outputs
    **Easy Extraction**: All key outputs use uppercase keywords for easy regex extraction

---

## ⚠️ Key Improvement Points

1. **Multi-scenario Adaptation**: main function must demonstrate tool applications under different parameters/problems
2. **Professional Library Enforcement**: Clearly list domain-specific libraries for physics/chemistry/materials, cannot use only general libraries
3. **Function Calling Format**: Return dict instead of single value, include metadata
4. **Tool Call Chain**: High-level functions call low-level functions, reflecting tool ecosystem
5. **Database Integration**: Must demonstrate access to at least one free database
6. **🆕 Comment Specification**: Each function call must be preceded by a comment indicating function name and call relationship
