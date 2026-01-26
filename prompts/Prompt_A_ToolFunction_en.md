## Task Objective
Based on given domain knowledge, specific problems, and available tool sets, build generalizable mathematical computational tools and complete problem solving.
---
## Execution Requirements

- Strictly output complete Python code files following the process below:

### 1. Mathematical Principle Analysis
- Identify core mathematical concepts, theorems, and formulas involved in the problem
- Analyze solution paths, clarify mathematical tools required for each calculation step
- Build a complete mathematical reasoning framework

### 2. Core Algorithm Abstraction
- Extract the computational essence of the problem, design reusable core computational modules
- Analyze required Python scientific computing libraries (such as numpy, scipy, matplotlib, etc.)
- Design modular, extensible function architecture
- Design deconstructive func tools, reduce class definitions

### 3. Code Implementation Standards
- Write high-quality tool functions with complete docstrings
- Function parameters must be generalized, avoid hardcoding specific values. For problems requiring piecewise solutions, implement branch structures well, as different intervals may require different solving equations. You need to be as rigorous as writing a git repository～
- Provide detailed parameter descriptions, return value descriptions, and usage examples
- Visualization optional code: "Only provide when necessary", visualization is a measure to check if key results are correct, so simplify or omit it
- main function demonstrates the specific solution process for **current problem**

---

## 📌 Code Template Specification

```python
# Filename: <tool_name>.py
# (Note: <tool_name> must be reasonably named based on tool theme, e.g., pendulum_solver)

def calculate_function_name(param1, param2, param3=default_value):
    """
    Concise description of function functionality
    Detailed algorithm principle explanation, including applicable scenarios and theoretical basis.
    
    Parameters:
    -----------
    param1 : type
        Detailed parameter description, including units, value ranges, physical meaning, etc.
    param2 : type
        Detailed parameter description, including units, value ranges, physical meaning, etc.
    param3 : type, optional
        Optional parameter description, explain default value selection rationale
    
    Returns:
    --------
    type
        Return value description, including units, physical meaning, special value meanings
    """
    # Clear algorithm implementation
    # Include necessary intermediate variable comments
    return result

def main():
    """
    Main function: Demonstrate how to use tool functions to solve current problem
    """
    # Problem parameter definition (can hardcode specific values)
    # Call tool functions to solve
    # Output formatted results
    pass
```

---

## Output Format

Directly output as .py file content: ```python\n# Filename: <tool_name>.py\n# Complete tool function code\n```

**Notes:**
- Code must be complete and runnable
- File header must have `# Filename: <tool_name>.py` comment
- Function naming and parameter design must be generalizable
- I must consider visualization to be simple and straightforward, and labels on graphs should be displayable, both Chinese and English are acceptable, for example: ```python plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False```
- Plotting tool results must be saved, save directory is "./images"+image name
- Must reflect the "code as tool" mindset, don't just write dead code based on elements in the problem or copy answers. Remember code should be universal and generalizable
- Strictly refer to key domains given in the previous conversation, not limited to the specific problem currently being solved, generate integrated toolkits adapted to two key domains, consider more about computational tools and main design.
