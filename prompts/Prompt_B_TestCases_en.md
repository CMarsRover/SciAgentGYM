
Now please complete **unit test development** based on existing code.

## Task Objective
- Design unit tests for functions based on given function code and key tasks, verify that tool code correctly solves the problem
- Cover correctness, boundary conditions, exception handling, and other situations
---

## Execution Requirements

- Use `pytest` framework to write test cases
- Must first **analyze the mathematical problem and expected output in query**, derive correct results or ranges for assertions
- Write at least 3 sets of test data:
  - ✅ Normal case: Verify main calculation flow is correct
  - ⚠️ Boundary case: Extreme values, zero values, decimal precision, etc.
  - ❌ Exception case: Illegal input, type errors, etc.
- Explain the mathematical meaning and expected result source of each test case in comments
- Comments should be clear,便于后续添加新测试 (convenient for adding new tests later)
- Test path import is correct ```python from file import func``` Because files are in test path, while tools are in another path, so you need to handle it similarly:
    ```
    import os
    import sys 
    # Use import helper to solve module import issues
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ```
---

## Test Code Template Specification

```python
# Filename: test_<tool_name>.py

import pytest
from math_tool import calculate_function_name

def test_case_1():
    """
    Test case description: Based on typical input given in query, verify return value matches theoretical solution
    """
    result = calculate_function_name(input parameters...)
    assert pytest.approx(result, rel=1e-6) == expected_value

def test_case_2():
    """
    Boundary test: Very small/large input, verify calculation stability
    """
    ...

def test_case_3():
    """
    Exception test: Illegal parameters should raise ValueError
    """
    with pytest.raises(ValueError):
        calculate_function_name(illegal_parameters)

if __name__ == "__main__":
    # Allow direct running of this file for testing
    pytest.main([__file__])

---

## Output Format

Directly output as .py file content: ```python\n# Filename: test_<tool_name>.py\n# Complete test code\n```

**Notes:**
- Don't write complex test code like class
- Start writing code at file beginning, no reasoning, must clearly state test file name is `# Filename: test_<tool_name>.py` comment
- No need to repeat algorithm logic implementation
- Tests must be independently runnable
- Each test case must include: input parameters, expected output, assertions
