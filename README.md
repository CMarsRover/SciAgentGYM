<h1 align="center"> SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.12984">📃 Paper</a>
  •
  <a href="#">🤗 Data & Models</a>
  •
  <a href="#">🔧 Toolkits</a>
</p>

We present **SciAgentGym**, the first benchmark environment for evaluating LLM agents' capability in multi-step scientific tool-use. SciAgentGym provides a comprehensive suite of scientific tools across multiple disciplines, enabling rigorous evaluation of how well LLMs can solve complex scientific problems through sequential tool invocation.

<p align="center">
  <img src="./pic/main.png" width="100%" alt="SciAgentGym Overview">
</p>

---

## Overview

Complex scientific problems often require multiple steps of computation, each involving specialized domain tools. SciAgentGym addresses this challenge by providing:

- **1780+ Scientific Tools** across Physics, Chemistry, Materials Science, Life Science, and Astronomy
- **Multi-step Reasoning Tasks** requiring sequential tool calls to reach final answers
- **Standardized Evaluation Pipeline** with automated answer extraction and scoring
- **Flexible Tool Registration** enabling easy extension to new domains

## Environment Building

SciAgentGym provides an integrated execution environment comprising four components:

| Component | Description |
|-----------|-------------|
| **Toolkit** | 1,780 domain-specific scientific tools across multiple disciplines |
| **Filesystem** | Data storage and artifact management for intermediate results |
| **Databases** | Scientific knowledge retrieval (e.g., PubChem, local molecular DBs) |
| **Python Interpreter** | Flexible computation environment for tool execution |

Each task runs in an **isolated instance** with its own registered tools and filesystem, ensuring reproducibility and avoiding cross-task contamination.

<p align="center">
  <img src="./pic/interwenv.png" width="80%" alt="Environment Interaction">
</p>

### Design Principles

- **Type Safety**: Each tool specifies typed input/output signatures enabling automatic validation
- **Reproducibility**: All executions are recorded as structured traces with fixed random seeds
- **Extensibility**: Tools are organized by domain with standardized protocols, enabling researchers to register custom tools for specialized domains

### Tool Distribution

SciAgentGym contains **1,780+ scientific tools from 4 major disciplines**:

<p align="center">
  <img src="./pic/icml_treemap_final.png" width="90%" alt="Tool Distribution Treemap">
</p>

| Discipline | Topics |  Python files |
|------------|--------|-------|
| **Physics** | Optics, Mechanics, Electromagnetism, Thermodynamics, Acoustics, Plasma Physics, Fluid Dynamics, Atomic Physics, Condensed Matter | 96+ |
| **Chemistry** | Analytical, Physical, Computational, Organic, Environmental | 28+ |
| **Materials Science** | Crystallography, Spectroscopy, Structural Analysis, XRD | 24+ |
| **Life Science** | Structural Biology, Mass Spectrometry | 19+ |

<p align="center">
  <img src="./pic/tool_clustering_by_discipline.png" width="90%" alt="Tool Clustering by Discipline">
</p>
<p align="center"><em>Tool clustering results by discipline</em></p>





## Architecture

```
SciAgentGym/
├── gym/                      # Core evaluation environment
│   ├── env.py               # MinimalSciEnv - execution environment
│   ├── tool.py              # EnvironmentTool base class
│   ├── toolbox.py           # Tool registration system
│   ├── agent.py             # Multi-model LLM client
│   ├── test_executor.py     # Test execution engine
│   ├── test_querys.py       # Benchmark entry point
│   │
│   ├── core/                # Core modules
│   │   ├── tool_loader.py   # Dynamic tool loading
│   │   ├── data_loader.py   # Dataset loading
│   │   ├── evaluator.py     # Scoring algorithms
│   │   └── environment_fs.py # Result management
│   │
│   ├── config/              # Configuration
│   └── utils/               # Utilities
│
└── toolkits/                # Scientific tool implementations
    ├── physics/
    ├── chemistry/
    ├── materials_science/
    ├── life_science/
    └── astronomy/
```

## Toolkits Structure
```
toolkits/
├── physics/                          # Physics (96+ tools)
│   ├── optics/                       # Optics
│   │   ├── optics_tools_gym.py       # Tool registration module
│   │   ├── optical_interference_solver_204.py
│   │   └── thin_film_interference.py
│   ├── mechanics/                    # Mechanics (37 tools)
│   ├── electromagnetism/             # Electromagnetism (8 tools)
│   ├── thermodynamics/               # Thermodynamics (12 tools)
│   ├── acoustics/                    # Acoustics (4 tools)
│   ├── plasma_physics/               # Plasma Physics
│   ├── fluid_dynamics/               # Fluid Dynamics
│   ├── atomic_and_molecular_physics/ # Atomic & Molecular Physics
│   ├── condensed_matter_physics/     # Condensed Matter Physics
│   └── structural_mechanics/         # Structural Mechanics
│
├── chemistry/                        # Chemistry (28+ tools)
│   ├── analytical_chemistry/         # Analytical Chemistry
│   ├── physical_chemistry/           # Physical Chemistry
│   ├── computational_chemistry/      # Computational Chemistry
│   ├── organic_chemistry/            # Organic Chemistry
│   └── environmental_chemistry/      # Environmental Chemistry
│
├── materials_science/                # Materials Science (24+ tools)
│   ├── crystallography/              # Crystallography
│   ├── spectroscopy_analysis/        # Spectroscopy Analysis
│   ├── structural_analysis/          # Structural Analysis
│   └── x_ray_diffraction_analysis/   # XRD Analysis
│
├── life_science/                     # Life Science (19+ tools)
│   ├── structural_biology/           # Structural Biology
│   └── mass_spectrometry/            # Mass Spectrometry
│
├── astronomy/                        # Astronomy (6+ tools)
│
└── local_db/                         # Local Databases
    └── *.db / *.sqlite               # SQLite databases
```

### File Types

| File Type | Description | Example |
|-----------|-------------|---------|
| `*_tools_gym.py` | Tool registration module with `@Toolbox.register` decorated classes | `optics_tools_gym.py` |
| `*.py` | Core computation function implementations | `thin_film_interference.py` |
| `*.json` | Configuration or data files | `reaction_library.json` |
| `*.db` | SQLite databases for knowledge retrieval | `molecules.db` |

### Tool Class Example

```python
@Toolbox.register(name="calculate_thin_film_interference")
class CalculateThinFilmInterferenceTool(EnvironmentTool):
    """Calculate enhanced and weakened wavelengths in thin film interference."""
    
    name = "calculate_thin_film_interference"
    description = "Calculate enhanced and weakened wavelengths in thin film interference."
    arguments = {
        "n1": {"type": "number", "description": "Refractive index of incident medium"},
        "n2": {"type": "number", "description": "Refractive index of thin film"},
        "d": {"type": "number", "description": "Film thickness in nm"},
    }
    
    def use(self, environment, action) -> Observation:
        result = compute_interference(...)
        return Observation(self.name, json.dumps(result))
```

## Quick Start

**Installation**

We recommend the one-shot script — it creates a conda env named
`sciagentgym` (Python 3.11), installs every scientific package via
conda-forge for binary compatibility, then runs a soft-fail verification
step:

```bash
git clone git@github.com:CMarsRover/SciAgentGYM.git
cd SciAgentGYM

# Option A: one-shot script (recommended)
bash install.sh

# Useful flags:
#   REBUILD=1        recreate the env from scratch
#   USE_TSINGHUA=1   route conda + pip through Tsinghua mirrors (mainland China)
REBUILD=1 USE_TSINGHUA=1 bash install.sh

# Option B: manual (equivalent to Option A minus the verification step)
conda env create -f environment.yml
conda activate sciagentgym
```

**Configure API Keys**

Edit `gym/config/config.py`:

```python
SUPPORTED_MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "your-api-key",
    },
    # Add more models...
}
```

**Datasets**

`dataset/refine_merged_multi_questions.json` (83 cases, multi-modal) and
`dataset/refine_merged_single_questions.json` (48 cases, text-only) —— combined
they form the paper's 131 original + 128 refined = **259 tasks / 1134 subquestions**.

Each case is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | int | Case identifier |
| `question` | str | Original problem statement (multi-modal cases reference images via `metadata.image_path`) |
| `answer` | str | Gold answer for the original question (used by boxed-answer evaluation) |
| `metadata.subject` / `topic` | str | Discipline / sub-field, used to auto-infer toolkit directories |
| `metadata.image_path` | list[str] | Image files attached to the question (multi-modal input) |
| `metadata.solution_steps` | list | Reference reasoning steps |
| `metadata.tool_expected` | list[str] | Expected tool function names |
| `metadata.golden_answer` | list | Structured gold answers keyed by sub-question |
| `usage_tool_protocol` | list | OpenAI function-calling schemas, injected into the tool-call turn |
| `refined_versions` | list | Refined variants: `refined_question` + `final_answer` (a dict of sub-question → answer) |

**Run Evaluation**

```bash
# See all flags
python gym/test_querys.py --help

# Full multi-modal benchmark (default)
python gym/test_querys.py

# Single-modal split
python gym/test_querys.py --dataset single

# Both splits
python gym/test_querys.py --dataset both

# Pick model & disable tools (LLM baseline)
python gym/test_querys.py --model gpt-4o --no-tools

# Run one specific case id (repeat --case-id to select several)
python gym/test_querys.py --case-id 5 --case-id 12

# Structured (refine) evaluation instead of boxed-answer matching
python gym/test_querys.py --test-type refine
```

## Evaluation Pipeline

SciAgentGym provides a standardized evaluation pipeline:

1. **Tool Loading**: Automatically infer and load required tools from metadata
2. **Agent Execution**: LLM generates tool calls, environment executes them
3. **Answer Extraction**: Extract final answer from `\boxed{}` format
4. **Scoring**: Compare against gold standard with flexible matching

```python
from gym.core.tool_loader import prepare_env_from_query
from gym.core.evaluator import extract_boxed_answer, is_answer_correct

# Load environment and tools
env, tools, schema, registry = prepare_env_from_query(test_case)

# Run agent interaction...

# Evaluate
answer = extract_boxed_answer(model_response)
correct = is_answer_correct(question, answer, gold_answer, case_id)
```

## Extending SciAgentGym

**Adding New Tools**

```python
# toolkits/physics/optics/my_tool.py
from gym.toolbox import Toolbox
from gym.entities import Observation

@Toolbox.register
class RefractionCalculator:
    name = "refraction_calculator"
    description = "Calculate refraction angle using Snell's law"
    
    parameters = {
        "type": "object",
        "properties": {
            "n1": {"type": "number", "description": "Refractive index of medium 1"},
            "n2": {"type": "number", "description": "Refractive index of medium 2"},
            "theta1": {"type": "number", "description": "Incident angle in degrees"}
        },
        "required": ["n1", "n2", "theta1"]
    }
    
    def use(self, n1: float, n2: float, theta1: float) -> Observation:
        import math
        theta2 = math.degrees(math.asin(n1 * math.sin(math.radians(theta1)) / n2))
        return Observation(source=self.name, observation=f"Refraction angle: {theta2:.2f}°")
```


## Citation

```bibtex
@misc{shen2026sciagentgymbenchmarkingmultistepscientific,
      title={SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents}, 
      author={Yujiong Shen and Yajie Yang and Zhiheng Xi and Binze Hu and Huayu Sha and Jiazheng Zhang and Qiyuan Peng and Junlin Shang and Jixuan Huang and Yutao Fan and Jingqi Tong and Shihan Dou and Ming Zhang and Lei Bai and Zhenfei Yin and Tao Gui and Xingjun Ma and Qi Zhang and Xuanjing Huang and Yu-Gang Jiang},
      year={2026},
      eprint={2602.12984},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.12984}, 
}
```
