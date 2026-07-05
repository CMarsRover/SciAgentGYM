#!/usr/bin/env bash
# =============================================================================
# SciAgentGYM — install.sh
#
# One-shot installer:
#   1. conda creates a fresh Python 3.11 env from environment.yml.
#   2. environment.yml transparently runs `pip install -r requirements.txt`.
#   3. A quick import check reports what's usable.
#
# Usage:
#   bash install.sh                 # install (fails if env already exists)
#   REBUILD=1 bash install.sh       # remove env first, then rebuild
#   USE_TSINGHUA=1 bash install.sh  # route conda + pip through Tsinghua mirrors
#
# Requires: conda on PATH; internet access to conda-forge and PyPI.
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-sciagentgym}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

command -v conda >/dev/null || { echo "ERROR: conda not on PATH" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "${USE_TSINGHUA:-0}" = "1" ]; then
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
fi

if [ "${REBUILD:-0}" = "1" ]; then
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
fi

echo "=== Creating conda env '${ENV_NAME}' ==="
conda env create -n "${ENV_NAME}" -f "${HERE}/environment.yml"
conda activate "${ENV_NAME}"

echo "=== Verifying imports ==="
python <<'PY'
import importlib, sys
mods = ['openai','zhipuai','tiktoken','numpy','scipy','sklearn','pandas','matplotlib',
        'seaborn','plotly','networkx','sympy','rdkit','pymatgen','spglib','ase','chempy',
        'CoolProp','pubchempy','cirpy','mp_api','mendeleev','py3Dmol','pyscf','qutip',
        'quspin','qiskit','pythtb','lmfit','astropy','astroquery','Bio','pulp','asteval']
fail = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        fail.append(f'{m} ({type(e).__name__})')
print(f'OK {len(mods)-len(fail)}/{len(mods)}' + (f'; MISSING: {fail}' if fail else ''))
sys.exit(1 if fail else 0)
PY

echo ""
echo "=== Done. Activate with: conda activate ${ENV_NAME} ==="
