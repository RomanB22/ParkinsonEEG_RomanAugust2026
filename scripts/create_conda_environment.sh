#!/usr/bin/env bash

# Create or update the conda environment used by every project workflow.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="MNE_August2026"
PYTHON_VERSION="3.14"
RUN_TESTS=false

usage() {
    cat <<'EOF'
Usage:
  bash scripts/create_conda_environment.sh [options]

Options:
  --env NAME          Environment name (default: MNE_August2026)
  --python VERSION    Python version for a new environment (default: 3.14)
  --run-tests         Run preprocessing tests that do not require analysis outputs
  -h, --help          Show this message

If the named environment already exists, it is kept and its Python packages
are updated to the versions pinned in requirements.txt.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            [[ $# -ge 2 ]] || { echo "ERROR: --env requires a name" >&2; exit 2; }
            CONDA_ENV="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "ERROR: --python requires a version" >&2; exit 2; }
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --run-tests)
            RUN_TESTS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not available on PATH" >&2
    exit 1
}

# Match Conda's previously implicit channel choice explicitly. This prevents
# the deprecation warning without changing the package source used by setup.
export CONDA_CHANNELS="${CONDA_CHANNELS:-defaults}"

cd "$PROJECT_ROOT"

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/parkinson_eeg_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/parkinson_eeg_cache"
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/parkinson_eeg_pip"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$PIP_CACHE_DIR"

environment_exists() {
    conda env list | awk -v environment_name="$CONDA_ENV" '
        $1 == environment_name { found = 1 }
        END { exit !found }
    '
}

named_environment_prefix() {
    printf '%s/envs/%s\n' "$(conda info --base)" "$CONDA_ENV"
}

backup_incomplete_prefix() {
    local prefix="$1"
    local backup
    backup="${prefix}.incomplete.$(date '+%Y%m%d_%H%M%S')"
    echo "Found an incomplete environment prefix: $prefix"
    echo "It will be preserved at: $backup"
    mv "$prefix" "$backup"
}

if environment_exists; then
    echo "Using existing conda environment: $CONDA_ENV"
else
    environment_prefix="$(named_environment_prefix)"
    if [[ -e "$environment_prefix" ]]; then
        # A real Conda environment has both its history and Python executable.
        # A non-empty orphan prefix makes `conda create --name` fail. Preserve
        # it under a timestamped name so recovery remains possible.
        if [[ ! -f "$environment_prefix/conda-meta/history" ]] \
            || [[ ! -x "$environment_prefix/bin/python" ]]; then
            backup_incomplete_prefix "$environment_prefix"
        else
            echo "ERROR: a valid-looking but unregistered prefix already exists:" >&2
            echo "  $environment_prefix" >&2
            echo "Refusing to alter it automatically; inspect Conda's environment registry." >&2
            exit 1
        fi
    fi
    echo "Creating conda environment: $CONDA_ENV (Python $PYTHON_VERSION)"
    conda create --yes --name "$CONDA_ENV" --channel defaults \
        "python=$PYTHON_VERSION" pip
fi

echo "Installing pinned project requirements"
conda run --name "$CONDA_ENV" python -m pip install --requirement requirements.txt
conda run --name "$CONDA_ENV" python -m pip check

echo "Verifying environment imports and package versions"
conda run --name "$CONDA_ENV" python -c '
from importlib.metadata import version

import matplotlib
import mne
import mne_icalabel
import numpy
import ordpy
import pandas
import scipy
import sklearn
import specparam
import tqdm
import bycycle
import ebosc
import neurodsp

packages = (
    "mne",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "mne-icalabel",
    "onnxruntime",
    "ordpy",
    "tqdm",
    "specparam",
    "bycycle",
    "ebosc",
    "neurodsp",
)
for package in packages:
    print(f"{package}=={version(package)}")
'

if [[ "$RUN_TESTS" == true ]]; then
    echo "Running preprocessing tests"
    conda run --name "$CONDA_ENV" python -m unittest -v \
        tests.test_cleaning \
        tests.test_config \
        tests.test_dataset \
        tests.test_ica
fi

echo
echo "Environment is ready: $CONDA_ENV"
echo "Run the ordinal analysis with:"
if [[ "$CONDA_ENV" == "MNE_August2026" ]]; then
    echo "  bash ordinal_analysis/run_ordinal_analysis.sh --overwrite"
else
    echo "  conda run --name $CONDA_ENV python ordinal_analysis/run_ordinal_analysis.py --overwrite"
fi
