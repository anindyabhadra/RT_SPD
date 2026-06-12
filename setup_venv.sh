#!/usr/bin/env bash
set -euo pipefail

# setup_venv.sh
#
# Create and populate a Python virtual environment for the RT-SPD repository.
# Run from the repository root:
#
#   bash setup_venv.sh
#   source .venv_rt/bin/activate
#
# Optional environment variables:
#   RT_VENV_NAME=.venv_rt        # venv directory name
#   RT_PYTHON=python3.12         # Python executable to use
#   RT_TORCH_BACKEND=default     # default | cpu | cuda121 | cuda124 | cuda126 | skip
#   RT_RUN_VERIFY=1              # run basic RT verification after install
#
# Examples:
#   RT_TORCH_BACKEND=cpu bash setup_venv.sh
#   RT_TORCH_BACKEND=cuda124 bash setup_venv.sh
#   RT_TORCH_BACKEND=skip bash setup_venv.sh   # use an already-installed torch

VENV_NAME="${RT_VENV_NAME:-.venv_rt}"
PYTHON_BIN="${RT_PYTHON:-python3.12}"
TORCH_BACKEND="${RT_TORCH_BACKEND:-default}"
RUN_VERIFY="${RT_RUN_VERIFY:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}; falling back to python3."
  PYTHON_BIN="python3"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Could not find a usable Python executable."
  echo "Set RT_PYTHON, e.g. RT_PYTHON=/path/to/python3.12 bash setup_venv.sh"
  exit 1
fi

PY_VERSION="$(${PYTHON_BIN} - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
echo "Using Python ${PY_VERSION} from $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

case "${PY_VERSION}" in
  3.10.*|3.11.*|3.12.*)
    ;;
  *)
    echo "WARNING: This repo is tested most safely with Python 3.10--3.12."
    echo "         If PyTorch installation fails, recreate with Python 3.12."
    ;;
esac

if [ ! -d "${VENV_NAME}" ]; then
  echo "Creating virtual environment: ${VENV_NAME}"
  "${PYTHON_BIN}" -m venv "${VENV_NAME}"
else
  echo "Using existing virtual environment: ${VENV_NAME}"
fi

# shellcheck disable=SC1090
source "${VENV_NAME}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "${TORCH_BACKEND}" in
  default)
    echo "Installing PyTorch from PyPI default wheels."
    python -m pip install torch
    ;;
  cpu)
    echo "Installing PyTorch CPU wheels."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ;;
  cuda121)
    echo "Installing PyTorch CUDA 12.1 wheels."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
    ;;
  cuda124)
    echo "Installing PyTorch CUDA 12.4 wheels."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    ;;
  cuda126)
    echo "Installing PyTorch CUDA 12.6 wheels."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
    ;;
  skip)
    echo "Skipping PyTorch installation."
    ;;
  *)
    echo "ERROR: Unknown RT_TORCH_BACKEND='${TORCH_BACKEND}'."
    echo "Allowed: default, cpu, cuda121, cuda124, cuda126, skip."
    exit 1
    ;;
esac

if [ -f requirements.txt ]; then
  echo "Installing repo requirements from requirements.txt"
  python -m pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found. Run this script from the repo root."
  exit 1
fi

python - <<'PY'
import numpy, scipy, sklearn, matplotlib, torch
print("\nInstalled core packages:")
print("  numpy       ", numpy.__version__)
print("  scipy       ", scipy.__version__)
print("  scikit-learn", sklearn.__version__)
print("  matplotlib  ", matplotlib.__version__)
print("  torch       ", torch.__version__)
print("  cuda avail. ", torch.cuda.is_available())
PY

if [ "${RUN_VERIFY}" = "1" ]; then
  echo "\nRunning basic RT verification..."
  python basic_rt/verify_rt.py --p 8 --seed 22
fi

echo "\nEnvironment setup complete."
echo "Activate it with:"
echo "  source ${VENV_NAME}/bin/activate"
