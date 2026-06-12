#!/usr/bin/env bash
# cleanup_venv.sh
#
# Remove the virtual environment created by setup_venv.sh and common temporary
# or cached files from the repository tree.
#
# Usage:
#   bash cleanup_venv.sh
#   bash cleanup_venv.sh --all
#   bash cleanup_venv.sh --dry-run
#
# By default this removes only caches/build artifacts and .venv_rt.
# With --all it also removes common generated experiment-output directories/files.

set -euo pipefail

DRY_RUN=0
REMOVE_OUTPUTS=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --all)
      REMOVE_OUTPUTS=1
      ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

# Resolve repo root as the directory containing this script.
# This assumes cleanup_venv.sh is kept in the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_rm() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] rm -rf %q\n' "$@"
  else
    rm -rf "$@"
  fi
}

run_find_delete() {
  local desc="$1"
  shift
  echo "Removing $desc..."
  if [ "$DRY_RUN" -eq 1 ]; then
    find . "$@" -print
  else
    find . "$@" -exec rm -rf {} + 2>/dev/null || true
  fi
}

echo "Cleaning repository at: $SCRIPT_DIR"

# Virtual environment created by setup_venv.sh.
if [ -d ".venv_rt" ]; then
  echo "Removing virtual environment .venv_rt..."
  run_rm ".venv_rt"
else
  echo "No .venv_rt directory found."
fi

# Python caches.
run_find_delete "Python __pycache__ directories" -type d -name "__pycache__" -prune
run_find_delete "Python bytecode files" -type f \( -name "*.pyc" -o -name "*.pyo" \)

# Test/type/lint caches.
run_find_delete "pytest caches" -type d -name ".pytest_cache" -prune
run_find_delete "mypy caches" -type d -name ".mypy_cache" -prune
run_find_delete "ruff caches" -type d -name ".ruff_cache" -prune
run_find_delete "Jupyter checkpoint directories" -type d -name ".ipynb_checkpoints" -prune

# Python packaging/build artifacts.
run_find_delete "build directories" -type d -name "build" -prune
run_find_delete "dist directories" -type d -name "dist" -prune
run_find_delete "egg-info directories" -type d -name "*.egg-info" -prune

# macOS metadata.
run_find_delete "macOS .DS_Store files" -type f -name ".DS_Store"
if [ -d "__MACOSX" ]; then
  echo "Removing __MACOSX directory..."
  run_rm "__MACOSX"
fi

# Optional generated outputs. Kept by default to avoid deleting experiment results.
if [ "$REMOVE_OUTPUTS" -eq 1 ]; then
  echo "Removing generated experiment outputs because --all was provided..."
  run_rm "results_wishart"
  run_find_delete "generated NumPy archives" -type f -name "*.npz"
  run_find_delete "generated PNG figures" -type f -name "*.png"
  run_find_delete "generated JSON diagnostics" -type f -name "*.json"
else
  echo "Keeping experiment outputs. Use --all to also remove results_wishart/, *.npz, *.png, and *.json."
fi

echo "Cleanup complete."
