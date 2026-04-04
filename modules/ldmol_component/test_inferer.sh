#!/usr/bin/env bash
set -euo pipefail

# currentscriptdirectorydirectory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# PWD directory
export PYTHONPATH="${PWD}"

# parameterdefaultvalue
: "${DEVICE:=cuda:0}"
: "${SEED:=0}"

python3 -m ldmol_component \
 --device "${DEVICE}" \
 --seed "${SEED}"
