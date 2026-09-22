#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT=/path/to/project_OC2M/test/test_18

if [[ ! -f "$TEST_ROOT/AdsDrift/activate_as2p_conda.sh" ]]; then
    echo "Activation script not found under $TEST_ROOT/AdsDrift" >&2
    exit 1
fi

source "$TEST_ROOT/AdsDrift/activate_as2p_conda.sh"

python -m pip install --no-deps "fairchem-data-oc==1.0.2"
python -m pip install --no-deps "ase==3.22.1" "pymatgen==2023.5.10"
python - <<'PY'
import ase
import numpy
import pymatgen
import fairchem.data.oc

print("OC20_PREP_ENV_OK")
print("numpy", numpy.__version__)
print("ase", ase.__version__)
print("fairchem-data-oc", fairchem.data.oc.__file__)
PY
