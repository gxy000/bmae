#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Run after: conda env create -f environment.yml && conda activate brain-mae
python -c 'import sys; assert sys.version_info[:2] == (3, 7), "Activate the Python 3.7 brain-mae environment first"'
python -m pip install -r requirements.txt

# timm 0.3.2 imports an alias removed by PyTorch 1.12. Update only that import;
# model implementations and checkpoint parameter names remain unchanged.
python - <<'PY'
import importlib.util
from pathlib import Path
import re

spec = importlib.util.find_spec("timm")
helpers = Path(spec.origin).parent / "models" / "layers" / "helpers.py"
source = helpers.read_text()
alias = "import collections.abc as container_abcs"
if alias not in source:
    updated, count = re.subn(
        r"^(?:from torch\._six import container_abcs|import collections\.abc)\s*$",
        alias,
        source,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Unexpected timm helpers.py; inspect " + str(helpers))
    helpers.write_text(updated)
print("timm compatibility import ready:", helpers)
PY

python -m pip check
python - <<'PY'
import torch
import timm
import precomputed_tif
import blockfs
from timm.models.layers.helpers import to_2tuple

assert to_2tuple(16) == (16, 16)
print("PyTorch:", torch.__version__, "timm:", timm.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
command -v rsync
