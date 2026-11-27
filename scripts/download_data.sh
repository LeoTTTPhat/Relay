#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" - "${1:-data/who_and_when}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Kevin355/Who_and_When", repo_type="dataset",
                  local_dir=sys.argv[1])
PY
