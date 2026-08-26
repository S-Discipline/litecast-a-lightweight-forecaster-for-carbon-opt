#!/usr/bin/env bash
set -euo pipefail
# Fixed run command: install deps, then run the configured experiment.
pip install -q --disable-pip-version-check -r requirements.txt
python3 - <<'PYEOF'
import sys, os
import yaml
cfg = yaml.safe_load(open("config.yaml"))
exp = cfg.get("experiment", "baseline")
script = "run.py" if exp == "baseline" else f"run_{exp}.py"
os.execv(sys.executable, [sys.executable, script, "--config", "config.yaml"])
PYEOF
