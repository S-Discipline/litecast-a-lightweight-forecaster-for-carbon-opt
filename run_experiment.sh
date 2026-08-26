#!/usr/bin/env bash
set -euo pipefail
# Fixed run command: install deps, then run the configured experiment.
pip install -q --disable-pip-version-check -r requirements.txt
python3 run.py --config config.yaml
