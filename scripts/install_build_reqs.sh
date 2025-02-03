#!/usr/bin/env bash
set -exuo pipefail

# These two lines get the build requirements from `pyproject.toml`
pip install pip-tools
pip-compile --all-build-deps -o ./build-reqs.txt ./pyproject.toml
# We install them since we're building without isolation.
pip install -r ./build-reqs.txt
pip install build
