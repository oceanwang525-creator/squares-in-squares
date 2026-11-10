#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 poc/free_angle_repolish/poc/free_angle_repolish.py "$@"
