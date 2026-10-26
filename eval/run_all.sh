#!/usr/bin/env bash
# Held-out evaluation (seeds never used during tuning: tuning used 5000-5149).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m eval.run_eval -n 500 --seed 20000 --out eval/results/station
python3 -m eval.run_eval -n 500 --seed 20000 --out eval/results/upload --no-recapture
