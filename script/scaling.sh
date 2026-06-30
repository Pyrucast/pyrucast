#!/usr/bin/env bash
# Parallel scaling test — builds and runs the scaling harness in release mode.
# Reports threads / time / speedup / efficiency for the hot FE paths
# (assemble::stiffness and behavior::integrate). Run it on any machine.
#
# Usage: script/scaling.sh [n] [reps]
#   n     grid size  → n×n QUA4 cells          (default 60)
#   reps  timed repetitions per thread count   (default 20)
set -euo pipefail
cd "$(dirname "$0")/.."
exec cargo run --release --quiet --bin scaling -- "${1:-60}" "${2:-20}"
