#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=cube-build
#SBATCH --output=build-%j.out

# Build the single-file CUDA cube renderer using the same pattern
# as your other labs (cmake -B build && cmake --build build).
#
# Submit:   sbatch compile.sh
# Output:   cat build-*.out
# Binary:   ./build/release/cube

set -e

cmake -B build
cmake --build build

echo
echo "BUILD OK"
ls -lh build/release/cube
