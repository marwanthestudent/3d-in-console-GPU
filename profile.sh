#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=cube-ncu
#SBATCH --output=ncu-%j.out

# Nsight Compute profile of the cube renderer.
# Mirrors the lab-2 / lab-3 pattern:
#   ncu --set full -o <name> -f <binary> <args>
#
# Submit:   sbatch profile.sh
# Result:   cube_profile.ncu-rep   (open in Nsight Compute UI)
#
# Notes:
#   - ncu replays each kernel several times to gather metrics, so use
#     a small iters count (5) here. The captured per-kernel metrics
#     are independent of how many launches you do.
#   - --set full collects every section (memory, scheduler, occupancy,
#     warp state, source counters). Same flag your other labs used.

ncu --set full -o cube_profile -f ./build/release/cube 100 5
