#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=cube-bench
#SBATCH --output=bench-%j.out

# Sweep N and collect CPU vs GPU timings.
# Submit:   sbatch benchmark.sh
# Outputs:
#   results/cube-N<N>.txt    per-run raw output (ASCII frame + timings)
#   results/summary.csv      one row per N for plotting / reports

set -e

mkdir -p results
SUMMARY=results/summary.csv
echo "N,vertices,h2d_ms,kernel_ms,d2h_ms,cpu_per_frame_ms,gpu_per_frame_ms,speedup" > "$SUMMARY"

# Sizes to sweep. Add or remove as you like.
# At N=200 the CPU side does ~8M vertices/iter so 100 iters takes a while.
SIZES=(30 50 75 100 125 150 175 200)
ITERS=100

for N in "${SIZES[@]}"; do
    OUT=results/cube-N${N}.txt
    echo
    echo "============================================================"
    echo "  N = $N   ($((N*N*N)) vertices,  $ITERS iterations)"
    echo "============================================================"
    ./build/release/cube "$N" "$ITERS" | tee "$OUT"

    verts=$((N*N*N))

    # Grab numeric values out of the program's text output.
    h2d=$(grep  "H2D copy"   "$OUT" | sed 's/[^0-9.]*\([0-9.]\+\).*/\1/')
    kern=$(grep "^Kernels"   "$OUT" | sed 's/[^0-9.]*\([0-9.]\+\).*/\1/')
    d2h=$(grep  "D2H copy"   "$OUT" | sed 's/[^0-9.]*\([0-9.]\+\).*/\1/')
    cpuf=$(grep "CPU total"  "$OUT" | sed 's/.*(\s*\([0-9.]\+\).*/\1/')
    gpuf=$(grep "GPU kernel" "$OUT" | sed 's/.*(\s*\([0-9.]\+\).*/\1/')
    sp=$(grep   "Speedup"    "$OUT" | sed 's/[^0-9.]*\([0-9.]\+\).*/\1/')

    echo "$N,$verts,$h2d,$kern,$d2h,$cpuf,$gpuf,$sp" >> "$SUMMARY"
done

echo
echo "============================================================"
echo "  Summary"
echo "============================================================"
column -t -s, "$SUMMARY"
echo
echo "CSV written to $SUMMARY"
