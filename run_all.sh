#!/bin/bash
# Runs the experiments of the paper. Every command is resumable: running it again only does what is missing.
#   ./run_all.sh light           # or complexity, grid, all
#   JOBS=32 ./run_all.sh grid    # number of parallel runs (default: all cores)
# Never run the same (instance, algorithm) twice at the same time, both would write the same result file.

set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-$($PYTHON -c "import os; print(os.cpu_count())")}
ALGORITHMS="STS-LB STS-LB-Naive ASTS SGTS MuSTS"
mkdir -p logs

run_simulations () {  # instances, runs, extra arguments, log name
    echo "$(date '+%F %T') instances $1"
    $PYTHON -u simulate.py --algorithm $ALGORITHMS --instance_index $1 --runs $2 --jobs $JOBS $3 >> logs/sim_$4.log 2>&1
}

wants () {  # is $1 one of the targets given on the command line
    for target in $TARGETS; do
        if [ "$target" = "$1" ] || [ "$target" = "all" ]; then return 0; fi
    done
    return 1
}
TARGETS="$*"

if wants light; then
    # a few minutes each, also fine on a laptop
    $PYTHON -u bound_tightness.py --jobs $JOBS > logs/bound_tightness.log 2>&1
    # GLR trace: the same instance with delta = 1e-3 (instance 2) and 1e-9 (instance 9)
    $PYTHON -u glr_trace.py --instance_index 2 --log_period 40 --jobs $JOBS > logs/glr_trace_2.log 2>&1
    $PYTHON -u glr_trace.py --instance_index 9 --log_period 50 --jobs $JOBS > logs/glr_trace_9.log 2>&1
    $PYTHON -u heatmaps.py --instance_index 6 --div 61 --jobs $JOBS > logs/heatmaps_6.log 2>&1
fi
if wants complexity; then
    # sample complexity on the two instances of the main text
    run_simulations "4 5" 100 "" complexity
fi
if wants grid; then
    # sample complexity on the random instances, n in {4, 8, 12, 16, 20} and k in {2, 4, 6, 8}
    run_simulations "42 44 46 48 82 84 86 88 122 124 126 128 162 164 166 168 202 204 206 208" 20 "" grid
fi
$PYTHON plot.py all > logs/plot.log 2>&1
echo "$(date '+%F %T') done"
