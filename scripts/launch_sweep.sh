#!/usr/bin/env bash

set -exu

sweep_id=$1
num_machines=${2:-1}
threads=${3:-4}
mem=${4:-35000}

TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

sweep_name=${sweep_id##*/}
job_name="$sweep_name-$TIME"
log_dir=logs/$sweep_name
log_base=$log_dir/log

partition='defq'

mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-10:00 \
            --array=0-$num_machines \
            scripts/run_sweep.sh $sweep_id $threads
