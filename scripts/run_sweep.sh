#!/usr/bin/env bash

set -ex

source scripts/setup.sh

sweep_id=$1
threads=$2

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

$wandb agent $sweep_id