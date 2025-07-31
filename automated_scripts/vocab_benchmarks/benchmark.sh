#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH -o ./.Report/KGE/vocab_benchmarks.%j.out 
#STDOUT
echo "$0" "$@"
# This script conducts 5 experiments.
for run in 0 1 2 3 4; do
  "$@"
done