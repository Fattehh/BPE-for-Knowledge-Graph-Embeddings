#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH -o ./.Report/KGE/linear_exploration.%j.out 
#STDOUT
echo "$0" "$@"
# This script conducts 1 experiments.
for run in 0; do
  "$@"
done