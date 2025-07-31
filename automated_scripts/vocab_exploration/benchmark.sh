#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH -o ./.Report/KGE/vocab_exploration.%j.out 
#STDOUT
echo "$0" "$@"
# This script conducts 12 experiments.
for run in 0; do
  for model in ComplEx Keci; do
    for embedding_dim in 16 32 64; do
      for lr in 0.1 0.01; do
        "$@" --model $model --embedding_dim $embedding_dim --lr $lr
      done
    done
  done
done