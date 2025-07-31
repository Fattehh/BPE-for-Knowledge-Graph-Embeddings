#!/bin/bash
# This script hands in 36 batch jobs containing 36 experiments total
  for bpe_with_RNN in RNN GRU LSTM; do
    for num_epochs in 200; do
      for base_seed in 10; do
        for model in ComplEx Keci; do
          for embedding_dim in 16 32 64; do
            for lr in 0.1 0.01; do
              sbatch automated_scripts\linear_exploration\benchmark.sh .venv/bin/python easier_BytE_run.py KGs NELL-995-h25 --KG_pool KG-pretrained --corpus_input VandR --tie_breaker descending_size --forced_truncation 100 --bpe_truncation None --multiple_bpe_encodings 1 --multiple_bpe_loss False --min_epochs 100 --vocab_size 2.0 --fix_missing 0 --exp_root automated_scripts\linear_exploration --bpe_with_RNN $bpe_with_RNN --num_epochs $num_epochs --base_seed $base_seed --model $model --embedding_dim $embedding_dim --lr $lr
            done
          done
        done
      done
    done
  done
