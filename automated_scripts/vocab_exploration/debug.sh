#!/bin/bash
# This script is to check if experiments run bugfree
for model in ComplEx Keci; do
  for embedding_dim in 16 32 64; do
    for lr in 0.1 0.01; do
      .venv/Scripts/python.exe easier_BytE_run.py KGs "$training_kg" --KG_pool KG-custom --corpus_input VandR --tie_breaker ascending_size --vocab_size 4.0 --num_epochs 100 --min_epochs 100 --base_seed 10 --forced_truncation 100 --bpe_truncation None --bpe_with_RNN Linear --multiple_bpe_encodings 1 --fix_missing 0 --exp_root automated_scripts/vocab_exploration --model $model --embedding_dim $embedding_dim --lr $lr
    done
  done
done
