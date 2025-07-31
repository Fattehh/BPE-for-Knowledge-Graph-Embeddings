#!/bin/bash
# This script is to check if experiments run bugfree
for fix_missing in 5; do
  for base_seed in 20; do
    .venv/Scripts/python.exe easier_BytE_run.py KGs NELL-995-h25 --KG_pool KG-pretrained --corpus_input VandR --tie_breaker descending_size --forced_truncation 100 --bpe_truncation None --bpe_with_RNN Linear --multiple_bpe_encodings 1 --multiple_bpe_loss False --model ComplEx --embedding_dim 64 --lr 0.01 --num_epochs 1000 --min_epochs 500 --vocab_size 2.0 --exp_root automated_scripts/vocab_benchmarks --fix_missing $fix_missing --base_seed $base_seed
    .venv/Scripts/python.exe easier_BytE_run.py KGs NELL-995-h25 --KG_pool KG-custom --corpus_input E --tie_breaker descending_size --forced_truncation 100 --bpe_truncation None --bpe_with_RNN Linear --multiple_bpe_encodings 1 --multiple_bpe_loss False --model ComplEx --embedding_dim 16 --lr 0.1 --num_epochs 1000 --min_epochs 500 --vocab_size 1.0 --exp_root automated_scripts/vocab_benchmarks --fix_missing $fix_missing --base_seed $base_seed
  done
done
