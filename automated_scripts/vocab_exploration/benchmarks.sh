#!/bin/bash
# This script hands in 217 batch jobs containing 2604 experiments total
for training_kg in FB-v3-ind NL-v1-ind NL-v4-ind wn-v2-ind wn-v4-ind NL-25 NELL-995-h25; do
  for KG_pool in KG-custom; do
    for corpus_input in VandR E randomWalks; do
      for tie_breaker in ascending_size descending_size; do
        for vocab_size in 0.5 1.0 2.0; do
          sbatch automated_scripts\vocab_exploration\benchmark.sh .venv/bin/python easier_BytE_run.py KGs "$training_kg" --num_epochs 100 --min_epochs 100 --base_seed 10 --forced_truncation 100 --bpe_truncation None --bpe_with_RNN Linear --multiple_bpe_encodings 1 --fix_missing 0 --exp_root automated_scripts\vocab_exploration --KG_pool $KG_pool --corpus_input $corpus_input --tie_breaker $tie_breaker --vocab_size $vocab_size
        done
      done
    done
  done
  for KG_pool in KG-pretrained KG-finetuned; do
    for corpus_input in VandR; do
      for tie_breaker in ascending_size descending_size; do
        for vocab_size in 0.5 1.0 2.0; do
          sbatch automated_scripts\vocab_exploration\benchmark.sh .venv/bin/python easier_BytE_run.py KGs "$training_kg" --num_epochs 100 --min_epochs 100 --base_seed 10 --forced_truncation 100 --bpe_truncation None --bpe_with_RNN Linear --multiple_bpe_encodings 1 --fix_missing 0 --exp_root automated_scripts\vocab_exploration --KG_pool $KG_pool --corpus_input $corpus_input --tie_breaker $tie_breaker --vocab_size $vocab_size
        done
      done
    done
  done
  for KG_pool in original_BytE; do
    sbatch automated_scripts\vocab_exploration\benchmark.sh .venv/bin/python easier_BytE_run.py KGs "$training_kg" --num_epochs 100 --min_epochs 100 --base_seed 10 --forced_truncation 100 --bpe_truncation None --bpe_with_RNN Linear --multiple_bpe_encodings 1 --fix_missing 0 --exp_root automated_scripts\vocab_exploration --KG_pool $KG_pool
  done
done