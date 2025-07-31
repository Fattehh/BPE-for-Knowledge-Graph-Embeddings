#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
for kg_pool in KG-pretrained KG-finetuned KG-custom; do
  for corpus_input in VandR E randomWalks; do
    for tie_breaker in default ascending_size descending_size; do
      .venv/bin/python tokenization_script.py KGs NELL-995 "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
      .venv/bin/python tokenization_script.py KGs nell_v1_ind "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
      .venv/bin/python tokenization_script.py KGs nell_v2_ind "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
      .venv/bin/python tokenization_script.py KGs NL-25 "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
      .venv/bin/python tokenization_script.py KGs NL-75 "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
    done
  done
done
for corpus_input in VandR E randomWalks; do
    .venv/bin/python tokenization_script.py KGs NELL-995 KG-custom "$corpus_input" &
    .venv/bin/python tokenization_script.py KGs nell_v1_ind KG-custom "$corpus_input" &
    .venv/bin/python tokenization_script.py KGs nell_v2_ind "$kg_pool" "$corpus_input" &
    .venv/bin/python tokenization_script.py KGs NL-25 KG-custom "$corpus_input" &
    .venv/bin/python tokenization_script.py KGs NL-75 KG-custom "$corpus_input" &
done

wait