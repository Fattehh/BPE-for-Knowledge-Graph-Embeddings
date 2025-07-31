#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
for corpus_input in VandR E randomWalks; do
  for tie_breaker in default ascending_size descending_size; do
    .venv/bin/python tokenization_script.py KGs WK-25_labeled KG-custom "$corpus_input" --tie_breaker "$tie_breaker" &
    .venv/bin/python tokenization_script.py KGs WK-25_uslabeled KG-custom "$corpus_input" --tie_breaker "$tie_breaker" &
    .venv/bin/python tokenization_script.py KGs WK-50_labeled KG-custom "$corpus_input" --tie_breaker "$tie_breaker" &
    .venv/bin/python tokenization_script.py KGs WK-75_labeled KG-custom "$corpus_input" --tie_breaker "$tie_breaker" &
    .venv/bin/python tokenization_script.py KGs WK-100_labeled KG-custom "$corpus_input" --tie_breaker "$tie_breaker" &
  done
done
#for kg_pool in KG-pretrained KG-finetuned KG-custom; do
#  for corpus_input in VandR E randomWalks; do
#    for tie_breaker in default ascending_size descending_size; do
#      .venv/bin/python tokenization_script.py KGs WK-25_labeled "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
#      .venv/bin/python tokenization_script.py KGs WK-50_labeled "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
#      .venv/bin/python tokenization_script.py KGs WK-75_labeled "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
#      .venv/bin/python tokenization_script.py KGs WK-100_labeled "$kg_pool" "$corpus_input" --tie_breaker "$tie_breaker" &
#    done
#  done
#done
#for corpus_input in VandR E randomWalks; do
#    .venv/bin/python tokenization_script.py KGs WK-25_labeled KG-custom "$corpus_input" &
#    .venv/bin/python tokenization_script.py KGs WK-50_labeled KG-custom "$corpus_input" &
#    .venv/bin/python tokenization_script.py KGs WK-75_labeled KG-custom "$corpus_input" &
#    .venv/bin/python tokenization_script.py KGs WK-100_labeled KG-custom "$corpus_input" &
#done

wait