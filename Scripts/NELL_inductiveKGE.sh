#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v1"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v1_ind"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v2"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v2_ind"
for size in 0.25 0.5 1.0 2.0; do
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v1" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v1_ind" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v2" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/nell_v2_ind" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
done