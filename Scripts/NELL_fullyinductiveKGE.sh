#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-0"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-25"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-75"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-100"
for size in 0.25 0.5 1.0 2.0; do
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-0" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-25" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-75" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NL-100" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
done