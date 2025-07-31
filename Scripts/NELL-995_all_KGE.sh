#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/NELL-995_all/NELL-995"
for size in 0.25 0.5 1.0 2.0; do
    .venv/bin/python standard_BytE_run.py "KGs/NELL-995_all/NELL-995" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
done