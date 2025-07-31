#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_labeled"
.venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_uslabeled"
.venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-75_uslabeled"
for size in 0.25 0.5 1.0 2.0; do
    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_labeled" --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_uslabeled" --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-75_uslabeled" --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
done