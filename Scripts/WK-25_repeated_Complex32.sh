#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python standard_BytE_run.py --embedding_dim 16 "KGs/Wikidata/WK-25_brokenlabeled"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/Wikidata/WK-25_brokenlabeled"
.venv/bin/python standard_BytE_run.py --embedding_dim 16 "KGs/Wikidata/WK-25_brokenlabeled"
.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/Wikidata/WK-25_brokenlabeled"



#.venv/bin/python standard_BytE_run.py --embedding_dim 16 "KGs/Wikidata/WK-25_uslabeled"
#.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/Wikidata/WK-25_uslabeled"
#.venv/bin/python standard_BytE_run.py --embedding_dim 16 "KGs/Wikidata/WK-25_uslabeled"
#.venv/bin/python standard_BytE_run.py --embedding_dim 32 "KGs/Wikidata/WK-25_uslabeled"
#for size in 0.25 0.5 1.0 2.0; do
#    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_uslabeled" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
#    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_uslabeled" --embedding_dim 16 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
#    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_uslabeled" --embedding_dim 32 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
#    .venv/bin/python standard_BytE_run.py "KGs/Wikidata/WK-25_uslabeled" --embedding_dim 16 --custom_tokenization --KG_pool "KG-custom" --corpus_input "VandR" --vocab_size "$size"
#done
