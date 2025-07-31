#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python -m "Data analysis".find_token_sizes --DS "FB15K237_all"
.venv/bin/python -m "Data analysis".find_token_sizes --DS "NELL-995_all"
.venv/bin/python -m "Data analysis".find_token_sizes --DS "Wikidata"
.venv/bin/python -m "Data analysis".find_token_sizes --DS "WNN18RR_all"
wait