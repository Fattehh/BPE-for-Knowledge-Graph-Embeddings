#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python tokenization_script.py KGs/ WNN18RR_all KG-pretrained &
.venv/bin/python tokenization_script.py KGs/ WNN18RR_all KG-Custom &
.venv/bin/python tokenization_script.py KGs/ WNN18RR_all KG-finetuned &
wait
