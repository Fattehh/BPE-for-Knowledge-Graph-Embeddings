#!/bin/bash
echo "Please call from Custom KG Tokenizer folder!"
.venv/bin/python tokenization_script.py KGs/ FB15K237_all KG-pretrained &
.venv/bin/python tokenization_script.py KGs/ FB15K237_all KG-Custom &
.venv/bin/python tokenization_script.py KGs/ FB15K237_all KG-finetuned &
wait