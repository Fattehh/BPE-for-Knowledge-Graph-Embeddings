#!/bin/bash
echo "Initiate KGE for wk-100!"
source "../torchvenv/bin/activate"
python standard_BytE_run.py KGs/Wikidata/WK-100_labeled False
deactivate
