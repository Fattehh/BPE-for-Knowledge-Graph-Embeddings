# Thesis Project: Byte-Pair-Encoding for Knowledge Graph Embeddings

This repository contains the code, models, and experiments for my thesis:  
**"Byte-Pair-Encoding for Knowledge Graph Embeddings"**  
Fatos Ferati, Universität Stuttgart, 2025

# Project details
This repository contains a modified version of dicee 0.1.4, scripts for creating custom BPE vocabularies, and scripts for systematic hyperparameter searching and benchmarking.
It is based on python 3.11 torch 2.6.0. 
Any scripts are intended to be used on a system using Slurm, and with relative paths in mind. Therefore, any script should be executed from the project root. 

# Setup
For setup on a HoreKa node, use the following:
```bash
git clone https://github.com/your-username/BPE-for-Knowledge-Graph-Embeddings.git
cd BPE-for-Knowledge-Graph-Embeddings
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt       # or venv\Scripts\Activate.ps1 on Windows
```
See automated_scripts/automated_scripts.py for an example of creating bash scripts for hyperparameter search, benchmarking, and vocabulary creation.
sh automated_script/xxx/benchmarks.sh, sets up a set of slurm jobs of the script automated_script/xxx/benchmark.sh. 
Use .venv/bin/python custom_BytE_run -h to see an overview of parameters to use for an individual run of BytE. 
Use .venv/bin/python tokenization_script.py -h to see an overview of parameters to use for creating KG token vocabularies.
