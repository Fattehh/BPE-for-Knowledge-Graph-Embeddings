#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpuonly
#SBATCH -o ./.Report/tokenization/Tokenize.%j.out # STDOUT
.venv/bin/python tokenization_script.py "$@"