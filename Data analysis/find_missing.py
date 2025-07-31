import argparse
import os
import random
import re
import pathlib

import pandas as pd

import custom_BytE_util
from custom_BytE_util import dir_path, ParameterSpace


def find_experiments(path: str) -> (list[str], list[list[str]], int):
    found_experiments = []
    unfound_experiments = []
    try:
        experiment_done = False
        temp_duplicates = -1
        if "_vocab" in path:
            vocab_path = re.search(".*_vocab", path).group()
            if any([str(num) in os.listdir(vocab_path) for num in [4.0, 8.0, 12.0]]):
                path_parts = list(pathlib.Path(path).parts)
                for i in range(len(path_parts)):
                    if "_vocab" in path_parts[i]:
                        path_parts[i + 1] = str(float(path_parts[i + 1]) * 4)
                        break
                path = os.path.join(*path_parts)
        for experiment in os.listdir(path):
            if "eval_report.json" in os.listdir(os.path.join(path, experiment)):
                experiment_done = True
                found_experiments.append(os.path.join(path, experiment))
                temp_duplicates += 1
        if not experiment_done:
            raise FileNotFoundError
    except FileNotFoundError:
        unfound_experiments.append(path)
    return found_experiments, unfound_experiments, max(temp_duplicates, 0)


parser = argparse.ArgumentParser()
parser.add_argument("--exp_path", type=dir_path, required=True)
parser.add_argument("--summary_of", type=str, default="summary_of")
args = parser.parse_args()

KG_pools = custom_BytE_util.KG_pools + ["original_BytE"]
corpus_inputs = custom_BytE_util.corpus_inputs
tie_breakers = custom_BytE_util.tie_breakers
vocab_sizes = [0.25, 0.5, 1.0, 2.0, 3.0]
models = ["Keci", "ComplEx"]
emb_dims = [16, 32, 64]
lrs = [0.1, 0.01]

exp_datasets = set()
for directory in os.listdir(args.exp_path):
    if directory not in ["oldExperiments", "Summaries"]:
        for root, dirs, files in os.walk(os.path.join(args.exp_path, directory)):
            if any([pool in dirs for pool in KG_pools]):
                exp_datasets.add((root, os.path.basename(root)))
experiment_list = []
missing_experiments = []
duplicates = 0
counter = 0
for exp_root, exp_dataset in exp_datasets:
    print(exp_root, "\t\tDS:", exp_dataset)
    for model in models:
        for emb_dim in emb_dims:
            for lr in lrs:
                for pool in KG_pools:
                    if pool == "original_BytE":
                        counter += 1
                        config_path = os.path.join(exp_root, pool, model, str(emb_dim), str(lr))
                        found_exp, miss_exp, dups = find_experiments(config_path)
                        experiment_list += found_exp
                        missing_experiments += miss_exp
                        duplicates += dups
                    else:
                        for corpus_input in corpus_inputs:
                            if pool == "KG-custom" or corpus_input == "VandR":
                                for vocab_size in vocab_sizes:
                                    for tie_breaker in tie_breakers:
                                        if pool == "KG-custom" or tie_breaker != "default":
                                            counter += 1
                                            config_path = os.path.join(exp_root, pool, corpus_input,
                                                                       tie_breaker + "_vocab", str(vocab_size),
                                                                       model, str(emb_dim), str(lr))
                                            found_exp, miss_exp, dups = find_experiments(config_path)
                                            experiment_list += found_exp
                                            missing_experiments += miss_exp
                                            duplicates += dups

expected_exp_count = ((len(exp_datasets) * len(models) * len(emb_dims) * len(lrs))
                      * (1 + len(corpus_inputs) * len(tie_breakers) * len(vocab_sizes) + 2 * 2 * len(vocab_sizes)))
if len(missing_experiments)>5:
    debug_missing = random.sample(missing_experiments, k=5)
else:
    debug_missing = missing_experiments
print("Up to 5 Missing experiments:\n", debug_missing)
print("exp that should exist, according to counter:", counter)
print("expected exp:", expected_exp_count)
print("missing exp:", len(missing_experiments))
print("dups:", duplicates)
print("Total number of experiments:", expected_exp_count - len(missing_experiments) + duplicates)
print("Unique number of experiments:", len(experiment_list))
